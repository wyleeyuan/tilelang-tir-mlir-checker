"""Dependency-free MCP stdio transport for the existing checker API."""
from __future__ import annotations

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

from . import __version__, check
from .capture import export_trace
from .cli import _response
from .naming import validate_operator_name
from .trace_io import InputError, excerpt


VERSIONS = ("2025-03-26", "2024-11-05")


def _string(description, **limits):
    return dict(type="string", minLength=1, description=description, **limits)


def _integer(description, **limits):
    return dict(type="integer", description=description, **limits)


def _flag(description):
    return dict(type="boolean", default=False, description=description)


_TIMEOUT = _integer("Timeout in seconds for each export or check invocation.", minimum=1, maximum=3600)
_EXPORT_TIMEOUT = dict(_TIMEOUT, default=600)
_OUT = _string("New server-side output directory. Omit for a unique directory.")
_ADAPTER = _string("Trusted server-side Python adapter with make_kernel(); dynamic run also needs make_cases(seed).")
_TRACE = _string("Server-side directory for a single operator's five-stage trace.")
_COMMON = {
    "operator_name": _string("Display name: 1–100 letters, digits, underscores, hyphens or dots.", maxLength=100),
    "out_dir": _OUT,
    "seed": _integer("Seed for the adapter's case generator.", default=0),
    "max_steps": _integer("Step budget per interpreted stage.", minimum=100, maximum=20_000_000, default=2_000_000),
    "tir_backend": _string("TIR frontend. Deployment default or ast; explicit tvm never falls back.",
                           enum=["ast", "tvm"]),
    "mlir_backend": _string("MLIR frontend. Deployment default or text; explicit official never falls back.",
                            enum=["text", "official"]),
    "ir_python": _string("Python executable on the server for the selected official frontend(s)."),
    "mlir_opt": _string("Compatible mlir-opt executable on the server for additional verification."),
}


def _tool(name, description, *required, **properties):
    return dict(name=name, description=description, inputSchema={
        "type": "object", "properties": properties, "required": list(required), "additionalProperties": False,
    })


TOOLS = {item["name"]: item for item in (
    _tool("run_operator", "Compile and export five stages, then check them with the same adapter.", "adapter",
          adapter=_ADAPTER, static_only=_flag("Skip dynamic comparisons."), timeout=_EXPORT_TIMEOUT, **_COMMON),
    _tool("check_trace", "Check a five-stage trace. Dynamic comparisons require a trusted case adapter.",
          "trace_dir", trace_dir=_TRACE, dynamic=_flag("Compare each stage with reference cases."),
          semantic=_flag("Request bounded symbolic analysis; use with a case adapter."),
          case=_string("Trusted server-side Python file with make_cases(seed)."),
          translation=_string("Trusted supplemental translation for dynamic checks with case."),
          timeout=dict(_TIMEOUT, default=120), **_COMMON),
    _tool("show_ir", "Read original IR lines; verify recorded fingerprints when present.", "trace_dir", "stage",
          trace_dir=_TRACE, stage=_string("Stage number.", enum=["00", "01", "02", "03", "04"]),
          start=_integer("First line, inclusive.", minimum=1, default=1),
          end=_integer("Last line, inclusive; at most 400 lines total.", minimum=1)),
    _tool("export_trace", "Compile one real TileLang RISC-V adapter and capture its five stages.", "adapter",
          adapter=_ADAPTER, out_dir=_OUT, timeout=_EXPORT_TIMEOUT),
)}


class RPCError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _arguments(schema, args):
    fields = schema["properties"]
    if not isinstance(args, dict):
        raise RPCError(-32602, "Tool arguments must be an object")
    if extra := args.keys() - fields.keys():
        raise RPCError(-32602, "Unknown arguments: " + ", ".join(sorted(extra)))
    if missing := set(schema["required"]) - args.keys():
        raise RPCError(-32602, "Missing arguments: " + ", ".join(sorted(missing)))
    for key, value in args.items():
        field = fields[key]
        expected = {"boolean": bool, "integer": int, "string": str}[field["type"]]
        if (type(value) is not expected
                or (expected is str and (not value.strip() or "\0" in value))
                or ("enum" in field and value not in field["enum"])
                or ("minimum" in field and value < field["minimum"])
                or ("maximum" in field and value > field["maximum"])):
            raise RPCError(-32602, f"Invalid argument: {key}")
    if "start" in fields:
        start = args.get("start", 1)
        if not start <= args.get("end", start + 79) < start + 400:
            raise RPCError(-32602, "Require start <= end and at most 400 lines")


def _output(value, trace=None):
    if value is None:
        root = os.getenv("TIR_MLIR_DEFAULT_OUT_ROOT")
        root = Path(root).expanduser().resolve() if root else Path(tempfile.gettempdir()).resolve()
        if trace is not None and root.is_relative_to(trace):
            raise InputError("默认输出根目录不能位于输入 Trace 内")
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="tir-mlir-mcp-", dir=root)) / "result"
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise InputError("输出目录必须不存在")
    if trace is not None and path.is_relative_to(trace):
        raise InputError("输出目录不能位于输入 Trace 内")
    return path


def _check_options(args):
    tir = args.get("tir_backend", os.getenv("TIR_MLIR_DEFAULT_TIR_BACKEND") or "ast")
    mlir = args.get("mlir_backend", os.getenv("TIR_MLIR_DEFAULT_MLIR_BACKEND") or "text")
    if tir not in ("ast", "tvm") or mlir not in ("text", "official"):
        raise InputError("MCP 默认后端配置无效")
    ir_python = args.get("ir_python")
    if ir_python is None and (tir == "tvm" or mlir == "official"):
        ir_python = os.getenv("TIR_MLIR_IR_PYTHON") or os.getenv("TIR_MLIR_TIR_PYTHON") or None
    if ir_python is not None and tir != "tvm" and mlir != "official":
        raise InputError("ir_python 需要至少一个官方前端")
    return dict(tir_backend=tir, mlir_backend=mlir, ir_python=ir_python,
                mlir_opt=args.get("mlir_opt", os.getenv("TIR_MLIR_MLIR_OPT") or None),
                seed=args.get("seed", 0), max_steps=args.get("max_steps", 2_000_000),
                operator_name=args.get("operator_name"))


def _checked(trace, out, **options):
    result = check(trace, out_dir=out, **options)
    response = _response(result)
    if "mlir_verification" in result:
        response["mlir_verification"] = result["mlir_verification"]
    return response


def _execute(name, args):
    if name == "show_ir":
        start = args.get("start", 1)
        return dict(excerpt(args["trace_dir"], args["stage"], start, args.get("end", start + 79)),
                    exit_code=0)
    timeout = args.get("timeout", 120 if name == "check_trace" else 600)
    if name == "export_trace":
        return export_trace(args["adapter"], _output(args.get("out_dir")), timeout=timeout)
    options = _check_options(args)
    if options["operator_name"] is not None:
        validate_operator_name(options["operator_name"])
    if name == "run_operator":
        out = _output(args.get("out_dir"))
        trace = out / "trace"
        captured = export_trace(args["adapter"], trace, timeout=timeout)
        if captured["exit_code"] != 0:
            return captured
        dynamic = not args.get("static_only", False)
        return dict(_checked(trace, out, dynamic=dynamic, case=args["adapter"] if dynamic else None,
                             timeout=timeout, **options), capture=captured)
    trace = Path(args["trace_dir"]).expanduser().resolve()
    out = _output(args.get("out_dir"), trace)
    return _checked(trace, out, dynamic=args.get("dynamic", False), semantic=args.get("semantic", False),
                    case=args.get("case"), translation=args.get("translation"), timeout=timeout, **options)


def _dumps(value):
    # ASCII escapes also cover unusual paths and preserve the one-line wire format.
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _wire_value(value):
    # JSON itself has no IEEE non-finite numbers. Preserve their spelling in
    # diagnostic payloads without changing the checker result or disk evidence.
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, float) and value != value:
        return "NaN"
    if isinstance(value, dict):
        return {k: _wire_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_value(v) for v in value]
    return value


def _call(params):
    name = params.get("name")
    if not isinstance(name, str) or name not in TOOLS:
        raise RPCError(-32602, "Unknown tool")
    args = params.get("arguments", {})
    _arguments(TOOLS[name]["inputSchema"], args)
    try:
        result = _execute(name, args)
    except InputError as exc:
        result = dict(exit_code=2, tool_status="input_error", diagnostics=[str(exc)])
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result = dict(exit_code=3, tool_status="tool_error", diagnostics=[f"{type(exc).__name__}: {exc}"])
    result = _wire_value({"semantic_conclusion": "not_established", "summary": None,
                          "diagnostics": [], "artifacts": {}, **result})
    return {"isError": result["exit_code"] in (2, 3),
            "content": [{"type": "text", "text": _dumps(result)}],
            "structuredContent": result}


def _error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class Session:
    def __init__(self):
        self.version = None

    def dispatch(self, message):
        request_id = message.get("id") if isinstance(message, dict) else None
        has_id = isinstance(message, dict) and "id" in message
        if (not isinstance(message, dict) or message.get("jsonrpc") != "2.0"
                or not isinstance(message.get("method"), str)
                or (has_id and type(request_id) not in (int, str))):
            return _error(request_id if type(request_id) in (int, str) else None, -32600, "Invalid Request")
        if not has_id:
            # The server sends no requests of its own and never replies to notifications.
            return None
        try:
            params = message.get("params", {})
            if not isinstance(params, dict) or ("_meta" in params and not isinstance(params["_meta"], dict)):
                raise RPCError(-32602, "params and its optional _meta must be objects")
            method = message["method"]
            if method == "initialize":
                version = params.get("protocolVersion")
                client = params.get("clientInfo")
                if (self.version is not None or not isinstance(version, str)
                        or not isinstance(params.get("capabilities"), dict) or not isinstance(client, dict)
                        or any(not isinstance(client.get(key), str) for key in ("name", "version"))):
                    raise RPCError(-32602, "Expected initialize with protocolVersion, capabilities and clientInfo once")
                self.version = version if version in VERSIONS else VERSIONS[0]
                result = {"protocolVersion": self.version, "capabilities": {"tools": {}},
                          "serverInfo": {"name": "tir-mlir-checker", "version": __version__}}
            elif method == "ping":
                result = {}
            elif method in ("tools/list", "tools/call"):
                if self.version is None:
                    raise RPCError(-32000, "Server not initialized")
                if method == "tools/list":
                    if "cursor" in params:
                        raise RPCError(-32602, "No additional tools page")
                    result = {"tools": list(TOOLS.values())}
                else:
                    result = _call(params)
            else:
                raise RPCError(-32601, "Method not found")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except RPCError as exc:
            return _error(request_id, exc.code, str(exc))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return _error(request_id, -32603, "Internal error")


def _invalid_constant(token):
    raise ValueError(f"Not a JSON number: {token}")


def serve(source, sink):
    session = Session()
    for line in source:
        try:
            message = json.loads(line.decode("utf-8"), parse_constant=_invalid_constant)
        except (ValueError, RecursionError):
            response = _error(None, -32700, "Parse error")
        else:
            if isinstance(message, list) and message:
                # Batching is part of the 2025-03-26 stdio transport.
                response = [reply for item in message if (reply := session.dispatch(item)) is not None] or None
            else:
                response = session.dispatch(message)
        if response is not None:
            sink.write(_dumps(response) + "\n")
            sink.flush()


def main():
    # Keep the protocol pipe private even from native writes and subprocesses
    # that inherit file descriptor 1. Python prints also follow stderr.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n") as wire:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            with redirect_stdout(sys.stderr):
                serve(sys.stdin.buffer, wire)
        except BrokenPipeError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
