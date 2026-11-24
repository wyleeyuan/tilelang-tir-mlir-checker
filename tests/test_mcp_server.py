import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tir_mlir_checker import __version__, mcp_server


ROOT = Path(__file__).resolve().parents[1]
WHERE = ROOT / "tests/fixtures/official_tir/where"
CASE = WHERE.with_name("where_case.py")
NAMES = {"run_operator", "check_trace", "show_ir", "export_trace"}


def request(method, params=None, *, id=1):
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}


def initialize(version="2025-03-26"):
    return request("initialize", {"protocolVersion": version, "capabilities": {},
                                  "clientInfo": {"name": "test-client", "version": "1"}})


def notify():
    return {"jsonrpc": "2.0", "method": "notifications/initialized"}


def call(name, **arguments):
    return request("tools/call", {"name": name, "arguments": arguments}, id="tool")


def unwrap(reply, code):
    result = reply["result"]
    assert result["isError"] is (code in (2, 3))
    data = result["structuredContent"]
    assert json.loads(result["content"][0]["text"]) == data
    assert data["exit_code"] == code
    assert {"tool_status", "semantic_conclusion", "summary", "diagnostics", "artifacts"} <= data.keys()
    return data


@pytest.fixture
def session(monkeypatch, tmp_path):
    for name in ("DEFAULT_TIR_BACKEND", "DEFAULT_MLIR_BACKEND", "IR_PYTHON", "TIR_PYTHON", "MLIR_OPT"):
        monkeypatch.delenv("TIR_MLIR_" + name, raising=False)
    monkeypatch.setenv("TIR_MLIR_DEFAULT_OUT_ROOT", str(tmp_path / "results"))
    server = mcp_server.Session()
    server.dispatch(initialize())
    assert server.dispatch(notify()) is None
    return server


@pytest.mark.parametrize(("requested", "negotiated"), [
    ("2025-03-26", "2025-03-26"), ("2024-11-05", "2024-11-05"), ("2099-01-01", "2025-03-26"),
])
def test_initialization_and_discovery(requested, negotiated):
    server = mcp_server.Session()
    assert server.dispatch(request("ping", id=0)) == {"jsonrpc": "2.0", "id": 0, "result": {}}
    assert server.dispatch(initialize(requested))["result"] == {
        "protocolVersion": negotiated, "capabilities": {"tools": {}},
        "serverInfo": {"name": "tir-mlir-checker", "version": __version__},
    }
    assert server.dispatch(notify()) is None
    tools = {item["name"]: item for item in server.dispatch(request("tools/list"))["result"]["tools"]}
    assert tools.keys() == NAMES
    for item in tools.values():
        assert item["description"]
        schema = item["inputSchema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) <= schema["properties"].keys()
    assert tools["show_ir"]["inputSchema"]["required"] == ["trace_dir", "stage"]


def test_bad_messages_and_batch():
    lines = [b"{", b"\xff", b"null", b"[]", json.dumps(initialize()).encode(),
             json.dumps([notify(), request("ping", id="α"), request("unknown", id=2),
                         request("tools/list", [1], id=3)]).encode(),
             json.dumps([notify()]).encode()]
    out = io.StringIO()
    mcp_server.serve(io.BytesIO(b"\n".join(lines) + b"\n"), out)
    replies = list(map(json.loads, out.getvalue().splitlines()))
    assert len(replies) == 6
    assert [x["error"]["code"] for x in replies[:4]] == [-32700, -32700, -32600, -32600]
    assert replies[-1][0] == {"jsonrpc": "2.0", "id": "α", "result": {}}
    assert [x["error"]["code"] for x in replies[-1][1:]] == [-32601, -32602]


def test_stdio_process_routes_python_native_and_child_stdout_to_stderr():
    # Inject diagnostic producers before dispatch; the actual transport must
    # isolate writes even through fd 1 and inherited child stdout.
    program = """
import os, subprocess, sys
from tir_mlir_checker import mcp_server as m
original = m.excerpt
def noisy(*args):
    print("python diagnostic", flush=True)
    os.write(1, b"native diagnostic\\n")
    subprocess.run([sys.executable, "-c", "print('child diagnostic')"], check=True)
    return original(*args)
m.excerpt = noisy
raise SystemExit(m.main())
"""
    requests = [initialize(), notify(), request("tools/list", id=2),
                call("show_ir", trace_dir=str(WHERE), stage="03", start=1, end=2)]
    text = "".join(json.dumps(x) + "\n" for x in requests)
    for command in ([sys.executable, "-m", "tir_mlir_checker.mcp_server"],
                    [sys.executable, "-c", program]):
        done = subprocess.run(command, cwd=ROOT, input=text, text=True, capture_output=True, timeout=30, check=True)
        replies = [json.loads(line) for line in done.stdout.splitlines()]
        assert [item["id"] for item in replies] == [1, 2, "tool"]
        assert len(replies[1]["result"]["tools"]) == 4
        assert unwrap(replies[2], 0)["lines"][0] == {"line": 1, "text": "module {"}
    assert all(f"{writer} diagnostic" in done.stderr for writer in ("python", "native", "child"))


def test_show_ir_integrity_and_validation(session, tmp_path):
    args = {"trace_dir": str(WHERE), "stage": "03", "start": 2, "end": 3}
    shown = unwrap(session.dispatch(call("show_ir", **args)), 0)
    assert [line["line"] for line in shown["lines"]] == [2, 3]
    assert shown["sha256"] == hashlib.sha256(Path(shown["file"]).read_bytes()).hexdigest()
    for bad in ({"stage": "3"}, {"start": True}, {"end": 1}, {"end": 402}, {"extra": ""}):
        assert session.dispatch(call("show_ir", **(args | bad)))["error"]["code"] == -32602
    assert session.dispatch(call("unknown-tool"))["error"]["code"] == -32602
    tampered = tmp_path / "tampered"
    shutil.copytree(WHERE, tampered)
    path = tampered / Path(shown["file"]).name
    path.write_bytes(path.read_bytes() + b"\n")
    result = unwrap(session.dispatch(call("show_ir", **(args | {"trace_dir": str(tampered)}))), 2)
    assert "sha256" in result["diagnostics"][0]


def test_check_trace_success_and_reversed_condition(session, tmp_path):
    args = {"trace_dir": str(WHERE), "dynamic": True, "case": str(CASE), "timeout": 120}
    baseline = unwrap(session.dispatch(call("check_trace", **args)), 0)
    assert baseline["semantic_conclusion"] == "bounded_checks_passed"
    assert baseline["dynamic_status"] == "passed"
    assert Path(baseline["artifacts"]["evidence"]).is_file()

    mutant = tmp_path / "mutant"
    shutil.copytree(WHERE, mutant)
    stage = "03_mlir_linalg"
    path = mutant / f"{stage}.mlir"
    text = path.read_text()
    changed = text.replace("arith.cmpi ne,", "arith.cmpi eq,")
    assert changed != text
    raw = changed.encode("utf-8")
    path.write_bytes(raw)
    manifest_path = mutant / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(x for x in manifest["stages"] if x["file"] == path.name)
    entry.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), lines=len(raw.splitlines()))
    manifest_path.write_text(json.dumps(manifest))
    bad = unwrap(session.dispatch(call("check_trace", **(args | {"trace_dir": str(mutant)}))), 1)
    assert bad["semantic_conclusion"] == "bounded_counterexample"
    assert Path(bad["artifacts"]["evidence"]).parent != Path(baseline["artifacts"]["evidence"]).parent


def test_input_error_and_default_output_guard(session, monkeypatch, tmp_path):
    result = unwrap(session.dispatch(call("check_trace", trace_dir=str(tmp_path / "missing"))), 2)
    assert result["tool_status"] == "input_error" and result["diagnostics"]
    result = unwrap(session.dispatch(call("check_trace", trace_dir=str(WHERE), out_dir=str(tmp_path))), 2)
    assert result["tool_status"] == "input_error"
    protected = tmp_path / "protected"
    protected.mkdir()
    inside = protected / "forbidden"
    monkeypatch.setenv("TIR_MLIR_DEFAULT_OUT_ROOT", str(inside))
    unwrap(session.dispatch(call("check_trace", trace_dir=str(protected))), 2)
    assert not inside.exists()


def test_official_default_no_fallback_and_explicit_override(session, monkeypatch, tmp_path):
    args = {"trace_dir": str(WHERE)}
    missing = str(tmp_path / "missing-python")
    monkeypatch.setenv("TIR_MLIR_DEFAULT_TIR_BACKEND", "tvm")
    monkeypatch.setenv("TIR_MLIR_DEFAULT_MLIR_BACKEND", "official")
    monkeypatch.setenv("TIR_MLIR_TIR_PYTHON", missing)
    refused = unwrap(session.dispatch(call("check_trace", **args)), 4)
    assert refused["tool_status"] == "unsupported"
    assert refused["tir_verification"]["status"] != "passed"
    assert refused["mlir_python_verification"]["status"] != "passed"
    assert any(missing in d for d in refused["diagnostics"])
    override = unwrap(session.dispatch(call("check_trace", **args, tir_backend="ast", mlir_backend="text")), 0)
    assert override["tir_verification"]["status"] == "not_requested"
    monkeypatch.setenv("TIR_MLIR_IR_PYTHON", "/primary")
    monkeypatch.setenv("TIR_MLIR_MLIR_OPT", "/default-opt")
    assert mcp_server._check_options({})["ir_python"] == "/primary"
    assert mcp_server._check_options({})["mlir_opt"] == "/default-opt"
    opts = mcp_server._check_options({"ir_python": "/explicit", "mlir_opt": "/explicit-opt"})
    assert opts["ir_python"] == "/explicit" and opts["mlir_opt"] == "/explicit-opt"


def test_run_preserves_export_error_and_check_arguments(session, monkeypatch, tmp_path):
    captured = []
    def capture(adapter, out, *, timeout):
        captured.append((adapter, out, timeout))
        return {"exit_code": 3, "tool_status": "tool_error", "diagnostics": ["compiler timeout"]}
    monkeypatch.setattr(mcp_server, "export_trace", capture)
    out = tmp_path / "explicit"
    args = {"adapter": "/adapter.py", "out_dir": str(out), "timeout": 6, "seed": 11, "max_steps": 1000}
    result = unwrap(session.dispatch(call("run_operator", **args)), 3)
    assert result["diagnostics"] == ["compiler timeout"]
    assert captured == [("/adapter.py", out / "trace", 6)]

    monkeypatch.setattr(mcp_server, "export_trace", lambda *a, **kw: {"exit_code": 0})
    def checked(trace, destination, **options):
        assert trace == out / "trace" and destination == out
        assert options["case"] == "/adapter.py" and options["dynamic"]
        assert options["timeout"] == 6 and options["seed"] == 11 and options["max_steps"] == 1000
        return {"exit_code": 0, "tool_status": "ok"}
    monkeypatch.setattr(mcp_server, "_checked", checked)
    assert unwrap(session.dispatch(call("run_operator", **args)), 0)["capture"] == {"exit_code": 0}
