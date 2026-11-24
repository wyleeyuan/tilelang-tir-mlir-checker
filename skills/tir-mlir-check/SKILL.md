---
name: tir-mlir-check
description: Run and diagnose TileLang TIR-to-MLIR semantic equivalence checks via the tir-mlir-checker MCP tools or CLI. Use whenever compiling/exporting 5-stage IR traces, checking 02->03 lowering bugs, tracing numerical mismatches to original IR lines, or writing the 3-section operator check report.
---

# TileLang TIR→MLIR Semantic Check Skill

## Core Principles & Evidence Discipline

Never blur the boundary between five distinct kinds of evidence:
1. **Official structural verification** (`tir_verification`, `mlir_python_verification`, `mlir_verification`) proves IR syntax/SSA well-formedness, NOT numerical equivalence.
2. **Interpreter preflight admission** proves the op subset is modeled, NOT that test cases pass.
3. **Bounded numeric checks** (`semantic_conclusion=bounded_checks_passed`) prove equivalence only on the tested finite inputs; never generalize to arbitrary shapes or hardware machine code correctness.
4. **Agent translation supplement** (`--translation`) is recorded separately under `agent_translation` and MUST NEVER upgrade or mask an underlying `unsupported` (`4`), `mismatch` (`1`), or exit code.
5. **Real machine-code execution** on target hardware must be run and logged separately.

Strict prohibitions:
- **Never** silently fall back from requested official frontends (`tvm` / `official` / `mlir_opt`) to lightweight AST/text mode.
- **Never** relax tolerances, skip unknown operations, or bypass `tvm_storage_sync` / warp / pointer checks just to obtain exit code `0`.
- **Never** fabricate or overwrite raw Trace files or manifests to hide an error.

---

## Tooling: MCP Tools & Exit-Code Semantics

Prefer the 4 MCP tools when connected to `tir-mlir-checker` (all file paths refer to the host where the MCP server executes, e.g., SG2044 over SSH):

| MCP Tool | CLI Equivalent | Purpose |
| --- | --- | --- |
| `run_operator` | `tir-mlir-check run` | Compile `adapter` → export 5-stage Trace (`00–04`) → run static & dynamic checks. |
| `check_trace` | `tir-mlir-check check` | Validate manifest hashes and run static/official/dynamic checks on an existing Trace directory. |
| `show_ir` | `tir-mlir-check show` | Read verified original IR lines (`stage`: `"00"`–`"04"`, `start`, `end`) for root-cause tracing. |
| `export_trace` | `tir-mlir-check export` | Compile `adapter` and save 5-stage IRs, TIR object JSONs, and SHA-256 `manifest.json`. |

### Interpreting Exit Codes & `isError`

| `exit_code` | MCP `isError` | Meaning & Required Agent Action |
| --- | --- | --- |
| `0` | `false` | Request finished cleanly. Inspect `dynamic_status`, `semantic_conclusion`, and `tir/mlir_*_verification` before claiming a pass. |
| `1` | `false` | **Mismatch detected** (`NUM-OUTPUT-DIFFERENCE` or contract failure). Use `show_ir` from the symptom write-back line backward to locate the root cause. |
| `4` | `false` | **Explicitly unsupported** operation or out-of-contract construct (e.g., conditional sync, atomics, `local.fragment`). Report exact blocker honestly; optionally use `--translation` if semantic contract is clear. |
| `2` | `true` | Input error (missing path, hash mismatch, invalid adapter, existing `out_dir`). Fix input/path — always use a fresh `out_dir`. |
| `3` | `true` | Tool error or timeout (or backend crash). Inspect stderr/diagnostics and `timeout`/`max_steps`; do not misclassify as a numerical pass. |

---

## Standard 4-Step Inspection Workflow

### Step 1: Prepare or Verify the Adapter (`adapter.py`)
If testing a TileLang kernel from scratch, write an adapter exposing:
- `make_kernel()`: returns an **uncompiled** `PrimFunc` plus `compile_options` explicitly specifying `target` and `out_idx`. **Do NOT compile at module import top-level.**
- `make_cases(seed)`: returns deterministic `TestCase` instances (from `tir_mlir_checker.cases`) specifying input tensors, `inouts` (if mutated in place), expected outputs, comparison strategy (bitwise / tolerance / NaN / signed-zero), and an **independent reference** (NumPy / PyTorch).
  - Reference outputs MUST NOT be computed from any IR stage under test.
  - Note: adapters and `--translation` execute Python in an isolated subprocess; only pass trusted code.

### Step 2: Export & Check into a Fresh Output Directory
- For a new kernel: call `run_operator` (with `tir_backend="tvm"`, `mlir_backend="official"`, and `mlir_opt` configured; omitting `out_dir` uses a fresh directory if `TIR_MLIR_DEFAULT_OUT_ROOT` is set).
- For an existing snapshot directory: call `check_trace` with `dynamic=True`, `case=<adapter.py>`, and `operator_name=<Name>`.
- **Always pass a non-existent `out_dir`** (or omit it when auto-managed) to avoid overwriting earlier evidence.
- Parse the returned JSON fields: `tool_status`, `exit_code`, `dynamic_status`, `semantic_conclusion`, `tir_verification`, `mlir_python_verification`, `mlir_verification`, `findings_summary`, `diagnostics`, and `artifacts`.

### Step 3: Diagnose & Trace Root Causes with `show_ir`
1. **If `exit_code == 1` (`mismatch`)**:
   - Read the failing stage, rule ID (`NUM-OUTPUT-DIFFERENCE`), symptom file/line (often a write-back `memref.store` / `BufferStore`), and first counterexample index/values.
   - Call `show_ir` around the symptom line in that stage, then trace dataflow backward through `03_mlir_linalg` and compare against `02_tir_optimized` (check condition polarity, indexing/strides, buffer initialization, precision cast/rounding order, reduction axis/init, guards, and sync boundaries).
   - Distinguish **symptom line** (where bad value is stored) from **root-cause line** (where computation/condition diverged, e.g., inverted `arith.cmpi` predicate).
2. **If `exit_code == 4` (`unsupported`)**:
   - Inspect `support_inventory` and diagnostics for the exact blocking op/annotation/line (`show_ir` if needed).
   - Do not guess semantics for concurrency/pointer red lines. If a deterministic non-concurrent gap exists and the user wants a finite-input supplement, write an explicit `--translation` file bound by exact stage `sha256` and `line_map` without altering primary exit status.

### Step 4: Deliver the 3-Section Human Report
Keep the tool-generated `<算子名称> 自动检查记录.md` and `<算子名称> 检查证据.zip` intact. Write `<算子名称> 单算子完整检查报告.md` whose `# <算子名称> 单算子完整检查报告` H1 matches the filename and strictly follows 3 sections:

1. **`## 1. 检查结论`**: Clear verdict (pass / mismatch found / unsupported / timeout), stages verified, input case count, simulation outcome, and verification status.
2. **`## 2. 怎么查的与测试范围`**: How IR reading, official structural verification, and independent simulation ran; deduplicated tensor signatures, test cases, backends (`ast/text` vs `tvm/official` + `mlir-opt`), comparison tolerances, and observed counts per check.
3. **`## 3. 问题定位与 Agent 复核`**: Deduplicated findings grouped by stage/rule/location, affected test cases, symptom line + first counterexample, and Agent root-cause trace (`show_ir` line references) or `uncertain`/`unsupported` review. Place raw logs and command dumps in the evidence archive and link cleanly.
