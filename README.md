# TileLang TIR→MLIR Checker

检查 TileLang 从 TIR 到结构化 MLIR（Linalg）降级过程中的编译语义一致性，通过结构校验与独立数值模拟对照，捕捉“IR 语法结构合法，但编译后算错数”的静默编译 Bug。

```text
TileLang 算子 + 适配文件
          |
          v
export: 真实编译，截取 5 阶段 IR
  00_tir_initial -> 01_tir_legalized -> 02_tir_optimized
                                             |
                                             v  (主检查边界: 02 -> 03)
                        03_mlir_linalg -> 04_mlir_final
          |
          v
check: 文件校验 + 基础规则 + 官方结构验证(可选) + 各阶段独立数值模拟
          |
          v
诊断 + 首个反例 + 原始 IR + 数组证据
          |
          v
show 回查行号 -> Agent 溯源分析 -> 三段式人类报告

run = export + check
```

五份快照均取自 LLVM 后端生成机器码之前；支持 **CLI 独立运行**、**Python API** 以及 **零依赖 MCP `stdio` 服务**（配合配套 [Agent Skill](skills/tir-mlir-check/SKILL.md) 使用）。

---

## 当前能力与验证基线

- **实测覆盖率（SG2044 · 520 份正式算子快照，官方对象模式）：**
  - **03 / 04 · MLIR**：`519 / 520`（**99.8%**，仅余 1 份多 `func.func` 入口限制），官方结构校验 `520 / 520` 通过；
  - **02 · TIR**：`346 / 520`（**66.5%**）；
  - **02 + 03 + 04 全链**：`345 / 520`（**66.3%**）。
- **回归测试：** SG2044 真实 TVM / MLIR 对象全套 `pytest` **417 通过 / 3 预期跳过 / 0 失败**（另 90 个子测试通过）；19 项端到端回归零退化，6 项注入假 Bug 全检出。
- **核心支持范围：**
  - **标量与位运算**：浮点数学函数（`erf / log2 / log1p / sin / cos / pow / fma / frcp` 等）、整数位运算/移位/`clz`/`popcount`、`bitcast`、`fptosi / fptoui`；
  - **TileOp 与归约**：`fill`、`async/tma_copy`、`gemm / gemm_py / wgmma_gemm`、8 种 `tl.tileop.reduce`、`cumsum`、`linalg.matmul` 及转置变体；
  - **视图、布局与有界指针**：`memref.reinterpret_cast`、`tl.local_var_init`、安全连续行主序 `layout_map`、受控 `address_of` 与 `tvm_access_ptr`；
  - **控制流与受控外部调用**：`scf.while` + `scf.condition`、串行 `break_loop`、`tvm_tuple`、受控 `call_extern`（`__fdividef / __float2half_rz`）；
  - **受控线程协作重放**：单线程及顶层一致分相 `tvm_storage_sync`、受控静态完整 Warp Shuffle（`shfl_* / tvm_warp_shuffle*`）串行重放。
- **主动保留的拒绝红线（显式返回退出码 `4`，不做猜测性模拟）：** 条件分支内不可证明一致到达的同步与 shuffle、多线程原子操作、`local.fragment`、复杂 swizzle 布局、未建模裸指针逃逸以及多入口 MLIR 模块。

---

## MCP 服务接入与 Agent Skill

本包使用 Python 标准库实现了零外部依赖的 MCP `stdio` 服务（`tir-mlir-mcp` / `python -m tir_mlir_checker.mcp_server`，支持 `2024-11-05` 与 `2025-03-26` 协议），无需常驻守护进程，客户端连接时按需启动、关闭 `stdin` 后自动退出。

配套工作流规范见：**[skills/tir-mlir-check/SKILL.md](skills/tir-mlir-check/SKILL.md)**（可直接安装或引用到 Agent 中，规范化编排适配、检查、溯源和三段式报告）。

### 暴露的 4 个 MCP 工具

| 工具名称 | 用途 | 核心输入 |
| --- | --- | --- |
| `run_operator` | 串联 `export → check`，单算子一键编译导出并语义检查 | `adapter`、`out_dir`（可选）、`operator_name`、`static_only` 等 |
| `check_trace` | 校验并动态模拟已有五阶段 Trace（支持官方 TVM/MLIR 与独立参考） | `trace_dir`、`dynamic`、`case`、`tir_backend`、`mlir_backend` 等 |
| `show_ir` | 校验 manifest 指纹后按阶段（`00`–`04`）和行号返回原始 IR 片段 | `trace_dir`、`stage`、`start`、`end` |
| `export_trace` | 在兼容 TileLang RISC-V 环境中真实编译适配器，导出五阶段快照 | `adapter`、`out_dir`、`timeout` |

> **响应约定：** 工具响应同时在 `content[0].text`（JSON 文本）和 `structuredContent` 返回完整结构化字段；退出码 `0`（完成）、`1`（检出数值差异）、`4`（能力不支持）均属于正常检查结论（`isError: false`），仅 `2`（输入错误）或 `3`（工具异常/超时）标为 `isError: true`。工具执行期间 `sys.stdout` 自动重定向至 `sys.stderr`，确保 MCP 协议流零杂讯污染。

### 客户端接入配置样例

**方式一：本机通过 SSH 按需远程拉起服务器 MCP（推荐）**

先配置可非交互认证且核验过主机密钥的 SSH 别名（如 `<sg2044-host>`），再在部署目录放置 [scripts/run-tir-mlir-mcp.sh](scripts/run-tir-mlir-mcp.sh)（按服务器实际路径调整脚本内环境变量），客户端 `mcpServers` 配置如下：

```json
{
  "mcpServers": {
    "tir-mlir-checker": {
      "command": "ssh",
      "args": ["-T", "<sg2044-host>", "/path/to/tir-mlir-checker-deploy/run-tir-mlir-mcp.sh"]
    }
  }
}
```

**方式二：在已安装环境的同机直接启动**

```json
{
  "mcpServers": {
    "tir-mlir-checker": {
      "command": "/path/to/tir-mlir-checker-deploy/run-tir-mlir-mcp.sh",
      "args": []
    }
  }
}
```

> **提示：** SSH 接入时，传入的 `trace_dir`、`adapter` 路径以及返回的产物路径均位于远程服务器；省略 `out_dir` 且已设置 `TIR_MLIR_DEFAULT_OUT_ROOT` 时，会在其下创建不冲突的独立运行目录。可通过 `TIR_MLIR_DEFAULT_TIR_BACKEND`、`TIR_MLIR_DEFAULT_MLIR_BACKEND`、`TIR_MLIR_IR_PYTHON`、`TIR_MLIR_MLIR_OPT` 配置默认官方后端（显式工具入参优先级更高）。

---

## 快速上手：5 分钟跑通 Where 示例

### 安装与运行（本机轻量模式）

需要 Python 3.10+，在本仓库根目录执行（纯静态与 MCP 协议仅依赖标准库；动态数值模拟需安装 `execution` 可选依赖）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[execution]'
tir-mlir-check --version

# 使用仓库内置的真实五阶段 IR 和独立 NumPy 参考；每次输出到新目录
where_run="$(mktemp -d "${TMPDIR:-/tmp}/where-check.XXXXXX")"
tir-mlir-check check --trace tests/fixtures/traces/十算子-2026-11-11/Where \
  --operator Where --dynamic --case tests/adapters/where_case.py \
  --out "$where_run/normal"

# 回查参与条件判断的原始 MLIR 行号
tir-mlir-check show --trace tests/fixtures/traces/十算子-2026-11-11/Where \
  --stage 03 --start 160 --end 174
```

打开 `$where_run/normal/Where 自动检查记录.md` 查看三段式摘要，完整数组、IR 与诊断信息保存在同目录 `Where 检查证据.zip`。

### 正常算子 vs 条件反转 Bug 对比效果

`Where` 计算 `out = np.where(cond != 0, x, y)`（`cond: uint8[2048]`、`x/y: float32[2048]`，9 组用例）：

- **正常版**：9 组输入 × 5 阶段共 **45/45 次逐位一致**，最大有限误差 `0`（退出码 `0`）。
- **注入条件反转 Bug 版**：仅将临时副本 `03_mlir_linalg.mlir` 第 167 行 `arith.cmpi ne` 改为 `eq` 并更新 manifest 指纹。官方结构校验依然通过，但 Checker 精确检出 Stage 03 全部 `9/9` 组失败（退出码 `1`，其余 4 阶段 `36/36` 通过）：
  - 诊断：`03_mlir_linalg.mlir:226: error: 实际输出与独立参考不一致 [NUM-OUTPUT-DIFFERENCE; mismatch]`；
  - 首个反例：`zero` 用例 `out[0]` 实际 `0.25`，参考 `-4096.5`；
  - Agent 调用 `show` 沿第 226 行写回值回溯到第 167 行即可定位条件反转根因。

---

## CLI 速查与运行模式

### 核心命令

| 命令 | 输入与用途 |
| --- | --- |
| `export --adapter ... --out ...` | 在兼容 TileLang RISC-V 环境中真实编译并截取五阶段快照与对象 JSON |
| `run --adapter ... --operator ... --out ...` | `export → check`，默认包含动态对照；`--static-only` 仅静态检查 |
| `check --trace ...` | 检查单算子五阶段目录；`--dynamic --case ...` 启用各阶段独立数值模拟与有限形状符号检查 |
| `show --trace ... --stage 03 --start 60 --end 85` | 校验输入完整性后返回原始 IR 片段、指纹与行号 |

命令不在 `PATH` 时可用 `python -m tir_mlir_checker`；Python API 为 `tir_mlir_checker.check()`。常用参数包括 `--operator`、`--out`、`--tir-backend ast|tvm`、`--mlir-backend text|official`、`--mlir-opt`、`--ir-python`、`--seed`、`--timeout`（`export/run` 默认 600s，`check` 默认 120s，范围 1–3600s）、`--max-steps`（默认 2,000,000）及 `--translation`。

### 轻量模式与官方增强模式

| 模式 | 结构解析与验证 | 数值模拟 |
| --- | --- | --- |
| **轻量模式（默认 `ast/text`）** | TIR 用 Python AST，MLIR 用受限文本解析；静态检查零外部依赖 | 内置 Python 解释器（NumPy / ml_dtypes） |
| **官方增强模式（`tvm/official`）** | `tvm`：`ir.load_json` / TVMScript + `verify_ssa/verify_well_formed`；`official`：`Module.parse/verify()`；`--mlir-opt` 独立校验 03/04 | 直接解释真实 TIR 对象节点与 MLIR SSA/Attributes/Regions；**版本或依赖不兼容时不静默回退** |

官方增强模式示例：

```bash
tir-mlir-check check --trace /path/to/trace --operator Where \
  --tir-backend tvm --mlir-backend official \
  --mlir-opt /path/to/mlir-opt --ir-python /path/to/compatible/python \
  --dynamic --case /path/to/adapter.py --out /tmp/where-official-001
```

### 退出码与交付产物

| 退出码 | 含义 | 主要产出物 |
| --- | --- | --- |
| `0` | 请求完成（仍须结合 `dynamic_status` 与 `semantic_conclusion` 读取结论） | 五阶段 IR、`*_object.json`、`manifest.json`（`export/run` 产生） |
| `1` | 检查发现具体数值差异（`mismatch`） | `<算子名称> 自动检查记录.md`（Python 自动生成三段式记录） |
| `2` | 输入错误（缺失文件、指纹不匹配或非法参数） | `<算子名称> 检查证据.zip`（含 `result.json`、`facts.json`、`diagnostics.txt`、数组与证据） |
| `3` | 工具错误或执行超时 | stdout JSON（返回 `tool_status`、`artifacts` 与摘要） |
| `4` | 遇到显式不支持的操作或超出当前安全契约 | `<算子名称> 单算子完整检查报告.md`（由 Agent 复核并补充根因后另写） |

---

<details>
<summary>支持范围与证据严谨性细节说明</summary>

1. **五类证据严格区分：** 官方结构验证（`tir_verification` / `mlir_python_verification` / `mlir_verification`）、解释器预检准入、有限用例数值一致性（`semantic_conclusion=bounded_checks_passed`）、Agent 翻译补执行（`agent_translation`，不会覆盖或升级主退出码），以及目标硬件机器码实机执行，五者各自独立记录、互不替代。
2. **指纹与行号：** SHA-256 绑定文件与对象打印一致性；当 TIR `original_span` 或 MLIR `loc` 指向 DSL 源码而非 Trace 行号时保留节点路径，不混淆两者。
3. **数值与舍入契约：** 标量运算、归约次序与低精度转换保留原位宽与语义；`f32↔bf16/f16` 默认 RNE；`--seed` 默认 `0`，不可为求通过而放宽容差、忽略未知操作或跳过屏障/同步。
4. **安全边界：** `case`、`TVMScript` 与 `--translation` 在隔离子进程运行普通 Python，仅应用于可信输入。

</details>
