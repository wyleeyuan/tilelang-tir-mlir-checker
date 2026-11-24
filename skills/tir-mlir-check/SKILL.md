---
name: tir-mlir-check
description: 通过 tir-mlir-checker 的 MCP 工具或 CLI 执行并诊断 TileLang TIR→MLIR 语义等价性检查。适用于编译导出五阶段 IR 快照、排查 02→03 降级数值差异、回溯原始 IR 行号定位根因，以及撰写三段式单算子完整检查报告。
---

# TileLang TIR→MLIR 语义检查 Skill

## 核心原则与证据边界

严格区分以下五类证据，绝不混淆或替代：
1. **官方结构验证**（`tir_verification`、`mlir_python_verification`、`mlir_verification`）：仅证明 IR 语法、SSA 与注册方言约束合法，**不证明**跨阶段数值等价。
2. **解释器预检准入**（`support_inventory`）：仅证明操作子集位于当前已建模范围内，**不证明**具体用例计算正确。
3. **有限输入数值对照**（`semantic_conclusion=bounded_checks_passed`）：仅证明已测有限用例输出一致，不可泛化为任意形状或目标硬件机器码正确。
4. **Agent 翻译补执行**（`--translation`）：单列记录于 `agent_translation`，**绝不得覆盖或升级**底层原有的 `unsupported`（`4`）、`mismatch`（`1`）结论与主退出码。
5. **目标硬件机器码实机执行**：须单独在目标硬件编译运行并独立留存日志。

严格红线：
- **禁止**在显式指定官方前端（`tvm` / `official` / `mlir_opt`）时静默回退到轻量 AST / 文本模式。
- **禁止**为获得退出码 `0` 而放宽比较容差、跳过未知操作，或绕过 `tvm_storage_sync` / Warp Shuffle / 指针边界校验。
- **禁止**伪造、篡改或覆盖原始五阶段 Trace 文件及 `manifest.json` 以掩盖报错。

---

## MCP 工具与退出码语义

连接 `tir-mlir-checker` MCP 服务时优先调用以下 4 个工具（通过 SSH 接入 SG2044 等远端主机时，所有传入与返回路径均指服务端本地路径）：

| MCP 工具 | 对应 CLI 命令 | 用途 |
| --- | --- | --- |
| `run_operator` | `tir-mlir-check run` | 编译 `adapter` → 导出五阶段 Trace（`00–04`）→ 执行静态与动态语义检查。 |
| `check_trace` | `tir-mlir-check check` | 校验 `manifest.json` 哈希，对已有五阶段 Trace 执行静态、官方结构校验与动态模拟。 |
| `show_ir` | `tir-mlir-check show` | 校验文件指纹后，按阶段（`stage`: `"00"`–`"04"`）与行号区间（`start`、`end`）读取原始 IR 片段用于溯源。 |
| `export_trace` | `tir-mlir-check export` | 在兼容 TileLang RISC-V 环境中真实编译 `adapter`，导出五阶段 IR、TIR 对象 JSON 与 SHA-256 `manifest.json`。 |

### 退出码与 `isError` 解读

| `exit_code` | MCP `isError` | 含义与 Agent 必须采取的动作 |
| --- | --- | --- |
| `0` | `false` | 请求正常完成。须核对 `dynamic_status`、`semantic_conclusion` 及 `tir/mlir_*_verification` 后再下通过结论。 |
| `1` | `false` | **检出数值差异或契约违背**（如 `NUM-OUTPUT-DIFFERENCE`）。从表象写回行出发，使用 `show_ir` 逆向追溯至真正根因行。 |
| `4` | `false` | **遇到显式不支持的操作或超出安全契约**（如条件内并发同步、多线程原子、`local.fragment`）。如实报告阻塞项；仅在确定性串行语义明确时，按需补充 `--translation`。 |
| `2` | `true` | 输入错误（路径不存在、SHA-256 指纹失配、适配器非法或 `out_dir` 已存在）。修正输入与路径，始终使用全新 `out_dir`。 |
| `3` | `true` | 工具异常、后端崩溃或执行超时。排查 stderr、诊断信息与 `timeout`/`max_steps`，绝不可误判为数值通过。 |

---

## 标准四步检查流程

### 第一步：准备或核对适配器（`adapter.py`）
从零检查 TileLang 算子时，编写包含以下入口的适配器文件：
- `make_kernel()`：返回**未编译**的 `PrimFunc` 以及显式指定 `target` 与 `out_idx` 的 `compile_options`。**禁止在模块导入顶层执行编译。**
- `make_cases(seed)`：返回确定性的 `TestCase`（来自 `tir_mlir_checker.cases`），明确输入张量、`inouts`（存在原地写回时）、输出位置、比较策略（逐位 / 容差 / NaN / 有符号零）及**独立参考实现**（NumPy / PyTorch）。
  - 参考输出严禁由被测 IR 阶段计算得出。
  - 安全说明：适配器与 `--translation` 会在隔离子进程运行普通 Python 代码，仅应用于可信输入。

### 第二步：在全新输出目录执行导出与检查
- 新算子：调用 `run_operator`（启用 `tir_backend="tvm"`、`mlir_backend="official"` 及 `mlir_opt`；服务端已配置 `TIR_MLIR_DEFAULT_OUT_ROOT` 时，省略 `out_dir` 会自动创建不冲突的新目录）。
- 已有快照：调用 `check_trace`，设置 `dynamic=True`、`case=<adapter.py>` 与 `operator_name=<算子名称>`。
- **始终传入不存在的 `out_dir`**（或交由服务端自动分配），切勿覆盖历史证据。
- 重点解析返回 JSON 字段：`tool_status`、`exit_code`、`dynamic_status`、`semantic_conclusion`、`tir_verification`、`mlir_python_verification`、`mlir_verification`、`findings_summary`、`agent_diagnostics`、`diagnostics` 与 `artifacts`。

### 第三步：使用 `show_ir` 诊断并溯源根因
1. **当 `exit_code == 1`（`mismatch`）时**：
   - 提取失败阶段、规则 ID（如 `NUM-OUTPUT-DIFFERENCE`）、表象报错行（常为最终写回处的 `memref.store` / `BufferStore`）以及首个反例的下标与实测/参考值。
   - 调用 `show_ir` 查看表象行附近 IR，沿数据流在 `03_mlir_linalg` 中逆向追踪并与 `02_tir_optimized` 逐项对照（排查条件极性、下标与步长、局部缓冲初始化、类型转换与舍入顺序、归约轴与初值、访存保护及同步边界）。
   - 严格区分**表象行**（错误值写回处）与**根因行**（计算或判断逻辑真正偏离处，例如被反转的 `arith.cmpi` 谓词）。
2. **当 `exit_code == 4`（`unsupported`）时**：
   - 结合 `support_inventory`、`agent_diagnostics` 与 `diagnostics` 查明精确阻塞操作/注解与行号（必要时调用 `show_ir` 核对）。
   - 对并发同步、多线程原子、`local.fragment`、复杂 swizzle 布局或裸指针逃逸等红线不做猜测性模拟。若仅属语义明确的确定性串行缺口且用户需要有限用例补测，可编写绑定原阶段 `sha256` 与 `line_map` 的 `--translation` 补充模型，同时保留原始主结论与退出码不变。

### 第四步：交付三段式人类检查报告
保留工具生成的 `<算子名称> 自动检查记录.md` 与 `<算子名称> 检查证据.zip` 原样不动。另行撰写 `<算子名称> 单算子完整检查报告.md`，一级标题 `# <算子名称> 单算子完整检查报告` 须与文件名一致，严格按以下三节组织：

1. **`## 1. 检查结论`**：明确给出总评结论（通过 / 检出差异 / 不支持 / 超时异常）、已查阶段、测试用例数、动态模拟结果与官方结构校验状态。
2. **`## 2. 怎么查的与测试范围`**：说明 IR 读取、官方结构验证及独立数值模拟的执行方式；列出合并去重后的张量签名、测试输入集、所用后端（`ast/text` 或 `tvm/official` + `mlir-opt`）、比较容差以及各项检查的实际计数。
3. **`## 3. 问题定位与 Agent 复核`**：按阶段、规则与位置去重归类问题，写明受影响用例、表象行与首个反例，给出 Agent 结合 `show_ir` 行号回溯的根因判断，以及 `uncertain`/`unsupported` 项的复核意见；原始日志与命令输出保存在证据压缩包中并清晰关联。
