# TileLang TIR→MLIR Checker

检查 TileLang 从 TIR 到结构化 MLIR（Linalg）降级过程中的编译语义一致性，通过结构校验与独立数值对照，捕捉“结构合法但算错了数”的静默编译 Bug。

```text
TileLang 算子 + 适配文件
          |
          v
export: 真实编译，截取 5 阶段 IR
  00_tir_initial -> 01_tir_legalized -> 02_tir_optimized
                                             |
                                             v
                        03_mlir_linalg -> 04_mlir_final
          |
          v
check: 文件校验 + 基础规则
       官方结构验证（可选）+ 各阶段独立数值模拟
          |
          v
诊断 + 首个反例 + 原始 IR + 数组证据
          |
          v
show 回查原始行号 -> Agent 溯源分析 -> 三段式人类报告

run = export + check
```

主检查边界是 **02→03**。五份快照均在 LLVM 后端生成机器码之前；工具可独立运行，也可由 Agent 通过 CLI 调用。

## MCP 服务接入（SG2044 按需启动）

服务通过 stdio 暴露 `run_operator`、`check_trace`、`show_ir`、`export_trace`，客户端连入时启动，关闭 stdin 后退出。无需额外 MCP Python 依赖。在本机可用 `python -m tir_mlir_checker.mcp_server`，安装包后也可用 `tir-mlir-mcp`。

**从本机通过 SSH 启动。** 先配置名为 `sg2044` 的 SSH 别名（用户名、端口、已核验的主机密钥和非交互认证），再把以下内容加入支持 `mcpServers` 的客户端配置：

```json
{
  "mcpServers": {
    "tir-mlir-checker": {
      "command": "ssh",
      "args": ["-T", "sg2044", "/home/gudingrui/sg2044-tilelang-riscv/checker/phase4-mcp.msbbKuO3/run-tir-mlir-mcp.sh"]
    }
  }
}
```

**从 SG2044 本机启动。** 同一套部署也可直接接入：

```json
{
  "mcpServers": {
    "tir-mlir-checker": {
      "command": "/home/gudingrui/sg2044-tilelang-riscv/checker/phase4-mcp.msbbKuO3/run-tir-mlir-mcp.sh",
      "args": []
    }
  }
}
```

[启动脚本](scripts/run-tir-mlir-mcp.sh) 与部署的 `tir_mlir_checker/` 放在同一目录。脚本默认选用 TVM 和官方 MLIR，并调用配置的 `mlir-opt`；显式工具入参优先。SSH 接入时，Trace、适配器和返回的证据路径均位于 SG2044。省略 `out_dir` 时会在部署目录的 `runs/` 下生成独立结果。适配器和补充翻译是可执行 Python，仅传入信任的文件。

退出码 `0/1/4` 都是检查结果（分别为完成、检出差异、不支持），`2/3` 才标记为 MCP 工具错误。响应在 `content` 提供 JSON 文本，并在 `structuredContent` 提供同一对象。真实编译导出、官方结构校验、有限用例数值对照分别记录；数值对照不执行编译产物的机器码。

## 快速上手：5 分钟跑通 Where 示例

### 安装与运行

需要 Python 3.10+，在本仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[execution]'
tir-mlir-check --version

# 使用仓库内的真实五阶段 IR 和独立 NumPy 参考。
# 每轮建立新目录，重复运行不会覆盖已有结果。
where_run="$(mktemp -d "${TMPDIR:-/tmp}/where-check.XXXXXX")"
tir-mlir-check check --trace tests/fixtures/traces/十算子-2026-11-11/Where \
  --operator Where --dynamic --case tests/adapters/where_case.py \
  --out "$where_run/normal"

# 查看参与条件判断的原始 MLIR。
tir-mlir-check show --trace tests/fixtures/traces/十算子-2026-11-11/Where \
  --stage 03 --start 160 --end 174
```

这条命令使用默认轻量前端，无需安装 TileLang、TVM 或 MLIR。静态检查仅依赖标准库；数值模拟需要 NumPy 和 ml_dtypes，已包含在 `execution` 安装选项中。包不附带服务器配置或编译工具链。

打开 `$where_run/normal/Where 自动检查记录.md` 查看结果；完整数据保存在同目录的 `Where 检查证据.zip`。

### 运行结果：正常算子与条件反转 Bug

Where 计算 `out = np.where(cond != 0, x, y)`。输入为 `cond: uint8[2048]`、`x/y: float32[2048]`，独立参考来自 NumPy。9 组用例覆盖全零、非零、交替、字节类别、随机、块边界及 NaN/Inf/正负零。

**正常版：新版三段式报告的核心输出**

```markdown
# Where 自动检查记录

## 1. 检查结论
✅ **通过**（未发现语义错误）
9 组输入 × 5 阶段，共 45/45 次逐位一致，最大有限误差 0；4 项待 Agent 复核。

## 2. 怎么查的与测试范围
读取真实五阶段快照 → 基础结构检查 → 内置解释器分别执行，与独立 NumPy 参考比较。
张量签名：arg0 uint8[2048]；arg1、arg2、arg3(out) float32[2048]。
执行方式：轻量前端；官方 TVM/MLIR 验证与算子机器码执行未请求。

## 3. 问题定位与 Agent 复核
自动报错诊断：无。
公式、数据流、精度、控制流：[待 Agent 结合原始 IR 复核]。
```

**条件反转版：同一算子的临时副本**

将 Stage 03 第 167 行的 `arith.cmpi ne` 改为 `eq`，并更新副本的 manifest 指纹后，以同一组输入检查。原始 Trace 保持不变。报告将同位置的失败折叠为一条：

```markdown
# Where 自动检查记录

## 1. 检查结论
❌ **失败**（在 03_mlir_linalg 发现错误，共 9 次模拟失败）
Stage 03 的 9 组全部失败；其余 4 阶段的 36 次对照通过。退出码为 1。

## 2. 怎么查的与测试范围
与正常版使用相同的 9 组输入、五阶段范围和独立 NumPy 参考。

## 3. 问题定位与 Agent 复核
错误规则：NUM-OUTPUT-DIFFERENCE。
症状位置：03_mlir_linalg.mlir:226，向输出 arg3 写回结果。
首个反例：zero 用例 out[0] 实际 0.25，参考 -4096.5。
Agent 根因分析：[待沿第 226 行写回值回溯]。
```

证据中的原始诊断：

```text
03_mlir_linalg.mlir:226: error: 实际输出与独立参考不一致 [NUM-OUTPUT-DIFFERENCE; mismatch]
```

Agent 顺着第 226 行的写回值追踪到第 167 行，确认条件被反转。可用以下命令查看症状位置：

```bash
tir-mlir-check show --trace tests/fixtures/traces/十算子-2026-11-11/Where \
  --stage 03 --start 220 --end 230
```

## 核心命令与工作模式

### CLI 速查

| 命令 | 输入与用途 |
| --- | --- |
| `export --adapter ... --out ...` | 在兼容的 TileLang RISC-V 环境中真实编译，导出一个算子的五阶段 IR |
| `run --adapter ... --operator ... --out ...` | `export → check`，默认包含动态对照；`--static-only` 仅静态检查 |
| `check --trace ...` | 检查已有单算子 Trace；`--dynamic --case ...` 增加模拟与独立参考对照 |
| `show --trace ... --stage 03 --start 60 --end 85` | 校验输入后返回原始 IR 片段、指纹与行号 |

命令不在 PATH 时使用 `python -m tir_mlir_checker`。Python API 为 `tir_mlir_checker.check()`，完整签名见源码；各命令可用 `--help` 查询参数。

| 参数 | 适用命令与含义 |
| --- | --- |
| `--adapter` / `--trace` | `export/run` 传适配文件；`check/show` 传一个算子目录 |
| `--operator` | `run/check` 的算子展示名称，用于报告及证据命名 |
| `--out` | `export/run/check` 的新输出目录；`check` 省略时在系统临时目录生成 |
| `--static-only` | `run` 只执行采集和静态检查 |
| `--dynamic --case` | `check` 执行各阶段与独立参考的动态对照，同时做有限形状符号检查 |
| `--semantic --case` | `check` 仅请求有限形状的逐元素表达式与覆盖检查 |
| `--translation` | `check` 配合 `--dynamic --case`，单列 Agent 翻译补执行结果 |
| `--tir-backend ast\|tvm` / `--mlir-backend text\|official` | `run/check` 选择解析及模拟前端；默认 `ast/text` |
| `--mlir-opt` | `run/check` 指定兼容工具，仅验证 03/04，不附加转换 passes |
| `--ir-python` / `--tir-python` | `run/check` 指定同机兼容 Python；两者同时设置时须相同 |
| `--seed` | `run/check` 用例种子，默认 `0` |
| `--timeout` | `export/run` 默认 `600` 秒，`check` 默认 `120` 秒；执行请求有效范围为 `1–3600` 秒 |
| `--max-steps` | `run/check` 每次单阶段解释的步数上限，默认 `2_000_000` |
| `--stage` / `--start` / `--end` | `show` 选择 `00`–`04`，默认第 1–80 行，每次最多 400 行 |
| `--version` | 查看包版本 |

### 轻量模式与官方增强模式

| 模式 | 结构解析与验证 | 数值模拟 |
| --- | --- | --- |
| **轻量模式（默认）** | TIR 用 Python AST，MLIR 用有限文本解析；静态检查无需外部框架 | 本包 Python 解释器，数值依赖为 NumPy/ml_dtypes |
| **官方增强模式** | `--tir-backend tvm`：`ir.load_json` 或 TVMScript parser，执行 `verify_ssa/verify_well_formed`；`--mlir-backend official`：`Module.parse/operation.verify()`；`--mlir-opt` 独立验证 03/04 | 本包解释器直接访问 TIR 对象节点，以及 MLIR 的 SSA、属性和 regions |

两侧官方前端与 `mlir-opt` 可分别启用，也可组合使用。依赖已配置时：

```bash
tir-mlir-check check --trace /path/to/trace --operator Where \
  --tir-backend tvm --mlir-backend official \
  --mlir-opt /path/to/mlir-opt --ir-python /path/to/compatible/python \
  --dynamic --case /path/to/adapter.py --out /tmp/where-official-001
```

官方检查结果分别位于 `tir_verification`、`mlir_python_verification`、`mlir_verification`。`--ir-python` 为所选前端提供同机 Python；`--tir-python` 保留 TIR 前端入口。安装、环境配置和服务器连接由调用者或 Agent 完成。

### 输入与六类核心功能

传入完整的**单算子目录**，包含 `00_tir_initial`、`01_tir_legalized`、`02_tir_optimized`、`03_mlir_linalg`、`04_mlir_final`。前三份为 TIR，后两份为 MLIR；01/02 相同也保留，快照不是每个 pass 的逐次记录。

`export` 同时保存 00–02 的 `*_object.json`，manifest 绑定对象、阶段文本和 TVM 版本/commit；MLIR Python 绑定可用时记录依赖身份。传递时保留整个目录，包括对象文件和 manifest。

| 核心功能 | 提供的信息 |
| --- | --- |
| 完整性校验 | 五阶段文件、manifest、字节数、行数、SHA-256 |
| IR 事实提取 | 参数、形状、类型、循环、读写、转换及阶段/位置 |
| 基础错误检查 | 支持模型内的越界、未初始化读取、漏写、只读输入改写和写冲突 |
| 有限动态对照 | 各阶段独立执行，与显式用例的独立参考比较 |
| 来源与定位 | 触发条件、首个反例、相关读写及输入→临时区→输出导航 |
| 稳定诊断与证据 | JSON 诊断、MLIR 风格 `diagnostics.txt`、原始 IR 与数组 |

<details>
<summary>支持范围与语义验证边界说明</summary>

**完整性、定位与诊断**

- SHA-256 绑定文件内容，不能独立证明文件来自同一次编译。绑定的 TIR 对象还会核对打印文本与原阶段文本。
- 来源导航最多 64 条，可能截断；它提供回查线索，不自动证明根因。JSON 保留 `rule_id/status/method/scope/location` 和相关位置 `notes`，TXT 以 `note` 展示补充位置。
- TIR `original_span`、MLIR 显式 `loc` 可能指向 DSL 源码；不能对应 Trace 行号时保留节点路径和原位置，不能将源码行号当作 Trace 行号。
- 未知操作、缺参考、未执行、超时和具体反例分别记录。规则不按算子名称内置公式；转换次数或表达式外观不同不独立构成语义错误。

**官方前端与依赖**

- 默认 `ast/text` 不运行官方 verifier。显式选择官方前端后，版本、扩展、方言、属性或依赖不兼容会保留诊断，不静默回退到轻量解析。
- TIR 官方前端优先 `ir.load_json` 读取绑定对象，无对象时尝试 TVMScript parser。MLIR 官方前端要求相关方言已注册，并按 manifest 核对依赖身份；缺身份记录时注明。
- 官方结构验证与本包模拟分开记录。TVM、MLIR 提供解析与验证，本包实现数值解释器。

**数值模型与支持范围**

- 以 `support_inventory` 和本次诊断为准。循环、内存、标量运算和转换保留原位宽及运算次序；`f32↔bf16/f16` 默认 RNE（最近偶数舍入）。
- 比较保留显式的逐位/容差、NaN 与正负零策略；算术 NaN 按约定分类比较，不默认保证 payload 一致。
- TVM 对象支持标准未调度 Block init：完整直接轴映射、串行归约。MLIR 对象支持受限 `scf.for iter_args`。
- 调度/rfactor、并行归约、复杂布局、未知属性/fastmath、物理调度及特殊同步不作通用支持承诺；逻辑并行模拟不能证明真机线程映射。
- 一次阶段执行可比较多个输出；报告中的执行次数按“阶段 × 用例”去重。规则条目统计保留在 `result.json`，不混入正文执行次数。

**执行与结论**

- case、TVMScript parser 和翻译适配会处理普通 Python；隔离子进程与超时不是安全沙箱，应使用可信输入。
- 退出 0 表示请求完成，静态 ok 不替代动态结果。`semantic_conclusion=bounded_checks_passed` 表示有限检查通过，不能推广为任意输入等价、机器码正确或 AME 验证通过。
- 官方验证、内置模拟、Agent 翻译及真机结果分别记录；机器码测试由 Agent 另配环境并保存执行证据。不得为取得通过而放宽容差、删除未知操作或省略同步。

</details>

## AI Agent 自动化协作指南

### 四步标准流程

1. **编写适配。** `make_kernel()` 返回未编译 PrimFunc，`compile_options` 显式指定 `target/out_idx`，模块顶层不要编译。`make_cases(seed)` 返回 [TestCase](tir_mlir_checker/cases.py)，声明参数、输出、可写 `inouts`、独立参考和比较策略；参考不能来自待测阶段。[Where 适配](tests/adapters/where_case.py) 展示已有 Trace 的动态用例。Agent 在已授权服务器配置兼容环境，管理连接、上传和下载，凭证不写入源码或日志。
2. **采集与检查。** `export` 只采集，`run` 采集后检查，`check` 读取已有目录；每次使用新输出路径。读取退出码和 stdout JSON 的 `tool_status/dynamic_status/semantic_conclusion/ir/agent_diagnostics/artifacts`，再打开证据中的完整结果、事实与诊断。采集失败可能仅留下部分 IR，先确认五阶段是否完整。
3. **回查与溯源。** 先处理具体 error，再分析 `uncertain/unsupported`。通过 `show` 优先核对 02→03 的参数/布局、公式与初始化、源头下标、精度/归约顺序、守卫、输出覆盖及同步。区分写回等症状位置与根因位置；契约不匹配、缺参考、超时或不支持不自动判为算子缺陷，不改写原自动标签。
4. **完成三段式报告。** 保留 Python 自动记录，另写 `<算子名称> 单算子完整检查报告.md`，一级标题与文件名文字一致。解释缺口可按下方协议补执行；Agent 将复核结论、实际执行结果和关键证据填入下表对应章节。

| 最终报告章节 | Agent 填写内容 |
| --- | --- |
| `1. 检查结论` | 通过、失败或未完成；已检查阶段、输入组数、模拟结果与复核情况 |
| `2. 怎么查的与测试范围` | 读取 IR、官方结构验证、独立模拟三步；去重的张量签名、用例、后端、比较策略与各项实际结果 |
| `3. 问题定位与 Agent 复核` | 同阶段/规则/位置的错误合并，列影响用例、症状位置及首个反例；填写 Agent 根因分析和 uncertain 复核结论 |

长日志、完整命令、数组和规则明细放证据，报告保留关键行号与证据链接。

### 产物与命名

用 `--operator <算子名称>` 明确名称；省略时依次取 `manifest.name`、唯一 IR 函数名、Trace 目录名。名称支持字母、数字、下划线、连字符和点，不含空白或路径。展示名称与真实 IR 函数名同时保留，不用于选择数学契约。

建议每轮目录为 `<算子名称>/<日期与运行ID>/`。

| 交付产物 | 生成方与用途 |
| --- | --- |
| 五阶段 IR、`*_object.json`、`manifest.json` | `export` 直接写入输出目录，`run` 写入其 `trace/` 子目录 |
| stdout JSON | 工具返回状态、IR 索引、诊断与产物路径 |
| `<算子名称> 自动检查记录.md` | Python 生成的三段式记录，保留自动标签 |
| `<算子名称> 检查证据.zip` | 完整 `result.json`、`facts.json`、`diagnostics.txt`、原始 IR、数组与执行证据 |
| `<算子名称> 单算子完整检查报告.md` | Agent 另写，填写复核结论与根因分析 |

以 `artifacts` 返回的真实位置为准；`final_report` 仅是命名提示，不代表工具已经生成 Agent 最终报告。

### 退出码速查

| 退出码 | 含义 |
| --- | --- |
| `0` | 请求完成；结合实际静态、动态状态读取结论 |
| `1` | 原检查发现具体 `mismatch` |
| `2` | 输入错误 |
| `3` | 工具错误或超时 |
| `4` | 当前能力不支持 |


<details>
<summary>Agent 翻译补执行协议（解释器不支持时）</summary>

先区分缺参考、预算不足与语义缺口。缺参考先补独立契约；超时先缩小样本。语义明确时分别按原 IR 翻译阶段模型，优先 02 和 03，不从上层公式重写两个相同答案。

```bash
tir-mlir-check check --trace /path/to/trace --operator abs \
  --dynamic --case /path/to/case.py --translation /path/to/translation.py \
  --out /tmp/abs-translation-001
```

接口示意（哈希、行号和函数体须来自实际 IR）：

```python
import numpy as np

def run_stage(arguments, symbols):
    x, _ = arguments
    y = np.empty_like(x)
    for i in range(x.size):
        y[i] = np.abs(x[i])
    return {1: y}

TRANSLATIONS = {
    '03_mlir_linalg': {
        'sha256': '<原始阶段文本的 SHA-256>',
        'entry': 'run_stage',
        'line_map': [{'ir': [1, 10], 'python': [4, 10]}],
        'assumptions': ['固定 f32 连续一维输入，无别名；逻辑串行枚举']
    }
}
```

- 函数只接收参数副本与动态符号，不接收参考答案，返回 `{输出参数位置: ndarray}`；覆盖全部输出，保留 inouts 初值。保留原搬运、下标、初始化、dtype、舍入和归约次序，不能默默省略 extern/barrier/shuffle。
- `sha256` 绑定原 IR；`line_map` 是原 IR 与 Python 的闭区间映射。工具只验证格式、范围与指纹，翻译忠实性由 Agent 复核。NumPy 默认广播、负下标、溢出和归约可能与 IR 不同，必须显式处理。
- 补充结果在 `agent_translation` 单列；`input_binding=identical` 才称同输入补充验证。原输入不可用时会明确标记，不能假设绑定。
- 原 unsupported/mismatch、主结论及退出码不被补充结果升级。生成代码报错先查翻译；通过也只称 Agent 翻译后的样本验证，不继承内置解释器的内存/来源/冲突证明。
- 翻译受子进程超时约束，不受解释器 max_steps 计数。证据中 `agent_translation/` 保存代码、适配、阶段快照和数组；数组按原始 uint8 与 array_dtypes 恢复。包不内置模型或自动翻译器。

</details>
