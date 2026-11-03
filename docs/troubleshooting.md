# 排障指南

先保留终端错误和对应 `run-*` 目录，再按下面步骤检查。
所有命令从仓库根目录执行；后端接入见[使用指南](usage.md)。

## 读取退出码与报告

判定脚本的 0 表示故障仍在；下面是 **onnx-reduce CLI** 的退出码，含义不同：

| CLI 退出码 | 含义与下一步 |
|---|---|
| `0` | 正常结束，包括固定点和预算耗尽；阅读报告判断是否变小及最小性状态 |
| `1` | 最终输出未通过独立复检，`stop_reason=unstable_check`；不要直接当作稳定复现模型交付 |
| `2` | 参数、配置、输入或执行错误，也包括原模型无法复现；查看终端错误与已保留证据 |
| `130` | Ctrl-C 中断；最终输出可能不存在或尚未验证，检查 checkpoint |

正常收尾时，运行目录总会有 `report.json`；`--report` 额外导出一份相同报告。
初检失败、加载失败或中断时可能没有最终报告。

当前 JSON schema 的版本字段为 `version: 6`，保留实际产物的 `artifact_recheck`，增加优化配置 `optimizations`，
staged 还会扩展 scheduler 中的阶段和收尾信息。主要字段如下：

| 字段 | 读法 |
|---|---|
| `original` / `final` | 主图节点数、字节数、输入、输出、initializer 数量 |
| `node_reduction_rate` / `byte_reduction_rate` | `1 - final/original`；原值为 0 时为 `null`，字节缩减率可能为负 |
| `stop_reason` | `fixed_point`：完整一轮未接受；`budget_exhausted`：预算停止；`unstable_check`：复检失败；staged 另有 `search_exhausted`、`plateau`、`coverage_inconclusive` |
| `minimality.status` | `pass_fixed_point` 或 `unproven`；精确条件见[架构说明](architecture.md#接受条件与最小性) |
| `minimality.sweep_inconclusive_candidate_count` | 最终模型搜索轮次中的无法判定候选数 |
| `failure_checks.calls` | 实际调用数，包含初检和最终复检；因此可能大于候选配额 |
| `failure_checks.cache_hits` | 经编辑或内容缓存复用判定的次数，不增加 `calls`；准备失败的复用不计入 |
| `failure_checks.interesting` / `not_interesting` / `inconclusive` | 实际调用经目标匹配后的结果计数，不含缓存命中 |
| `failure_checks.target_identity_mismatches` | 实际调用中结束方式或签名不符的次数 |
| `candidates` | `generated`、`invalid`、`accepted`、`not_cheaper` 等候选统计 |
| `candidates.valid_rate` | `(generated - not_cheaper - invalid) / (generated - not_cheaper)`，分母为 0 时为 `null`；不是后端复现率 |
| `passes` | 每个 pass 的候选、无效、实际调用、接受、成本未下降统计 |
| `scheduler.strategy` / `model_version` | 本次策略，以及接受编辑后递增的模型版本 |
| `scheduler.coverage` | 被扫描的摘要、版本、候选范围、是否完成及不确定候选数 |
| `scheduler.sources` | heuristic / coverage，或 coarse / regions / fine / cleanup / coverage 的尝试、实际调用、缓存命中和接受数 |
| `optimizations` | 模型编辑模式、续扫/历史/方向配置、校验缓存与 checkpoint 复用开关 |
| `scheduler.history` | 历史延后/重新尝试次数及有界历史占用；不代表缓存判定 |
| `scheduler.finalization` | staged 独立收尾是否启动、尝试、接受后重启与同版本去重次数 |
| `scheduler.stage_accepted_cost_delta` | staged 各阶段接受编辑的完整成本变化；某些分量可能为负 |
| `artifact_recheck` | 实际输出路径、摘要、是否读回校验、强制复检结果及缓存标志 |
| `scheduler.buckets` | 按 pass / 方向 / 二次幂尺度桶汇总的拒绝类型、接受率、缓存来源和接受成本变化 |
| `scheduler.phase_revisits` / `scale_reuses` / `deduplicated` | 跨版本阶段重访、接受后尺度续用、启发式节点编辑去重次数 |
| `requested_validation_mode` / `effective_validation_modes` / `custom_domains` | 请求的校验策略和运行中观察到的模式与自定义 domain |
| `final_hash` | 最终适配器序列化内容的 SHA-256 |
| `work_directory` / `duration_seconds` | 本次独立运行目录和记录时的已用时间 |

`fixed_point` 不一定意味着 `pass_fixed_point`：如果最终轮次有无法判定的候选，最小性仍为
`unproven`。即使 `pass_fixed_point` 也只针对 `minimality.scope` 描述的候选范围，不保证全局最小。
`scheduler.coverage` 只证明所记录摘要对应的模型，不能把旧版本的完成标志用于新模型。
staged fast 的 `search_exhausted`、`plateau` 均表示快速阶段停止，最小性始终未证明；
staged complete 的 `coverage_inconclusive` 表示完成扫描但仍有无法判定候选，同样为 unproven。
`scheduler.sources`、`buckets` 汇总整个运行，包含此前失效扫描；预算中止的在途尝试不计为
完成尝试，见 `scheduler.jsonl`。计时仅在 `--performance` 开启时采集，否则为 `null`。
其中 preparation 是候选执行中判定路径之外的耗时；evaluation 包含目标匹配和判定证据记录，
不含模型落盘。精确外部执行耗时仍看 `performance.stages.external_check`。

## 找到日志与判定证据

默认工作根目录为 `OUTPUT.work/`，每次运行创建独立 `run-*`，不会复用旧目录。
正常完成后，从报告的 `work_directory` 获取精确路径；出错时查看终端输出的工作根目录，
按本次运行时间确认对应目录。

| 相对运行目录的路径 | 内容与生成时机 |
|---|---|
| `task.json` | 引擎配置、判定配置、调用者工作目录；引擎开始时写入 |
| `initial/model.onnx` | 导入并重新序列化的原模型，供初检使用 |
| `target.json` | 初检成功后记录目标模式、签名表达式与基线身份 |
| `candidates.jsonl` | 候选编辑、决策及相关成本/校验原因；处理候选时生成 |
| `scheduler.jsonl` | 按候选序号关联的版本、阶段、方向、尺度、区域锚点与反馈；不改变候选重放格式 |
| `evaluations.jsonl` | 实际判定与缓存命中的结果、模型路径和摘要；有判定事件时生成 |
| `accepted.jsonl` | 被接受的候选编辑、摘要和成本；有接受事件时生成 |
| `checks/00000001/stdout.log` / `stderr.log` | 第一次实际调用的日志；后续实际调用按序编号，缓存命中不新建日志目录 |
| `candidates/00000000/model.onnx` | 合法且成本更低、已落盘候选的模型；序号可能跳跃 |
| `checkpoint.json` / `checkpoint/*.onnx` | 初检成功后及后续提交时保存的模型快照和报告 |
| `report.json` / `replay.json` | 正常收尾时写出的报告与复现配置，包括最终复检失败的收尾 |

`evaluations.jsonl` 顶层 `outcome` 是目标匹配后的结果；
`observation.outcome` 是进程执行的原始分类。两者不同通常意味着故障身份不匹配。
`observation` 还包含 `returncode`、`termination`、`timed_out`、`output_truncated`
和日志目录 `logs`。缓存命中记录没有新的 `observation`。
命中记录的 `cache_source` 为 `edit` 或 `content`，`model` 指向已有的实际产物；
当前尝试由 `candidate.sequence_number` 区分，不会为命中再写一份模型。
`candidates.jsonl` 中 `edit_cache_hit=true` 也可用于识别编辑失败、校验失败或成本未下降的准备复用。

`replay.json` 记录模型路径、摘要、判定配置、基线身份和工作目录，供手工复现参考；
当前没有直接读取它自动重放的 CLI 参数，也不会保存全部环境变量或后端程序。

## 原模型无法通过初检

### 命令无法执行或结果方向相反

若终端提示 `cannot execute interestingness test`，检查 `--test` 的可执行文件路径和权限。
命令以调用者工作目录运行，`.venv/bin/python`、脚本和配置的相对路径都基于此目录。
工具不执行 shell 展开；含 `|`、`>`、`$VAR` 的行为请移入脚本。

若提示 `the original model is not interesting`，从 `checks/00000001/` 查看诊断，
直接执行[使用指南中的判定脚本检查](usage.md#完整的编译错误判定示例)。
后端编译失败时，包装脚本应该识别目标错误并返回 0；直接把编译器的普通非零错误码
交给 reducer 会被视为未复现。

### 无法判定

若提示 `the original model check is inconclusive`，查看 `evaluations.jsonl`：

- `timed_out=true`：确认是环境变慢还是目标挂起；只有排查挂起时才启用 `--timeout-interesting`。
- `returncode=125` 或自定义无法判定码：修复脚本报告的环境或资源问题。
- `output_truncated=true` 且没有签名匹配：将稳定目标标记提前输出，见下一节。

初检失败不会开始搜索，通常也没有 checkpoint；修复判定后从原模型重新运行。

## 签名没有匹配或发生漂移

`--signature-regex` 对保留的 stdout、换行、stderr 拼接内容取第一个匹配。
有捕获组时只比较全部捕获组，否则比较完整匹配。
例如 `assertion: ([A-Za-z0-9_]+)` 比整行中含地址或路径的签名更稳定。

每个日志流最多保留前 1 MiB **原始字节**，之后的输出继续被读取但不保留。
如果需要的标记出现太晚，可以让脚本提取它并在精简输出中打印。
原始结果为 `interesting` 但没有正则匹配时：

- 日志未截断：判为 `not_interesting`，并记录身份不匹配。
- 日志被截断：判为 `inconclusive`，不会缓存该结果。

正则在截断后的保留内容中已经匹配时，仍按匹配签名比较。
若 `failure_checks.target_identity_mismatches` 增多，检查候选是否触发了其他缺陷，
或表达式是否包含随机值。不要通过扩大匹配范围将无关故障混入目标。

## 输入或候选校验失败

`input model is invalid` 表示原模型未通过选定的校验策略。
加载错误可能更早发生，此时还没有 `run-*` 目录；检查 ONNX 文件格式、external data
路径和文件完整性。`permissive` 也不能跳过文件加载和 IR 导入。

候选无效通常是正常搜索结果，不会终止运行。查看 `candidates.jsonl` 的状态：

- `edit_failed`：编辑无法应用，记录中有 `reason`。
- `invalid`：编辑后未通过校验，查看 `validation.reason`。
- `not_cheaper`：成本没有严格下降，不会调用判定脚本。

自定义 domain 默认自动使用 `structural`，它仍启用 ONNX checker 的完整检查。
只有当目标就是某类不符合完整 ONNX 校验的模型时，才考虑 `--validation-mode permissive`。
各策略保证见[模型校验策略](architecture.md#模型校验策略)。

## 预算耗尽或模型没有变小

`budget_exhausted` 表示搜索未完成，最终复检通过时 CLI 仍返回 0，
报告的 `minimality.status` 为 `unproven`。增大预算或以输出模型为新输入再运行。
`--max-candidate-check-calls 0` 跳过候选搜索，仍执行初检与最终复检。

报告的实际调用数包含初检和最终复检，而 `passes.*.failure_check_calls` 只统计候选调用。
缓存命中、编辑失败、校验失败、成本未降低均不消耗候选调用配额。
时间预算是软限制，已开始的工作及最终复检可使总耗时超过预算。

如果到达固定点却没变小，结合 `candidates`、`passes` 和决策日志区分：
所有候选都无法通过校验、都未复现目标故障，或都没有更低成本。
现有 pass 无法继续缩减并不代表不存在其他形式的更小模型。

## 最终输出复检失败

CLI 返回 1，报告记录 `stop_reason=unstable_check`、`minimality.status=unproven`。
输出文件可能已经写出；文件存在不代表故障仍然稳定。

对照 `evaluations.jsonl` 的初检和最后一次实际调用，以及相应 `checks/` 日志，检查：

1. 判定是否依赖文件名或目录。初检位于 `initial/`，候选位于 `candidates/`，最终检查使用实际输出路径。
2. 后端是否有随机性、设备负载变化、临时资源或环境状态依赖。
3. 结束方式是否变化，或签名是否包含不稳定字段。

`--no-cache` 可用于检查同一内容重复判定的差异；首尾复检本来就不走缓存，
所以关闭缓存不会自动修复不稳定故障。先固定判定条件，再重复确认模型。

## 从 checkpoint 重新运行

Ctrl-C 时 CLI 返回 130。只有初检成功后才会有 checkpoint；最终输出可能还未校验。
选定本次运行的 `checkpoint.json`，从仓库根目录执行下列命令：

```bash
# 替换为需要恢复的那个运行目录，勿选错其他 run-*。
checkpoint_json=/absolute/path/to/run-xxxx/checkpoint.json
checkpoint_model=$(uv run --locked python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$checkpoint_json")
uv run --locked onnx-reduce "$checkpoint_model" \
  --test '.venv/bin/python check_backend.py' \
  --signature-regex 'onnx-reduce-target: (YOUR_TARGET_ERROR)' \
  --max-candidate-check-calls 100 \
  -o results/resumed.onnx --report results/resumed-report.json
```

示例沿用[使用指南](usage.md)中的脚本；将判定命令、签名和其他选项改为原运行配置，
可从 `task.json` 与 `target.json` 核对，并保持相同工作目录和环境。
先查看 checkpoint 内嵌报告，若其 `stop_reason=unstable_check`，先排查最终复检失败。

这会开启**新一轮搜索**：重新初检并建立基线，使用新预算，不恢复旧缓存、旧搜索位置或剩余预算。
若要保留此前目标身份并只做独立收尾，应使用已经正常复检的最终输出和
`--scheduler staged --finalize-from RUN_DIR`，详见[使用指南](usage.md#分阶段搜索与独立收尾实验)。
保留 checkpoint 引用的 `.onnx` 文件；仅复制 JSON 清单不足以恢复模型。

## 准备缺陷报告

附上 CLI 命令、版本、操作系统、后端版本、终端错误、可共享的最小模型和判定脚本，
以及相关 `report.json` 和日志片段。说明是否能在最终输出路径上重复复现。
日志与配置可能包含本地路径或后端诊断，共享前检查其中内容。
