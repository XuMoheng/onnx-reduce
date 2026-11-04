# 使用指南

本文介绍如何为自己的后端编写故障判定脚本，并运行模型缩减。
安装与可直接运行的教学示例见 [README](../README.md)。所有命令从仓库根目录执行。

## 运行前准备

准备可以稳定复现目标问题的 ONNX 模型、后端程序及其依赖。先在同一工作目录和环境中
直接调用后端，确认模型确实触发目标诊断、崩溃或挂起。固定后端版本、编译选项和设备配置。
external data 模型还需要原始权重文件；加载时会将这些数据读入内存。

接下来的完整示例假定：

- 已在仓库根目录运行 `uv sync --locked`，存在 `.venv/bin/python`。
- 将原模型放到 `cases/original.onnx`，external data 文件保持原有相对位置。
- 后端可通过 `your-backend MODEL_PATH` 运行，并在复现时输出 `YOUR_TARGET_ERROR`。
  这两个名称需要按真实后端替换。

输出使用单独的 `results/` 目录，避免覆盖原模型。已有同名输出会被覆盖；
`--report` 不允许与输入或输出模型指向同一个文件。

## 编写故障判定脚本

### 参数与退出码契约

`--test CMD` 使用 shell 风格的引号拆分参数，然后直接启动进程。
工具将候选模型的**绝对路径追加为最后一个参数**，保留调用者的工作目录和环境，
标准输入关闭。候选文件名和目录会变化，脚本应只通过传入的路径读取模型。

不会执行管道、重定向、变量或通配符展开；需要这些行为时写成独立脚本，
例如通过 `--test 'bash check.sh'` 调用。判定进程以调用者权限执行，工具不提供执行沙箱。

| 判定进程结果 | 默认含义 |
|---|---|
| 退出码 `0` | 目标故障仍在（`interesting`） |
| 退出码 `125` | 无法判定（`inconclusive`），不接受候选 |
| 其他普通非零退出码 | 目标故障不在（`not_interesting`） |
| 被信号终止 | 故障仍在；`--no-crash-interesting` 可改为不在 |
| 超过 `--timeout` | 无法判定；`--timeout-interesting` 可改为仍在 |

**退出码 0 表示复现目标故障。** 后端失败不等于判定脚本应该返回非零；
脚本需要把后端结果转换为上述约定。环境缺失、设备不可用等情况应返回 125，
不要把它们当作目标缺陷。

### 完整的编译错误判定示例

在仓库根目录保存为 `check_backend.py`，替换后端命令及目标诊断标记：

```python
import subprocess
import sys

try:
    result = subprocess.run(
        ["your-backend", sys.argv[-1]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
except OSError as exc:
    print(f"cannot execute backend: {exc}", file=sys.stderr)
    raise SystemExit(125) from exc

print(result.stdout, end="")
reproduced = result.returncode != 0 and "YOUR_TARGET_ERROR" in result.stdout
if reproduced:
    print("onnx-reduce-target: YOUR_TARGET_ERROR")
raise SystemExit(0 if reproduced else 1)
```

该模板适用于运行时间和日志量可控的编译后端；它把后端日志暂存在内存中。
如需识别许可证、设备或服务错误，请在目标判定前为这些情况添加返回 125 的分支。
工具的 `--timeout` 约束整个脚本；超时会终止其 POSIX 进程组。

先直接验证脚本。原模型复现时应打印目标标记并返回 0：

```bash
uv run --locked python check_backend.py cases/original.onnx
echo $?
```

再用一个能正常通过后端的模型执行同一命令，应返回 1。
模拟缺少后端的环境时应返回 125，以确认脚本没有颠倒契约。

### 保持目标故障身份

默认 `--target-mode baseline` 要求候选与原模型的**判定进程**结束方式一致：
相同退出码、相同信号编号或同为超时。上面的包装脚本把所有目标故障转换为退出码 0，
因此仍需脚本本身或签名区分具体缺陷。

`--target-mode script` 信任脚本给出的 `interesting`，不比较结束方式。
只有脚本已能可靠识别目标故障时才适合使用它。

两种模式均可配合 `--signature-regex`：在保留的 stdout 与 stderr 中查找第一个匹配。
有捕获组时以全部捕获组作为签名，否则使用完整匹配；候选签名必须与原模型相同。
应选稳定的错误编号或断言名称，避免把地址、临时路径或时间戳包含在签名中。
原始结果还必须是 `interesting`，正则匹配本身不会把非零退出判为复现。
签名缺失和日志截断的处理见[排障指南](troubleshooting.md#签名没有匹配或发生漂移)。

## 执行缩减

替换好脚本并准备 `cases/original.onnx` 后，从仓库根目录执行：

```bash
mkdir -p results
uv run --locked onnx-reduce cases/original.onnx \
  --test '.venv/bin/python check_backend.py' \
  --signature-regex 'onnx-reduce-target: (YOUR_TARGET_ERROR)' \
  --max-candidate-check-calls 100 \
  --time-budget-seconds 600 \
  --timeout 60 \
  -o results/reduced.onnx \
  --report results/report.json
```

这里最多执行 100 次候选判定；原模型检查和最终复检不计入该配额，缓存命中也不消耗配额。
600 秒是软墙钟预算，从本次运行开始计时，包含初检；已经开始的生成、编辑、校验或判定
可能超过预算。最终复检仍会执行，所以它不是进程总时长的硬上限。

### 常用选项

| 选项 | 默认值与作用 |
|---|---|
| `--test CMD` | 必填；判定命令 |
| `-o, --output PATH` | 必填；交付模型路径 |
| `--report PATH` | 不额外导出；指定后复制一份 JSON 报告到此处 |
| `--timeout SECONDS` | `60`；单次判定超时，必须为有限正数 |
| `--max-candidate-check-calls N` | 不设上限；非负整数，`0` 跳过候选搜索 |
| `--time-budget-seconds N` | 不设上限；软墙钟预算，必须为有限正数 |
| `--scheduler legacy\|stateful\|staged` | `legacy`；另提供有状态调度与分阶段搜索两种实验策略 |
| `--goal fast\|complete` | 仅 staged；默认 complete，fast 省略固定点扫描 |
| `--finalize-from RUN_DIR` | 仅 staged complete；恢复源运行目标，独立收尾其已验证输出 |
| `--validation-mode strict\|structural\|permissive` | `strict`；[模型校验策略](architecture.md#模型校验策略) |
| `--target-mode baseline\|script` | `baseline`；匹配原模型结束方式或信任脚本 |
| `--signature-regex REGEX` | 不提取签名；指定后要求基线签名匹配 |
| `--inconclusive-code CODE` | `125`；可重复，显式指定后替换默认集合；范围 1–255 |
| `--no-crash-interesting` | 默认信号终止视为复现；使用此选项后视为未复现 |
| `--timeout-interesting` | 默认超时无法判定；使用此选项后视为复现 |
| `--model-edit-mode ir\|auto` | `ir`；auto 对支持的平坦图使用 protobuf 快速删除，其余回退 IR |
| `--search-policy restart\|cursor` | `restart`；cursor 仅用于 staged，接受后延续细搜索游标 |
| `--direction-order fixed\|adaptive` | `fixed`；adaptive 仅用于 staged/cursor，调整区域删除/保留顺序 |
| `--history-order off\|defer` | `off`；defer 仅用于 staged/cursor，旧拒绝只延后，不跳过 |
| `--no-cache` | 关闭运行内的编辑、shape、内容判定、校验与域缓存；快照、调度去重及显式历史排序仍存在 |
| `--no-validation-cache` | 仅关闭校验/域缓存，供性能对照使用 |
| `--no-checkpoint-reuse` | 关闭同内容 checkpoint 去重，供存储对照使用 |
| `--work-dir DIR` | `OUTPUT.work`；日志和 checkpoint 的根目录，每次创建独立 `run-*` |
| `-v, --verbose` | 默认 INFO；开启 DEBUG 日志 |

完整帮助：`uv run --locked onnx-reduce --help`。旧参数 `--max-tests`、`--checker`
已移除，分别改用 `--max-candidate-check-calls`、`--validation-mode`。

### 选择搜索调度

默认 `legacy` 按 `drop_outputs → remove_unused → delete_nodes` 搜索，接受编辑后从首个 pass
重启。`--scheduler stateful` 会延续有效阶段、分区尺度和位置，节点搜索同时尝试删除区域
和仅保留区域；阶段最多连续尝试 8 个候选后轮转。两种策略使用相同的接受条件和缓存。

stateful 当前为实验选项，不保证每个模型更快，也可能在相同预算下留下更大的模型。
对照时保持输入、判定器、缓存和预算一致，只切换该参数；结果见
[第三批实测](performance/performance-batch3-results.md)。

stateful 的启发式搜索结束后，还会在当前模型上扫描全部配置 pass 及标准分区的保留候选。
扫描中接受编辑就重新开始新版本的搜索与覆盖。预算停止或最终扫描有无法判定候选时，
最小性为 `unproven`；报告的 `minimality.scope` 和 `scheduler.coverage` 描述实际保证范围。
这里只搜索主图节点集合，保留区域不会自动保留其依赖闭包，边界提升规则见
[架构说明](architecture.md#搜索调度与节点身份)。

### 崩溃与挂起场景

直接判定崩溃时，将 `--test` 设置为实际后端命令，候选路径仍追加在最后。
默认基线模式要求相同信号，不能仅凭“进程崩溃”证明是同一个缺陷，应尽量再匹配稳定诊断。
如果通过 Python 包装后端，`subprocess.run()` 的子进程信号会变成包装脚本可见的
负 `returncode`，不会自动让包装脚本也被同一信号终止；应显式识别并转换结果。

排查挂起时，在已确认原模型会稳定超时的命令上添加
`--timeout 10 --timeout-interesting`，将 10 秒替换为正常运行与挂起能清晰区分的阈值。
如果设备负载也可能导致超时，这一设置会将它视为故障；应先稳定运行环境。

## 确认结果

工具先在工作目录中检查原模型，接受缩减后写 checkpoint，最后将模型写到 `-o` 指定位置，
先重新读取实际文件并校验，再在实际交付路径上绕过缓存独立复检。正常收尾后应能找到：

- `results/reduced.onnx`：自包含的输出模型，external data 已内嵌。
- `results/report.json`：通过 `--report` 指定的报告副本。
- `results/reduced.onnx.work/run-*/`：每次运行的判定证据、checkpoint 和报告。

CLI 退出 0 包含固定点与预算耗尽，不能据此推断模型一定变小。
读取报告的 `original`、`final`、`stop_reason` 和 `minimality.status`：

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("results/report.json").read_text())
for key in ("original", "final", "stop_reason", "minimality"):
    print(key, report[key])
print("run directory:", report["work_directory"])
PY
```

交付前还可以手动确认目标诊断：

```bash
uv run --locked python check_backend.py results/reduced.onnx
echo $?
```

应返回 0 并打印相同目标标记。手动运行包装脚本不会执行 reducer 的基线签名比较，
因此也要检查输出中的具体错误。报告含义与 CLI 退出码见[排障指南](troubleshooting.md#读取退出码与报告)。

## 常见问题

- 原模型无法复现：先直接执行判定脚本，核对 0/1/125 方向、工作目录与签名；
  参见[原模型无法通过初检](troubleshooting.md#原模型无法通过初检)。
- 预算耗尽：已有结果仍会最终复检，但最小性通常未证明；
  参见[预算耗尽或模型没有变小](troubleshooting.md#预算耗尽或模型没有变小)。
- 最终复检失败：输出文件可能仍存在，CLI 返回 1；
  参见[最终输出复检失败](troubleshooting.md#最终输出复检失败)。

## 适用范围

当前用于不依赖具体输入数据的编译错误、崩溃和挂起。编辑仅作用于主图，可能将中间值
提升为新图输入，保留故障不代表计算语义等价。暂不缩减输入数据或比较数值差异；
模型规模、平台、缓存和子图边界见[当前边界](architecture.md#当前边界)。

## 分阶段搜索与独立收尾（实验）

`--scheduler staged` 按粗分组、图区域、小组/单项搜索，然后独立验证固定点。
默认目标为 `complete`；`--goal fast` 省略固定点扫描，无论模型多小都只报告最小性未证明。
搜索和收尾共享现有总预算。完整目标若预算用尽，同样为 `unproven`。

```bash
uv run --locked onnx-reduce cases/original.onnx \
  --test '.venv/bin/python check_backend.py' \
  --signature-regex 'onnx-reduce-target: (YOUR_TARGET_ERROR)' \
  --scheduler staged --goal fast --time-budget-seconds 60 \
  -o results/fast.onnx --report results/fast.json
```

需要随后独立收尾时，从 `results/fast.json` 读取 `work_directory`，替换下面的 `run-EXAMPLE`。
保持判定命令、签名、超时、校验模式和工作目录与第一次一致：

```bash
uv run --locked onnx-reduce results/fast.onnx \
  --test '.venv/bin/python check_backend.py' \
  --signature-regex 'onnx-reduce-target: (YOUR_TARGET_ERROR)' \
  --scheduler staged --goal complete \
  --finalize-from results/fast.onnx.work/run-EXAMPLE \
  --time-budget-seconds 600 -o results/final.onnx
```

独立入口验证源产物摘要、恢复原始目标故障身份，再进行新的初检与完整扫描；
不会重新建立目标身份，也不恢复旧缓存和旧搜索进度。必须保留完整的源运行目录，
输入须与源运行最终输出一致；未正常结束的 checkpoint 不能使用此入口。

区域首先读取显式 `onnx_reduce.module` 节点标注，再使用无分叉链，最后回退拓扑分区。
未标注模型无需额外准备。边界缺类型会影响排序，不会永久跳过该区域。
当前标注格式、范围保证和内部额度见[第四批设计](design/staged-search-design.md)，
质量与成本对照见[第四批实测](performance/performance-batch4-results.md)。

### 第五、六批实验优化

`--model-edit-mode auto` 可与三种 scheduler 配合。支持范围外的模型自动回退 IR，
包括子图、空可选输入、未支持类型和提升边界的元数据；实际输出仍读回校验和独立复检。
`--scheduler staged --search-policy cursor` 进一步启用节点/资源续扫；普通自定义 pass 保持原接口。

```bash
uv run --locked onnx-reduce input.onnx --test '/absolute/path/check-backend' \
  -o reduced.onnx --scheduler staged --goal complete \
  --model-edit-mode auto --search-policy cursor --performance
```

`--direction-order adaptive`、`--history-order defer` 是进一步的排序实验，默认关闭。
它们可能改变紧预算下的结果质量；fast 永远不证明最小性，complete 仍需在总预算内完成独立扫描。
独立 `--finalize-from` 可改变这些搜索选项，但判定命令与故障身份须保持一致。
各项性能证据与已知退化见[第五、六批结果](performance/performance-batch56-results.md)。
