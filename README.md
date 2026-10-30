# onnx-reduce

在保留用户指定故障的前提下，将 ONNX 模型缩减为更小的复现模型，用于定位后端问题、
提交缺陷报告和构建回归测试。当前版本为 `0.1.0`，面向编译错误、崩溃和挂起等
不依赖具体输入数据的故障。

你提供一个回答“目标故障是否仍在”的命令，onnx-reduce 负责生成候选、校验模型、
执行判定、保存缩减结果和复现证据。它不内置特定编译器或推理后端。

## 安装

需要 **Python >= 3.11** 和 **Linux / macOS**；判定进程使用 POSIX 进程组，
Windows 暂不支持。CI 覆盖 Linux / macOS 上的 Python 3.11、3.12、3.13。

安装 [uv](https://docs.astral.sh/uv/) 后，从源码运行：

```bash
git clone https://github.com/XuMoheng/onnx-reduce.git
cd onnx-reduce
uv sync --locked
uv run --locked onnx-reduce --help
```

`uv sync --locked` 会安装项目及开发依赖。运行依赖是 `onnx` 和 `onnxscript`，
具体版本约束见 [pyproject.toml](pyproject.toml)，锁定环境见 [uv.lock](uv.lock)。

也可以在仓库根目录使用 pip 安装：

```bash
python -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m onnx_reduce --help
```

以下示例使用 uv，并从仓库根目录执行。

## 快速开始

仓库附带一个可直接运行的教学示例：生成 `Identity → Relu → Identity → Identity`
四节点模型，判定脚本在主图含 `Relu` 时返回 0。

```bash
uv run --locked python examples/relu/make_model.py /tmp/onnx-reduce-demo/original.onnx
uv run --locked onnx-reduce /tmp/onnx-reduce-demo/original.onnx \
  --test '.venv/bin/python examples/relu/check.py' \
  --signature-regex 'assertion: ([A-Za-z0-9_]+)' \
  -o /tmp/onnx-reduce-demo/reduced.onnx \
  --report /tmp/onnx-reduce-demo/report.json
```

结果应只保留一个 `Relu`，报告中 `stop_reason` 为 `fixed_point`，
`minimality.status` 为 `pass_fixed_point`。这只是教学判定条件，不代表 `Relu`
存在真实缺陷。示例源码见 [模型生成器](examples/relu/make_model.py)和
[判定脚本](examples/relu/check.py)。

## 接入自己的故障判定

通过 `--test CMD` 指定判定命令。工具将候选模型的**绝对路径追加为最后一个参数**，
保留调用者的工作目录和环境；命令使用 shell 风格的引号解析，但不会执行管道、
重定向、变量或通配符展开。需要这些行为时，将命令写进独立脚本。

| 判定进程结果 | 默认含义 |
|---|---|
| 退出码 `0` | 目标故障仍在（`interesting`） |
| 退出码 `125` | 无法判定（`inconclusive`） |
| 其他普通非零退出码 | 目标故障不在（`not_interesting`） |
| 被信号终止 | 故障仍在；可用 `--no-crash-interesting` 改为不在 |
| 超过 `--timeout` | 无法判定；排查挂起时可用 `--timeout-interesting` 改为仍在 |

**退出码 0 的含义是“复现了目标故障”**。后端自身的失败退出码通常需要脚本转换。
例如，下面是编译错误判定脚本的模板；将 `your-backend` 命令及错误标记替换为
真实后端的调用方式和目标诊断信息，保存为 `check_backend.py`：

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
raise SystemExit(0 if reproduced else 1)
```

```bash
uv run --locked onnx-reduce original.onnx \
  --test '.venv/bin/python check_backend.py' \
  --signature-regex 'YOUR_TARGET_ERROR' \
  --max-candidate-check-calls 100 \
  --time-budget-seconds 600 \
  -o reduced.onnx --report report.json
```

默认 `--target-mode baseline` 要求候选与原模型的判定进程以相同方式结束，
包括退出码、具体信号或超时。`--target-mode script` 信任脚本的故障判定，
不比较结束方式。两种模式都支持 `--signature-regex`：在 stdout 和 stderr 中
提取签名，有捕获组时比较捕获组，否则比较完整匹配，要求与原模型一致。
仅匹配结束方式不足以区分不同缺陷，应让脚本识别具体故障，或额外匹配稳定签名。

## 常用选项

| 选项 | 说明 |
|---|---|
| `--test CMD` | 必填，故障判定命令 |
| `-o, --output PATH` | 必填，结果路径；已有同名文件会被覆盖 |
| `--report PATH` | JSON 统计报告 |
| `--timeout SECONDS` | 单次判定超时，默认 60 秒 |
| `--max-candidate-check-calls N` | 候选实际判定调用上限；原模型检查和最终复检不计入，缓存命中不消耗配额；`0` 跳过搜索 |
| `--time-budget-seconds N` | 软墙钟预算，已开始的操作可能越界，尾检仍执行 |
| `--signature-regex REGEX` | 匹配基线故障签名，避免同类退出背后的不同错误 |
| `--target-mode baseline\|script` | 默认 `baseline`，匹配原模型的结束方式；`script` 信任脚本判定 |
| `--validation-mode strict\|structural\|permissive` | 模型校验策略，默认 `strict`，见下文 |
| `--inconclusive-code CODE` | 指定无法判定退出码，可重复；显式指定后替换默认的 `125` |
| `--no-crash-interesting` | 将被信号终止的判定进程视为未复现故障 |
| `--timeout-interesting` | 将超时视为复现故障，用于挂起问题 |
| `--no-cache` | 关闭单次运行内的内容判定缓存 |
| `--work-dir DIR` | 日志与 checkpoint 根目录，默认 `OUTPUT.work` |

完整参数见 `uv run --locked onnx-reduce --help`。旧参数 `--max-tests` 和
`--checker` 已移除，请分别使用 `--max-candidate-check-calls` 和 `--validation-mode`。

## 缩减与校验方式

默认依次运行三个缩减 pass，每接受一次候选就从第一个 pass 重新开始：

1. `drop_outputs`：逐个裁剪主图输出，至少保留一个输出。
2. `remove_unused`：尝试清理不影响图输出的节点、未使用输入和权重。
3. `delete_nodes`：从较大的连续节点块到单节点，尝试删除主图节点；必要时将
   删除边界的中间值提升为新图输入，并尝试推断类型和形状。

包括清理在内的所有编辑都必须通过故障判定。候选还必须通过所选校验，并按
`(node_count, byte_size, output_count, input_count, initializer_count)` 的字典序
严格降低成本；节点更少时允许文件更大。

- `strict`：执行 ONNX checker 和完整检查；发现非内置 domain 时自动使用 `structural`。
- `structural`：仍执行 ONNX checker 和完整检查，但关闭自定义 domain 的 schema 检查，
  不保证后端支持相应算子。
- `permissive`：只执行基本图输出检查，跳过 ONNX checker；适合有意保留某些
  不满足完整 ONNX 校验的模型。

校验同样适用于原模型。报告记录 `requested_validation_mode`、
`effective_validation_modes` 和 `custom_domains`，便于确认实际使用的策略。

## 结果与边界

原模型首先必须复现故障；缩减结束后，输出文件会在实际交付路径上绕过缓存独立复检。
CLI 的退出码与判定脚本的退出码含义不同：

| CLI 退出码 | 含义 |
|---|---|
| `0` | 正常结束，包括固定点和预算耗尽；不保证模型一定变小 |
| `1` | 最终输出复检未通过，`stop_reason=unstable_check` |
| `2` | 配置、输入或执行错误，包括原模型无法复现故障 |
| `130` | Ctrl-C 中断；最终输出可能尚未验证 |

JSON 报告当前为 **schema version 3**，主要字段包括：

- `original` / `final`：节点数、字节数、输入、输出和权重数量。
- `failure_checks`：实际调用数（含原模型检查和最终复检）、缓存命中、故障签名
  或结束方式不匹配次数，以及各类判定结果计数。
- `candidates` / `passes`：候选生成、无效、接受和成本未下降等统计。
- `stop_reason`：`fixed_point`、`budget_exhausted` 或 `unstable_check`。
- `minimality.status`：只有搜索到固定点、最终一轮没有无法判定的候选且复检通过时，
  才为 `pass_fixed_point`；否则为 `unproven`。历史轮次出现无法判定并不单独阻止此结论。

固定点只针对当前 pass 生成的合法、成本更低的候选，**不保证全局最小**。
时间预算为软限制，已开始的生成、编辑和校验可能越界，最终复检仍会执行。

## 日志与中断恢复

默认在 `OUTPUT.work/` 下为每次运行创建独立的 `run-*` 目录，也可通过
`--work-dir` 指定根目录。关键文件包括：

| 路径 | 内容 |
|---|---|
| `task.json` / `target.json` | 配置、工作目录与基线故障身份 |
| `candidates.jsonl` / `evaluations.jsonl` / `accepted.jsonl` | 候选决策、判定记录与接受记录；有相应事件时生成 |
| `checks/*/stdout.log` / `stderr.log` | 判定进程日志，每个流最多保留 1 MiB |
| `checkpoint.json` / `checkpoint/*.onnx` | 最近已接受的模型及其统计，首次成功判定后创建 |
| `report.json` / `replay.json` | 正常收尾时写出的报告及复现所需配置 |

`--report` 会额外将报告写到指定位置。原模型判定失败时，可从 `evaluations.jsonl`
和 `checks/` 排查命令、退出码、超时或签名；最终复检失败时还应检查路径依赖或故障随机性。

中断后，读取对应 `checkpoint.json` 的 `model` 字段，将该模型路径作为新的输入，
使用相同判定命令重新运行。这会开启新一轮搜索，不会自动恢复旧缓存或剩余预算。

## 当前限制

- 保留故障不要求计算语义等价；删除边界可能将中间值变成新的图输入。
- 只编辑主图；暂不编辑子图内部、属性或权重数值，也不缩减输入数据或比较数值差异。
- external data 读入内存并内嵌到输出；不支持超过 2 GB 的模型，候选留盘也会占用磁盘。
- 缓存要求固定环境及基于内容的判定；首尾复检不能保证随机故障的统计可靠性。
- 判定命令以调用者权限执行；工具不提供执行沙箱。

## 开发与验证

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests examples
uv run --locked ruff format --check src tests examples
uv build
uv run --no-project --with twine python -m twine check --strict dist/*
```

CI 还会验证 wheel 的隔离安装，并从源码分发包解压后运行测试。`README.md` 被用作
发布包的长描述，必须有有效内容，才能通过 `twine check --strict`。

源码入口：

- [cli.py](src/onnx_reduce/cli.py)：命令行与退出码。
- [engine.py](src/onnx_reduce/engine.py)：搜索、预算、checkpoint 和最终复检。
- [model.py](src/onnx_reduce/model.py)：基于 `onnxscript.ir` 的模型编辑、校验与序列化。
- [failure_check/](src/onnx_reduce/failure_check/)：判定契约、进程执行、故障匹配和缓存。
- [passes/](src/onnx_reduce/passes/)：内置缩减策略。
- [report.py](src/onnx_reduce/report.py)：JSON 报告格式。
- [tests/](tests/)：行为与回归测试；[CI 配置](.github/workflows/ci.yml)包含分发包验证步骤。

[MIT License](LICENSE)
