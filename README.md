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

这个教学示例无需安装真实编译器，用“模型包含 `Relu`”模拟需要保留的故障条件，
不代表 `Relu` 存在真实缺陷。目标是删除无关节点，同时保留这一条件：

```text
缩减前：Identity → Relu → Identity → Identity
预期：             Relu
```

完成上述 uv 安装步骤后，从仓库根目录依次执行以下命令。
示例文件写入 `/tmp/onnx-reduce-demo/`，重复运行会覆盖同名模型与报告。

### 1. 生成原始模型

```bash
uv run --locked python examples/relu/make_model.py \
  /tmp/onnx-reduce-demo/original.onnx
```

[模型生成器](examples/relu/make_model.py)会自动创建目标目录，生成包含四个节点的
`original.onnx`，作为后续缩减的输入。

### 2. 手动验证故障判定

先直接运行[判定脚本](examples/relu/check.py)，确认原模型满足目标条件：

```bash
uv run --locked python examples/relu/check.py \
  /tmp/onnx-reduce-demo/original.onnx
echo $?
```

预期输出：

```text
assertion: DEMO_RELU
0
```

脚本加载传入的模型，检查主图是否包含 `Relu`：包含时输出目标标记并返回 `0`，
不包含时返回 `1`，模型无法加载时返回 `125`。**这里的退出码 0 表示目标条件仍在**，
它是缩减过程中判断候选能否保留的依据。

### 3. 执行缩减

```bash
uv run --locked onnx-reduce /tmp/onnx-reduce-demo/original.onnx \
  --test '.venv/bin/python examples/relu/check.py' \
  --signature-regex 'assertion: ([A-Za-z0-9_]+)' \
  -o /tmp/onnx-reduce-demo/reduced.onnx \
  --report /tmp/onnx-reduce-demo/report.json
```

这条命令中：

- `--test` 指定判定命令，工具会把候选模型的绝对路径追加为最后一个参数。
- `--signature-regex` 从输出中提取 `DEMO_RELU`，要求候选保留与原模型相同的签名。
- `-o` 指定缩减模型路径。
- `--report` 指定 JSON 统计报告路径。

工具会尝试删除节点等编辑，只有模型通过校验、成本严格下降且目标条件仍成立时才接受。
本例中可以逐步删除 `Identity`，但必须保留 `Relu`。终端会显示接受的缩减及最终复检结果。

### 4. 检查缩减模型并重新判定

列出缩减前后的节点，确认模型确实变小：

```bash
uv run --locked python - <<'PY'
import onnx

for name in ("original", "reduced"):
    model = onnx.load(f"/tmp/onnx-reduce-demo/{name}.onnx")
    operators = [node.op_type for node in model.graph.node]
    print(f"{name}: {len(operators)} nodes, {operators}")
PY
```

预期输出：

```text
original: 4 nodes, ['Identity', 'Relu', 'Identity', 'Identity']
reduced: 1 nodes, ['Relu']
```

再次运行判定脚本，确认缩减模型仍满足目标条件：

```bash
uv run --locked python examples/relu/check.py \
  /tmp/onnx-reduce-demo/reduced.onnx
echo $?
```

应继续输出 `assertion: DEMO_RELU`，退出码为 `0`。
这同时验证了“模型变小”和“目标条件保留”。

### 5. 查看报告与运行证据

本次示例生成的文件如下：

```text
/tmp/onnx-reduce-demo/
├── original.onnx       # 原始模型
├── reduced.onnx        # 缩减模型
├── report.json         # 缩减统计
└── reduced.onnx.work/  # 每次运行独立的 run-* 日志与 checkpoint 目录
```

读取报告中的关键结果：

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("/tmp/onnx-reduce-demo/report.json").read_text())
print("stop_reason:", report["stop_reason"])
print("minimality.status:", report["minimality"]["status"])
PY
```

预期输出：

```text
stop_reason: fixed_point
minimality.status: pass_fixed_point
```

`fixed_point` 表示当前缩减策略完成一轮搜索后，无法再接受更小的候选；
`pass_fixed_point` 表示最终搜索轮次没有无法判定的候选，且最终复检通过。
这一结论仅针对当前策略生成的合法、成本更低候选，不保证全局最小。
若结果与预期不同，参见[排障指南](docs/troubleshooting.md)。

跑通后，可按[使用指南](docs/usage.md)将教学判定脚本替换为实际故障判定脚本。

## 按任务阅读

| 你要完成的任务 | 阅读入口 |
|---|---|
| 查看全部文档，包括设计与性能资料 | [文档导航](docs/README.md) |
| 为自己的后端编写判定脚本、设置预算、确认结果 | [使用指南](docs/usage.md) |
| 理解候选如何生成、校验和接受，定位相关源码 | [架构说明](docs/architecture.md) |
| 排查无法启动、判定异常、最终复检失败或中断 | [排障指南](docs/troubleshooting.md) |
| 搭建开发环境、运行验证、提交贡献 | [贡献指南](CONTRIBUTING.md) |

首次接入建议按「README → 使用指南 → 排障指南」阅读；修改实现前再阅读架构说明。

## 用于实际故障

处理真实问题时，编写一个脚本，调用实际出问题的 ONNX 编译器或推理引擎，
判断传入模型是否仍触发同一故障，再通过 `--test` 调用它。
脚本需要识别具体目标诊断，并将工具自身的运行结果转换为故障判定退出码。
完整脚本、参数传入方式及信号和超时约定见[使用指南](docs/usage.md#编写故障判定脚本)。

原模型必须先复现故障；最终输出会在实际交付路径上独立复检。
CLI 退出 0 包含固定点和预算耗尽，并不保证模型一定变小。
请结合报告判断结果；[排障指南](docs/troubleshooting.md#读取退出码与报告)说明各退出码和字段。

## 适用范围

当前只缩减主图结构，面向不依赖具体输入数据的编译错误、崩溃和挂起。
缩减可能将中间值提升为新图输入，保留故障不保证计算语义等价；
固定点也不保证全局最小。子图、数值问题、模型规模等限制见
[当前边界](docs/architecture.md#当前边界)。

[MIT License](LICENSE)
