# 性能基线实测：2026-11-02

本次实现完成了第一批的采集、合成基准和行为对照。真实后端故障仍待提供模型及判定器后验收。

## 输入模型与判定条件

本报告的性能数据来自基准脚本人工生成的 ONNX 模型，生成逻辑见
[benchmarks/reduction.py 的 `make_model()`](../../benchmarks/reduction.py)。没有使用实际业务网络或真实后端故障模型。
所有模型均使用标准 ONNX 算子、默认 domain 的 opset 17，只包含主图，没有属性子图。
图输入均为 `x`，类型为 `float32`、形状为 `[1]`。基准不生成输入数据，也不执行数值推理。

### 主要基线：512 个无用 initializer

`--case initializers --size 512` 生成的输入模型如下：

```text
主图计算路径：
  x: float32[1] ── Relu ──► y: float32[1]

同一主图中另有 512 个 initializer，均未连接到任何节点，也不是图输出：
  unused_0000: float32[1] = [1.0]
  unused_0001: float32[1] = [1.0]
  ...
  unused_0511: float32[1] = [1.0]
```

缩减前只有 **1 个节点、1 个输入、1 个输出和 512 个 initializer**，序列化大小为 **26698 字节**。
512 个 initializer 的张量数据合计只有 2048 字节，其余大小来自名称、类型、图结构等序列化信息；
这是用于暴露“逐项删除资源”的小模型，不是大权重模型。它测量的是在保留 Relu 的条件下，
清理这些无用 initializer 所产生的候选准备、判定和 checkpoint 开销。

### 其他合成输入

五类场景回归使用 `--case all --size 32`，其中 `size` 的含义因场景而异：
initializer 场景表示权重数量，链图表示节点数量，分支和多输出图表示分支数量。
下表的原始成本按 `(nodes, bytes, outputs, inputs, initializers)` 排列，字节数取自当次报告的
`original.byte_size`，即适配器导出的自包含序列化快照大小。

| 场景 | 缩减前的图结构与 shape 信息 | 原始成本 |
| --- | --- | --- |
| `initializers` | `x → Relu → y`，另附 32 个未使用的 `float32[1] = [1.0]` initializer；与 512 场景结构相同 | `(1, 1737, 1, 1, 32)` |
| `chain` | `x → 16 个 Identity → Relu → 15 个 Identity → y`；共 32 个串联节点，Relu 位于从 0 开始的索引 16；输入、输出和所有中间值均声明 `float32[1]` | `(32, 1307, 1, 1, 0)` |
| `branches` | `x → Relu → trigger` 后分成 32 路，每路经过一个 Identity，再由 `Concat(axis=0)` 汇合为 `y`；共 34 个节点，中间值均声明 `float32[1]`，输出 `y` 为 `float32[32]` | `(34, 2271, 1, 1, 0)` |
| `outputs` | `x → Relu → trigger` 后分成 32 路，每路经过一个 Identity，直接成为图输出 `branch_0` 至 `branch_31`；没有 Concat，共 33 个节点、32 个输出，各值均声明 `float32[1]` | `(33, 1884, 32, 1, 0)` |
| `missing-shape` | 节点连接与 `chain` 相同，但仅图输入和输出声明 `float32[1]`；省略所有中间值的 `value_info`，因此中间类型和 shape 需要在必要时推导 | `(32, 736, 1, 1, 0)` |

除 initializer 场景外，其余输入均不含 initializer。`chain` 与 `missing-shape` 的差别是中间值元信息，
用于对照边界提升时的显式 shape 推导成本。

### 保留什么条件

所有合成场景共用脚本中的 `ReluCheck`：在缩减进程内调用 `onnx.load()` 读取候选，
只要主图仍含至少一个 `Relu` 就返回 `interesting`，否则返回 `not_interesting`。
模型有效性由缩减器另行校验。这个条件模拟需要保留的故障，不表示 Relu 存在真实缺陷，
也不要求候选与原模型数值等价。

因此，报告中的 `external_check` 在这些场景下实际是“加载模型并查找 Relu”的耗时，
不含启动后端进程、编译或推理的成本。以下耗时占比和吞吐只适用于上述合成输入及判定条件。

## 环境与复现

- 环境：Linux-7.0.0-38-generic-x86_64-with-glibc2.43，Python 3.11.17，ONNX 1.23.2，onnxscript 0.7.2。
- 依赖安装受离线缓存限制，使用相邻 `1012/.venv/bin/python` 加载本仓库源码；未修改相邻仓库。
- 每项重复三次，每次使用新进程，开关采集的顺序交替执行；判定器为进程内 Relu 条件替身。
- 以下耗时为中位数，不含输入模型构建和 Python 启动。所有最终成本包含主图节点、序列化字节、输出、输入和 initializer。
- 512 场景原始记录：`output/performance/20261102-151240/baseline.json`。
- 五类合成场景原始记录：`output/performance/20261102-151316/baseline.json`。
- 原始记录位于被 Git 忽略的 `output/`，包含源码 SHA-256、环境、参数和逐次报告；完整口径见[性能基线](performance-baseline.md)。

```bash
../1012/.venv/bin/python benchmarks/reduction.py \
  --case initializers --size 512 --repeat 3 --budget both \
  --candidate-budget 600 --time-budget 1 --compare-instrumentation
../1012/.venv/bin/python benchmarks/reduction.py \
  --case all --size 32 --repeat 3 --budget both \
  --candidate-budget 100 --time-budget 0.02 --compare-instrumentation
```

## 512 个无用 initializer

检查预算 600 次时，三轮全部达到固定点，保留一个 Relu，删除全部 512 个 initializer。
最终成本 `(nodes, bytes, outputs, inputs, initializers) = (1, 72, 1, 1, 0)`。
每次产生 513 个候选，接受 512 个；实际判定 515 次 = 512 次清理 + 1 次删除 Relu 被拒绝 + 强制首尾检查各 1 次。
采集开关前后的有序候选决策、最终内容摘要、计数和停止原因完全一致。

| 模式 | 墙钟耗时中位数 |
| --- | ---: |
| 关闭采集 | 3.637 s |
| 开启采集 | 3.749 s |

观测采集开销约 3.1%。开启时峰值 RSS 中位数 120.5 MiB，累计写入约 16.14 MiB，写入 4117 次，checkpoint 514 次。

按阶段自身耗时统计（扣除嵌套阶段），以开启采集的总窗口中位数为分母：

| 阶段 | 次数 | 自身耗时中位数 | 占总窗口 |
| --- | ---: | ---: | ---: |
| export | 1028 | 1.960 s | 52.3% |
| clone | 513 | 0.669 s | 17.8% |
| write | 4117 | 0.287 s | 7.7% |
| validation | 514 | 0.268 s | 7.1% |
| checkpoint | 514 | 0.165 s | 4.4% |
| external_check | 515 | 0.081 s | 2.2% |
| candidate_generation | 2055 | 0.046 s | 1.2% |
| clone_and_edit | 513 | 0.042 s | 1.1% |
| serialization | 514 | 0.022 s | 0.6% |
| hash | 514 | 0.006 s | 0.2% |

各项独立取中位数，不要求表中耗时精确相加到总窗口。剩余时间包含阶段之外的 Python 控制逻辑和采集维护。
本场景无需显式补全边界 shape，因此无 `shape_inference` 阶段。校验内部的 shape 检查属于 `validation`。

**主要发现：导出和克隆合计约占 70%。** 共发生 1028 次导出；候选序列化和校验分别导出，确实存在准备成本放大。
外部判定替身仅约占 2%，不能据此推断真实后端也有相同比例。逐项清理还带来了 512 次成功检查及相应 checkpoint。

## 同等时间预算的结果质量

512 场景使用 1 秒软预算，三次结果均因预算耗尽停止，主图均保留一个节点、一个输入和一个输出：

| 采集 | 剩余 initializer：三次结果 | 最终字节：三次结果 |
| --- | --- | --- |
| 关 | 430, 429, 429 | 22434, 22382, 22382 |
| 开 | 434, 432, 431 | 22642, 22538, 22486 |

计时开销会消耗时间预算，所以此处不要求轨迹或最终大小一致。实际墙钟略超预算，因为检查点、同步准备和最终复检仍须完成。

## 五类场景回归

规模均设为 32，固定检查预算 100 次；下表为三次中位数，全部达到固定点。
每类每次的采集开关对照都通过了有序决策日志和最终内容的一致性检查。

| 场景 | 关闭采集 | 开启采集 | 判定次数（含首尾） | 最终成本 `(nodes, bytes, outputs, inputs, initializers)` |
| --- | ---: | ---: | ---: | --- |
| initializers | 0.058 s | 0.066 s | 35 | `(1, 72, 1, 1, 0)` |
| chain | 0.023 s | 0.026 s | 13 | `(1, 107, 1, 2, 0)` |
| branches | 0.027 s | 0.031 s | 15 | `(1, 114, 1, 2, 0)` |
| outputs | 0.108 s | 0.117 s | 37 | `(1, 127, 1, 2, 0)` |
| missing-shape | 0.025 s | 0.027 s | 13 | `(1, 96, 1, 2, 0)` |

同时运行了 0.02 秒预算对照，所有场景均报告预算耗尽并保留有效产物；该极短预算用于验证预算路径，不作为稳定的吞吐排名。逐次完整成本见原始 JSON。
短运行中，采集额外增加约 2–10 ms，观测相对开销高于 512 场景，应保持默认关闭。

## 后续顺序

1. 先做第二批的批量清理，用相同 512 场景证明候选判定和 checkpoint 次数显著减少。
2. 独立实现准备缓存与快照复用，重点比较 `export`、`clone`、`validation` 和累计写入量。
3. 获得真实后端故障后重复两类预算评估，再据此决定调度器和并行方案。

真实后端的耗时及内存尚未测量；`backend` 命令入口可用，但教学判定器冒烟测试不能替代真实后端故障验收。

## 实现验证

- 全量测试：`103 passed`；新增覆盖嵌套计时、异常清理、UTF-8 累计写入、缓存命中、拒绝原因、shape 推导失败及采集开关的决策一致性。
- Ruff 检查与格式检查通过，范围包括 `src tests examples benchmarks`。
- `uv build --offline` 成功生成 wheel 和源码包；源码包包含基准脚本和本次文档。
- `backend` 入口用教学 Relu 判定脚本完成了进程调用和签名匹配的冒烟验证，采集开关的固定检查预算对照一致。
- `twine check --strict` 未完成：离线环境未缓存 twine，未进行发布元数据的该项校验。
