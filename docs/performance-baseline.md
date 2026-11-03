# 性能基线

性能采集默认关闭。开启后，报告增加可选的 `performance` 字段；原有搜索顺序、接受条件、
缓存和预算规则保持不变。采集本身有开销，固定时间预算下可能影响最终缩减质量，需实测。

## 运行基准

从 `1012-perf/`（包含 `pyproject.toml` 的仓库根目录）执行，先按贡献指南安装依赖：

```bash
uv run --locked python benchmarks/reduction.py \
  --case initializers --size 512 --repeat 3 \
  --budget both --candidate-budget 600 --time-budget 5 \
  --compare-instrumentation
```

该场景包含一个必须保留的 Relu 和 512 个无用 initializer。
判定替身在当前进程加载模型并检查 Relu，适合观察缩减器自身成本；不代表真实后端性能。
每次缩减使用新进程、新模型和独立产物目录，避免复用模型快照或进程峰值内存。
脚本交替开启/关闭采集的执行顺序，记录中位数和观测到的耗时差；短运行差异可能来自噪声。
基准不并发运行，不要在同时运行测试或其他负载时采样。

默认输出到 `output/performance/<批次时间>/baseline.json`，包括环境、参数、每次报告、
墙钟耗时、进程峰值内存和汇总；各运行保留候选日志、checkpoint 与最终模型。
固定检查预算按有序候选决策日志的 SHA-256、最终成本及内容摘要、计数和停止原因验证
开关前后一致；不一致会报错。时间预算只比较结果质量，不要求候选轨迹一致。

其他合成场景可分别选择 `chain`、`branches`、`outputs`、`missing-shape`，例如：

```bash
uv run --locked python benchmarks/reduction.py \
  --case all --size 32 --repeat 3 --budget both \
  --candidate-budget 100 --time-budget 2 --compare-instrumentation
```

`--size` 对 initializer 场景表示权重数量，对链图表示节点数量，对分支/多输出图表示分支数量。
链图包含中间值的 shape 信息；`missing-shape` 省略它们，用于暴露提升边界时的 shape 推导成本。
运行到预算上限的结果与固定点结果需区分；检查配额不含强制初检、最终复检，缓存命中也不消耗配额。

真实后端需要用户自己的稳定故障模型和判定命令：

```bash
uv run --locked python benchmarks/reduction.py --case backend \
  --model /absolute/path/failing.onnx \
  --test '/absolute/path/check-backend' \
  --signature-regex 'assertion: ([A-Za-z0-9_]+)' \
  --repeat 3 --budget both --candidate-budget 100 --time-budget 60 \
  --compare-instrumentation
```

判定命令收到候选模型路径作为最后一个参数，退出 0 表示目标故障仍在。
此入口使用实际判定进程，按默认规则匹配终止身份和可选签名。
没有真实故障模型时，不把合成场景结果视为真实后端验收。

### 准备缓存基准

`preparation` 不包含在 `all` 的五类场景中。它生成省略中间 shape 的链图，对每个节点
依次生成单次删除、重复索引删除、相同单次删除三种候选。判定条件是必须保留全部节点，
因此所有候选均被拒绝，可以分别观察 shape 共享、相同内容和相同编辑的命中。
以下两条命令使用相同候选序列，分别开启和关闭三类缓存：

```bash
uv run --locked python benchmarks/reduction.py \
  --case preparation --size 128 --repeat 3 --budget calls \
  --candidate-budget 600 --compare-instrumentation
uv run --locked python benchmarks/reduction.py \
  --case preparation --size 128 --repeat 3 --budget calls \
  --candidate-budget 600 --compare-instrumentation --no-cache
```

缓存开关会改变实际判定次数；该对照应使用足够完成扫描的调用预算，并比较编辑、状态、
内容摘要与最终结果，不能要求包含命中字段的原始日志字节完全一致。
采集开关的原有严格轨迹对照仍分别执行。

## 对单次缩减采集

在已有 CLI 命令中增加 `--performance`，例如：

```bash
uv run --locked python examples/relu/make_model.py /tmp/onnx-perf/original.onnx
uv run --locked onnx-reduce /tmp/onnx-perf/original.onnx \
  --test '.venv/bin/python examples/relu/check.py' \
  -o /tmp/onnx-perf/reduced.onnx --performance \
  --report /tmp/onnx-perf/report.json
```

Python 接口对应 `EngineConfig(collect_performance=True)`。

## 统计口径

| 字段 | 含义 |
| --- | --- |
| `stages.*.calls` | 实际进入阶段的次数；缓存属性只在首次计算时计数 |
| `inclusive_seconds` | 阶段总耗时，包含内部被采集的阶段 |
| `exclusive_seconds` | 扣除内部采集阶段后的自身耗时，可跨阶段相加 |
| `elapsed_seconds` | 从 engine 入口到内部报告和复现配置写完的采集窗口 |
| `counters.written_bytes` | 模型、JSON、JSONL、判定输出日志提交给文件系统的累计字节，包含覆盖和重复写入 |
| `counters.write_calls` | 上述文件写入次数，空日志也计一次 |
| `counters.candidate_status.*` | 候选拒绝/判定/预算停止的原因，细节仍在 `candidates.jsonl` |
| `counters.content_cache_*` | 候选内容缓存命中、未命中或禁用次数，不含强制首尾检查 |
| `counters.edit_cache_*` / `shape_cache_*` | 编辑准备和 shape 摘要的命中、未命中或禁用次数 |
| `counters.*_cache_evictions` / `*_cache_oversized` | 各缓存的淘汰、超限不保留次数 |
| `samples.*_cache_bytes` / `*_cache_entries` | 各缓存保留的编码键值字节数和项数采样，`max` 为本次运行峰值 |
| `samples.*` | 候选编辑数量、选择项数量、编辑后节点/字节规模的数量、总和、最大值 |
| `peak_rss_bytes` | 当前缩减进程生命周期的峰值 RSS，含原生库分配，排除子进程 |

采集阶段包括 `candidate_generation`、`clone_and_edit`、`clone`、`shape_inference`、
`export`、`serialization`、`validation`、`hash`、`write`、`external_check`、`checkpoint`。
候选生成测量实际的惰性迭代，包括迭代结束；shape 推导仅在确实缺失边界信息时计数。
ONNX checker 内部的 shape 检查包含在 `validation`，不与显式 `shape_inference` 混算。
`serialization` 包含导出，`clone_and_edit` 包含克隆/推导，checkpoint 包含落盘，
因此分析主要耗时应按 **exclusive_seconds** 排序，不能累加 inclusive_seconds。
未进入的阶段不出现在字典中。

计时窗口不含 CLI 输入加载、基准模型构建和最终冻结的性能报告写入；checkpoint 报告为
当时的快照，最终 `report.json` 才包含完整的采集窗口。累计写入量不含最终冻结报告、
CLI 额外指定报告和后端自行写的文件，也不是磁盘物理写入量。RSS 含 Python 启动和输入加载，
不等于本次缩减的内存增量；真实后端子进程内存需另外测量。
阶段之外的 Python 调度、统计维护等开销是 elapsed 与 exclusive 总和之间的余量。

## 验收

1. 所有合成场景可重复运行，固定检查预算下采集开关前后的候选轨迹与最终内容一致。
2. 报告能区分主要耗时来源、重复导出/推导、实际判定与缓存命中，以及累计写入。
3. 同时报出固定检查预算和固定时间预算下的最终完整成本元组。
4. 单独记录真实后端验证状态；未提供模型与判定器时保留为待验证项。

本批仅建立测量基础；批量清理、准备缓存和有状态调度器在后续独立改动中实施。
