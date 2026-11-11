# 缩减质量验证

本轮实现覆盖目标选择、shape / dense 权重裁剪、递归控制流子图和有界组合编辑。
全部候选由同一套合法性检查、故障判定、预算、checkpoint 和最终实际产物复检约束。

## 可复现质量基准

从仓库根目录执行：

```bash
.venv/bin/python benchmarks/quality.py --out /tmp/onnx-reduce-quality
```

脚本在同一个引擎、相同成本目标与预算下比较原有三个删除 pass 和扩展 pass。
默认每次最多 500 个候选判定、60 秒，每类组合最多 32 个候选。记录节点数、张量元素数、
字节数、判定次数、耗时和最终复检结果；完整运行证据保留在输出目录。
此基准测量候选能力的差异，不是旧版引擎与新版引擎的纯速度比较。

六个案例均为确定性生成模型，使用模拟故障条件，不能作为真实编译器缺陷的复现证据。
本次实测两侧均到固定点并通过最终复检：

| 案例 | 递归节点数（原策略→扩展） | 张量元素数（原策略→扩展） | 字节数（原策略→扩展） | 判定次数（原策略→扩展） |
|---|---|---|---|---|
| MatMul | 1→1 | 32→2 | 179→120 | 3→9 |
| Reshape | 1→1 | 18→3 | 137→140 | 3→9 |
| If | 5→3 | 17→2 | 262→228 | 3→9 |
| Loop | 4→2 | 36→6 | 325→276 | 3→14 |
| Scan | 4→2 | 96→4 | 371→319 | 4→11 |
| 成对旁路 | 4→2 | 16→1 | 118→92 | 8→36 |

Reshape 的张量规模减小，但常量编码转换使文件增加 3 字节；默认 nodes 目标允许这种结果。
需要文件最小时选择 `--objective bytes`。新增搜索一般需要更多判定，不能视为速度优化。
实测摘要见 [JSON 结果](reduction-quality-results.json)。

## 典型架构覆盖

复用了相邻 compare-bench 中 19 个典型架构生成器，并复用其原有条件：

```bash
.venv/bin/python benchmarks/quality.py --architecture-zoo ../compare-bench \
  --out /tmp/onnx-reduce-architecture-quality \
  --max-calls 200 --time-budget 15 --combination-limit 16
```

两侧共 38 次运行均到固定点、最终产物复检通过；15/19 个案例的扩展结果字节数下降，
其余 4 个相同，没有字节数回退。BERT embedding 从 128275→261 字节，LSTM 从
25930→313 字节，GRU 从 13383→286 字节，external-data embedding 从
33097→286 字节。Loop-with-If 的递归节点数 5→4。完整指标见
[架构结果 JSON](reduction-architecture-results.json)。

这些模型覆盖 CNN、Transformer、循环网络、量化、动态 shape、控制流和外部权重，
但判定条件仍是模拟条件；其中 byte/name 条件可能被残留名称元数据满足，不能据此认定
真实编译器故障仍在。新增搜索增加了判定成本，本次每个案例的调用数均高于原策略。
结果为单次质量覆盖，不提供统计意义上的性能加速结论。

## 接入真实故障

准备 JSON 列表（模型相对路径以清单目录为基准，判定命令在当前工作目录执行）：

```json
[
  {
    "name": "compiler-case",
    "model": "models/failure.onnx",
    "test": ["/absolute/path/python", "/absolute/path/check_backend.py"],
    "signature_regex": "assertion: ([A-Za-z0-9_]+)"
  }
]
```

判定脚本遵循工具契约：0 表示通过、故障不存在，普通非零表示仍复现目标故障，125 表示无法判定。
基准调用脚本时追加模型路径；脚本应将编译器崩溃或超时转换成自己的目标判断。
此基准把脚本本身的崩溃和超时视作不可接受候选，不直接视作目标故障。

```bash
.venv/bin/python benchmarks/quality.py --manifest /path/cases.json \
  --out /tmp/onnx-reduce-real-quality --max-calls 1000 --time-budget 300
```

每侧独立执行初检和最终复检；无法复现或最终复检失败会终止基准并返回非零。
应结合 `stop_reason` 区分预算停止与固定点。脚本的 --timeout 为单次判定上限。
当前尚未提供真实故障清单，因此真实编译器故障质量结论待该清单运行后补充。

## 相同时间预算下比较两个源码版本

前述质量表保留为改动时的历史结果。后续修复了组合枚举的深度/位置偏差、pass 轮转、deep 搜索，
并增加局部算子约束裁剪和收益时间序列。历史表不用于证明这些调度修复对真实模型的收益。
新增回归模型验证：80 个节点中只有末端三个非连续节点同时删除才保留条件，修复后能在
256 个组合上限、300 次判定预算内删去这三个节点；350 节点生成用例验证末端三元候选可被覆盖。

准备两个独立的源码目录（baseline 必须保留修改前的源码），复用上文的真实故障清单：

```bash
.venv/bin/python benchmarks/strategy_compare.py \
  --manifest /path/cases.json \
  --baseline-source /path/to/before \
  --current-source "$PWD" \
  --budgets 60,120,300 \
  --combination-limit 1024 \
  --out /tmp/onnx-reduce-strategy-compare
```

每次使用相同解释器、依赖、工作目录、判定命令、签名、目标与组合参数；source tree 通过
独立进程隔离。两侧均启用 staged complete 和 performance，交替先运行的版本以减少顺序偏差。
两侧源码须使用相同的判定退出码约定；旧版“0 表示故障仍在”的源码不能直接复用当前判定脚本进行对照。
summary.json 保存每个预算两侧的完整报告，运行证据、日志和实际产物位于各自工作目录。
预算包含初检，是软限制，最终复检始终执行，因此总耗时可能超过设置值。
对比最终成本、调用数、停止原因及新版本的收益曲线，不仅比较退出码。
尚未收到 1573 节点模型及其判定命令，因此目前没有该模型修改前后 60/120/300 秒的实测结论。

本轮使用同一个模拟 MatMul 故障条件做了独立源码对比：条件要求保留一个无关输入的 4×4 形状，
修改前的全局同值维度替换因此无法缩减；修改后局部收缩 K 轴不改变该无关输入。
60/120/300 秒预算下，两侧均到固定点并复检通过，修改前张量元素数均为 48，修改后均为 24。
这是局部协调候选能力的回归证据，不是对 1573 节点真实模型收益的预测，也不意味着节点数减半。
结果与基线 commit 记录见 [后续修复结果](strategy-followup-results.json)。
