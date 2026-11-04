# 职责重构验收

本轮将搜索、模型查询与编辑、候选执行、记录和产物分开。默认参数、候选顺序、严格成本下降、
独立首尾判定及报告 version 7 保持兼容。当前模块和配置边界见[架构说明](../architecture.md)。

## 对照方法

基线为 `21b7b0f`，即职责拆分前的版本。`benchmarks/refactor_compare.py` 对两份源码逐次启动
独立 Python 进程，使用相同解释器、依赖、工作目录、模型构造和模拟故障条件；交替两侧顺序，
所有最终产物均独立复检。每次保留完整日志和报告。

固定调用预算为 16 次候选实际判定，首尾判定不计入；另对成对旁路案例完整扫描。
对照有序 candidates.jsonl、accepted.jsonl、scheduler.jsonl 的 SHA-256，以及完整最终成本、
模型摘要、停止原因、最小性、pass 统计、判定计数、覆盖摘要/版本/完成状态。
时间与绝对产物路径不参与轨迹相等判断。脚本遇到决策差异立即失败。

模型包括 MatMul、Reshape、If、Loop、Scan、成对旁路、链图、门控链图、无用 initializer 和
重复准备场景。覆盖 legacy/stateful/staged、IR/auto、缓存开/关；另测 staged cursor + defer +
adaptive，并关闭校验缓存与 checkpoint 复用。构造模型和模拟条件不能证明真实编译器故障仍在。

时间预算为 0.5 秒，链图、门控链图和准备压力场景均使用 256 节点，每配置重复 3 次。
三种调度器和两种编辑后端均测量，缓存开启。时间预算是软限制，比较最终质量和开销，
不要求时限附近的候选轨迹相同。

## 复现命令

从仓库根目录执行；基线目录仅保存已提交源码：

```bash
mkdir -p /tmp/onnx-reduce-before-refactor
git archive 21b7b0f | tar -x -C /tmp/onnx-reduce-before-refactor
.venv/bin/python benchmarks/refactor_compare.py \
  --baseline-source /tmp/onnx-reduce-before-refactor \
  --out output/refactor-verification --time-budget 0.5 --repeat 3
.venv/bin/python benchmarks/refactor_compare.py \
  --baseline-source /tmp/onnx-reduce-before-refactor \
  --out output/refactor-experimental-verification --experimental-only
```

每个输出目录的 summary.json 包含配置、逐次决策摘要、计时、原始报告路径和汇总。
命令可用 `--cases gated` 筛选案例。性能采样应串行执行，避免同时跑测试或其他负载。
本次常规矩阵分两次运行，门控补测位于 output/refactor-gated-verification；配置与完整命令相同。

## 本次结果

- 固定调用/完整扫描：154 组决策对照完全一致。
- 固定时间：54 组对照，18 个配置各重复 3 次；最终完整成本的中位结果一致。
- 被采集的非判定阶段累计耗时，中位数比值为原版的 0.932–1.004 倍。该指标累加各阶段的
  exclusive_seconds，包含准备、调度、记录和 checkpoint，不含 external_check；不覆盖未被采集的 Python 开销。
- 准备压力场景的候选吞吐略有波动；没有据此声称加速。短预算、单机测量不构成统计性能保证。
- 当前 439 项测试通过；静态检查、格式检查、wheel/sdist 构建、已安装 wheel 的隔离示例验证通过。

| 256 节点场景 | 配置 | 原版→重构后节点数（3 次中位结果） |
|---|---|---|
| 链图 | 三种调度器 × 两种编辑后端 | 1→1 |
| 门控链图 | legacy / IR、auto | 256→256 |
| 门控链图 | stateful / IR | 250→250 |
| 门控链图 | stateful / auto | 249→249 |
| 门控链图 | staged / IR | 256→256 |
| 门控链图 | staged / auto | 251→251 |
| 准备压力 | 三种调度器 × 两种编辑后端 | 256→256（条件要求全部节点保留） |

节点数表仅供浏览；[证据摘要](refactor-evidence.json) 保存完整成本与环境。原始输出位于
Git 忽略的 output/refactor-*verification/，包含每次 report.json、独立复检、全部候选及接受记录。
计时反馈现在在结果交给记录模块前完成，scheduler.buckets 中的 preparation_seconds 不再包含
后续候选记录的时间；报告字段保持原样，比较运行总开销应使用上述阶段累计或 elapsed_seconds。

真实后端模型和故障判定器未参与本次验收。实际接入可使用
[相同清单的双源码时间对照](reduction-quality.md#接入真实故障)及 benchmarks/strategy_compare.py；
本次结论限于行为回归和模拟负载，不能推广为真实后端缩减质量或速度保证。
