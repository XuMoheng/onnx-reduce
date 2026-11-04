# 第五、六批性能实测：2026-11-03

第五批准备优化和第六批 staged 续扫已实现，默认继续使用 `legacy` / `model_edit_mode=ir`。
显式启用 `--model-edit-mode auto` 获得准备加速；增加
`--scheduler staged --search-policy cursor` 试验接受后续扫。设计与边界见
[第五、六批设计](../design/free-try-followup-design.md)。

本批实测的主要收益是快速编辑与 fine 游标。方向排序和拒绝历史没有显示额外的稳定收益，
仅保留显式实验/消融入口，默认分别为 fixed、off，不推荐作为常规配置。
所有结果来自合成故障，不能据此承诺真实后端普遍加速。

## 方法与证据

基线为 perf 的 `3f7cb3a64d84c6ce1324a94da24c9f3cd2c86fd1`，不是 free-try 的 main 基线。
原 perf 源码先冻结到独立目录，只为基准移入同一份 gated / one-stubborn 输入与判定定义。
`prepare_batch56_baseline.py` 可重建该目录，已核对原始源码逐字节一致、基准定义 AST 一致。

环境：Linux x86-64、Python 3.11、onnx 1.23.2、onnxscript 0.7.2、onnx-ir 1.0.0、protobuf 7.36.2。
合成判定在进程内加载模型；不含外部编译器启动或推理执行。每项三次、新进程、新运行目录，
同一批内交替配置顺序，串行采样。大规模扩展单独分批，不宣称这些批次也交替了源码版本。
时间为 `engine.run()` 墙钟的三次中位数，不含模型构建和 Python 启动。

共 **714 次缩减**，不计探索性 smoke 和教学 CLI 验证：

| 批次 | 次数 | 内容 |
| --- | ---: | --- |
| 第五批主矩阵 | 285 | 五类输入、1000 节点、重复准备、关闭缓存，legacy/stateful/staged 对照与五步消融 |
| 第五批扩展 | 45 | gated-90，以及 2000 initializer / one-stubborn；同样五步消融 |
| 第六批主矩阵 | 270 | 15 个场景、6 种 staged 配置、三次重复；包括 0.2 秒预算 |
| gated-300 完整收敛 | 6 | restart-complete 与 cursor-complete，配额 50000 |
| 采集开关对照 | 48 | tight、threshold、old-rejection、gated-90；cursor / history，开启与关闭采集 |
| 5000 节点扩展 | 48 | chain / missing-shape，原 perf 与 auto，四种 scheduler/goal |
| 1 / 5 秒预算扩展 | 12 | gated-90、cursor-complete，采集开关各三次 |

完整源码 SHA-256、命令、环境、逐配置结果与原始记录位置见
[证据摘要](performance-batch56-evidence.json)。原始运行目录保留在本工作区的
`output/performance/`，不随 Git 提交。`audit_batch56.py` 独立核验了全部 714 份实际产物的
摘要、完整成本、ONNX full checker、首尾缓存绕过及固定点对应的覆盖版本。
另有 **288 对**原 perf / 当前源码的候选与调度轨迹等价检查，通过最终内容、计数和停止结论对照。
这些配对包含重复运行，不代表 288 个独立模型。

## 第五批：准备成本下降，资源清理基本持平

五步消融依次为：原 perf → 快照 IR（关闭新校验缓存和 checkpoint 复用）→
auto 快速编辑 → 新校验/域缓存 → checkpoint 去重。以下比较原 perf 与完整第五批配置，
搜索顺序保持不变，实际判定次数一致：

| 场景 / 策略 | 原 perf（ms） | 第五批 auto（ms） | 实际判定 |
| --- | ---: | ---: | ---: |
| chain-512 / legacy | 113.25 | 89.21 | 21 |
| branches-512 / legacy | 144.19 | 88.26 | 23 |
| missing-shape-512 / legacy | 127.08 | 106.58 | 21 |
| outputs-512 / legacy | 64.0 | 60.7 | 7 |
| initializers-512 / legacy | 17.19 | 17.10 | 4 |
| chain-1000 / legacy | 145.65 | 105.88 | 25 |
| missing-shape-1000 / legacy | 194.2 | 156.5 | 25 |
| preparation-128 / legacy | 1058.94 | 313.08 | 130 |
| gated-90 / staged-complete | 4510.64 | 2031.02 | 1449 |
| initializers-2000 / legacy | 41.81 | 42.41 | 4 |
| one-stubborn-2000 / legacy | 143.98 | 147.91 | 23 |

链图 512 的端到端改善约 21%，分支图约 39%。按 scheduler 累计的候选准备路径耗时，
chain-512、branches-512、gated-90 分别降低 **30.74%、50.44%、64.66%**，
通过设计中准备成本至少降低 20% 的目标。该路径包含候选准备和物化等工作，不等同于纯编辑微基准。
本矩阵 stored 相对 before 的峰值 RSS 中位数最高增幅约 1.17%，未触及 10% 调查阈值。

仅增加快照、不使用快速编辑的 snapshot-ir 组没有收益，部分样本略慢。
准备侧收益主要来自 protobuf 删除，不能归因于“又消除了一次原本不存在的重复导出”。
preparation 场景中 fast 为 360.04 ms、cached 为 315.21 ms；新校验缓存将 checker 调用
从 258 次降到 130 次，实际故障判定仍为 130 次。常规链/分支中缓存并无额外稳定的墙钟收益。

资源清理的调用次数早已由第二批降低，本批 2000 initializer 和 one-stubborn 仍分别为 4、23 次；
小幅耗时变化没有一般加速结论。checkpoint 去重省去重复模型写入，但独立摘要核验和 manifest
仍有成本，未观察到稳定端到端收益。新增报告/计数也会增加 JSON 写入，不能把模型写入减少
直接解释为总 `written_bytes` 必定减少。

5000 节点扩展中，legacy 的 chain 为 **747.8 → 447.9 ms**，missing-shape 为
**924.7 → 649.6 ms**；stateful 分别为 **669.6 → 301.0 ms**、**762.8 → 414.0 ms**。
四种 scheduler/goal 的原 perf 与 auto 均得到相同完整成本、最终内容及候选轨迹。

## 第六批：续扫消除 gated 的重复搜索

以下固定第五批 auto 实现，仅切换 staged 的 restart / cursor；两者均以 complete 目标收尾：

| 场景 | restart（ms / 总判定） | cursor（ms / 总判定） | 结果 |
| --- | --- | --- | --- |
| gated-30 | 258 / 206 | 181 / 115 | 同成本，均证明固定点 |
| gated-90 | 2018 / 1449 | 566 / 302 | 同成本，均证明固定点 |
| gated-300，1000 次候选配额 | 2901 / 1002 | 2779 / 810 | restart 停在 286 节点；cursor 到 100 节点并证明固定点 |
| gated-300，完整收敛 | 54627 / 22787 | 2802 / 810 | 同成本，均证明固定点 |

完整收敛的三次中位数为 **54.63 s → 2.80 s，约 19.5x**。
总判定降低约 **96.4%**；总数包括初检和最终复检，因此候选判定分别为 22785、808 次。
两边最终完整成本都是 `(100, 7656, 1, 101, 0)`。
最终字节摘要不同，符合策略顺序变化导致的等成本固定点差异；不宣称两种策略内容等价。

本批固定调用预算的 complete 策略配对共 120 对：**111 对成本持平、9 对改善、0 对退化**。
改善来自紧预算 gated-300；这些包含三次重复及多个排序变体，不代表 120 个独立故障模型。
cursor-fast 在 gated-300 同预算到达 100 节点，总判定 547 次，仍始终报告 unproven。

## 保留的退化与没有收益的改动

- threshold-128：restart-complete 与 cursor-complete 都用满 102 次总判定，得到 32 节点但未证明固定点。
  耗时约 159 → 132 ms；没有解决 legacy 在第四批只用 68 次判定完成覆盖的差距。
- 16/12/6：两者均为 23 次判定、约 37 ms、6 节点并证明固定点；没有回退到 stateful 的旧尺度问题，
  但仍多于第四批 legacy 的 16 次判定。
- cross-region：两者均为 60 次判定、约 109 ms；stateful 的 44 次判定优势仍在。
- tight-calls=4：两者均为 6 次总判定，得到 `(8, 395, 1, 2, 0)`，最小性未证明；没有偷加 cleanup 配额。
- adaptive 与 history 的判定次数通常不变；history 存在排序维护开销，未展示稳定的额外收益。
  不推广排序实验，也不切换默认 scheduler。历史排序不提供跨模型拒绝结论或固定点证据。

0.2 秒 gated-90 中，restart-complete 三次都保留 84 节点，cursor-complete 都保留 68 节点，
均未证明固定点；时间预算是软限制，最终复检计入总墙钟。
独立的 1 秒 / 5 秒 cursor-complete 对照均以 302 次总判定完成固定点；开启采集中位数约 567 ms，
关闭采集约 0.52–0.55 s。不能把采集成本算作算法收益，也不要求时间预算轨迹相同。

## 正确性与接口验证

- Python 3.11、3.13 下全量测试均为 **363 passed**，Ruff 检查、格式检查和差异空白检查通过。
- 快速/IR 差分覆盖 metadata 回退、多输出、非连续/逆序/重复/越界删除、符号 shape、推导失败、
  外部权重、initializer 同名 input、组合和后续编辑、区域摘要、ID 与三种校验模式。
- cursor 覆盖当前版本重绑定、新资源发现、旧拒绝反转、有界预取及接受后回退、普通自定义 pass、
  无节点协议适配器、预算与 fast 保证、收尾接受后重扫、采集开关一致性。
- 教学 CLI 完成 auto/cursor/fast（含显式排序实验）→ 独立 complete/ir/restart，目标签名保持，
  最终为单 Relu；搜索选项可改变，故障身份与实际产物复检未放宽。
- 报告升级为 schema 6，增加 optimizations；task 记录各实验选项。
  最终读回校验绕过搜索校验缓存，实际路径故障判定继续绕过内容缓存。

上述结果支持保留显式 auto / cursor 入口，未证明全局最小或真实后端普遍性能改善。

## 复现

从仓库根目录执行；基线目录必须尚不存在。使用同一已安装依赖的 Python 执行全部批次：

```bash
uv run --locked python benchmarks/prepare_batch56_baseline.py \
  --output /tmp/onnx-perf-batch56-baseline
uv run --locked python benchmarks/batch56.py \
  --baseline-root /tmp/onnx-perf-batch56-baseline --suite preparation --repeat 3
uv run --locked python benchmarks/batch56.py \
  --baseline-root /tmp/onnx-perf-batch56-baseline --suite search --repeat 3
uv run --locked python benchmarks/batch56.py \
  --baseline-root /tmp/onnx-perf-batch56-baseline --suite search \
  --only gated-300 --full-gated --variants restart-complete,cursor-complete --repeat 3
uv run --locked python benchmarks/batch56.py \
  --baseline-root /tmp/onnx-perf-batch56-baseline --suite search \
  --only tight,threshold,old-rejection,gated-90 --variants cursor-complete,history \
  --compare-instrumentation --repeat 3
```

主 preparation 矩阵及后补的三个场景在该命令中合为 330 次；原始分批记录见证据摘要。
5000 节点与 1 / 5 秒扩展的配置、源码和结果也保留在摘要中，可用 `benchmarks/reduction.py`
按其 `configuration` 重跑。用 `audit_batch56.py --batch .../summary.json --extended .../baseline.json`
逐次传入待核验批次，再指定教学 `--cli` 目录及 `--output` 汇总位置，重新核对实际文件和日志。
