# 架构说明

本文帮助开发者理解缩减决策、定位模块和判断结果的保证范围。
后端接入和运行命令见[使用指南](usage.md)，运行证据见[排障指南](troubleshooting.md)。

## 从入口找到模块

主要源码位于仓库根目录下的 `src/onnx_reduce/`：

```text
src/onnx_reduce/
├── __main__.py          # python -m onnx_reduce 入口
├── cli.py               # 参数解析、模型与判定器创建、CLI 退出码
├── config.py            # 搜索、执行、判定、实验配置及兼容 EngineConfig
├── engine.py            # 兼容旧导入路径，转发 runtime.engine
├── runtime/             # 一次运行的执行与证据
│   ├── engine.py        # 初检、循环、接受、覆盖核验与独立最终复检
│   ├── result.py        # 候选执行事实，供搜索反馈和记录消费
│   ├── candidate.py     # 单个候选的缓存查询、编辑、成本比较、校验与判定
│   ├── budget.py        # 软时间预算和候选实际判定次数配额
│   ├── reporting.py     # 候选、调度、接受记录及报告计数，不写执行缓存
│   ├── artifacts.py     # 已接受模型的 checkpoint 与清单
│   └── resume.py        # 跨运行目标身份恢复与配置核对
├── search/              # 候选选择；不执行编辑、故障判定或 checkpoint 写入
│   ├── base.py          # Scheduler 协议
│   ├── types.py         # 候选反馈、调度记录、覆盖摘要
│   ├── state.py         # 共享版本、覆盖记账与统计，不包含选择策略
│   ├── legacy.py        # 固定 pass 顺序，接受后重启
│   ├── stateful.py      # 有状态分区、尺度复用与轮转
│   ├── staged.py        # 编排阶段、区域和清理机会
│   ├── progress.py      # 阶段转换、停滞证据与接受成本
│   ├── rotation.py      # pass 轮转、量子及跨接受的公平性状态
│   ├── coverage.py      # 确定性覆盖枚举、独立 Finalizer 与版本失效
│   ├── regions.py       # 模块、无分叉链与拓扑区域
│   └── cursors.py       # 稳定身份游标与有界历史排序
├── types.py             # 模型协议、成本、校验结果、落盘产物与图摘要
├── edits.py             # 候选及有序编辑的数据描述
├── model/               # ONNX 模型表示与操作
│   ├── __init__.py      # 保留 OnnxModel、TensorLinks 导入入口
│   ├── adapter.py       # 表示所有权、稳定 ID、按需 IR 与对外接口
│   ├── queries.py       # 连接、区域、死节点和无用资源查询
│   ├── editing.py       # 有序编辑、克隆、边界提升与身份维护
│   ├── inference.py     # shape / type 推导摘要
│   ├── validation.py    # 成本、domain 策略、模型和实际产物校验
│   ├── protobuf.py      # 保守快速删除与平坦图查询，无隐式混入继承
│   ├── snapshot.py      # 拥有 proto、确定性字节和摘要；统一编辑后失效
│   └── operations/      # graphs、rewrites、tensors、local_tensors 的具体操作
├── passes/              # 内置候选生成规则和默认执行顺序
│   ├── base.py          # pass 协议及可选节点、游标、阶段能力
│   ├── drop_outputs.py
│   ├── remove_unused.py
│   ├── delete_nodes.py
│   ├── extract.py       # 有界快速探测与完整的活跃子图提取
│   ├── rewrite.py       # 旁路、属性、张量和子图 pass
│   ├── candidates.py    # ONNX 候选规则，按查询视图生成有序编辑
│   ├── combinations.py  # 有界复合候选
│   └── groups.py        # 资源分组规则
├── failure_check/
│   ├── contract.py      # 判定配置、退出码与三态结果
│   ├── runner.py        # 进程执行、超时清理与原始观察
│   └── evaluator.py     # 故障身份匹配、内容缓存与判定证据
├── performance.py       # 可选的运行内计时、规模和资源统计
├── preparation.py       # 有容量边界的元数据缓存；不携带编辑策略
├── report.py            # 统计项与 JSON schema version 7
└── storage.py           # 临时文件、原子替换与 checkpoint 文件复用
```

## 模块职责与修改入口

CLI 负责组装配置和输入；engine 编排任务生命周期。engine 将当前模型交给 scheduler
选择候选，再交给 `CandidateExecutor.execute()` 执行。执行器返回 `AttemptResult`，
engine 把反馈交回 scheduler，并在接受后切换当前模型、保存 checkpoint。

三种 scheduler 独立实现 `search/base.py` 中的协议，不互相继承。每个策略组合一个
`SearchState`，共享模型版本、候选绑定、覆盖状态与统计。策略自行决定接受后的游标
和阶段变化；`SearchState` 不选择下一候选。staged 的最终覆盖状态由独立 Finalizer 提供。

候选执行器显式接收本次运行的缓存、预算、判定器和报告。确定性拒绝的缓存写入
属于执行器；`runtime/reporting.py` 仅写证据、更新观察统计，停用候选日志不会改变
缓存命中和判定调用次数。执行器返回结果后，engine 调用记录模块消费结果；
调用预算使用 evaluator 的实际调用数，不依赖报告计数。shape 和校验缓存仍使用运行内上下文；
编辑后端通过 `with_edit_options(EditOptions(...))` 显式绑定到模型视图，克隆与快速结果继续携带选项，
退出缓存上下文不会改变编辑策略。模型表示由 `model/adapter.py` 持有；查询、编辑、推导和校验通过
显式函数接收模型，模型接口负责委托，不再继承 protobuf 实现。
复合编辑每步结束后调用统一失效入口，清除模型类及其父类声明的所有 `cached_property`；
新增派生缓存会自动参与失效，不需修改编辑分发列表。失效属性名按模型类型有界缓存，
不保存模型、IR 或张量引用。

| 修改需求 | 主要入口 |
|---|---|
| 新增缩减候选 | `passes/` 中的 pass 与候选规则 |
| 调整候选顺序、阶段或游标 | `search/` 中对应策略 |
| 调整接受条件或候选缓存 | `runtime/candidate.py` |
| 修改图查询或编辑 | `model/queries.py`、`model/editing.py`、`model/operations/` |
| 修改推导或校验 | `model/inference.py`、`model/validation.py` |
| 调整预算 | `runtime/budget.py` |
| 调整日志与计数 | `runtime/reporting.py`、`report.py` |
| 调整 checkpoint 或跨运行核对 | `runtime/artifacts.py`、`runtime/resume.py` |
| 调整后端进程或故障匹配 | `failure_check/runner.py`、`failure_check/evaluator.py` |

`EngineConfig` 的定义位于 `config.py`。原有
`from onnx_reduce.engine import EngineConfig` 入口继续可用，命令行参数、默认值、
任务配置和报告字段保持兼容。旧的根级搜索模块已迁移到 `search/`，仓库内部导入
直接使用新路径。历史批次设计和性能证据记录当时的实现；当前源码位置以本文为准。


## 一次缩减的流程

1. CLI 加载模型与 external data，创建工作根目录下独立的 `run-*`，记录任务配置。
2. 按选定策略校验原模型，在 `initial/model.onnx` 上执行强制初检。
   只有判为 `interesting` 才建立基线身份并写首次 checkpoint，否则报错退出。
3. 由所选 scheduler 返回下一候选；engine 分配序号并绑定、核对当前模型的内容摘要。
4. 候选执行器先查询同基准、同编辑序列的准备缓存；未命中时在模型副本上应用编辑，检查成本是否严格降低，再校验合法性。
   编辑失败、成本未降低、校验失败均记录决策，不执行后端判定。
5. 对合法候选先查询内容判定缓存；未命中且有调用预算时才落盘，检查时间预算并执行实际判定。
   只有匹配目标故障的 `interesting` 候选会被接受。
6. engine 返回结构化结果给 scheduler；接受后更新当前模型与 checkpoint。
   legacy 从首个 pass 重启，stateful 延续阶段并使全部旧覆盖失效。
   staged 依次执行粗、区域、细搜索；complete 目标增加深度搜索后进入独立收尾。
   当前模型的声明范围完成扫描且没有接受时达到固定点；预算阻止继续时标记预算耗尽。
7. 将当前模型写到实际输出路径，读取字节核对摘要、重新加载校验，再绕过缓存执行强制最终复检，
   写最终报告与复现配置。
   若复检不再为 `interesting`，停止原因改为 `unstable_check`。

这意味着清理无用内容也必须通过故障判定。即使节点不影响图输出，它也可能是编译器
故障的唯一触发因素，因此不能像常规优化器一样无条件删除。

## 内置 pass

默认顺序由 [default_passes()](../src/onnx_reduce/passes/__init__.py) 定义：

| pass | 候选生成方式 |
|---|---|
| `compact_fragments` | 优先以降 rank 等形状变化节点为终点，沿依赖回溯，原子地旁路 dtype 和具体 shape 相同的输入支路并提取活跃子图；dense Constant 边界保留为 initializer。每个版本最多 8 个锚点、4096 次依赖访问、每个候选最多 8 次旁路；缺少注释时仅对不超过 4096 节点、16 MiB 的模型复用 shape 推导缓存。仍未知的 shape 和控制流交给后续 pass |
| `probe_fragments` | 每个模型版本先尝试最多 16 个完整子图；按算子家族的稀有度选择代表节点，并交替尝试拓扑分区；一次编辑同时重建真实输入输出 |
| `drop_outputs` | 优先批量删除除第一个以外的输出，再尝试除最后一个以外的输出；未接受时按平衡分区细分到全部单项，始终保留至少一个输出；不隐式删除生产节点 |
| `remove_unused` | 先尝试整体删除不通向图输出的死节点，再分别整组删除未使用输入、initializer；未接受的资源批次细分到全部单项 |
| `shrink_tensors` | 对具体维度 d 尝试 1、d//2、d−1；协调修改图接口、dense initializer、Constant 张量和直接使用的 shape 向量；在常规节点删除前完成可接受的缩维，减少接受后重复的删除扫描 |
| `delete_nodes` | 按当前主图节点顺序删除连续块；块大小从节点数的一半向下整除到 1，单节点图尝试删除该节点 |
| `bypass_nodes` | 对有单个非空输出的主图节点，逐一尝试用其非空输入替代输出，然后移除该节点，不提升新输入 |
| `simplify_attributes` | 根据当前 opset 的标准算子 schema，逐一删除显式设置的可选属性；跳过必需属性、图属性、自定义算子和本地函数 |
| `reduce_subgraphs` | 递归访问属性子图，尝试旁路、可选属性删除和分组节点删除；保留子图接口槽位 |
| `combinations` | 有界枚举非连续节点删除、多个旁路、属性/维度编辑与旁路或删除的组合；组合后清理无用资源，整体判定 |
| `extract_nodes` | 在原有 pass 后完整扫描家族代表及标准尺度分区，包含每个主图单节点；其判定与成本规则和其他候选相同 |


旁路会更新消费者、图输出及嵌套子图中的外层捕获引用。替换来源在子图中被局部定义遮蔽、
或替换导致重复图输出时拒绝编辑；类型不匹配由候选校验拒绝。旁路及属性删除都可能改变
计算语义，每个候选仍须成本严格下降、通过模型校验并保留目标故障。
内置 pass 默认启用，接受后按调度策略重新访问；staged complete 的独立最终扫描包含全部配置 pass。
这扩展了固定点的范围，仍不证明任意节点组合或任意 ONNX 模型中的全局最小。

删除节点时，仍被删除区域之外使用、或仍是图输出的边界值会被替换为新图输入。
工具先尝试推断缺失的类型和形状；推断失败不会终止整轮搜索，后续校验会决定候选能否继续。
同一基准摘要共享推导结果或失败摘要；只缓存主图值的名字、类型和形状。复合编辑中的后续
推导针对前序编辑后的图，每次编辑后清除中间快照，避免把旧推导或旧字节用于新图。
删除含属性子图的主图节点时，会清理随之移除的嵌套节点引用；子图内部候选由 `reduce_subgraphs` 单独提供。

## 活跃子图与图接口

单纯删除尾部节点会把原图输出提升为新输入；保留下来的故障节点可能因此成为死代码，
真实后端若先消除无用节点，就不会再触发原故障。这种候选失败不意味着保留的节点本身不能复现。
`ExtractNodes` 因此明确表达另一种候选：保留指定主图节点，暴露仍向区域外使用或终端的输出，
用外部依赖组成新的图输入，并保留实际需要的 initializer、稀疏资源及外层捕获。
被裁掉的标准 Constant 若持有 TensorProto 值，可将该数据转为同名 initializer。
对 Expand / Reshape 的计算型 shape 边界，还增加 `materialize_shapes` 提取变体：
在源图中按原 opset 求值小规模、完全由常量组成的整数表达式，将结果转为同名 initializer，
再对候选推导 shape。普通提取候选仍会枚举；同等稀有的家族优先探测 shape 消费者，
避免粗分区被接受后重置枚举，使这类候选迟迟得不到机会。
求值仅支持 Identity、Add、Sub、Mul、Equal、Where、ConstantOfShape，
常量必须为标量或一维的 int32 / int64 / bool；每个值最多 1024 个元素，
每次提取最多访问 128 个值，依赖深度最多 64。
不执行自定义算子、本地函数或依赖图输入的表达式（包括带 initializer 默认值的图输入），
不分配超过限制的 ConstantOfShape 结果。不支持或求值失败时拒绝该变体，继续普通候选。
这不读取实际运行数据，也不合成任意 shape；候选仍须复现同一故障才能接受。
构造验证见[计算型 shape 边界](performance/dynamic-expand.md)。

每个单节点还尝试分别暴露各个命名输出，避免多输出算子因额外输出而无法被单独接受。
另提供 `preserve_interface` 变体：先匹配同名输入，再按完全一致的 TypeProto 匹配边界，
保留原图输入输出的名字、顺序、类型、shape 和必要默认 initializer；原有未使用输入也可保留。
不能匹配类型或输出数量时拒绝该变体，不会把内部值强行标成另一种类型。
重命名同步更新外层捕获，拒绝与保留定义冲突或导致捕获被子图局部定义遮蔽的映射。
这些变换不插入额外的 Identity 节点，因此仍可产生真正的一节点图。

普通投影先尝试两个分区，再处理延后的接口变体；唯一算子家族仍优先尝试其接口变体，
兼顾固定接口的故障判定与原有快速分区。前置 16 次上限涵盖全部变体。

提取会改变图接口和计算语义，始终经过正常的成本、校验和故障判定。旧输出及无关资源在同一候选内移除，
不能先接受半个变换。边界类型从规范化快照读取，缺失时使用已有的运行内 shape 推导缓存；
推导失败不会编造类型或维度，仍由配置的 checker 拒绝不合法的候选。
控制流内部接口、节点属性和模型本地函数保持原有表示；主图节点稳定 ID 只保留选中的谱系。

提取查询索引按模型版本缓存，输出只复制保留的节点和数据；不先克隆整张图或复制将要抛弃的大权重。
新索引属于统一的派生属性失效机制。两种编辑模式使用相同的提取实现。
`probe_fragments` 的前置上限避免新策略独占有限预算；后面的 `extract_nodes` 仍枚举完整集合，
staged complete 的 Finalizer 会在每个新版本重建扫描。报告固定点仍只针对声明的候选集合，不能作为全局最小证明。

## 搜索调度与节点身份

`EngineConfig.scheduler` / `--scheduler` 默认为 `legacy`。三种调度器共用候选执行路径，
不执行编辑或判定，不决定接受条件，也不写 checkpoint。反馈区分编辑失败、成本未下降、
校验失败、未复现、无法判定和接受，同时提供缓存来源、实际调用数及完整成本。

`stateful` 以当前拓扑的连续分区搜索，块大小从 `max(1, N//2)` 开始，每块尝试删除和仅保留。
接受后保留成功方向、尺度及下一存活锚点；同尺度无进展才减半。每阶段最多连续尝试 8 个候选，
包含接受和准备失败；版本更新不重置该额度。仅保留会删除补集，仍沿用边界提升为图输入的语义，
不隐式清理输出、输入或 initializer。允许删除最后一个节点，仍须保留合法输出。

`OnnxModel.node_ids` 和只读 `id_to_index` 是主图节点的旁路身份。克隆按对应顺序传递 ID，
每次删除、排序后重建当前映射；复合编辑按各自的中间图更新。ID 不写入 ONNX，也不影响
摘要、成本和判定缓存；重新加载模型产生新的身份空间。搜索不使用旧模型的位置索引。

节点 pass 通过可选的 `NodeSearchPass.node_regions()` 显式启用专用搜索；适配器另提供
`NodeSearchModel` 协议。只有 `generate(model)` 的第三方 pass 仍可轮转执行，但模型版本变化
就丢弃其生成器并重建。不会按 pass 名称替换自定义实现。所需节点能力在首次后端调用前检查。

任何接受都增加模型版本，清空当前版本节点去重表并使全部阶段待重访。旧模型上的失败
不能证明新模型上可以跳过同一区域。每个启用节点搜索的 pass 最多保存 4096 个去重摘要，淘汰只影响效率；
游标保证有限扫描。启发式耗尽后独立扫描配置 pass 的原候选及标准尺度的保留候选，
该扫描不受启发式去重影响。覆盖接受后返回搜索；`--no-cache` 时也必须实际执行覆盖候选。
历史设计与验收边界见[有状态搜索设计](history/design/stateful-search-design.md)。

`staged` 增加 coarse、regions、fine 阶段及有额度限制的资源清理。complete 目标再经过 deep 阶段，随后进入
独立 Finalizer，收尾接受后留在收尾阶段并重新扫描；fast 目标结束搜索后直接复检产物，
始终不声明最小性。区域使用明确模块标注、无分叉链与拓扑回退，旧失败只影响尝试历史。
可选的 RegionSearchModel 提供当前稳定 ID 下的图链接，包含控制流捕获；缺少该能力时
仍可用节点身份执行拓扑分区。跨运行的 `--finalize-from` 恢复源目标并重新初检，
不恢复旧游标或缓存。历史规则、范围和接口见[分阶段搜索设计](history/design/staged-search-design.md)。

## 配置与测试边界

`EngineConfig` 组合 `SearchConfig`、`ExecutionConfig`、`CheckConfig`、`ExperimentConfig`。
各组验证自身的值域，跨组限制集中在组合入口；候选执行器只接收执行配置与校验模式。
CLI 的任务参数保持原名、默认值与语义；游标、历史排序、编辑后端、校验缓存和 checkpoint
对照开关集中在 experimental comparisons 参数组。进程命令、退出码和超时配置继续由
`failure_check/contract.py` 持有。

已有 `EngineConfig(scheduler=..., ...)` 关键字、属性读取及 `dataclasses.replace` 继续可用；
新代码优先显式构造四组配置。`asdict(config)` 展示分组结构，`config.to_dict()` 用于保留
旧版平铺的 task.json 格式，因此目标恢复和报告 version 7 不变。

ONNX pass 从 `candidate_snapshot()` 读取共享的只读约定视图，生成规则均位于 pass 层；
编辑、校验和后端判定不在生成器中执行。公开 `to_proto()` 返回独立副本。第三方适配器已有的
`rewrite_candidates()` 扩展仍由 pass 入口兼容，内置模型不再生成候选。

测试按 `tests/model`、`tests/search`、`tests/runtime`、`tests/integration` 归类；公共模型
构造位于 `tests/helpers.py`，已安装分发包的独立验证仍在 `tests/smoke_distribution.py`。
重构对照方法和测量边界见[行为与性能回归](performance/refactor-verification.md)。

## 接受条件与最小性

通过 `--objective` 选择确定性的字典序成本：

| 目标 | 比较顺序 |
|---|---|
| `nodes`（默认） | 递归节点数、张量元素数、字节数、输出数、输入数、initializer 数 |
| `bytes` | 字节数、递归节点数、张量元素数、输出数、输入数、initializer 数 |
| `tensors` | 张量元素数、递归节点数、字节数、输出数、输入数、initializer 数 |

节点、输入输出及 initializer 数统计主图和递归属性子图，不包含本地函数定义。
用于主图搜索索引的 `OnnxModel.node_count` 仍只统计主图，报告中的成本 `node_count` 为递归统计。
`tensor_elements` 是 dense initializer、张量属性的存储元素数，加上未与 initializer 重名的
图输入具体形状元素数；标量为 1，符号/未知维度按 1 计，零维度按 0 计，稀疏数据不计入。
字节数取自自包含序列化快照。默认目标允许在张量元素数下降时字节数略增；若只关心文件大小，
选择 `bytes`。JSON 报告版本升级为 7，记录实际目标、组合边界和节点统计范围。

接受候选要求：编辑成功、成本严格降低、校验通过、故障判定与目标身份匹配。
严格下降的成本防止在相同大小模型之间来回移动。

`minimality.status=pass_fixed_point` 只在达到固定点且最终模型的完整搜索轮次中
没有无法判定的候选、最终复检通过时出现。此前轮次的无法判定不单独阻止这一结论。
legacy 的作用域为当前配置 pass 所生成的合法、成本更低候选；stateful 在启用节点搜索时
还覆盖当前标准尺度分区的保留候选。自适应中出现的其他尺度不属于最终声明范围。
staged complete 还覆盖最终模块、链和拓扑区域的删除与保留候选；staged fast 始终不证明最小性。
所有策略都**不保证全局最小**。报告记录覆盖的摘要、版本和完成标志。
预算停止、最终轮次无法判定或最终复检失败时，状态为 `unproven`。

## 模型校验策略

原模型与候选使用相同策略。所有模式首先检查主图至少有一个输出、输出名称非空且唯一，
并确认输出有生产者或本身是输入/initializer。

| 请求模式 | 后续检查 |
|---|---|
| `strict`（默认） | `onnx.checker.check_model(full_check=True, check_custom_domain=True)`；发现非内置 domain 自动改为 `structural` |
| `structural` | 同样调用 ONNX checker 并启用 `full_check=True`，但 `check_custom_domain=False` |
| `permissive` | 仅执行前述基本输出检查，跳过 ONNX checker |

内置 domain 集合为 `""`、`ai.onnx`、`ai.onnx.ml`、`ai.onnx.training`。
`structural` 仍可能因标准算子、类型或形状问题拒绝模型，也不证明后端支持自定义算子。
`permissive` 不会绕过模型加载或 IR 导入错误。
报告用 `requested_validation_mode`、`effective_validation_modes` 和 `custom_domains`
记录请求策略及整个运行中观察到的模式与 domain。

## 判定、缓存与预算的分工

Runner 负责执行并观察，matcher 负责判定是否仍是目标故障。两层结果可能不同：
例如进程退出 1，但签名变化，最终结果为 `not_interesting`。
具体退出码、超时及签名契约见[使用指南](usage.md#编写故障判定脚本)。

缓存仅存在于一次运行内，固定该次运行的判定器、目标身份、签名配置和校验策略。
缓存键包含版本（当前为 1）：编辑缓存以基准内容 SHA-256、有序编辑序列和校验模式为身份，
保存确定性的编辑失败、成本未下降、校验失败与未复现决策；内容缓存以精确内容 SHA-256 为身份，
保存已匹配的确定判定；shape 缓存以推导前基准内容 SHA-256 为身份，保存推导摘要（包括失败）。
新增校验缓存以内容摘要、请求模式和有效模式为身份，保存完整 ValidationResult；
域缓存以内容摘要为身份，保存域名摘要。接受候选立即切换基准，成功候选不保留在编辑缓存中。

五类缓存各自采用 LRU，最多 4096 项、8 MiB 编码键值，合计最多 40 MiB 编码元数据，另有
受项数限制的 Python 容器开销；不保留候选模型、权重或有图引用的 IR 值。超大条目不保留，
淘汰后重新计算。此限制不包含当前模型、序列化快照及推导过程的临时内存。

不做语义等价判断，也不跨运行复用。`inconclusive` 判定不进入决策缓存；
shape 推导失败属于固定基准的准备结果，可以缓存。初检和最终复检始终绕过判定缓存。
判定缓存命中不执行进程、不消耗候选调用配额；编辑缓存命中还省掉克隆与校验，
内容命中省掉候选落盘。`--no-cache` 同时关闭上述五类缓存；一次导出快照复用仍生效。
`--no-validation-cache` 单独关闭校验和域缓存。最终实际产物重新加载后的校验不复用搜索校验缓存。
显式启用的 `--history-order defer` 使用另一份至多 4096 项 / 8 MiB 历史，只有排序作用；
最多 8 项当前版本预取尝试及当前阶段游标单独占用内存。`--no-cache` 不关闭这种调度状态。
该选项不提供随机故障的多次采样或统计保证。

调用配额只限制候选的实际判定。时间预算是包含初检的软墙钟限制，
同步候选生成、编辑、校验及已启动的判定可能越界，最终复检仍执行。
两个预算都未设置时持续搜索，直到固定点或发生错误/中断。

## 落盘与中断边界

模型适配器在副本上编辑，随后用同一份确定性序列化快照执行校验、计算成本、摘要和输出文件，
避免“检查的内容”和“保存的内容”分离。checkpoint 先保存以摘要命名的模型，
再原子替换 `checkpoint.json`；清单引用完整模型。初检成功后才首次创建 checkpoint。

原子替换降低进程中断造成半写入的风险；当前不调用 `fsync`，不保证掉电持久性。
JSONL 事件日志按事件追加，不与模型和报告组成跨文件事务。
checkpoint 是已接受模型的快照；正常收尾时还会记录最终状态，不能仅凭 checkpoint 存在
断言最终复检成功。恢复时应读取其中的报告，按[恢复步骤](troubleshooting.md#从-checkpoint-重新运行)
启动新运行。

## 当前边界

- Linux / macOS、Python >= 3.11；判定进程使用 POSIX 进程组，Windows 暂不支持。
- 缩减主图和递归属性子图、标准算子的可选属性，并尝试具体 shape 与 dense 权重的协调裁剪。
  不展开本地函数，不独立改变 Loop/Scan 接口槽位，不缩减稀疏权重或具体运行时输入数据。
- 面向不依赖具体输入数据的故障；不缩减输入数据，也不比较数值差异。
- 删除边界可以引入新输入；输出保留目标故障，不保证与原模型计算语义等价。
- external data 加载到内存并内嵌到输出；当前不支持超过 2 GB 的模型，候选落盘也会占用磁盘。
- 判定环境应固定，缓存要求结果依赖模型内容；首尾复检不能代替随机故障的统计验证。

这里的未支持能力均不是可用选项，也不代表已承诺的发布计划。

## 快照编辑与阶段续扫实验

`model_edit_mode=auto` 从规范化快照复制并执行单个 DeleteNodes 编辑，保留稳定 ID，
以按需 IR 支持复杂查询及后续组合编辑；公开 to_proto 返回独立副本。快速路径自身的查询
避免重新导入 IR，不改变 checker 和成本规则。未证明等价的边界退回 IR。

staged/cursor 通过 CursorPass 可选能力读取当前 RemovalGroup，按稳定 ID 或资源名重绑定。
coarse 重新恢复新规模，fine 延续粒度与下一存活身份；每个无接受尺度尝试 8 次后细分。
接受会丢弃当前版本生成器，有历史预取时先回退未执行游标，最终收尾范围与版本规则不变。
历史设计和证据见[续扫实验设计](history/design/free-try-followup-design.md)与[结果](history/performance/performance-batch56-results.md)。

checkpoint 仅复用本次运行已发布的私有产物，复用前核验实际字节摘要，缺失/损坏则重新物化。
不链接后端可写的候选文件；模型/资源完成后再原子替换 checkpoint.json。

## 协调缩减与有界组合搜索

张量 pass 先尝试边界上的局部算子约束：二维 MatMul/Gemm 协调收缩维度（含 transA/transB），
Conv 协调输入通道或终端输出通道、权重、偏置并保留 group 整除约束，终端 Reshape 协调
输入首维与正整数 shape 向量的元素数。源必须是图输入或 dense initializer，无法解析的中间生产者、
符号维度、复杂广播和带 0/-1 的 Reshape 暂不生成局部候选，最终合法性仍由 checker 验证。
随后按维度值在全模型范围做一致替换，并裁剪相关 dense 张量的前缀切片。
直接连接 Reshape、Expand、ConstantOfShape 的整型 shape 向量同步替换维度值；
由复杂计算生成的 shape、任意算子属性约束、符号维度求解仍不支持。
同值维度并不一定属于同一语义维度，因此该变换可能被校验或目标判定拒绝。
它是候选生成规则，不是任意 ONNX 图都能成功的形状求解器。

子图编辑使用 `(父节点索引, 属性名, 图列表索引)` 路径寻址；普通 GRAPH 的列表索引为 -1。
子图不把断边提升成新输入，避免改变 Loop/Scan 的位置接口；无法保持合法性的删除会被拒绝。
旁路和属性候选复用主图规则，任意接受后重新生成路径，主图稳定 ID 保持不变。

`--combination-depth` 为 1..3，默认 2；1 关闭组合搜索。
`--combination-limit` 默认 256，0 关闭；上限分别作用于四类候选：非连续删除并清理、
多个旁路并清理、属性/张量与旁路并清理、属性/张量与删除并清理。
每类最多输出上限数量，二元和三元深度交替输出，共享该类上限；拓扑关联候选与全图分散候选
交替执行。分散顺序覆盖两端及逐步细分的中间区域，避免字典序前缀耗尽额度。
旁路池按同样方式分散采样；属性与张量配对分别保有候选池，交替输出，不能相互挤掉。
有限候选池也受此上限限制；采样不意味着穷举所有组合。
多个旁路按节点索引降序执行，避免前序删除使后续地址失效。中间编辑不要求单独变小或复现，
只有整体最终模型通过检查且降低选定成本才接受。这允许部分有界路径越过局部固定点，
不维护无限的等成本状态，也不保证任意编辑组合可达。

所有新增 pass 都纳入标准完整扫描及 staged complete 最终扫描。预算耗尽或无法判定仍为
`unproven`；`pass_fixed_point` 仅针对配置的有限候选集合，组合上限会影响覆盖范围。
质量和真实故障接入方式见 [缩减质量验证](performance/reduction-quality.md)。

## 搜索公平性与收益诊断

coarse/fine/deep 的 pass 按批轮转，默认每轮 8 个尝试。最近 16 个执行中至少有 8 个历史样本、
接受率至少 25% 的 pass 最多获得 16 个尝试，其余策略仍保留自己的轮次。
轮次计数跨成功接受保留，模型变更后丢弃全部旧生成器并重新绑定地址，不能靠连续成功独占搜索。
fine/coarse 的连续失败上限按 pass 分别统计为 64；deep 为 256。区域阶段仍有全局 64 次无收益上限。
报告的 pass_states 在接受后失效并重置，pass_state_model_version 标记状态所属的当前模型版本；
状态区分 not_visited、no_candidates、visited、exhausted 和 plateau。

complete 的 deep 阶段在正式证明前轮转完整 pass 候选；对节点搜索 pass，优先补充最近删改区域附近的
小区域删除和保留尝试。deep 受时间和调用预算约束，不产生固定点证明。
一旦进入 finalization，仍在每次接受后重建当前模型的完整扫描，不复用旧模型的失败结论。
`--finalize-from` 继续直接进入独立 finalization，不经过 deep；fast 也不执行 deep。

`--performance` 的嵌套性能报告版本为 2，增加 progress 时间序列：记录初始、每次接受、
约 5 秒间隔的搜索采样和最终结果，包含节点数、张量元素数、字节数及调用次数。
长时间单次判定无法中途采样。报告内序列最多 2048 点，超出后抽稀并记录 progress_compactions，
完整采样另存 progress.jsonl。每个接受的模型都降低选定的成本，但其他分量可能增加，
不能把三项指标分别最低的数值拼成一个不存在的模型。

各 pass 的准备/判定时间、接受成本变化及拒绝原因见 scheduler.buckets；生成耗时见
performance.stages 中的 generate/PASS_NAME。计时和曲线只用于诊断，不参与候选顺序决策。
