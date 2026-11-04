# 第五、六批：参考 free-try 的后续性能设计

> 本文保留该批次的设计和当时的源码位置。当前模块职责与路径见[架构说明](../../architecture.md)。

日期：2026-11-03。状态：已实现实验入口；性能与推广结论见[第五、六批实测](../performance/performance-batch56-results.md)。

参考提交为 `perf/free-try` 的
[`666fc9b545596c61893d5afc882e3e9bae7299f7`](https://github.com/XuMoheng/onnx-reduce/commit/666fc9b545596c61893d5afc882e3e9bae7299f7)，
设计基线为 `1012-perf` / `perf/reduction-strategy` 的
`3f7cb3a64d84c6ce1324a94da24c9f3cd2c86fd1`。两边共同起点为 `b7c7631`。

建议先降低同一候选的准备成本，再改善接受后的续扫。继续保留默认 `legacy`，
复用 staged 的独立 Finalizer，不整体移植 free-try 的引擎循环和拒绝跳过机制。
第五批要求同一策略的行为等价；第六批承认搜索路径会变化，按预算下的质量和成本验收。

## 1. 与已有工作的差集

free-try 相对 main 的 126x、185x 是该提交报告的结果，不是相对当前 perf 的增量收益。
perf 已有批量清理、shape 摘要缓存、编辑缓存、内容缓存前移、稳定节点 ID、区域搜索和独立收尾。
[第四批实测](../performance/performance-batch4-results.md)显示，512 initializer 已只需 4 次实际判定；
同一场景无法再次获得“515 → 4”的策略收益。

| free-try 机制 | perf 当前状态与代码入口 | 后续决定 |
| --- | --- | --- |
| 整组清理、失败细分 | `passes/groups.py` 已有整组、分区、单项回退 | 保留候选集合，增加接受后的游标；不重新计算一遍批量清理收益 |
| 单次导出复用 | `model._payload` 已供成本、checker、摘要、落盘共享 | 不作为独立新增收益；增加 protobuf 快照是为快速编辑服务 |
| shape 缓存 | `preparation.py` 已按当前基准摘要缓存类型摘要和失败，且有容量限制 | 复用，不引入谱系共享的无界字典或 `ir.Value` 缓存 |
| protobuf 快速删除 | `model.apply_candidate()` 仍统一 IR clone/edit/export | 第五批首要工作，先支持单个 `DeleteNodes` 编辑 |
| 有序图跳过排序 | `model._delete_nodes()` 每次调用 `graph.sort()` | 第五批单独验证；只在可证明顺序不变时省略 |
| 校验、域扫描缓存 | 同内容的不同编辑仍可能重复执行 `validate()`；内容缓存查询在校验后 | 按实测加入有界元数据缓存；最终产物读回校验绕过校验缓存 |
| checkpoint 复用 | `_checkpoint()` 每次 `materialize()`，已有 `_payload`，主要重复成本是写入 | 第五批先避免同摘要 checkpoint 重写，再评估文件复用 |
| 接受后反馈续扫 | stateful 有粒度/锚点；staged 接受后丢弃阶段生成器 | 第六批在 scheduler 内维护显式游标，不让 engine 管理 `generator.send()` |
| 跨基准拒绝记忆 | 当前设计明确禁止把旧失败当作新模型的排除依据 | 只试验延后排序，不移植跳过语义 |
| 无跳过扫描证明固定点 | staged 已有范围版本 1 的独立 Finalizer | 直接复用；不增加另一套“诚实扫描”状态机 |

已有退化样本继续作为约束：threshold-128 中 legacy 68 次判定完成固定点，
staged-complete 用满 102 次总判定仍未证明；16/12/6 中分别为 16、23 次。
同时保留 stateful 在 cross-region 上 44 次判定优于 staged-complete 60 次的样本。
这些数字均来自第四批报告，不能以单一 gated 场景取代。

## 2. 移植前需要收紧的两条结论

### 2.1 相同物理编辑的旧拒绝不能证明新拒绝

令故障条件为 `T 存在且（A 存在或 B 不存在）`，初始图含 A、B、T。
初始删除 A 会失败；先成功删除 B 后，删除 A 就可以保留故障。
这是已有 `old-rejection` 基准所覆盖的模式，既适用于节点，也适用于资源及类型边界。
本地 `invalid`、`not_cheaper` 也可能随基准改变而失效。

free-try 的 `_memo_skips()` 对部分旧拒绝跨基准跳过，之后再无跳过收尾，
可以支持最终扫描范围内的固定点；这不能证明与关闭记忆的运行到达相同固定点，
也不能保证紧预算下成本不退化。有限的哈希一致性测试不能升级为一般等价结论。
perf 只允许同基准的精确缓存直接复用结论，旧基准历史仅影响启发式排序。

### 2.2 快速路径需要覆盖元数据的字节等价

本次在参考提交原样源码上核验：Python 3.13.17、onnx 1.23.2、onnxscript 0.7.2。
两节点 `Identity(x→a) → Relu(a→y)`，给完整类型的边界 a 添加
`ValueInfoProto.metadata_props["boundary_tag"]="kept?"`，删除第一个节点。
快速路径保留了提升输入的 metadata，IR 路径未保留；两者 strict 校验均通过，但 payload 不同。
参考测试矩阵未覆盖该情况。以下命令从 `1012-perf/` 执行，仅在参考进程中切换路径：

```bash
(cd ../1012-free-try && .venv/bin/python - <<'PY'
from onnx import TensorProto, helper
from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.model import OnnxModel

def v(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])

a = v("a")
a.metadata_props.add(key="boundary_tag", value="kept?")
p = helper.make_model(helper.make_graph([
    helper.make_node("Identity", ["x"], ["a"]),
    helper.make_node("Relu", ["a"], ["y"]),
], "metadata", [v("x")], [v("y")], value_info=[a]),
    opset_imports=[helper.make_opsetid("", 17)])
c = Candidate((DeleteNodes((0,)),))
fast = OnnxModel.from_proto(p).apply_candidate(c)
original = OnnxModel._try_fast_delete_nodes
try:
    OnnxModel._try_fast_delete_nodes = lambda *args, **kwargs: None
    slow = OnnxModel.from_proto(p).apply_candidate(c)
finally:
    OnnxModel._try_fast_delete_nodes = original
print(fast._payload == slow._payload)  # False
print(fast.validate("strict").valid, slow.validate("strict").valid)  # True True
PY
)
```

移植时以 perf 当前 IR 行为为参照。首版遇到尚未证明等价的边界元数据就回退 IR；
若要改进元数据保留，应另立功能变更，同时修改两条路径，不能混入性能等价改动。

## 3. 第五批：降低准备和存储成本

### 3.1 模型表示及快速删除

在 `model.py` 内提供私有、只读 protobuf 快照和按需创建的 IR。
公开 `to_proto()` 继续返回独立副本；内部序列化直接使用快照，不经 `to_proto()` 再复制一次。
输入先沿用现有 IR 导入/导出规范化，避免直接从原始输入 protobuf 编辑而改变序列化基准。

```text
IR 模型 ──一次规范化导出──> 只读 protobuf 快照 ──确定性序列化──> payload / digest
                                  │
                          CopyFrom + 单次节点删除
                                  ↓
                           新的只读快照及节点 ID
                                  │ 只有不支持的查询/编辑才需要
                                  └────────按需导入 IR
```

首版进入快速路径必须同时满足：

1. 候选只有一个 `DeleteNodes` 编辑，基准摘要匹配，所有索引合法。
   “一个编辑”允许不连续、逆序及重复的多个索引；去重保留第一次出现顺序。
2. 主图无 GRAPH/GRAPHS 属性；节点引用的值均具名，名称与定义无歧义；省略可选输入的空名回退。
3. 注解为已支持的普通张量类型；序列、可选、稀疏张量等未验证形式回退。
4. 原图拓扑有序，删除后无需重排；非规范顺序回退，不能把新快照无条件标记为已排序。
5. 被提升边界的类型、shape、doc string、metadata 的构造已与 IR 路径证明等价。
   首版未覆盖的字段组合直接回退，不能以“两边都能过 checker”替代字节等价。

步骤保持现有语义：按编辑索引顺序找边界，判断剩余消费者及图输出，按原顺序追加提升输入，
复用基准 shape 摘要，移除删除节点及对应中间注解，保留无关资源，生成新快照。
图输出被提升时也需同步类型；缺类型且推导失败时保持原 IR 路径的拒绝行为。
组合候选继续走 IR，并按每个中间状态推导、失效，不能使用最初基准的类型摘要处理所有编辑。

继续实现现有 `node_ids` / `id_to_index`，不改成 free-try 的 `stable_node_ids()` 接口。
ID 来源是谱系身份，不能由节点名称或新索引重建；删除后筛选存活 ID，实际重排时保持
IR 对象与 ID 的对应。`region_graph()` 必须保留 `onnx_reduce.module` 元信息和现有边界摘要。

**避免按需 IR 变成逐候选 IR：**对符合条件的平坦图，node/input/output 数量、op types、
死节点、无用资源、域扫描和 `region_graph()` 读取快照；按当前 ID 构造区域依赖。
复杂图统一回退现有 IR 查询。分别记录 clone/export/lazy-import 次数，验证完整搜索链路
确实省掉转换，而不只是在 `apply_candidate()` 微基准中省掉。

快照、payload、IR、区域摘要都绑定不可变模型实例；编辑只能操作新对象。
不在父模型上失效缓存，不在全局缓存里保留完整模型、权重或可变 protobuf。
每实例的 capability/拓扑状态可惰性计算；真正改图后重算或按已证明规则继承。
需要测量 protobuf + bytes + 按需 IR 同时存活的 RSS，不能把降低 CPU 当作内存改善。

### 3.2 排序与精确元数据缓存

- 有序图的删除只移除依赖、把边界变为输入，不产生反向依赖；仅在当前图及嵌套图的
  顺序检查能证明现有稳定排序不改序时跳过 `sort()`。复杂图和不确定输入保持旧路径。
- 校验缓存接入 `PreparationCaches` / `MetadataCache`，键包含缓存版本、内容摘要、
  请求模式和有效模式；值编码完整 `ValidationResult`。域扫描缓存同样只保存域名摘要。
  分别沿用每缓存 4096 项 / 8 MiB 上限，过大不保留；这些是元数据上限，不是进程内存上限。
- 不跨运行共享；`--no-cache` 同时禁用新增校验/域元数据缓存，保持现有首尾强制判定。
  模型自身不可变快照和 ID 映射仍然存在，避免把内容表示与判定缓存混为一谈。
- `validate_artifact()` 的实际字节摘要核验、重新加载和 checker 不能命中搜索校验缓存；
  最终实际路径的后端判定也继续绕过内容缓存。

先测命中率再决定域缓存是否值得保留。已有 `_payload` 复用 checker 字节，不重复列为新增优化。

### 3.3 checkpoint：先去重，再评估复用

现有 `_payload` 已缓存，因此这里的主要目标是减少重复写入，不是再次消除序列化。

1. run 内记录已成功发布的 checkpoint artifact。相同摘要且私有文件完整时只更新 manifest；
   `checkpoint.json` 仍记录最新计数、耗时和报告。文件缺失或核验不符时从模型重新物化。
2. 新摘要首版仍使用 `model.materialize()` 原子写入；模型/资源先完成，manifest 最后替换。
   通用 `ModelAdapter` 不增加必需方法，带 `resources` 的 artifact 继续走现有完整物化路径。
3. 后续独立评估从已判定 artifact 复制/克隆的收益。`Evaluation.model_path` 可能指向缓存中的
   旧候选路径，不能假定当前 `candidate_path` 存在；文件摘要必须匹配当前模型。
4. 不直接硬链接判定脚本可写的候选文件到 checkpoint。硬链接共享 inode，脚本/清理过程的
   原地写入会损坏证据。只在有不可变私有源文件的明确保障后研究硬链接；否则原子复制或物化。
   `FileExistsError` 也不能代替摘要核验。最终交付继续独立物化、读回和强制复检。

存储改动单独验收文件缺失、源被修改、目标已存在但损坏、跨文件系统失败和 manifest 写入中断。
缓存命中接受时缺少可复用文件，必须安全回退，不改变候选接受结论。

## 4. 第六批：在 staged 内试验反馈续扫

第五批通过后再实施。保留 `legacy`、`stateful` 及当前 staged 为可复现实验对照，
不改普通 pass 的 `generate(model)` 协议，也不把 free-try 的引擎循环合并进来。

已实现以下实验选项；默认保持原搜索策略及 IR 编辑，独立开关用于消融和回退：

| 选项 | 初始默认 | 范围 |
| --- | --- | --- |
| `--model-edit-mode ir\|auto` | `ir`，第五批验收后再评估 `auto` | 全部 scheduler 的模型准备；`ir` 保留回退对照 |
| `--search-policy restart\|cursor` | `restart` | 仅 staged，选择当前行为或反馈续扫 |
| `--history-order off\|defer` | `off` | 仅 staged/cursor，选择是否试验旧拒绝的延后排序 |
| `--direction-order fixed\|adaptive` | `fixed` | 仅 staged/cursor，按当前阶段接受/实际调用计数排序，双方向均保留 |
| `--no-validation-cache` | 不指定时启用 | 独立关闭 checker/域元数据缓存，供消融使用 |
| `--no-checkpoint-reuse` | 不指定时复用 | 独立关闭 checkpoint 去重，供消融使用 |

不兼容配置在初检前报错；基准脚本需同步透传。配置写入 task/report，
`resume.py` 明确区分搜索选项与故障身份选项：独立收尾可改变搜索选项，不改变判定身份，
且不恢复历史或游标。新增报告字段/状态按仓库规则更新 schema 及读取测试。

### 4.1 游标状态归 scheduler 所有

内部游标记录 stage/pass、粒度、稳定 ID 锚点或资源名、待细分组；只输出对当前版本绑定的候选。
接受后先增加模型版本，再从当前 `node_ids` / 资源清单重绑定，不能保存旧位置或可执行候选。
普通自定义 pass 没有显式续扫能力时，继续在新基准重新生成；不用 pass 名称探测能力。
实现以可选 `CursorPass.removal_groups(model)` 暴露当前 `RemovalGroup`；`requires_node_ids`
明确声明是否需要节点协议，在初检前验证。旧 `generate()` 和 Finalizer 的枚举不变。

| 阶段 | 接受后的续扫规则 | 防止的退化 |
| --- | --- | --- |
| coarse | 从新模型重建粗尺度，维持各资源/pass 的轮转位置 | 保留 16→12 后恢复 6 节点尺度的能力 |
| regions | 丢弃旧区域快照，在新模型重建，优先锚点附近仍存在的区域 | 不执行旧区域索引，不因旧失败永久过滤新边界 |
| fine 节点 | 优先保留成功尺度；从删除处的下一存活节点绑定当前区块，逐步细分至单项 | 避免 gated 每接受一个节点就重跑全部粗块 |
| fine 资源 | 按资源名维护细分组，接受后去掉已消失项；新出现的无用资源加入待扫描组 | 避免 one-stubborn 清理反复回到整组失败 |
| cleanup | 继续最多两个尝试，接受也消耗额度；完成后回到更新后的阶段游标 | 不靠额外清理额度制造性能/质量收益 |
| finalization | 不续扫，不读取旧历史；接受后按新版本重启完整扫描 | 保留现有范围内的固定点证明 |

`cursors.py` 的游标每个失败尺度最多执行 8 次尝试后继续细分，接受则清零该计数；
这是避免 64 次阶段窗口在到达单项前耗尽的启发式规则，未扫描的组由完整收尾补足。
区域接受后按旧区域后方的下一存活 ID 作为新锚点；所有索引仍在新模型上重新绑定。
资源组的原始成员列表和复杂区域不能无限累积；只保留当前模型及当前阶段的有界状态。
无接受尝试仍计入现有 64 次 plateau 窗口；cache hit、invalid 等不能绕过窗口和时间检查。
总调用/时间预算、目标身份与成本严格下降的门槛不变；fast 一直是 unproven。

### 4.2 删除/保留方向及拒绝历史

先只实现游标，再单独试验方向排序，最后试验历史排序，避免无法归因。
方向可以基于同一阶段的实际接受与调用次数优先；不根据采集计时决定顺序，
以保持固定调用预算下采集开关的一致性。不预设删除方向总是更好，cross-region 仍需保留方向。

拒绝历史键使用 pass 身份、编辑种类、稳定节点 ID 或资源名，以及有序编辑结构；
节点 ID 的顺序需保留首次出现顺序，因为边界提升顺序可能影响 payload，不能一律改成集合。
首版仅支持单编辑，复合编辑不用原始基准索引构造“稳定签名”。
每运行至多 4096 项 / 8 MiB 编码元数据，条目记录基准摘要和拒绝类别，不缓存模型。
INCONCLUSIVE 不记作拒绝；本地校验失败与后端拒绝分列。接受相应编辑后清除其旧历史。

`defer` 只把旧基准拒绝放入当前阶段的延后队列；每最多 8 个普通尝试后安排 1 个延后尝试，
普通队列为空就直接排空延后队列。该比例是待测的起点，不是已验证的默认值。
延后尝试按当前模型重新生成和校验，不能伪装成 NOT_INTERESTING 或 exact cache hit。
预算不足时仍可能影响质量，必须单独记录退化；若没有稳定收益，保留 `off` 并不发布此选项。
延后队列最多保留 8 项当前版本尝试。接受发生时，未执行的 fine 尝试按保存的游标状态回退，
避免生成器预取越过尚未执行的编辑；下一尝试重新绑定当前版本，不能执行旧候选。
阶段切换清空队列，进入收尾后释放 fine 游标。Finalizer 始终忽略历史，
只保留既有同版本编辑去重和精确缓存。`--no-cache` 禁用精确缓存；显式启用的历史排序继续存在。

## 5. 验收与归因

### 5.1 分开衡量三类改动

| 类型 | 必须保持 | 可以变化 |
| --- | --- | --- |
| 快照/快速编辑/省排序/存储 | 同策略、固定调用预算下的编辑顺序、状态、payload、成本、停止及最小性；候选实际判定次数 | 准备耗时、clone/export/write 次数、RSS、性能统计 |
| 新校验/域缓存 | 内容与校验结论、候选状态、后端调用和最终结果 | checker/域扫描次数、缓存统计 |
| 游标/方向/历史排序 | 有效性、严格成本下降、故障身份、首尾复检、覆盖声明及预算契约 | 路径、摘要、判定次数和最终固定点；须逐项报告质量差异 |

同一配置的采集开关仍按既有决策/调度摘要严格比较；新增缓存的计数差异与
模型编辑模式的差异需要归一化比较，不能把包含时间和配置的整份 JSON 强行逐字节比较。
策略之间不承诺摘要一致。有限样本无退化，只能作为这些样本的验收结论。

第五批快速/IR 差分矩阵至少增加：多消费者/多输出、删除全部/空选/逆序/重复/越界、
缺失/部分/符号 shape、推导失败、自定义域、QDQ、外部权重读入、节点和 value 元数据、
提升图输出、initializer 同名 input、空可选输入、无序图、控制流捕获与非张量类型。
还要比较二次编辑、混合编辑、`region_graph()`、ID 与查询结果，确认支持与回退两种路径。
校验覆盖 strict/structural/permissive，比较完整 ValidationResult，而不只比较 valid 布尔值。

第六批增加 A/B/T 旧拒绝反转、资源旧拒绝反转、尺度恢复、单项卡点、阶段饥饿、
INCONCLUSIVE、零预算、恰好耗尽调用、时间中止、收尾接受重扫和历史队列淘汰测试。
复用已有恢复目标身份、实际产物损坏和 legacy/stateful 轨迹回归；不因新增模式削弱这些测试。

### 5.2 统一基准矩阵

在 perf 的 `benchmarks/reduction.py` 中补充 free-try 的 `gated`、`one-stubborn-init`，
保持同一输入构造与判定条件；`gated`、`one-stubborn-init` 已加入 `--case` 和 `all`。
先把改动后的同一脚本运行在原 perf 基线，再逐项开启改动；free-try 仅作额外参照，
它相对 main 的历史耗时不能充当 perf 的优化前结果。

| 组别 | 输入与预算 | 主要问题 |
| --- | --- | --- |
| 常规准备 | 五类 128/512；chain 与 missing-shape 1000/5000；固定调用及 1s/5s | 每候选准备、快照命中与规模增长 |
| 细分/重启 | one-stubborn-init 512/2000；gated 30/90/300，1000 后续扩展 | 接受后续扫的独立收益，大小与检查次数曲线 |
| 既有退化 | tight-calls=4、threshold-128、16/12/6、cross-region、non-monotonic、old-rejection | 同预算完整成本及固定点完成率 |
| 区域与回退 | 非连续模块、缺元信息、控制流、空输入、边界元数据 | 支持率及自动回退的总开销 |
| 缓存/存储 | preparation 缓存开/关；大 payload 的重复 checkpoint | checker 命中率、写入量与存储核验成本 |
| 真实后端 | 稳定故障模型与实际判定命令，首尾匹配同一身份 | 减少准备与减少判定各自占比；子进程资源另测 |

每组至少三次、新进程、新 run 目录，交替配置顺序，串行采样。先小规模和有限调用预算，
再对可承受的场景进行完整收敛；运行到预算的记录不能与收敛记录直接算加速比。
同时报告 `(nodes, bytes, outputs, inputs, initializers)`、最小性、实际调用、
全部尝试、耗时中位数/范围、峰值 RSS、写入量、回退次数及覆盖阶段成本。
合成判定的进程内 load 成本与真实后端进程启动、编译成本分开解释。

消融顺序：原 perf → 省排序/快照 → 快速删除 → 精确元数据缓存 → checkpoint 去重，
然后固定模型层实现比较 restart/cursor → 方向排序 → history defer。
保留 legacy/stateful/staged-fast/staged-complete 四组；搜索与收尾时间、实际判定分列。

正式验收前确定性能阈值：第五批以 chain/branches/gated 的准备成本至少降低 20% 为目标，
常规端到端耗时及 RSS 中位数恶化超过 10% 的样本逐项调查；毫秒级样本同时查看绝对差与范围。
第六批以 gated-300 同质量、同覆盖保证下实际判定至少减少 30% 为目标，且保留矩阵中
固定调用预算不发生完整成本退化、既有阈值样本的覆盖完成率不降低，才考虑推广。
这些是待验证门槛，不是性能预测；未达标时保留实验开关或撤下对应优化，不切默认策略。

### 5.3 当前可运行的对照入口

以下使用现有参数，可用于采集设计基线；不是本次已运行的性能结果。
新增模式的完整对照由 `benchmarks/batch56.py` 组织，正式命令及记录见实测报告。

```bash
uv run --locked python benchmarks/reduction.py \
  --case all --size 128 --repeat 3 --budget both \
  --candidate-budget 100 --time-budget 1 \
  --compare-staged --compare-instrumentation
uv run --locked python benchmarks/reduction.py \
  --case chain --size 128 --condition old-rejection --repeat 3 \
  --budget calls --candidate-budget 600 --compare-staged
uv run --locked python benchmarks/reduction.py \
  --case preparation --size 128 --repeat 3 --budget calls \
  --candidate-budget 600 --compare-instrumentation
```

## 6. 实施顺序与交付物

| 顺序 | 独立改动 | 完成条件 |
| --- | --- | --- |
| 5A | 统一 gated/one-stubborn 基准，保存原 perf 记录，补边界反例 | 固定输入、源码摘要、命令、环境与质量记录可复跑 |
| 5B | 快照所有权、按需 IR、有序图省排序、查询适配 | 原路径回退可用；三策略固定调用预算行为等价 |
| 5C | 单 `DeleteNodes` 快速路径 | 扩展差分矩阵全过，报告支持率、准备收益与 RSS |
| 5D | 校验/域元数据缓存；checkpoint 去重各自独立 | 分开证实 checker/写入减少；最终读回校验强制执行 |
| 6A | staged 可选 cursor，仅节点和资源续扫 | gated 改善且尺度恢复、cleanup、预算与覆盖测试通过 |
| 6B | 方向排序、历史延后各自独立实验 | 对阈值、跨区域与旧拒绝反转给出消融和退化表 |
| 7 | 真实后端、权重存储、并行的后续专项 | 先按新 profile 确定瓶颈，再决定投入 |

每批交付设计对应实现、必要回归、消融结果与可追溯证据摘要；默认值切换另行评审。
大权重共享存储仍需解决实际权重内容身份和最终独立打包；protobuf CopyFrom 不会自动共享权重。
暂不省略 checker，不把增量校验当作已证明安全的行为保持优化，也不同时引入并行判定。

第五、六批实现与设计对应关系、实测收益及退化样本见[本批结果](../performance/performance-batch56-results.md)。
本批结果不替换原有四批证据，也不支持宣称真实后端普遍加速。
