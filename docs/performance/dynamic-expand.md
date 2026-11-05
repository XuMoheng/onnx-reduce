# 计算型 shape 边界

用户提供的内网模型描述包含 1573 个节点，opset 12、IR 7，没有中间 value_info。
节点 252 为 Unsqueeze，节点 260 为唯一的 Expand，第二个输入由
Constant / ConstantOfShape / Mul / Equal / Where 计算产生。人工单节点复现则将 shape 写为 initializer。

这暴露了与死代码删除不同的候选缺口。基线 `2995683` 已能直接提取单节点；
但它只保留 initializer 和直接 Constant 的数据，切断 Where 后会把 shape 提升成
`int64[8]` 图输入。这个类型只说明有八个整数，不保留八个整数的值。
本地构造模型上，该候选通过严格 ONNX 校验，但不再满足“编译时知道扩展尺寸”的模拟故障条件。
单纯增加删除次数或启用 `infer_shapes(data_prop=True)` 没有补回这组值。

## 新候选

提取 Expand / Reshape 时，可尝试 `ExtractNodes(materialize_shapes=True)`：

1. 找到保留节点的外部 shape 依赖。
2. 在源图上求值有大小和深度限制的纯常量表达式。
3. 将已知 shape 写成 initializer，重建数据输入和活跃输出，并重新推导候选的 shape。
4. 经正常的成本、模型校验与目标故障判定后决定是否接受。

原来的提取方式仍保留。每次提取最多访问 128 个值、深度 64，每个值最多 1024 个元素，
仅支持小规模标量或一维整数 / 布尔表达式。输入依赖、可覆盖默认值、自定义算子、本地函数、
不支持的计算和超限表达式不会被当成常量。完整边界见[架构说明](../architecture.md#活跃子图与图接口)。

候选顺序也有影响：较粗的保留分区若先被接受，会重建提取流；
有轮转额度的调度器可能长期到不了后面的 Expand。
现在同等稀有的家族优先探测 shape 消费者，其常量 shape 变体紧邻普通候选，先获得尝试机会。

## 构造验证

`tests/helpers.py::dynamic_expand_model` 构造 1573 个节点，保留上述节点索引、opset、IR、
输入 shape、目标 shape 和零 value_info，其余部分用 Identity 链模拟。
它不重建用户的完整 transformer 模型。

独立模拟判定器 `StaticExpandCheck` 要求 Expand 通向真实图输出、输入为
`[1,6,1,2,2,12,32,32]`，并且 shape 可以静态求值为 `[1,6,3,2,2,12,32,32]`。
这用于检验缩减器能否保留尺寸信息，不模拟 Typhoon 的真实 tiling 阈值。

2026-11-05 在 Python 3.13.17、ONNX 1.23.2、onnxscript 0.7.2 下，串行比较基线 `2995683` 和本次修改，
各调度器各运行一次，候选实际判定预算均为 64。判定数包含强制初检和独立最终复检：

| 调度器 | 节点数（基线 → 修改后） | 判定数（基线 → 修改后） | 修改后停止原因 |
|---|---:|---:|---|
| legacy | 7 → 1 | 66 → 37 | fixed_point |
| stateful | 8 → 1 | 66 → 37 | fixed_point |
| staged | 5 → 1 | 66 → 37 | fixed_point |

基线均因调用预算结束；两侧最终复检均为 interesting。修改后的模型为 310 字节，
仅保留 Expand，shape 为 initializer，数据输入直接采用原 Unsqueeze 输出的八维 shape。
这个比较没有复现用户所述“停在 350 节点”，也不能据此量化真实后端上的收益。

回归覆盖三种调度器、IR / auto 编辑、缓存开启 / 关闭。
开启缓存使用 64 次候选判定预算；关闭缓存用 128 次，以容纳收尾扫描中重复执行的候选。
本地原始对照、构造模型、最终模型与运行脚本保存在忽略目录 `output/dynamic-expand-20261105/`。
脚本使用第一个参数指定证据目录，第二个参数指定结果标签；通过 `PYTHONPATH` 选择源码的 `src/`：

```bash
PYTHONPATH=/path/to/source/src uv run --locked python \
  output/dynamic-expand-20261105/compare.py \
  output/dynamic-expand-20261105 rerun
```

回归执行：

```bash
uv run --locked pytest -q tests/model/test_constant_shapes.py tests/integration/test_live_fragments.py
```

## 内网复验与剩余边界

沿用原判定命令及故障签名，对原模型或已有缩减结果运行修改后的版本。
检查候选 / 接受日志中的 `materialize_shapes: true`、实际 initializer 数值、
最终复检及停止原因。不要把后端拒绝动态 shape、模型非法等其他错误判成原 Typhoon 故障。

目前只有用户提供的模型描述，没有原文件、完整常量内容、内网判定器及运行报告。
因此尚不能确认原 Where 链是否全部满足求值条件，也不能宣布真实模型已自动缩到单节点。
此次补充没有增加 rank 降低、轴合并或非单调 shape 阈值搜索；
即使得到八维单节点，也不表示自动得到 `[1,1024,1024] → [3,1024,1024]`，
更不表示找到全局最小触发尺寸。
