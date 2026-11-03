# 文档导航

onnx-reduce 在保留用户指定故障的前提下，缩减 ONNX 模型。以下文档描述当前
`0.1.0` 实现；未支持的能力单独列在[架构说明的当前边界](architecture.md#当前边界)中。

## 按任务阅读

| 你要完成的任务 | 阅读入口 |
|---|---|
| 安装并跑通第一个示例 | [项目 README](../README.md) |
| 为自己的后端编写判定脚本、设置预算、确认结果 | [使用指南](usage.md) |
| 理解候选如何生成、校验和接受，定位相关源码 | [架构说明](architecture.md) |
| 排查无法启动、判定异常、最终复检失败或中断 | [排障指南](troubleshooting.md) |
| 搭建开发环境、运行验证、提交贡献 | [贡献指南](../CONTRIBUTING.md) |
| 查看当前版本与依赖 | [项目元数据](../pyproject.toml) |

首次接入建议按「README → 使用指南 → 排障指南」阅读；修改实现前再阅读架构说明。

## 开发计划

- [性能优化计划](performance-todo.md)
- [第三批设计：有状态搜索调度器](stateful-search-design.md)
- [第四批设计：分阶段搜索和独立收尾](staged-search-design.md)
- [性能基线：采集与基准命令](performance-baseline.md)
- [性能基线实测：2026-11-02](performance-results.md)
- [第二批性能实测：2026-11-03](performance-batch2-results.md)
- [第三批性能实测：2026-11-03](performance-batch3-results.md)
- [第四批性能实测：2026-11-03](performance-batch4-results.md)

## 命令与路径约定

文档中的命令均从 **Git 仓库根目录**（包含 `pyproject.toml` 的目录）执行。
当前性能工作区对应 `1012-perf/`，克隆到其他位置时以实际仓库根目录为准。
先按 README 安装依赖。`your-backend`、`YOUR_TARGET_ERROR` 和 `/absolute/path/...`
表示必须替换的内容；`examples/relu/` 中的教学示例无需外部后端。

参数名和默认值以当前 `uv run --locked onnx-reduce --help` 为准。
判定契约与选项集中在使用指南，报告、日志和恢复操作集中在排障指南，内部设计集中在架构说明。
