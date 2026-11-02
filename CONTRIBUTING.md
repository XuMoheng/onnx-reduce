# 贡献指南

## 准备开发环境

需要 Python >= 3.11、Linux / macOS 和 uv。获取源码见 [README](README.md#安装)。
以下命令均从 Git 仓库根目录（包含 `pyproject.toml`）运行：

```bash
uv sync --locked
uv run --locked onnx-reduce --help
```

`uv sync --locked` 安装运行依赖及 pytest、ruff 等开发依赖。
开发前阅读[架构说明](docs/architecture.md)；判定相关改动同时核对
[判定契约](docs/usage.md#编写故障判定脚本)和[报告说明](docs/troubleshooting.md#读取退出码与报告)。

## 验证改动

常规代码验证与当前 [CI](.github/workflows/ci.yml) 一致：

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests examples
uv run --locked ruff format --check src tests examples
```

测试只使用仓库内的模型和判定替身，不需要安装真实推理后端。
CI 在 Linux / macOS 上覆盖 Python 3.11、3.12、3.13。

涉及打包、依赖或发布元数据时，再执行：

```bash
uv build
uv run --no-project --with twine python -m twine check --strict dist/*
```

CI 还会验证 wheel 的隔离安装，以及从源码分发包解压后运行测试。
修改依赖时同步更新 `uv.lock`；不要为了运行验证顺手升级无关依赖。
`README.md` 用作发布包长描述，必须保留有效内容以通过严格元数据检查。

文档改动需检查相对链接，核对参数、默认值、退出码和报告字段，并实际运行涉及的教学命令。
完整教学流程见 [README](README.md#快速开始)。

## 提交贡献

1. 基于当前代码创建主题分支，保持改动围绕一个明确问题。
2. 行为修复附上能复现问题的最小模型或回归用例；说明输入、预期行为与实际行为。
3. 更新行为的主要文档位置，README 保留概述并链接详细说明；不要将计划能力写成已实现功能。
4. 提交前运行相关验证，在 Pull Request 中说明问题、最终行为和实际执行的验证结果。

提交 issue 时所需的信息见[准备缺陷报告](docs/troubleshooting.md#准备缺陷报告)。
