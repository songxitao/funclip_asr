# Contributing to FunClip Pro

感谢你考虑为 FunClip Pro 贡献代码！以下指南帮助你在本地搭建开发环境并参与开发。

## 环境搭建

1. **Fork 仓库**，克隆到本地：
   ```bash
   git clone https://github.com/<your-username>/funclip_asr.git
   cd funclip_asr
   ```

2. **创建虚拟环境**（推荐 conda）：
   ```bash
   conda create -n funclip-pro python=3.11
   conda activate funclip-pro
   pip install -r requirements.lock
   ```

3. **安装 pre-commit 钩子**：
   ```bash
   pip install pre-commit
   pre-commit install
   ```
   pre-commit 会自动在独立 venv 中管理 pytest 等工具，不污染你的开发环境。

## 代码风格

- 使用 [ruff](https://docs.astral.sh/ruff/) 进行代码检查和格式化
- pre-commit 会自动运行 ruff，提交前确保通过
- 导入排序、行长度等规则遵循 `pyproject.toml` 中的 `[tool.ruff]` 配置

## 运行测试

### 单元测试（无需加载 ML 模型）
```bash
pytest tests/unit/ -x -q
```

### 集成测试
```bash
pytest tests/integration/ -x -q
```

### 运行所有测试（跳过慢测试）
```bash
pytest -x -q -m 'not slow'
```

### 运行慢测试（需 GPU 和模型文件）
```bash
pytest -x -q -m slow
```

## 提交 PR

1. 从 `main` 分支创建新分支：`git checkout -b feature/your-feature`
2. 在源码 `src/funclip_pro/` 下修改
3. 为新增功能编写单元测试（放在 `tests/unit/`）
4. 确保所有测试通过
5. 提交前确认 pre-commit 检查通过：`pre-commit run --all-files`
6. 提交并推送，创建 Pull Request

## 项目结构

```
funclip_pro/
├── src/funclip_pro/      # 核心源码
│   ├── core/              # ASR 引擎、模型封装
│   ├── pipeline/          # 处理管线
│   ├── config/            # 配置管理
│   └── utils/             # 工具函数
├── tests/
│   ├── unit/              # 单元测试（无模型依赖）
│   └── integration/       # 集成测试（轻量 mock 或外部服务）
├── docs/                  # 文档
└── scripts/               # 辅助脚本
```

## 问题反馈

- 提交 [Issue](https://github.com/songxitao/funclip_asr/issues) 报告 bug 或功能建议
- 项目讨论可在 Issue 中进行
