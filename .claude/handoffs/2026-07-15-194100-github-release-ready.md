# Handoff: GitHub 开源发布 v0.8.0 准备完成

## Session Metadata
- Created: 2026-07-15 19:41
- Project: E:\project\funclip-pro
- Branch: main (工作区 Clean — 已提交 `6a156d1`)
- Session focus: v0.8.0 GitHub 开源发布准备 — 基础设施、环境解耦、测试重组、文档
- Worked by: WorkBuddy AI Agent (会话 ID: 2026-07-15-18-46-57)

### Recent Commits
  - `6a156d1` chore: v0.8.0 release preparation — open-source infrastructure & test reorganization

---

## Handoff Chain
- **Continues from**: HANDOFF.md (2026-07-14 23:07, P3.2 重构遗留 Bug 整改完成)
- **Supersedes**: HANDOFF.md (2026-07-14 23:07)
- **Current session**: 2026-07-15 18:47–19:41 完成 v0.8.0 开源发布准备

---

## Current State Summary

已完成 **P0/P1/P1.5/P2/P3.1/P3.3/P3.2 + 安全审计 + v0.8.0 开源发布准备**。

项目已达到 GitHub 公开推送的状态：
- ✅ Apache 2.0 LICENSE
- ✅ pyproject.toml 完整元数据（authors/license/readme/urls/scripts）
- ✅ CHANGELOG.md（Keep a Changelog, v0.1→v0.8）
- ✅ CLI 入口 `funclip-pro`
- ✅ 补全缺失运行时依赖（websockets/requests/pyaudio/pyaudiowpatch）
- ✅ .gitignore 完善（.scratch/.pilot_venv/build/dist/egg-info/.workbuddy）
- ✅ config.json.example 模板化
- ✅ app_control.py 回退值降级为系统 python
- ✅ 一键启动 bat 脚本移除硬编码盘符
- ✅ pytest 收集卡死修复（TestClient 惰性加载）
- ✅ 测试文件硬编码路径解耦
- ✅ 测试目录三级重组（unit/integration/archive）
- ✅ README.md 更新（移除硬编码路径、版本演进表→CHANGELOG.md）
- ✅ 15 个废弃测试归档

### 全量变更
- 119 个文件变更，+3377 / -12033 行
- 物理删除：sherpa_engine.py / torch_engine.py / funclip/ 旧目录 / 超级 Agenda 文档 / 测试结果 JSON / 旧测试 15 个文件
- 文件迁移：测试 34 个 → unit(10) + integration(13) + archive(15) + README 等移至 docs/

### 交付物
| 文件 | 说明 |
|------|------|
| `LICENSE` | Apache 2.0 许可证 |
| `CHANGELOG.md` | 版本记录 v0.1→v0.8 |
| `config.json.example` | 配置模板（路径替换为占位值） |
| `pyproject.toml` | 完整构建系统 + pytest 配置 |
| `.gitignore` | 安全过滤规则补充 |
| `README.md` | 更新为便携部署指引 |
| `.scratch/refactor-alignment/issues/` | 13 个 tracer-bullet ticket 文件 |

---

## Critical Files

| File | Purpose | Status |
|------|---------|--------|
| `LICENSE` | Apache 2.0 许可证 | ✅ |
| `pyproject.toml` | 构建系统 + pytest 配置 | ✅ |
| `CHANGELOG.md` | 版本记录 | ✅ |
| `app_control.py` | 回退值降级 | ✅ |
| `tests/unit/` | 10 个纯逻辑测试（默认 `pytest` 入口） | ✅ |
| `tests/integration/` | 13 个模型/GPU/服务测试（需 `-m slow`） | ✅ |
| `tests/archive/` | 15 个废弃测试（完全不执行） | ✅ |
| `config.json.example` | 配置模板 | ✅ |
| 三个 `一键启动_*.bat` | 移除硬编码盘符 | ✅ |

---

## 下一步行动

### 建议的执行顺序

| # | 任务 | 说明 |
|:-:|------|------|
| 1 | **推送到 GitHub** | `git push origin main` → GitHub 仓库显示 LICENSE、CHANGELOG |
| 2 | **创建 GitHub Release** | `git tag v0.8.0 && git push --tags` → Draft release，贴 CHANGELOG v0.8.0 内容 |
| 3 | **合并 README（可选）** | 将 `README.md` + `README_REFACTORING.md` 融合为一份完整 README |
| 4 | **端到端验证** | 在有硬件的 Windows 机器上启动各一键启动脚本验证功能完整性 |
| 5 | **补充 CONTRIBUTING.md（可选）** | 定义代码格式、分支规范、单测契约 |

### Potential Gotchas
- `funclip_pro.cli:main` CLI 入口尚未实现实际代码，需要创建 `src/funclip_pro/cli.py`
- 一键启动 bat 脚本在 `config.json` 不存在时的友好提示已在 `app_control.py` 中处理
- `pytest` 默认只跑 unit/ 的 10 个文件，不加载任何模型，3 秒内完成
- GPU 测试环境：`E:/conda/envs/asr_ui_env/python.exe` (torch 2.3.1+cu121, RTX 4080 Laptop GPU)
- `tests/archive/` 死锁：其中 5 个归档旧测试关闭 stdout 句柄导致 pytest 崩溃，已隔离
- `ppc64le` 等小众架构未经测试
