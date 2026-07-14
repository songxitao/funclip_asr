# 04 — 重塑生产接口与客户端 (完成 P1 收口)

**What to build:**
清空 FastAPI 主服务与 CLI 中的全部冗余算法实现，完成生产系统与命令行接口向重构后本地包的无缝切换。

**Blocked by:**
03 — 构建统一离线 Pipeline 处理类

**Status:** ready-for-agent

- [ ] 清理 `asr_onnx_service.py` 内部除了 FastAPI 接口路由和跨域/基础配置外的一切算法函数，直接实例化 `OfflinePipeline` 处理端点 `/transcribe` 的业务。
- [ ] 修改 `cli_transcribe.py`，使命令行工具统一消费 `OfflinePipeline` 提取数据。
- [ ] 精度回归验证：在 8003 端口运行测试服务，跑评测脚本 `python ali_der_eval.py R8002_M8002 seg_clustering`。
- [ ] 确保测试通过，且算出的 DER 结果对齐重构前指标（14.60% - 15.13% 范围），无精度和转写文本回归。
