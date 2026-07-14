# 03 — 构建统一离线 Pipeline 处理类

**What to build:**
提供一个统一的 `OfflinePipeline` 流水线控制器，作为核心包向外的统一转写接口，实现与 FastAPI 路由的完全脱耦。

**Blocked by:**
02 — 提取时序对齐与 SRT/同人合并工具

**Status:** ready-for-agent

- [ ] 在 `src/funclip_pro/pipeline/` 目录下新建 `offline.py`，设计并实现 `OfflinePipeline` 业务流水线类。
- [ ] 该 Pipeline 类的构造器读取统一配置，内部按顺序加载并调度 ASR -> Segmentation -> Speaker (Cam++) -> Clustring -> Alignment -> Merge (VAD 段内合并) 的逻辑。
- [ ] 编写集成测试 `tests/test_p1_pipeline.py`，在不依赖网络和 Web API 服务的前提下，通过实例化 `OfflinePipeline` 直接对测试音频进行完整的分人转写流程，校验转写结果及格式的正确性。
