# 设计：AliMeeting 近场说话人分离评测 + 滑窗 Segmentation 改造

> 日期：2026-07-12 ｜ 状态：设计待审（尖子进一步判断中）
> 前置：2026-07-11-cer-der-test-plan-design.md（旧 AISHELL-4 方案，已判不匹配，本方案替代）
> 约束：本项目读代码走 codegraph，不直读 .py 源码

---

## 0. 背景与动机

### 0.1 旧方案的死穴
原 DER 评测用 **AISHELL-4**（远场 8 麦克风阵列会议），与实际部署场景（近场单声道实时字幕/剪辑）不匹配：
- 远场单通道 Cam++ embedding 区分度差，DER 虚高
- MISS 稳定 45K 查不出根因——实为远场通道对齐问题，非模型问题
- 多麦克风阵列增益被丢弃（只挑 1 路单通道）

### 0.2 换近场测试集的验证
试跑 AliMeeting near（领夹麦，近场信号质量）单场 R8002_M8002：
- **MISS 从 45K 降到 0.7%**——证明旧 MISS 是远场问题，换近场自动消失
- 但 DER=49.21%，其中 **CONF 占 48.3%**（说话人混淆），FA/MISS 几乎为 0
- 即：水分消失了，暴露的真实问题是"说话人分不开"

### 0.3 CONF 主因诊断
服务端说话人分离流程为：`VAD 切段 → 每段提 1 个 Cam++ embedding → 聚类 → 贴 speaker`。
- VAD（funasr FSMN）为 **ASR 设计**，倾向合并完整句，段长不固定且偏大（~8s）
- 8s 段里多人交替说话 → 这段的 embedding 是"混合向量"
- 混合向量聚类 → 同人向量黏在一起分不开 → CONF 高
- 数据佐证：hyp 102 段 vs ref 866 段，段粒度差 8 倍，说明 VAD 合并太狠

**结论：CONF 48% 的根因不是 Cam++ 模型不行，是喂给它的"段"不对——VAD 段太大，段内多人，向量被污染。**

---

## 1. 核心方案：滑窗 Segmentation（业界标准）

### 1.1 思想
说话人分离的"最小单位"从 **VAD 段** 换成 **固定滑窗**：
- 窗长 1.5s，步长 0.5s（相邻窗重叠 1s）
- 每个窗独立提 Cam++ embedding → 聚类 → 每窗各贴 speaker 标签
- 相邻同人窗合并 → 还原完整说话人时间轴
- **VAD 在 diarization 里退场**，仅用于"框出有语音的大区域"避开纯静音窗，不当输出单位、不贴标签

### 1.2 为什么滑窗不怕"切在句子中间"
关键认知：**窗是"提向量的临时工具"，不是"最终输出段"**。
- 1.5s 窗内基本是单人说话，embedding 纯净
- 重叠窗（重叠 1s）信息不丢
- 窗切在句中无所谓——提的向量照样代表"当时在说话的那个人"
- 最后按 speaker 合并相邻同人窗，还原完整段

对比 VAD 段：VAD 段是"最终输出段"，切坏了直接导致结果坏；滑窗是中间产物，最后合并还原。

### 1.3 跨界窗（A/B 边界处的混合窗）处理
- 聚类是**全局找自然分堆**，不是"逐窗判断这个窗是谁"
- 跨界窗是少数（约 8%），纯 A 窗、纯 B 窗是多数（约 92%）
- 少数混合向量淹没不了多数纯向量的分堆结构
- 重叠窗让边界被多个窗覆盖，后处理看相邻窗标签可纠正孤立错分
- 举例：A 说 10s + B 说 10s → 约 39 窗，其中 36 纯 / 3 混，聚类看 36 个纯窗的结构

### 1.4 局限（诚实声明）
- **快速抢话/叠词场景**（说话人切换 < 窗长 1.5s）会失效：每个窗都是混合，聚类崩
- 会议对话一般人说话 ≥ 1-2s，1.5s 窗够用；叠词场景需额外处理（未来工作）
- 这是滑窗的天花板，但比 VAD 段（几乎段段混合）好得多

---

## 2. 评测设计：AliMeeting near 近场

### 2.1 数据集
- AliMeeting Test set（已下至 `testset/Test_Ali/`）
- near 子集：20 场，每场 2-4 人领夹麦（每人 1 个单声道 wav）+ TextGrid 标注
- 中文会议，15-30 分钟/场

### 2.2 预处理（已试跑验证 ✅）
脚本 `ali_near_prep.py`（零第三方依赖，纯标准库 wave/array/re）：
1. **混音**：每场 N 个领夹麦 wav 相加 → 单通道混合 wav（模拟 1 个近场麦收多人）
   - 对齐到最长长度，短的补 0，相加后除路数防爆音，clip 到 int16
2. **TextGrid → RTTM**：每场 N 个 TextGrid 合并 → 参考 RTTM
   - 解析每个 TextGrid 的 interval（xmin/xmax/text），text 为空跳过（静音段）
   - speaker ID 从文件名 SPK 编号提取
   - 输出标准 RTTM：`SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>`
- 已修 bug：TextGrid 头部 xmin/xmax 误匹配（正则 `.*?` → `\s*` 紧邻）

### 2.3 评测流程
脚本 `ali_der_eval.py`（用 conda asr_ui_env python 跑）：
1. 混合 wav → flac（规避服务 50MB 上传上限）
2. POST `:8002/transcribe`，参数 `diarize=true`、`num_speakers=<ref人数>`（oracle-K）、`vad_strategy=always`、`diarize_strategy=spectral`
3. 取 segments（start/end 单位 ms）→ hyp_segs（转秒）
4. 解析参考 RTTM → ref_segs
5. 算 DER（0.25s collar + 贪心说话人映射，复用 `der_eval.compute_der`）
6. 全量 20 场，按参考语音时长加权平均

### 2.4 试跑结果（R8002_M8002，4 人，2062s）
```
DER = 49.21%
FA=361(0.3%)  MISS=920(0.7%)  CONF=66145(48.3%)  REF=137027
hyp 102 段，4 人都分出（811/789/150/250s），未崩塌
latency=26.8s, engine=torch
```
- ✅ MISS/FA 几乎为 0（vs AISHELL-4 的 45K MISS）→ 换近场对了
- ⚠️ DER 49% 主因 CONF 48% → 需滑窗改造（本设计核心）

---

## 3. 改造范围

| 模块 | 改动 | 说明 |
|---|---|---|
| `speaker_engine.py` | 新增滑窗 segmentation | `cluster()` 新增 segmentation 模式：输入整段音频 → 内部滑窗切 → 逐窗提 embedding → 聚类 → 合并。Cam++ 和聚类逻辑不变 |
| `asr_onnx_service.py` | diarize 分支接滑窗 | `diarize=true` 时走滑窗 segmentation（VAD 仅框区域），ASR 仍走原 VAD |
| `der_eval.py` | 不变 | 接收 segments 算 DER，与 segmentation 方式解耦 |
| `ali_near_prep.py` | 已就绪 | 近场混音 + RTTM 生成 |
| `ali_der_eval.py` | 已就绪 | 调服务 + 算 DER |

### 3.1 关键设计约束
- **Cam++ 模型和聚类不动**——只换"怎么切段喂给它"
- **ASR 的 VAD 不动**——VAD 仍为 ASR 服务，只在 diarization 里退场
- **两套 segmentation 各管各的**：ASR 用 VAD 段识别文字，diarization 用滑窗分人

---

## 4. 方向决策记录

| 方向 | 描述 | 优劣 | 决策 |
|---|---|---|---|
| A 调小 VAD max_duration | VAD 段超时硬切 | 治标+硬切副作用（切断话语），仅验证假设用 | ❌ 放弃（副作用） |
| B 滑窗 segmentation | 窗为最小单位，VAD 退场 | 根本解决，对齐业界标准（pyannote），需改架构 | ✅ 选定 |
| C VAD 段内二次切分 | 长段按能量谷再切 | 折中，不如 B 彻底 | ❌ 备选 |

---

## 5. 待决问题（尖子进一步判断）

1. **窗长/步长参数**：1.5s 窗 + 0.5s 步长是业界经验值，是否需在本数据集上调优
2. **叠词场景**：快速抢话/叠词是滑窗死穴，是否需额外机制（如重叠检测）
3. **far 对照**：是否同时跑 AliMeeting far 单通道，出"近场 vs 远场"对比论据
4. **oracle-K**：评测用真实人数作弊，实际部署不知人数；是否加"自动估 K"对照组
5. **滑窗实现位置**：speaker_engine 内部 vs 服务端预处理层

---

## 6. 已验证 / 待验证

| 项 | 状态 |
|---|---|
| near 混音 + RTTM 生成 | ✅ 试跑通过（ali_near_prep.py）|
| DER 评测管线（调服务+算DER）| ✅ 试跑一场（ali_der_eval.py）|
| MISS 是远场问题（换近场消失）| ✅ 验证 |
| CONF 主因是 VAD 段污染 | 🟡 强烈怀疑（数据佐证），待滑窗改造后对比确认 |
| 滑窗 segmentation 实现 | ⏸ 待尖子确认方向后实施 |
| 全量 20 场 DER | ⏸ 待滑窗改造后跑 |

---

## 7. 复跑命令

```bash
# 环境
E:/conda/envs/asr_ui_env/python.exe  # 唯一可用环境

# 1. 近场预处理（混音+RTTM，零依赖，受管python即可）
python ali_near_prep.py R8002_M8002

# 2. DER 评测（需 :8002 服务在跑）
E:/conda/envs/asr_ui_env/python.exe ali_der_eval.py R8002_M8002

# 3. 起服务（端口 8002）
E:/conda/envs/asr_ui_env/python.exe asr_onnx_service.py
```

---

*本设计基于 2026-07-12 brainstorm 会话，核心结论：VAD 段为 ASR 设计不适合 diarization，滑窗 segmentation 是业界标准根本解。待尖子审阅判断后，转 writing-plans 出实施计划。*
