# SenseVoiceSmall ASR 微服务 OpenVINO 引擎集成与 CPU 推理优化设计规约

## 1. 背景与目标 (Background & Goals)

在当前的 ASR CPU 推理基准评估中，虽然我们已经完成了 NumPy 向量化批量解码以消除 Python 层面的 GIL 锁瓶颈，但 `ONNX-CPU` (36.85秒) 依旧显著慢于 `PyTorch-CPU` (24.61秒)。

其根本性能死穴在于：**ONNX Runtime 的默认 CPU EP 缺乏对量化 Transformer 算子的全图融合**，这导致推理过程中存在高频的 `INT8 ➡️ FP32 动态反量化` 与内存拷贝开销，占用了前向推理（Forward Pass）98% 以上的耗时。

本设计的目标是：
1.  正式引入 **Intel OpenVINO 推理引擎**，完全用 OpenVINO 原生 Python Core 代替 ONNX Runtime 的 `InferenceSession`。
2.  通过 OpenVINO 底层强大的 Transformer 算子融合与 CPU JIT 编译优化，将 CPU 推理的整体耗时压缩至 **12~15秒** 左右，实现对 PyTorch-CPU 基线的强力反超。
3.  补齐物理绑核与多线程配置，将核心锁定与多线程库防线限制在 **6 核 / 6 线程**，杜绝过度抢占（Oversubscription）开销。

---

## 2. 方案权衡与架构决策 (Approaches & Trade-offs)

### 2.1 方案 A (已选定)：直接使用 OpenVINO 原生 Python API (Core)
*   **实现方式**：在 `asr_onnx_service.py` 内部引入 `openvino.Core`，直接读取 ONNX 文件并对其进行编译，不再调用 `onnxruntime`。
*   **权衡**：
    *   **优点**：加速性能最佳，底层 Transformer Attention 被合并为单个执行 Kernel，能完美应对 SenseVoiceSmall 的动态输入维度。
    *   **缺点**：需对 `SenseVoiceSmall` 包装类的载入与 `infer` 逻辑进行小幅代码重构。

### 2.2 方案 B (已弃用)：使用 ORT + OpenVINO Execution Provider (ORT-OV-EP)
*   **实现方式**：依然在 ONNX Runtime 的 `InferenceSession` 下运行，仅在 providers 中添加 `OpenVINOExecutionProvider`。
*   **权衡**：
    *   **优点**：代码侵入极小。
    *   **缺点**：在多 Batch 动态尺寸输入下兼容性较差，常遭遇算子映射失效或反复在底层重新编译的开销，加速效果远逊于原生 API。

---

## 3. 核心设计与代码重构路径 (Detailed Design)

### 3.1 物理核心与多线程防线对齐 (Task 1 融合)
在 `asr_onnx_service.py` 头部，利用 `psutil` 硬绑定 CPU 的前 6 个物理核心，且设置多线程环境变量软防线为 **6**，防止 CPU 线程无秩序暴涨抢占导致性能崩塌：

```python
import os
import psutil

# 1. 强力锁定 CPU 核心在前 6 个上，防止上下文切换与卡死
try:
    psutil.Process().cpu_affinity([0, 1, 2, 3, 4, 5])
except Exception as e:
    print(f"警告：设置 CPU 亲和性失败: {e}")

# 2. 设置多线程计算库的环境变量软防线
for env_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[env_var] = "6"
```

### 3.2 ASR 模型加载重构
修改 `asr_onnx_service.py` 中的 `SenseVoiceSmall` 类的初始化，使用 OpenVINO 替代 ONNX Runtime，并配置编译线程：

```python
from openvino import Core

class SenseVoiceSmall(SenseVoiceSmallONNX):
    def __init__(self, model_dir, batch_size=1, quantize=True, device_id="-1", **kwargs):
        # 保留原基类基础属性的初始化
        super().__init__(model_dir, batch_size=batch_size, device_id=device_id, quantize=quantize, **kwargs)
        
        tokens_path = os.path.join(model_dir, "tokens.json")
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)
            
        # 1. 初始化 OpenVINO Core
        self.ie = Core()
        model_path = os.path.join(model_dir, "model_quant.onnx" if quantize else "model.onnx")
        
        # 2. 载入模型并配置 CPU 物理线程编译限制
        ov_model = self.ie.read_model(model_path)
        self.compiled_model = self.ie.compile_model(ov_model, "CPU", config={
            "INFERENCE_NUM_THREADS": "6",
            "NUM_STREAMS": "1"
        })
```

### 3.3 ASR 推理方法重构
重写 `SenseVoiceSmall.infer` 接口，以便直接利用 OpenVINO 接口处理 NumPy 数据输入并映射输出节点：

```python
    def infer(self, feats, feats_len, language, textnorm):
        # OpenVINO compilation model 直接接收 numpy list 的输入组合
        # 传入的 inputs 顺序与结构需要与 ONNX 模型的 input nodes 完全一致
        results = self.compiled_model([feats, feats_len, language, textnorm])
        
        # 提取转写 Logits 与实际编码长度
        ctc_logits = results[self.compiled_model.output(0)]
        encoder_out_lens = results[self.compiled_model.output(1)]
        
        return ctc_logits, encoder_out_lens
```

---

## 4. 验证与测试方案 (Verification Plan)

### 4.1 跑评时延维度修复 (`test_openvino_speed.py`)
由于 SenseVoiceSmall 包含低帧率拼接 (LFR)，拼接因子为 7，因此原本的 speech 输入特征维度应为 $7 \times 80 = 560$。
*   **修改**：将 `tests/test_openvino_speed.py` 第 14 行的模拟特征维度 `feat_dim = 80` 修改为 `feat_dim = 560`。
*   **运行**：`python tests/test_openvino_speed.py`，验证 OpenVINO 与 ORT 在 CPU 推理上的吞吐加速对比。

### 4.2 接口与大 Batch 流水线集成测试 (`test_onnx_service_integration.py`)
编写一个全面的集成测试，通过 `/transcribe` FastAPI 端点测试：
1.  验证服务能正常初始化并载入 `PUNC_MODEL`、`VAD_MODEL` 及 `OpenVINO ASR` 模型。
2.  模拟真实长音频，验证 VAD 分段 ➡️ 合并段 ➡️ 传入 ASR 大 Batch (16) 推理 ➡️ 标点后处理的最终文本还原度和可读性。

### 4.3 物理核绑定验证 (`test_affinity.py`)
编写 `tests/test_affinity.py`，在导入 `asr_onnx_service` 后断言当前进程的 `cpu_affinity()` 严格处于 `[0, 1, 2, 3, 4, 5]`（绑定的 6 个大核心物理核）之内。
