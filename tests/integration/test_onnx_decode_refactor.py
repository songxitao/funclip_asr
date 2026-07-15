import numpy as np
import torch

def test_numpy_decode_equivalence():
    """
    测试 NumPy 批量向量化解码算法与原 PyTorch/Python 循环解码算法在多 Batch 时的完全等价性。
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 模拟 3 个 batch，最长时间步 15，词表大小 50
    batch_size = 3
    max_time = 15
    vocab_size = 50
    blank_id = 0
    
    ctc_logits_np = np.random.randn(batch_size, max_time, vocab_size).astype(np.float32)
    encoder_out_lens = np.array([12, 15, 8], dtype=np.int32)
    
    # ----------------- 算法一：原 PyTorch 循环解码 -----------------
    ctc_logits_torch = torch.from_numpy(ctc_logits_np).float()
    res_loop = []
    for b in range(batch_size):
        x = ctc_logits_torch[b, : encoder_out_lens[b].item(), :]
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)
        
        mask = yseq != blank_id
        token_int = yseq[mask].tolist()
        res_loop.append(token_int)
        
    # ----------------- 算法二：NumPy 向量化批量解码 -----------------
    token_ids = np.argmax(ctc_logits_np, axis=-1)  # [B, T]
    
    # 构造有效时间长度掩码
    time_indices = np.arange(max_time)[None, :]  # [1, T]
    valid_mask = time_indices < encoder_out_lens[:, None]  # [B, T]
    
    # 构造 CTC 去重掩码
    shifted = np.roll(token_ids, 1, axis=-1)
    shifted[:, 0] = -1  # 确保每行的第一个 token 绝不因 roll 循环而被误判为重复
    repeat_mask = (token_ids == shifted)
    
    # 过滤掉 padding 区域、去重区域以及 blank token
    keep_mask = valid_mask & (~repeat_mask) & (token_ids != blank_id)
    
    res_numpy = []
    for b in range(batch_size):
        token_int_np = token_ids[b][keep_mask[b]].tolist()
        res_numpy.append(token_int_np)
        
    # 校验结果是否绝对相同
    print("原循环解码结果:", res_loop)
    print("NumPy 解码结果: ", res_numpy)
    
    assert res_loop == res_numpy, f"等价性校验失败！\nLoop: {res_loop}\nNumpy: {res_numpy}"
    print("等价性校验通过！结果完全一致。")

if __name__ == "__main__":
    test_numpy_decode_equivalence()
