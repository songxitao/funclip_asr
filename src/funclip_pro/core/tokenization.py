"""字符级分词与解码工具（core 下沉版）。

职责：
  - CharTokenizer：字符级 token -> 文本还原（等价抽取自原 asr_onnx_service.py
    内 `__call__` 中的 `DefaultTokenizer` 内部类，按契约重命名为公开 `CharTokenizer`）。
  - 保留 tokens2text 原始逻辑，未做任何 45454 交替 / 字级时间戳对齐相关优化
    （Out of Scope）。

本模块为纯 Python，不依赖任何深度学习框架，可独立导入。
"""

from __future__ import annotations

# 等价原 asr_onnx_service.py:103 的 DefaultTokenizer.tokens2text 逻辑。
_SPACE = "<space>"
_UNK = "<unk>"


class CharTokenizer:
    """字符级 token 还原器。

    与原 `DefaultTokenizer` 完全一致：遍历 token id，按 tokens 表还原文本，
    - 跳过 <|...|> 类标签
    - "<space>" -> 空格
    - "<unk>"   -> 跳过
    """

    def __init__(self, tokens):
        # tokens: list[str]，索引即 token id
        self.tokens = tokens

    def tokens2text(self, ids):
        res = []
        for i in ids:
            t = self.tokens[i]
            if t.startswith("<|") and t.endswith("|>"):
                continue
            if t == _SPACE:
                res.append(" ")
            elif t == _UNK:
                continue
            else:
                res.append(t)
        return "".join(res)
