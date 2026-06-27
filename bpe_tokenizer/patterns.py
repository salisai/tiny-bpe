"""Pre-tokenization regex patterns (applied before byte-level BPE)."""

# GPT-2 / tiktoken-style: splits contractions, words, numbers, punctuation, whitespace.
# \p{L} and \p{N} match letters and numbers in any script (Arabic, CJK, etc.).
GPT2_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Simpler fallback if you want fewer boundaries (whole lines / whitespace splits).
SIMPLE_PATTERN = r"\S+|\s+"
