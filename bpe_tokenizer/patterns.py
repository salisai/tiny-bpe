
GPT2_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Simpler fallback if you want fewer boundaries (whole lines / whitespace splits).
SIMPLE_PATTERN = r"\S+|\s+"
