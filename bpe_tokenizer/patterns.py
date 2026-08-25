import regex


# openai cl100k approach. 
CODE_AWARE_PATTERN = (
    r"'(?i:s|t|re|ve|m|ll|d)"          # contractions, case-insensitive
    r"| ?[\p{L}_][\p{L}\p{N}_]*"       # identifiers: word or _word or snake_case or __dunder__
    r"|\p{N}{1,3}"                      # numbers, capped at 3 digits per piece
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"       # punctuation/operator runs, absorbing trailing newlines
    r"|\s*[\r\n]+"                      # indentation glued to its newline
    r"|\s+(?!\S)"                       # trailing whitespace at end of text
    r"|\s+"                             # any remaining whitespace
)