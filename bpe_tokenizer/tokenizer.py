from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import regex

from .core import (
    build_vocab_from_merges,
    chunks_from_texts,
    encode_chunk,
    train_bpe,
)
from .patterns import GPT2_PATTERN
from .io import stream_files

Pair = tuple[int, int]


class BPETokenizer:
    """
    Byte-level BPE tokenizer with GPT-2-style regex pre-tokenization.

    Works on any UTF-8 text (all languages). Train on your corpus, then use
    encode/decode in LLM training or inference pipelines.
    """

    def __init__(
        self,
        *,
        merges: Optional[Mapping[Pair, int]] = None,
        pattern: str = GPT2_PATTERN,
        special_tokens: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.pattern = pattern
        self._regex = regex.compile(pattern)

        self.merges: Dict[Pair, int] = dict(merges or {})
        self.special_tokens: Dict[str, int] = dict(special_tokens or {})
        self.inverse_special: Dict[int, str] = {v: k for k, v in self.special_tokens.items()} #for decoding 

        # rank: lower = merge earlier during encoding
        self._ranks = {
            pair: rank 
            for rank, (pair, _ ) in enumerate(
                sorted(
                    self.merges.items(), key=lambda x: x[1]
                )
            )
        }

        # pair -> output token id (same data, convenient lookup)
        self._merge_out = dict(self.merges)

        self._vocab = build_vocab_from_merges(self.merges)

        #add special tokens to vocab
        for token_id, token_str in self.inverse_special.items():
            self._vocab[token_id] = token_str.encode("utf-8")



    @classmethod
    def train(
        cls,
        texts: Union[str, Iterable[str]],
        vocab_size: int = 8192,
        *,
        pattern: str = GPT2_PATTERN,
        special_tokens: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> "BPETokenizer":
        """
        Train a new tokenizer on text(s).

        `vocab_size` includes the 256 byte tokens plus learned merges.
        Special tokens are assigned ids starting at `vocab_size` after training.
        """
        if isinstance(texts, str):
            texts = [texts]

        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        chunks = chunks_from_texts(texts, pattern)
        if not chunks:
            raise ValueError("No text chunks found — is the corpus empty?")

        if show_progress:
            total_pieces = sum(chunks.values())
            print(f"Training on {total_pieces:,} pre-tokenized pieces, vocab_size={vocab_size}")

        merges = train_bpe(chunks, vocab_size, show_progress=show_progress)

        specials = special_tokens or []
        special_map = {tok: vocab_size + i for i, tok in enumerate(specials)}

        return cls(merges=merges, pattern=pattern, special_tokens=special_map)

    @classmethod
    def train_from_files(
        cls,
        paths: Union[str, Path, Iterable[Union[str, Path]]],
        vocab_size: int = 8192,
        **kwargs,
    ) -> "BPETokenizer":
        """Train from one or more UTF-8 text files."""
        if isinstance(paths, (str, Path)):
            path_list = [Path(paths)]
        else:
            path_list = [Path(p) for p in paths]

        # texts = []
        # for path in path_list:
        #     texts.append(path.read_text(encoding="utf-8"))

        # return cls.train(texts, vocab_size, **kwargs)
        texts = stream_files(path_list)

        return cls.train(
            texts, 
            vocab_size, 
            **kwargs
        )



    def encode(self, text: str, *, allowed_special: Optional[set[str]] = None) -> List[int]:
        """
        Text -> token ids.

        Splits on special tokens first (if any), then regex + BPE per chunk.
        """
        if not self.special_tokens:
            return self._encode_text(text)

        allowed = allowed_special or set()

        # Build a regex that splits on known special token strings
        parts = self._split_special(text, allowed)
        ids: List[int] = []
        for kind, segment in parts:
            if kind == "special":
                ids.append(self.special_tokens[segment])
            else:
                ids.extend(self._encode_text(segment))
        return ids

    def encode_batch(self, texts: Sequence[str], **kwargs) -> List[List[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    def decode(self, ids: Sequence[int], *, errors: str = "replace") -> str:
        """Token ids -> text. Invalid UTF-8 byte runs are handled per `errors`."""
        parts: List[bytes] = []
        for token_id in ids:
            if token_id in self.inverse_special:
                parts.append(self.inverse_special[token_id].encode("utf-8"))
            elif token_id in self._vocab:
                parts.append(self._vocab[token_id])
            else:
                raise KeyError(f"Unknown token id: {token_id}")

        return b"".join(parts).decode("utf-8", errors=errors)

    def tokenize(self, text: str) -> List[str]:
        """Human-readable tokens (decoded byte pieces)."""
        return [self._vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]


    def save(self, path: Union[str, Path]) -> None:
        """Save merges, pattern, and special tokens to JSON."""
        path = Path(path)
        data = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [[a, b, idx] for (a, b), idx in sorted(self.merges.items(), key=lambda x: x[1])],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        merges = {(int(a), int(b)): int(idx) for a, b, idx in data["merges"]}
        return cls(
            merges=merges,
            pattern=data.get("pattern", GPT2_PATTERN),
            special_tokens=data.get("special_tokens", {}),
        )



    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including bytes, merges, and special tokens."""
        base = 256 + len(self.merges)
        if self.special_tokens:
            return max(base, max(self.special_tokens.values()) + 1)
        return base

    def get_vocab(self) -> Dict[int, bytes]:
        return dict(self._vocab)

    def __len__(self) -> int:
        return self.vocab_size



    def _encode_text(self, text: str) -> List[int]:
        ids: List[int] = []
        
        for piece in self._regex.findall(text):
            if not piece:
                continue
            chunk = list(piece.encode("utf-8"))
            ids.extend(encode_chunk(chunk, self._ranks, self._merge_out))
        return ids

    def _split_special(self, text: str, allowed: set[str]) -> List[tuple[str, str]]:
        """Split text into ('text'|'special', segment) tuples."""
        if not self.special_tokens:
            return [("text", text)]

        # Sort longest-first so multi-char specials match correctly
        tokens = sorted(self.special_tokens.keys(), key=len, reverse=True)
        escaped = [regex.escape(t) for t in tokens]
        split_re = regex.compile("(" + "|".join(escaped) + ")")

        parts: List[tuple[str, str]] = []
        pos = 0
        for match in split_re.finditer(text):
            start, end = match.span()
            if start > pos:
                parts.append(("text", text[pos:start]))
            token = match.group(0)
            if token not in allowed:
                raise ValueError(f"Disallowed special token in text: {token!r}")
            parts.append(("special", token))
            pos = end
        if pos < len(text):
            parts.append(("text", text[pos:]))
        return parts
