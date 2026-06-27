#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer from text file(s)."""

from __future__ import annotations

import argparse
from pathlib import Path

from bpe_tokenizer import BPETokenizer
from bpe_tokenizer.patterns import GPT2_PATTERN, SIMPLE_PATTERN


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument(
        "input",
        nargs="+",
        help="UTF-8 text file(s) to train on",
    )
    parser.add_argument(
        "-o", "--output",
        default="tokenizer.json",
        help="Output path for saved tokenizer (default: tokenizer.json)",
    )
    parser.add_argument(
        "-v", "--vocab-size",
        type=int,
        default=8192,
        help="Vocabulary size: 256 bytes + merges (default: 8192)",
    )
    parser.add_argument(
        "--pattern",
        choices=["gpt2", "simple"],
        default="gpt2",
        help="Pre-tokenization pattern (default: gpt2)",
    )
    parser.add_argument(
        "--special",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to register after training",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print training progress",
    )
    args = parser.parse_args()

    pattern = GPT2_PATTERN if args.pattern == "gpt2" else SIMPLE_PATTERN
    paths = [Path(p) for p in args.input]

    for p in paths:
        if not p.is_file():
            raise SystemExit(f"File not found: {p}")

    print(f"Training on {len(paths)} file(s), vocab_size={args.vocab_size}")
    tokenizer = BPETokenizer.train_from_files(
        paths,
        vocab_size=args.vocab_size,
        pattern=pattern,
        special_tokens=args.special,
        show_progress=args.progress,
    )

    tokenizer.save(args.output)
    print(f"Saved tokenizer to {args.output}")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    print(f"  merges:     {len(tokenizer.merges)}")
    print(f"  special:    {list(tokenizer.special_tokens.keys())}")

    # Quick sanity check on first file
    sample = paths[0].read_text(encoding="utf-8")[:500]
    ids = tokenizer.encode(sample)
    recovered = tokenizer.decode(ids)
    ok = recovered == sample
    print(f"  round-trip: {'OK' if ok else 'FAILED'} on {len(sample)} char sample")


if __name__ == "__main__":
    main()
