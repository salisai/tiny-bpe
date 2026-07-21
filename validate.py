from bpe_tokenizer import BPETokenizer

CORPUS = """
The night sky has fascinated humans for millennia. Ancient civilizations mapped
the stars and used them for navigation, agriculture, and religious ceremonies.
مرحبا بالعالم — hello world in Arabic.
def add(a, b):
    return a + b
The café serves crème brûlée daily.
"""

TEST_CASES = [
    "The quick brown fox.",
    "The naïve résumé of François Müller",
    "مرحبا بالعالم",
    "def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)",
    "Temperature: -40°C = -40°F",
    "Don't you think it's amazing?",
]


def round_trip(tokenizer: BPETokenizer, text: str) -> bool:
    return tokenizer.decode(tokenizer.encode(text)) == text


def main() -> None:
    print("Training tokenizer (vocab_size=512)...")
    tok = BPETokenizer.train(
        CORPUS,
        vocab_size=512,
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )

    print(f"vocab_size={tok.vocab_size}, merges={len(tok.merges)}")

    passed = 0
    for text in TEST_CASES:
        ok = round_trip(tok, text)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {text[:50]!r}")
        passed += int(ok)

    # save / load round-trip
    path = "/tmp/test_tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    ok = round_trip(loaded, TEST_CASES[0])
    print(f"  [{'OK' if ok else 'FAIL'}] save/load round-trip")

    # special token
    with_special = "<|endoftext|>Hello"
    ids = tok.encode(with_special, allowed_special={"<|endoftext|>"})
    decoded = tok.decode(ids)
    ok = decoded == with_special
    print(f"  [{'OK' if ok else 'FAIL'}] special token encode/decode")

    print(f"\n{passed}/{len(TEST_CASES)} text round-trips passed")


if __name__ == "__main__":
    main()
