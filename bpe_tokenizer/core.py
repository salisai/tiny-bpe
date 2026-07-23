"""Low-level BPE merge logic and training."""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

Pair = Tuple[int, int]
Word = Tuple[int, ...]


def get_pair_counts(ids: Sequence[int]) -> Dict[Pair, int]:
    """Count consecutive token pairs in a single sequence."""
    counts: Dict[Pair, int] = defaultdict(int)
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


# list-based - this function is not used
def merge_ids(ids: Sequence[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`."""
    first, second = pair
    out: List[int] = []

    i = 0
    n = len(ids)

    while i < n:
        if i < n - 1 and ids[i] == first and ids[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def merge_word(word: Word, pair: Pair, new_id: int) -> Word:
    """Same as merge_ids but returns a tuple (hashable chunk key)."""
    first, second = pair
    out: List[int] = []

    i = 0
    n = len(word)

    while i < n:
        if i < n - 1 and word[i] == first and word[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def build_vocab_from_merges(merges: Dict[Pair, int]) -> Dict[int, bytes]:
    """Reconstruct byte sequences for every token id from merge rules."""
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    for (a, b), idx in sorted(merges.items(), key=lambda x: x[1]):
        vocab[idx] = vocab[a] + vocab[b]
    return vocab


def train_bpe(
    chunks: Dict[Word, int],
    vocab_size: int,
    *,
    show_progress: bool = False,
) -> Dict[Pair, int]:
    """
    Train byte-level BPE on pre-tokenized chunks.

    `chunks` maps a byte sequence (one regex piece) to its frequency in the corpus.
    Returns merge rules: (token_a, token_b) -> new_token_id.
    """
    num_merges = max(0, vocab_size - 256)
    merges: Dict[Pair, int] = {}

    if num_merges == 0:
        return merges

    words: Dict[int, List[int]] = {}
    freqs: Dict[int, int] = {}

    for wid, (word, freq) in enumerate(chunks.items()):
        words[wid] = list(word) # mutable
        freqs[wid] = freq

    pair_counts: Dict[Pair, int] = defaultdict(int)
    pair_to_words: Dict[Pair, set] = defaultdict(set)

    for wid, word in words.items():
        if len(word) < 2: 
            continue 

        f = freqs[wid]
        for a, b in zip(word, word[1:]):
            pair_counts[(a, b)] += f
            pair_to_words[(a, b)].add(wid)

    
    heap: List[tuple] = [
        (-cnt, (-a, -b), (a, b)) for (a, b), cnt in pair_counts.items()
    ]
    heapq.heapify(heap)

    # whenever a  pair's count change after merge, we need to add an update entry
    def push(pair: Pair) -> None: 
        cnt = pair_counts.get(pair, 0)
       
        if cnt > 0: 
            a, b = pair  
            heapq.heappush(heap, (-cnt, (-a, -b), pair))


    for step in range(num_merges):
        best_pair = None 
        best_count = 0 

        while heap: 
            neg_cnt, _, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current > 0 and -neg_cnt == current: 
                best_pair, best_count = pair, current 
                break 

        if best_pair is None:
            break

        new_id = 256 + step 
        merges[best_pair] = new_id 

        affected = list(pair_to_words.get(best_pair, ()))

        for wid in affected:
            word = words[wid]
            if len(word) < 2: #no pairing needed
                continue
            f = freqs[wid]
 
            old_local: Dict[Pair, int] = defaultdict(int)
            for a, b in zip(word, word[1:]):
                old_local[(a, b)] += 1
 
            new_word = _merge_word(word, best_pair, new_id)
 
            new_local: Dict[Pair, int] = defaultdict(int)
            for a, b in zip(new_word, new_word[1:]):
                new_local[(a, b)] += 1

            for p in old_local:
                pair_to_words[p].discard(wid)
            for p in new_local:
                pair_to_words[p].add(wid)
 
            touched = set(old_local) | set(new_local)
            for p in old_local:
                pair_counts[p] -= old_local[p] * f
            for p in new_local:
                pair_counts[p] += new_local[p] * f
 
            for p in touched:
                cnt = pair_counts.get(p, 0)
                if cnt <= 0:
                    pair_counts.pop(p, None)
                    pair_to_words.pop(p, None)
                else:
                    push(p)  # fresh heap entry reflecting the new count
 
            words[wid] = new_word
 
        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)
 
        if show_progress and (step + 1) % 500 == 0:
            print(
                f"  merge {step + 1}/{num_merges}  "
                f"pair={best_pair}  freq={best_count}  "
                f"affected_words={len(affected)}"
            )
 
    return merges
        
        


def encode_chunk(ids: List[int], ranks: Dict[Pair, int], merge_out: Dict[Pair, int]) -> List[int]:
    """
    Greedy BPE on one pre-tokenized piece.

    Repeatedly merge the adjacent pair with the lowest rank (earliest learned merge).
    """
    while len(ids) >= 2:
        # (position, rank) for pairs that have a learned merge
        candidates = []

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            rank = ranks.get(pair)
            
            if rank is not None:
                candidates.append((rank, i))

        if not candidates:
            break

        _, pos = min(candidates)
        pair = (ids[pos], ids[pos + 1])
        ids = ids[:pos] + [merge_out[pair]] + ids[pos + 2 :]

    return ids


def chunks_from_texts(
    texts: Iterable[str],
    pattern,
) -> Dict[Word, int]:
    """Split texts with regex, count byte sequences per piece."""
    import regex as re

    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    chunks: Dict[Word, int] = defaultdict(int)

    for text in texts:
        for piece in compiled.findall(text):
            if not piece:
                continue
            chunks[tuple(piece.encode("utf-8"))] += 1 #convert to utf-8 bytes

    return dict(chunks)




def _merge_word(word: Sequence[int], pair: Pair, new_id: int) -> List[int]:
    """Same semantics as core.merge_word, operating on a plain list."""
    first, second = pair
    out: List[int] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == first and word[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out
 