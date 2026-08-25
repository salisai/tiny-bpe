"""
Fast greedy BPE encoding for a single pre-tokenized chunk.
"""

from __future__ import annotations 

import heapq 
from typing import Dict, List, Tuple 

Pair = Tuple[int, int]

def encode_chunk(
        ids: List[int], 
        ranks: Dict[Pair, int], 
        merge_out: Dict[Pair, int]
) -> List[int]:
    n = len(ids)
    if n < 2: 
        return list(ids)

    val = list(ids)
    nxt = list(range(1, n)) + [-1]
    prev = [-1] + list(range(0, n-1))
    alive = [True] * n 

    heap: List[tuple] = []

    def try_push(i: int) -> None: 
        if i == -1 or not alive[i]:
            return 

        j = nxt[i]
        if j == -1 or not alive[j]:
            return 

        pair = (val[i], val[j])
        rank = ranks.get(pair)
        if rank is not None: 
            heapq.heappush(heap, (rank, i, pair))

    for i in range(n - 1):
        try_push(i)

    while heap: 
        rank, i, pair = heapq.heappop(heap)

        if not alive[i]:
            continue 

        j = nxt[i]
        if j == -1 or not alive[j]: 
            continue 

        if (val[i], val[j]) != pair:
            continue 


        new_id = merge_out[pair]
        val[i] = new_id 
        k = nxt[j]
        nxt[i] = k 

        if k!=-1:
            prev[k] = i

        alive[j] = False

        try_push(prev[i])
        try_push(i)


    out: List[int] = []
    i = 0 

    while i != -1: 
        out.append(val[i])
        i = nxt[i]

    return out 