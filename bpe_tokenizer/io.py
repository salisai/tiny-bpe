from pathlib import Path 
from typing import Iterator, Iterable 

def stream_files(paths: Iterable[Path]) -> Iterator[str]:
    """stream text from the list of files paths"""
    for path in paths: 
        with open(path, "r", encoding="utf-8") as f: 
            for line in f: 
                yield line 