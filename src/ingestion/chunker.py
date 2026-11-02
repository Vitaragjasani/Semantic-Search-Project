"""Split cleaned text into overlapping, stably-identified chunks.

Chunking is character-based rather than token-based: it's deterministic,
has no extra tokenizer dependency, and the embedding model truncates any
chunk that exceeds its own token limit anyway. Each page is chunked
independently (not after concatenating the whole document) so every chunk
keeps an accurate page_number instead of one that only matches its first
page.
"""
import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    page_number: Optional[int]
    char_count: int


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{doc_id}:{chunk_index}".encode("utf-8")).hexdigest()
    return digest[:16]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    step = chunk_size - overlap
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start += step
    return chunks


def chunk_document(
    doc_id: str,
    pages: list[tuple[Optional[int], str]],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for page_number, page_text in pages:
        for piece in chunk_text(page_text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc_id, index),
                    doc_id=doc_id,
                    chunk_index=index,
                    text=piece,
                    page_number=page_number,
                    char_count=len(piece),
                )
            )
            index += 1
    return chunks
