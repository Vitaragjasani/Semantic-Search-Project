"""Build bounded, de-duplicated, source-attributed context for the LLM.

Retrieval can return many candidate chunks; this module decides which of
them are actually worth spending LLM tokens on. The context budget is
measured in characters, not LLM tokens -- the same reasoning as Phase 2's
character-based chunking: no extra tokenizer dependency, and roughly 4
characters per token is a conservative enough rule of thumb that a
character budget set comfortably below the model's real context window
avoids truncation in practice. If MAX_CONTEXT_CHARS is set too close to
the model's actual token limit, that margin disappears -- leave headroom.

The first chunk is always included even if it alone exceeds the budget,
so a single misconfigured (too-small) budget can't silently produce a
completely empty context; every chunk after it is still bounded.
"""
from dataclasses import dataclass
from typing import Optional


class ContextBuilderError(Exception):
    pass


@dataclass
class ContextChunk:
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: Optional[int]
    text: str
    retrieval_score: float


@dataclass
class BuiltContext:
    context_text: str
    sources: list[ContextChunk]  # chunks actually included, in presentation order


def build_context(
    ranked_chunk_ids: list[tuple[str, float]],
    chunk_lookup: dict[str, dict],
    max_context_chars: int,
) -> BuiltContext:
    """ranked_chunk_ids: (chunk_id, score) pairs, best first -- typically
    the output of HybridSearch.search_hybrid/search_semantic/search_keyword.
    chunk_lookup: chunk_id -> chunk metadata dict, as persisted by
    src.utils.metadata, used to fetch chunk text and source info.
    """
    if max_context_chars <= 0:
        raise ContextBuilderError("max_context_chars must be positive")

    included: list[ContextChunk] = []
    seen_text: set[str] = set()
    total_chars = 0

    for chunk_id, score in ranked_chunk_ids:
        record = chunk_lookup.get(chunk_id)
        if record is None:
            continue  # stale id, e.g. chunk deleted since the index was last built

        text = record["text"]
        if text in seen_text:
            continue  # exact-duplicate text (e.g. a re-ingested document)

        if included and total_chars + len(text) > max_context_chars:
            break

        seen_text.add(text)
        included.append(
            ContextChunk(
                chunk_id=chunk_id,
                doc_id=record["doc_id"],
                source_filename=record["source_filename"],
                page_number=record.get("page_number"),
                text=text,
                retrieval_score=score,
            )
        )
        total_chars += len(text)

    return BuiltContext(context_text=_format_context(included), sources=included)


def _format_context(chunks: list[ContextChunk]) -> str:
    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        location = chunk.source_filename
        if chunk.page_number is not None:
            location += f", page {chunk.page_number}"
        blocks.append(f"[Source {position}: {location}]\n{chunk.text}")
    return "\n\n".join(blocks)
