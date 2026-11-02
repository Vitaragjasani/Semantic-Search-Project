"""Keyword retrieval over chunk text using BM25Okapi.

BM25 scores a chunk by exact-token statistics: how often each query token
appears in it (term frequency), weighted by how rare that token is across
the whole corpus (inverse document frequency), normalized for chunk
length. It has no notion of meaning or paraphrase -- "car" and
"automobile" are two unrelated tokens to BM25 -- but that's exactly why it
complements semantic search: an exact product code, error ID, part number,
or proper name that an embedding model might blur into a nearby-but-wrong
match is still retrieved by BM25 with high precision, because matching
only requires the literal characters to line up.

The BM25Okapi object itself is never persisted. rank_bm25 objects aren't a
stable, portable serialization format, and the whole index can be rebuilt
deterministically, in well under a second, from the tokenized chunk text
that Phase 2 already persists as chunk metadata. So instead of trying to
restore a saved index after a restart, `BM25Store.from_chunk_records`
re-tokenizes the stored chunks and rebuilds BM25Okapi from scratch every
time -- a reliable rebuild, rather than an assumption that a serialized
index is still valid.
"""
import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization. Chunks and queries must
    go through this exact same function -- BM25 only matches tokens that
    are literally equal after tokenization.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25StoreError(Exception):
    pass


class BM25Store:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise BM25StoreError(f"Got {len(chunk_ids)} chunk_ids but {len(texts)} texts.")

        self._chunk_ids = list(chunk_ids)
        if not texts:
            self._bm25 = None
            return

        tokenized_corpus = [tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @classmethod
    def from_chunk_records(cls, chunk_records: list[dict]) -> "BM25Store":
        store = cls()
        chunk_ids = [record["chunk_id"] for record in chunk_records]
        texts = [record["text"] for record in chunk_records]
        store.build(chunk_ids, texts)
        return store

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise BM25StoreError("top_k must be positive")
        if self._bm25 is None or self.size == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        # A zero score means the chunk shares no token with the query --
        # excluded rather than returned as a misleading "match".
        matches = [(chunk_id, float(score)) for chunk_id, score in ranked if score > 0]
        return matches[:top_k]
