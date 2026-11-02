"""Combine FAISS (semantic) and BM25 (keyword) retrieval via Reciprocal
Rank Fusion.

FAISS similarity scores (cosine, bounded in [-1, 1], usually clustered
near 1 for relevant hits) and BM25 scores (unbounded term-frequency sums
that can range from near 0 to double digits depending on corpus and
query) live on two incompatible scales. Averaging or adding them directly
would let whichever method happens to produce larger raw numbers dominate
the fused ranking by scale alone, not by actual relevance.

Reciprocal Rank Fusion (RRF) sidesteps the scale problem by discarding raw
scores entirely and working only with each list's *rank positions*:

    rrf_score(chunk) = sum over lists containing chunk of 1 / (k + rank)

where rank is 1-indexed and k is a small damping constant (60 is the
commonly cited default from the original RRF paper). A chunk ranked near
the top of both the semantic and keyword lists outscores a chunk that's
#1 in only one list -- which is the behavior "hybrid" retrieval is
supposed to produce, and it falls out of rank-based fusion without ever
needing the two methods' scores to be comparable.
"""
from dataclasses import dataclass

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.bm25_store import BM25Store
from src.retrieval.faiss_store import FaissStore

RRF_K_DEFAULT = 60


class HybridSearchError(Exception):
    pass


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K_DEFAULT,
) -> list[tuple[str, float]]:
    """Each entry in ranked_lists is a list of chunk_ids, already sorted
    best-first by one retrieval method. Returns (chunk_id, rrf_score)
    pairs sorted descending by fused score. A chunk_id absent from a given
    list simply contributes nothing from that list -- it is not penalized
    beyond not receiving that list's rank term.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for position, chunk_id in enumerate(ranked_list):
            rank = position + 1  # RRF is defined over 1-indexed ranks
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridSearch:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        faiss_store: FaissStore,
        bm25_store: BM25Store,
    ):
        self._embedding_service = embedding_service
        self._faiss_store = faiss_store
        self._bm25_store = bm25_store

    def search_semantic(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vector = self._embedding_service.embed_query(query)
        return self._faiss_store.search(query_vector, top_k)

    def search_keyword(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self._bm25_store.search(query, top_k)

    def search_hybrid(
        self,
        query: str,
        top_k: int,
        semantic_candidates: int = 20,
        keyword_candidates: int = 20,
        rrf_k: int = RRF_K_DEFAULT,
    ) -> list[tuple[str, float]]:
        """Retrieves semantic_candidates from FAISS and keyword_candidates
        from BM25 independently, fuses the two ranked lists with RRF, and
        returns the top_k fused results. The candidate counts are kept
        separate from the final top_k so each method can be given a wider
        net to pull from than what's ultimately returned.
        """
        if top_k <= 0:
            raise HybridSearchError("top_k must be positive")

        semantic_results = self.search_semantic(query, semantic_candidates)
        keyword_results = self.search_keyword(query, keyword_candidates)

        semantic_ids = [chunk_id for chunk_id, _ in semantic_results]
        keyword_ids = [chunk_id for chunk_id, _ in keyword_results]

        fused = reciprocal_rank_fusion([semantic_ids, keyword_ids], k=rrf_k)
        return fused[:top_k]
