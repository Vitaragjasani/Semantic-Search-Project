import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.bm25_store import BM25Store
from src.retrieval.faiss_store import FaissStore
from src.retrieval.hybrid_search import HybridSearch, HybridSearchError, reciprocal_rank_fusion

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384


# --- Pure RRF math, no model needed -----------------------------------

def test_rrf_prefers_chunk_ranked_in_both_over_chunk_ranked_in_one():
    semantic = ["a", "b", "c"]
    keyword = ["b", "d", "e"]
    fused = reciprocal_rank_fusion([semantic, keyword])
    fused_ids = [chunk_id for chunk_id, _ in fused]
    # "b" is #2 in semantic and #1 in keyword -- present in both lists.
    # "a" is #1 in semantic only. "b" should outrank "a" since it
    # accumulates score from two lists instead of one.
    assert fused_ids.index("b") < fused_ids.index("a")


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_preserves_relative_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]


def test_rrf_missing_from_one_list_still_included():
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    fused_ids = {chunk_id for chunk_id, _ in fused}
    assert fused_ids == {"a", "b", "c"}


# --- Integration: real embedding model + real FAISS + real BM25 -------

@pytest.fixture(scope="module")
def hybrid_search(tmp_path_factory):
    embedding_service = EmbeddingService(model_name=MODEL_NAME, expected_dimension=DIMENSION)

    chunk_ids = ["c1", "c2", "c3", "c4"]
    texts = [
        "Apples are a good source of dietary fiber and vitamin C.",
        "Part number ACME-4471-X is the replacement cooling fan assembly.",
        "The stock market closed lower on Tuesday amid interest rate concerns.",
        "Our customer support hours are Monday through Friday, 9am to 5pm.",
    ]

    index_dir = tmp_path_factory.mktemp("faiss_index")
    faiss_store = FaissStore(dimension=DIMENSION, index_dir=index_dir)
    vectors = embedding_service.embed_documents(texts).vectors
    faiss_store.add(vectors, chunk_ids)

    bm25_store = BM25Store()
    bm25_store.build(chunk_ids, texts)

    return HybridSearch(embedding_service, faiss_store, bm25_store)


def test_semantic_search_finds_paraphrased_match(hybrid_search):
    results = hybrid_search.search_semantic("health benefits of eating fruit", top_k=1)
    assert results[0][0] == "c1"


def test_keyword_search_finds_exact_identifier(hybrid_search):
    results = hybrid_search.search_keyword("ACME-4471-X", top_k=1)
    assert results[0][0] == "c2"


def test_keyword_search_misses_pure_paraphrase(hybrid_search):
    # BM25 has no notion of "produce" meaning "apples" -- it should not
    # surface c1 for a query sharing no literal tokens with it. (No
    # stopword removal is implemented, so the query is deliberately
    # chosen to avoid even an incidental shared word like "of" or "a".)
    results = hybrid_search.search_keyword("health benefits for eating produce", top_k=4)
    returned_ids = {chunk_id for chunk_id, _ in results}
    assert "c1" not in returned_ids


def test_hybrid_search_surfaces_exact_identifier_even_with_low_semantic_similarity(hybrid_search):
    # A query that's almost entirely the exact part number: keyword search
    # nails it, semantic search alone might not rank it first. Hybrid
    # should still surface it in the final fused top_k.
    results = hybrid_search.search_hybrid("ACME-4471-X", top_k=2)
    returned_ids = {chunk_id for chunk_id, _ in results}
    assert "c2" in returned_ids


def test_search_hybrid_rejects_nonpositive_top_k(hybrid_search):
    with pytest.raises(HybridSearchError):
        hybrid_search.search_hybrid("anything", top_k=0)


def test_search_hybrid_respects_top_k(hybrid_search):
    results = hybrid_search.search_hybrid("apples fan support stocks", top_k=2)
    assert len(results) <= 2
