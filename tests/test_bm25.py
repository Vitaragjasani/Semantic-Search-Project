import pytest

from src.retrieval.bm25_store import BM25Store, BM25StoreError, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Order #ACME-4471, please!") == ["order", "acme", "4471", "please"]


def test_build_rejects_mismatched_lengths():
    store = BM25Store()
    with pytest.raises(BM25StoreError):
        store.build(["a", "b"], ["only one text"])


def test_search_on_empty_corpus_returns_empty_list():
    store = BM25Store()
    store.build([], [])
    assert store.search("anything", top_k=5) == []


def test_search_with_empty_query_returns_empty_list():
    store = BM25Store()
    store.build(["a"], ["some chunk text"])
    assert store.search("   ", top_k=5) == []


def test_search_rejects_nonpositive_top_k():
    store = BM25Store()
    store.build(["a"], ["some chunk text"])
    with pytest.raises(BM25StoreError):
        store.search("text", top_k=0)


def test_search_finds_exact_identifier_match():
    store = BM25Store()
    chunk_ids = ["c1", "c2", "c3"]
    texts = [
        "The refund policy covers returns within thirty days of purchase.",
        "Part number ACME-4471-X is the replacement cooling fan assembly.",
        "Our support hours are Monday through Friday, 9am to 5pm.",
    ]
    store.build(chunk_ids, texts)

    results = store.search("ACME-4471-X", top_k=3)

    assert results[0][0] == "c2"
    returned_ids = {chunk_id for chunk_id, _ in results}
    assert "c1" not in returned_ids
    assert "c3" not in returned_ids


def test_search_respects_top_k():
    # Classic BM25 IDF (log((N - n + 0.5) / (n + 0.5))) hits zero or goes
    # negative once a term appears in half or more of the corpus, so the
    # term under test must stay a minority to get a positive, rankable
    # score -- here "widget" appears in only 2 of 5 chunks.
    store = BM25Store()
    chunk_ids = [f"c{i}" for i in range(5)]
    texts = [
        "widget assembly instructions",
        "widget warranty terms",
        "completely unrelated topic one",
        "completely unrelated topic two",
        "completely unrelated topic three",
    ]
    store.build(chunk_ids, texts)

    results = store.search("widget", top_k=2)
    assert len(results) == 2


def test_search_excludes_zero_score_chunks():
    # "apples" must appear in fewer than half the docs for its IDF to be
    # positive rather than exactly zero -- use 3 docs, 1 match.
    store = BM25Store()
    chunk_ids = ["c1", "c2", "c3"]
    texts = [
        "apples and oranges",
        "completely unrelated terminology here",
        "another unrelated passage about something else",
    ]
    store.build(chunk_ids, texts)

    results = store.search("apples", top_k=5)
    returned_ids = {chunk_id for chunk_id, _ in results}
    assert returned_ids == {"c1"}


def test_from_chunk_records_builds_searchable_index():
    records = [
        {"chunk_id": "c1", "text": "invoice number INV-9910 is overdue"},
        {"chunk_id": "c2", "text": "general shipping information for all orders"},
        {"chunk_id": "c3", "text": "return policy covers thirty days"},
    ]
    store = BM25Store.from_chunk_records(records)

    results = store.search("INV-9910", top_k=1)
    assert results[0][0] == "c1"


def test_rebuild_after_document_deletion_excludes_removed_chunks():
    records = [
        {"chunk_id": "c1", "text": "widget specifications and dimensions"},
        {"chunk_id": "c2", "text": "gadget specifications and warranty"},
        {"chunk_id": "c3", "text": "completely different topic about weather"},
        {"chunk_id": "c4", "text": "another filler document about shipping"},
    ]
    store = BM25Store.from_chunk_records(records)
    assert store.size == 4

    # Delete c2 (doesn't contain "widget") and rebuild from what's left.
    remaining_records = [r for r in records if r["chunk_id"] != "c2"]
    rebuilt_store = BM25Store.from_chunk_records(remaining_records)

    assert rebuilt_store.size == 3
    results = rebuilt_store.search("widget", top_k=5)
    returned_ids = {chunk_id for chunk_id, _ in results}
    assert returned_ids == {"c1"}
