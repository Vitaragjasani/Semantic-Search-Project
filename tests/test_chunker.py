import pytest

from src.ingestion.chunker import chunk_document, chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_chunk_text_shorter_than_chunk_size_returns_single_chunk():
    assert chunk_text("short text", chunk_size=100, overlap=10) == ["short text"]


def test_chunk_text_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_text_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0, overlap=0)


def test_chunk_document_preserves_page_numbers():
    pages = [(1, "a" * 50), (2, "b" * 50)]
    chunks = chunk_document("doc1", pages, chunk_size=20, overlap=5)
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
    assert all(len(c.chunk_id) == 16 for c in chunks)


def test_chunk_document_chunk_ids_are_stable_across_runs():
    pages = [(None, "x" * 30)]
    chunks_a = chunk_document("doc1", pages, chunk_size=10, overlap=2)
    chunks_b = chunk_document("doc1", pages, chunk_size=10, overlap=2)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_chunk_document_indexes_are_sequential_across_pages():
    pages = [(1, "a" * 15), (2, "b" * 15)]
    chunks = chunk_document("doc1", pages, chunk_size=10, overlap=2)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
