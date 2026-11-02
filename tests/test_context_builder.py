import pytest

from src.rag.context_builder import ContextBuilderError, build_context


def _lookup(records: list[dict]) -> dict:
    return {r["chunk_id"]: r for r in records}


def test_build_context_basic_ordering_and_fields():
    records = [
        {"chunk_id": "c1", "doc_id": "d1", "source_filename": "manual.pdf", "page_number": 3, "text": "First chunk."},
        {"chunk_id": "c2", "doc_id": "d1", "source_filename": "manual.pdf", "page_number": 5, "text": "Second chunk."},
    ]
    ranked = [("c1", 0.9), ("c2", 0.5)]

    built = build_context(ranked, _lookup(records), max_context_chars=1000)

    assert [c.chunk_id for c in built.sources] == ["c1", "c2"]
    assert "[Source 1: manual.pdf, page 3]" in built.context_text
    assert "First chunk." in built.context_text
    assert "[Source 2: manual.pdf, page 5]" in built.context_text


def test_build_context_skips_stale_chunk_ids():
    records = [{"chunk_id": "c1", "doc_id": "d1", "source_filename": "a.txt", "page_number": None, "text": "Alive chunk."}]
    ranked = [("c1", 0.9), ("deleted_chunk", 0.8)]

    built = build_context(ranked, _lookup(records), max_context_chars=1000)

    assert [c.chunk_id for c in built.sources] == ["c1"]


def test_build_context_dedupes_identical_text():
    records = [
        {"chunk_id": "c1", "doc_id": "d1", "source_filename": "a.txt", "page_number": None, "text": "Same text."},
        {"chunk_id": "c2", "doc_id": "d2", "source_filename": "b.txt", "page_number": None, "text": "Same text."},
    ]
    ranked = [("c1", 0.9), ("c2", 0.8)]

    built = build_context(ranked, _lookup(records), max_context_chars=1000)

    assert len(built.sources) == 1
    assert built.sources[0].chunk_id == "c1"


def test_build_context_respects_char_budget():
    records = [
        {"chunk_id": "c1", "doc_id": "d1", "source_filename": "a.txt", "page_number": None, "text": "x" * 50},
        {"chunk_id": "c2", "doc_id": "d1", "source_filename": "a.txt", "page_number": None, "text": "y" * 50},
        {"chunk_id": "c3", "doc_id": "d1", "source_filename": "a.txt", "page_number": None, "text": "z" * 50},
    ]
    ranked = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]

    built = build_context(ranked, _lookup(records), max_context_chars=80)

    # Only the first chunk fits cleanly; the second would push past budget.
    assert [c.chunk_id for c in built.sources] == ["c1"]


def test_build_context_always_includes_first_chunk_even_if_oversized():
    records = [{"chunk_id": "c1", "doc_id": "d1", "source_filename": "a.txt", "page_number": None, "text": "x" * 500}]
    ranked = [("c1", 0.9)]

    built = build_context(ranked, _lookup(records), max_context_chars=10)

    assert [c.chunk_id for c in built.sources] == ["c1"]


def test_build_context_empty_input_returns_empty():
    built = build_context([], {}, max_context_chars=1000)
    assert built.sources == []
    assert built.context_text == ""


def test_build_context_rejects_nonpositive_budget():
    with pytest.raises(ContextBuilderError):
        build_context([], {}, max_context_chars=0)
