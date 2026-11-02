import numpy as np
import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.bm25_store import BM25Store
from src.retrieval.faiss_store import FaissStore
from src.utils.indexing import (
    build_chunk_lookup,
    delete_and_reindex,
    full_rebuild_from_scratch,
    ingest_and_index,
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384
SUPPORTED_EXTENSIONS = {".txt"}


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService(model_name=MODEL_NAME, expected_dimension=DIMENSION)


@pytest.fixture
def workspace(tmp_path):
    documents_dir = tmp_path / "documents"
    metadata_dir = tmp_path / "metadata"
    index_dir = tmp_path / "indexes"
    documents_dir.mkdir()
    metadata_dir.mkdir()
    index_dir.mkdir()
    return {"documents_dir": documents_dir, "metadata_dir": metadata_dir, "index_dir": index_dir}


def _write_source(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_ingest_and_index_makes_document_searchable(workspace, embedding_service, tmp_path):
    faiss_store = FaissStore(dimension=DIMENSION, index_dir=workspace["index_dir"])
    bm25_store = BM25Store()

    # BM25's IDF is zero/negative once a term appears in half or more of
    # the corpus (see Phase 5 notes), so two unrelated filler documents are
    # ingested alongside the target one to keep "ACME-4471-X" a genuine
    # minority term and get a positive, rankable score.
    for name, content in [
        ("filler_one.txt", "Our customer support hours are Monday through Friday."),
        ("filler_two.txt", "The quarterly report covers revenue and expenses."),
    ]:
        filler_source = _write_source(tmp_path, name, content)
        ingest_and_index(
            filler_source, name, filler_source.stat().st_size,
            workspace["documents_dir"], workspace["metadata_dir"],
            chunk_size=200, overlap=20, max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
            embedding_service=embedding_service, faiss_store=faiss_store, bm25_store=bm25_store,
        )

    source = _write_source(tmp_path, "widgets.txt", "Part number ACME-4471-X is a cooling fan assembly.")
    record = ingest_and_index(
        source, "widgets.txt", source.stat().st_size,
        workspace["documents_dir"], workspace["metadata_dir"],
        chunk_size=200, overlap=20,
        max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
        embedding_service=embedding_service, faiss_store=faiss_store, bm25_store=bm25_store,
    )

    assert record.num_chunks > 0
    assert faiss_store.size == bm25_store.size

    results = bm25_store.search("ACME-4471-X", top_k=1)
    assert results[0][0] in {c for c in build_chunk_lookup(workspace["metadata_dir"])}


def test_ingest_and_index_is_incremental_across_two_documents(workspace, embedding_service, tmp_path):
    faiss_store = FaissStore(dimension=DIMENSION, index_dir=workspace["index_dir"])
    bm25_store = BM25Store()

    source_a = _write_source(tmp_path, "a.txt", "The first document talks about apples and fruit.")
    source_b = _write_source(tmp_path, "b.txt", "The second document talks about cars and engines.")

    record_a = ingest_and_index(
        source_a, "a.txt", source_a.stat().st_size,
        workspace["documents_dir"], workspace["metadata_dir"],
        chunk_size=200, overlap=20, max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
        embedding_service=embedding_service, faiss_store=faiss_store, bm25_store=bm25_store,
    )
    size_after_first = faiss_store.size

    record_b = ingest_and_index(
        source_b, "b.txt", source_b.stat().st_size,
        workspace["documents_dir"], workspace["metadata_dir"],
        chunk_size=200, overlap=20, max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
        embedding_service=embedding_service, faiss_store=faiss_store, bm25_store=bm25_store,
    )

    assert faiss_store.size == size_after_first + record_b.num_chunks
    # The first document's vectors are still retrievable without having
    # been touched by the second ingestion.
    lookup = build_chunk_lookup(workspace["metadata_dir"])
    first_doc_chunk_id = next(cid for cid, rec in lookup.items() if rec["doc_id"] == record_a.doc_id)
    reconstructed = faiss_store.get_vectors([first_doc_chunk_id])
    assert reconstructed.shape == (1, DIMENSION)


def test_delete_and_reindex_removes_only_the_deleted_document(workspace, embedding_service, tmp_path):
    faiss_store = FaissStore(dimension=DIMENSION, index_dir=workspace["index_dir"])
    bm25_store = BM25Store()

    # Two filler documents stay in the corpus after "a" is deleted, so the
    # remaining corpus has 3 chunks and "ACME-4471-X" (in b only) stays a
    # genuine minority term rather than hitting BM25's zero-IDF boundary.
    docs = [
        ("a.txt", "Apples are rich in fiber and vitamin C for health."),
        ("b.txt", "Part number ACME-4471-X is a cooling fan assembly."),
        ("c.txt", "Our customer support hours are Monday through Friday."),
        ("d.txt", "The quarterly report covers revenue and expenses."),
    ]
    records = {}
    for name, content in docs:
        source = _write_source(tmp_path, name, content)
        records[name] = ingest_and_index(
            source, name, source.stat().st_size,
            workspace["documents_dir"], workspace["metadata_dir"],
            chunk_size=200, overlap=20, max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
            embedding_service=embedding_service, faiss_store=faiss_store, bm25_store=bm25_store,
        )

    record_a = records["a.txt"]
    total_chunks = sum(r.num_chunks for name, r in records.items() if name != "a.txt")

    delete_and_reindex(
        record_a.doc_id,
        workspace["documents_dir"], workspace["metadata_dir"], DIMENSION,
        faiss_store, bm25_store,
    )

    assert faiss_store.size == total_chunks
    assert bm25_store.size == total_chunks
    assert not (workspace["documents_dir"] / record_a.doc_id).exists()

    lookup = build_chunk_lookup(workspace["metadata_dir"])
    assert all(rec["doc_id"] != record_a.doc_id for rec in lookup.values())

    # Document b's content must still be findable after a's deletion.
    results = bm25_store.search("ACME-4471-X", top_k=1)
    assert len(results) == 1


def test_full_rebuild_from_scratch_reprocesses_all_chunks(workspace, embedding_service, tmp_path):
    faiss_store = FaissStore(dimension=DIMENSION, index_dir=workspace["index_dir"])
    bm25_store = BM25Store()

    source = _write_source(tmp_path, "a.txt", "Some reasonably long piece of sample text content here.")
    record = ingest_and_index(
        source, "a.txt", source.stat().st_size,
        workspace["documents_dir"], workspace["metadata_dir"],
        chunk_size=200, overlap=20, max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
        embedding_service=embedding_service, faiss_store=faiss_store, bm25_store=bm25_store,
    )

    # Simulate index corruption/drift by wiping the in-memory index.
    faiss_store.rebuild(np.empty((0, DIMENSION), dtype=np.float32), [])
    bm25_store.build([], [])

    processed = full_rebuild_from_scratch(
        workspace["metadata_dir"], embedding_service, faiss_store, bm25_store
    )

    assert processed == record.num_chunks
    assert faiss_store.size == record.num_chunks
    assert bm25_store.size == record.num_chunks


def test_build_chunk_lookup_reflects_persisted_chunks(workspace, embedding_service, tmp_path):
    faiss_store = FaissStore(dimension=DIMENSION, index_dir=workspace["index_dir"])
    bm25_store = BM25Store()

    source = _write_source(tmp_path, "a.txt", "Lookup table content for the chunk lookup test.")
    record = ingest_and_index(
        source, "a.txt", source.stat().st_size,
        workspace["documents_dir"], workspace["metadata_dir"],
        chunk_size=200, overlap=20, max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
        embedding_service=embedding_service, faiss_store=faiss_store, bm25_store=bm25_store,
    )

    lookup = build_chunk_lookup(workspace["metadata_dir"])
    assert len(lookup) == record.num_chunks
    assert all(rec["doc_id"] == record.doc_id for rec in lookup.values())
