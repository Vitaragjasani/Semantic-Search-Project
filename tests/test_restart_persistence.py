"""Simulates an actual application restart: a brand-new FaissStore and a
brand-new BM25Store, pointed at the same storage directories a prior
"session" wrote to, with nothing shared in memory between the two. Earlier
phases' tests proved save()/load() round-trip within one test function and
proved rebuild-from-chunk-records works in isolation; this file is the one
that proves the two together survive a real restart, including after a
document is deleted -- which is what Feature K actually asks for.
"""
import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.bm25_store import BM25Store
from src.retrieval.faiss_store import FaissStore
from src.utils.indexing import build_chunk_lookup, delete_and_reindex, ingest_and_index
from src.utils.metadata import load_all_chunks

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384
SUPPORTED_EXTENSIONS = {".txt"}


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService(model_name=MODEL_NAME, expected_dimension=DIMENSION)


@pytest.fixture
def storage_dirs(tmp_path):
    documents_dir = tmp_path / "documents"
    metadata_dir = tmp_path / "metadata"
    index_dir = tmp_path / "indexes"
    documents_dir.mkdir()
    metadata_dir.mkdir()
    index_dir.mkdir()
    return documents_dir, metadata_dir, index_dir


def _write_source(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _new_session_stores(index_dir, metadata_dir):
    """Constructs fresh store objects with no in-memory ties to any prior
    session -- the only thing connecting them to prior data is disk state
    under index_dir/metadata_dir, exactly as a real process restart would
    see it.
    """
    faiss_store = FaissStore(dimension=DIMENSION, index_dir=index_dir)
    loaded = faiss_store.load()
    bm25_store = BM25Store.from_chunk_records(load_all_chunks(metadata_dir))
    return faiss_store, bm25_store, loaded


def test_documents_and_indexes_survive_a_restart(embedding_service, storage_dirs, tmp_path):
    documents_dir, metadata_dir, index_dir = storage_dirs

    # --- "Session 1": ingest two documents, persisting as we go. ---
    session1_faiss = FaissStore(dimension=DIMENSION, index_dir=index_dir)
    session1_bm25 = BM25Store()

    # A third, unrelated filler document keeps "ACME-4471-X" a genuine
    # minority term (1 of 3) -- with only 2 documents it would sit at
    # exactly half the corpus, where BM25's IDF hits zero (see Phase 5).
    docs = [
        ("widgets.txt", "Part number ACME-4471-X is a cooling fan assembly."),
        ("apples.txt", "Apples are rich in fiber and vitamin C for health."),
        ("filler.txt", "Our customer support hours are Monday through Friday."),
    ]
    records = {}
    for name, content in docs:
        source = _write_source(tmp_path, name, content)
        records[name] = ingest_and_index(
            source, name, source.stat().st_size,
            documents_dir, metadata_dir, chunk_size=200, overlap=20,
            max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
            embedding_service=embedding_service, faiss_store=session1_faiss, bm25_store=session1_bm25,
        )
    session1_total_chunks = session1_faiss.size

    # --- "Restart": drop session 1's objects entirely, build fresh ones. ---
    del session1_faiss, session1_bm25
    session2_faiss, session2_bm25, loaded = _new_session_stores(index_dir, metadata_dir)

    assert loaded is True
    assert session2_faiss.size == session1_total_chunks
    assert session2_bm25.size == session1_total_chunks

    # The restored FAISS index must still answer a semantic query correctly.
    query_vector = embedding_service.embed_query("information about fruit nutrition")
    results = session2_faiss.search(query_vector, top_k=1)
    chunk_lookup = build_chunk_lookup(metadata_dir)
    assert chunk_lookup[results[0][0]]["doc_id"] == records["apples.txt"].doc_id

    # The restored BM25 index must still answer an exact-identifier query.
    bm25_results = session2_bm25.search("ACME-4471-X", top_k=1)
    assert chunk_lookup[bm25_results[0][0]]["doc_id"] == records["widgets.txt"].doc_id


def test_deletion_persists_across_a_second_restart(embedding_service, storage_dirs, tmp_path):
    documents_dir, metadata_dir, index_dir = storage_dirs

    session1_faiss = FaissStore(dimension=DIMENSION, index_dir=index_dir)
    session1_bm25 = BM25Store()

    # Two filler documents stay after "apples.txt" is deleted, so the
    # remaining corpus still has 3 chunks and "ACME-4471-X" (in widgets.txt
    # only) stays a minority term rather than landing at BM25's zero-IDF
    # 50% boundary (see Phase 5 notes).
    docs = [
        ("widgets.txt", "Part number ACME-4471-X is a cooling fan assembly."),
        ("apples.txt", "Apples are rich in fiber and vitamin C for health."),
        ("filler.txt", "Our customer support hours are Monday through Friday."),
        ("filler2.txt", "The quarterly report covers revenue and expenses."),
    ]
    records = {}
    for name, content in docs:
        source = _write_source(tmp_path, name, content)
        records[name] = ingest_and_index(
            source, name, source.stat().st_size,
            documents_dir, metadata_dir, chunk_size=200, overlap=20,
            max_upload_size_mb=10, supported_extensions=SUPPORTED_EXTENSIONS,
            embedding_service=embedding_service, faiss_store=session1_faiss, bm25_store=session1_bm25,
        )
    del session1_faiss, session1_bm25

    # --- Restart 1: a new session deletes "apples.txt". ---
    session2_faiss, session2_bm25, _ = _new_session_stores(index_dir, metadata_dir)
    delete_and_reindex(
        records["apples.txt"].doc_id, documents_dir, metadata_dir, DIMENSION,
        session2_faiss, session2_bm25,
    )
    del session2_faiss, session2_bm25

    # --- Restart 2: an independent third session must see the deletion. ---
    session3_faiss, session3_bm25, loaded = _new_session_stores(index_dir, metadata_dir)

    assert loaded is True
    chunk_lookup = build_chunk_lookup(metadata_dir)
    assert records["apples.txt"].doc_id not in {rec["doc_id"] for rec in chunk_lookup.values()}
    assert not (documents_dir / records["apples.txt"].doc_id).exists()

    # The surviving documents must still be fully searchable after the
    # deletion carried across two separate restarts.
    bm25_results = session3_bm25.search("ACME-4471-X", top_k=1)
    assert chunk_lookup[bm25_results[0][0]]["doc_id"] == records["widgets.txt"].doc_id

    query_vector = embedding_service.embed_query("customer support availability")
    faiss_results = session3_faiss.search(query_vector, top_k=1)
    assert chunk_lookup[faiss_results[0][0]]["doc_id"] == records["filler.txt"].doc_id


def test_restart_with_no_prior_data_starts_clean(storage_dirs):
    _, metadata_dir, index_dir = storage_dirs
    faiss_store, bm25_store, loaded = _new_session_stores(index_dir, metadata_dir)

    assert loaded is False
    assert faiss_store.size == 0
    assert bm25_store.size == 0
