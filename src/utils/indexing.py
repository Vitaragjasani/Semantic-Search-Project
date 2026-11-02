"""Wires ingestion, embedding, and both retrieval indexes together.

Kept separate from app.py so the Streamlit layer never contains business
logic -- the UI only calls these functions and renders what comes back.
"""
import shutil
from pathlib import Path

import numpy as np

from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.pipeline import ingest_document, validate_upload
from src.retrieval.bm25_store import BM25Store
from src.retrieval.faiss_store import FaissStore
from src.utils.metadata import DocumentRecord, delete_document, load_all_chunks, load_chunks


def ingest_and_index(
    source_path: Path,
    original_filename: str,
    file_size_bytes: int,
    documents_dir: Path,
    metadata_dir: Path,
    chunk_size: int,
    overlap: int,
    max_upload_size_mb: int,
    supported_extensions: set,
    embedding_service: EmbeddingService,
    faiss_store: FaissStore,
    bm25_store: BM25Store,
) -> DocumentRecord:
    """Ingests one new document and updates both indexes incrementally.

    Only this document's own chunks are embedded here -- every previously
    indexed vector is left untouched (see FaissStore.add's docstring).
    BM25 has no incremental add, but rebuilding it costs no embeddings,
    only re-tokenizing already-stored text, so it's rebuilt from every
    persisted chunk each call -- cheap, and always internally consistent.
    """
    validate_upload(original_filename, file_size_bytes, max_upload_size_mb, supported_extensions)

    record = ingest_document(
        source_path, original_filename, documents_dir, metadata_dir, chunk_size, overlap
    )

    new_chunks = load_chunks(metadata_dir, record.doc_id)
    chunk_ids = [c["chunk_id"] for c in new_chunks]
    texts = [c["text"] for c in new_chunks]
    vectors = embedding_service.embed_documents(texts).vectors
    faiss_store.add(vectors, chunk_ids)
    faiss_store.save()

    _rebuild_bm25_from_all_chunks(metadata_dir, bm25_store)
    return record


def delete_and_reindex(
    doc_id: str,
    documents_dir: Path,
    metadata_dir: Path,
    embedding_dimension: int,
    faiss_store: FaissStore,
    bm25_store: BM25Store,
) -> None:
    """Deletes a document's files/metadata, then rebuilds both indexes
    from the chunks that remain.

    The remaining vectors are read back out of the FAISS index itself
    (FaissStore.get_vectors) rather than re-embedded, so deleting one
    document never costs re-running the embedding model over every other
    document still in the corpus.
    """
    delete_document(metadata_dir, doc_id)
    shutil.rmtree(documents_dir / doc_id, ignore_errors=True)

    remaining = load_all_chunks(metadata_dir)
    remaining_ids = [c["chunk_id"] for c in remaining]
    remaining_texts = [c["text"] for c in remaining]

    if remaining_ids:
        vectors = faiss_store.get_vectors(remaining_ids)
    else:
        vectors = np.empty((0, embedding_dimension), dtype=np.float32)
    faiss_store.rebuild(vectors, remaining_ids)
    faiss_store.save()

    bm25_store.build(remaining_ids, remaining_texts)


def full_rebuild_from_scratch(
    metadata_dir: Path,
    embedding_service: EmbeddingService,
    faiss_store: FaissStore,
    bm25_store: BM25Store,
) -> int:
    """Re-embeds every persisted chunk and rebuilds both indexes from
    zero. Returns the number of chunks processed.

    Unlike ingest_and_index/delete_and_reindex above, this deliberately
    recomputes every embedding. It exists only as a manual recovery tool
    for when the index and the stored chunk metadata have drifted out of
    sync (e.g. after manually editing files under storage/) -- not for
    routine use, since routine ingestion/deletion never need to re-embed
    documents that haven't changed.
    """
    chunk_records = load_all_chunks(metadata_dir)
    chunk_ids = [c["chunk_id"] for c in chunk_records]
    texts = [c["text"] for c in chunk_records]

    if chunk_ids:
        vectors = embedding_service.embed_documents(texts).vectors
    else:
        vectors = np.empty((0, embedding_service.expected_dimension), dtype=np.float32)

    faiss_store.rebuild(vectors, chunk_ids)
    faiss_store.save()
    bm25_store.build(chunk_ids, texts)
    return len(chunk_ids)


def build_chunk_lookup(metadata_dir: Path) -> dict:
    return {c["chunk_id"]: c for c in load_all_chunks(metadata_dir)}


def _rebuild_bm25_from_all_chunks(metadata_dir: Path, bm25_store: BM25Store) -> None:
    chunk_records = load_all_chunks(metadata_dir)
    bm25_store.build(
        [c["chunk_id"] for c in chunk_records],
        [c["text"] for c in chunk_records],
    )
