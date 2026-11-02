"""Orchestrates ingestion: validate -> load -> clean -> chunk -> persist.

This module isn't in the original spec'd file list but is added here as the
single entry point the Streamlit UI (Phase 8) and tests call, instead of the
UI re-assembling loader/cleaner/chunker/metadata calls itself. It keeps that
wiring in one tested place rather than duplicated in UI code.
"""
import shutil
from pathlib import Path

from src.ingestion.chunker import chunk_document
from src.ingestion.document_loader import load_document
from src.ingestion.text_cleaner import clean_text
from src.utils.metadata import (
    ChunkRecord,
    DocumentRecord,
    load_documents_index,
    new_doc_id,
    save_chunks,
    save_documents_index,
    utc_now_iso,
)


class IngestionError(Exception):
    """Raised for validation/ingestion failures that aren't extraction errors."""


def validate_upload(
    filename: str,
    size_bytes: int,
    max_size_mb: int,
    supported_extensions: set[str],
) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in supported_extensions:
        raise IngestionError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(supported_extensions)}"
        )
    if size_bytes == 0:
        raise IngestionError(f"File '{filename}' is empty.")
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise IngestionError(
            f"File '{filename}' is {size_bytes / 1_048_576:.1f} MB, "
            f"exceeds the {max_size_mb} MB limit."
        )


def ingest_document(
    source_path: Path,
    original_filename: str,
    documents_dir: Path,
    metadata_dir: Path,
    chunk_size: int,
    overlap: int,
) -> DocumentRecord:
    extracted = load_document(source_path)

    cleaned_pages = [(p.page_number, clean_text(p.text)) for p in extracted.pages]
    cleaned_pages = [(page_number, text) for page_number, text in cleaned_pages if text]
    if not cleaned_pages:
        raise IngestionError(f"'{original_filename}' has no usable text after cleaning.")

    doc_id = new_doc_id()
    dest_dir = documents_dir / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_dir / original_filename)

    chunks = chunk_document(doc_id, cleaned_pages, chunk_size, overlap)
    chunk_records = [
        ChunkRecord(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            chunk_index=c.chunk_index,
            text=c.text,
            page_number=c.page_number,
            char_count=c.char_count,
            source_filename=original_filename,
        )
        for c in chunks
    ]
    save_chunks(metadata_dir, doc_id, chunk_records)

    record = DocumentRecord(
        doc_id=doc_id,
        filename=original_filename,
        file_type=extracted.file_type,
        uploaded_at=utc_now_iso(),
        num_pages=len(extracted.pages),
        num_chunks=len(chunk_records),
        char_count=sum(c.char_count for c in chunk_records),
    )
    index = load_documents_index(metadata_dir)
    index[doc_id] = record.__dict__
    save_documents_index(metadata_dir, index)
    return record
