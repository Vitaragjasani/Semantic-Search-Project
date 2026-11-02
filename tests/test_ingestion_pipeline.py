import pytest

from src.ingestion.pipeline import IngestionError, ingest_document, validate_upload


def test_ingest_txt_document(tmp_path):
    documents_dir = tmp_path / "documents"
    metadata_dir = tmp_path / "metadata"
    documents_dir.mkdir()
    metadata_dir.mkdir()

    source = tmp_path / "upload.txt"
    source.write_text(
        "This is a reasonably long piece of text used for chunking tests. " * 5,
        encoding="utf-8",
    )

    record = ingest_document(
        source, "upload.txt", documents_dir, metadata_dir, chunk_size=100, overlap=20
    )

    assert record.num_chunks > 0
    assert (documents_dir / record.doc_id / "upload.txt").exists()
    assert (metadata_dir / f"chunks_{record.doc_id}.json").exists()


def test_ingest_same_filename_twice_does_not_overwrite(tmp_path):
    documents_dir = tmp_path / "documents"
    metadata_dir = tmp_path / "metadata"
    documents_dir.mkdir()
    metadata_dir.mkdir()

    source = tmp_path / "dup.txt"
    source.write_text("Some content for the duplicate filename test.", encoding="utf-8")

    record_a = ingest_document(
        source, "dup.txt", documents_dir, metadata_dir, chunk_size=50, overlap=10
    )
    record_b = ingest_document(
        source, "dup.txt", documents_dir, metadata_dir, chunk_size=50, overlap=10
    )

    assert record_a.doc_id != record_b.doc_id
    assert (documents_dir / record_a.doc_id / "dup.txt").exists()
    assert (documents_dir / record_b.doc_id / "dup.txt").exists()


def test_validate_upload_rejects_unsupported_extension():
    with pytest.raises(IngestionError):
        validate_upload("data.csv", 100, max_size_mb=10, supported_extensions={".pdf", ".txt"})


def test_validate_upload_rejects_oversized_file():
    with pytest.raises(IngestionError):
        validate_upload(
            "big.txt", 20 * 1024 * 1024, max_size_mb=10, supported_extensions={".txt"}
        )


def test_validate_upload_rejects_empty_file():
    with pytest.raises(IngestionError):
        validate_upload("empty.txt", 0, max_size_mb=10, supported_extensions={".txt"})


def test_validate_upload_accepts_valid_file():
    validate_upload("notes.md", 1024, max_size_mb=10, supported_extensions={".md"})
