"""Document/chunk metadata records and their JSON persistence.

Documents are keyed by a generated doc_id (uuid4 hex), not by filename, and
each stored under storage/documents/<doc_id>/. Two uploads of "report.pdf"
therefore get distinct doc_ids and distinct directories -- nothing is ever
silently overwritten by filename collision.
"""
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DOCUMENTS_INDEX_FILENAME = "documents.json"


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    file_type: str
    uploaded_at: str
    num_pages: int
    num_chunks: int
    char_count: int


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    page_number: Optional[int]
    char_count: int
    source_filename: str


def new_doc_id() -> str:
    return uuid.uuid4().hex


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_documents_index(metadata_dir: Path) -> dict:
    path = metadata_dir / DOCUMENTS_INDEX_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_documents_index(metadata_dir: Path, index: dict) -> None:
    path = metadata_dir / DOCUMENTS_INDEX_FILENAME
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _chunks_path(metadata_dir: Path, doc_id: str) -> Path:
    return metadata_dir / f"chunks_{doc_id}.json"


def save_chunks(metadata_dir: Path, doc_id: str, chunks: list[ChunkRecord]) -> None:
    path = _chunks_path(metadata_dir, doc_id)
    path.write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")


def load_chunks(metadata_dir: Path, doc_id: str) -> list[dict]:
    path = _chunks_path(metadata_dir, doc_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_chunks(metadata_dir: Path) -> list[dict]:
    all_chunks = []
    for path in sorted(metadata_dir.glob("chunks_*.json")):
        all_chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return all_chunks


def delete_document(metadata_dir: Path, doc_id: str) -> None:
    index = load_documents_index(metadata_dir)
    index.pop(doc_id, None)
    save_documents_index(metadata_dir, index)
    chunks_path = _chunks_path(metadata_dir, doc_id)
    if chunks_path.exists():
        chunks_path.unlink()
