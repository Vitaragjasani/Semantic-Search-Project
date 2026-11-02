"""Extract raw text (and page numbers, where the format has pages) from
PDF, DOCX, TXT, and Markdown files.

PDF is the only format with a native notion of "page" here; DOCX/TXT/MD are
treated as a single unpaged block of text (page_number=None in their
PageText), since python-docx and plain text files don't carry page breaks.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import docx
import fitz  # PyMuPDF


class DocumentLoadError(Exception):
    """Base class for all document extraction failures."""


class UnsupportedFileTypeError(DocumentLoadError):
    pass


class EmptyDocumentError(DocumentLoadError):
    pass


class CorruptedDocumentError(DocumentLoadError):
    pass


@dataclass
class PageText:
    page_number: Optional[int]  # 1-indexed; None when the format has no pages
    text: str


@dataclass
class ExtractedDocument:
    pages: list[PageText]
    file_type: str


def load_document(file_path: Path) -> ExtractedDocument:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix == ".docx":
        return _load_docx(file_path)
    if suffix == ".txt":
        return _load_txt(file_path)
    if suffix == ".md":
        return _load_markdown(file_path)
    raise UnsupportedFileTypeError(f"Unsupported file type: '{suffix}'")


def _load_pdf(file_path: Path) -> ExtractedDocument:
    try:
        pdf = fitz.open(file_path)
    except Exception as exc:
        raise CorruptedDocumentError(
            f"Could not open PDF '{file_path.name}': {exc}"
        ) from exc

    try:
        pages = [
            PageText(page_number=i, text=page.get_text("text"))
            for i, page in enumerate(pdf, start=1)
        ]
    finally:
        pdf.close()

    if not any(p.text.strip() for p in pages):
        raise EmptyDocumentError(
            f"PDF '{file_path.name}' has no extractable text "
            "(it may be a scanned/image-only PDF, which needs OCR not covered here)."
        )
    return ExtractedDocument(pages=pages, file_type="pdf")


def _load_docx(file_path: Path) -> ExtractedDocument:
    try:
        document = docx.Document(file_path)
    except Exception as exc:
        raise CorruptedDocumentError(
            f"Could not open DOCX '{file_path.name}': {exc}"
        ) from exc

    full_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    if not full_text.strip():
        raise EmptyDocumentError(f"DOCX '{file_path.name}' contains no readable text.")
    return ExtractedDocument(pages=[PageText(page_number=None, text=full_text)], file_type="docx")


def _load_txt(file_path: Path) -> ExtractedDocument:
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = file_path.read_text(encoding="latin-1")
        except Exception as exc:
            raise CorruptedDocumentError(
                f"Could not decode TXT '{file_path.name}': {exc}"
            ) from exc
    except OSError as exc:
        raise CorruptedDocumentError(f"Could not read TXT '{file_path.name}': {exc}") from exc

    if not text.strip():
        raise EmptyDocumentError(f"TXT '{file_path.name}' is empty.")
    return ExtractedDocument(pages=[PageText(page_number=None, text=text)], file_type="txt")


def _load_markdown(file_path: Path) -> ExtractedDocument:
    extracted = _load_txt(file_path)
    return ExtractedDocument(pages=extracted.pages, file_type="md")
