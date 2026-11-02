import docx
import fitz
import pytest

from src.ingestion.document_loader import (
    CorruptedDocumentError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_document,
)


def test_load_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello world", encoding="utf-8")
    doc = load_document(p)
    assert doc.file_type == "txt"
    assert doc.pages[0].text == "hello world"
    assert doc.pages[0].page_number is None


def test_load_empty_txt_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(p)


def test_load_markdown(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Heading\n\nbody text", encoding="utf-8")
    doc = load_document(p)
    assert doc.file_type == "md"
    assert "Heading" in doc.pages[0].text


def test_load_docx(tmp_path):
    p = tmp_path / "report.docx"
    d = docx.Document()
    d.add_paragraph("First paragraph.")
    d.add_paragraph("Second paragraph.")
    d.save(p)

    doc = load_document(p)
    assert doc.file_type == "docx"
    assert "First paragraph." in doc.pages[0].text
    assert "Second paragraph." in doc.pages[0].text


def test_load_pdf_multi_page(tmp_path):
    p = tmp_path / "sample.pdf"
    pdf = fitz.open()
    for text in ("Page one content", "Page two content"):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(p)
    pdf.close()

    doc = load_document(p)
    assert doc.file_type == "pdf"
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Page one content" in doc.pages[0].text
    assert "Page two content" in doc.pages[1].text


def test_load_image_only_pdf_raises_empty(tmp_path):
    p = tmp_path / "blank.pdf"
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(p)
    pdf.close()

    with pytest.raises(EmptyDocumentError):
        load_document(p)


def test_load_corrupted_pdf_raises(tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"not a real pdf")
    with pytest.raises(CorruptedDocumentError):
        load_document(p)


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(p)
