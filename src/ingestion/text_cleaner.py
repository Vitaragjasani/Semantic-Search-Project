"""Normalize raw extracted text before chunking.

Deliberately conservative: it collapses whitespace noise that PDF/DOCX
extraction introduces, but leaves punctuation, casing, and line structure
(headings, list markers, paragraph breaks) untouched, since those carry
meaning that both BM25 and the LLM rely on.
"""
import re

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = _CONTROL_CHARS_RE.sub("", text)
    # Re-join words that PDF extraction split across a line wrap, e.g.
    # "informa-\ntion" -> "information".
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _INLINE_WHITESPACE_RE.sub(" ", text)
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()
