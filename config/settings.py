"""Central configuration loaded from environment variables / .env.

Chunk size and overlap matter because:
- Larger chunks give the LLM more surrounding context per retrieved hit, but
  dilute embedding specificity (a 2000-char chunk's vector represents an
  "average" of many ideas, so it matches fewer queries precisely) and cost
  more tokens once sent to the LLM.
- Smaller chunks embed more precisely and are cheaper per hit, but a single
  chunk may lack enough context to answer on its own, forcing retrieval of
  more chunks (higher top-K) to compensate.
- Overlap prevents a sentence that answers the query from being split across
  a chunk boundary and losing meaning in both halves. Overlap that's too
  large wastes storage and retrieval slots on duplicated text.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STORAGE_DIR = PROJECT_ROOT / "storage"
DOCUMENTS_DIR = STORAGE_DIR / "documents"
INDEXES_DIR = STORAGE_DIR / "indexes"
METADATA_DIR = STORAGE_DIR / "metadata"

for _dir in (DOCUMENTS_DIR, INDEXES_DIR, METADATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# EMBEDDING_DIMENSION must match the actual output size of EMBEDDING_MODEL_NAME.
# EmbeddingService checks this at load time and raises rather than silently
# using whatever dimension the model happens to produce, because a silent
# mismatch here would only surface later as confusing FAISS errors.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Hybrid retrieval / context budget
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
SEMANTIC_CANDIDATES = int(os.getenv("SEMANTIC_CANDIDATES", "20"))
KEYWORD_CANDIDATES = int(os.getenv("KEYWORD_CANDIDATES", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

# OpenRouter (OpenAI-compatible chat completions router over many models).
# Free-model availability/names change over time -- check
# https://openrouter.ai/models?max_price=0 before relying on a ":free"
# model staying served.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "512"))
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30"))
