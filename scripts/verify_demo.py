"""One-off verification that the demo questions in README.md actually
retrieve what they claim to, using the real ingestion -> embedding ->
FAISS -> BM25 -> hybrid pipeline against sample_docs/. Not a pytest test
(it writes into storage/ and is meant to be read, not asserted in CI) --
run manually after `python scripts/generate_sample_docs.py`:

    python scripts/verify_demo.py
"""
import shutil
from pathlib import Path

from config import settings
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.bm25_store import BM25Store
from src.retrieval.faiss_store import FaissStore
from src.retrieval.hybrid_search import HybridSearch
from src.utils.indexing import build_chunk_lookup, ingest_and_index

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
DEMO_QUESTIONS = [
    "How long do refunds take to process?",
    "What is part number ACME-4471-X?",
    "How many vacation days do new employees get?",
    "How long does IT equipment take to arrive before a start date?",
    # This one is deliberately off-topic. FAISS's IndexFlatIP.search always
    # returns its k nearest vectors, however weak the real similarity is --
    # it never returns "nothing," so this will still show some chunk below.
    # The "no relevant evidence" behavior for a genuinely off-topic question
    # isn't a retrieval-layer short-circuit here; it relies on the LLM
    # itself following system-prompt rule #4 ("if CONTEXT does not contain
    # enough information, say so") once it actually reads the weak context
    # below. Verifying that requires a real HUGGINGFACE_API_TOKEN and an
    # actual Q&A call -- this script only verifies retrieval, not that rule.
    "What is the company's policy on space travel?",
]


def main():
    # Use a throwaway copy of storage/ so this script never pollutes the
    # real app's persisted data.
    demo_root = Path(__file__).resolve().parent.parent / ".demo_verify_tmp"
    if demo_root.exists():
        shutil.rmtree(demo_root)
    documents_dir = demo_root / "documents"
    metadata_dir = demo_root / "metadata"
    index_dir = demo_root / "indexes"
    for d in (documents_dir, metadata_dir, index_dir):
        d.mkdir(parents=True)

    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL_NAME, expected_dimension=settings.EMBEDDING_DIMENSION
    )
    faiss_store = FaissStore(dimension=settings.EMBEDDING_DIMENSION, index_dir=index_dir)
    bm25_store = BM25Store()

    for path in sorted(SAMPLE_DOCS_DIR.iterdir()):
        ingest_and_index(
            path, path.name, path.stat().st_size,
            documents_dir, metadata_dir,
            settings.CHUNK_SIZE, settings.CHUNK_OVERLAP,
            settings.MAX_UPLOAD_SIZE_MB, settings.SUPPORTED_EXTENSIONS,
            embedding_service, faiss_store, bm25_store,
        )

    hybrid = HybridSearch(embedding_service, faiss_store, bm25_store)
    chunk_lookup = build_chunk_lookup(metadata_dir)

    print(
        "(scores below are RRF-fused rank scores, not raw cosine similarity --"
        " see src/retrieval/hybrid_search.py for why the two aren't comparable)"
    )
    for question in DEMO_QUESTIONS:
        print(f"\nQ: {question}")
        results = hybrid.search_hybrid(
            question, top_k=3,
            semantic_candidates=settings.SEMANTIC_CANDIDATES,
            keyword_candidates=settings.KEYWORD_CANDIDATES,
        )
        if not results:
            print("  -> no relevant chunks found")
            continue
        for chunk_id, score in results:
            record = chunk_lookup[chunk_id]
            location = record["source_filename"]
            if record.get("page_number") is not None:
                location += f", page {record['page_number']}"
            print(f"  [{score:.4f}] {location}: {record['text'][:80]!r}")

    shutil.rmtree(demo_root)


if __name__ == "__main__":
    main()
