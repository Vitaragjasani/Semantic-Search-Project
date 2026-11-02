"""Streamlit UI. Deliberately thin: every widget here calls into
src/ (ingestion, embeddings, retrieval, rag) and renders the result --
no business logic lives in this file.
"""
import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.document_loader import DocumentLoadError
from src.ingestion.pipeline import IngestionError
from src.llm.openrouter_client import OpenRouterClientError, OpenRouterLLMClient
from src.rag.rag_pipeline import RagPipeline
from src.retrieval.bm25_store import BM25Store
from src.retrieval.faiss_store import FaissStore
from src.retrieval.hybrid_search import HybridSearch
from src.utils.indexing import build_chunk_lookup, delete_and_reindex, full_rebuild_from_scratch, ingest_and_index
from src.utils.timing import measure
from src.utils.metadata import load_documents_index

st.set_page_config(page_title="Semantic Search + RAG", layout="wide")


# ---------------------------------------------------------------------
# Cached resources: loaded once per running process, mutated in place by
# ingest/delete/rebuild so they stay in sync with disk without reloading
# the embedding model on every rerun.
# ---------------------------------------------------------------------

@st.cache_resource
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        model_name=settings.EMBEDDING_MODEL_NAME,
        expected_dimension=settings.EMBEDDING_DIMENSION,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


@st.cache_resource
def get_faiss_store() -> FaissStore:
    store = FaissStore(dimension=settings.EMBEDDING_DIMENSION, index_dir=settings.INDEXES_DIR)
    store.load()
    return store


@st.cache_resource
def get_bm25_store() -> BM25Store:
    from src.utils.metadata import load_all_chunks

    return BM25Store.from_chunk_records(load_all_chunks(settings.METADATA_DIR))


def get_llm_client() -> OpenRouterLLMClient:
    # Not cached: constructing it is cheap, and a missing/changed key
    # should be picked up on the next Q&A attempt rather than requiring a
    # process restart.
    return OpenRouterLLMClient(
        model=settings.OPENROUTER_MODEL,
        temperature=settings.OPENROUTER_TEMPERATURE,
        max_tokens=settings.OPENROUTER_MAX_TOKENS,
        timeout=settings.OPENROUTER_TIMEOUT_SECONDS,
    )


def get_hybrid_search() -> HybridSearch:
    return HybridSearch(get_embedding_service(), get_faiss_store(), get_bm25_store())


if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # list of dicts: question, answer, sources
if "last_retrieval_seconds" not in st.session_state:
    st.session_state.last_retrieval_seconds = None
if "last_generation_seconds" not in st.session_state:
    st.session_state.last_generation_seconds = None
if "last_search_seconds" not in st.session_state:
    st.session_state.last_search_seconds = None


st.title("Semantic Search + RAG Document Q&A")

tab_docs, tab_search, tab_qa, tab_config = st.tabs(
    ["Document Management", "Semantic Search", "Document Q&A", "Configuration & Diagnostics"]
)


# ---------------------------------------------------------------------
# Tab 1: Document Management
# ---------------------------------------------------------------------
with tab_docs:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, or Markdown",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process uploaded files", disabled=not uploaded_files):
        embedding_service = get_embedding_service()
        faiss_store = get_faiss_store()
        bm25_store = get_bm25_store()

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)
            try:
                record = ingest_and_index(
                    tmp_path, uploaded_file.name, len(file_bytes),
                    settings.DOCUMENTS_DIR, settings.METADATA_DIR,
                    settings.CHUNK_SIZE, settings.CHUNK_OVERLAP,
                    settings.MAX_UPLOAD_SIZE_MB, settings.SUPPORTED_EXTENSIONS,
                    embedding_service, faiss_store, bm25_store,
                )
                st.success(f"'{uploaded_file.name}' ingested: {record.num_chunks} chunks.")
            except (IngestionError, DocumentLoadError) as exc:
                st.error(f"'{uploaded_file.name}' failed: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)

    st.divider()
    st.header("Uploaded documents")
    documents_index = load_documents_index(settings.METADATA_DIR)

    if not documents_index:
        st.info("No documents uploaded yet.")
    else:
        for doc_id, record in sorted(documents_index.items(), key=lambda kv: kv[1]["uploaded_at"]):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 2, 1])
            col1.write(record["filename"])
            col2.write(record["file_type"])
            col3.write(f"{record['num_chunks']} chunks")
            col4.write(record["uploaded_at"][:19])
            if col5.button("Delete", key=f"delete_{doc_id}"):
                delete_and_reindex(
                    doc_id, settings.DOCUMENTS_DIR, settings.METADATA_DIR,
                    settings.EMBEDDING_DIMENSION, get_faiss_store(), get_bm25_store(),
                )
                st.rerun()

    st.divider()
    st.caption(
        "Rebuild re-embeds every stored chunk from scratch -- only use it to "
        "recover from a corrupted or manually-edited index, not routinely."
    )
    if st.button("Rebuild indexes from scratch"):
        with st.spinner("Re-embedding all chunks..."):
            count = full_rebuild_from_scratch(
                settings.METADATA_DIR, get_embedding_service(), get_faiss_store(), get_bm25_store()
            )
        st.success(f"Rebuilt indexes from {count} chunks.")


# ---------------------------------------------------------------------
# Tab 2: Semantic Search
# ---------------------------------------------------------------------
with tab_search:
    st.header("Search your documents")
    query = st.text_input("Query", key="search_query")
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=5, key="search_top_k")
    mode = st.radio(
        "Retrieval mode",
        ["Hybrid (RRF)", "Semantic only", "Keyword (BM25) only", "Compare all three"],
        horizontal=True,
    )

    def _render_results(results, chunk_lookup):
        if not results:
            st.info("No matching chunks.")
            return
        for chunk_id, score in results:
            record = chunk_lookup.get(chunk_id)
            if record is None:
                continue
            location = record["source_filename"]
            if record.get("page_number") is not None:
                location += f", page {record['page_number']}"
            with st.expander(f"{location} -- score {score:.3f}"):
                st.write(record["text"])

    if st.button("Search", disabled=not query):
        hybrid_search = get_hybrid_search()
        chunk_lookup = build_chunk_lookup(settings.METADATA_DIR)

        with measure() as search_timer:
            if mode == "Compare all three":
                col_sem, col_kw, col_hyb = st.columns(3)
                with col_sem:
                    st.subheader("Semantic")
                    _render_results(hybrid_search.search_semantic(query, top_k), chunk_lookup)
                with col_kw:
                    st.subheader("Keyword (BM25)")
                    _render_results(hybrid_search.search_keyword(query, top_k), chunk_lookup)
                with col_hyb:
                    st.subheader("Hybrid (RRF)")
                    _render_results(
                        hybrid_search.search_hybrid(
                            query, top_k,
                            semantic_candidates=settings.SEMANTIC_CANDIDATES,
                            keyword_candidates=settings.KEYWORD_CANDIDATES,
                        ),
                        chunk_lookup,
                    )
            else:
                if mode == "Semantic only":
                    results = hybrid_search.search_semantic(query, top_k)
                elif mode == "Keyword (BM25) only":
                    results = hybrid_search.search_keyword(query, top_k)
                else:
                    results = hybrid_search.search_hybrid(
                        query, top_k,
                        semantic_candidates=settings.SEMANTIC_CANDIDATES,
                        keyword_candidates=settings.KEYWORD_CANDIDATES,
                    )
                _render_results(results, chunk_lookup)

        st.session_state.last_search_seconds = search_timer.elapsed_seconds
        st.caption(f"Retrieved in {search_timer.elapsed_seconds * 1000:.0f} ms")


# ---------------------------------------------------------------------
# Tab 3: Document Q&A
# ---------------------------------------------------------------------
with tab_qa:
    st.header("Ask questions about your documents")

    for turn in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                with st.expander("Sources"):
                    for source in turn["sources"]:
                        location = source.source_filename
                        if source.page_number is not None:
                            location += f", page {source.page_number}"
                        st.markdown(f"- **{location}** (score {source.retrieval_score:.3f})")

    question = st.chat_input("Ask a question about your uploaded documents")

    if st.button("Clear conversation"):
        st.session_state.conversation_history = []
        st.rerun()

    if question:
        try:
            llm_client = get_llm_client()
        except OpenRouterClientError as exc:
            st.error(str(exc))
        else:
            history_pairs = [(t["question"], t["answer"]) for t in st.session_state.conversation_history]
            pipeline = RagPipeline(
                hybrid_search=get_hybrid_search(),
                llm_client=llm_client,
                chunk_lookup_provider=lambda: build_chunk_lookup(settings.METADATA_DIR),
                max_context_chars=settings.MAX_CONTEXT_CHARS,
                top_k=settings.RAG_TOP_K,
                semantic_candidates=settings.SEMANTIC_CANDIDATES,
                keyword_candidates=settings.KEYWORD_CANDIDATES,
            )
            try:
                result = pipeline.answer_question(question, conversation_history=history_pairs)
            except OpenRouterClientError as exc:
                st.error(str(exc))
            else:
                st.session_state.last_retrieval_seconds = result.retrieval_seconds
                st.session_state.last_generation_seconds = result.generation_seconds
                st.session_state.conversation_history.append(
                    {"question": question, "answer": result.answer, "sources": result.sources}
                )
                st.rerun()


# ---------------------------------------------------------------------
# Tab 4: Configuration & Diagnostics
# ---------------------------------------------------------------------
with tab_config:
    st.header("Configuration")
    st.write(f"Embedding model: `{settings.EMBEDDING_MODEL_NAME}` ({settings.EMBEDDING_DIMENSION}-dim)")
    st.write(f"LLM model (via OpenRouter): `{settings.OPENROUTER_MODEL}`")
    st.write(f"Chunk size / overlap: {settings.CHUNK_SIZE} / {settings.CHUNK_OVERLAP} characters")
    st.write(f"RAG top-K: {settings.RAG_TOP_K}, max context: {settings.MAX_CONTEXT_CHARS} characters")

    import os

    key_set = bool(os.getenv("OPENROUTER_API_KEY"))
    st.write(f"OPENROUTER_API_KEY set: {'yes' if key_set else 'no (Q&A will fail until this is set in .env)'}")

    st.divider()
    st.header("Diagnostics")
    documents_index = load_documents_index(settings.METADATA_DIR)
    total_chunks = sum(r["num_chunks"] for r in documents_index.values())
    st.write(f"Documents: {len(documents_index)}")
    st.write(f"Chunks: {total_chunks}")
    st.write(f"FAISS index size: {get_faiss_store().size}")
    st.write(f"BM25 corpus size: {get_bm25_store().size}")
    if st.session_state.last_search_seconds is not None:
        st.write(f"Last Semantic Search latency: {st.session_state.last_search_seconds:.2f}s")
    if st.session_state.last_retrieval_seconds is not None:
        st.write(f"Last Q&A retrieval latency: {st.session_state.last_retrieval_seconds:.2f}s")
    if st.session_state.last_generation_seconds is not None:
        st.write(f"Last Q&A generation latency: {st.session_state.last_generation_seconds:.2f}s")
