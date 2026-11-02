"""Orchestrates the end-to-end RAG flow: question -> hybrid retrieval ->
bounded context -> prompt -> LLM -> answer with sources.

If retrieval finds nothing (empty corpus, or a question with no matching
chunks), the LLM is never called -- there's nothing useful it could ground
an answer in, and skipping the call also means skipping its cost. The
fixed NO_EVIDENCE_MESSAGE is returned instead, satisfying "provide a clear
response when no relevant evidence is found" without making that response
depend on the LLM working at all.

Follow-up questions ("what about the second one?") are contextualized for
*retrieval* by prepending the immediately preceding question to the
current one before searching -- a cheap heuristic that gives FAISS/BM25
more lexical surface to match against, without a separate LLM call to
rewrite the query. The *prompt* sent to the LLM still asks the original
question verbatim and carries the conversation history in its own
separate section (see prompt_builder) -- retrieval contextualization and
"what the LLM is told the user asked" are deliberately kept independent,
so a follow-up never gets silently rephrased in front of the model.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from src.llm.base import LLMClient
from src.rag.context_builder import ContextChunk, build_context
from src.rag.prompt_builder import build_messages
from src.retrieval.hybrid_search import HybridSearch
from src.utils.timing import measure

NO_EVIDENCE_MESSAGE = (
    "I couldn't find anything in your uploaded documents relevant to this question."
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[ContextChunk]
    context_used: bool
    retrieval_seconds: float
    generation_seconds: float


class RagPipeline:
    def __init__(
        self,
        hybrid_search: HybridSearch,
        llm_client: LLMClient,
        chunk_lookup_provider: Callable[[], dict[str, dict]],
        max_context_chars: int,
        top_k: int = 5,
        semantic_candidates: int = 20,
        keyword_candidates: int = 20,
    ):
        self._hybrid_search = hybrid_search
        self._llm_client = llm_client
        self._chunk_lookup_provider = chunk_lookup_provider
        self._max_context_chars = max_context_chars
        self._top_k = top_k
        self._semantic_candidates = semantic_candidates
        self._keyword_candidates = keyword_candidates

    def answer_question(
        self,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> RagAnswer:
        retrieval_query = question
        if conversation_history:
            retrieval_query = f"{conversation_history[-1][0]} {question}"

        with measure() as retrieval_timer:
            ranked = self._hybrid_search.search_hybrid(
                retrieval_query,
                top_k=self._top_k,
                semantic_candidates=self._semantic_candidates,
                keyword_candidates=self._keyword_candidates,
            )
            chunk_lookup = self._chunk_lookup_provider()
            context = build_context(ranked, chunk_lookup, self._max_context_chars)

        if not context.sources:
            return RagAnswer(
                answer=NO_EVIDENCE_MESSAGE,
                sources=[],
                context_used=False,
                retrieval_seconds=retrieval_timer.elapsed_seconds,
                generation_seconds=0.0,
            )

        messages = build_messages(question, context, conversation_history)
        with measure() as generation_timer:
            answer_text = self._llm_client.generate(messages)

        return RagAnswer(
            answer=answer_text,
            sources=context.sources,
            context_used=True,
            retrieval_seconds=retrieval_timer.elapsed_seconds,
            generation_seconds=generation_timer.elapsed_seconds,
        )
