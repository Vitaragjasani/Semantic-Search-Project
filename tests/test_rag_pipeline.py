"""RagPipeline orchestration tests, using lightweight fakes for
HybridSearch and the LLM client (any object with a generate(messages)
method -- see src/llm/base.py) so this suite doesn't need a real embedding
model, a real FAISS/BM25 index, or a network call -- those are each
already covered by their own test files. This isolates exactly what
RagPipeline itself is responsible for: calling retrieval, building
context, skipping the LLM when there's no evidence, and passing the
right prompt through when there is.
"""
from src.rag.rag_pipeline import NO_EVIDENCE_MESSAGE, RagPipeline


class FakeHybridSearch:
    def __init__(self, ranked_results):
        self.ranked_results = ranked_results
        self.last_call = None

    def search_hybrid(self, query, top_k, semantic_candidates=20, keyword_candidates=20):
        self.last_call = {
            "query": query,
            "top_k": top_k,
            "semantic_candidates": semantic_candidates,
            "keyword_candidates": keyword_candidates,
        }
        return self.ranked_results


class FakeLLMClient:
    def __init__(self, response_text="Fake answer."):
        self.response_text = response_text
        self.received_messages = None
        self.call_count = 0

    def generate(self, messages):
        self.call_count += 1
        self.received_messages = messages
        return self.response_text


def _chunk_lookup():
    return {
        "c1": {
            "chunk_id": "c1",
            "doc_id": "d1",
            "source_filename": "policy.pdf",
            "page_number": 2,
            "text": "Refunds are processed within 10 business days.",
        }
    }


def test_answer_question_calls_llm_with_retrieved_context():
    hybrid = FakeHybridSearch(ranked_results=[("c1", 0.9)])
    llm = FakeLLMClient(response_text="Refunds take 10 business days [Source 1].")

    pipeline = RagPipeline(
        hybrid_search=hybrid,
        llm_client=llm,
        chunk_lookup_provider=_chunk_lookup,
        max_context_chars=1000,
        top_k=5,
    )

    result = pipeline.answer_question("How long do refunds take?")

    assert result.context_used is True
    assert result.answer == "Refunds take 10 business days [Source 1]."
    assert [c.chunk_id for c in result.sources] == ["c1"]
    assert llm.call_count == 1
    assert "Refunds are processed within 10 business days." in llm.received_messages[1]["content"]
    assert "How long do refunds take?" in llm.received_messages[1]["content"]
    assert result.retrieval_seconds >= 0.0
    assert result.generation_seconds >= 0.0


def test_answer_question_with_no_retrieval_hits_skips_llm_entirely():
    hybrid = FakeHybridSearch(ranked_results=[])
    llm = FakeLLMClient()

    pipeline = RagPipeline(
        hybrid_search=hybrid,
        llm_client=llm,
        chunk_lookup_provider=lambda: {},
        max_context_chars=1000,
    )

    result = pipeline.answer_question("Anything in here about dragons?")

    assert result.answer == NO_EVIDENCE_MESSAGE
    assert result.sources == []
    assert result.context_used is False
    assert llm.call_count == 0  # never pay for an LLM call with nothing to ground it
    assert result.retrieval_seconds >= 0.0
    assert result.generation_seconds == 0.0  # generation never ran at all


def test_answer_question_with_history_contextualizes_retrieval_query():
    hybrid = FakeHybridSearch(ranked_results=[("c1", 0.9)])
    llm = FakeLLMClient()

    pipeline = RagPipeline(
        hybrid_search=hybrid,
        llm_client=llm,
        chunk_lookup_provider=_chunk_lookup,
        max_context_chars=1000,
    )

    history = [("What is the refund policy?", "Refunds take 10 business days.")]
    pipeline.answer_question("What about exchanges?", conversation_history=history)

    # Retrieval query is contextualized with the prior question...
    assert hybrid.last_call["query"] == "What is the refund policy? What about exchanges?"
    # ...but the LLM still sees the original question verbatim, with history
    # passed through separately rather than folded into it.
    assert "What about exchanges?" in llm.received_messages[1]["content"]
    assert "PREVIOUS CONVERSATION" in llm.received_messages[1]["content"]


def test_answer_question_without_history_uses_question_as_is():
    hybrid = FakeHybridSearch(ranked_results=[("c1", 0.9)])
    llm = FakeLLMClient()

    pipeline = RagPipeline(
        hybrid_search=hybrid,
        llm_client=llm,
        chunk_lookup_provider=_chunk_lookup,
        max_context_chars=1000,
    )

    pipeline.answer_question("How long do refunds take?")

    assert hybrid.last_call["query"] == "How long do refunds take?"
    assert "PREVIOUS CONVERSATION" not in llm.received_messages[1]["content"]


def test_answer_question_passes_configured_candidate_counts_through():
    hybrid = FakeHybridSearch(ranked_results=[("c1", 0.9)])
    llm = FakeLLMClient()

    pipeline = RagPipeline(
        hybrid_search=hybrid,
        llm_client=llm,
        chunk_lookup_provider=_chunk_lookup,
        max_context_chars=1000,
        top_k=3,
        semantic_candidates=15,
        keyword_candidates=10,
    )

    pipeline.answer_question("question")

    assert hybrid.last_call == {
        "query": "question",
        "top_k": 3,
        "semantic_candidates": 15,
        "keyword_candidates": 10,
    }
