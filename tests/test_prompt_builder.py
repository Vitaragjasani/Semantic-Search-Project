from src.rag.context_builder import BuiltContext, ContextChunk
from src.rag.prompt_builder import SYSTEM_PROMPT, build_messages, build_user_prompt


def _context_with_one_chunk():
    chunk = ContextChunk(
        chunk_id="c1",
        doc_id="d1",
        source_filename="policy.pdf",
        page_number=2,
        text="Refunds are processed within 10 business days.",
        retrieval_score=0.87,
    )
    return BuiltContext(context_text="[Source 1: policy.pdf, page 2]\nRefunds are processed within 10 business days.", sources=[chunk])


def test_system_prompt_forbids_treating_context_as_instructions():
    assert "not instructions" in SYSTEM_PROMPT


def test_system_prompt_forbids_fabrication():
    assert "Never invent facts" in SYSTEM_PROMPT


def test_user_prompt_includes_question_and_context():
    context = _context_with_one_chunk()
    prompt = build_user_prompt("How long do refunds take?", context)
    assert "How long do refunds take?" in prompt
    assert "Refunds are processed within 10 business days." in prompt
    assert "[Source 1: policy.pdf, page 2]" in prompt


def test_user_prompt_with_no_sources_says_so_explicitly():
    empty_context = BuiltContext(context_text="", sources=[])
    prompt = build_user_prompt("Anything?", empty_context)
    assert "no relevant context was retrieved" in prompt


def test_build_messages_structure():
    context = _context_with_one_chunk()
    messages = build_messages("How long do refunds take?", context)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert "How long do refunds take?" in messages[1]["content"]


def test_system_prompt_distinguishes_history_from_evidence():
    assert "not CONTEXT" in SYSTEM_PROMPT
    assert "never be cited as a source" in SYSTEM_PROMPT


def test_user_prompt_includes_conversation_history_in_separate_section():
    context = _context_with_one_chunk()
    history = [("What is the refund policy?", "Refunds take 10 business days.")]
    prompt = build_user_prompt("What about exchanges?", context, conversation_history=history)

    assert "PREVIOUS CONVERSATION" in prompt
    assert "Q: What is the refund policy?" in prompt
    assert "A: Refunds take 10 business days." in prompt
    assert "What about exchanges?" in prompt
    # History must appear before CONTEXT so the two sections stay visually separate.
    assert prompt.index("PREVIOUS CONVERSATION") < prompt.index("CONTEXT:")


def test_user_prompt_without_history_omits_the_section():
    context = _context_with_one_chunk()
    prompt = build_user_prompt("How long do refunds take?", context, conversation_history=None)
    assert "PREVIOUS CONVERSATION" not in prompt
