from src.ingestion.text_cleaner import clean_text


def test_collapses_repeated_spaces_and_tabs():
    assert clean_text("a   b\tc") == "a b c"


def test_caps_excess_blank_lines_at_one():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_dehyphenates_line_wrapped_words():
    assert clean_text("informa-\ntion") == "information"


def test_strips_control_characters():
    assert clean_text("a\x00b\x0bc") == "abc"


def test_preserves_punctuation_and_structure():
    text = "Section 1: Overview\n\n- item one\n- item two"
    assert clean_text(text) == text


def test_trims_surrounding_whitespace():
    assert clean_text("  \n  hello  \n  ") == "hello"
