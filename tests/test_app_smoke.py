"""Headless smoke test for app.py using Streamlit's official AppTest API
(streamlit.testing.v1) -- runs the actual script and inspects it for
uncaught exceptions, without needing a real browser. This is the
supported way to test a Streamlit script's top-level execution; it does
not replace clicking through the UI by hand, which the project's
guidelines ask for separately before calling a UI phase done.
"""
from streamlit.testing.v1 import AppTest


def test_app_runs_without_exceptions():
    at = AppTest.from_file("app.py")
    at.run(timeout=60)
    assert not at.exception


def test_app_renders_four_tabs():
    at = AppTest.from_file("app.py")
    at.run(timeout=60)
    assert len(at.tabs) == 4
