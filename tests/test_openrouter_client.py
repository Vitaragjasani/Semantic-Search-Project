"""Tests for the error-handling/translation logic in OpenRouterLLMClient.

No real network calls are made: requests.post is monkeypatched on the
`requests` module as seen by openrouter_client, simulating each response.
This is the one legitimate place in the project to stub an external
boundary rather than exercise it for real, since the real boundary is a
paid, rate-limited third-party service, not business logic.
"""
import pytest

from src.llm import openrouter_client
from src.llm.openrouter_client import (
    LLMInferenceError,
    MissingApiKeyError,
    OpenRouterLLMClient,
)


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        return self._json_data


def _success_response(content: str) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def test_missing_api_key_raises_when_no_env_and_no_arg(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        OpenRouterLLMClient(model="some/model")


def test_explicit_api_key_overrides_missing_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterLLMClient(model="some/model", api_key="explicit-key")
    assert client.model == "some/model"


def test_generate_returns_content_on_success(monkeypatch):
    client = OpenRouterLLMClient(model="some/model", api_key="fake-key")
    monkeypatch.setattr(
        openrouter_client.requests, "post", lambda *a, **k: _success_response("The refund takes 10 days.")
    )

    result = client.generate([{"role": "user", "content": "hi"}])
    assert result == "The refund takes 10 days."


def test_generate_maps_401_to_actionable_error(monkeypatch):
    client = OpenRouterLLMClient(model="some/model", api_key="fake-key")
    monkeypatch.setattr(openrouter_client.requests, "post", lambda *a, **k: FakeResponse(401))

    with pytest.raises(LLMInferenceError, match="401"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_maps_402_to_actionable_error(monkeypatch):
    client = OpenRouterLLMClient(model="some/model", api_key="fake-key")
    monkeypatch.setattr(openrouter_client.requests, "post", lambda *a, **k: FakeResponse(402))

    with pytest.raises(LLMInferenceError, match="credit"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_maps_429_to_actionable_error(monkeypatch):
    client = OpenRouterLLMClient(model="some/model", api_key="fake-key")
    monkeypatch.setattr(openrouter_client.requests, "post", lambda *a, **k: FakeResponse(429))

    with pytest.raises(LLMInferenceError, match="rate-limited"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_maps_502_to_actionable_error_naming_the_model(monkeypatch):
    client = OpenRouterLLMClient(model="some/model", api_key="fake-key")
    monkeypatch.setattr(openrouter_client.requests, "post", lambda *a, **k: FakeResponse(502))

    with pytest.raises(LLMInferenceError, match="some/model"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_maps_timeout_to_actionable_error(monkeypatch):
    client = OpenRouterLLMClient(model="some/model", api_key="fake-key", timeout=5.0)

    def _raise_timeout(*args, **kwargs):
        raise openrouter_client.requests.exceptions.Timeout()

    monkeypatch.setattr(openrouter_client.requests, "post", _raise_timeout)

    with pytest.raises(LLMInferenceError, match="timed out"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_does_not_fabricate_on_empty_response(monkeypatch):
    client = OpenRouterLLMClient(model="some/model", api_key="fake-key")
    monkeypatch.setattr(openrouter_client.requests, "post", lambda *a, **k: _success_response(""))

    with pytest.raises(LLMInferenceError, match="empty response"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_raises_on_unexpected_response_shape(monkeypatch):
    client = OpenRouterLLMClient(model="some/model", api_key="fake-key")
    monkeypatch.setattr(openrouter_client.requests, "post", lambda *a, **k: FakeResponse(200, {"unexpected": "shape"}))

    with pytest.raises(LLMInferenceError, match="unexpected response shape"):
        client.generate([{"role": "user", "content": "hi"}])
