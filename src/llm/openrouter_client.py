"""Thin wrapper around OpenRouter's chat completions API.

OpenRouter exposes an OpenAI-compatible endpoint
(POST https://openrouter.ai/api/v1/chat/completions) that routes to many
backend models, including a number of ":free"-suffixed models with no
per-token cost. Confirmed shape (checked against openrouter.ai's own docs
and its free-models router guide): Authorization: Bearer <OPENROUTER_API_KEY>
header, JSON body with `model` and `messages`, response JSON with
choices[0].message.content -- the same response schema as OpenAI's Chat
Completions API, which is why this is a plain `requests` call rather than
a new SDK dependency.

Free-model availability changes often -- labs add prerelease checkpoints
and pull them when a promotional window ends -- so OPENROUTER_MODEL is
read from config rather than hardcoded. Check
https://openrouter.ai/models?max_price=0 for the current free-model list
before relying on one.
"""
import os
from typing import Optional

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClientError(Exception):
    pass


class MissingApiKeyError(OpenRouterClientError):
    pass


class LLMInferenceError(OpenRouterClientError):
    pass


class OpenRouterLLMClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 30.0,
    ):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise MissingApiKeyError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file "
                "(see .env.example) -- get a key at https://openrouter.ai/keys."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = key

    def generate(self, messages: list[dict]) -> str:
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMInferenceError(
                f"OpenRouter request timed out after {self.timeout}s."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMInferenceError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code == 401:
            raise LLMInferenceError(
                "OpenRouter rejected the API key (401 Unauthorized). "
                "Check OPENROUTER_API_KEY in your .env file."
            )
        if response.status_code == 402:
            raise LLMInferenceError(
                "OpenRouter rejected this request for lack of credit "
                "(402 Payment Required). Check https://openrouter.ai/credits, "
                "or switch OPENROUTER_MODEL to a ':free'-suffixed model."
            )
        if response.status_code == 429:
            raise LLMInferenceError(
                "OpenRouter rate-limited this request (429 Too Many "
                "Requests). Free models are rate-limited hard during peak "
                "hours -- wait and retry, or switch OPENROUTER_MODEL."
            )
        if response.status_code == 502:
            raise LLMInferenceError(
                f"Model '{self.model}' failed to generate upstream (502 Bad "
                "Gateway). This happens with free/prerelease models -- try "
                "a different OPENROUTER_MODEL."
            )
        if not response.ok:
            raise LLMInferenceError(
                f"OpenRouter inference failed: {response.status_code} {response.text[:300]}"
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMInferenceError(
                f"OpenRouter returned an unexpected response shape: {data}"
            ) from exc

        if not content:
            raise LLMInferenceError("OpenRouter returned an empty response.")
        return content
