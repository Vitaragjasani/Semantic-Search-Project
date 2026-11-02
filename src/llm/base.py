"""The minimal interface RagPipeline needs from any LLM client.

Kept as a Protocol (structural typing) rather than an abstract base class
so swapping providers (Hugging Face, OpenRouter, anything else later)
never requires the client to inherit from anything -- it just needs a
generate(messages) -> str method.
"""
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, messages: list[dict]) -> str: ...
