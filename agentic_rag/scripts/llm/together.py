"""Together AI chat-model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.settings import RagSettings


@dataclass(frozen=True)
class TogetherLLMs:
    responder: Any
    pruning: Any


def create_together_llms(settings: RagSettings) -> TogetherLLMs:
    """Create deterministic responder and pruning clients for the same model."""

    api_key = settings.require_together_api_key()
    try:
        from langchain_together import ChatTogether
    except ImportError as exc:
        raise RuntimeError(
            "langchain-together is required. Install the packages in requirements.txt."
        ) from exc

    common = {
        "model": settings.together_model,
        "api_key": api_key,
        "temperature": 0,
        "timeout": 60,
        "max_retries": 2,
    }
    return TogetherLLMs(
        responder=ChatTogether(**common),
        pruning=ChatTogether(**common),
    )
