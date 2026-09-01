"""Shared LLM client.

Currently backed by Hugging Face Inference Providers for testing (HF_TOKEN
required). Swap this out for ChatOpenAI pointed at your own vLLM server once
that's ready to test against.
"""

import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


def get_llm(temperature: float = 0.0) -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id=os.environ.get("HF_MODEL", "Qwen/Qwen3.5-9B"),
        provider=os.environ.get("HF_PROVIDER", "auto"),
        temperature=temperature,
    )
    return ChatHuggingFace(llm=endpoint)
