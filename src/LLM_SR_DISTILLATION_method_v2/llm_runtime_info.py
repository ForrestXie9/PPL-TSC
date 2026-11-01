"""Display the resolved LLM settings without credentials."""
from __future__ import annotations

import os


def print_llm_startup_banner(
    llm_api_endpoint: str,
    llm_model: str,
    *,
    title: str = "LLM configuration",
) -> None:
    backend = os.getenv("LLM_BACKEND") or "openai_compatible"
    request_path = os.getenv("LLM_REQUEST_PATH", "/v1/chat/completions")
    temperature = os.getenv("LLM_TEMPERATURE", "0.8")
    print(f"{title}: backend={backend}; host={llm_api_endpoint}; "
          f"model={llm_model}; path={request_path}; temperature={temperature}")
