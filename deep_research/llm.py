"""Centralized model factory for Gemini (cloud) and Ollama (local)."""
from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model


def get_model(
    model_name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    temperature: float = 0.2,
):
    """Build a LangChain chat model using runtime config, then environment defaults."""
    configurable = (config or {}).get("configurable", config or {})
    provider = str(configurable.get("llm_provider") or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if provider not in {"ollama", "gemini"}:
        raise ValueError("LLM_PROVIDER must be either 'ollama' or 'gemini'.")

    if provider == "ollama":
        selected_model = model_name or configurable.get("research_model_name") or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        base_url = configurable.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return init_chat_model(
            selected_model,
            model_provider="ollama",
            base_url=base_url,
            temperature=temperature,
        )

    selected_model = model_name or configurable.get("research_model_name") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise ValueError("Gemini selected, but GEMINI_API_KEY (or GOOGLE_API_KEY) is missing from .env.")
    return init_chat_model(
        selected_model,
        model_provider="google_genai",
        temperature=temperature,
    )


def get_model_for(config: dict[str, Any] | None, role: str, *, temperature: float = 0.2):
    """Resolve a role-specific model override or the shared provider/model."""
    configurable = (config or {}).get("configurable", config or {})
    role_model = configurable.get(f"{role}_model_name")
    return get_model(role_model, config=config, temperature=temperature)
