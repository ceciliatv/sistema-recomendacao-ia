"""Fábrica de provedores por nome."""

from __future__ import annotations

from repertorio.agent.providers.base import Provider, ProviderError
from repertorio.config import get_llm_settings

PROVIDERS = ("anthropic", "openai", "heuristic")


def get_provider(name: str, model: str | None = None) -> Provider:
    settings = get_llm_settings()
    if name == "anthropic":
        from repertorio.agent.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model or settings.anthropic_model, settings.anthropic_effort)
    if name == "openai":
        from repertorio.agent.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model or settings.openai_model, settings.openai_reasoning_effort)
    if name == "heuristic":
        from repertorio.agent.providers.heuristic import HeuristicProvider

        return HeuristicProvider()
    raise ProviderError(f"provedor desconhecido: {name!r} (use {', '.join(PROVIDERS)})")
