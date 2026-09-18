"""Shared LLM layer: one router, one registry, one set of clients.

An earlier pass added a second registry here (a provider Protocol, a
ModelSpec-based ModelRegistry, an LLMFactory and a retry/fallback policy)
alongside the router. Nothing in production ever called it -- the adapters
that bridged it to the real clients were imported only by their own tests --
so it was a parallel description of a job ``router.py`` was already doing.
Two registries in one package is how a model gets configured in one place and
resolved from the other, which is the failure it was supposed to prevent. It
was removed rather than carried.

Dependency injection is still here, and is actually used: the factory is
``ascent_domain.non_omop.llm_provider.get_llm_service()`` and every non-OMOP workflow
takes ``llm_service=None``.
"""

from ascent_platform.llm.router import DEFAULT_REGISTRY, LLMRouter, ModelConfig, ModelRegistry

__all__ = ["DEFAULT_REGISTRY", "LLMRouter", "ModelConfig", "ModelRegistry"]
