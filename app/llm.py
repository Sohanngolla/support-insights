"""LLM provider seam.

Two providers behind one interface, tried in the order given by
settings.provider_chain (Groq first -- higher free-tier rate limit, no
credit card). On a rate limit, timeout, or any provider-side error, the
caller falls through to the next configured provider rather than failing
the request. This is the whole reason two providers exist: Gemini barely
gets used, but it means a demo doesn't die on a single 429.

Each provider's SDK import is lazy (inside __init__), so a machine that only
has one API key installed doesn't need the other package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings, settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A provider failed to produce a completion."""


class AllProvidersFailedError(LLMError):
    """Every configured provider in the chain failed."""


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class GroqProvider:
    api_key: str
    model: str
    name: str = "groq"

    def __post_init__(self) -> None:
        from groq import Groq  # lazy import

        self._client = Groq(api_key=self.api_key)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


@dataclass
class GeminiProvider:
    api_key: str
    model: str
    name: str = "gemini"

    def __post_init__(self) -> None:
        from google import genai  # lazy import, new SDK (google-genai)
        from google.genai import types

        self._client = genai.Client(api_key=self.api_key)
        self._types = types

    def complete(self, system: str, user: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system, temperature=0
            ),
        )
        return resp.text or ""


_BUILDERS = {
    "groq": lambda s: GroqProvider(api_key=s.groq_api_key, model=s.groq_model),
    "gemini": lambda s: GeminiProvider(api_key=s.gemini_api_key, model=s.gemini_model),
}


def build_providers(cfg: Settings | None = None) -> list[Provider]:
    """Instantiate providers for every configured chain entry that has a key.

    Order follows settings.provider_chain. A provider with no key configured
    is skipped, not errored -- lets you run with just one API key set.
    """
    cfg = cfg or settings
    providers: list[Provider] = []
    for name in cfg.provider_chain:
        if name not in _BUILDERS:
            log.warning("unknown provider in chain: %s", name)
            continue
        if name not in cfg.configured_providers():
            continue
        providers.append(_BUILDERS[name](cfg))
    return providers


def complete_with_fallback(
    system: str, user: str, providers: list[Provider]
) -> tuple[str, str]:
    """Try each provider in order. Returns (text, provider_name_used).

    Raises AllProvidersFailedError only if every provider raised -- including
    the case of an empty provider list, which means no API key is configured
    at all.
    """
    if not providers:
        raise AllProvidersFailedError(
            "No LLM provider is configured. Set GROQ_API_KEY and/or "
            "GEMINI_API_KEY in .env."
        )

    errors: list[str] = []
    for provider in providers:
        try:
            text = provider.complete(system, user)
            if not text.strip():
                raise LLMError(f"{provider.name} returned an empty response")
            return text, provider.name
        except Exception as exc:  # provider SDKs raise their own exception types
            log.warning("provider %s failed: %s", provider.name, exc)
            errors.append(f"{provider.name}: {exc}")

    raise AllProvidersFailedError(
        "All configured providers failed: " + "; ".join(errors)
    )
