"""Tests for app.llm's fallback logic. Uses fake providers -- no network,
no real API keys needed. What matters here is the *fallback behaviour*, not
whether Groq or Gemini's SDKs work, which can't be verified without network
access anyway (confirm that manually against the real APIs).

Runnable two ways:
    pytest -q
    python tests/test_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import AllProvidersFailedError, complete_with_fallback  # noqa: E402


class _FakeProvider:
    """A provider stub. Set `text` to succeed, or `error` to raise."""

    def __init__(self, name: str, text: str | None = None, error: Exception | None = None):
        self.name = name
        self._text = text
        self._error = error
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self._error:
            raise self._error
        return self._text


def test_first_provider_used_when_it_succeeds():
    p1 = _FakeProvider("groq", text="SELECT 1")
    p2 = _FakeProvider("gemini", text="should not be called")
    text, used = complete_with_fallback("sys", "user", [p1, p2])
    assert text == "SELECT 1"
    assert used == "groq"
    assert p2.calls == 0


def test_falls_through_to_second_provider_on_error():
    p1 = _FakeProvider("groq", error=RuntimeError("429 rate limited"))
    p2 = _FakeProvider("gemini", text="SELECT 2")
    text, used = complete_with_fallback("sys", "user", [p1, p2])
    assert text == "SELECT 2"
    assert used == "gemini"


def test_empty_response_counts_as_failure_and_falls_through():
    p1 = _FakeProvider("groq", text="   ")
    p2 = _FakeProvider("gemini", text="SELECT 3")
    text, used = complete_with_fallback("sys", "user", [p1, p2])
    assert used == "gemini"


def test_all_providers_failing_raises():
    p1 = _FakeProvider("groq", error=RuntimeError("boom"))
    p2 = _FakeProvider("gemini", error=RuntimeError("also boom"))
    try:
        complete_with_fallback("sys", "user", [p1, p2])
        assert False, "expected AllProvidersFailedError"
    except AllProvidersFailedError as exc:
        assert "groq" in str(exc) and "gemini" in str(exc)


def test_empty_provider_list_raises_with_helpful_message():
    try:
        complete_with_fallback("sys", "user", [])
        assert False, "expected AllProvidersFailedError"
    except AllProvidersFailedError as exc:
        assert "API_KEY" in str(exc)


TESTS = [
    test_first_provider_used_when_it_succeeds,
    test_falls_through_to_second_provider_on_error,
    test_empty_response_counts_as_failure_and_falls_through,
    test_all_providers_failing_raises,
    test_empty_provider_list_raises_with_helpful_message,
]


if __name__ == "__main__":
    passed = 0
    for t in TESTS:
        t()
        print(f"  pass  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(TESTS)} passed")
