"""llm/router.py — 작업 유형·복잡도별 LLM 선택."""

from __future__ import annotations

import pytest

from llm.router import LLMRouter


def _router(local_available: bool) -> LLMRouter:
    r = LLMRouter(threshold=0.6)
    r._local_available = local_available
    return r


@pytest.mark.parametrize("task", ["parse", "analyze", "generate", "review"])
def test_high_complexity_always_openai(task):
    assert _router(True).route(task, "high") == "openai"


@pytest.mark.parametrize("task", ["analyze", "generate", "review"])
def test_quality_sensitive_medium_tasks_use_openai(task):
    assert _router(True).route(task, "medium") == "openai"


def test_parse_medium_and_low_use_local_when_available():
    r = _router(True)
    assert r.route("parse", "medium") == "local"
    assert r.route("analyze", "low") == "local"


def test_falls_back_to_openai_when_local_unavailable():
    assert _router(False).route("parse", "low") == "openai"


def test_route_by_score():
    r = _router(True)
    assert r.route_by_score(0.9) == "openai"
    assert r.route_by_score(0.1) == "local"
    assert r.route_by_score(0.5, task_type="generate") == "openai"


def test_unreachable_ollama_counts_as_unavailable():
    """OLLAMA_BASE_URL 이 응답하지 않으면 예외 없이 openai 로 라우팅."""
    r = LLMRouter()
    assert r.route("parse", "low") == "openai"
    r.invalidate_local_cache()
    assert r._local_available is None
