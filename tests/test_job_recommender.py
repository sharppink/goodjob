"""
job_recommender: 프로필 기반 검색어 생성과 여러 검색어 결과 병합.

Tavily 검색(_collect_postings)과 공고 구조화·점수화는 가짜로 바꿔서
"검색어가 어떻게 정해지고 어떤 공고가 후보가 되는지"만 검증합니다.
"""

from __future__ import annotations

import pytest

from agents import job_recommender
from agents.job_recommender import (
    _collect_postings_multi,
    job_recommender_node,
    suggest_queries_from_profile,
)


def _posting(url: str) -> dict:
    return {"raw_text": "x" * 120, "title": url, "company": "C", "url": url, "requirements": {}}


@pytest.fixture
def fake_search(monkeypatch):
    """검색어별로 정해진 공고를 돌려주고, 호출된 검색어를 기록."""
    calls: list[str] = []
    results: dict[str, list[dict]] = {}

    def collect(query, max_count=10):
        calls.append(query)
        return results.get(query, [])

    def structure(candidates):
        for c in candidates:
            c["requirements"] = {"is_job_posting": True, "company_name": c["company"]}
        return candidates

    def rank(candidates):
        return [{**c, "fit_score": 0.8, "fit_feedback": "ok"} for c in candidates]

    monkeypatch.setattr(job_recommender, "_collect_postings", collect)
    monkeypatch.setattr(job_recommender, "_structure_postings", structure)
    monkeypatch.setattr(job_recommender, "_rank_candidates", rank)
    return calls, results


# ------------------------------------------------------------------ #
# suggest_queries_from_profile                                         #
# ------------------------------------------------------------------ #

def test_suggest_queries_cleans_and_dedupes(fake_llm):
    fake_llm.structured["ProfileSearchQueries"] = {
        "queries": ['"Python 백엔드"', "python  백엔드", "  ", "LLM 엔지니어", "데이터 엔지니어", "MLOps"],
    }
    queries = suggest_queries_from_profile(["기술: Python, FastAPI", "경력: 백엔드 4년"])
    assert queries == ["Python 백엔드", "LLM 엔지니어", "데이터 엔지니어"]
    prompt = fake_llm.calls[0][2]
    assert "FastAPI" in prompt and "백엔드 4년" in prompt


def test_suggest_queries_empty_profile_skips_llm(fake_llm):
    assert suggest_queries_from_profile(["", "  "]) == []
    assert fake_llm.calls == []


def test_suggest_queries_llm_error_returns_empty(fake_llm):
    fake_llm.structured["ProfileSearchQueries"] = RuntimeError("API down")
    assert suggest_queries_from_profile(["기술: Python"]) == []


# ------------------------------------------------------------------ #
# _collect_postings_multi                                              #
# ------------------------------------------------------------------ #

def test_collect_multi_interleaves_and_dedupes(fake_search):
    _, results = fake_search
    results["A"] = [_posting("a1"), _posting("dup"), _posting("a3")]
    results["B"] = [_posting("dup"), _posting("b2")]
    merged = _collect_postings_multi(["A", "B"], max_count=3)
    assert [p["url"] for p in merged] == ["a1", "dup", "b2"]


def test_collect_multi_dedupes_same_title_with_different_url(fake_search):
    """#033: 같은 공고가 다른 주소로 두 검색어에 걸리면 하나만 남김."""
    _, results = fake_search
    a = {**_posting("https://saramin/1"), "title": "(주)차후 2026 백엔드 개발자 경력 채용"}
    b = {**_posting("https://jobkorea/9"), "title": "㈜차후 2026 백엔드  개발자 경력 채용 | 잡코리아"}
    other = {**_posting("https://x/2"), "title": "차후 2026 프론트엔드 개발자 채용"}
    results["A"] = [a]
    results["B"] = [b, other]
    merged = _collect_postings_multi(["A", "B"])
    assert [p["url"] for p in merged] == ["https://saramin/1", "https://x/2"]


@pytest.mark.parametrize("title, expected", [
    ("$128k-$180k Fastapi Jobs (NOW HIRING) Oct 2026", True),
    ("Best Python Jobs in NYC, NY 2026 | Built In NYC", True),
    ("1,234 Python developer jobs", True),
    ("백엔드 개발자 채용공고 120건", True),
    ("[(주)더페어글로벌] 백엔드 개발자 (Python)", False),
    ("Backend Engineer (Python) - Toss", False),
    ("[리소리우스] [인턴]백엔드 개발자 (Python, FastAPI/Django) 채용 공고 | 원티드", False),
    ("2026 백엔드 개발자 3명 채용", False),
])
def test_is_listing_title(title, expected):
    """#027: URL 로 못 거르는 목록 페이지를 제목으로 판별."""
    assert job_recommender._is_listing_title(title) is expected


# ------------------------------------------------------------------ #
# job_recommender_node                                                 #
# ------------------------------------------------------------------ #

def test_node_without_query_uses_profile_queries(fake_llm, fake_search, loaded_profile):
    calls, results = fake_search
    fake_llm.structured["ProfileSearchQueries"] = {"queries": ["Python 백엔드", "FastAPI 개발자"]}
    results["Python 백엔드"] = [_posting("p1")]
    results["FastAPI 개발자"] = [_posting("f1")]

    state = job_recommender_node({"recommendation_query": "", "errors": []})

    assert calls == ["Python 백엔드", "FastAPI 개발자"]
    assert state["recommendation_queries"] == ["Python 백엔드", "FastAPI 개발자"]
    assert [m["url"] for m in state["ranked_matches"]] == ["p1", "f1"]
    # 프로필 내용이 검색어 생성 프롬프트에 들어감
    assert "FastAPI" in fake_llm.calls[0][2]


def test_node_with_query_skips_profile_queries(fake_llm, fake_search, loaded_profile):
    calls, results = fake_search
    results["AI 엔지니어"] = [_posting("ai1")]

    state = job_recommender_node({"recommendation_query": " AI 엔지니어 ", "errors": []})

    assert calls == ["AI 엔지니어"]
    assert state["recommendation_queries"] == ["AI 엔지니어"]
    assert fake_llm.count("structured", "ProfileSearchQueries") == 0


def test_node_reports_when_profile_queries_fail(fake_llm, fake_search, loaded_profile):
    calls, _ = fake_search
    fake_llm.structured["ProfileSearchQueries"] = {"queries": []}

    state = job_recommender_node({"errors": []})

    assert calls == []
    assert state["ranked_matches"] == []
    assert any("검색어" in e for e in state["errors"])
