"""
#028: 자동 검색이 채용공고가 아닌 글을 공고로 넘기지 않는지 확인.

- search.pipeline.find_job_postings : 페이지별로 거르고 공고로 판정된 것만 반환
- 공고 입력 화면 '자동 검색'      : 공고 후보만 보여 주고 고른 공고를 입력으로 사용
- 분석 화면                      : 공고가 아닌 글이면 파싱 후 멈추고 사용자 확인
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import SAMPLE_REQUIREMENTS
from tests.test_app_login import APP, NO_AUTH_SECRETS

LONG = "본문 " * 120  # MIN_POSTING_CHARS 이상


def _page(url: str, title: str, content: str = LONG) -> dict:
    return {"url": url, "title": title, "content": content}


@pytest.fixture
def fake_search(monkeypatch):
    """Tavily 검색 결과와 공고 판정을 가짜로. 본문에 '블로그' 가 있으면 공고 아님으로 판정."""
    import agents.job_parser
    import search.company_searcher

    pages: list[dict] = []
    parsed: list[str] = []

    def results(self, company_name, role="개발자"):
        return pages

    def parse(raw_text, raise_on_error=False):
        parsed.append(raw_text)
        if "블로그" in raw_text:
            return {**SAMPLE_REQUIREMENTS, "is_job_posting": False}
        company = "당근" if "당근" in raw_text else "다른회사"
        return {**SAMPLE_REQUIREMENTS, "company_name": company, "job_title": "백엔드 엔지니어"}

    monkeypatch.setattr(search.company_searcher.CompanySearcher, "__init__", lambda self: None)
    monkeypatch.setattr(search.company_searcher.CompanySearcher, "search_job_posting_results", results)
    monkeypatch.setattr(agents.job_parser, "parse_posting_structured", parse)
    return pages, parsed


# ------------------------------------------------------------------ #
# find_job_postings                                                    #
# ------------------------------------------------------------------ #

def test_find_job_postings_keeps_only_postings(fake_search):
    from search.pipeline import find_job_postings

    pages, parsed = fake_search
    pages += [
        _page("https://blog/1", "당근 채용 블로그 — 개발 문화 Q&A"),
        _page("https://saramin/list/x", "목록"),                    # 목록 URL
        _page("https://x/2", "Backend Jobs in Seoul"),               # 목록 제목
        _page("https://x/3", "짧은 글", content="짧음"),              # 너무 짧음
        _page("https://other/4", "다른회사 백엔드 엔지니어 채용"),
        _page("https://wanted/5", "[당근] 백엔드 엔지니어"),
    ]
    result = find_job_postings("당근", "백엔드 엔지니어")

    assert result["found"] == 6
    assert result["checked"] == 3, "목록·짧은 글은 LLM 판정 전에 제외"
    assert len(parsed) == 3
    # 블로그 글은 빠지고, 입력한 회사(당근) 공고가 먼저
    assert [p["url"] for p in result["postings"]] == ["https://wanted/5", "https://other/4"]
    assert result["postings"][0]["raw_text"].startswith("[당근] 백엔드 엔지니어\n\n")


def test_find_job_postings_none_found(fake_search):
    from search.pipeline import find_job_postings

    pages, _ = fake_search
    pages.append(_page("https://blog/1", "당근 채용 블로그"))
    result = find_job_postings("당근")
    assert result["postings"] == []
    assert result["checked"] == 1


# ------------------------------------------------------------------ #
# Streamlit 화면                                                       #
# ------------------------------------------------------------------ #

def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets = NO_AUTH_SECRETS  # type: ignore[assignment]
    return at.run()


def _goto(at: AppTest, page: str) -> None:
    at.sidebar.radio(key="page_radio").set_value(page).run()


def test_auto_search_lets_user_pick_a_posting(fake_search, loaded_profile):
    pages, _ = fake_search
    pages += [_page("https://blog/1", "당근 채용 블로그"), _page("https://wanted/5", "[당근] 백엔드 엔지니어")]

    at = _app()
    _goto(at, "공고 입력")
    at.text_input[0].input("당근").run()
    next(r for r in at.radio if r.label == "입력 방법").set_value("자동 검색").run()
    at.button(key="search_btn").click().run()

    assert not at.exception
    assert at.session_state["job_posting_text"] == "", "고르기 전에는 입력되지 않음"
    pick = at.radio(key="search_pick")
    assert len(pick.options) == 1 and "당근 · 백엔드 엔지니어" in pick.options[0]

    at.button(key="search_use").click().run()
    assert at.session_state["job_posting_text"].startswith("[당근] 백엔드 엔지니어")


def test_auto_search_warns_when_no_posting(fake_search, loaded_profile):
    pages, _ = fake_search
    pages.append(_page("https://blog/1", "당근 채용 블로그"))

    at = _app()
    _goto(at, "공고 입력")
    at.text_input[0].input("당근").run()
    next(r for r in at.radio if r.label == "입력 방법").set_value("자동 검색").run()
    at.button(key="search_btn").click().run()

    assert not at.exception
    assert any("개별 채용공고로 판정된 것이 없습니다" in w.value for w in at.warning)
    assert at.session_state["job_posting_text"] == ""


@pytest.fixture
def fake_pipeline(monkeypatch):
    """파싱 결과만 정하고, 이후 노드는 적합도 낮음으로 끝나게 해서 호출 여부만 기록."""
    import agents.fit_analyzer
    import agents.job_parser
    import agents.rag_retriever

    calls: list[str] = []
    verdict = {"is_job_posting": False}

    def parser(state):
        calls.append("job_parser")
        state["job_requirements"] = {**SAMPLE_REQUIREMENTS, **verdict}
        return state

    def retriever(state):
        calls.append("rag_retriever")
        state["retrieved_experiences"] = []
        return state

    def fit(state):
        calls.append("fit_analyzer")
        state["fit_score"] = 0.0
        return state

    monkeypatch.setattr(agents.job_parser, "job_parser_node", parser)
    monkeypatch.setattr(agents.rag_retriever, "rag_retriever_node", retriever)
    monkeypatch.setattr(agents.fit_analyzer, "fit_analyzer_node", fit)
    return calls, verdict


def _run_analysis(text: str) -> AppTest:
    at = _app()
    at.session_state["job_posting_text"] = text
    _goto(at, "분석 & 이력서")
    next(b for b in at.button if b.label == "▶ 파이프라인 실행").click().run()
    return at


def test_analysis_stops_on_non_posting_until_confirmed(fake_pipeline, loaded_profile):
    calls, _ = fake_pipeline
    at = _run_analysis("당근 개발 문화 블로그 글")

    assert not at.exception
    assert calls == ["job_parser"], "공고가 아니면 파싱 후 멈춤"
    assert any("채용공고가 아닌 것으로 판정" in w.value for w in at.warning)
    assert not at.session_state["pipeline_ran"]

    next(b for b in at.button if b.label == "그래도 분석 계속").click().run()
    assert not at.exception
    assert calls[1:] == ["job_parser", "rag_retriever", "fit_analyzer"]
    assert at.session_state["pipeline_ran"]


def test_analysis_runs_normally_for_posting(fake_pipeline, loaded_profile):
    calls, verdict = fake_pipeline
    verdict["is_job_posting"] = True
    at = _run_analysis("[당근] 백엔드 엔지니어 채용")

    assert not at.exception
    assert calls == ["job_parser", "rag_retriever", "fit_analyzer"]
    assert not any("채용공고가 아닌 것으로 판정" in w.value for w in at.warning)
