"""공개 배포용 사용량 제한 (llm/usage_limit.py) — 계정별·전체 하루 한도, 제외 계정, 화면 연결."""

from __future__ import annotations

import pytest

from llm import usage_limit
from llm.usage_limit import UsageLimitExceeded, consume, remaining_all


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    from config.settings import settings
    from rag.vectorstore import set_current_user

    usage_limit.reset_memory()
    monkeypatch.setattr(settings, "USAGE_DAILY_LIMITS", "analysis=2,recommend=1")
    monkeypatch.setattr(settings, "USAGE_GLOBAL_DAILY_LIMIT", 100)
    monkeypatch.setattr(settings, "USAGE_EXEMPT_EMAILS", "owner@example.com")
    yield settings
    set_current_user(None)
    usage_limit.reset_memory()


def _login(email):
    from rag.vectorstore import set_current_user
    set_current_user(email)


def test_settings_parse_limits(clean):
    clean.USAGE_DAILY_LIMITS = " analysis = 5 , star=10, broken, x=abc,recommend=0"
    assert clean.usage_daily_limits == {"analysis": 5, "star": 10, "recommend": 0}


def test_no_limit_without_login():
    for _ in range(10):
        consume("analysis")
    assert remaining_all() == {}


def test_per_account_daily_limit():
    _login("a@example.com")
    consume("analysis")
    assert remaining_all() == {"analysis": 1, "recommend": 1}
    consume("analysis")
    with pytest.raises(UsageLimitExceeded, match="분석 & 이력서 생성.*2회"):
        consume("analysis")
    consume("star")  # 한도가 없는 기능은 계속 가능

    _login("b@example.com")
    consume("analysis")  # 다른 계정은 따로 셈
    assert remaining_all()["analysis"] == 1


def test_exempt_account_and_privacy_mode_are_not_limited(clean):
    _login("owner@example.com")
    for _ in range(5):
        consume("recommend")
    assert remaining_all() == {}

    _login("a@example.com")
    clean.PRIVACY_MODE = True  # 로컬 처리라 API 비용 없음
    try:
        for _ in range(5):
            consume("recommend")
    finally:
        clean.PRIVACY_MODE = False


def test_zero_limit_blocks_feature(clean):
    clean.USAGE_DAILY_LIMITS = "recommend=0"
    _login("a@example.com")
    with pytest.raises(UsageLimitExceeded):
        consume("recommend")


def test_global_limit_refunds_personal_count(clean):
    clean.USAGE_GLOBAL_DAILY_LIMIT = 2
    _login("a@example.com")
    consume("analysis")
    _login("b@example.com")
    consume("star")
    with pytest.raises(UsageLimitExceeded, match="서비스 전체"):
        consume("analysis")
    assert remaining_all()["analysis"] == 2, "전체 한도로 막힌 실행은 개인 횟수에 남지 않음"


# ------------------------------------------------------------------ #
# 화면 연결                                                            #
# ------------------------------------------------------------------ #

def test_app_blocks_recommend_over_limit_and_shows_remaining(monkeypatch, clean):
    import agents.job_recommender
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from tests.test_app_login import APP, AUTH_SECRETS, FakeUser

    called: list[dict] = []

    def fake_node(state):
        called.append(state)
        return {"recommendation_queries": ["Python"], "ranked_matches": [], "errors": []}

    monkeypatch.setattr(agents.job_recommender, "job_recommender_node", fake_node)
    monkeypatch.setattr(st, "user", FakeUser("viewer@example.com"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets = AUTH_SECRETS  # type: ignore[assignment]
    at.run()
    assert any("오늘 남은 횟수" in c.value and "공고 추천 1회" in c.value for c in at.sidebar.caption)

    # 이 계정에는 프로필이 없어 추천 화면이 막히므로 등록된 것으로 표시
    at.session_state["profile_indexed"] = True
    at.sidebar.radio(key="page_radio").set_value("공고 추천").run()

    def click_search():
        next(b for b in at.button if b.label == "🔍 내 프로필로 공고 탐색").click().run()

    click_search()
    assert len(called) == 1
    click_search()
    assert not at.exception
    assert len(called) == 1, "한도를 넘으면 추천을 실행하지 않음"
    assert any("공고 추천" in w.value and "한도" in w.value for w in at.warning)
