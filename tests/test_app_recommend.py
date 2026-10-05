"""Streamlit '공고 추천' 화면 — 프로필 기반 탐색이 기본이고, 키워드 입력으로도 바꿀 수 있음."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from tests.test_app_login import APP, NO_AUTH_SECRETS


def _open_recommend_page(monkeypatch, fake_node) -> AppTest:
    import agents.job_recommender

    monkeypatch.setattr(agents.job_recommender, "job_recommender_node", fake_node)
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets = NO_AUTH_SECRETS  # type: ignore[assignment]
    at.run()
    at.sidebar.radio(key="page_radio").set_value("공고 추천").run()
    return at


def test_profile_mode_runs_without_keyword(monkeypatch, loaded_profile):
    seen: list[dict] = []

    def fake_node(state):
        seen.append(dict(state))
        return {
            "recommendation_queries": ["Python 백엔드", "FastAPI 개발자"],
            "ranked_matches": [{
                "rank": 1, "company": "테스트컴퍼니", "title": "$100k-$150k 백엔드 개발자", "url": "https://x",
                "fit_score": 0.8, "fit_feedback": "좋음", "requirements": {"required_skills": ["Python"]},
            }],
            "errors": [],
        }

    at = _open_recommend_page(monkeypatch, fake_node)
    assert not at.exception
    assert at.header[0].value == "🔎 내 프로필에 맞는 공고 추천"
    assert not at.text_input, "프로필 모드에서는 키워드 입력칸이 없어야 함"

    next(b for b in at.button if b.label == "🔍 내 프로필로 공고 탐색").click().run()

    assert not at.exception
    assert seen[0]["recommendation_query"] == ""
    assert any("Python 백엔드" in c.value and "FastAPI 개발자" in c.value for c in at.caption)
    assert "내 프로필에 맞는 공고 1개 추천" in at.success[0].value
    # #034: 제목의 '$' 가 수식으로 해석되지 않게 이스케이프
    assert any(r"[\$100k-\$150k 백엔드 개발자](https://x)" in m.value for m in at.markdown)


def test_keyword_mode_passes_keyword(monkeypatch, loaded_profile):
    seen: list[dict] = []

    def fake_node(state):
        seen.append(dict(state))
        return {"recommendation_queries": [state["recommendation_query"]], "ranked_matches": [], "errors": []}

    at = _open_recommend_page(monkeypatch, fake_node)
    mode = next(r for r in at.radio if r.label == "검색 방식")
    mode.set_value("keyword").run()
    at.text_input[0].input("AI 엔지니어").run()
    next(b for b in at.button if b.label == "🔍 공고 탐색").click().run()

    assert not at.exception
    assert seen[0]["recommendation_query"] == "AI 엔지니어"
    assert any("찾지 못했습니다" in i.value for i in at.info)
