"""지원 현황 보드 — 이력서 자동 추가, 단계 이동(드래그 결과 반영), 계정 분리, 화면."""

from __future__ import annotations

import pytest

from tracker import applications as ap


@pytest.fixture(autouse=True)
def clean():
    from rag.vectorstore import set_current_user
    from config.settings import settings
    from pathlib import Path

    path = Path(settings.CHROMA_PERSIST_DIR) / "applications.json"
    path.unlink(missing_ok=True)
    yield
    set_current_user(None)
    path.unlink(missing_ok=True)


def _login(email):
    from rag.vectorstore import set_current_user
    set_current_user(email)


# ------------------------------------------------------------------ #
# 저장                                                                 #
# ------------------------------------------------------------------ #

def test_save_draft_goes_to_first_stage_and_updates_same_posting():
    first = ap.save_draft("당근", "백엔드 엔지니어", "# 이력서 v1", 0.7, "공고 원문")
    assert first["stage"] == "작성 중"

    ap.set_stages({first["id"]: "지원 완료"})
    again = ap.save_draft("당근", "백엔드 엔지니어", "# 이력서 v2", 0.8, "공고 원문")
    apps = ap.list_applications()
    assert len(apps) == 1, "같은 공고로 다시 만들면 새 카드가 아니라 갱신"
    assert again["id"] == first["id"]
    assert apps[0]["resume"] == "# 이력서 v2" and apps[0]["fit_score"] == 0.8
    assert apps[0]["stage"] == "지원 완료", "갱신해도 옮겨 둔 단계는 유지"

    ap.save_draft("당근", "백엔드 엔지니어", "# 다른 공고", 0.5, "다른 공고 원문")
    assert len(ap.list_applications()) == 2


def test_set_stages_ignores_unknown_stage_and_delete():
    app = ap.save_draft("토스", "", "# r", None, "p")
    ap.set_stages({app["id"]: "없는 단계"})
    assert ap.list_applications()[0]["stage"] == "작성 중"
    ap.delete_application(app["id"])
    assert ap.list_applications() == []


def test_accounts_are_separated():
    _login("a@example.com")
    ap.save_draft("A사", "", "# a", 0.6, "pa")
    _login("b@example.com")
    assert ap.list_applications() == []
    ap.save_draft("B사", "", "# b", 0.6, "pb")
    _login("a@example.com")
    assert [a["company"] for a in ap.list_applications()] == ["A사"]


def test_login_required_without_user_raises():
    from rag import vectorstore
    vectorstore.require_login(True)
    try:
        with pytest.raises(RuntimeError):
            ap.list_applications()
    finally:
        vectorstore.require_login(False)


# ------------------------------------------------------------------ #
# 보드 형식 (드래그 결과 → 단계 변경)                                    #
# ------------------------------------------------------------------ #

def test_board_columns_and_stage_changes():
    a = ap.save_draft("당근", "백엔드", "# a", 0.72, "pa")
    b = ap.save_draft("토스", "", "# b", None, "pb")
    apps = ap.list_applications()

    cols = ap.board_columns(apps)
    assert [c["header"] for c in cols] == ap.STAGES
    assert sorted(cols[0]["items"]) == sorted([f"당근 · 백엔드 72% [#{a['id'][:6]}]", f"토스 [#{b['id'][:6]}]"])

    # 사용자가 당근 카드를 '면접' 칸으로 끌어다 놓은 결과
    dragged = [dict(c, items=list(c["items"])) for c in cols]
    label = next(x for x in dragged[0]["items"] if x.startswith("당근"))
    dragged[0]["items"].remove(label)
    dragged[ap.STAGES.index("면접")]["items"].append(label)
    assert ap.stage_changes(apps, dragged) == {a["id"]: "면접"}
    assert ap.stage_changes(apps, cols) == {}, "움직이지 않으면 변경 없음"


# ------------------------------------------------------------------ #
# 화면                                                                 #
# ------------------------------------------------------------------ #

def _app():
    from streamlit.testing.v1 import AppTest
    from tests.test_app_login import APP, NO_AUTH_SECRETS

    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets = NO_AUTH_SECRETS  # type: ignore[assignment]
    return at.run()


def test_board_page_empty_and_with_cards():
    at = _app()
    at.sidebar.radio(key="page_radio").set_value("지원 현황").run()
    assert not at.exception
    assert any("아직 카드가 없습니다" in i.value for i in at.info)

    ap.save_draft("당근", "백엔드", "# 당근 이력서", 0.7, "pa")
    at.run()
    assert not at.exception
    assert any("총 1개" in c.value and "작성 중 1" in c.value for c in at.caption)
    assert at.selectbox(key="board_pick").options[0].startswith("[작성 중] 당근")

    app_id = ap.list_applications()[0]["id"]
    at.checkbox(key=f"board_del_ok_{app_id}").check().run()
    at.button(key=f"board_del_{app_id}").click().run()
    assert ap.list_applications() == []


def test_pipeline_with_resume_adds_card(monkeypatch, loaded_profile):
    """분석 & 이력서에서 이력서가 만들어지면 '작성 중' 에 자동으로 들어감."""
    import agents.fit_analyzer
    import agents.interview_coach
    import agents.job_parser
    import agents.rag_retriever
    import agents.resume_writer
    import agents.reviewer
    from tests.conftest import SAMPLE_REQUIREMENTS

    def parser(state):
        state["job_requirements"] = dict(SAMPLE_REQUIREMENTS)
        return state

    def fit(state):
        state["fit_score"] = 0.8
        return state

    monkeypatch.setattr(agents.job_parser, "job_parser_node", parser)
    monkeypatch.setattr(agents.rag_retriever, "rag_retriever_node", lambda s: {**s, "retrieved_experiences": []})
    monkeypatch.setattr(agents.fit_analyzer, "fit_analyzer_node", fit)
    monkeypatch.setattr(agents.resume_writer, "stream_resume_writer", lambda s: iter(["# 초안"]))
    monkeypatch.setattr(agents.reviewer, "stream_reviewer", lambda s: iter(["# 최종 이력서"]))
    monkeypatch.setattr(agents.interview_coach, "interview_coach_node", lambda s: s)

    at = _app()
    at.session_state["company_name"] = "테스트컴퍼니"
    at.session_state["job_posting_text"] = "백엔드 개발자 채용 공고"
    at.sidebar.radio(key="page_radio").set_value("분석 & 이력서").run()
    next(b for b in at.button if b.label == "▶ 파이프라인 실행").click().run()

    assert not at.exception
    apps = ap.list_applications()
    assert len(apps) == 1
    assert apps[0]["company"] == "테스트컴퍼니" and apps[0]["stage"] == "작성 중"
    assert apps[0]["job_title"] == SAMPLE_REQUIREMENTS["job_title"]
    assert apps[0]["resume"] == "# 최종 이력서"
