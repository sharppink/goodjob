"""agents/graph.py — 전체 LangGraph 파이프라인 (LLM 은 가짜, 벡터 DB 는 임시 Chroma)."""

from __future__ import annotations

from agents.fit_analyzer import LOW_FIT_THRESHOLD
from agents.graph import _route_after_fit_analysis, app
from conftest import INTERVIEW_SET, SAMPLE_POSTING, SAMPLE_REQUIREMENTS, make_fit


def _initial_state(**overrides):
    state = {"company_name": "테스트컴퍼니", "job_posting_raw": SAMPLE_POSTING,
             "messages": [], "errors": [], "current_step": "start"}
    state.update(overrides)
    return state


def test_route_after_fit_analysis_threshold():
    assert _route_after_fit_analysis({"fit_score": LOW_FIT_THRESHOLD - 0.01}) == "__end__"
    assert _route_after_fit_analysis({"fit_score": LOW_FIT_THRESHOLD}) == "resume_writer"
    assert _route_after_fit_analysis({}) == "__end__"


def test_full_pipeline_generates_resume_and_interview(fake_llm, loaded_profile):
    fake_llm.structured["JobRequirements"] = SAMPLE_REQUIREMENTS
    fake_llm.structured["FitAnalysis"] = make_fit(0.8)
    fake_llm.structured["InterviewQuestionSet"] = INTERVIEW_SET
    fake_llm.texts = ["# 초안", "```markdown\n# 최종본\n```"]

    final = app.invoke(_initial_state())

    assert final["errors"] == []
    assert final["fit_score"] == 0.8
    assert final["retrieved_experiences"], "임시 DB 의 샘플 프로필이 검색돼야 함"
    assert final["resume_draft"] == "# 초안"
    assert final["resume_final"] == "# 최종본"
    assert len(final["interview_questions"]) == 2
    assert final["current_step"] == "interview_coach"


def test_low_fit_ends_before_resume(fake_llm, loaded_profile):
    fake_llm.structured["JobRequirements"] = SAMPLE_REQUIREMENTS
    fake_llm.structured["FitAnalysis"] = make_fit(0.1)

    final = app.invoke(_initial_state())

    assert final["current_step"] == "fit_analyzer"
    assert not final.get("resume_draft")
    assert fake_llm.count("generate") == 0, "적합도가 낮으면 이력서 LLM 호출이 없어야 함"


def test_interview_opt_out_skips_question_generation(fake_llm, loaded_profile):
    fake_llm.structured["JobRequirements"] = SAMPLE_REQUIREMENTS
    fake_llm.structured["FitAnalysis"] = make_fit(0.8)

    final = app.invoke(_initial_state(generate_interview=False))

    assert final["resume_final"]
    assert not final.get("interview_questions")
    assert fake_llm.count("structured", "InterviewQuestionSet") == 0
