"""agents/* 노드 단위 테스트 (LLM 은 FakeChatClient)."""

from __future__ import annotations

import pytest

from agents.coverletter_writer import count_chars, write_answer
from agents.fit_analyzer import fit_analyzer_node
from agents.interview_coach import interview_coach_node
from agents.rag_retriever import MAX_RESULTS, SMALL_PROFILE_CHUNKS, _build_queries, rag_retriever_node
from agents.resume_writer import build_resume_prompt, resume_writer_node, strip_markdown_fence
from agents.reviewer import reviewer_node
from agents.star_converter import MISSING_MARK, convert_to_star, star_to_text
from conftest import INTERVIEW_SET, SAMPLE_REQUIREMENTS, make_fit


# ------------------------------------------------------------------ #
# fit_analyzer                                                         #
# ------------------------------------------------------------------ #

def test_fit_analyzer_without_requirements_skips_llm(fake_llm):
    state = fit_analyzer_node({"retrieved_experiences": ["x"]})
    assert state["fit_score"] == 0.0
    assert state["errors"]
    assert fake_llm.calls == []


@pytest.mark.parametrize("raw, expected", [(1.7, 1.0), (-0.2, 0.0), (0.72, 0.72)])
def test_fit_analyzer_clamps_score(fake_llm, raw, expected):
    fake_llm.structured["FitAnalysis"] = make_fit(raw)
    state = fit_analyzer_node({"job_requirements": SAMPLE_REQUIREMENTS,
                               "retrieved_experiences": ["FastAPI 4년"]})
    assert state["fit_score"] == expected
    assert state["matched_skills"] == ["Python", "FastAPI"]
    assert state["missing_skills"] == ["Kubernetes"]


def test_fit_analyzer_prompt_contains_requirements_and_experiences(fake_llm):
    fake_llm.structured["FitAnalysis"] = make_fit(0.5)
    fit_analyzer_node({"job_requirements": SAMPLE_REQUIREMENTS,
                       "retrieved_experiences": ["FastAPI 서버 4년 운영"]})
    prompt = fake_llm.calls[0][2]
    assert "Required skills: Python, FastAPI" in prompt
    assert "Minimum experience: 3 years" in prompt
    assert "[1] FastAPI 서버 4년 운영" in prompt


def test_fit_analyzer_llm_error_is_non_fatal(fake_llm):
    fake_llm.structured["FitAnalysis"] = RuntimeError("timeout")
    state = fit_analyzer_node({"job_requirements": SAMPLE_REQUIREMENTS, "retrieved_experiences": []})
    assert state["fit_score"] == 0.0
    assert any("timeout" in e for e in state["errors"])


# ------------------------------------------------------------------ #
# resume_writer / reviewer                                             #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("text, expected", [
    ("```markdown\n# 이력서\n내용\n```", "# 이력서\n내용"),
    ("```\n# 이력서\n```", "# 이력서"),
    ("# 이력서\n```python\ncode\n```\n끝", "# 이력서\n```python\ncode\n```\n끝"),
    ("  # 그대로  ", "# 그대로"),
])
def test_strip_markdown_fence(text, expected):
    assert strip_markdown_fence(text) == expected


def test_resume_prompt_lists_matched_and_missing_skills():
    prompt, _ = build_resume_prompt({
        "company_name": "테스트컴퍼니", "job_requirements": SAMPLE_REQUIREMENTS,
        "fit_score": 0.8, "matched_skills": ["Python"], "missing_skills": ["Kubernetes"],
        "retrieved_experiences": ["경험 A"],
    })
    assert "근거가 확인된 기술: Python" in prompt
    assert "근거가 없는 기술: Kubernetes" in prompt
    assert "적합도 점수: 80%" in prompt
    assert "[경험 1]\n경험 A" in prompt


def test_resume_writer_strips_fence(fake_llm):
    fake_llm.texts = ["```markdown\n# 이력서 초안\n```"]
    state = resume_writer_node({"job_requirements": SAMPLE_REQUIREMENTS})
    assert state["resume_draft"] == "# 이력서 초안"


def test_reviewer_keeps_draft_when_llm_fails(fake_llm):
    fake_llm.texts = [RuntimeError("500")]
    state = reviewer_node({"resume_draft": "# 초안", "job_requirements": SAMPLE_REQUIREMENTS})
    assert state["resume_final"] == "# 초안"
    assert any("500" in e for e in state["errors"])


def test_reviewer_skips_empty_draft(fake_llm):
    state = reviewer_node({"resume_draft": "  "})
    assert fake_llm.calls == []
    assert state["errors"]


def test_reviewer_passes_missing_skills_to_prompt(fake_llm):
    reviewer_node({"resume_draft": "# 초안", "missing_skills": ["Kubernetes"]})
    assert "Kubernetes" in fake_llm.calls[0][2]


# ------------------------------------------------------------------ #
# interview_coach                                                      #
# ------------------------------------------------------------------ #

def test_interview_coach_respects_opt_out(fake_llm):
    state = interview_coach_node({"generate_interview": False, "resume_final": "# 이력서"})
    assert "interview_questions" not in state
    assert fake_llm.calls == []


def test_interview_coach_generates_questions(fake_llm):
    fake_llm.structured["InterviewQuestionSet"] = INTERVIEW_SET
    state = interview_coach_node({"resume_final": "# 이력서", "job_requirements": SAMPLE_REQUIREMENTS})
    assert [q["category"] for q in state["interview_questions"]] == ["기술", "약점 보완"]


def test_interview_coach_without_resume_returns_empty(fake_llm):
    state = interview_coach_node({})
    assert state["interview_questions"] == []
    assert fake_llm.calls == []


# ------------------------------------------------------------------ #
# rag_retriever                                                        #
# ------------------------------------------------------------------ #

def test_build_queries_covers_each_requirement_field():
    queries = _build_queries(SAMPLE_REQUIREMENTS)
    assert queries[0].startswith("기술 경험: Python, FastAPI")
    assert any(q.startswith("업무 경험:") for q in queries)
    assert any(q.startswith("우대 기술:") for q in queries)
    assert _build_queries({}) == ["소프트웨어 개발 경험"]


def test_retriever_with_empty_store_reports_error():
    state = rag_retriever_node({"job_requirements": SAMPLE_REQUIREMENTS})
    assert state["retrieved_experiences"] == []
    assert any("프로필" in e for e in state["errors"])


def test_retriever_small_profile_returns_all_chunks(loaded_profile):
    """작은 프로필은 검색으로 거르지 않고 전부 사용 (#008)."""
    total = loaded_profile.get_stored_count()
    assert 0 < total <= SMALL_PROFILE_CHUNKS
    state = rag_retriever_node({"job_requirements": SAMPLE_REQUIREMENTS})
    assert len(state["retrieved_experiences"]) == total


def test_retriever_large_profile_searches_and_dedupes():
    from rag.vectorstore import VectorStore

    vs = VectorStore()
    docs = [f"기타 경험 {i} 마케팅 영업" for i in range(SMALL_PROFILE_CHUNKS + 5)]
    docs.append("기술 경험: Python FastAPI 백엔드 서버 개발")
    vs.add_documents(docs)

    state = rag_retriever_node({"job_requirements": SAMPLE_REQUIREMENTS})
    found = state["retrieved_experiences"]
    assert 0 < len(found) <= MAX_RESULTS
    assert len(found) == len(set(found)), "쿼리 간 중복 결과는 제거돼야 함"
    assert found[0] == "기술 경험: Python FastAPI 백엔드 서버 개발"


# ------------------------------------------------------------------ #
# star_converter                                                       #
# ------------------------------------------------------------------ #

def test_convert_to_star_empty_input_skips_llm(fake_llm):
    assert convert_to_star("   ") == []
    assert fake_llm.calls == []


def test_star_to_text_drops_missing_fields():
    text = star_to_text({
        "title": "API 성능 개선", "situation": "응답 지연", "task": MISSING_MARK,
        "action": "쿼리 튜닝", "result": MISSING_MARK, "skills": ["PostgreSQL"],
    })
    assert "과제" not in text and "결과" not in text
    assert "- 행동: 쿼리 튜닝" in text
    assert "- 사용 기술: PostgreSQL" in text


def test_star_experiences_are_stored_one_chunk_each(fake_llm, loaded_profile):
    fake_llm.structured["STARResult"] = {"experiences": [
        {"title": "캐시 도입", "situation": "느린 조회", "task": "개선", "action": "Redis",
         "result": MISSING_MARK, "skills": ["Redis"], "missing_info": ["개선 전후 응답시간은?"]},
        {"title": "", "situation": "", "task": "", "action": "", "result": "",
         "skills": [], "missing_info": []},
    ]}
    before = loaded_profile.get_stored_count()
    stars = convert_to_star("Redis 캐시 넣어서 빨라짐")
    assert stars[0]["missing_info"] == ["개선 전후 응답시간은?"]
    assert loaded_profile.load_star_experiences(stars) == 1, "제목 없는 항목은 저장하지 않음"
    assert loaded_profile.get_stored_count() == before + 1


# ------------------------------------------------------------------ #
# coverletter_writer                                                   #
# ------------------------------------------------------------------ #

def _answer(text: str) -> dict:
    return {"question_intent": "의도", "key_message": "핵심", "used_experiences": ["A"], "answer": text}


def test_count_chars():
    assert count_chars("가 나\n다") == (5, 3)


def test_coverletter_within_limit_needs_no_retry(fake_llm):
    fake_llm.structured["CoverLetterAnswer"] = _answer("가" * 90)
    result = write_answer("지원 동기는?", char_limit=100)
    assert result["within_limit"] is True
    assert result["retries"] == 0
    assert fake_llm.count("generate") == 0


def test_coverletter_shortens_when_over_limit(fake_llm):
    """LLM 이 글자수를 넘기면 코드가 세서 줄여 쓰기를 다시 요청."""
    fake_llm.structured["CoverLetterAnswer"] = _answer("가" * 150)
    fake_llm.texts = ["가" * 120, "가" * 95]
    result = write_answer("지원 동기는?", char_limit=100)
    assert result["retries"] == 2
    assert result["chars_with_spaces"] == 95
    assert result["within_limit"] is True


def test_coverletter_reports_failure_after_max_retries(fake_llm):
    fake_llm.structured["CoverLetterAnswer"] = _answer("가" * 150)
    fake_llm.texts = ["가" * 140, "가" * 130, "가" * 10]
    result = write_answer("지원 동기는?", char_limit=100)
    assert result["retries"] == 2
    assert result["within_limit"] is False
    assert fake_llm.count("generate") == 2


def test_coverletter_uses_stored_experiences(fake_llm, loaded_profile):
    fake_llm.structured["CoverLetterAnswer"] = _answer("답변")
    write_answer("가장 도전적인 경험은?", char_limit=500)
    prompt = fake_llm.calls[0][2]
    assert "FastAPI 서버 개발" in prompt
    assert "(등록된 경험 없음)" not in prompt
