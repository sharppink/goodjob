"""
agents/interview_coach.py

LangGraph 노드: interview_coach

역할
----
최종 이력서(resume_final) + 채용공고 요구사항 + 부족한 기술(missing_skills)을 바탕으로
면접관이 물어볼 가능성이 높은 질문과 답변 전략을 생성합니다.

질문 구성
---------
- 기술     : 공고 필수 기술을 이력서 경험과 엮은 심화 질문
- 경험     : 이력서에 적힌 성과·수치의 검증 질문 ("40%는 어떻게 측정했나요?")
- 약점 보완 : missing_skills 를 파고드는 질문과 방어 전략
- 컬처핏   : 회사 문화·협업 관련 질문

state["generate_interview"] 가 False 이면 건너뜁니다.
결과는 state["interview_questions"] 에 저장됩니다.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from agents.state import GoodJobState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 한국 IT 기업의 시니어 개발자 면접관이자 면접 코치입니다.
지원자의 이력서와 채용공고를 보고, 실제 면접에서 나올 가능성이 높은 질문을 뽑고
각 질문에 대한 답변 전략을 코칭합니다.

규칙:
1. 질문은 이력서에 실제로 적힌 경험·수치를 구체적으로 인용합니다.
2. answer_tip 은 "무엇을 말하라"는 방향만 제시하고, 이력서에 없는 경험을 지어내 답하라고 하지 않습니다.
3. 한국어로 작성합니다.
"""

USER_PROMPT_TEMPLATE = """\
### 지원 회사 / 직무
{company_name} / {job_title}

### 채용공고 요구사항
{requirements_text}

### 지원자가 근거를 보이지 못한 기술 (약점)
{missing_skills}

### 지원자 최종 이력서
{resume}

### 작성 지시
면접 예상 질문 {count}개를 만드세요. 카테고리 분배:
- 기술 2~3개, 경험 2~3개, 약점 보완 1~2개 (약점이 없으면 0개), 컬처핏 1개
중요도가 높은 질문부터 나열하세요.
"""

DEFAULT_QUESTION_COUNT = 7


class InterviewQuestion(BaseModel):
    category: Literal["기술", "경험", "약점 보완", "컬처핏"]
    question: str = Field(description="면접관이 실제로 할 법한 질문 문장")
    intent: str = Field(description="면접관이 이 질문으로 확인하려는 것 (1문장)")
    answer_tip: str = Field(description="답변 전략 힌트 (2~3문장, 언급할 포인트 중심)")


class InterviewQuestionSet(BaseModel):
    questions: list[InterviewQuestion]


def interview_coach_node(state: GoodJobState) -> GoodJobState:
    """최종 이력서 기반 면접 예상 질문과 답변 전략을 생성합니다."""
    state["current_step"] = "interview_coach"
    errors: list[str] = state.get("errors") or []

    if state.get("generate_interview") is False:
        logger.info("[interview_coach] generate_interview=False — 건너뜀.")
        return state

    resume = (state.get("resume_final") or state.get("resume_draft") or "").strip()
    if not resume:
        logger.warning("[interview_coach] 이력서가 없어 질문 생성을 건너뜁니다.")
        state["interview_questions"] = []
        return state

    logger.info("[interview_coach] 면접 질문 생성 시작.")
    try:
        state["interview_questions"] = generate_interview_questions(state)
    except Exception as exc:  # noqa: BLE001
        logger.error("[interview_coach] LLM 호출 실패: %s", exc)
        errors.append(f"interview_coach: LLM 오류 – {exc}")
        state["interview_questions"] = []

    state["errors"] = errors
    return state


def generate_interview_questions(
    state: GoodJobState, count: int = DEFAULT_QUESTION_COUNT
) -> list[dict]:
    """state 정보로 면접 질문을 생성해 dict 리스트로 반환합니다 (Streamlit 직접 호출용)."""
    from llm.factory import get_chat_client

    req: dict = state.get("job_requirements") or {}
    prompt = USER_PROMPT_TEMPLATE.format(
        company_name=state.get("company_name") or "지원 회사",
        job_title=req.get("job_title") or "개발자",
        requirements_text=_format_requirements(req),
        missing_skills=", ".join(state.get("missing_skills") or []) or "(없음)",
        resume=(state.get("resume_final") or state.get("resume_draft") or "")[:6000],
        count=count,
    )
    result: InterviewQuestionSet = get_chat_client().generate_structured(
        prompt=prompt, schema=InterviewQuestionSet, system=SYSTEM_PROMPT, temperature=0.4,
    )
    questions = [q.model_dump() for q in result.questions]
    logger.info("[interview_coach] 질문 %d개 생성.", len(questions))
    return questions


def _format_requirements(req: dict) -> str:
    parts = []
    if req.get("required_skills"):
        parts.append("필수: " + ", ".join(req["required_skills"]))
    if req.get("preferred_skills"):
        parts.append("우대: " + ", ".join(req["preferred_skills"]))
    if req.get("responsibilities"):
        parts.append("주요 업무: " + " / ".join(req["responsibilities"][:5]))
    if req.get("company_culture"):
        parts.append("회사 문화: " + req["company_culture"])
    return "\n".join(parts) or "(요구사항 없음)"
