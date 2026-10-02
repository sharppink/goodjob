"""
agents/resume_writer.py

LangGraph 노드: resume_writer

역할
----
job_requirements + retrieved_experiences + fit_feedback를 종합하여
해당 회사/공고에 최적화된 이력서 초안을 생성합니다.

결과는 state["resume_draft"] 에 저장됩니다.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.state import GoodJobState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 한국 IT 업계 최고의 이력서 작성 전문가입니다.
15년 경력으로 카카오, 네이버, 삼성, 쿠팡 등 대기업 합격 이력서를 수백 건 작성했습니다.

작성 원칙:
1. 채용공고의 키워드를 이력서에 자연스럽게 반영하여 ATS 통과율을 높입니다.
2. 모든 경험은 "동사 + 수치 + 결과" 형식으로 작성합니다. (예: FastAPI로 API 응답 시간 40% 개선)
3. 지원자가 실제로 보유한 경험만 사용하며 과장하지 않습니다.
4. 공고 언어(한국어/영어)에 맞춰 이력서 언어를 결정합니다.
5. 구조: 자기소개 → 핵심 기술 → 경력 → 프로젝트 → 학력/자격증
6. Markdown 형식으로 출력합니다.
"""

USER_PROMPT_TEMPLATE = """\
아래 정보를 바탕으로 맞춤 이력서를 작성해주세요.

### 지원 회사 및 직무
회사: {company_name}
직무: {job_title}

### 채용공고 요구사항
{job_requirements_text}

### 적합도 분석 결과
적합도 점수: {fit_score:.0%}
{fit_feedback}

### 지원자의 관련 경험 (프로필에서 검색됨)
{experiences_text}

### 작성 지시사항
1. 자기소개(3~4문장): 지원자의 강점을 {company_name}의 {job_title} 관점에서 어필
2. 핵심 기술: required_skills 우선 배치, 지원자가 보유한 것만 포함
3. 경력사항: 각 항목을 "동사 + 수치 + 결과" 형식으로 3~5개 bullet
4. 프로젝트: AI/기술 관련 프로젝트 강조
5. 누락된 필수 기술이 있다면 경험 섹션 하단에 "보완 계획" 항목 추가

Markdown 형식으로만 출력하세요.
"""


def build_resume_prompt(state: GoodJobState) -> tuple[str, str]:
    """프롬프트와 시스템 메시지를 반환합니다 (스트리밍 등 외부 호출용)."""
    job_requirements: dict = state.get("job_requirements") or {}
    retrieved_experiences: list[str] = state.get("retrieved_experiences") or []
    fit_score: float = state.get("fit_score") or 0.0
    fit_feedback: str = state.get("fit_feedback") or ""
    company_name: str = state.get("company_name") or "지원 회사"

    job_req_text = _format_requirements(job_requirements)
    experiences_text = _format_experiences(retrieved_experiences)
    job_title = job_requirements.get("job_title") or "개발자"

    prompt = USER_PROMPT_TEMPLATE.format(
        company_name=company_name,
        job_title=job_title,
        job_requirements_text=job_req_text,
        fit_score=fit_score,
        fit_feedback=fit_feedback,
        experiences_text=experiences_text,
    )
    return prompt, SYSTEM_PROMPT


def stream_resume_writer(state: GoodJobState):
    """이력서 초안을 토큰 단위로 스트리밍합니다 (Streamlit st.write_stream 용)."""
    from llm.openai_client import OpenAIClient
    prompt, system = build_resume_prompt(state)
    client = OpenAIClient()
    yield from client.stream(prompt=prompt, system=system, max_tokens=3000)


def resume_writer_node(state: GoodJobState) -> GoodJobState:
    """채용공고에 최적화된 이력서 초안을 생성합니다."""
    logger.info("[resume_writer] 이력서 생성 시작.")
    state["current_step"] = "resume_writer"
    errors: list[str] = state.get("errors") or []

    job_requirements: dict = state.get("job_requirements") or {}
    retrieved_experiences: list[str] = state.get("retrieved_experiences") or []
    fit_score: float = state.get("fit_score") or 0.0
    fit_feedback: str = state.get("fit_feedback") or ""
    company_name: str = state.get("company_name") or "지원 회사"

    job_req_text = _format_requirements(job_requirements)
    experiences_text = _format_experiences(retrieved_experiences)
    job_title = job_requirements.get("job_title") or "개발자"

    prompt = USER_PROMPT_TEMPLATE.format(
        company_name=company_name,
        job_title=job_title,
        job_requirements_text=job_req_text,
        fit_score=fit_score,
        fit_feedback=fit_feedback,
        experiences_text=experiences_text,
    )

    try:
        from llm.openai_client import OpenAIClient
        client = OpenAIClient()
        resume_draft = client.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=3000,
            temperature=0.4,
        )
    except Exception as exc:
        logger.error("[resume_writer] LLM 호출 실패: %s", exc)
        errors.append(f"resume_writer: LLM 오류 – {exc}")
        resume_draft = f"# 이력서 생성 실패\n\n오류: {exc}"

    logger.info("[resume_writer] 초안 생성 완료 (%d자).", len(resume_draft))
    state["resume_draft"] = resume_draft
    state["errors"] = errors
    return state


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _format_requirements(req: dict) -> str:
    lines = []
    if req.get("required_skills"):
        lines.append("필수 기술: " + ", ".join(req["required_skills"]))
    if req.get("preferred_skills"):
        lines.append("우대 기술: " + ", ".join(req["preferred_skills"]))
    if req.get("experience_years"):
        lines.append(f"최소 경력: {req['experience_years']}년")
    if req.get("responsibilities"):
        lines.append("주요 업무:")
        for r in req["responsibilities"][:6]:
            lines.append(f"  - {r}")
    if req.get("keywords"):
        lines.append("핵심 키워드: " + ", ".join(req["keywords"][:10]))
    return "\n".join(lines) or "(요구사항 없음)"


def _format_experiences(experiences: list[str]) -> str:
    if not experiences:
        return "(검색된 경험 없음 — 프로필을 먼저 등록해주세요)"
    return "\n\n---\n\n".join(
        f"[경험 {i+1}]\n{exp}" for i, exp in enumerate(experiences)
    )
