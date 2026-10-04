"""
agents/reviewer.py

LangGraph 노드: reviewer

역할
----
resume_draft를 자기검토하여 개선된 resume_final을 생성합니다.

검토 체크리스트
---------------
1. 키워드 매칭  : 필수 기술이 이력서에 자연스럽게 포함됐는가
2. 문체/어조    : 능동적이고 자신감 있는 표현인가
3. 수치화       : 경험의 60% 이상에 수치(숫자, %)가 있는가
4. 구조/포맷    : ATS 친화적 Markdown, 일관된 구조인가
5. 일관성       : 날짜, 회사명, 직무명이 논리적으로 일치하는가
6. 한국어 완성도: 맞춤법, 어색한 표현 수정 (한국어 공고인 경우)

결과는 state["resume_final"] 에 저장됩니다.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.state import GoodJobState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 엄격한 이력서 교정 전문가이자 ATS 최적화 전문가입니다.
이력서 초안을 검토하고 완성도 높은 최종본을 출력합니다.
개선된 Markdown 이력서만 출력하세요. 설명이나 부연은 불필요합니다.
전체를 코드블록(```)으로 감싸지 마세요.
초안에 없는 사실(회사, 기간, 수치, 보유 기술)을 새로 추가하지 마세요.
초안에 있더라도 "지원자 경험 원문"에 근거가 없는 문장은 삭제하세요.
"""

REVIEW_PROMPT_TEMPLATE = """\
### 지원자 경험 원문 (사실 확인 기준 — 이력서의 모든 사실은 여기에 근거가 있어야 함)
{experiences_text}

### 채용공고 요구사항 (키워드 참조용 — 지원자의 경험이 아님)
{job_requirements_text}

### 지원자가 보유 근거가 없는 기술 (핵심 기술·경력에 추가 금지, "보완 계획"에만 언급 가능)
{missing_skills}

### 검토할 이력서 초안
{resume_draft}

### 검토 체크리스트 — 아래 항목을 모두 수정하세요:

0. 사실 검증 (가장 먼저)
   - 경력 bullet: 해당 회사의 경력 원문에 적힌 업무·성과가 아니면 삭제
     · 보유 기술 목록에만 있는 기술로 만든 업무 (예: 기술 목록에 Docker 만 있는데 "Docker 로 배포 자동화")
     · 채용공고의 업무 문구를 옮겨 온 것
     · 원문에서 "프로젝트"인 경험을 회사 경력에 넣은 것 (프로젝트 섹션으로 옮김)
   - 자기소개·프로젝트: 원문에 없는 경험·규모·표현(예: "다양한 회사에서")은 삭제하거나 원문대로 고침
   - 삭제해서 bullet 이 줄어도 괜찮음. 개수를 맞추려고 새 내용을 만들지 말 것

1. 키워드 매칭
   - 초안에 이미 있는 경험 중 공고 키워드와 관련된 표현을 공고 용어로 맞춤
     (예: 초안에 "Postgres" → 공고 용어 "PostgreSQL")
   - 보유 근거가 없는 기술은 핵심 기술·경력에 추가하지 않음

2. 문체/어조
   - 수동태 → 능동태 변환 (예: "담당했음" → "주도함")
   - 약한 표현 → 강한 동사 (예: "참여" → "리드", "개발에 기여" → "개발")

3. 수치화
   - 초안에 이미 있는 수치는 유지하고 문장 앞쪽으로 배치
   - 초안에 없는 숫자는 절대 새로 만들지 말 것 (사실이 아닌 수치는 면접에서 치명적)
   - 수치가 없는 항목은 원문에 적힌 범위 안에서만 구체적으로 다듬음 (원문에 없는 원인·방법·결과 추가 금지)

4. 구조/포맷
   - ## 헤더, 불릿 포인트 일관성 확인
   - 각 경력 항목: 회사명 / 기간 / 직무 형식 통일

5. 일관성
   - 중복 내용 제거
   - 날짜 순서 오류 수정 (최신순)

6. 마무리
   - 전체 분량: A4 1~2페이지 분량으로 조정
   - 지원 직무와 무관한 내용 제거

개선된 최종 이력서를 Markdown으로만 출력하세요.
"""


def build_review_prompt(state: GoodJobState) -> tuple[str, str]:
    """프롬프트와 시스템 메시지를 반환합니다 (스트리밍 등 외부 호출용)."""
    # 초안을 원문과 대조해야 지어낸 업무를 걸러낼 수 있음 — 예전에는 원문을 주지 않아
    # 초안의 허위 bullet 이 그대로 최종본에 남았음 (PROJECT_DOCS #025)
    from agents.resume_writer import _format_experiences

    resume_draft: str = state.get("resume_draft") or ""
    job_requirements: dict = state.get("job_requirements") or {}
    prompt = REVIEW_PROMPT_TEMPLATE.format(
        experiences_text=_format_experiences(state.get("retrieved_experiences") or []),
        job_requirements_text=_format_requirements(job_requirements),
        missing_skills=", ".join(state.get("missing_skills") or []) or "(없음)",
        resume_draft=resume_draft,
    )
    return prompt, SYSTEM_PROMPT


def stream_reviewer(state: GoodJobState):
    """최종 이력서를 토큰 단위로 스트리밍합니다 (Streamlit st.write_stream 용)."""
    from llm.factory import get_chat_client
    if not (state.get("resume_draft") or "").strip():
        return
    prompt, system = build_review_prompt(state)
    client = get_chat_client()
    yield from client.stream(prompt=prompt, system=system, max_tokens=3000)


def reviewer_node(state: GoodJobState) -> GoodJobState:
    """이력서 초안을 검토하고 최종본을 생성합니다."""
    logger.info("[reviewer] 이력서 검토 시작.")
    state["current_step"] = "reviewer"
    errors: list[str] = state.get("errors") or []

    resume_draft: Optional[str] = state.get("resume_draft") or ""

    if not resume_draft.strip():
        logger.warning("[reviewer] resume_draft 가 비어있습니다.")
        errors.append("reviewer: 검토할 이력서 초안이 없습니다.")
        state["errors"] = errors
        state["resume_final"] = resume_draft
        return state

    prompt, system = build_review_prompt(state)

    try:
        from llm.factory import get_chat_client
        client = get_chat_client()
        resume_final = client.generate(
            prompt=prompt,
            system=system,
            max_tokens=3000,
            temperature=0.2,  # 낮은 temperature → 일관된 교정
        )
    except Exception as exc:
        logger.error("[reviewer] LLM 호출 실패: %s", exc)
        errors.append(f"reviewer: LLM 오류 – {exc}")
        resume_final = resume_draft  # 실패 시 초안을 그대로 사용

    from agents.resume_writer import strip_markdown_fence
    resume_final = strip_markdown_fence(resume_final)
    logger.info("[reviewer] 최종 이력서 완성 (%d자).", len(resume_final))
    state["resume_final"] = resume_final
    state["errors"] = errors
    return state


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _format_requirements(req: dict) -> str:
    parts = []
    if req.get("required_skills"):
        parts.append("필수: " + ", ".join(req["required_skills"]))
    if req.get("preferred_skills"):
        parts.append("우대: " + ", ".join(req["preferred_skills"]))
    if req.get("keywords"):
        parts.append("키워드: " + ", ".join(req["keywords"][:12]))
    return "\n".join(parts) or "(요구사항 없음)"
