"""
agents/job_parser.py

LangGraph 노드: job_parser

역할
----
state["job_posting_raw"] 의 채용공고 텍스트를 분석해서
구조화된 job_requirements 딕셔너리를 추출합니다.

Structured Output 사용으로 JSON 파싱 실패 없음.
복잡도가 낮은 공고는 로컬 LLM으로 라우팅합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.state import GoodJobState
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

_router = LLMRouter()

SYSTEM_PROMPT = """\
당신은 한국 및 글로벌 IT 채용공고 분석 전문가입니다.
채용공고에서 구조화된 정보를 정확하게 추출하는 것이 역할입니다.
"""

USER_PROMPT_TEMPLATE = """\
아래 채용공고를 분석하여 요청된 항목을 추출하세요.

### 채용공고
{raw_text}
"""


# ------------------------------------------------------------------ #
# Structured Output 스키마                                             #
# ------------------------------------------------------------------ #

class JobRequirements(BaseModel):
    """채용공고 구조화 스키마 — OpenAI Structured Output 전용."""
    job_title: str = Field(description="채용 직무명")
    required_skills: list[str] = Field(description="필수 기술/자격 목록")
    preferred_skills: list[str] = Field(description="우대 기술/자격 목록")
    experience_years: int = Field(description="최소 경력 연수 (신입/미명시=0)")
    responsibilities: list[str] = Field(description="주요 업무 목록 (최대 7개)")
    company_culture: str = Field(description="회사 문화 및 근무 환경 요약 (2~3문장)")
    keywords: list[str] = Field(description="ATS 통과용 핵심 키워드 (10~15개)")
    location: str = Field(default="", description="근무 지역")
    remote_policy: str = Field(default="", description="재택/출근/하이브리드 여부")
    job_type: str = Field(default="정규직", description="고용 형태 (정규직/계약직/인턴)")
    salary_range: str = Field(default="", description="급여 범위 (명시된 경우)")


# ------------------------------------------------------------------ #
# Node function                                                        #
# ------------------------------------------------------------------ #

def job_parser_node(state: GoodJobState) -> GoodJobState:
    """채용공고 텍스트를 구조화된 요구사항으로 파싱합니다."""
    logger.info("[job_parser] 채용공고 파싱 시작.")
    state["current_step"] = "job_parser"
    errors: list[str] = state.get("errors") or []

    raw_text: str = state.get("job_posting_raw") or ""
    if not raw_text.strip():
        logger.warning("[job_parser] job_posting_raw 가 비어있습니다.")
        errors.append("job_parser: 채용공고 텍스트가 없습니다.")
        state["errors"] = errors
        state["job_requirements"] = _empty_requirements()
        return state

    trimmed = raw_text[:4000]
    complexity = _estimate_complexity(trimmed)
    llm_choice = _router.route(task_type="parse", complexity=complexity)
    logger.info("[job_parser] LLM 선택: %s (복잡도=%s)", llm_choice, complexity)

    prompt = USER_PROMPT_TEMPLATE.format(raw_text=trimmed)

    try:
        if llm_choice == "openai":
            from llm.openai_client import OpenAIClient
            client = OpenAIClient()
            parsed: JobRequirements = client.generate_structured(
                prompt=prompt,
                schema=JobRequirements,
                system=SYSTEM_PROMPT,
            )
            requirements = parsed.model_dump()
        else:
            # 로컬 LLM은 Structured Output 미지원 → JSON 텍스트 생성 후 파싱
            from llm.local_llm import LocalLLM
            import json, re
            local = LocalLLM()
            raw_resp = local.generate(prompt=prompt, system=SYSTEM_PROMPT)
            clean = re.sub(r"```(?:json)?", "", raw_resp).strip().rstrip("`").strip()
            data = json.loads(clean)
            requirements = _normalize(data)

    except Exception as exc:
        logger.error("[job_parser] LLM 호출 실패: %s", exc)
        errors.append(f"job_parser: LLM 오류 – {exc}")
        requirements = _empty_requirements()

    state["job_requirements"] = requirements
    state["errors"] = errors
    logger.info(
        "[job_parser] 추출 완료: 필수기술 %d개, 우대기술 %d개",
        len(requirements.get("required_skills", [])),
        len(requirements.get("preferred_skills", [])),
    )
    return state


# ------------------------------------------------------------------ #
# 공개 헬퍼 — job_recommender 등에서 재사용                            #
# ------------------------------------------------------------------ #

def parse_posting_structured(raw_text: str) -> dict[str, Any]:
    """
    단일 채용공고 텍스트를 Structured Output으로 파싱하여 dict 반환.
    job_recommender 등 외부 호출용.
    """
    trimmed = raw_text[:4000]
    prompt = USER_PROMPT_TEMPLATE.format(raw_text=trimmed)
    try:
        from llm.openai_client import OpenAIClient
        client = OpenAIClient()
        parsed: JobRequirements = client.generate_structured(
            prompt=prompt,
            schema=JobRequirements,
            system=SYSTEM_PROMPT,
        )
        return parsed.model_dump()
    except Exception as exc:
        logger.error("[job_parser] parse_posting_structured 실패: %s", exc)
        return _empty_requirements()


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _estimate_complexity(text: str) -> str:
    length = len(text)
    if length < 500:
        return "low"
    if length < 2000:
        return "medium"
    return "high"


def _normalize(data: dict) -> dict[str, Any]:
    """로컬 LLM JSON 응답을 표준 형태로 정규화합니다."""
    return {
        "job_title":        data.get("job_title", ""),
        "required_skills":  data.get("required_skills", []),
        "preferred_skills": data.get("preferred_skills", []),
        "experience_years": int(data.get("experience_years", 0) or 0),
        "responsibilities": data.get("responsibilities", []),
        "company_culture":  data.get("company_culture", ""),
        "keywords":         data.get("keywords", []),
        "location":         data.get("location", ""),
        "remote_policy":    data.get("remote_policy", ""),
        "job_type":         data.get("job_type", "정규직"),
        "salary_range":     data.get("salary_range", ""),
    }


def _empty_requirements() -> dict[str, Any]:
    return {
        "job_title": "", "required_skills": [], "preferred_skills": [],
        "experience_years": 0, "responsibilities": [], "company_culture": "",
        "keywords": [], "location": "", "remote_policy": "",
        "job_type": "정규직", "salary_range": "",
    }
