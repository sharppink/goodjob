"""
agents/star_converter.py

자유 텍스트로 적은 경험을 STAR(Situation / Task / Action / Result) 구조로 변환합니다.

- 한 번에 여러 경험을 입력해도 경험별로 나눠서 변환합니다.
- 원문에 없는 사실(수치, 기간, 성과)은 만들어내지 않습니다.
  빠진 정보는 ``missing_info`` 에 "사용자에게 물어볼 질문" 형태로 남깁니다.
- 변환 결과는 ``ProfileLoader.load_star_experiences()`` 로 벡터 DB에 저장하면
  RAG 검색 시 상황·행동·결과가 한 청크에 묶여 검색 품질이 올라갑니다.

사용 예
-------
    from agents.star_converter import convert_to_star, star_to_text
    stars = convert_to_star("FastAPI 서버 개발해서 성능 개선")
    for s in stars:
        print(star_to_text(s))
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MISSING_MARK = "(보완 필요)"

SYSTEM_PROMPT = f"""\
당신은 IT 개발자 커리어 코치입니다.
지원자가 자유롭게 적은 경험을 STAR 기법(Situation, Task, Action, Result)으로 구조화합니다.

규칙:
1. 원문에 적힌 사실만 사용합니다. 수치·기간·회사명·성과를 지어내지 않습니다.
2. 원문에서 알 수 없는 STAR 항목은 "{MISSING_MARK}" 라고만 적습니다.
3. 빠진 정보는 missing_info 에 지원자에게 되물을 구체적인 질문으로 적습니다.
   (예: "응답 속도가 개선 전후로 각각 몇 ms 였나요?")
4. 서로 다른 경험이 여러 개 섞여 있으면 경험별로 나눕니다.
5. 한국어로 작성합니다.
"""

USER_PROMPT_TEMPLATE = """\
아래 경험 메모를 STAR 구조로 변환하세요.

### 경험 메모
{raw_text}
"""


# ------------------------------------------------------------------ #
# Structured Output 스키마                                             #
# ------------------------------------------------------------------ #

class STARExperience(BaseModel):
    title: str = Field(description="경험을 한 줄로 요약한 제목 (예: 'FastAPI 전환으로 API 성능 개선')")
    situation: str = Field(description="배경·문제 상황")
    task: str = Field(description="지원자가 맡은 역할과 목표")
    action: str = Field(description="지원자가 실제로 한 행동 (사용 기술 포함)")
    result: str = Field(description="결과와 성과 (원문에 있는 수치만)")
    skills: list[str] = Field(description="이 경험에서 사용한 기술/도구")
    missing_info: list[str] = Field(description="보완이 필요한 정보를 묻는 질문 목록 (없으면 빈 리스트)")


class STARResult(BaseModel):
    experiences: list[STARExperience]


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def convert_to_star(raw_text: str) -> list[dict[str, Any]]:
    """
    자유 텍스트 경험을 STAR 구조 리스트로 변환합니다.

    Returns
    -------
    list[dict]
        ``STARExperience.model_dump()`` 리스트. 입력이 비어 있으면 빈 리스트.
    """
    if not raw_text.strip():
        return []

    from llm.openai_client import OpenAIClient, FAST_MODEL

    client = OpenAIClient(model=FAST_MODEL)
    result: STARResult = client.generate_structured(
        prompt=USER_PROMPT_TEMPLATE.format(raw_text=raw_text[:4000]),
        schema=STARResult,
        system=SYSTEM_PROMPT,
    )
    experiences = [e.model_dump() for e in result.experiences]
    logger.info("[star_converter] %d개 경험 변환 완료.", len(experiences))
    return experiences


def star_to_text(star: dict[str, Any]) -> str:
    """STAR 딕셔너리를 벡터 DB 저장용 텍스트로 변환합니다 (보완 필요 항목은 제외)."""
    lines = [f"STAR 경험: {star.get('title', '')}"]
    for key, label in (("situation", "상황"), ("task", "과제"),
                       ("action", "행동"), ("result", "결과")):
        value = (star.get(key) or "").strip()
        if value and value != MISSING_MARK:
            lines.append(f"- {label}: {value}")
    if star.get("skills"):
        lines.append("- 사용 기술: " + ", ".join(star["skills"]))
    return "\n".join(lines)
