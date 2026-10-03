"""
agents/coverletter_writer.py

자소서 문항별 작성 모드.

입력: 문항 텍스트 + 글자수 제한 (+ 회사명, 분석된 공고 요구사항이 있으면 함께 사용)
흐름:
  1. 프로필에서 문항과 관련된 경험 검색 (작은 프로필은 전체 사용)
  2. 문항 의도 분석 + 답변 작성 (Structured Output)
  3. 글자수(공백 포함)를 코드로 직접 세서, 제한을 넘으면 줄여 쓰도록 최대 2회 재요청

LLM 은 글자수를 정확히 세지 못하므로 길이 검증은 반드시 코드에서 합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_SHORTEN_RETRIES = 2
# 제한의 이 비율 이상을 채우도록 요청 (너무 짧은 답변 방지)
TARGET_MIN_RATIO = 0.85

SYSTEM_PROMPT = """\
당신은 한국 IT 기업 자기소개서 첨삭 전문가입니다.
지원자의 실제 경험만으로 자소서 문항에 대한 답변을 작성합니다.

규칙:
1. 경험·수치·회사명은 "지원자 경험"에 있는 것만 사용합니다. 지어내지 않습니다.
2. 문항이 묻는 것에 정확히 답합니다 (지원동기 문항에 성장과정을 쓰지 않음).
3. 두괄식: 첫 문장에 핵심 메시지를 둡니다.
4. 구체적인 경험 1~2개를 STAR 흐름(상황→행동→결과)으로 풀어 씁니다.
5. 회사·직무와의 연결로 마무리합니다.
6. 마크다운, 제목, 글머리표 없이 자연스러운 문단으로 작성합니다.
"""

USER_PROMPT_TEMPLATE = """\
### 지원 회사 / 직무
{company_name} / {job_title}

### 채용공고 요구사항
{requirements_text}

### 지원자 경험
{experiences_text}

### 자소서 문항
{question}

### 분량
공백 포함 {min_chars}~{char_limit}자 (절대 {char_limit}자를 넘지 말 것)
"""

SHORTEN_PROMPT_TEMPLATE = """\
아래 자소서 답변은 공백 포함 {current}자로, 제한 {char_limit}자를 넘었습니다.
핵심 경험과 메시지는 유지하고 덜 중요한 문장을 줄여서
공백 포함 {target}자 이내로 다시 작성하세요. 새로운 사실을 추가하지 마세요.

### 문항
{question}

### 기존 답변
{answer}
"""


class CoverLetterAnswer(BaseModel):
    question_intent: str = Field(description="이 문항으로 회사가 확인하려는 것 (1~2문장)")
    key_message: str = Field(description="답변의 핵심 메시지 한 줄")
    used_experiences: list[str] = Field(description="답변에 사용한 지원자 경험 (짧게)")
    answer: str = Field(description="자소서 답변 본문")


def count_chars(text: str) -> tuple[int, int]:
    """(공백 포함 글자수, 공백 제외 글자수)."""
    return len(text), len("".join(text.split()))


def write_answer(
    question: str,
    char_limit: int,
    company_name: str = "",
    job_requirements: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    자소서 문항 답변을 생성합니다.

    Returns
    -------
    dict
        answer, question_intent, key_message, used_experiences,
        chars_with_spaces, chars_without_spaces, char_limit, within_limit, retries
    """
    from llm.factory import get_chat_client

    req = job_requirements or {}
    experiences = _retrieve_experiences(question, req)
    min_chars = int(char_limit * TARGET_MIN_RATIO)

    client = get_chat_client()
    result: CoverLetterAnswer = client.generate_structured(
        prompt=USER_PROMPT_TEMPLATE.format(
            company_name=company_name or "지원 회사",
            job_title=req.get("job_title") or "개발자",
            requirements_text=_format_requirements(req),
            experiences_text="\n\n".join(f"[{i+1}] {e}" for i, e in enumerate(experiences))
                             or "(등록된 경험 없음)",
            question=question,
            min_chars=min_chars,
            char_limit=char_limit,
        ),
        schema=CoverLetterAnswer,
        system=SYSTEM_PROMPT,
        temperature=0.5,
    )
    answer = result.answer.strip()

    # 글자수 초과 시 줄여 쓰기 재요청
    retries = 0
    while count_chars(answer)[0] > char_limit and retries < MAX_SHORTEN_RETRIES:
        retries += 1
        current = count_chars(answer)[0]
        logger.info("[coverletter] %d자 > 제한 %d자 — 줄여 쓰기 %d회차", current, char_limit, retries)
        answer = client.generate(
            prompt=SHORTEN_PROMPT_TEMPLATE.format(
                current=current, char_limit=char_limit,
                target=int(char_limit * 0.95), question=question, answer=answer,
            ),
            system=SYSTEM_PROMPT,
            temperature=0.3,
        ).strip()

    with_spaces, without_spaces = count_chars(answer)
    return {
        "answer": answer,
        "question_intent": result.question_intent,
        "key_message": result.key_message,
        "used_experiences": result.used_experiences,
        "chars_with_spaces": with_spaces,
        "chars_without_spaces": without_spaces,
        "char_limit": char_limit,
        "within_limit": with_spaces <= char_limit,
        "retries": retries,
    }


def _retrieve_experiences(question: str, req: dict) -> list[str]:
    from agents.rag_retriever import SMALL_PROFILE_CHUNKS
    from rag.vectorstore import VectorStore

    vs = VectorStore()
    total = vs.count()
    if total == 0:
        return []
    if total <= SMALL_PROFILE_CHUNKS:
        return vs.get_all_documents()

    from rag.profile_loader import ProfileLoader
    loader = ProfileLoader()
    query = question + " " + " ".join((req.get("keywords") or [])[:8])
    return loader.search(query, k=6)


def _format_requirements(req: dict) -> str:
    parts = []
    if req.get("required_skills"):
        parts.append("필수: " + ", ".join(req["required_skills"]))
    if req.get("responsibilities"):
        parts.append("주요 업무: " + " / ".join(req["responsibilities"][:5]))
    if req.get("company_culture"):
        parts.append("회사 문화: " + req["company_culture"])
    return "\n".join(parts) or "(공고 정보 없음 — 회사명과 문항만으로 작성)"
