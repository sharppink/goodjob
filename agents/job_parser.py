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
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.state import GoodJobState
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

_router = LLMRouter()

# 공고 원문 최대 길이 (추론·학습 데이터 공통)
MAX_POSTING_CHARS = 4000

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
    is_job_posting: bool = Field(
        description=(
            "특정 회사의 단일 채용공고면 true. 커리어 조언 글, 뉴스, 여러 공고를 모은 "
            "검색/목록 페이지, 연봉 통계 등은 false"
        )
    )
    company_name: str = Field(
        description="채용하는 회사명 (채용 사이트 이름이 아닌 실제 고용 회사, (주)·주식회사 표기 제외, 모르면 빈 문자열)"
    )
    job_title: str = Field(description="채용 직무명")
    required_skills: list[str] = Field(
        description=(
            "필수 자격요건의 기술·역량을 짧은 이름으로 (예: 'Python', 'Playwright', 'SQL', "
            "'REST API 설계', 'E2E 테스트'). 문장·'~경험'·'~우대' 표현 금지, 경력 연수는 "
            "experience_years 로만 표현, 괄호로 묶인 도구는 각각 분리"
        )
    )
    preferred_skills: list[str] = Field(
        description="우대사항의 기술·역량·자격증을 짧은 이름으로 (required_skills 와 같은 규칙)"
    )
    experience_years: int = Field(description="최소 경력 연수 (신입/미명시=0)")
    responsibilities: list[str] = Field(description="주요 업무 목록 (최대 7개)")
    company_culture: str = Field(description="회사 문화 및 근무 환경 요약 (2~3문장)")
    keywords: list[str] = Field(description="ATS 통과용 핵심 키워드 10~15개 (각 1~3단어)")
    location: str = Field(default="", description="근무 지역")
    remote_policy: str = Field(default="", description="재택/출근/하이브리드 여부")
    job_type: str = Field(default="정규직", description="고용 형태 (정규직/계약직/인턴)")
    salary_range: str = Field(default="", description="급여 범위 (명시된 경우)")


# 로컬 LLM 전용: Structured Output 대신 프롬프트로 JSON 키를 명시
LOCAL_JSON_INSTRUCTION = """

### 출력 형식
아래 키를 모두 가진 JSON 객체 하나만 출력하세요. 설명 문장은 쓰지 마세요.
모든 값은 공고 원문의 언어(한국어 또는 영어)로만 쓰고, 중국어는 절대 쓰지 마세요.
원문에 없는 내용은 지어내지 말고 빈 문자열이나 빈 리스트로 두세요.
remote_policy 는 재택/출근/하이브리드 여부만 적습니다 (복장·근무시간 등은 넣지 않음).
{
  "is_job_posting": true,
  "company_name": "실제 고용 회사명 ((주)·㈜ 제외, 점핏·사람인 같은 채용 사이트명 아님)",
  "job_title": "직무명",
  "required_skills": ["짧은 기술명 (예: Python, Playwright, SQL)"],
  "preferred_skills": ["짧은 기술명/자격증"],
  "experience_years": 0,
  "responsibilities": ["주요 업무 (최대 7개)"],
  "company_culture": "회사 문화 요약 (2~3문장)",
  "keywords": ["ATS 핵심 키워드 10~15개 (각 1~3단어)"],
  "location": "근무 지역",
  "remote_policy": "재택/출근/하이브리드",
  "job_type": "정규직/계약직/인턴",
  "salary_range": "급여 범위 (없으면 빈 문자열)"
}
"""


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

    trimmed = clean_posting_text(raw_text)
    from config.settings import settings
    if settings.PRIVACY_MODE:
        # 개인정보 보호 모드: 파인튜닝 sLLM → 실패 시 로컬 범용 모델 (OpenAI 사용 안 함)
        try:
            requirements = parse_privately(trimmed)
        except Exception as exc:
            logger.error("[job_parser] 로컬 파싱 실패: %s", exc)
            errors.append(f"job_parser: 로컬 파싱 실패 – {exc}")
            requirements = _empty_requirements()
        state["job_requirements"] = requirements
        state["errors"] = errors
        return state

    if settings.PARSE_WITH_LOCAL_LLM:
        # 파인튜닝한 파서 모델을 쓰는 경우: 길이와 관계없이 로컬 우선
        llm_choice = "local" if _router._check_local_available() else "openai"
    else:
        complexity = _estimate_complexity(trimmed)
        llm_choice = _router.route(task_type="parse", complexity=complexity)
    logger.info("[job_parser] LLM 선택: %s", llm_choice)

    requirements = None
    if llm_choice == "local":
        try:
            requirements = parse_with_local_llm(trimmed)
        except Exception as exc:
            # 로컬 실패 시 OpenAI 로 폴백 (빈 결과로 끝내지 않음)
            logger.warning("[job_parser] 로컬 LLM 실패 → OpenAI 폴백: %s", exc)
            errors.append(f"job_parser: 로컬 LLM 실패, OpenAI 로 대체 – {exc}")

    if requirements is None:
        try:
            requirements = parse_posting_structured(trimmed, raise_on_error=True)
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

_DATA_URI_IMAGE = re.compile(r"!\[[^\]]*\]\(data:[^)]*\)")
_DATA_URI = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def clean_posting_text(raw_text: str) -> str:
    """
    공고 원문의 노이즈 제거 후 MAX_POSTING_CHARS 로 자름 (추론·학습 데이터 공통).

    웹 수집 원문에는 base64 이미지(data:image/png;base64,...)와 마크다운 링크 URL 이
    섞여 있어 토큰을 낭비하고 모델이 노이즈를 읽게 만듦 (PROJECT_DOCS #014).
    """
    text = _DATA_URI_IMAGE.sub("", raw_text)
    text = _DATA_URI.sub("", text)
    text = _MD_LINK.sub(r"\1", text)          # [쏘카](/company/...) → 쏘카
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()[:MAX_POSTING_CHARS]


_UNKNOWN_VALUES = {"n/a", "na", "none", "null", "정보 없음", "정보없음", "상세 불명", "불명",
                   "미정", "명시되지 않음", "명시 안 됨", "알 수 없음", "해당 없음", "-"}


def normalize_remote_policy(value: str) -> str:
    """근무 형태를 출근 / 재택 / 하이브리드 / "" 로 정규화 (근무시간·복장 제도는 "")."""
    v = (value or "").replace(" ", "")
    if not v:
        return ""
    has_remote = any(k in v for k in ("재택", "원격", "리모트", "remote", "Remote"))
    has_office = any(k in v for k in ("출근", "오피스", "현장", "상주", "사무실"))
    if "하이브리드" in v or "hybrid" in v.lower() or (has_remote and has_office):
        return "하이브리드"
    if has_remote:
        # "재택근무 가능", "주 2일 재택" 등 부분 재택 표현이면 하이브리드
        partial = any(k in v for k in ("가능", "일부", "주", "선택", "병행", "partial"))
        return "하이브리드" if partial else "재택"
    if has_office:
        return "출근"
    return ""


def normalize_requirements(req: dict[str, Any]) -> dict[str, Any]:
    """
    LLM 출력(OpenAI·로컬 공통) 정규화.

    gpt-4o 라벨도 '정보 없음'/'N/A'/'상세 불명' 등 빈 값 표기가 제각각이고 remote_policy 에
    유연근무제·복장 규정이 섞여 있어, 학생 모델이 이를 그대로 배움 (PROJECT_DOCS #019).
    """
    out = dict(req)
    for key, value in out.items():
        if isinstance(value, str) and value.strip().lower() in _UNKNOWN_VALUES:
            out[key] = ""
        elif isinstance(value, list):
            out[key] = [x for x in value
                        if not (isinstance(x, str) and x.strip().lower() in _UNKNOWN_VALUES)]
    out["remote_policy"] = normalize_remote_policy(out.get("remote_policy", ""))
    return out


def build_local_messages(raw_text: str) -> list[dict[str, str]]:
    """
    로컬 LLM 에 보내는 채팅 메시지 (system + user).

    파인튜닝 학습 데이터(finetune/data_collector.py)도 이 함수로 만들어서
    학습 때와 실제 추론 때의 입력 형식이 정확히 같도록 유지합니다.
    """
    user = USER_PROMPT_TEMPLATE.format(raw_text=clean_posting_text(raw_text)) + LOCAL_JSON_INSTRUCTION
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_local_response(raw_resp: str) -> dict[str, Any]:
    """로컬 LLM 응답 문자열 → 정규화된 requirements dict (실패 시 예외)."""
    import json
    import re
    if raw_resp.startswith("[LocalLLM error"):
        raise RuntimeError(raw_resp)
    clean = re.sub(r"```(?:json)?", "", raw_resp).strip().rstrip("`").strip()
    return normalize_requirements(_normalize(json.loads(clean)))


def parse_with_local_llm(raw_text: str, model: str | None = None) -> dict[str, Any]:
    """Ollama 로컬 모델(JSON 모드)로 공고를 파싱합니다. model 미지정 시 LOCAL_MODEL_NAME."""
    from llm.local_llm import LocalLLM
    messages = build_local_messages(raw_text)
    raw_resp = LocalLLM(model=model).generate(
        prompt=messages[1]["content"],
        system=messages[0]["content"],
        format="json",
    )
    requirements = parse_local_response(raw_resp)
    if _has_chinese_leak(requirements):
        # Qwen 계열의 중국어 혼입 — 예외를 내서 job_parser_node 가 OpenAI 로 재파싱 (PROJECT_DOCS #018)
        raise ValueError("로컬 모델 출력에 중국어가 섞임")
    return requirements


# finetune/metrics.py 의 no_chinese_leak 지표와 같은 규칙
# (metrics 는 Colab 번들에 단독으로 들어가야 해서 agents 를 import 하지 않음)
_CHINESE_LEAK = re.compile(r"[一-鿿]{4,}|[，。；：！？]")


def _has_chinese_leak(data: Any) -> bool:
    if isinstance(data, str):
        return bool(_CHINESE_LEAK.search(data))
    if isinstance(data, dict):
        return any(_has_chinese_leak(v) for v in data.values())
    if isinstance(data, list):
        return any(_has_chinese_leak(v) for v in data)
    return False


def parse_privately(raw_text: str) -> dict[str, Any]:
    """
    개인정보 보호 모드 파싱 — 외부 API 를 쓰지 않음.

    1) 파인튜닝 sLLM(PRIVACY_PARSER_MODEL) — 학습 때와 같은 프롬프트·JSON 모드
    2) 실패/중국어 혼입 시 로컬 범용 모델(PRIVACY_CHAT_MODEL) + 구조화 출력
    """
    from config.settings import settings
    try:
        return parse_with_local_llm(raw_text, model=settings.PRIVACY_PARSER_MODEL)
    except Exception as exc:
        logger.warning("[job_parser] 파인튜닝 파서 실패 → 로컬 범용 모델: %s", exc)
    from llm.local_llm import LocalChatClient
    parsed = LocalChatClient().generate_structured(
        prompt=USER_PROMPT_TEMPLATE.format(raw_text=clean_posting_text(raw_text)),
        schema=JobRequirements,
        system=SYSTEM_PROMPT,
    )
    return normalize_requirements(parsed.model_dump())


def parse_posting_structured(raw_text: str, raise_on_error: bool = False) -> dict[str, Any]:
    """
    단일 채용공고 텍스트를 Structured Output(OpenAI)으로 파싱하여 dict 반환.
    job_recommender, 파인튜닝 데이터 라벨링(teacher) 등 외부 호출용.
    """
    from llm.factory import is_privacy_mode
    if is_privacy_mode():
        try:
            return parse_privately(raw_text)
        except Exception as exc:
            logger.error("[job_parser] 로컬 파싱 실패: %s", exc)
            if raise_on_error:
                raise
            return _empty_requirements()

    prompt = USER_PROMPT_TEMPLATE.format(raw_text=clean_posting_text(raw_text))
    try:
        from llm.openai_client import OpenAIClient
        client = OpenAIClient()
        parsed: JobRequirements = client.generate_structured(
            prompt=prompt,
            schema=JobRequirements,
            system=SYSTEM_PROMPT,
        )
        return normalize_requirements(parsed.model_dump())
    except Exception as exc:
        logger.error("[job_parser] parse_posting_structured 실패: %s", exc)
        if raise_on_error:
            raise
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
        "is_job_posting":   bool(data.get("is_job_posting", True)),
        "company_name":     data.get("company_name", ""),
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
        "is_job_posting": False, "company_name": "",
        "job_title": "", "required_skills": [], "preferred_skills": [],
        "experience_years": 0, "responsibilities": [], "company_culture": "",
        "keywords": [], "location": "", "remote_policy": "",
        "job_type": "정규직", "salary_range": "",
    }
