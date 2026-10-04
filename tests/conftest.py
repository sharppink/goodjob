"""
tests/conftest.py

공통 픽스처. 모든 테스트는 외부 API·실제 DB 없이 돌아갑니다.

안전장치
--------
1. 환경변수를 config.settings 가 로드되기 전에 덮어써서 .env 의 실제 키를 쓰지 않음
   (pydantic-settings 는 환경변수가 .env 보다 우선)
2. 루프백(127.0.0.1) 외 네트워크 접속을 차단 — 실수로 OpenAI/Tavily 를 호출하면 즉시 실패
3. Chroma 경로를 테스트마다 임시 폴더로 바꾸고, 실제 chroma_db 를 가리키면 테스트를 중단
   (평가 스크립트가 실제 DB 를 덮어쓴 사고 재발 방지 — PROJECT_DOCS #009)
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_CHROMA_DIR = (PROJECT_ROOT / "chroma_db").resolve()

sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------ #
# 1. 환경변수 (settings import 전에 설정)                              #
# ------------------------------------------------------------------ #

_SESSION_TMP = Path(tempfile.mkdtemp(prefix="goodjob_test_"))

os.environ.update({
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "TAVILY_API_KEY": "",
    "LANGCHAIN_API_KEY": "",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "PRIVACY_MODE": "false",
    "PARSE_WITH_LOCAL_LLM": "false",
    "REDIS_ENABLED": "false",
    "RERANKER_ENABLED": "false",
    "OLLAMA_BASE_URL": "http://127.0.0.1:9",
    "CHROMA_PERSIST_DIR": str(_SESSION_TMP / "chroma"),
    "SUPABASE_DB_URL": "",
    "ANONYMIZED_TELEMETRY": "False",
})

# ------------------------------------------------------------------ #
# 2. 외부 네트워크 차단                                                #
# ------------------------------------------------------------------ #

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _check_address(address: Any) -> None:
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and host not in _LOOPBACK:
        raise RuntimeError(f"테스트 중 외부 네트워크 접속 차단: {address}")


def _guarded_connect(self, address):
    _check_address(address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    _check_address(address)
    return _real_connect_ex(self, address)


socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]


# ------------------------------------------------------------------ #
# Fakes                                                                #
# ------------------------------------------------------------------ #

class FakeEmbeddingModel:
    """
    단어 해시 기반 결정적 임베딩 (bag-of-words).

    같은 단어를 공유하는 문서끼리 코사인 유사도가 높아서 검색 순서를 검증할 수 있습니다.
    """

    DIM = 256

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in re.findall(r"\w+", text.lower()):
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


class FakeChatClient:
    """
    OpenAIClient / LocalChatClient 와 같은 인터페이스의 가짜 클라이언트.

    - structured[스키마 이름] = dict | 스키마 객체 | 예외 | callable(prompt) → 위 셋 중 하나
    - texts : generate() 가 순서대로 반환할 문자열(또는 예외) 큐
    - calls : 호출 기록 (kind, 스키마 이름 또는 None, prompt)
    """

    DEFAULT_TEXT = "## 자기소개\n테스트 이력서입니다."

    def __init__(self) -> None:
        self.structured: dict[str, Any] = {}
        self.texts: list[Any] = []
        self.calls: list[tuple[str, str | None, str]] = []

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        self.calls.append(("generate", None, prompt))
        resp = self.texts.pop(0) if self.texts else self.DEFAULT_TEXT
        if isinstance(resp, Exception):
            raise resp
        return resp

    def stream(self, prompt: str, system: str | None = None, **kwargs: Any):
        yield self.generate(prompt, system)

    def generate_structured(self, prompt: str, schema: type, system: str | None = None, **kwargs: Any):
        self.calls.append(("structured", schema.__name__, prompt))
        value = self.structured.get(schema.__name__)
        if value is None:
            raise AssertionError(f"FakeChatClient: {schema.__name__} 응답이 설정되지 않음")
        if callable(value) and not isinstance(value, type):
            value = value(prompt)
        if isinstance(value, Exception):
            raise value
        return schema.model_validate(value) if isinstance(value, dict) else value

    def count(self, kind: str, schema: str | None = None) -> int:
        return sum(1 for k, s, _ in self.calls if k == kind and (schema is None or s == schema))


# ------------------------------------------------------------------ #
# 샘플 데이터                                                          #
# ------------------------------------------------------------------ #

SAMPLE_POSTING = """\
[테스트컴퍼니] 백엔드 개발자 채용

■ 담당 업무
- Python/FastAPI 기반 REST API 개발
- PostgreSQL 데이터 모델링

■ 자격 요건
- Python 백엔드 경력 3년 이상
- FastAPI 실무 경험

■ 우대 사항
- Docker, Kubernetes 운영 경험
"""

SAMPLE_REQUIREMENTS: dict[str, Any] = {
    "is_job_posting": True,
    "company_name": "테스트컴퍼니",
    "job_title": "백엔드 개발자",
    "required_skills": ["Python", "FastAPI"],
    "preferred_skills": ["Docker", "Kubernetes"],
    "experience_years": 3,
    "responsibilities": ["REST API 개발", "PostgreSQL 데이터 모델링"],
    "company_culture": "자율적인 문화",
    "keywords": ["Python", "FastAPI", "PostgreSQL", "REST API"],
    "location": "서울",
    "remote_policy": "하이브리드",
    "job_type": "정규직",
    "salary_range": "",
}

SAMPLE_PROFILE: dict[str, Any] = {
    "이름": "테스트 지원자",
    "기술": ["Python", "FastAPI", "Docker"],
    "경력": ["A사 백엔드 개발자 4년 - FastAPI 서버 개발, PostgreSQL 설계"],
    "프로젝트": ["LangGraph 기반 이력서 생성 서비스"],
    "학력": "테스트대학교 컴퓨터공학과",
}


def make_fit(score: float, matched=("Python", "FastAPI"), missing=("Kubernetes",)) -> dict[str, Any]:
    return {
        "reasoning": "테스트 추론",
        "matched_skills": list(matched),
        "missing_skills": list(missing),
        "fit_score": score,
        "fit_feedback": f"적합도 {score}",
    }


INTERVIEW_SET: dict[str, Any] = {
    "questions": [
        {"category": "기술", "question": "FastAPI 를 고른 이유는?", "intent": "기술 선택 근거",
         "answer_tip": "비교 대상과 결정 기준을 말하세요."},
        {"category": "약점 보완", "question": "Kubernetes 경험이 없는데요?", "intent": "학습 의지",
         "answer_tip": "Docker 경험과 학습 계획을 연결하세요."},
    ]
}


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """테스트마다 임시 Chroma 경로 + 가짜 임베딩 + 기본 설정값."""
    from config.settings import settings
    import rag.vectorstore as vectorstore

    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", str(chroma_dir))
    monkeypatch.setattr(settings, "PRIVACY_MODE", False)
    monkeypatch.setattr(settings, "PARSE_WITH_LOCAL_LLM", False)
    monkeypatch.setattr(settings, "RERANKER_ENABLED", False)
    monkeypatch.setattr(settings, "REDIS_ENABLED", False)
    monkeypatch.setattr(settings, "SUPABASE_DB_URL", "")
    monkeypatch.setattr(vectorstore, "EmbeddingModel", FakeEmbeddingModel)
    # 로그인 계정은 테스트마다 공용(None)에서 시작
    vectorstore.set_current_user(None)
    vectorstore.require_login(False)

    if Path(settings.CHROMA_PERSIST_DIR).resolve() == REAL_CHROMA_DIR:
        pytest.exit("테스트가 실제 chroma_db 를 가리킵니다 — 중단 (PROJECT_DOCS #009)", returncode=3)
    yield


@pytest.fixture
def fake_llm(monkeypatch) -> FakeChatClient:
    """모든 채팅 LLM 호출(get_chat_client, OpenAIClient)을 FakeChatClient 로 대체."""
    import llm.factory
    import llm.openai_client
    from agents import job_parser

    fake = FakeChatClient()
    monkeypatch.setattr(llm.factory, "get_chat_client", lambda fast=False: fake)
    monkeypatch.setattr(llm.openai_client, "OpenAIClient", lambda *a, **k: fake)
    # 공고 파싱 라우터가 Ollama 상태를 확인하지 않도록 "로컬 없음"으로 고정
    monkeypatch.setattr(job_parser._router, "_local_available", False)
    return fake


@pytest.fixture
def loaded_profile():
    """임시 Chroma 에 샘플 프로필을 저장하고 ProfileLoader 를 반환."""
    from rag.profile_loader import ProfileLoader

    loader = ProfileLoader()
    loader.load_from_dict(SAMPLE_PROFILE)
    return loader
