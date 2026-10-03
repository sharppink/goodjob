"""
개인정보 보호 모드 — 켜져 있으면 어떤 경로로도 OpenAI 를 호출하지 않아야 함.

여기서는 fake_llm 을 쓰지 않습니다. 실제 클라이언트 선택·차단 로직을 그대로 검증합니다.
"""

from __future__ import annotations

import pytest

from conftest import SAMPLE_POSTING, SAMPLE_REQUIREMENTS


@pytest.fixture
def privacy_on(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "PRIVACY_MODE", True)


def test_factory_returns_local_client(privacy_on):
    from llm.factory import get_chat_client
    from llm.local_llm import LocalChatClient

    assert isinstance(get_chat_client(), LocalChatClient)
    assert isinstance(get_chat_client(fast=True), LocalChatClient)


def test_factory_returns_openai_client_when_off():
    from llm.factory import get_chat_client
    from llm.openai_client import FAST_MODEL, OpenAIClient

    assert isinstance(get_chat_client(), OpenAIClient)
    assert get_chat_client(fast=True)._model == FAST_MODEL


def test_openai_client_is_blocked(privacy_on):
    from llm.openai_client import OpenAIClient, PrivacyModeError

    with pytest.raises(PrivacyModeError):
        OpenAIClient()


def test_openai_embedding_is_blocked(privacy_on):
    from llm.openai_client import PrivacyModeError
    from rag.embeddings import EmbeddingModel

    with pytest.raises(PrivacyModeError):
        EmbeddingModel(use_openai=True)._openai_embed(["이력서 내용"])


def test_default_embedding_uses_local_model(privacy_on, monkeypatch):
    from rag.embeddings import EmbeddingModel

    em = EmbeddingModel()
    monkeypatch.setattr(em, "_ollama_embed", lambda texts: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(em, "_openai_embed", lambda texts: pytest.fail("OpenAI 임베딩 호출됨"))
    assert em.embed_texts(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert em.embed_query("q") == [0.1, 0.2]


def test_vectorstore_uses_separate_collection(privacy_on):
    """bge-m3(1024차원) 저장소는 OpenAI(1536차원) 저장소와 분리돼야 함."""
    from rag.vectorstore import PRIVATE_COLLECTION_NAME, VectorStore

    vs = VectorStore()
    vs.initialize()
    assert vs._collection_name == PRIVATE_COLLECTION_NAME


def test_job_parser_falls_back_to_local_chat_model(privacy_on, monkeypatch):
    """파인튜닝 파서가 실패하면 로컬 범용 모델로 — OpenAI 로 가지 않음."""
    from agents import job_parser
    from agents.job_parser import JobRequirements
    from llm.local_llm import LocalChatClient

    def broken_finetuned(raw_text, model=None):
        raise ValueError("파서 모델 없음")

    monkeypatch.setattr(job_parser, "parse_with_local_llm", broken_finetuned)
    monkeypatch.setattr(LocalChatClient, "generate_structured",
                        lambda self, **kw: JobRequirements.model_validate(SAMPLE_REQUIREMENTS))

    state = job_parser.job_parser_node({"job_posting_raw": SAMPLE_POSTING})
    assert state["job_requirements"]["job_title"] == "백엔드 개발자"
    assert state.get("errors") == []


def test_job_parser_records_error_when_all_local_models_fail(privacy_on, monkeypatch):
    from agents import job_parser
    from llm.local_llm import LocalChatClient

    def fail(*a, **k):
        raise ConnectionError("ollama down")

    monkeypatch.setattr(job_parser, "parse_with_local_llm", fail)
    monkeypatch.setattr(LocalChatClient, "generate_structured", fail)

    state = job_parser.job_parser_node({"job_posting_raw": SAMPLE_POSTING})
    assert state["job_requirements"]["required_skills"] == []
    assert any("로컬 파싱 실패" in e for e in state["errors"])
