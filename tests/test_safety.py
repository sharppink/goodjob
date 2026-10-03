"""conftest 안전장치 자체 검증 — 이게 깨지면 다른 테스트가 실제 API·DB 를 건드릴 수 있음."""

from __future__ import annotations

import os
import socket

import pytest

from conftest import REAL_CHROMA_DIR


def test_external_network_is_blocked():
    with pytest.raises(RuntimeError, match="외부 네트워크"):
        socket.create_connection(("api.openai.com", 443), timeout=1)


def test_real_keys_are_not_loaded():
    from config.settings import settings

    assert settings.OPENAI_API_KEY == "sk-test-not-a-real-key"
    assert settings.TAVILY_API_KEY == ""
    assert not settings.langsmith_enabled


def test_chroma_points_to_temp_dir():
    from config.settings import settings

    assert os.path.abspath(settings.CHROMA_PERSIST_DIR) != str(REAL_CHROMA_DIR)
