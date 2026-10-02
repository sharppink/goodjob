"""
config/tracing.py

LangSmith 트레이싱 초기화.
애플리케이션 시작 시 setup_tracing() 한 번만 호출하면
LangGraph 실행 흐름이 자동으로 LangSmith에 기록됩니다.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing() -> bool:
    """
    LangSmith 트레이싱을 환경변수로 활성화합니다.

    Returns
    -------
    bool
        True면 트레이싱 활성화됨, False면 키 없어서 비활성.
    """
    from config.settings import settings

    if not settings.langsmith_enabled:
        logger.info("[Tracing] LangSmith 비활성 (키 없거나 LANGCHAIN_TRACING_V2=false)")
        return False

    # LangChain/LangGraph가 읽는 환경변수 세팅
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT

    logger.info(
        "[Tracing] LangSmith 활성화 — 프로젝트: '%s'  대시보드: https://smith.langchain.com",
        settings.LANGCHAIN_PROJECT,
    )
    return True
