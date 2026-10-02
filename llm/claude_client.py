"""
llm/claude_client.py

하위 호환 래퍼 — 내부적으로 OpenAIClient를 사용합니다.
기존 코드에서 ClaudeClient를 import해도 그대로 동작합니다.

    from llm.claude_client import ClaudeClient  # 변경 불필요
"""

from llm.openai_client import OpenAIClient as ClaudeClient  # noqa: F401

__all__ = ["ClaudeClient"]
