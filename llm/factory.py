"""
llm/factory.py

현재 모드에 맞는 채팅 클라이언트를 반환합니다.

- 일반 모드          : OpenAIClient (gpt-4o, fast=True 면 gpt-4o-mini)
- 개인정보 보호 모드 : LocalChatClient (Ollama, settings.PRIVACY_CHAT_MODEL)

두 클라이언트는 generate / stream / generate_structured 인터페이스가 같아서
호출하는 쪽 코드는 모드를 신경 쓰지 않아도 됩니다.
"""

from __future__ import annotations

from config.settings import settings


def is_privacy_mode() -> bool:
    return bool(settings.PRIVACY_MODE)


def get_chat_client(fast: bool = False):
    if is_privacy_mode():
        from llm.local_llm import LocalChatClient
        return LocalChatClient()
    from llm.openai_client import FAST_MODEL, OpenAIClient
    return OpenAIClient(model=FAST_MODEL) if fast else OpenAIClient()
