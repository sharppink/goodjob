"""
llm/openai_client.py

OpenAI API 클라이언트.

지원 기능
---------
- 텍스트 생성 (generate)
- 이미지 + 텍스트 비전 생성 (generate_with_vision)
- 스트리밍 생성 (stream)

기본 모델: gpt-4o (Vision 포함, 고품질)
저비용 모델: gpt-4o-mini (빠른 파싱 작업용)
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 2048

# Process-wide singleton — reusing one httpx connection pool avoids GC-triggered
# segfaults that occur when many short-lived OpenAI() instances are collected.
_shared_openai: Optional[object] = None  # openai.OpenAI


def _get_openai_client(api_key: str):
    global _shared_openai
    if _shared_openai is None:
        from openai import OpenAI
        # 429(분당 토큰 한도) 시 SDK 가 retry-after 를 지켜 지수 백오프로 재시도.
        # 기본 2회로는 tier 1 한도(gpt-4o 30k TPM)에서 연속 호출 시 실패함 (PROJECT_DOCS #016)
        _shared_openai = OpenAI(api_key=api_key, max_retries=8)
    return _shared_openai


class PrivacyModeError(RuntimeError):
    """개인정보 보호 모드에서 외부(OpenAI) 호출을 시도했을 때 발생."""


def _assert_openai_allowed() -> None:
    """
    개인정보 보호 모드면 OpenAI 호출을 차단합니다.

    각 기능이 로컬 클라이언트로 전환되도록 구현했지만, 놓친 경로가 있어도
    데이터가 외부로 나가지 않도록 마지막 방어선 역할을 합니다.
    """
    from config.settings import settings
    if settings.PRIVACY_MODE:
        raise PrivacyModeError("개인정보 보호 모드에서는 OpenAI API 를 호출할 수 없습니다.")


class OpenAIClient:
    """
    OpenAI Python SDK 래퍼.

    Usage
    -----
    >>> client = OpenAIClient()
    >>> text = client.generate("이 채용공고를 분석해줘: …")

    >>> # 빠른 파싱 작업 (저비용)
    >>> client_fast = OpenAIClient(model=FAST_MODEL)
    >>> text = client_fast.generate("회사명만 추출해줘: …")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        _assert_openai_allowed()  # 클라이언트 생성 단계부터 차단
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._model = model
        self._client = self._build_client()

    def _build_client(self):
        try:
            return _get_openai_client(self._api_key)
        except ImportError as exc:
            raise ImportError("openai 패키지가 필요합니다: pip install openai") from exc

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
    ) -> str:
        """
        텍스트 응답 생성.

        Parameters
        ----------
        prompt : str
            사용자 메시지.
        system : str, optional
            시스템 프롬프트 (모델 역할/지침 설정).
        max_tokens : int
            최대 출력 토큰 수.
        temperature : float
            0에 가까울수록 결정적, 1에 가까울수록 창의적.
        """
        logger.debug("[OpenAIClient] generate() (model=%s)", self._model)
        _assert_openai_allowed()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def generate_with_vision(
        self,
        prompt: str,
        image_path: str | Path,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        이미지 + 텍스트 비전 응답 생성.

        Parameters
        ----------
        prompt : str
            텍스트 질문.
        image_path : str | Path
            이미지 파일 경로 (PNG, JPEG, WEBP, GIF).
        """
        logger.debug("[OpenAIClient] generate_with_vision() for: %s", image_path)
        _assert_openai_allowed()
        image_data, media_type = self._encode_image(image_path)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    },
                },
                {"type": "text", "text": prompt},
            ],
        })

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        prompt: str,
        schema: type,
        system: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Structured Output — 응답을 Pydantic 모델로 강제 파싱합니다.

        Parameters
        ----------
        prompt : str
            사용자 메시지.
        schema : type
            pydantic.BaseModel 서브클래스.  OpenAI가 이 스키마에 맞는
            JSON만 출력하도록 강제합니다.
        system : str, optional
            시스템 프롬프트.
        temperature : float
            구조화 출력에는 낮은 값(0.0~0.2)을 권장합니다.

        Returns
        -------
        pydantic.BaseModel instance
            ``schema`` 타입의 인스턴스.
        """
        logger.debug("[OpenAIClient] generate_structured() (model=%s)", self._model)
        _assert_openai_allowed()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=schema,
            temperature=temperature,
        )
        return response.choices[0].message.parsed

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[str]:
        """
        토큰 단위 스트리밍 응답.

        Yields
        ------
        str
            순차적인 텍스트 델타 청크.
        """
        logger.debug("[OpenAIClient] stream() called.")
        _assert_openai_allowed()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_image(image_path: str | Path) -> tuple[str, str]:
        """이미지를 base64로 인코딩하여 (data, media_type) 반환."""
        path = Path(image_path)
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(path.suffix.lower(), "image/png")
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return data, media_type
