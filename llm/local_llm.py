"""
llm/local_llm.py

Ollama-based local LLM client used for low-complexity tasks to reduce
API costs and latency.

The Ollama server must be running locally (default: http://localhost:11434).
Start it with: ``ollama serve``
Pull the model with: ``ollama pull qwen2.5:7b``
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048
# Ollama 기본 컨텍스트(2048 토큰)는 공고 원문(~3천 토큰)보다 짧아 입력이 잘림 → 확장
DEFAULT_NUM_CTX = 8192


class LocalLLM:
    """
    Thin wrapper around the ``ollama`` Python client.

    Usage
    -----
    >>> llm = LocalLLM()
    >>> if llm.is_available():
    ...     text = llm.generate("Extract skills from: …")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        model : str, optional
            Ollama model tag.  Falls back to ``settings.LOCAL_MODEL_NAME``.
        base_url : str, optional
            Ollama server URL.  Falls back to ``settings.OLLAMA_BASE_URL``.
        """
        self._model = model or settings.LOCAL_MODEL_NAME
        self._base_url = base_url or settings.OLLAMA_BASE_URL

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        format: Optional[str] = None,
    ) -> str:
        """
        Generate a text response using the local Ollama model.

        Parameters
        ----------
        prompt : str
            User message.
        system : str, optional
            System prompt.
        max_tokens : int
            Maximum number of tokens in the response.
        temperature : float
            Sampling temperature (lower = more deterministic).
        format : str, optional
            ``"json"`` forces Ollama to emit a valid JSON object.

        Returns
        -------
        str
            Generated text, or an error string if Ollama is unavailable.
        """
        logger.debug("[LocalLLM] generate() called (model=%s)", self._model)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            import ollama  # type: ignore
            client = ollama.Client(host=self._base_url)
            extra = {"format": format} if format else {}
            response = client.chat(
                model=self._model,
                messages=messages,
                **extra,
                options={
                    "num_predict": max_tokens,
                    "num_ctx": DEFAULT_NUM_CTX,
                    "temperature": temperature,
                },
            )
            return response["message"]["content"]
        except ImportError as exc:
            raise ImportError("ollama package is required: pip install ollama") from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("[LocalLLM] Ollama call failed: %s", exc)
            return f"[LocalLLM error: {exc}]"

    def is_available(self) -> bool:
        """
        Check whether the Ollama server is reachable and the model is loaded.

        Returns
        -------
        bool
            ``True`` if a test ping succeeds, ``False`` otherwise.
        """
        try:
            import ollama  # type: ignore
            client = ollama.Client(host=self._base_url)
            models = client.list()
            # ollama-python 0.4+ 는 객체(.model), 이전 버전은 dict("name") 반환
            # (예전 코드는 m["name"] 만 써서 신버전에서 항상 KeyError → 사용 불가 판정, PROJECT_DOCS #013)
            entries = getattr(models, "models", None)
            if entries is None:
                entries = models.get("models", [])
            available_names = [
                getattr(m, "model", None) or (m.get("name") if isinstance(m, dict) else "") or ""
                for m in entries
            ]
            # Accept both "qwen2.5:7b" and "qwen2.5" as matching
            base_name = self._model.split(":")[0]
            reachable = any(
                self._model in name or base_name in name
                for name in available_names
            )
            logger.info(
                "[LocalLLM] is_available=%s (model=%s, server=%s)",
                reachable,
                self._model,
                self._base_url,
            )
            return reachable
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LocalLLM] Availability check failed: %s", exc)
            return False

    @property
    def model_name(self) -> str:
        """Return the configured Ollama model name."""
        return self._model


class LocalChatClient:
    """
    OpenAIClient 와 같은 인터페이스(generate / stream / generate_structured)를 가진
    Ollama 클라이언트. 개인정보 보호 모드에서 llm.factory.get_chat_client() 가 반환합니다.

    generate_structured 는 Ollama 구조화 출력(format=<JSON 스키마>)으로 스키마에 맞는
    JSON 만 생성하게 한 뒤 pydantic 으로 검증합니다.
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self._model = model or settings.PRIVACY_CHAT_MODEL
        self._base_url = base_url or settings.OLLAMA_BASE_URL

    @property
    def model_name(self) -> str:
        return self._model

    def _client(self):
        import ollama  # type: ignore
        return ollama.Client(host=self._base_url)

    @staticmethod
    def _messages(prompt: str, system: Optional[str]) -> list[dict]:
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _options(self, max_tokens: int, temperature: float) -> dict:
        return {"num_predict": max_tokens, "temperature": temperature, "num_ctx": DEFAULT_NUM_CTX}

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.3) -> str:
        resp = self._client().chat(model=self._model, messages=self._messages(prompt, system),
                                   options=self._options(max_tokens, temperature))
        return resp["message"]["content"]

    def stream(self, prompt: str, system: Optional[str] = None,
               max_tokens: int = DEFAULT_MAX_TOKENS):
        for chunk in self._client().chat(model=self._model, messages=self._messages(prompt, system),
                                         options=self._options(max_tokens, 0.3), stream=True):
            delta = chunk["message"]["content"]
            if delta:
                yield delta

    def generate_structured(self, prompt: str, schema: type, system: Optional[str] = None,
                            temperature: float = 0.1):
        resp = self._client().chat(model=self._model, messages=self._messages(prompt, system),
                                   format=schema.model_json_schema(),
                                   options=self._options(DEFAULT_MAX_TOKENS, temperature))
        return schema.model_validate_json(resp["message"]["content"])
