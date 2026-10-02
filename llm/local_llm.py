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
            response = client.chat(
                model=self._model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
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
            available_names = [m["name"] for m in models.get("models", [])]
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
