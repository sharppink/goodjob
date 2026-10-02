"""
llm/router.py

Automatically selects the most appropriate LLM (Claude or local Ollama)
based on the task type and complexity.

Routing rules
-------------
+-------------+-----------+-------------------------------------------+
| task_type   | complexity | decision                                  |
+-------------+-----------+-------------------------------------------+
| any         | high       | Claude (always)                           |
| analyze     | medium     | Claude (nuanced reasoning required)       |
| generate    | medium     | Claude (creative quality matters)         |
| review      | medium     | Claude (editorial quality matters)        |
| parse       | medium     | local if available, else Claude           |
| any         | low        | local if available, else Claude           |
+-------------+-----------+-------------------------------------------+

If the local model is not available (Ollama down / model not loaded),
all calls are automatically routed to Claude regardless of complexity.
"""

from __future__ import annotations

import logging
from typing import Literal

from config.settings import settings

logger = logging.getLogger(__name__)

TaskType = Literal["parse", "analyze", "generate", "review"]
Complexity = Literal["low", "medium", "high"]
LLMChoice = Literal["openai", "local"]  # "openai" = OpenAI API, "local" = Ollama

# Tasks where Claude is always preferred at medium complexity
_CLAUDE_PREFERRED_TASKS: frozenset[str] = frozenset({"analyze", "generate", "review"})


class LLMRouter:
    """
    Decides which LLM to use for a given (task_type, complexity) pair.

    Usage
    -----
    >>> router = LLMRouter()
    >>> router.route("parse", "low")
    'local'
    >>> router.route("analyze", "high")
    'openai'
    """

    def __init__(self, threshold: float | None = None) -> None:
        """
        Parameters
        ----------
        threshold : float, optional
            Complexity score cut-off [0, 1] for routing to Claude.
            Defaults to ``settings.LLM_ROUTER_THRESHOLD``.
        """
        self._threshold = threshold if threshold is not None else settings.LLM_ROUTER_THRESHOLD
        self._local_available: bool | None = None  # cached

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def route(self, task_type: TaskType, complexity: Complexity) -> LLMChoice:
        """
        Return the recommended LLM for the given task and complexity.

        Parameters
        ----------
        task_type : str
            One of ``"parse"``, ``"analyze"``, ``"generate"``, ``"review"``.
        complexity : str
            One of ``"low"``, ``"medium"``, ``"high"``.

        Returns
        -------
        str
            ``"openai"`` or ``"local"``.
        """
        decision = self._decide(task_type, complexity)
        logger.info(
            "[LLMRouter] task=%s, complexity=%s → %s",
            task_type,
            complexity,
            decision,
        )
        return decision

    def route_by_score(self, score: float, task_type: TaskType = "parse") -> LLMChoice:
        """
        Route based on a raw numeric complexity score [0.0, 1.0].

        Parameters
        ----------
        score : float
            Complexity score.  Values above ``self._threshold`` → Claude.
        task_type : TaskType
            Task type hint (used when score is near the threshold).

        Returns
        -------
        str
            ``"openai"`` or ``"local"``.
        """
        if score > self._threshold:
            return "openai"
        complexity: Complexity = "low" if score < 0.3 else "medium"
        return self.route(task_type, complexity)

    def invalidate_local_cache(self) -> None:
        """Force a fresh availability check for the local LLM on next call."""
        self._local_available = None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _decide(self, task_type: str, complexity: str) -> LLMChoice:
        if complexity == "high":
            return "openai"

        if complexity == "medium" and task_type in _CLAUDE_PREFERRED_TASKS:
            return "openai"

        # For low complexity or non-critical medium tasks, try local first
        if self._check_local_available():
            return "local"

        return "openai"

    def _check_local_available(self) -> bool:
        """
        Check local LLM availability (result is cached per instance lifetime).
        """
        if self._local_available is None:
            try:
                from llm.local_llm import LocalLLM  # noqa: PLC0415
                self._local_available = LocalLLM().is_available()
            except Exception:  # noqa: BLE001
                self._local_available = False
        return self._local_available
