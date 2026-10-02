"""
llm package

Provides clients for Claude (cloud) and Ollama (local) LLMs, plus an
automatic router that selects the best model for each task.
"""

from llm.claude_client import ClaudeClient
from llm.local_llm import LocalLLM
from llm.router import LLMRouter

__all__ = ["ClaudeClient", "LocalLLM", "LLMRouter"]
