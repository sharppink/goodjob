"""
config/settings.py

Centralised application settings loaded from environment variables / .env file.
Uses pydantic-settings so every field is validated at startup.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.

    All values can be overridden via environment variables or a .env file
    placed in the project root.  See .env.example for a full list.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # LLM API keys                                                         #
    # ------------------------------------------------------------------ #
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key (required for cloud LLM calls).",
    )
    TAVILY_API_KEY: str = Field(
        default="",
        description="Tavily Search API key (required for company/job searches).",
    )

    # ------------------------------------------------------------------ #
    # LangSmith (LLMOps / tracing)                                        #
    # ------------------------------------------------------------------ #
    LANGCHAIN_API_KEY: str = Field(
        default="",
        description="LangSmith API key for LangGraph tracing and observability.",
    )
    LANGCHAIN_TRACING_V2: str = Field(
        default="false",
        description="Set to 'true' to enable LangSmith tracing.",
    )
    LANGCHAIN_PROJECT: str = Field(
        default="goodjob",
        description="LangSmith project name.",
    )
    LANGCHAIN_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint.",
    )

    # ------------------------------------------------------------------ #
    # Redis (session cache)                                                #
    # ------------------------------------------------------------------ #
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for session/result caching.",
    )
    REDIS_ENABLED: bool = Field(
        default=False,
        description="Set True to use Redis instead of in-memory session store.",
    )

    # ------------------------------------------------------------------ #
    # RAG quality (RAGAS + Reranker)                                      #
    # ------------------------------------------------------------------ #
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace model ID for 2-stage retrieval reranking.",
    )
    RERANKER_ENABLED: bool = Field(
        default=True,
        description="Enable BGE reranker after initial vector search.",
    )
    RAG_TOP_K_INITIAL: int = Field(
        default=20,
        description="Number of candidates fetched before reranking.",
    )
    RAG_TOP_K_FINAL: int = Field(
        default=5,
        description="Number of results returned after reranking.",
    )

    # ------------------------------------------------------------------ #
    # MLflow (fine-tune experiment tracking)                              #
    # ------------------------------------------------------------------ #
    MLFLOW_TRACKING_URI: str = Field(
        default="./mlflow_runs",
        description="MLflow tracking URI (local dir or remote server).",
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="goodjob-finetune",
        description="MLflow experiment name for fine-tuning runs.",
    )

    # ------------------------------------------------------------------ #
    # Ollama / local LLM                                                   #
    # ------------------------------------------------------------------ #
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for the running Ollama server.",
    )
    LOCAL_MODEL_NAME: str = Field(
        default="qwen2.5:7b",
        description="Ollama model tag to use for low-complexity tasks.",
    )

    # ------------------------------------------------------------------ #
    # Vector store                                                         #
    # ------------------------------------------------------------------ #
    CHROMA_PERSIST_DIR: str = Field(
        default="./chroma_db",
        description="Directory where Chroma persists its index files.",
    )

    # ------------------------------------------------------------------ #
    # LLM routing                                                          #
    # ------------------------------------------------------------------ #
    LLM_ROUTER_THRESHOLD: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Complexity score threshold [0, 1].  Tasks with a score above "
            "this value are routed to Claude; tasks at or below go to the "
            "local model (if available)."
        ),
    )

    # ------------------------------------------------------------------ #
    # FastAPI server                                                        #
    # ------------------------------------------------------------------ #
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)

    # ------------------------------------------------------------------ #
    # Streamlit frontend                                                   #
    # ------------------------------------------------------------------ #
    STREAMLIT_PORT: int = Field(default=8501)
    BACKEND_URL: str = Field(
        default="http://localhost:8000",
        description="URL of the FastAPI backend that the Streamlit app calls.",
    )

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #
    @property
    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def has_tavily_key(self) -> bool:
        return bool(self.TAVILY_API_KEY)

    @property
    def has_langsmith_key(self) -> bool:
        return bool(self.LANGCHAIN_API_KEY)

    @property
    def langsmith_enabled(self) -> bool:
        return self.has_langsmith_key and self.LANGCHAIN_TRACING_V2.lower() == "true"


# Module-level singleton – import this instead of instantiating Settings yourself.
settings = Settings()
