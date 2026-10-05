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
    # Embeddings                                                           #
    # ------------------------------------------------------------------ #
    USE_OPENAI_EMBEDDINGS: bool = Field(
        default=True,
        description=(
            "True: OpenAI text-embedding-3-small, False: local BGE-M3. "
            "Changing this requires deleting chroma_db and re-indexing."
        ),
    )

    # ------------------------------------------------------------------ #
    # RAG quality (RAGAS + Reranker)                                      #
    # ------------------------------------------------------------------ #
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace model ID for 2-stage retrieval reranking.",
    )
    RERANKER_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable BGE reranker after initial vector search. Off by default: RAGAS eval "
            "showed it lowers context_recall 0.90 -> 0.65 on Korean IT profiles (PROJECT_DOCS #010)."
        ),
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
    # 개인정보 보호 모드 (전체 로컬 처리)                                  #
    # ------------------------------------------------------------------ #
    PRIVACY_MODE: bool = Field(
        default=False,
        description=(
            "True 면 모든 LLM·임베딩을 로컬 Ollama 로 처리하고 OpenAI 호출을 차단. "
            "Streamlit 사이드바 토글로 실행 중 변경 가능."
        ),
    )
    PRIVACY_PARSER_MODEL: str = Field(default="goodjob-parser", description="공고 파싱용 파인튜닝 sLLM")
    PRIVACY_CHAT_MODEL: str = Field(default="qwen2.5:7b", description="적합도·이력서·교정·면접 등 범용 로컬 모델")
    PRIVACY_EMBED_MODEL: str = Field(default="bge-m3", description="로컬 임베딩 모델 (Ollama)")

    PARSE_WITH_LOCAL_LLM: bool = Field(
        default=False,
        description=(
            "True: job_parser always tries the local model first (regardless of posting "
            "length) and falls back to OpenAI on failure. Use with the fine-tuned parser "
            "(LOCAL_MODEL_NAME=goodjob-parser)."
        ),
    )

    # ------------------------------------------------------------------ #
    # Vector store                                                         #
    # ------------------------------------------------------------------ #
    CHROMA_PERSIST_DIR: str = Field(
        default="./chroma_db",
        description="Directory where Chroma persists its index files.",
    )
    CHROMA_HOST: str = Field(
        default="",
        description=(
            "Chroma 서버 주소. 비어 있으면 CHROMA_PERSIST_DIR 로컬 파일 사용(단일 프로세스 전용). "
            "API 와 Streamlit 을 동시에 띄울 때(Docker 등)는 Chroma 서버를 공유해야 함 (PROJECT_DOCS #024)."
        ),
    )
    CHROMA_PORT: int = Field(default=8000, description="Chroma 서버 포트")
    SUPABASE_DB_URL: str = Field(
        default="",
        description=(
            "Supabase Postgres 접속 문자열 (pgvector). 있으면 Chroma 대신 여기에 계정별 프로필을 저장 — "
            "Streamlit Cloud 는 재시작 시 로컬 파일이 지워지므로 영구 보관용 (PROJECT_DOCS #029). "
            "Streamlit Cloud 는 IPv6 직접 접속이 안 될 수 있어 Connection pooler 주소 사용 권장. "
            "개인정보 보호 모드에서는 사용하지 않음 (항상 로컬 Chroma)."
        ),
    )

    # ------------------------------------------------------------------ #
    # 로그인 (Streamlit Google 로그인, Secrets 의 [auth] 가 있을 때만 사용) #
    # ------------------------------------------------------------------ #
    ALLOWED_EMAILS: str = Field(
        default="",
        description=(
            "로그인을 허용할 이메일 목록 (쉼표 구분). 비어 있으면 Google 계정이 있는 누구나 로그인 가능 — "
            "분석마다 OpenAI 비용이 드니 공개 배포 시에는 채워 두는 것을 권장."
        ),
    )

    # ------------------------------------------------------------------ #
    # 사용량 제한 — 로그인한 계정에만 적용 (로그인 없는 로컬 실행은 제한 없음)  #
    # ------------------------------------------------------------------ #
    USAGE_DAILY_LIMITS: str = Field(
        default="analysis=5,recommend=2,search=5,image=5,coverletter=5,star=10",
        description=(
            "계정당 하루(한국 시간) 기능별 실행 횟수 한도. '기능=횟수' 를 쉼표로 구분. "
            "목록에 없는 기능은 제한 없음, 0 이면 사용 불가."
        ),
    )
    USAGE_GLOBAL_DAILY_LIMIT: int = Field(
        default=100,
        ge=0,
        description="모든 계정을 합친 하루 실행 횟수 한도 (OpenAI·Tavily 비용 상한). 0 이면 제한 없음.",
    )
    USAGE_EXEMPT_EMAILS: str = Field(
        default="",
        description="사용량 제한을 받지 않는 이메일 목록 (쉼표 구분, 예: 앱 소유자).",
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
    def allowed_emails(self) -> set[str]:
        return {e.strip().lower() for e in self.ALLOWED_EMAILS.split(",") if e.strip()}

    @property
    def usage_exempt_emails(self) -> set[str]:
        return {e.strip().lower() for e in self.USAGE_EXEMPT_EMAILS.split(",") if e.strip()}

    @property
    def usage_daily_limits(self) -> dict[str, int]:
        limits: dict[str, int] = {}
        for item in self.USAGE_DAILY_LIMITS.split(","):
            name, _, value = item.partition("=")
            if name.strip() and value.strip().isdigit():
                limits[name.strip()] = int(value)
        return limits

    @property
    def has_langsmith_key(self) -> bool:
        return bool(self.LANGCHAIN_API_KEY)

    @property
    def langsmith_enabled(self) -> bool:
        return self.has_langsmith_key and self.LANGCHAIN_TRACING_V2.lower() == "true"


# Module-level singleton – import this instead of instantiating Settings yourself.
settings = Settings()
