# ============================================================
# GoodJob - Docker 이미지 (API / Streamlit 공용, 빌드 대상만 다름)
#
#   docker build --target api      -t goodjob-api .
#   docker build --target frontend -t goodjob-frontend .
#
# 보통은 docker compose up -d 로 함께 실행합니다 (docker-compose.yml).
# ============================================================

FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # 비루트 사용자도 쓸 수 있도록 Playwright 브라우저를 공용 경로에 설치
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# 의존성 레이어 — 코드만 바뀌면 이 레이어는 캐시 재사용
COPY requirements-app.txt requirements.lock.txt ./
RUN apt-get update \
    # 이력서 PDF 한국어 폰트 (frontend/pdf_exporter.py 가 나눔고딕 경로를 찾음)
    && apt-get install -y --no-install-recommends fonts-nanum \
    && pip install -r requirements-app.txt -c requirements.lock.txt \
    # JS 렌더링 공고 페이지(원티드 등) 수집용 Chromium + 시스템 라이브러리
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 goodjob \
    && mkdir -p /app/chroma_db \
    && chown goodjob:goodjob /app/chroma_db

# 앱 코드 (.dockerignore 로 .env, chroma_db, .venv, 학습 데이터 등 제외)
COPY --chown=goodjob:goodjob config ./config
COPY --chown=goodjob:goodjob agents ./agents
COPY --chown=goodjob:goodjob api ./api
COPY --chown=goodjob:goodjob llm ./llm
COPY --chown=goodjob:goodjob rag ./rag
COPY --chown=goodjob:goodjob search ./search

USER goodjob

# ------------------------------------------------------------------ #
# FastAPI                                                              #
# ------------------------------------------------------------------ #
FROM base AS api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------ #
# Streamlit (에이전트를 직접 호출 — API 컨테이너를 거치지 않음)        #
# ------------------------------------------------------------------ #
FROM base AS frontend

COPY --chown=goodjob:goodjob frontend ./frontend
COPY --chown=goodjob:goodjob .streamlit ./.streamlit

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
# .streamlit/config.toml 은 localhost 전용이라 컨테이너 안에서는 0.0.0.0 으로 덮어씀.
# 외부 노출 범위는 docker-compose.yml 의 포트 바인딩(127.0.0.1)으로 제한
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
