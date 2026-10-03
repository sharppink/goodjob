"""
api/main.py

FastAPI application entry point for the GoodJob service.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import company, profile, recommend, resume
from config.settings import settings
from config.tracing import setup_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# App factory                                                          #
# ------------------------------------------------------------------ #

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Fully configured application instance.
    """
    application = FastAPI(
        title="GoodJob API",
        description=(
            "AI-powered job matching and resume generation service. "
            "Upload your profile, provide a target company, and receive "
            "a tailored resume in seconds."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ---- CORS (allow Streamlit frontend and local dev) ----
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            settings.BACKEND_URL,
            "http://localhost:8501",    # Streamlit default
            "http://127.0.0.1:8501",
            "http://localhost:3000",    # React / Next.js dev server
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Include routers ----
    application.include_router(profile.router, prefix="/profile", tags=["Profile"])
    application.include_router(company.router, prefix="/company", tags=["Company"])
    application.include_router(resume.router, prefix="/resume", tags=["Resume"])
    application.include_router(recommend.router, prefix="/recommend", tags=["Recommend"])

    # ---- Lifecycle events ----
    @application.on_event("startup")
    async def _startup() -> None:
        setup_tracing()
        logger.info("GoodJob API starting up …")

    @application.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("GoodJob API shutting down.")

    return application


app = create_app()


# ------------------------------------------------------------------ #
# Routes defined directly on the root app                             #
# ------------------------------------------------------------------ #

@app.get("/", tags=["Root"])
async def root() -> dict:
    """Redirect hint for API root."""
    return {"message": "GoodJob API is running. Visit /docs for the API reference."}


@app.get("/health", tags=["Root"])
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns a simple status dict so load balancers / k8s probes can
    verify the service is alive.
    """
    return {"status": "ok", "service": "goodjob-api", "version": app.version}


# ------------------------------------------------------------------ #
# Dev runner                                                           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
