"""api/* — FastAPI 엔드포인트 (TestClient, 가짜 LLM, 임시 Chroma)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import INTERVIEW_SET, SAMPLE_POSTING, SAMPLE_REQUIREMENTS, make_fit


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ------------------------------------------------------------------ #
# /profile                                                             #
# ------------------------------------------------------------------ #

def test_profile_status_empty_then_ready(client):
    assert client.get("/profile/status").json() == {"total_chunks": 0, "status": "empty"}

    resp = client.post("/profile/manual", json={
        "name": "테스트 지원자", "skills": ["Python", "FastAPI"],
        "experience": "A사 백엔드 4년", "education": "테스트대학교",
    })
    assert resp.status_code == 201
    stored = resp.json()["chunks_stored"]
    assert stored >= 1

    status = client.get("/profile/status").json()
    assert status == {"total_chunks": stored, "status": "ready"}


def test_profile_manual_requires_name(client):
    assert client.post("/profile/manual", json={"skills": ["Python"]}).status_code == 422


def test_profile_upload_rejects_non_pdf(client):
    resp = client.post("/profile/upload", files={"file": ("resume.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_profile_upload_pdf(client):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Skills: Python, FastAPI")
    pdf_bytes = doc.tobytes()
    doc.close()

    resp = client.post("/profile/upload", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")})
    assert resp.status_code == 201
    assert resp.json()["chunks_stored"] >= 1


def test_profile_upload_broken_pdf_returns_500(client, monkeypatch):
    """손상된 PDF → 원인이 담긴 500. 임시 파일 삭제 실패가 응답을 덮어쓰면 안 됨 (#022)."""
    import tempfile

    created = []
    real_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(*args, **kwargs):
        f = real_ntf(*args, **kwargs)
        created.append(f.name)
        return f

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracking_ntf)
    resp = client.post("/profile/upload", files={"file": ("resume.pdf", b"not a pdf", "application/pdf")})
    assert resp.status_code == 500
    assert "Failed to process PDF" in resp.json()["detail"]

    import os
    assert created and not os.path.exists(created[0]), "임시 파일이 남으면 안 됨"


# ------------------------------------------------------------------ #
# /resume                                                              #
# ------------------------------------------------------------------ #

def test_resume_generate_requires_posting(client):
    resp = client.post("/resume/generate", json={"company_name": "테스트컴퍼니"})
    assert resp.status_code == 400


def test_resume_generate_and_fetch_session(client, fake_llm, loaded_profile):
    fake_llm.structured["JobRequirements"] = SAMPLE_REQUIREMENTS
    fake_llm.structured["FitAnalysis"] = make_fit(0.8)
    fake_llm.structured["InterviewQuestionSet"] = INTERVIEW_SET
    fake_llm.texts = ["# 초안", "# 최종본"]

    resp = client.post("/resume/generate", json={
        "company_name": "테스트컴퍼니", "job_posting_text": SAMPLE_POSTING,
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["fit_score"] == 0.8
    assert body["resume_final"] == "# 최종본"
    assert body["missing_skills"] == ["Kubernetes"]
    assert len(body["interview_questions"]) == 2
    assert body["errors"] == []

    saved = client.get(f"/resume/{body['session_id']}")
    assert saved.status_code == 200
    assert saved.json()["resume_final"] == "# 최종본"
    assert saved.json()["created_at"]


def test_resume_generate_pipeline_error_returns_500(client, monkeypatch):
    import agents.graph

    class Broken:
        def invoke(self, state):
            raise RuntimeError("graph exploded")

    monkeypatch.setattr(agents.graph, "app", Broken())
    resp = client.post("/resume/generate", json={
        "company_name": "테스트컴퍼니", "job_posting_text": SAMPLE_POSTING,
    })
    assert resp.status_code == 500
    assert "graph exploded" in resp.json()["detail"]


def test_resume_unknown_session_404(client):
    assert client.get("/resume/does-not-exist").status_code == 404


def test_resume_url_fetch_failure_returns_400(client, monkeypatch):
    from api.routes import resume

    async def no_text(url):
        return ""

    monkeypatch.setattr(resume, "_fetch_from_url", no_text)
    resp = client.post("/resume/generate", json={
        "company_name": "테스트컴퍼니", "job_url": "https://example.com/job/1",
    })
    assert resp.status_code == 400
    assert "example.com" in resp.json()["detail"]


def test_html_to_text_skips_script_and_style():
    from api.routes.resume import _html_to_text

    html = ("<html><head><title>t</title></head><body><script>var x=1;</script>"
            "<style>.a{}</style><h1>백엔드 개발자</h1><p>Python 필수</p></body></html>")
    assert _html_to_text(html) == "백엔드 개발자\nPython 필수"


# ------------------------------------------------------------------ #
# /recommend                                                           #
# ------------------------------------------------------------------ #

def test_recommend_maps_ranked_matches(client, monkeypatch):
    import agents.job_recommender

    def fake_node(state):
        return {"ranked_matches": [{
            "rank": 1, "company": "테스트컴퍼니", "title": "백엔드", "url": "https://x",
            "fit_score": 0.7, "fit_feedback": "좋음", "requirements": {"job_title": "백엔드 개발자"},
        }], "errors": []}

    monkeypatch.setattr(agents.job_recommender, "job_recommender_node", fake_node)
    resp = client.post("/recommend", json={"query": "Python 백엔드"})
    assert resp.status_code == 200
    match = resp.json()["ranked_matches"][0]
    assert match["job_title"] == "백엔드 개발자"
    assert "requirements" not in match


# ------------------------------------------------------------------ #
# SessionStore                                                         #
# ------------------------------------------------------------------ #

def test_session_store_memory_backend():
    from api.session_store import SessionStore

    store = SessionStore()
    assert store.backend == "memory"
    store.save("s1", {"a": 1})
    assert store.get("s1") == {"a": 1}
    assert store.get("missing") is None


def test_session_store_falls_back_when_redis_unreachable(monkeypatch):
    from api.session_store import SessionStore
    from config.settings import settings

    monkeypatch.setattr(settings, "REDIS_ENABLED", True)
    monkeypatch.setattr(settings, "REDIS_URL", "redis://127.0.0.1:1")
    store = SessionStore()
    assert store.backend == "memory"
    store.save("s1", {"a": 1})
    assert store.get("s1") == {"a": 1}


def test_session_store_falls_back_when_redis_write_fails():
    from api.session_store import SessionStore

    class FlakyRedis:
        def set(self, *a, **k):
            raise ConnectionError("down")

        def get(self, *a, **k):
            raise ConnectionError("down")

    store = SessionStore()
    store._redis = FlakyRedis()
    store.save("s1", {"a": 1})
    assert store.get("s1") == {"a": 1}
