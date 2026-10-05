"""
Supabase(pgvector) 저장소 통합 테스트 — 실제 Postgres 가 있을 때만 실행.

CI 는 pgvector 컨테이너를 띄우고 TEST_DATABASE_URL 을 넘김 (.github/workflows/ci.yml).
로컬에서는 TEST_DATABASE_URL 이 없으면 건너뜀. 실제 Supabase 주소는 넣지 말 것 (테이블을 비움).
"""

from __future__ import annotations

import os

import pytest

DB_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL 없음 (CI 의 pgvector 컨테이너에서 실행)")


@pytest.fixture
def pg(monkeypatch):
    import psycopg
    from config.settings import settings
    from rag import pg_store

    monkeypatch.setattr(settings, "SUPABASE_DB_URL", DB_URL)
    pg_store._schema_ready.clear()
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(f"drop table if exists {pg_store.TABLE_NAME}")
        conn.execute(f"drop table if exists {pg_store.USAGE_TABLE_NAME}")
        conn.execute(f"drop table if exists {pg_store.APPLICATIONS_TABLE_NAME}")
    yield DB_URL


def test_profile_roundtrip_and_account_isolation(pg):
    from rag.vectorstore import VectorStore, set_current_user

    set_current_user("a@example.com")
    vs = VectorStore()
    vs.add_documents(["Python FastAPI 백엔드", "React 프론트엔드", "Kubernetes 운영"],
                     metadatas=[{"source": "manual"}, {}, {}])
    assert vs.count() == 3
    assert vs.similarity_search("FastAPI 백엔드", k=1) == ["Python FastAPI 백엔드"]
    assert len(vs.similarity_search("아무거나", k=10)) == 3
    entries = vs.get_all_entries()
    assert [e["document"] for e in entries] == ["Python FastAPI 백엔드", "React 프론트엔드", "Kubernetes 운영"]
    assert entries[0]["metadata"] == {"source": "manual"}

    set_current_user("b@example.com")
    other = VectorStore()
    assert other.count() == 0
    other.add_documents(["B 의 경력"])
    other.delete_all()
    assert other.count() == 0

    set_current_user("a@example.com")
    assert sorted(VectorStore().get_all_documents()) == sorted(
        ["Python FastAPI 백엔드", "React 프론트엔드", "Kubernetes 운영"]
    ), "다른 계정의 삭제가 영향을 주면 안 됨"


def test_table_has_row_level_security(pg):
    """Supabase Data API(anon 키)로 프로필이 노출되지 않도록 RLS 가 켜져 있어야 함."""
    import psycopg
    from rag import pg_store
    from rag.vectorstore import VectorStore

    VectorStore().count()  # 스키마 생성
    with psycopg.connect(DB_URL) as conn:
        rows = conn.execute(
            "select relname, relrowsecurity from pg_class where relname = any(%s) order by relname",
            ([pg_store.TABLE_NAME, pg_store.USAGE_TABLE_NAME, pg_store.APPLICATIONS_TABLE_NAME],),
        ).fetchall()
    assert rows == [(pg_store.APPLICATIONS_TABLE_NAME, True), (pg_store.TABLE_NAME, True),
                    (pg_store.USAGE_TABLE_NAME, True)]


def test_email_is_not_stored(pg):
    import psycopg
    from rag import pg_store
    from rag.vectorstore import VectorStore, set_current_user

    set_current_user("secret.person@example.com")
    VectorStore().add_documents(["경력"])
    with psycopg.connect(DB_URL) as conn:
        dump = str(conn.execute(f"select * from {pg_store.TABLE_NAME}").fetchall())
    assert "secret.person" not in dump


def test_usage_limit_persists_in_db(pg, monkeypatch):
    """사용 횟수가 Supabase 에 저장돼 재시작(메모리 초기화) 후에도 유지되고, 한도에서 멈춤."""
    import psycopg
    from config.settings import settings
    from llm import usage_limit
    from rag import pg_store
    from rag.vectorstore import set_current_user

    monkeypatch.setattr(settings, "USAGE_DAILY_LIMITS", "analysis=2")
    monkeypatch.setattr(settings, "USAGE_GLOBAL_DAILY_LIMIT", 100)
    monkeypatch.setattr(settings, "USAGE_EXEMPT_EMAILS", "")
    set_current_user("quota@example.com")
    try:
        usage_limit.consume("analysis")
        usage_limit.reset_memory()  # DB 에 저장되므로 메모리와 무관
        assert usage_limit.remaining_all() == {"analysis": 1}
        usage_limit.consume("analysis")
        with pytest.raises(usage_limit.UsageLimitExceeded):
            usage_limit.consume("analysis")
    finally:
        set_current_user(None)

    with psycopg.connect(DB_URL) as conn:
        rows = conn.execute(
            f"select user_key, action, count from {pg_store.USAGE_TABLE_NAME} order by action"
        ).fetchall()
    assert [(a, c) for _, a, c in rows] == [("*all*", 2), ("analysis", 2)]
    assert "quota" not in str(rows), "이메일 원문은 저장하지 않음"


def test_applications_roundtrip_in_db(pg):
    """지원 현황 카드가 Supabase 에 계정별로 저장되고 단계 이동·갱신·삭제가 반영됨."""
    from rag.vectorstore import set_current_user
    from tracker import applications as ap

    set_current_user("board@example.com")
    try:
        card = ap.save_draft("당근", "백엔드", "# v1", 0.7, "공고")
        ap.set_stages({card["id"]: "면접"})
        ap.save_draft("당근", "백엔드", "# v2", 0.9, "공고")  # 같은 공고 → 갱신
        apps = ap.list_applications()
        assert len(apps) == 1
        assert (apps[0]["stage"], apps[0]["resume"], apps[0]["fit_score"]) == ("면접", "# v2", 0.9)

        set_current_user("other@example.com")
        assert ap.list_applications() == []

        set_current_user("board@example.com")
        ap.delete_application(card["id"])
        assert ap.list_applications() == []
    finally:
        set_current_user(None)

