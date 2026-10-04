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
        row = conn.execute(
            "select relrowsecurity from pg_class where relname = %s", (pg_store.TABLE_NAME,)
        ).fetchone()
    assert row == (True,)


def test_email_is_not_stored(pg):
    import psycopg
    from rag import pg_store
    from rag.vectorstore import VectorStore, set_current_user

    set_current_user("secret.person@example.com")
    VectorStore().add_documents(["경력"])
    with psycopg.connect(DB_URL) as conn:
        dump = str(conn.execute(f"select * from {pg_store.TABLE_NAME}").fetchall())
    assert "secret.person" not in dump
