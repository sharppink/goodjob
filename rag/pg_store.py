"""
rag/pg_store.py

Supabase Postgres(pgvector) 에 프로필 청크를 저장하는 컬렉션.

VectorStore 가 Chroma 컬렉션과 같은 방식으로 쓸 수 있도록 add / query / get / delete / count
만 같은 모양으로 제공합니다. 모든 계정의 청크는 테이블 하나에 두고 (collection, user_key) 로 나눕니다.

- 임베딩 차원은 고정하지 않음 (OpenAI 1536 / bge-m3 1024). 계정별 청크가 수십 개 수준이라
  인덱스 없이 해당 계정 행만 거리순 정렬해도 충분히 빠름
- Supabase 는 public 테이블을 Data API 로 노출하므로 RLS 를 켜고 정책을 만들지 않음 →
  anon/authenticated 키로는 접근 불가, 테이블 소유자(접속 문자열의 postgres 계정)만 사용
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

TABLE_NAME = "goodjob_profile_chunks"
USAGE_TABLE_NAME = "goodjob_usage"  # 계정별 하루 사용 횟수 (llm/usage_limit.py)

_SCHEMA_SQL = f"""
create extension if not exists vector;
create table if not exists {TABLE_NAME} (
    id          text primary key,
    collection  text not null,
    user_key    text not null,
    content     text not null,
    metadata    jsonb not null default '{{}}'::jsonb,
    embedding   vector not null,
    created_at  timestamptz not null default now()
);
create index if not exists {TABLE_NAME}_owner_idx on {TABLE_NAME} (collection, user_key);
create table if not exists {USAGE_TABLE_NAME} (
    user_key    text not null,
    day         date not null,
    action      text not null,
    count       integer not null default 0,
    primary key (user_key, day, action)
);
alter table {TABLE_NAME} enable row level security;
alter table {USAGE_TABLE_NAME} enable row level security;
-- Supabase 는 public 새 테이블에 anon/authenticated 권한을 기본으로 줌 → 회수
-- (일반 Postgres(CI)에는 이 역할이 없어서 있는 경우에만)
do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        execute 'revoke all on table {TABLE_NAME} from anon';
        execute 'revoke all on table {USAGE_TABLE_NAME} from anon';
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        execute 'revoke all on table {TABLE_NAME} from authenticated';
        execute 'revoke all on table {USAGE_TABLE_NAME} from authenticated';
    end if;
end $$;
"""

# 프로세스 전체에서 접속 하나를 공유 (Streamlit 세션마다 새로 접속하면 Supabase 접속 수 한도에 걸림)
_conn_lock = threading.Lock()
_conn: Any = None
_conn_url: Optional[str] = None
_schema_ready: set[str] = set()


def _connect(url: str):
    import psycopg

    # prepare_threshold=None: Supabase transaction pooler(6543) 는 prepared statement 미지원
    return psycopg.connect(url, autocommit=True, prepare_threshold=None, connect_timeout=10)


def _get_conn(url: str):
    global _conn, _conn_url
    if _conn is None or _conn.closed or _conn_url != url:
        _conn = _connect(url)
        _conn_url = url
    if url not in _schema_ready:
        with _conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        _schema_ready.add(url)
    return _conn


def execute(url: str, sql: str, params: tuple = (), fetch: bool = False) -> list[tuple]:
    """공유 접속으로 SQL 하나를 실행 (스키마는 첫 접속 때 만들어짐)."""
    global _conn
    import psycopg

    with _conn_lock:
        for attempt in range(2):
            try:
                conn = _get_conn(url)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall() if fetch else []
            except psycopg.OperationalError:
                # 오래 쉬면 pooler 가 접속을 끊음 → 한 번만 다시 접속
                _conn = None
                if attempt:
                    raise
    return []


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


class PgCollection:
    """한 계정·한 모드의 프로필 청크 묶음 (Chroma Collection 과 같은 호출 방식)."""

    def __init__(self, url: str, collection: str, user_key: str) -> None:
        self._url = url
        self.name = collection
        self._user_key = user_key

    def _run(self, sql: str, params: tuple = (), fetch: bool = False) -> list[tuple]:
        return execute(self._url, sql, params, fetch)

    def count(self) -> int:
        rows = self._run(
            f"select count(*) from {TABLE_NAME} where collection = %s and user_key = %s",
            (self.name, self._user_key), fetch=True,
        )
        return int(rows[0][0])

    def add(self, documents: list[str], embeddings: list[list[float]],
            metadatas: list[dict], ids: list[str]) -> None:
        for doc_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self._run(
                f"insert into {TABLE_NAME} (id, collection, user_key, content, metadata, embedding) "
                # 접속이 끊겨 다시 보낸 경우 같은 행이 두 번 들어가지 않도록
                f"values (%s, %s, %s, %s, %s::jsonb, %s::vector) on conflict (id) do nothing",
                (doc_id, self.name, self._user_key, doc, json.dumps(meta, ensure_ascii=False),
                 _vector_literal(emb)),
            )

    def query(self, query_embeddings: list[list[float]], n_results: int) -> dict[str, Any]:
        rows = self._run(
            f"select content from {TABLE_NAME} where collection = %s and user_key = %s "
            f"order by embedding <=> %s::vector limit %s",
            (self.name, self._user_key, _vector_literal(query_embeddings[0]), n_results),
            fetch=True,
        )
        return {"documents": [[r[0] for r in rows]]}

    def get(self, include: Optional[list[str]] = None) -> dict[str, Any]:
        rows = self._run(
            f"select id, content, metadata from {TABLE_NAME} where collection = %s and user_key = %s "
            f"order by created_at, id",
            (self.name, self._user_key), fetch=True,
        )
        return {"ids": [r[0] for r in rows], "documents": [r[1] for r in rows],
                "metadatas": [r[2] or {} for r in rows]}

    def delete(self, ids: list[str]) -> None:
        self._run(
            f"delete from {TABLE_NAME} where collection = %s and user_key = %s and id = any(%s)",
            (self.name, self._user_key, ids),
        )
