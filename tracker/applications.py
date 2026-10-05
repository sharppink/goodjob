"""
tracker/applications.py

지원 현황 보드 — 생성한 이력서를 '작성 중' 에 자동으로 넣고, 사용자가 단계별로 옮겨 관리합니다.

- 저장 위치는 프로필(VectorStore)과 같은 규칙:
    Supabase 설정이 있고 개인정보 보호 모드가 아니면 goodjob_applications 테이블 (계정별)
    그 외에는 이 PC 의 JSON 파일 (CHROMA_PERSIST_DIR/applications.json, 계정 해시별로 구분)
- 계정은 rag.vectorstore 의 현재 로그인 계정(해시)으로 구분. 이메일 원문은 저장하지 않음
- 같은 공고로 이력서를 다시 만들면 새 카드 대신 기존 카드의 이력서·점수를 갱신 (단계는 유지)

보드 화면(streamlit-sortables)과 주고받는 형식 변환도 여기 둡니다 (board_columns / stage_changes).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

STAGES: list[str] = ["작성 중", "지원 완료", "서류 합격", "면접", "최종 합격", "불합격"]
DRAFT_STAGE = STAGES[0]

_FIELDS = ["id", "company", "job_title", "stage", "fit_score", "resume", "posting_hash", "created_at", "updated_at"]
_file_lock = threading.Lock()


# ------------------------------------------------------------------ #
# 공개 API                                                             #
# ------------------------------------------------------------------ #

def save_draft(
    company: str, job_title: str, resume: str, fit_score: Optional[float], posting_text: str
) -> dict[str, Any]:
    """생성한 이력서를 '작성 중' 카드로 저장. 같은 공고의 카드가 있으면 이력서·점수만 갱신해 반환."""
    posting_hash = hashlib.sha256(f"{company}\n{posting_text}".encode("utf-8")).hexdigest()[:16]
    now = _now()
    store = _store()
    existing = next((a for a in store.list() if a["posting_hash"] == posting_hash), None)
    if existing:
        existing.update(resume=resume, fit_score=fit_score, job_title=job_title or existing["job_title"],
                        updated_at=now)
        store.update(existing["id"], resume=resume, fit_score=fit_score,
                     job_title=existing["job_title"], updated_at=now)
        return existing
    app = {
        "id": uuid.uuid4().hex, "company": company or "회사 미입력", "job_title": job_title or "",
        "stage": DRAFT_STAGE, "fit_score": fit_score, "resume": resume,
        "posting_hash": posting_hash, "created_at": now, "updated_at": now,
    }
    store.insert(app)
    return app


def list_applications() -> list[dict[str, Any]]:
    """현재 계정의 지원 카드 (최근 수정 순)."""
    return sorted(_store().list(), key=lambda a: a["updated_at"], reverse=True)


def set_stages(changes: dict[str, str]) -> None:
    """{카드 id: 새 단계} 반영. 알 수 없는 단계는 무시."""
    now = _now()
    store = _store()
    for app_id, stage in changes.items():
        if stage in STAGES:
            store.update(app_id, stage=stage, updated_at=now)


def delete_application(app_id: str) -> None:
    _store().delete(app_id)


# ------------------------------------------------------------------ #
# 보드 형식 변환                                                        #
# ------------------------------------------------------------------ #

_LABEL_ID = re.compile(r"\[#([0-9a-f]{6})\]$")


def card_label(app: dict[str, Any]) -> str:
    """보드 카드 문구. 끝의 [#짧은id] 로 드래그 결과를 카드에 다시 연결."""
    title = " · ".join(x for x in [app["company"], app["job_title"]] if x)
    score = f" {app['fit_score']:.0%}" if isinstance(app.get("fit_score"), (int, float)) else ""
    return f"{title}{score} [#{app['id'][:6]}]"


def board_columns(apps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """sort_items(multi_containers=True) 에 넘길 단계별 칸."""
    return [
        {"header": stage, "items": [card_label(a) for a in apps if a["stage"] == stage]}
        for stage in STAGES
    ]


def stage_changes(apps: list[dict[str, Any]], result: list[dict[str, Any]]) -> dict[str, str]:
    """드래그 후 칸 상태(result)와 저장된 단계를 비교해 바뀐 카드만 {id: 새 단계} 로 반환.

    칸은 순서(index)로 단계에 대응시킴 — 헤더 문구에 의존하지 않음.
    """
    by_short = {a["id"][:6]: a for a in apps}
    changes: dict[str, str] = {}
    for stage, column in zip(STAGES, result or []):
        for label in column.get("items", []):
            match = _LABEL_ID.search(label)
            app = by_short.get(match.group(1)) if match else None
            if app and app["stage"] != stage:
                changes[app["id"]] = stage
    return changes


# ------------------------------------------------------------------ #
# 저장소                                                               #
# ------------------------------------------------------------------ #

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _owner_key() -> str:
    from rag.vectorstore import _login_required, get_current_user, user_key

    email = get_current_user()
    if email is None and _login_required:
        raise RuntimeError("로그인한 계정이 없어 지원 현황에 접근할 수 없습니다.")
    return user_key(email)


def _store():
    from config.settings import settings

    key = _owner_key()
    if settings.SUPABASE_DB_URL and not settings.PRIVACY_MODE:
        return _PgStore(settings.SUPABASE_DB_URL, key)
    return _FileStore(Path(settings.CHROMA_PERSIST_DIR) / "applications.json", key)


class _FileStore:
    """이 PC 의 JSON 파일 — {계정 해시: [카드, ...]}."""

    def __init__(self, path: Path, key: str) -> None:
        self._path = path
        self._key = key

    def _read_all(self) -> dict[str, list[dict[str, Any]]]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write_all(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def list(self) -> list[dict[str, Any]]:
        with _file_lock:
            return [dict(a) for a in self._read_all().get(self._key, [])]

    def insert(self, app: dict[str, Any]) -> None:
        with _file_lock:
            data = self._read_all()
            data.setdefault(self._key, []).append(app)
            self._write_all(data)

    def update(self, app_id: str, **fields: Any) -> None:
        with _file_lock:
            data = self._read_all()
            for a in data.get(self._key, []):
                if a["id"] == app_id:
                    a.update(fields)
            self._write_all(data)

    def delete(self, app_id: str) -> None:
        with _file_lock:
            data = self._read_all()
            data[self._key] = [a for a in data.get(self._key, []) if a["id"] != app_id]
            self._write_all(data)


class _PgStore:
    """Supabase goodjob_applications 테이블 (rag/pg_store.py 의 공유 접속 사용)."""

    def __init__(self, url: str, key: str) -> None:
        self._url = url
        self._key = key

    def _run(self, sql: str, params: tuple = (), fetch: bool = False) -> list[tuple]:
        from rag.pg_store import execute
        return execute(self._url, sql, params, fetch)

    def list(self) -> list[dict[str, Any]]:
        from rag.pg_store import APPLICATIONS_TABLE_NAME as T
        rows = self._run(
            f"select id, company, job_title, stage, fit_score, resume, posting_hash, "
            f"created_at::text, updated_at::text from {T} where user_key = %s",
            (self._key,), fetch=True,
        )
        return [dict(zip(_FIELDS, r)) for r in rows]

    def insert(self, app: dict[str, Any]) -> None:
        from rag.pg_store import APPLICATIONS_TABLE_NAME as T
        self._run(
            f"insert into {T} (id, user_key, company, job_title, stage, fit_score, resume, posting_hash, "
            f"created_at, updated_at) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            f"on conflict (id) do nothing",
            (app["id"], self._key, app["company"], app["job_title"], app["stage"], app["fit_score"],
             app["resume"], app["posting_hash"], app["created_at"], app["updated_at"]),
        )

    def update(self, app_id: str, **fields: Any) -> None:
        from rag.pg_store import APPLICATIONS_TABLE_NAME as T
        cols = [c for c in fields if c in _FIELDS and c != "id"]
        if not cols:
            return
        sets = ", ".join(f"{c} = %s" for c in cols)
        self._run(
            f"update {T} set {sets} where id = %s and user_key = %s",
            (*[fields[c] for c in cols], app_id, self._key),
        )

    def delete(self, app_id: str) -> None:
        from rag.pg_store import APPLICATIONS_TABLE_NAME as T
        self._run(f"delete from {T} where id = %s and user_key = %s", (app_id, self._key))
