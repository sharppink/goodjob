"""계정별 프로필 분리 (로그인 기능) — 저장소 선택·계정 격리."""

from __future__ import annotations

import threading

import pytest

import rag.vectorstore as vectorstore
from rag.vectorstore import VectorStore, require_login, set_current_user, user_key


def test_profiles_are_separated_per_account():
    set_current_user("a@example.com")
    VectorStore().add_documents(["A 의 경력"])

    set_current_user("b@example.com")
    assert VectorStore().count() == 0, "다른 계정의 프로필이 보이면 안 됨"
    VectorStore().add_documents(["B 의 경력"])

    set_current_user("A@Example.com ")  # 대소문자·공백이 달라도 같은 계정
    assert VectorStore().get_all_documents() == ["A 의 경력"]

    set_current_user(None)
    assert VectorStore().count() == 0, "로그인하지 않은 공용 프로필과도 분리"


def test_user_key_hides_email():
    key = user_key("someone@example.com")
    assert "someone" not in key and len(key) == 32
    assert key == user_key(" SomeOne@Example.com")
    assert user_key(None) == "local"


def test_login_required_blocks_shared_profile():
    """로그인을 쓰는 배포에서 계정이 정해지지 않았으면 공용 데이터를 읽지 말고 실패해야 함."""
    require_login(True)
    with pytest.raises(RuntimeError):
        VectorStore().count()
    set_current_user("a@example.com")
    assert VectorStore().count() == 0


def test_account_does_not_leak_between_sessions():
    """Streamlit 세션은 각자 스레드에서 실행 — 한 세션의 계정 지정이 다른 세션에 보이면 안 됨."""
    seen: dict[str, str | None] = {}
    ready = threading.Barrier(2)

    def session(name: str, email: str) -> None:
        set_current_user(email)
        ready.wait()  # 두 세션이 모두 계정을 정한 뒤에 확인
        seen[name] = vectorstore.get_current_user()

    threads = [threading.Thread(target=session, args=("s1", "a@example.com")),
               threading.Thread(target=session, args=("s2", "b@example.com"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"s1": "a@example.com", "s2": "b@example.com"}


def test_supabase_used_when_configured(monkeypatch):
    from config.settings import settings
    from rag.pg_store import PgCollection

    monkeypatch.setattr(settings, "SUPABASE_DB_URL", "postgresql://user:pw@127.0.0.1:5/db")
    monkeypatch.setattr(PgCollection, "count", lambda self: 0)  # 실제 접속 없이 선택만 확인
    set_current_user("a@example.com")
    vs = VectorStore()
    vs.initialize()
    assert isinstance(vs._collection, PgCollection)
    assert vs._collection._user_key == user_key("a@example.com")


def test_privacy_mode_never_sends_profile_to_supabase(monkeypatch):
    """보호 모드는 프로필을 PC 밖으로 보내지 않는다는 약속 — Supabase 설정이 있어도 로컬 Chroma."""
    from config.settings import settings
    from rag.pg_store import PgCollection

    monkeypatch.setattr(settings, "SUPABASE_DB_URL", "postgresql://user:pw@127.0.0.1:5/db")
    monkeypatch.setattr(settings, "PRIVACY_MODE", True)
    vs = VectorStore()
    vs.initialize()
    assert not isinstance(vs._collection, PgCollection)


def test_allowed_emails_parsing(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "ALLOWED_EMAILS", " A@x.com, b@y.com ,,")
    assert settings.allowed_emails == {"a@x.com", "b@y.com"}
    monkeypatch.setattr(settings, "ALLOWED_EMAILS", "")
    assert settings.allowed_emails == set()
