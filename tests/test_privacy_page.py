"""개인정보처리방침 공개 화면(?page=privacy)과 '내 프로필 삭제'."""

from __future__ import annotations

import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.test_app_login import APP, AUTH_SECRETS, FakeUser


def _app(monkeypatch, email: str | None, query: dict | None = None) -> AppTest:
    monkeypatch.setattr(st, "user", FakeUser(email))
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets = AUTH_SECRETS  # type: ignore[assignment]
    for k, v in (query or {}).items():
        at.query_params[k] = v
    return at.run()


def test_privacy_page_is_public_without_login(monkeypatch):
    at = _app(monkeypatch, None, {"page": "privacy"})
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "개인정보처리방침" in text
    for word in ("SHA-256", "Supabase", "OpenAI", "Tavily", "내 프로필 삭제"):
        assert word in text
    assert "Google 계정으로 로그인" not in [b.label for b in at.button], "방침 화면에서는 로그인 요구 안 함"


def test_login_screen_links_privacy_policy(monkeypatch):
    at = _app(monkeypatch, None)
    assert any("?page=privacy" in c.value for c in at.caption)


def test_delete_my_profile(monkeypatch):
    from rag.vectorstore import VectorStore, set_current_user

    set_current_user("deleter@example.com")
    try:
        VectorStore().add_documents(["삭제될 경력"])
    finally:
        set_current_user(None)

    at = _app(monkeypatch, "deleter@example.com")
    assert at.session_state["profile_indexed"]
    delete = at.button(key="profile_delete_btn")
    assert delete.disabled, "동의 체크 전에는 삭제 버튼이 꺼져 있음"

    at.checkbox(key="profile_delete_confirm").check().run()
    at.button(key="profile_delete_btn").click().run()

    assert not at.exception
    assert not at.session_state["profile_indexed"]
    set_current_user("deleter@example.com")
    try:
        assert VectorStore().count() == 0
    finally:
        set_current_user(None)
