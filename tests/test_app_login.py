"""Streamlit 앱 로그인 흐름 — st.user 를 가짜 사용자로 바꿔 AppTest 로 실행."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")
AUTH_SECRETS = {
    "auth": {
        "redirect_uri": "http://localhost:8501/oauth2callback",
        "cookie_secret": "test-cookie-secret",
        "client_id": "test.apps.googleusercontent.com",
        "client_secret": "test",
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
    }
}


class FakeUser:
    def __init__(self, email: str | None) -> None:
        self.is_logged_in = email is not None
        self._email = email

    def get(self, key: str, default=None):
        return self._email if key == "email" else default


def _run(monkeypatch, email: str | None, secrets: dict | None = AUTH_SECRETS) -> AppTest:
    monkeypatch.setattr(st, "user", FakeUser(email))
    at = AppTest.from_file(APP, default_timeout=60)
    if secrets:
        at.secrets = secrets  # type: ignore[assignment]
    return at.run()


def test_login_screen_before_login(monkeypatch):
    at = _run(monkeypatch, None)
    assert not at.exception
    assert [b.label for b in at.button] == ["Google 계정으로 로그인"]
    assert not at.header, "로그인 전에는 앱 화면이 보이면 안 됨"


def test_logged_in_user_gets_own_empty_profile(monkeypatch, loaded_profile):
    """공용 프로필(loaded_profile)이 있어도 새 계정에는 보이지 않아야 함."""
    at = _run(monkeypatch, "new.user@example.com")
    assert not at.exception
    assert at.header[0].value == "👤 프로필 등록"
    assert not at.success, "다른 프로필의 '등록되어 있습니다' 안내가 보이면 안 됨"
    assert any("new.user@example.com" in c.value for c in at.sidebar.caption)
    assert "로그아웃" in [b.label for b in at.sidebar.button]


def test_email_not_in_allowlist_is_blocked(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "ALLOWED_EMAILS", "owner@example.com")
    at = _run(monkeypatch, "stranger@example.com")
    assert not at.exception
    assert at.error and "사용 권한이 없습니다" in at.error[0].value
    assert not at.header


def test_without_auth_secrets_app_works_as_before(monkeypatch, loaded_profile):
    at = _run(monkeypatch, None, secrets=None)
    assert not at.exception
    assert at.header[0].value == "👤 프로필 등록"
    assert at.success, "로그인을 쓰지 않으면 기존 공용 프로필을 그대로 사용"
    assert "로그아웃" not in [b.label for b in at.sidebar.button]
