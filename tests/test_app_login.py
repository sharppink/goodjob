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


# 로그인 설정이 없는 경우. at.secrets 를 비워 두면 AppTest 가 이 PC 의 실제
# .streamlit/secrets.toml (로컬 로그인 테스트용 실제 키) 을 읽으므로 무관한 값으로 채움
NO_AUTH_SECRETS = {"GOODJOB_TEST_NO_AUTH": "1"}


def _run(monkeypatch, email: str | None, secrets: dict = AUTH_SECRETS) -> AppTest:
    monkeypatch.setattr(st, "user", FakeUser(email))
    at = AppTest.from_file(APP, default_timeout=60)
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
    at = _run(monkeypatch, None, secrets=NO_AUTH_SECRETS)
    assert not at.exception
    assert at.header[0].value == "👤 프로필 등록"
    assert at.success, "로그인을 쓰지 않으면 기존 공용 프로필을 그대로 사용"
    assert "로그아웃" not in [b.label for b in at.sidebar.button]


def test_logged_in_user_can_view_own_profile(monkeypatch):
    """'내 프로필 보기'에는 로그인한 계정이 등록한 내용만 보여야 함."""
    from rag.profile_loader import ProfileLoader
    from rag.vectorstore import set_current_user

    set_current_user("other@example.com")
    ProfileLoader().load_from_dict({"name": "다른사람", "skills": ["Go"], "experience": "다른회사 경력"})
    set_current_user("me@example.com")
    ProfileLoader().load_from_dict({"name": "나", "skills": ["Python", "FastAPI"],
                                    "experience": "샘플컴퍼니 백엔드 개발"})
    set_current_user(None)

    at = _run(monkeypatch, "me@example.com")
    assert not at.exception
    assert at.expander and at.expander[0].label == "📋 내 프로필 보기"
    shown = "\n".join(t.value for t in at.expander[0].text)
    assert "샘플컴퍼니" in shown and "Python" in shown
    assert "다른회사" not in shown and "다른사람" not in shown
    labels = "\n".join(m.value for m in at.expander[0].markdown)
    assert "✏️ 직접 입력" in labels
