"""
llm/usage_limit.py

공개 배포에서 OpenAI·Tavily 비용이 새지 않도록 계정별·전체 하루 실행 횟수를 제한합니다.

- 로그인한 계정(rag.vectorstore.get_current_user)에만 적용. 로그인 없는 로컬 실행은 제한 없음
- 하루 기준은 한국 시간 자정
- 기능별 한도 settings.USAGE_DAILY_LIMITS, 전체 한도 settings.USAGE_GLOBAL_DAILY_LIMIT,
  제외 계정 settings.USAGE_EXEMPT_EMAILS
- 횟수는 Supabase 가 있으면 goodjob_usage 테이블(재시작해도 유지), 없으면 프로세스 메모리에 저장

사용 예
-------
    from llm.usage_limit import UsageLimitExceeded, consume
    try:
        consume("analysis")
    except UsageLimitExceeded as exc:
        print(exc)          # 사용자에게 보여 줄 안내 문구
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone
from typing import Optional

GLOBAL_KEY = "*all*"  # 전체 합계 행의 user_key / action

ACTION_LABELS = {
    "analysis": "분석 & 이력서 생성",
    "recommend": "공고 추천",
    "search": "공고 자동 검색",
    "image": "공고 이미지 분석",
    "coverletter": "자소서 작성",
    "star": "STAR 변환",
}

_KST = timezone(timedelta(hours=9))
_mem_lock = threading.Lock()
_mem_counts: dict[tuple[str, date, str], int] = {}


class UsageLimitExceeded(Exception):
    """오늘 사용 한도를 넘음. 메시지는 화면에 그대로 보여 줄 수 있는 안내 문구."""


def today() -> date:
    return datetime.now(_KST).date()


def _limited_user() -> Optional[str]:
    """제한 대상 계정 이메일. 로그인하지 않았거나 제외 계정이거나 보호 모드(로컬 처리, API 비용 없음)면 None."""
    from config.settings import settings
    from rag.vectorstore import get_current_user

    email = get_current_user()
    if not email or email in settings.usage_exempt_emails or settings.PRIVACY_MODE:
        return None
    return email


def remaining_all() -> dict[str, int]:
    """기능별 오늘 남은 횟수 (한도가 정해진 기능만). 제한 대상이 아니면 빈 dict."""
    from config.settings import settings
    from rag.vectorstore import user_key

    email = _limited_user()
    if email is None:
        return {}
    used = _get_all(user_key(email))
    return {a: max(n - used.get(a, 0), 0) for a, n in settings.usage_daily_limits.items()}


def consume(action: str) -> None:
    """이 기능을 1회 사용한 것으로 기록. 한도를 넘으면 기록하지 않고 UsageLimitExceeded."""
    from config.settings import settings
    from rag.vectorstore import user_key

    email = _limited_user()
    if email is None:
        return
    label = ACTION_LABELS.get(action, action)
    limit = settings.usage_daily_limits.get(action)
    key = user_key(email)

    if limit is not None and not _increment(key, action, limit):
        raise UsageLimitExceeded(
            f"오늘 '{label}' 사용 한도({limit}회)를 모두 썼습니다. 내일(한국 시간 자정 이후) 다시 이용해 주세요."
        )
    global_limit = settings.USAGE_GLOBAL_DAILY_LIMIT
    if global_limit and not _increment(GLOBAL_KEY, GLOBAL_KEY, global_limit):
        if limit is not None:
            _decrement(key, action)  # 전체 한도로 막힌 실행은 개인 횟수에서 빼 줌
        raise UsageLimitExceeded(
            "오늘 서비스 전체 사용량이 한도에 도달했습니다. 내일 다시 이용해 주세요."
        )


# ------------------------------------------------------------------ #
# 저장소 (Supabase 또는 메모리)                                          #
# ------------------------------------------------------------------ #

def _db_url() -> str:
    from config.settings import settings
    return settings.SUPABASE_DB_URL


def _get_all(key: str) -> dict[str, int]:
    day = today()
    url = _db_url()
    if url:
        from rag.pg_store import USAGE_TABLE_NAME, execute
        rows = execute(
            url, f"select action, count from {USAGE_TABLE_NAME} where user_key = %s and day = %s",
            (key, day), fetch=True,
        )
        return {a: int(c) for a, c in rows}
    with _mem_lock:
        return {a: c for (k, d, a), c in _mem_counts.items() if k == key and d == day}


def _increment(key: str, action: str, limit: int) -> bool:
    """한도 미만이면 1 올리고 True, 이미 한도면 그대로 두고 False (DB 에서는 한 문장으로 처리)."""
    day = today()
    url = _db_url()
    if url:
        from rag.pg_store import USAGE_TABLE_NAME, execute
        if limit <= 0:
            return False
        rows = execute(
            url,
            f"insert into {USAGE_TABLE_NAME} as u (user_key, day, action, count) values (%s, %s, %s, 1) "
            f"on conflict (user_key, day, action) do update set count = u.count + 1 "
            f"where u.count < %s returning u.count",
            (key, day, action, limit), fetch=True,
        )
        return bool(rows)
    with _mem_lock:
        current = _mem_counts.get((key, day, action), 0)
        if current >= limit:
            return False
        _mem_counts[(key, day, action)] = current + 1
        return True


def _decrement(key: str, action: str) -> None:
    day = today()
    url = _db_url()
    if url:
        from rag.pg_store import USAGE_TABLE_NAME, execute
        execute(
            url,
            f"update {USAGE_TABLE_NAME} set count = greatest(count - 1, 0) "
            f"where user_key = %s and day = %s and action = %s",
            (key, day, action),
        )
        return
    with _mem_lock:
        k = (key, day, action)
        _mem_counts[k] = max(_mem_counts.get(k, 0) - 1, 0)


def reset_memory() -> None:
    """메모리 저장 횟수 초기화 (테스트용)."""
    with _mem_lock:
        _mem_counts.clear()
