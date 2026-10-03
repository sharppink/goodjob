"""
api/session_store.py

이력서 생성 결과(세션) 저장소.

- REDIS_ENABLED=true  → Redis 에 JSON 으로 저장 (TTL 7일, 서버 재시작 후에도 유지)
- REDIS_ENABLED=false → 프로세스 메모리 딕셔너리 (재시작 시 소실)

Redis 연결에 실패하면 경고를 남기고 메모리 저장소로 자동 폴백합니다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
_KEY_PREFIX = "goodjob:session:"


class SessionStore:
    def __init__(self) -> None:
        self._memory: dict[str, dict[str, Any]] = {}
        self._redis = self._connect_redis() if settings.REDIS_ENABLED else None

    @staticmethod
    def _connect_redis():
        try:
            import redis

            client = redis.Redis.from_url(
                settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2
            )
            client.ping()
            logger.info("[SessionStore] Redis 연결 성공: %s", settings.REDIS_URL)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SessionStore] Redis 연결 실패 → 메모리 저장소 사용: %s", exc)
            return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def save(self, session_id: str, record: dict[str, Any]) -> None:
        if self._redis is not None:
            try:
                self._redis.set(
                    _KEY_PREFIX + session_id,
                    json.dumps(record, ensure_ascii=False),
                    ex=SESSION_TTL_SECONDS,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("[SessionStore] Redis 저장 실패 → 메모리에 저장: %s", exc)
        self._memory[session_id] = record

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        if self._redis is not None:
            try:
                raw = self._redis.get(_KEY_PREFIX + session_id)
                if raw is not None:
                    return json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[SessionStore] Redis 조회 실패: %s", exc)
        return self._memory.get(session_id)


session_store = SessionStore()
