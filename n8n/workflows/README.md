# n8n Workflows for GoodJob

GoodJob FastAPI 백엔드를 호출하는 n8n 워크플로우 3종입니다.
모든 워크플로우는 비활성(`active: false`) 상태로 들어 있으니, import 후 환경변수를 설정하고 활성화하세요.

## Workflows

### 1. `profile_ingest_webhook.json` — 외부에서 프로필 등록
- Trigger: Webhook `POST /webhook/goodjob/profile`
- Body 를 그대로 `POST /profile/manual` 에 전달 → 저장된 청크 수를 응답
- Body 예시: `{"name": "홍길동", "skills": ["Python"], "experience": "..."}`

### 2. `daily_job_alert.json` — 매일 아침 추천 공고 알림
- Trigger: 평일 08:00 (Asia/Seoul)
- `Keywords` Code 노드의 키워드 목록마다 `POST /recommend` 호출
  (Tavily 검색 → 개별 공고만 필터 → 적합도 분석 → 기준 이상 상위 5개)
- 결과를 하나의 메시지로 묶어 Slack Incoming Webhook 으로 전송
- 키워드당 30~90초 소요 (HTTP 타임아웃 300초로 설정)

### 3. `resume_generation_pipeline.json` — 이력서 생성 요청
- Trigger: Webhook `POST /webhook/goodjob/resume`
- Body: `{"company_name": "...", "job_posting_text": "..." | "job_url": "...", "generate_interview": true}`
- `POST /resume/generate` 호출 (동기 API라 폴링 불필요, 1~2분 소요)
- 이력서 생성 여부에 따라 Slack 으로 성공(적합도·보유/부족 기술·면접 질문 수) 또는 기준 미달 알림
- 웹훅 응답으로 전체 결과(JSON) 반환 — 세션 ID 로 `GET /resume/{session_id}` 재조회 가능

## How to Import

1. n8n 실행 (`npx n8n` 또는 Docker, 기본 `http://localhost:5678`)
2. **Workflows → Import from File** 에서 `.json` 선택
3. 아래 환경변수 설정 후 워크플로우 활성화

## Environment Variables (n8n 프로세스에 설정)

| Variable | Description |
|---|---|
| `GOODJOB_API_URL` | FastAPI 백엔드 URL (예: `http://localhost:8000`) |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL |
| `N8N_BLOCK_ENV_ACCESS_IN_NODE` | `false` — 워크플로우 표현식에서 `$env` 를 읽기 위해 필요 |

## 검증 상태 (2026-10-03)

- JSON 구조, 노드 연결, Code 노드 JS 문법: 스크립트로 검증 완료
- `Build digest` JS: 실제 `/recommend` 응답으로 실행해 Slack 메시지 생성 확인
- 호출하는 API(`/profile/manual`, `/recommend`, `/resume/generate`): 실제 호출 확인
- **n8n 앱에 import 해서 실행하는 것은 아직 미검증** (노드 typeVersion 호환성 등)
