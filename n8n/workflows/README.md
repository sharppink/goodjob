# n8n Workflows for GoodJob

This directory stores exported n8n workflow JSON files that complement the
GoodJob Python backend.

## Planned Workflows

### 1. `profile_ingest_webhook.json`
Trigger: Webhook (POST /webhook/profile)
- Receives profile data from external sources (e.g. LinkedIn, Notion).
- Calls `POST /profile/manual` on the GoodJob FastAPI backend.
- Sends a confirmation email via Gmail node.

### 2. `daily_job_alert.json`
Trigger: Cron (every day at 08:00 KST)
- Reads a list of target companies from a Google Sheet.
- For each company, calls `POST /company/search`.
- Filters new postings since yesterday.
- Sends a Slack/email digest with matching positions.

### 3. `resume_generation_pipeline.json`
Trigger: Webhook or manual
- Accepts `company_name` + `job_posting_text`.
- Calls `POST /resume/generate`.
- Polls until `fit_score` and `resume_final` are available.
- Saves the result to Notion / Google Docs.

## How to Import

1. Open your n8n instance (`http://localhost:5678`).
2. Go to **Workflows → Import from File**.
3. Select the `.json` file from this directory.
4. Update credential nodes (API keys, OAuth) as required.
5. Activate the workflow.

## Environment Variables Required in n8n

| Variable | Description |
|---|---|
| `GOODJOB_API_URL` | FastAPI backend URL (e.g. `http://localhost:8000`) |
| `SLACK_WEBHOOK_URL` | Incoming webhook for job alerts |
| `GMAIL_OAUTH_CREDENTIALS` | Gmail OAuth2 credentials |
| `NOTION_API_KEY` | Notion integration token |
