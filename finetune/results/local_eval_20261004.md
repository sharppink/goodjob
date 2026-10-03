# 로컬 비교 평가 (2026-10-04)

- 환경: 이 PC (RTX 2060 SUPER 8GB), Ollama 0.35.1, JSON 모드(format=json), num_ctx 8192
- 모델: 둘 다 Qwen2.5-7B-Instruct Q4_K_M — 차이는 파인튜닝 여부뿐
  - base: `qwen2.5:7b` (Ollama 공식)
  - fine-tuned: `goodjob-parser` (train 208건, 2 epoch, QLoRA r16, train_loss 0.164)
- 정답: gpt-4o Structured Output 라벨 (정규화 적용)

## eval 23건 (전체)

| metric | qwen2.5:7b | goodjob-parser |
|---|---|---|
| json_valid | 1.000 | 1.000 |
| is_job_posting_acc | 0.957 | 0.913 |
| required_skills_f1 | 0.660 | 0.630 |
| preferred_skills_f1 | 0.672 | 0.757 |
| keywords_f1 | 0.458 | 0.548 |
| experience_years_acc | 0.783 | 0.870 |
| job_title_match | 0.826 | 0.783 |
| company_name_match | 0.957 | 0.913 |
| no_chinese_leak | 1.000 | 1.000 |
| sec_per_sample | 8.5 | 7.6 |

## 목록 페이지 2건 제외 21건 (권장 기준)

eval 23건 중 2건이 사람인 직무 카테고리 목록 페이지(`/jobs/list/job-category`)로, 수집 필터 누락으로
들어간 데이터이며 라벨도 불안정함 (한 건은 "공고", 한 건은 "비공고"). 23건 기준 공고판정·회사명·직무명
하락은 전부 이 중 1건 때문이었음.

| metric | qwen2.5:7b | goodjob-parser | 변화 |
|---|---|---|---|
| json_valid | 1.000 | 1.000 | = |
| is_job_posting_acc | 1.000 | 1.000 | = |
| required_skills_f1 | 0.681 | 0.645 | -0.036 |
| preferred_skills_f1 | 0.593 | 0.731 | +0.138 |
| keywords_f1 | 0.487 | 0.592 | +0.105 |
| experience_years_acc | 0.810 | 0.905 | +0.095 |
| job_title_match | 0.857 | 0.857 | = |
| company_name_match | 1.000 | 1.000 | = |
| no_chinese_leak | 1.000 | 1.000 | = |

## 해석

- 파인튜닝 효과: 우대 기술·키워드·경력 연수 추출에서 +0.10~0.14 개선, 나머지는 동일, 필수 기술 소폭 하락
- 평가 21~23건은 작은 표본 — 1건 차이가 정확도 0.043~0.048 에 해당하므로 소폭 차이는 오차 범위
- gpt-4o(정답 기준 1.0) 대비 기술 추출은 여전히 0.6~0.75 수준 → 앱 기본값은 OpenAI 유지가 품질상 유리
- Colab 결과표(제약 없는 생성)는 base 를 과소평가하므로 이 로컬 결과를 공식 수치로 사용
