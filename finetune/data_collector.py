"""
finetune/data_collector.py

공고 파싱 sLLM 학습 데이터 생성: 실제 채용공고 → gpt-4o 구조화 결과(teacher) 쌍.

학습 목표
---------
"채용공고 원문 → JobRequirements JSON" (agents/job_parser.py 의 로컬 LLM 작업).

예전에는 "공고 → 이력서" 합성 쌍을 만들었으나 폐기했습니다. 프로필 없이 이력서를
쓰도록 학습하면 경력을 지어내는 법을 배우게 되기 때문입니다 (PROJECT_DOCS #007 참고).

형식
----
JSONL, 한 줄에 하나:
    {"messages": [system, user, assistant], "meta": {"url", "query", "source"}}
system/user 는 agents.job_parser.build_local_messages() 로 만들어 실제 추론 입력과
글자 하나까지 같게 맞춥니다. assistant 는 gpt-4o 결과 JSON 문자열입니다.

실행
----
    python -m finetune.data_collector --target 50          # 파일럿
    python -m finetune.data_collector --target 600 --append  # 본 수집 (이어서)

비용: 공고 1건당 gpt-4o 1회 (입력 ~3k 토큰) + Tavily 검색. 50건 ≈ 수백 원.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "dataset"
ALL_PATH = DATASET_DIR / "parser_all.jsonl"
TRAIN_PATH = DATASET_DIR / "parser_train.jsonl"
EVAL_PATH = DATASET_DIR / "parser_eval.jsonl"

MIN_POSTING_CHARS = 300
EVAL_RATIO = 0.1
SPLIT_SEED = 42
# 공고가 아닌 페이지(음성 예시)는 전체의 이 비율까지만 유지
MAX_NEGATIVE_RATIO = 0.15

ROLES = [
    "백엔드 개발자", "프론트엔드 개발자", "풀스택 개발자", "데이터 엔지니어",
    "머신러닝 엔지니어", "AI 엔지니어", "데이터 분석가", "DevOps 엔지니어",
    "SRE", "안드로이드 개발자", "iOS 개발자", "QA 엔지니어", "보안 엔지니어",
    "게임 서버 개발자", "임베디드 소프트웨어", "클라우드 엔지니어", "DBA",
    "Flutter 개발자", "Node.js 개발자", "Java 백엔드", "Go 개발자", "기술 PM",
]
SITES = ["wanted.co.kr/wd", "saramin.co.kr", "jumpit.saramin.co.kr", "jobkorea.co.kr"]


# ------------------------------------------------------------------ #
# 수집                                                                 #
# ------------------------------------------------------------------ #

def _search_queries(rng: random.Random) -> Iterator[tuple[str, str]]:
    """(role, query) 를 다양하게 섞어 무한히 생성합니다."""
    pairs = [(r, s) for r in ROLES for s in SITES]
    while True:
        rng.shuffle(pairs)
        for role, site in pairs:
            yield role, f"{role} 채용 공고 site:{site}"


def collect_postings(target: int, seen_urls: set[str], rng: random.Random) -> Iterator[dict[str, Any]]:
    """Tavily 로 공고 원문(raw_content)을 수집해 하나씩 반환합니다."""
    from agents.job_recommender import _is_listing_url
    from search.company_searcher import CompanySearcher

    client = CompanySearcher()._get_client()
    yielded, searches, max_searches = 0, 0, max(10, target)
    for role, query in _search_queries(rng):
        if yielded >= target or searches >= max_searches:
            return
        searches += 1
        try:
            resp = client.search(query=query, max_results=8, include_raw_content=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[collect] 검색 실패 (%s): %s", query, exc)
            continue
        for item in resp.get("results", []):
            url = item.get("url", "")
            text = (item.get("raw_content") or item.get("content") or "").strip()
            if not url or url in seen_urls or _is_listing_url(url) or len(text) < MIN_POSTING_CHARS:
                continue
            seen_urls.add(url)
            yielded += 1
            yield {"url": url, "query": query, "role": role, "text": text}
            if yielded >= target:
                return


# ------------------------------------------------------------------ #
# 라벨링 + 레코드 생성                                                 #
# ------------------------------------------------------------------ #

def label_posting(text: str) -> dict[str, Any]:
    """gpt-4o Structured Output 으로 정답 라벨 생성 (실패 시 예외)."""
    from agents.job_parser import parse_posting_structured
    return parse_posting_structured(text, raise_on_error=True)


def build_record(text: str, label: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    from agents.job_parser import build_local_messages
    messages = build_local_messages(text)
    messages.append({"role": "assistant", "content": json.dumps(label, ensure_ascii=False)})
    return {"messages": messages, "meta": meta}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_dataset(rows: list[dict[str, Any]]) -> tuple[list, list]:
    """URL 기준 고정 시드 분할 (같은 공고가 train/eval 에 동시에 들어가지 않음)."""
    rows = sorted(rows, key=lambda r: r["meta"]["url"])
    random.Random(SPLIT_SEED).shuffle(rows)
    n_eval = max(1, int(len(rows) * EVAL_RATIO))
    return rows[n_eval:], rows[:n_eval]


BUNDLE_PATH = Path(__file__).parent / "goodjob_colab.zip"
BUNDLE_FILES = ["finetune/__init__.py", "finetune/train.py", "finetune/metrics.py",
                "finetune/dataset/parser_train.jsonl", "finetune/dataset/parser_eval.jsonl"]


def write_colab_bundle() -> Path:
    """Colab 에 올릴 zip (학습 코드 + 데이터셋) 을 만듭니다."""
    import zipfile
    root = Path(__file__).parent.parent
    with zipfile.ZipFile(BUNDLE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in BUNDLE_FILES:
            zf.write(root / rel, rel)
    return BUNDLE_PATH


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [json.loads(r["messages"][-1]["content"]) for r in rows]
    pos = [l for l in labels if l.get("is_job_posting")]
    return {
        "total": len(rows),
        "job_postings": len(pos),
        "non_postings": len(rows) - len(pos),
        "avg_required_skills": round(sum(len(l["required_skills"]) for l in pos) / max(1, len(pos)), 1),
        "with_company_name": sum(1 for l in pos if l.get("company_name")),
        "roles": len({r["meta"]["role"] for r in rows}),
    }


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="공고 파싱 sLLM 학습 데이터 생성")
    parser.add_argument("--target", type=int, default=50, help="새로 수집할 공고 수")
    parser.add_argument("--append", action="store_true", help="기존 parser_all.jsonl 에 이어서 수집")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)

    rows = _read_jsonl(ALL_PATH) if args.append else []
    seen = {r["meta"]["url"] for r in rows}
    negatives = sum(1 for r in rows if not json.loads(r["messages"][-1]["content"]).get("is_job_posting"))
    rng = random.Random(args.seed + len(rows))

    added = failed = skipped_neg = 0
    for i, post in enumerate(collect_postings(args.target, seen, rng), 1):
        try:
            label = label_posting(post["text"])
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning("[label] 실패 (%s): %s", post["url"], exc)
            continue
        if not label.get("is_job_posting"):
            if negatives >= MAX_NEGATIVE_RATIO * max(len(rows) + 1, args.target):
                skipped_neg += 1
                continue
            negatives += 1
        rows.append(build_record(post["text"], label,
                                 {"url": post["url"], "query": post["query"], "role": post["role"],
                                  "source": "tavily+gpt-4o"}))
        added += 1
        print(f"[{i}/{args.target}] {'공고' if label.get('is_job_posting') else '비공고'} "
              f"| {label.get('company_name') or '-'} | {label.get('job_title') or '-'}", flush=True)
        _write_jsonl(ALL_PATH, rows)  # 중간에 끊겨도 진행분 보존

    train, eval_ = split_dataset(rows)
    _write_jsonl(TRAIN_PATH, train)
    _write_jsonl(EVAL_PATH, eval_)
    print(f"\n추가 {added}건 / 라벨링 실패 {failed}건 / 비공고 초과로 제외 {skipped_neg}건")
    print("통계:", json.dumps(stats(rows), ensure_ascii=False))
    print(f"train {len(train)}건 → {TRAIN_PATH}\neval  {len(eval_)}건 → {EVAL_PATH}")
    print(f"Colab 업로드용 번들 → {write_colab_bundle()}")


if __name__ == "__main__":
    main()
