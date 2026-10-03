"""
finetune/eval_local.py

Ollama 에 등록한 모델(파인튜닝 전/후)을 eval 세트로 평가해 비교합니다.
앱이 실제로 쓰는 경로(agents.job_parser.parse_with_local_llm 와 같은 입력)를 그대로 사용합니다.

실행 (Ollama 실행 중이어야 함):
    python -m finetune.eval_local --models qwen2.5:7b goodjob-parser
    python -m finetune.eval_local --models goodjob-parser --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from finetune.metrics import format_table, score_all


def evaluate_model(model_name: str, records: list[dict]) -> dict[str, float]:
    from llm.local_llm import LocalLLM

    llm = LocalLLM(model=model_name)
    if not llm.is_available():
        raise RuntimeError(f"Ollama 에서 '{model_name}' 모델을 찾을 수 없습니다 (ollama list 확인).")

    preds, t0 = [], time.time()
    for i, rec in enumerate(records, 1):
        system, user = rec["messages"][0]["content"], rec["messages"][1]["content"]
        preds.append(llm.generate(prompt=user, system=system, format="json", max_tokens=1536))
        print(f"  [{model_name}] {i}/{len(records)}", flush=True)
    golds = [json.loads(r["messages"][-1]["content"]) for r in records]
    scores = score_all(preds, golds)
    scores["sec_per_sample"] = round((time.time() - t0) / max(1, len(records)), 1)
    return scores


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Ollama 로컬 모델 공고 파싱 평가")
    p.add_argument("--models", nargs="+", default=["qwen2.5:7b", "goodjob-parser"])
    p.add_argument("--eval", default="finetune/dataset/parser_eval.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = records[: args.limit] if args.limit else records

    results = {m: evaluate_model(m, records) for m in args.models}
    print("\n" + format_table(results))


if __name__ == "__main__":
    main()
