"""
finetune/train.py

공고 파싱 sLLM 파인튜닝 — Qwen2.5-7B-Instruct + Unsloth QLoRA.
Google Colab(T4 16GB) 에서 colab_finetune.ipynb 가 이 모듈을 import 해서 사용합니다.

흐름
----
1. 베이스 모델로 eval 세트 평가 (비교 기준)
2. LoRA 학습 — assistant(JSON) 부분만 loss 계산 (train_on_responses_only)
3. 파인튜닝 모델로 같은 eval 세트 평가
4. GGUF(q4_k_m) 내보내기 → 로컬 Ollama 에 등록해서 사용

데이터 형식: finetune/data_collector.py 가 만든 {"messages": [system, user, assistant]} JSONL.
system/user 는 실제 추론 입력(agents.job_parser.build_local_messages)과 동일합니다.

CLI (Colab/Linux GPU):
    python -m finetune.train --train parser_train.jsonl --eval parser_eval.jsonl --epochs 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
CHAT_TEMPLATE = "qwen-2.5"
# 공고 4000자 ≈ 3천 토큰 + 지시문 + JSON 출력
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 1536  # 정답 JSON 최대 ~900 토큰 (파일럿 측정) + 여유

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
}

TRAINING_ARGS = {
    "per_device_train_batch_size": 1,   # 긴 시퀀스 → 배치 1 + 누적 8
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.05,
    "learning_rate": 2e-4,
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "report_to": "none",
}


# ------------------------------------------------------------------ #
# 모델 / 데이터                                                        #
# ------------------------------------------------------------------ #

def load_model(base_model: str = BASE_MODEL, max_seq_length: int = MAX_SEQ_LENGTH):
    from unsloth import FastLanguageModel  # type: ignore
    from unsloth.chat_templates import get_chat_template  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    return model, tokenizer


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ #
# 평가                                                                 #
# ------------------------------------------------------------------ #

def generate_predictions(model, tokenizer, records: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
    """각 레코드의 system+user 로 생성 (greedy). 정답 assistant 는 사용하지 않음."""
    import torch  # type: ignore
    from unsloth import FastLanguageModel  # type: ignore

    FastLanguageModel.for_inference(model)
    preds: list[str] = []
    for i, rec in enumerate(records, 1):
        prompt_ids = tokenizer.apply_chat_template(
            rec["messages"][:-1], add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        preds.append(tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip())
        if i % 5 == 0 or i == len(records):
            print(f"  생성 {i}/{len(records)}", flush=True)
    return preds


def evaluate(model, tokenizer, eval_records: list[dict]) -> tuple[dict[str, float], list[str]]:
    from finetune.metrics import score_all

    golds = [json.loads(r["messages"][-1]["content"]) for r in eval_records]
    t0 = time.time()
    preds = generate_predictions(model, tokenizer, eval_records)
    scores = score_all(preds, golds)
    scores["sec_per_sample"] = round((time.time() - t0) / max(1, len(eval_records)), 1)
    return scores, preds


# ------------------------------------------------------------------ #
# 학습                                                                 #
# ------------------------------------------------------------------ #

def train(model, tokenizer, train_records: list[dict], output_dir: str = "outputs",
          num_epochs: int = 2, max_seq_length: int = MAX_SEQ_LENGTH):
    """LoRA 어댑터를 붙여 학습하고 (model, trainer_stats) 를 반환합니다."""
    from datasets import Dataset  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore
    from unsloth import FastLanguageModel, is_bfloat16_supported  # type: ignore
    from unsloth.chat_templates import train_on_responses_only  # type: ignore

    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    texts = [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in train_records
    ]
    dataset = Dataset.from_dict({"text": texts})

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        fp16=not is_bfloat16_supported(),   # T4 / RTX 20xx 는 bf16 미지원
        bf16=is_bfloat16_supported(),
        **TRAINING_ARGS,
    )
    try:
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=config)
    except TypeError:
        # 최신 trl 은 tokenizer 대신 processing_class 인자를 사용
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=config)

    # 공고 원문(user)까지 학습하면 "공고를 베끼는" 쪽으로 학습됨 → JSON 답변에만 loss
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    stats = trainer.train()
    return model, stats


def export_gguf(model, tokenizer, out_dir: str = "goodjob-parser", quantization: str = "q4_k_m") -> list[str]:
    """GGUF 로 저장하고 생성된 .gguf 파일 경로 목록을 반환합니다."""
    model.save_pretrained_gguf(out_dir, tokenizer, quantization_method=quantization)
    # unsloth 버전에 따라 저장 위치/파일명이 달라서 검색으로 찾음
    found = sorted({str(p) for p in Path(".").glob("**/*.gguf") if out_dir in str(p) or p.parent.name == out_dir})
    return found or sorted(str(p) for p in Path(".").glob("**/*.gguf"))


# ------------------------------------------------------------------ #
# 전체 실행                                                            #
# ------------------------------------------------------------------ #

def run_all(train_path: str, eval_path: str, epochs: int = 2, eval_limit: int | None = None,
            skip_base_eval: bool = False, export: bool = True) -> dict[str, Any]:
    from finetune.metrics import format_table

    train_records = read_jsonl(train_path)
    eval_records = read_jsonl(eval_path)[:eval_limit] if eval_limit else read_jsonl(eval_path)
    print(f"train {len(train_records)}건 / eval {len(eval_records)}건")

    model, tokenizer = load_model()
    results: dict[str, dict] = {}

    if not skip_base_eval:
        print("\n[1/4] 베이스 모델 평가")
        results["base"], _ = evaluate(model, tokenizer, eval_records)

    print("\n[2/4] LoRA 학습")
    from unsloth import FastLanguageModel  # type: ignore
    FastLanguageModel.for_training(model)
    model, stats = train(model, tokenizer, train_records, num_epochs=epochs)
    print(stats)

    print("\n[3/4] 파인튜닝 모델 평가")
    results["finetuned"], preds = evaluate(model, tokenizer, eval_records)
    print("\n" + format_table(results))

    report = {"results": results, "train_size": len(train_records), "eval_size": len(eval_records),
              "epochs": epochs, "base_model": BASE_MODEL, "lora": LORA_CONFIG,
              "training_args": TRAINING_ARGS, "train_loss": getattr(stats, "training_loss", None),
              "sample_predictions": preds[:3]}
    Path("finetune_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    model.save_pretrained("goodjob-parser-lora")
    tokenizer.save_pretrained("goodjob-parser-lora")

    report["gguf_files"] = []
    if export:
        print("\n[4/4] GGUF 내보내기 (q4_k_m, 10분 내외)")
        try:
            report["gguf_files"] = export_gguf(model, tokenizer)
            print("GGUF:", report["gguf_files"])
        except Exception as exc:  # noqa: BLE001
            # 내보내기가 실패해도 LoRA·리포트는 이미 저장됨 → 다음 셀에서 Drive 로 복사 가능
            report["gguf_error"] = repr(exc)
            print("⚠️ GGUF 내보내기 실패:", exc)
    else:
        print("\n[4/4] EXPORT=False — GGUF 를 만들지 않았습니다 (로컬 Ollama 에서 쓰려면 True 필요)")
    Path("finetune_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GoodJob 공고 파싱 sLLM 파인튜닝 (Unsloth)")
    p.add_argument("--train", default="finetune/dataset/parser_train.jsonl")
    p.add_argument("--eval", default="finetune/dataset/parser_eval.jsonl")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--eval-limit", type=int, default=None, help="평가 건수 제한 (기본: eval 전체)")
    p.add_argument("--skip-base-eval", action="store_true")
    p.add_argument("--no-export", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    a = _parse_args()
    run_all(a.train, a.eval, epochs=a.epochs, eval_limit=a.eval_limit,
            skip_base_eval=a.skip_base_eval, export=not a.no_export)
