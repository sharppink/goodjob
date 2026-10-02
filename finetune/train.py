"""
finetune/train.py

Fine-tunes a quantised LLM using Unsloth + LoRA + HuggingFace TRL's SFTTrainer.

Base model  : unsloth/Qwen2.5-7B-Instruct-bnb-4bit
LoRA config : r=16, alpha=16, standard target modules
Dataset     : Alpaca-format JSONL produced by DataCollector

Requirements
------------
- GPU with ≥ 16 GB VRAM (A100-40G or better recommended)
- ``pip install unsloth`` (or the unsloth_zoo variant)
- ``pip install trl peft transformers datasets``

Usage
-----
    python -m finetune.train \\
        --dataset finetune/dataset/dataset.jsonl \\
        --output  finetune/outputs/qwen2.5-goodjob \\
        --epochs  3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_OUTPUT_DIR = "finetune/outputs/goodjob-lora"
DEFAULT_DATASET_PATH = "finetune/dataset/dataset.jsonl"

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_dropout": 0.0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
}

TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 60,          # Increase for full training runs
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 42,
}

# Alpaca prompt template
ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
EOS_TOKEN_PLACEHOLDER = "<|endoftext|>"


# ------------------------------------------------------------------ #
# Main training function                                               #
# ------------------------------------------------------------------ #

def train(
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_epochs: int = 1,
    max_seq_length: int = 2048,
) -> None:
    """
    Fine-tune the base model on the GoodJob dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the JSONL training dataset.
    output_dir : str
        Directory to save LoRA adapter weights.
    num_epochs : int
        Number of full passes over the dataset.
    max_seq_length : int
        Maximum token sequence length.
    """
    logger.info("[train] Loading Unsloth model: %s", BASE_MODEL)

    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "unsloth is required for fine-tuning. "
            "Install with: pip install unsloth"
        ) from exc

    # ---- Load base model + tokenizer ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=None,          # auto-detect float16 / bfloat16
        load_in_4bit=True,
    )

    # ---- Attach LoRA adapters ----
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    # ---- Load dataset ----
    from datasets import load_dataset  # type: ignore
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    logger.info("[train] Loaded %d training examples.", len(dataset))

    eos_token = tokenizer.eos_token or EOS_TOKEN_PLACEHOLDER

    def _format_example(examples: dict) -> dict:
        texts = [
            ALPACA_PROMPT.format(
                instruction=instr,
                input=inp,
                output=out,
            ) + eos_token
            for instr, inp, out in zip(
                examples["instruction"],
                examples["input"],
                examples["output"],
            )
        ]
        return {"text": texts}

    dataset = dataset.map(_format_example, batched=True)

    # ---- Training arguments ----
    from transformers import TrainingArguments  # type: ignore
    from trl import SFTTrainer  # type: ignore

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        **TRAINING_ARGS,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=training_args,
    )

    # ---- Run training ----
    logger.info("[train] Starting training …")
    trainer_stats = trainer.train()
    logger.info("[train] Training complete. Stats: %s", trainer_stats)

    # ---- Save LoRA adapter ----
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("[train] Saved adapter to '%s'.", output_dir)


# ------------------------------------------------------------------ #
# CLI entry point                                                      #
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GoodJob LLM with Unsloth.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="JSONL dataset path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for LoRA weights")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_len,
    )
