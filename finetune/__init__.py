"""
finetune package

Utilities for collecting training data, fine-tuning a local LLM with
Unsloth + LoRA, and running inference on the fine-tuned checkpoint.
"""

from finetune.data_collector import DataCollector
from finetune.inference import FinetunedModel

__all__ = ["DataCollector", "FinetunedModel"]
