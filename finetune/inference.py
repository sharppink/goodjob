"""
finetune/inference.py

Loads a fine-tuned LoRA checkpoint (produced by ``finetune/train.py``) and
exposes a simple ``generate()`` interface compatible with the rest of the
GoodJob LLM abstraction layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7


class FinetunedModel:
    """
    Inference wrapper for a fine-tuned Unsloth / PEFT LoRA model.

    Usage
    -----
    >>> model = FinetunedModel()
    >>> model.load_model("finetune/outputs/goodjob-lora")
    >>> text = model.generate("Write a resume for a Python engineer applying to Kakao.")
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._model_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_model(self, model_path: str | Path) -> None:
        """
        Load the fine-tuned model from a saved LoRA checkpoint directory.

        Parameters
        ----------
        model_path : str | Path
            Directory containing the LoRA adapter weights saved by
            ``finetune/train.py``.
        """
        model_path = str(Path(model_path).resolve())
        logger.info("[FinetunedModel] Loading model from: %s", model_path)

        try:
            from unsloth import FastLanguageModel  # type: ignore
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
        except ImportError:
            # Fall back to vanilla PEFT / transformers if Unsloth is not installed
            logger.warning("[FinetunedModel] Unsloth not found – using plain transformers.")
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_4bit=True,
            )

        self._model = model
        self._tokenizer = tokenizer
        self._model_path = model_path
        logger.info("[FinetunedModel] Model ready.")

    def generate(
        self,
        prompt: str,
        instruction: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """
        Generate text from the fine-tuned model.

        Parameters
        ----------
        prompt : str
            Free-form prompt OR the "input" field of the Alpaca template.
            If ``instruction`` is ``None``, ``prompt`` is used directly.
        instruction : str, optional
            Alpaca "instruction" field.  When provided, ``prompt`` becomes
            the "input" field and the full Alpaca template is used.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        temperature : float
            Sampling temperature.

        Returns
        -------
        str
            Generated text (response portion only, after "### Response:").
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if instruction is not None:
            formatted = ALPACA_PROMPT.format(instruction=instruction, input=prompt)
        else:
            formatted = prompt

        import torch  # type: ignore
        tokenizer = self._tokenizer
        model = self._model

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)  # type: ignore
        with torch.no_grad():
            output_ids = model.generate(  # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return generated.strip()

    def is_loaded(self) -> bool:
        """Return ``True`` if a model has been successfully loaded."""
        return self._model is not None

    @property
    def model_path(self) -> Optional[str]:
        """Return the path from which the model was loaded."""
        return self._model_path
