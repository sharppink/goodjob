"""
finetune/inference.py

Colab 에서 학습 직후 LoRA 체크포인트(goodjob-parser-lora/)로 공고 하나를 바로 파싱해 보는 용도.
앱에서는 GGUF → Ollama 경로(llm/local_llm.py)를 사용하므로 이 모듈을 쓰지 않습니다.

    from finetune.inference import FinetunedParser
    parser = FinetunedParser("goodjob-parser-lora")
    print(parser.parse("[카카오] 백엔드 개발자 채용 ..."))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class FinetunedParser:
    def __init__(self, model_path: str | Path, max_seq_length: int = 4096) -> None:
        from unsloth import FastLanguageModel  # type: ignore
        from unsloth.chat_templates import get_chat_template  # type: ignore

        self._model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path), max_seq_length=max_seq_length, load_in_4bit=True,
        )
        self._tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
        FastLanguageModel.for_inference(self._model)

    def generate_raw(self, posting_text: str, max_new_tokens: int = 1024) -> str:
        import torch  # type: ignore
        from agents.job_parser import build_local_messages

        ids = self._tokenizer.apply_chat_template(
            build_local_messages(posting_text), add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=self._tokenizer.eos_token_id)
        return self._tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def parse(self, posting_text: str) -> dict[str, Any]:
        """앱과 같은 후처리(parse_local_response)로 requirements dict 반환."""
        from agents.job_parser import parse_local_response
        return parse_local_response(self.generate_raw(posting_text))
