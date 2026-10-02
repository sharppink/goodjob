"""
finetune/data_collector.py

Collects and prepares (job posting → resume) training pairs for fine-tuning.

Two strategies are supported:
1. **Real collection** – query the Tavily API or Korean job sites for actual
   job postings and pair them with Claude-generated ideal resumes.
2. **Synthetic generation** – use Claude to create diverse job posting / resume
   pairs from a seed job posting.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_DATASET_DIR = Path(__file__).parent / "dataset"

# Alpaca-style instruction format
INSTRUCTION_TEMPLATE = (
    "Below is a job posting. Write a tailored, professional resume for this position.\n\n"
    "### Job Posting:\n{job_posting}\n\n### Resume:"
)


class DataCollector:
    """
    Collects and generates (instruction, output) pairs for fine-tuning.

    Usage
    -----
    >>> collector = DataCollector()
    >>> pairs = collector.generate_synthetic_pairs(sample_job_posting, count=10)
    >>> collector.save_dataset(pairs, "dataset/synthetic_v1.jsonl")
    """

    def __init__(self) -> None:
        from llm.claude_client import ClaudeClient
        self._claude = ClaudeClient()

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    def collect_job_postings(
        self,
        keywords: list[str],
        count: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Collect real job postings from the web using Tavily.

        Parameters
        ----------
        keywords : list[str]
            Search terms (e.g. ``["Python backend", "React frontend"]``).
        count : int
            Total number of postings to target.

        Returns
        -------
        list[dict]
            Each dict has keys: ``job_posting``, ``title``, ``company``, ``url``.
        """
        from search.company_searcher import CompanySearcher
        searcher = CompanySearcher()
        postings: list[dict[str, Any]] = []

        per_keyword = max(1, count // len(keywords))
        for kw in keywords:
            logger.info("[DataCollector] Searching postings for: %s", kw)
            result = searcher.search_job_postings(company_name=kw, role="")
            for item in result.get("results", [])[:per_keyword]:
                postings.append({
                    "job_posting": item.get("content", ""),
                    "title": item.get("title", ""),
                    "company": kw,
                    "url": item.get("url", ""),
                })

        logger.info("[DataCollector] Collected %d real postings.", len(postings))
        return postings

    def generate_synthetic_pairs(
        self,
        job_posting: str,
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Use Claude to generate diverse (job_posting → resume) training pairs
        derived from a seed job posting.

        Parameters
        ----------
        job_posting : str
            Seed job posting text.
        count : int
            Number of synthetic pairs to generate.

        Returns
        -------
        list[dict]
            Each dict: ``{"instruction": str, "input": str, "output": str}``.
        """
        pairs: list[dict[str, Any]] = []
        personas = self._sample_personas(count)

        for persona in personas:
            prompt = self._build_synthetic_prompt(job_posting, persona)
            try:
                resume = self._claude.generate(
                    prompt=prompt,
                    system="You are an expert resume writer. Write a realistic, detailed resume.",
                    max_tokens=3000,
                )
                pairs.append({
                    "instruction": INSTRUCTION_TEMPLATE.format(job_posting=job_posting),
                    "input": "",
                    "output": resume,
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("[DataCollector] Failed to generate pair: %s", exc)

        logger.info("[DataCollector] Generated %d synthetic pairs.", len(pairs))
        return pairs

    def save_dataset(
        self,
        data: list[dict[str, Any]],
        path: str | Path = DEFAULT_DATASET_DIR / "dataset.jsonl",
    ) -> Path:
        """
        Save a list of training records to a JSONL file.

        Parameters
        ----------
        data : list[dict]
            Training records.
        path : str | Path
            Output file path.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("[DataCollector] Saved %d records to %s.", len(data), output_path)
        return output_path

    def load_dataset(self, path: str | Path) -> list[dict[str, Any]]:
        """Load a JSONL dataset file into a list of dicts."""
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sample_personas(n: int) -> list[dict[str, Any]]:
        """Generate diverse candidate personas to vary the synthetic resumes."""
        experience_levels = ["junior (1-2 years)", "mid-level (3-5 years)", "senior (7+ years)"]
        backgrounds = ["bootcamp graduate", "CS degree", "self-taught", "career changer"]
        personas = []
        for _ in range(n):
            personas.append({
                "level": random.choice(experience_levels),
                "background": random.choice(backgrounds),
            })
        return personas

    @staticmethod
    def _build_synthetic_prompt(job_posting: str, persona: dict) -> str:
        return (
            f"Write a realistic resume for a {persona['level']} {persona['background']} "
            f"applying to the following position.\n\n"
            f"Job Posting:\n{job_posting}\n\n"
            "Requirements:\n"
            "- Use Markdown formatting\n"
            "- Include quantified achievements\n"
            "- Make it realistic and detailed\n"
            "- Tailor it specifically to the job requirements"
        )
