"""
agents/fit_analyzer.py

LangGraph node: **fit_analyzer**

Responsibility
--------------
Analyse how well the user's retrieved experiences match the job requirements.
Produces:

- ``fit_score``    : float in [0.0, 1.0]
- ``fit_feedback`` : detailed gap-analysis paragraph

Always uses OpenAI for comprehensive analysis (high-complexity task).
If ``fit_score < 0.3``, the graph router will short-circuit to END.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from agents.state import GoodJobState

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt templates                                                     #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are a senior technical recruiter and career coach.
Your task is to objectively assess how well a candidate's experience matches
a job's requirements and produce an honest, actionable gap analysis.
Always respond with valid JSON only.
"""

USER_PROMPT_TEMPLATE = """\
### Job Requirements
{job_requirements_text}

### Candidate's Relevant Experiences
{experiences_text}

### Task
1. Score the candidate's fit on a scale from 0.0 (no match) to 1.0 (perfect match).
2. Write a detailed gap-analysis (2–4 paragraphs) covering:
   - Strengths that align with the role
   - Skill gaps or missing experience
   - Specific advice to address the gaps
3. Return a JSON object with exactly these two keys:
   - "fit_score"    : number (0.0 – 1.0)
   - "fit_feedback" : string (the gap-analysis text)
"""


# ------------------------------------------------------------------ #
# Node function                                                        #
# ------------------------------------------------------------------ #

def fit_analyzer_node(state: GoodJobState) -> GoodJobState:
    """
    LangGraph node that computes a fit score and gap analysis.

    Parameters
    ----------
    state : GoodJobState
        Must contain ``job_requirements`` and ``retrieved_experiences``.

    Returns
    -------
    GoodJobState
        Updated state with ``fit_score``, ``fit_feedback``, and
        ``current_step`` set to ``"fit_analyzer"``.
    """
    logger.info("[fit_analyzer] Starting fit analysis.")
    state["current_step"] = "fit_analyzer"
    errors: list[str] = state.get("errors") or []

    job_requirements: Optional[dict] = state.get("job_requirements")
    retrieved_experiences: list[str] = state.get("retrieved_experiences") or []

    if not job_requirements:
        logger.warning("[fit_analyzer] No job_requirements – assigning default low score.")
        errors.append("fit_analyzer: job_requirements missing.")
        state.update({"fit_score": 0.0, "fit_feedback": "No job requirements available for analysis.", "errors": errors})
        return state

    job_req_text = _format_requirements(job_requirements)
    experiences_text = _format_experiences(retrieved_experiences)
    prompt = USER_PROMPT_TEMPLATE.format(
        job_requirements_text=job_req_text,
        experiences_text=experiences_text,
    )

    try:
        from llm.openai_client import OpenAIClient
        client = OpenAIClient()
        raw_response = client.generate(prompt=prompt, system=SYSTEM_PROMPT)
        result = _parse_response(raw_response)
    except Exception as exc:  # noqa: BLE001
        logger.error("[fit_analyzer] Claude API call failed: %s", exc)
        errors.append(f"fit_analyzer: Claude error – {exc}")
        result = {"fit_score": 0.0, "fit_feedback": f"Analysis failed: {exc}"}

    fit_score = float(max(0.0, min(1.0, result.get("fit_score", 0.0))))
    fit_feedback = str(result.get("fit_feedback", ""))

    logger.info("[fit_analyzer] fit_score=%.2f", fit_score)
    state["fit_score"] = fit_score
    state["fit_feedback"] = fit_feedback
    state["errors"] = errors
    return state


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _format_requirements(req: dict) -> str:
    """Convert requirements dict to a readable text block."""
    lines = []
    if req.get("required_skills"):
        lines.append("Required skills: " + ", ".join(req["required_skills"]))
    if req.get("preferred_skills"):
        lines.append("Preferred skills: " + ", ".join(req["preferred_skills"]))
    if req.get("experience_years") is not None:
        lines.append(f"Minimum experience: {req['experience_years']} years")
    if req.get("responsibilities"):
        lines.append("Responsibilities:")
        for r in req["responsibilities"]:
            lines.append(f"  - {r}")
    if req.get("company_culture"):
        lines.append(f"Company culture: {req['company_culture']}")
    return "\n".join(lines)


def _format_experiences(experiences: list[str]) -> str:
    """Join retrieved experience chunks into a numbered list."""
    if not experiences:
        return "(No experiences retrieved)"
    return "\n\n".join(f"[{i+1}] {exp}" for i, exp in enumerate(experiences))


def _parse_response(raw: str) -> dict:
    """Parse JSON from LLM response, with fallback regex extraction."""
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt to extract score with regex if full JSON failed
    score_match = re.search(r'"fit_score"\s*:\s*([\d.]+)', raw)
    feedback_match = re.search(r'"fit_feedback"\s*:\s*"(.*?)"', raw, re.DOTALL)
    return {
        "fit_score": float(score_match.group(1)) if score_match else 0.0,
        "fit_feedback": feedback_match.group(1).replace("\\n", "\n") if feedback_match else raw,
    }
