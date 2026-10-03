"""
agents/fit_analyzer.py

LangGraph node: **fit_analyzer**

Responsibility
--------------
Analyse how well the user's retrieved experiences match the job requirements.
Produces:

- ``fit_score``      : float in [0.0, 1.0]
- ``fit_feedback``   : detailed gap-analysis paragraph
- ``matched_skills`` : required/preferred skills evidenced in the experiences
- ``missing_skills`` : required/preferred skills with no evidence

Uses OpenAI Structured Output (``FitAnalysis`` schema) so the response is
always valid.  The schema puts ``reasoning`` first so the model works through
the rubric step by step before committing to a score (chain-of-thought).
If ``fit_score < LOW_FIT_THRESHOLD``, the graph router will short-circuit to END.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from agents.state import GoodJobState

logger = logging.getLogger(__name__)

# 이 점수 미만이면 이력서 생성을 건너뜀 (graph 라우팅, Streamlit, 공고 추천에서 공통 사용)
LOW_FIT_THRESHOLD = 0.3

# ------------------------------------------------------------------ #
# Prompt templates                                                     #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are a senior technical recruiter and career coach.
Your task is to objectively assess how well a candidate's experience matches
a job's requirements and produce an honest, actionable gap analysis.
Only count a skill as matched when the candidate's experiences give concrete
evidence of it. Never assume skills that are not mentioned.
Write `reasoning` and `fit_feedback` in Korean, unless the job requirements are entirely in English.
"""

USER_PROMPT_TEMPLATE = """\
### Job Requirements
{job_requirements_text}

### Candidate's Relevant Experiences
{experiences_text}

### Task — work through these steps in `reasoning` before scoring
1. For each required skill, decide matched / missing based on evidence.
2. Do the same for preferred skills.
3. Compare the candidate's years of experience with the minimum required.
4. Check whether past responsibilities resemble the role's responsibilities.

### Scoring rubric for `fit_score`
- 0.85–1.00 : all required skills evidenced, experience meets the minimum, most preferred skills present
- 0.65–0.84 : nearly all required skills evidenced, minor gaps
- 0.45–0.64 : about half of required skills evidenced, or clearly short on experience
- 0.25–0.44 : few required skills evidenced, mostly transferable experience
- 0.00–0.24 : different field or no relevant experience

### `fit_feedback` (2–4 paragraphs)
- Strengths that align with the role
- Skill gaps or missing experience
- Specific advice to address the gaps

Use the exact skill names from the job requirements in `matched_skills` / `missing_skills`.
"""


# ------------------------------------------------------------------ #
# Structured Output schema                                             #
# ------------------------------------------------------------------ #

class FitAnalysis(BaseModel):
    """적합도 분석 결과 — OpenAI Structured Output 전용."""
    reasoning: str = Field(description="Step-by-step evaluation following steps 1–4 (not shown to the user)")
    matched_skills: list[str] = Field(description="Required/preferred skills evidenced in the experiences")
    missing_skills: list[str] = Field(description="Required/preferred skills with no evidence")
    fit_score: float = Field(description="Overall fit score from 0.0 to 1.0 following the rubric")
    fit_feedback: str = Field(description="Gap analysis text shown to the user (2–4 paragraphs)")


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
        Updated state with ``fit_score``, ``fit_feedback``, ``matched_skills``,
        ``missing_skills`` and ``current_step`` set to ``"fit_analyzer"``.
    """
    logger.info("[fit_analyzer] Starting fit analysis.")
    state["current_step"] = "fit_analyzer"
    errors: list[str] = state.get("errors") or []

    job_requirements: Optional[dict] = state.get("job_requirements")
    retrieved_experiences: list[str] = state.get("retrieved_experiences") or []

    if not job_requirements:
        logger.warning("[fit_analyzer] No job_requirements – assigning default low score.")
        errors.append("fit_analyzer: job_requirements missing.")
        state.update({
            "fit_score": 0.0,
            "fit_feedback": "No job requirements available for analysis.",
            "matched_skills": [],
            "missing_skills": [],
            "errors": errors,
        })
        return state

    prompt = USER_PROMPT_TEMPLATE.format(
        job_requirements_text=_format_requirements(job_requirements),
        experiences_text=_format_experiences(retrieved_experiences),
    )

    try:
        from llm.factory import get_chat_client
        client = get_chat_client()
        result: FitAnalysis = client.generate_structured(
            prompt=prompt,
            schema=FitAnalysis,
            system=SYSTEM_PROMPT,
        )
        fit_score = float(max(0.0, min(1.0, result.fit_score)))
        fit_feedback = result.fit_feedback
        matched, missing = result.matched_skills, result.missing_skills
        logger.debug("[fit_analyzer] reasoning: %s", result.reasoning)
    except Exception as exc:  # noqa: BLE001
        logger.error("[fit_analyzer] LLM call failed: %s", exc)
        errors.append(f"fit_analyzer: LLM 오류 – {exc}")
        fit_score, fit_feedback = 0.0, f"Analysis failed: {exc}"
        matched, missing = [], []

    logger.info("[fit_analyzer] fit_score=%.2f (matched %d / missing %d)",
                fit_score, len(matched), len(missing))
    state["fit_score"] = fit_score
    state["fit_feedback"] = fit_feedback
    state["matched_skills"] = matched
    state["missing_skills"] = missing
    state["errors"] = errors
    return state


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _format_requirements(req: dict) -> str:
    """Convert requirements dict to a readable text block."""
    lines = []
    if req.get("job_title"):
        lines.append(f"Job title: {req['job_title']}")
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
