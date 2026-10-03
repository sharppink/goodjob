"""
agents/graph.py

Assembles the GoodJob LangGraph StateGraph from the six agent nodes and
compiles it into an executable ``app`` object.

Graph topology
--------------

    [START]
       |
       v
  job_parser           -- parses raw posting → job_requirements
       |
       v
  rag_retriever        -- fetches relevant user experiences
       |
       v
  fit_analyzer         -- scores fit [0, 1] and writes gap analysis
       |
       +--(fit_score < 0.3)--> [END]  (low fit – skip resume generation)
       |
       +--(fit_score >= 0.3)-->
       v
  resume_writer        -- generates resume_draft
       |
       v
  reviewer             -- self-critiques and produces resume_final
       |
       v
  interview_coach      -- interview questions + answer tips
                          (skipped when generate_interview is False)
       |
       v
    [END]
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from agents.state import GoodJobState
from agents.job_parser import job_parser_node
from agents.rag_retriever import rag_retriever_node
from agents.fit_analyzer import LOW_FIT_THRESHOLD, fit_analyzer_node
from agents.resume_writer import resume_writer_node
from agents.reviewer import reviewer_node
from agents.interview_coach import interview_coach_node

# ------------------------------------------------------------------ #
# Conditional edge logic                                               #
# ------------------------------------------------------------------ #

def _route_after_fit_analysis(
    state: GoodJobState,
) -> Literal["resume_writer", "__end__"]:
    """
    Routing function used as a conditional edge after ``fit_analyzer``.

    Returns ``"resume_writer"`` when the fit score is high enough to
    proceed with resume generation, or ``"__end__"`` when the user's
    profile is too far from the job requirements.
    """
    fit_score: float = state.get("fit_score") or 0.0
    if fit_score < LOW_FIT_THRESHOLD:
        return "__end__"
    return "resume_writer"


# ------------------------------------------------------------------ #
# Build the graph                                                      #
# ------------------------------------------------------------------ #

def build_graph() -> StateGraph:
    """
    Construct and return the compiled GoodJob StateGraph.

    The returned object is already compiled (``graph.compile()``) so it
    can be invoked directly with ``app.invoke(state)`` or streamed with
    ``app.stream(state)``.
    """
    graph = StateGraph(GoodJobState)

    # Register nodes
    graph.add_node("job_parser", job_parser_node)
    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("fit_analyzer", fit_analyzer_node)
    graph.add_node("resume_writer", resume_writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("interview_coach", interview_coach_node)

    # Linear edges
    graph.set_entry_point("job_parser")
    graph.add_edge("job_parser", "rag_retriever")
    graph.add_edge("rag_retriever", "fit_analyzer")

    # Conditional edge: low fit → END, otherwise continue to resume generation
    graph.add_conditional_edges(
        "fit_analyzer",
        _route_after_fit_analysis,
        {
            "resume_writer": "resume_writer",
            "__end__": END,
        },
    )

    graph.add_edge("resume_writer", "reviewer")
    graph.add_edge("reviewer", "interview_coach")
    graph.add_edge("interview_coach", END)

    return graph


# Module-level compiled app – import and call directly.
app = build_graph().compile()
