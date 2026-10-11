"""Milestone 5: Skeptic Agent for critical evidence evaluation and gap identification."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agents.evidence_agent import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    get_ollama_client,
    parse_model_json,
)
from app.schemas import EvidenceAssessment, SkepticAnalysis


def format_assessments_for_critique(assessments: List[EvidenceAssessment]) -> str:
    """Format structured evidence assessments into clean text for skeptic review."""
    if not assessments:
        return "No preliminary evidence assessments available (0 sources analyzed)."

    formatted_lines: List[str] = []
    for a in assessments:
        quote_text = f'"{a.quote}"' if a.quote else "None"
        limitations = a.limitations or "None reported"
        formatted_lines.append(
            f"- [{a.evidence_id}] {a.title} ({a.domain})\n"
            f"  Stance: {a.stance.upper()} (Confidence: {a.confidence})\n"
            f"  Explanation: {a.explanation}\n"
            f"  Quote: {quote_text}\n"
            f"  Limitations: {limitations}"
        )
    return "\n\n".join(formatted_lines)


def build_skeptic_prompt(claim: str, assessments: List[EvidenceAssessment]) -> str:
    """Construct an adversarial evaluation prompt challenging the preliminary findings."""
    evidence_summary = format_assessments_for_critique(assessments)

    prompt = f"""You are ClaimCheck's Skeptic Agent.
Your role is to critically evaluate preliminary evidence assessments, detect methodological weaknesses, identify bias, unstated assumptions, and discover missing perspectives.

CRITICAL EVALUATION RULES:
1. Do not invent evidence or claim that you have browsed the web.
2. A balanced count of supporting and contradicting passages does NOT automatically mean the evidence is credible or sufficient.
3. If the evidence has important gaps, unverified assumptions, or missing counter-evidence, set "is_sufficient" to false and formulate UP TO 2 focused follow-up search queries.
4. If the evidence is comprehensive, independent, and sufficient for a cautious conclusion, set "is_sufficient" to true and "follow_up_queries" to an empty list.
5. Provide a specific rationale for each follow-up query proposed.

CLAIM TO EVALUATE:
{claim}

PRELIMINARY EVIDENCE ASSESSMENTS:
{evidence_summary}

Respond ONLY with a valid JSON object matching this schema:
{{
  "is_sufficient": true | false,
  "strongest_weakness": "The primary methodological, evidentiary, or logical weakness in the preliminary findings",
  "missing_context": "Crucial context, nuances, or counter-hypotheses omitted by the evidence",
  "source_concerns": "Potential bias, conflict of interest, or lack of independent corroboration",
  "unanswered_questions": ["Key unanswered empirical or causal question 1", "Question 2"],
  "follow_up_queries": ["query 1", "query 2"],
  "query_reasons": {{
    "query 1": "Clear reason for query 1",
    "query 2": "Clear reason for query 2"
  }}
}}
"""
    return prompt


def evaluate_evidence_critique(
    claim: str,
    assessments: List[EvidenceAssessment],
    client: Optional[Any] = None,
    model: str = DEFAULT_OLLAMA_MODEL,
) -> Tuple[SkepticAnalysis, Optional[str]]:
    """Evaluate evidence gaps and determine whether targeted follow-up research is necessary."""
    # Handle empty preliminary assessments immediately
    if not assessments:
        analysis = SkepticAnalysis(
            is_sufficient=False,
            strongest_weakness="No evidence was retrieved or analyzed in the initial round.",
            missing_context="Complete absence of empirical data for the claim.",
            source_concerns="No sources available to evaluate.",
            unanswered_questions=["What empirical research exists regarding this claim?"],
            follow_up_queries=[f"{claim} empirical evidence study"[:100]],
            query_reasons={
                f"{claim} empirical evidence study"[:100]: "Initial retrieval returned zero evidence."
            },
        )
        return analysis, "Initial evidence set was empty."

    if client is None:
        client = get_ollama_client()

    if client is None:
        warning = "Ollama client unavailable for Skeptic Agent review."
        fallback = SkepticAnalysis(
            is_sufficient=False,
            strongest_weakness="Skeptic review unavailable (Ollama service offline).",
            missing_context="Automated gap analysis could not run.",
            source_concerns="Source independence could not be verified.",
            unanswered_questions=["Are there independent peer-reviewed sources verifying this claim?"],
            follow_up_queries=[],
            query_reasons={},
        )
        return fallback, warning

    prompt = build_skeptic_prompt(claim, assessments)

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.0},
        )
        raw_output = response.get("response", "") if isinstance(response, dict) else getattr(response, "response", "")
        data = parse_model_json(raw_output)

        is_sufficient = bool(data.get("is_sufficient", False))
        strongest_weakness = str(data.get("strongest_weakness", "")).strip() or "Unstated assumptions in source methodology."
        missing_context = str(data.get("missing_context", "")).strip() or None
        source_concerns = str(data.get("source_concerns", "")).strip() or None

        raw_questions = data.get("unanswered_questions", [])
        unanswered_questions = [str(q).strip() for q in raw_questions if str(q).strip()]

        raw_queries = data.get("follow_up_queries", [])
        follow_up_queries = [str(q).strip() for q in raw_queries if str(q).strip()][:2]

        raw_reasons = data.get("query_reasons", {})
        query_reasons: Dict[str, str] = {}
        if isinstance(raw_reasons, dict):
            for q in follow_up_queries:
                query_reasons[q] = str(raw_reasons.get(q, f"Targeted follow-up for {q}")).strip()

        # If model flagged sufficient, ensure queries list is empty
        if is_sufficient:
            follow_up_queries = []
            query_reasons = {}

        analysis = SkepticAnalysis(
            is_sufficient=is_sufficient,
            strongest_weakness=strongest_weakness,
            missing_context=missing_context,
            source_concerns=source_concerns,
            unanswered_questions=unanswered_questions,
            follow_up_queries=follow_up_queries,
            query_reasons=query_reasons,
        )
        return analysis, None

    except Exception as exc:
        warning = f"Skeptic Agent error: {str(exc)}"
        fallback = SkepticAnalysis(
            is_sufficient=False,
            strongest_weakness=f"Analysis failed due to model error or timeout: {str(exc)}",
            missing_context="Critical review could not be completed.",
            source_concerns="Unable to assess source independence.",
            unanswered_questions=["Are there independent peer-reviewed sources verifying this claim?"],
            follow_up_queries=[],
            query_reasons={},
        )
        return fallback, warning
