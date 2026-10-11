"""Milestone 5: Orchestration service for conditional follow-up investigation."""

from typing import Any, Dict, List, Optional, Set
import httpx

from app.agents.evidence_agent import analyze_evidence_passages
from app.agents.skeptic_agent import evaluate_evidence_critique
from app.schemas import EvidenceAssessment, InvestigateResponse, SkepticAnalysis
from app.services.evidence_pipeline import run_web_evidence_pipeline
from app.services.web_retrieval import (
    DEFAULT_CACHE_DIR,
    SimpleFileCache,
    retrieve_and_extract_candidate,
)
from app.services.web_search import extract_keywords, search_web_candidates


def run_investigation_loop(
    claim: str,
    max_candidates_per_round: int = 3,
    max_passages_per_round: int = 5,
    ddgs_client: Any = None,
    http_client: Optional[httpx.Client] = None,
    ollama_client: Optional[Any] = None,
    cache: Optional[SimpleFileCache] = None,
) -> Dict[str, Any]:
    """Execute multi-stage investigation with conditional follow-up research (maximum 2 rounds)."""
    if cache is None:
        cache = SimpleFileCache(cache_dir=DEFAULT_CACHE_DIR)

    claim_keywords = extract_keywords(claim)
    seen_urls: Set[str] = set()
    retrieved_urls: Set[str] = set()
    warnings: List[str] = []

    # -------------------------------------------------------------
    # ROUND 1: Initial Search & Retrieval
    # -------------------------------------------------------------
    r1_result = run_web_evidence_pipeline(
        claim=claim,
        max_candidates=max_candidates_per_round,
        max_passages=max_passages_per_round,
        ddgs_client=ddgs_client,
        http_client=http_client,
        cache=cache,
    )
    warnings.extend(r1_result.get("warnings", []))
    r1_passages = r1_result.get("passages", [])

    for p in r1_passages:
        url = p.get("url")
        if url:
            seen_urls.add(url)
            retrieved_urls.add(url)

    # Analyze Round 1 passages with Evidence Analysis Agent
    initial_assessments, _, agent_warnings = analyze_evidence_passages(
        claim=claim,
        passages=r1_passages,
        client=ollama_client,
    )
    warnings.extend(agent_warnings)

    # -------------------------------------------------------------
    # EVALUATION: Skeptic Agent
    # -------------------------------------------------------------
    skeptic_analysis, skeptic_warning = evaluate_evidence_critique(
        claim=claim,
        assessments=initial_assessments,
        client=ollama_client,
    )
    if skeptic_warning:
        warnings.append(skeptic_warning)

    # Determine whether follow-up round is justified
    follow_up_triggered = False
    follow_up_reason: Optional[str] = None
    follow_up_assessments: List[EvidenceAssessment] = []

    has_meaningful_gaps = (not skeptic_analysis.is_sufficient) and bool(skeptic_analysis.follow_up_queries)

    if has_meaningful_gaps:
        follow_up_triggered = True
        follow_up_reason = skeptic_analysis.strongest_weakness

        # ---------------------------------------------------------
        # ROUND 2: Targeted Follow-Up Research (Strict 2-round limit)
        # ---------------------------------------------------------
        own_client = False
        client = http_client
        if client is None:
            client = httpx.Client(follow_redirects=True, timeout=10.0)
            own_client = True

        r2_passages: List[Dict[str, Any]] = []

        try:
            for query in skeptic_analysis.follow_up_queries[:2]:
                candidates = search_web_candidates(
                    claim=query,
                    max_candidates=max_candidates_per_round,
                    ddgs_client=ddgs_client,
                    queries=[query],
                    warnings=warnings,
                )

                # Deduplicate against all previously seen sources
                new_candidates = [c for c in candidates if c["url"] not in seen_urls]
                for c in new_candidates:
                    seen_urls.add(c["url"])

                for candidate in new_candidates:
                    doc_data, error = retrieve_and_extract_candidate(
                        candidate=candidate,
                        claim_keywords=claim_keywords,
                        cache=cache,
                        client=client,
                    )
                    if error:
                        warnings.append(error)
                        continue

                    if doc_data:
                        retrieved_urls.add(candidate["url"])
                        doc_passages = doc_data.get("passages", [])
                        r2_passages.extend(doc_passages)
        finally:
            if own_client:
                client.close()

        # Rank and limit Round 2 passages
        r2_passages.sort(key=lambda p: (-p.get("score", 0), p.get("title", ""), p.get("url", "")))
        selected_r2_passages = r2_passages[:max_passages_per_round]

        if selected_r2_passages:
            # Analyze Round 2 passages
            r2_assessments, _, r2_agent_warnings = analyze_evidence_passages(
                claim=claim,
                passages=selected_r2_passages,
                client=ollama_client,
            )
            warnings.extend(r2_agent_warnings)

            # Re-label IDs to indicate round 2 clearly
            for idx, a in enumerate(r2_assessments, start=1):
                a.evidence_id = f"ev-r2-{idx}"
            follow_up_assessments = r2_assessments
        else:
            warnings.append("Follow-up research did not discover additional new passages.")
    else:
        follow_up_reason = (
            "Evidence considered sufficient by Skeptic Agent."
            if skeptic_analysis.is_sufficient
            else "No follow-up queries proposed."
        )

    all_assessments = initial_assessments + follow_up_assessments

    return {
        "claim": claim,
        "initial_assessments": initial_assessments,
        "skeptic_analysis": skeptic_analysis,
        "follow_up_triggered": follow_up_triggered,
        "follow_up_reason": follow_up_reason,
        "follow_up_assessments": follow_up_assessments,
        "all_assessments": all_assessments,
        "total_sources_discovered": len(seen_urls),
        "total_documents_retrieved": len(retrieved_urls),
        "sources_retrieved": sorted(list(retrieved_urls)),
        "unresolved_questions": skeptic_analysis.unanswered_questions,
        "warnings": warnings,
    }
