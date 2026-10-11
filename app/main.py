"""FastAPI application entry point."""

from typing import List
from fastapi import FastAPI
from app.agents.evidence_agent import analyze_evidence_passages
from app.schemas import (
    ClaimRequest,
    ClaimResponse,
    EvidenceAnalyzeRequest,
    EvidenceAnalyzeResponse,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
    HealthResponse,
    HomeResponse,
    InvestigateRequest,
    InvestigateResponse,
    WebEvidenceSearchRequest,
    WebEvidenceSearchResponse,
)
from app.services.evidence_pipeline import run_web_evidence_pipeline
from app.services.evidence_search import search_evidence
from app.services.investigation_loop import run_investigation_loop

app = FastAPI(title="ClaimCheck")


@app.get("/", response_model=HomeResponse)
def home():
    return {
        "message": "Incident War Room is running!",
        "status": "healthy",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@app.post("/claims", response_model=ClaimResponse)
def submit_claim(payload: ClaimRequest):
    return ClaimResponse(
        id="claim-1",
        status="received",
        claim=payload.claim,
        text=payload.text,
    )


@app.post("/evidence/search", response_model=EvidenceSearchResponse)
def search_evidence_endpoint(payload: EvidenceSearchRequest):
    matches = search_evidence(query=payload.query, limit=payload.limit)
    return EvidenceSearchResponse(
        query=payload.query,
        total_results=len(matches),
        results=matches,
    )


@app.post("/evidence/web-search", response_model=WebEvidenceSearchResponse)
def search_web_evidence_endpoint(payload: WebEvidenceSearchRequest):
    result = run_web_evidence_pipeline(
        claim=payload.claim,
        max_candidates=payload.max_candidates,
        max_passages=payload.max_passages,
    )
    return WebEvidenceSearchResponse(**result)


@app.post("/evidence/analyze", response_model=EvidenceAnalyzeResponse)
def analyze_evidence_endpoint(payload: EvidenceAnalyzeRequest):
    warnings: List[str] = []

    if payload.passages is not None and len(payload.passages) > 0:
        passages_to_analyze = [p.model_dump() for p in payload.passages]
    else:
        pipeline_result = run_web_evidence_pipeline(
            claim=payload.claim,
            max_candidates=payload.max_candidates,
            max_passages=payload.max_passages,
        )
        passages_to_analyze = pipeline_result.get("passages", [])
        warnings.extend(pipeline_result.get("warnings", []))

    assessments, counts, agent_warnings = analyze_evidence_passages(
        claim=payload.claim,
        passages=passages_to_analyze,
    )
    warnings.extend(agent_warnings)

    return EvidenceAnalyzeResponse(
        claim=payload.claim,
        assessments=assessments,
        counts=counts,
        warnings=warnings,
    )


@app.post("/evidence/investigate", response_model=InvestigateResponse)
@app.post("/investigate", response_model=InvestigateResponse)
def investigate_claim_endpoint(payload: InvestigateRequest):
    result = run_investigation_loop(
        claim=payload.claim,
        max_candidates_per_round=payload.max_candidates_per_round,
        max_passages_per_round=payload.max_passages_per_round,
    )
    return InvestigateResponse(**result)
