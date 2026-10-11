"""Pydantic schemas for request and response models."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str


class HomeResponse(BaseModel):
    message: str
    status: str


class ClaimRequest(BaseModel):
    claim: Optional[str] = None
    text: Optional[str] = None


class ClaimResponse(BaseModel):
    id: str = "claim-1"
    status: str = "received"
    claim: Optional[str] = None
    text: Optional[str] = None


class EvidenceDocument(BaseModel):
    id: str
    title: str
    content: str
    tags: List[str]
    score: float


class EvidenceSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query string",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of search results to return",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query cannot be blank or contain only whitespace.")
        return stripped


class EvidenceSearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[EvidenceDocument]


class WebEvidenceSearchRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Claim to investigate and retrieve evidence for",
    )
    max_candidates: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of candidate URLs to retrieve",
    )
    max_passages: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum number of relevant evidence passages to retain",
    )

    @field_validator("claim")
    @classmethod
    def claim_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Claim cannot be blank or contain only whitespace.")
        return stripped


class EvidencePassage(BaseModel):
    title: str
    url: str
    domain: str
    text: str
    score: float = 0.0


class WebEvidenceSearchResponse(BaseModel):
    claim: str
    queries: List[str]
    candidates_discovered: int
    documents_retrieved: int
    passages: List[EvidencePassage]
    warnings: List[str] = []


# --- Milestone 4: Evidence Analysis Agent Schemas ---

StanceType = Literal["supports", "contradicts", "neutral", "insufficient"]


class EvidenceAssessment(BaseModel):
    evidence_id: str
    title: str
    url: str
    domain: str
    stance: StanceType
    explanation: str
    quote: Optional[str] = None
    limitations: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    passage_text: Optional[str] = None


class StanceCounts(BaseModel):
    supports: int = 0
    contradicts: int = 0
    neutral: int = 0
    insufficient: int = 0


class EvidenceAnalyzeRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Claim to analyze against evidence passages",
    )
    passages: Optional[List[EvidencePassage]] = Field(
        default=None,
        description="Optional list of passages to analyze; if omitted, web search is automatically invoked",
    )
    max_candidates: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum candidates to discover if automatic search is used",
    )
    max_passages: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum passages to evaluate",
    )

    @field_validator("claim")
    @classmethod
    def claim_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Claim cannot be blank or contain only whitespace.")
        return stripped


class EvidenceAnalyzeResponse(BaseModel):
    claim: str
    assessments: List[EvidenceAssessment]
    counts: StanceCounts
    warnings: List[str] = []


# --- Milestone 5: Skeptic Agent & Investigation Schemas ---


class SkepticAnalysis(BaseModel):
    is_sufficient: bool = False
    strongest_weakness: str
    missing_context: Optional[str] = None
    source_concerns: Optional[str] = None
    unanswered_questions: List[str] = []
    follow_up_queries: List[str] = []
    query_reasons: Dict[str, str] = {}


class InvestigateRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Claim to investigate through multi-stage conditional research",
    )
    max_candidates_per_round: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum candidate URLs per search round",
    )
    max_passages_per_round: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum evidence passages per search round",
    )

    @field_validator("claim")
    @classmethod
    def claim_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Claim cannot be blank or contain only whitespace.")
        return stripped


class InvestigateResponse(BaseModel):
    claim: str
    initial_assessments: List[EvidenceAssessment]
    skeptic_analysis: SkepticAnalysis
    follow_up_triggered: bool
    follow_up_reason: Optional[str] = None
    follow_up_assessments: List[EvidenceAssessment] = []
    all_assessments: List[EvidenceAssessment] = []
    total_sources_discovered: int
    total_documents_retrieved: int
    sources_retrieved: List[str] = []
    unresolved_questions: List[str] = []
    warnings: List[str] = []
