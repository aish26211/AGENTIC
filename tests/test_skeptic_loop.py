"""Unit tests for Milestone 5: Skeptic Agent and conditional follow-up research loop."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from app.agents.skeptic_agent import (
    build_skeptic_prompt,
    evaluate_evidence_critique,
    format_assessments_for_critique,
)
from app.main import app
from app.schemas import EvidenceAssessment, SkepticAnalysis
from app.services.investigation_loop import run_investigation_loop
from app.services.web_retrieval import SimpleFileCache

client = TestClient(app)


SAMPLE_ASSESSMENT = EvidenceAssessment(
    evidence_id="ev-1",
    title="ICCT Global EV Study",
    url="https://theicct.org/ev-lifecycle",
    domain="theicct.org",
    stance="contradicts",
    explanation="Study confirms battery electric vehicles have 66% lower lifecycle emissions.",
    quote="substantially lower greenhouse gas emissions",
    limitations="Model assumes average grid mix in Europe.",
    confidence=0.9,
    passage_text="Sample text content.",
)


# 1. Test Skeptic Agent with Sufficient Evidence
def test_skeptic_evaluation_sufficient():
    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "response": json.dumps({
            "is_sufficient": True,
            "strongest_weakness": "Minor regional variance.",
            "missing_context": None,
            "source_concerns": None,
            "unanswered_questions": [],
            "follow_up_queries": [],
            "query_reasons": {},
        })
    }

    analysis, warning = evaluate_evidence_critique(
        claim="Electric vehicles have lower lifecycle emissions",
        assessments=[SAMPLE_ASSESSMENT],
        client=mock_client,
    )

    assert warning is None
    assert analysis.is_sufficient is True
    assert analysis.follow_up_queries == []


# 2. Test Skeptic Agent with Evidence Gaps and Follow-Up Queries
def test_skeptic_evaluation_with_gaps():
    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "response": json.dumps({
            "is_sufficient": False,
            "strongest_weakness": "Only examined passenger cars, omitting heavy transport and battery mineral extraction supply chains.",
            "missing_context": "Supply chain emissions in developing regions.",
            "source_concerns": "Industry funding of ICCT could present potential bias.",
            "unanswered_questions": ["What are the raw extraction emissions for cobalt in the DRC?"],
            "follow_up_queries": ["electric vehicle battery cobalt extraction emissions", "heavy transport electric vs diesel lifecycle emissions"],
            "query_reasons": {
                "electric vehicle battery cobalt extraction emissions": "Address upstream mineral supply chain gap",
                "heavy transport electric vs diesel lifecycle emissions": "Compare beyond passenger vehicles",
            },
        })
    }

    analysis, warning = evaluate_evidence_critique(
        claim="Electric vehicles produce zero emissions overall",
        assessments=[SAMPLE_ASSESSMENT],
        client=mock_client,
    )

    assert warning is None
    assert analysis.is_sufficient is False
    assert len(analysis.follow_up_queries) == 2
    assert "battery cobalt" in analysis.follow_up_queries[0]
    assert len(analysis.unanswered_questions) == 1


# 3. Test Skeptic Agent with Malformed Model Output
def test_skeptic_malformed_output():
    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "response": "Here is my detailed critique without any JSON formatting."
    }

    analysis, warning = evaluate_evidence_critique(
        claim="Test Claim",
        assessments=[SAMPLE_ASSESSMENT],
        client=mock_client,
    )

    assert analysis.is_sufficient is False
    assert warning is not None
    assert "Skeptic Agent error" in warning
    assert analysis.follow_up_queries == []


# 4. Test Skeptic Agent with Ollama Unavailable or Timeout
def test_skeptic_ollama_timeout():
    mock_client = MagicMock()
    mock_client.generate.side_effect = TimeoutError("Ollama connection timed out")

    analysis, warning = evaluate_evidence_critique(
        claim="Test Claim",
        assessments=[SAMPLE_ASSESSMENT],
        client=mock_client,
    )

    assert analysis.is_sufficient is False
    assert warning is not None
    assert "timed out" in warning


# 5. Test Investigation Loop Branch: Follow-Up Triggered & Deduplication
def test_investigation_loop_follow_up_triggered(tmp_path: Path):
    cache = SimpleFileCache(cache_dir=tmp_path)

    # Mock DDGS Search
    mock_ddgs = MagicMock()
    # Round 1 returns source A
    # Round 2 search for follow-up returns source A (duplicate) and source B (new)
    def ddgs_text_side_effect(query, max_results=8):
        if "follow-up" in query.lower() or "battery" in query.lower():
            return [
                {"href": "https://example.com/source-a", "title": "Source A Duplicate", "body": "Snippet A"},
                {"href": "https://example.com/source-b", "title": "Source B New", "body": "Snippet B"},
            ]
        return [
            {"href": "https://example.com/source-a", "title": "Source A", "body": "Snippet A"},
        ]
    mock_ddgs.text.side_effect = ddgs_text_side_effect

    # Mock HTTP Client
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.content = b"<html><body><p>Electric vehicle battery research and detailed lifecycle data.</p></body></html>"
    mock_resp.text = "<html><body><p>Electric vehicle battery research and detailed lifecycle data.</p></body></html>"
    mock_http.get.return_value = mock_resp

    # Mock Ollama Client
    mock_ollama = MagicMock()
    # 1st call: Evidence Agent Round 1
    # 2nd call: Skeptic Agent (flags is_sufficient=False, gives follow-up query)
    # 3rd call: Evidence Agent Round 2
    mock_ollama.generate.side_effect = [
        # Round 1 Evidence Agent
        {"response": json.dumps({"stance": "supports", "explanation": "Initial support.", "confidence": 0.8})},
        # Skeptic Agent
        {"response": json.dumps({
            "is_sufficient": False,
            "strongest_weakness": "Missed upstream battery supply chain data.",
            "missing_context": "Cobalt mining emissions.",
            "source_concerns": None,
            "unanswered_questions": ["What about raw extraction?"],
            "follow_up_queries": ["battery mineral lifecycle emissions follow-up"],
            "query_reasons": {"battery mineral lifecycle emissions follow-up": "Investigate upstream mining"},
        })},
        # Round 2 Evidence Agent
        {"response": json.dumps({"stance": "neutral", "explanation": "Detailed mineral data.", "confidence": 0.85})},
    ]

    result = run_investigation_loop(
        claim="Electric cars have lower emissions",
        max_candidates_per_round=2,
        max_passages_per_round=2,
        ddgs_client=mock_ddgs,
        http_client=mock_http,
        ollama_client=mock_ollama,
        cache=cache,
    )

    assert result["follow_up_triggered"] is True
    assert "battery supply chain" in result["follow_up_reason"]
    assert len(result["initial_assessments"]) == 1
    assert len(result["follow_up_assessments"]) == 1
    assert len(result["all_assessments"]) == 2
    # Verify deduplication: source-a was not duplicated in retrieved sources
    assert "https://example.com/source-a" in result["sources_retrieved"]
    assert "https://example.com/source-b" in result["sources_retrieved"]
    assert result["total_documents_retrieved"] == 2


# 6. Test Investigation Loop Branch: Follow-Up Skipped (Sufficient Evidence)
def test_investigation_loop_follow_up_skipped(tmp_path: Path):
    cache = SimpleFileCache(cache_dir=tmp_path)

    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"href": "https://example.com/source-a", "title": "Source A", "body": "Snippet A"},
    ]

    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    sample_html = "<html><body><article><p>Robust empirical lifecycle assessment studies confirm that electric cars have lower emissions overall compared to petrol vehicles across multiple driving cycles.</p></article></body></html>"
    mock_resp.content = sample_html.encode("utf-8")
    mock_resp.text = sample_html
    mock_http.get.return_value = mock_resp

    mock_ollama = MagicMock()
    mock_ollama.generate.side_effect = [
        # Evidence Agent
        {"response": json.dumps({"stance": "supports", "explanation": "Comprehensive support.", "confidence": 0.95})},
        # Skeptic Agent (is_sufficient=True)
        {"response": json.dumps({
            "is_sufficient": True,
            "strongest_weakness": "None significant.",
            "missing_context": None,
            "source_concerns": None,
            "unanswered_questions": [],
            "follow_up_queries": [],
            "query_reasons": {},
        })},
    ]

    result = run_investigation_loop(
        claim="Electric cars have lower emissions",
        max_candidates_per_round=2,
        max_passages_per_round=2,
        ddgs_client=mock_ddgs,
        http_client=mock_http,
        ollama_client=mock_ollama,
        cache=cache,
    )

    assert result["follow_up_triggered"] is False
    assert len(result["follow_up_assessments"]) == 0
    assert len(result["all_assessments"]) == 1
    assert "sufficient" in result["follow_up_reason"].lower()


# 7. Test Two-Round Ceiling (Infinite Loop Prevention)
def test_two_round_limit_enforced(tmp_path: Path):
    cache = SimpleFileCache(cache_dir=tmp_path)

    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"href": "https://example.com/endless-source", "title": "Endless", "body": "Endless"},
    ]

    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.content = b"<html><body><p>Endless research content.</p></body></html>"
    mock_resp.text = "<html><body><p>Endless research content.</p></body></html>"
    mock_http.get.return_value = mock_resp

    mock_ollama = MagicMock()
    # Even if skeptic asks for queries, loop runs round 2 and halts (does not re-invoke skeptic)
    mock_ollama.generate.side_effect = [
        {"response": json.dumps({"stance": "supports", "explanation": "R1 support.", "confidence": 0.7})},
        {"response": json.dumps({
            "is_sufficient": False,
            "strongest_weakness": "Persistent gap.",
            "unanswered_questions": ["Q?"],
            "follow_up_queries": ["further follow-up query"],
            "query_reasons": {"further follow-up query": "Never enough"},
        })},
        {"response": json.dumps({"stance": "neutral", "explanation": "R2 neutral.", "confidence": 0.8})},
    ]

    result = run_investigation_loop(
        claim="Endless loop test",
        ddgs_client=mock_ddgs,
        http_client=mock_http,
        ollama_client=mock_ollama,
        cache=cache,
    )

    # Completed round 2 and returned cleanly without infinite looping
    assert result["follow_up_triggered"] is True
    assert len(result["all_assessments"]) <= 2


# 8. Test POST /evidence/investigate Endpoint
@patch("app.main.run_investigation_loop")
def test_investigate_endpoint(mock_loop):
    mock_loop.return_value = {
        "claim": "Electric vehicles have lower emissions",
        "initial_assessments": [
            {
                "evidence_id": "ev-1",
                "title": "Study A",
                "url": "https://epa.gov/study",
                "domain": "epa.gov",
                "stance": "supports",
                "explanation": "Low emissions.",
                "quote": None,
                "limitations": None,
                "confidence": 0.9,
                "passage_text": "Sample text",
            }
        ],
        "skeptic_analysis": {
            "is_sufficient": False,
            "strongest_weakness": "Sample weakness",
            "missing_context": None,
            "source_concerns": None,
            "unanswered_questions": ["Sample question?"],
            "follow_up_queries": ["sample follow up"],
            "query_reasons": {"sample follow up": "Reason"},
        },
        "follow_up_triggered": True,
        "follow_up_reason": "Sample weakness",
        "follow_up_assessments": [],
        "all_assessments": [
            {
                "evidence_id": "ev-1",
                "title": "Study A",
                "url": "https://epa.gov/study",
                "domain": "epa.gov",
                "stance": "supports",
                "explanation": "Low emissions.",
                "quote": None,
                "limitations": None,
                "confidence": 0.9,
                "passage_text": "Sample text",
            }
        ],
        "total_sources_discovered": 1,
        "total_documents_retrieved": 1,
        "sources_retrieved": ["https://epa.gov/study"],
        "unresolved_questions": ["Sample question?"],
        "warnings": [],
    }

    response = client.post(
        "/evidence/investigate",
        json={"claim": "Electric vehicles have lower emissions"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["claim"] == "Electric vehicles have lower emissions"
    assert data["follow_up_triggered"] is True
    assert data["skeptic_analysis"]["strongest_weakness"] == "Sample weakness"
    assert len(data["all_assessments"]) == 1


def test_investigate_validation_error():
    response = client.post("/evidence/investigate", json={"claim": "   "})
    assert response.status_code == 422
