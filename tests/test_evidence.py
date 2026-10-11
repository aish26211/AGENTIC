"""Tests for the local evidence search service and endpoint."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_evidence_search_matching_query():
    """Verify that a query matching documents returns 200 with structured results."""
    response = client.post("/evidence/search", json={"query": "tailpipe zero emissions"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "tailpipe zero emissions"
    assert data["total_results"] > 0
    assert len(data["results"]) > 0

    top_result = data["results"][0]
    assert "id" in top_result
    assert "title" in top_result
    assert "content" in top_result
    assert "tags" in top_result
    assert "score" in top_result
    assert top_result["score"] > 0
    assert top_result["id"] == "ev-001"


def test_evidence_search_relevance_ordering():
    """Verify that search results are sorted in descending order of relevance score."""
    response = client.post("/evidence/search", json={"query": "battery emissions"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) >= 2

    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_evidence_search_empty_query():
    """Verify that an empty query string is rejected with a 422 validation error."""
    response = client.post("/evidence/search", json={"query": ""})
    assert response.status_code == 422


def test_evidence_search_whitespace_query():
    """Verify that a whitespace-only query is rejected with a 422 validation error."""
    response = client.post("/evidence/search", json={"query": "    "})
    assert response.status_code == 422


def test_evidence_search_no_matches():
    """Verify that an unmatchable query returns 200 with an empty list."""
    response = client.post(
        "/evidence/search",
        json={"query": "quantum spacecraft planetary orbital trajectory"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 0
    assert data["results"] == []


def test_evidence_search_result_limit():
    """Verify that the configurable limit restricts the maximum number of results."""
    response = client.post(
        "/evidence/search",
        json={"query": "emissions", "limit": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) <= 2


def test_existing_health_and_claim_endpoints_preserved():
    """Verify that existing health, home, and claims endpoints remain functional."""
    # Health endpoint
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    # Home endpoint
    res_home = client.get("/")
    assert res_home.status_code == 200
    assert res_home.json()["status"] == "healthy"

    # Claims endpoint
    res_claim = client.post(
        "/claims",
        json={"claim": "Electric vehicles produce zero emissions"},
    )
    assert res_claim.status_code == 200
    data_claim = res_claim.json()
    assert data_claim["status"] == "received"
    assert data_claim["claim"] == "Electric vehicles produce zero emissions"
