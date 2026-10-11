"""Unit tests for web search discovery, extraction, caching, and pipeline."""

from pathlib import Path
import time
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import app
from app.services.evidence_pipeline import run_web_evidence_pipeline
from app.services.web_retrieval import (
    SimpleFileCache,
    extract_document_text,
    fetch_page,
    retrieve_and_extract_candidate,
    split_text_into_passages,
)
from app.services.web_search import (
    canonicalize_url,
    extract_domain,
    generate_search_queries,
    score_candidate,
    search_web_candidates,
)

client = TestClient(app)


# 1. Test search query generation
def test_search_query_generation():
    claim = "Electric vehicles produce more emissions than petrol cars"
    queries = generate_search_queries(claim)
    assert len(queries) >= 2
    assert len(queries) <= 3
    # Check that the original claim or core terms are preserved
    assert any("electric vehicles" in q.lower() or "emissions" in q.lower() for q in queries)


# 2. Test candidate deduplication and canonicalization
def test_url_canonicalization_and_deduplication():
    url1 = "https://www.example.com/article/?utm_source=twitter&utm_medium=social"
    url2 = "https://example.com/article/"
    url3 = "https://example.com/article?ref=homepage"

    c1 = canonicalize_url(url1)
    c2 = canonicalize_url(url2)
    c3 = canonicalize_url(url3)

    assert c1 == "https://example.com/article"
    assert c2 == "https://example.com/article"
    assert c3 == "https://example.com/article"

    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"href": url1, "title": "Article 1", "body": "Snippet 1"},
        {"href": url2, "title": "Article 1 Duplicate", "body": "Snippet 1 Duplicate"},
        {"href": "https://example.org/other-page", "title": "Article 2", "body": "Snippet 2"},
    ]

    candidates = search_web_candidates(
        claim="electric vehicles",
        max_candidates=5,
        ddgs_client=mock_ddgs,
    )
    # url1 and url2 should collapse into a single candidate
    candidate_urls = [c["url"] for c in candidates]
    assert len(candidate_urls) == 2
    assert "https://example.com/article" in candidate_urls
    assert "https://example.org/other-page" in candidate_urls


# 3. Test candidate ranking
def test_candidate_ranking():
    claim_keywords = ["electric", "vehicles", "emissions"]
    score_high = score_candidate(
        title="Electric Vehicles Lifecycle Emissions Report",
        snippet="Comprehensive data on electric vehicles emissions.",
        domain="epa.gov",
        claim_keywords=claim_keywords,
    )
    score_low = score_candidate(
        title="Baking Sourdough Bread at Home",
        snippet="Flour, water, and yeast instructions.",
        domain="cooking.com",
        claim_keywords=claim_keywords,
    )
    assert score_high > score_low
    assert score_high >= 4.0
    assert score_low == 0.0


# 4. Test empty search results handling
def test_empty_search_results():
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = []

    result = run_web_evidence_pipeline(
        claim="totally unmatchable query with no internet hits",
        max_candidates=5,
        ddgs_client=mock_ddgs,
    )
    assert result["candidates_discovered"] == 0
    assert result["documents_retrieved"] == 0
    assert result["passages"] == []
    assert any("No search results found" in w for w in result["warnings"])


# 5. Test successful extraction and metadata
def test_successful_extraction():
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>EV Emissions Study</title></head>
    <body>
        <article>
            <h1>EV Lifecycle Emissions Analysis</h1>
            <p>Electric cars generate significantly fewer lifecycle emissions than internal combustion vehicles.</p>
            <p>Over a full lifespan, cleaner grids lead to progressive carbon reductions.</p>
        </article>
    </body>
    </html>
    """
    extracted = extract_document_text(
        html=sample_html,
        url="https://sciencedirect.com/article/123",
        fallback_title="Fallback Title",
    )
    assert "Electric cars generate" in extracted["text"]
    assert extracted["domain"] == "sciencedirect.com"

    passages = split_text_into_passages(extracted["text"], min_words=5, max_words=100)
    assert len(passages) >= 1
    assert any("lifecycle emissions" in p.lower() for p in passages)


# 6. Test retrieval failure handling (403, 404, 500)
def test_retrieval_failure_handling():
    mock_response = MagicMock()
    mock_response.status_code = 403

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response

    html, error = fetch_page("https://paywalled.com/locked", client=mock_client)
    assert html is None
    assert "Access restricted" in error


# 7. Test timeout handling
def test_timeout_handling():
    mock_client = MagicMock()
    mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")

    html, error = fetch_page("https://slow-server.org/data", client=mock_client)
    assert html is None
    assert error == "Request timed out"


# 8. Test passage-to-source attribution
def test_passage_to_source_attribution():
    sample_candidate = {
        "url": "https://epa.gov/reports/ev-study",
        "title": "EPA EV Study",
        "domain": "epa.gov",
    }
    sample_html = """
    <html><body><article><p>Electric vehicles operate with zero direct exhaust emissions on the road. Battery electric cars utilize electric motors powered by rechargeable onboard storage.</p></article></body></html>
    """

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.content = sample_html.encode("utf-8")
    mock_response.text = sample_html
    mock_client.get.return_value = mock_response

    doc_data, error = retrieve_and_extract_candidate(
        candidate=sample_candidate,
        claim_keywords=["electric", "vehicles", "zero", "emissions"],
        client=mock_client,
        cache=None,
    )
    assert error is None
    assert doc_data is not None
    passages = doc_data["passages"]
    assert len(passages) >= 1
    for p in passages:
        assert p["url"] == "https://epa.gov/reports/ev-study"
        assert p["title"] == "EPA EV Study"
        assert p["domain"] == "epa.gov"
        assert p["score"] > 0
        assert "zero direct exhaust emissions" in p["text"]


# 9. Test cache behavior and TTL
def test_cache_behavior(tmp_path: Path):
    cache = SimpleFileCache(cache_dir=tmp_path)
    url = "https://example.com/cached-article"
    data = {"title": "Cached", "text": "Cached text content"}

    assert cache.get(url) is None
    cache.set(url, data)
    assert cache.get(url) == data

    # Test TTL expiration
    assert cache.get(url, ttl_seconds=0) is None


# 10. Test POST /evidence/web-search endpoint with mocked pipeline
@patch("app.main.run_web_evidence_pipeline")
def test_web_evidence_search_endpoint(mock_pipeline):
    mock_pipeline.return_value = {
        "claim": "Electric cars produce lower emissions",
        "queries": ["electric cars produce lower emissions", "electric cars emissions study"],
        "candidates_discovered": 2,
        "documents_retrieved": 1,
        "passages": [
            {
                "title": "EV Study",
                "url": "https://epa.gov/ev",
                "domain": "epa.gov",
                "text": "Electric cars reduce emissions over their lifespan.",
                "score": 4.5,
            }
        ],
        "warnings": [],
    }

    response = client.post(
        "/evidence/web-search",
        json={"claim": "Electric cars produce lower emissions", "max_candidates": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["claim"] == "Electric cars produce lower emissions"
    assert data["candidates_discovered"] == 2
    assert data["documents_retrieved"] == 1
    assert len(data["passages"]) == 1
    assert data["passages"][0]["url"] == "https://epa.gov/ev"
    assert data["passages"][0]["score"] == 4.5


def test_web_evidence_search_validation_error():
    # Empty query or whitespace
    response = client.post("/evidence/web-search", json={"claim": "   "})
    assert response.status_code == 422


# 11. Regression tests for search provider diagnostics
def test_search_provider_rate_limit_diagnostic():
    mock_ddgs = MagicMock()
    mock_ddgs.text.side_effect = Exception("202 Ratelimit: rate limit exceeded")

    warnings = []
    candidates = search_web_candidates(
        claim="electric vehicles",
        ddgs_client=mock_ddgs,
        warnings=warnings,
    )
    assert candidates == []
    assert len(warnings) > 0
    assert any("rate limit encountered" in w for w in warnings)


def test_search_provider_zero_results_diagnostic():
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = []

    warnings = []
    candidates = search_web_candidates(
        claim="electric vehicles",
        ddgs_client=mock_ddgs,
        warnings=warnings,
    )
    assert candidates == []
    assert len(warnings) > 0
    assert any("Search provider returned 0 results" in w for w in warnings)


def test_search_provider_explicit_queries_support():
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"href": "https://epa.gov/study", "title": "EPA EV Study", "body": "Snippet text"}
    ]

    warnings = []
    candidates = search_web_candidates(
        claim="ignored fallback",
        queries=["exact targeted query"],
        ddgs_client=mock_ddgs,
        warnings=warnings,
    )
    assert len(candidates) == 1
    mock_ddgs.text.assert_called_once_with("exact targeted query", max_results=8)

