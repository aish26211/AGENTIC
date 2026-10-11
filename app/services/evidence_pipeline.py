"""Pipeline orchestrating discovery, candidate selection, document retrieval, and passage ranking."""

from typing import Any, Dict, List, Optional
import httpx

from app.services.web_search import (
    extract_keywords,
    generate_search_queries,
    search_web_candidates,
)
from app.services.web_retrieval import (
    DEFAULT_CACHE_DIR,
    SimpleFileCache,
    retrieve_and_extract_candidate,
)


def run_web_evidence_pipeline(
    claim: str,
    max_candidates: int = 5,
    max_passages: int = 10,
    ddgs_client: Any = None,
    http_client: Optional[httpx.Client] = None,
    cache: Optional[SimpleFileCache] = None,
) -> Dict[str, Any]:
    """Execute the two-stage web evidence retrieval pipeline."""
    queries = generate_search_queries(claim)
    claim_keywords = extract_keywords(claim)

    if cache is None:
        cache = SimpleFileCache(cache_dir=DEFAULT_CACHE_DIR)

    warnings: List[str] = []

    # Stage 1: Search & Discovery
    candidates = search_web_candidates(
        claim=claim,
        max_candidates=max_candidates,
        ddgs_client=ddgs_client,
        queries=queries,
        warnings=warnings,
    )

    if not candidates:
        warnings.append("No search results found for the generated queries.")

    # Stage 2: Document Retrieval & Extraction
    retrieved_docs: List[Dict[str, Any]] = []
    all_passages: List[Dict[str, Any]] = []

    own_client = False
    client = http_client
    if client is None:
        client = httpx.Client(follow_redirects=True, timeout=10.0)
        own_client = True

    try:
        for candidate in candidates:
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
                retrieved_docs.append(doc_data)
                doc_passages = doc_data.get("passages", [])
                all_passages.extend(doc_passages)
    finally:
        if own_client:
            client.close()

    # Rank all extracted passages across all documents by relevance score descending
    all_passages.sort(key=lambda p: (-p["score"], p["title"], p["url"]))

    return {
        "claim": claim,
        "queries": queries,
        "candidates_discovered": len(candidates),
        "documents_retrieved": len(retrieved_docs),
        "passages": all_passages[:max_passages],
        "warnings": warnings,
    }
