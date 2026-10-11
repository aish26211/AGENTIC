"""Stage 1: Web search query generation, discovery, deduplication, and ranking."""

import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None  # Handled gracefully if neither is installed

# Common stop words for keyword extraction
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "should", "it",
    "its", "this", "that", "these", "those", "they", "them", "their", "we",
    "us", "our", "you", "your", "he", "him", "his", "she", "her", "i", "me",
    "my", "as", "than", "so", "no", "not", "too", "very", "more", "less",
}

# Domain reputation hints (slight preference for official/research sources)
PREFERRED_TLDS = (".gov", ".edu", ".org", ".ac.uk")
PREFERRED_DOMAINS = (
    "nature.com", "sciencedirect.com", "iea.org", "epa.gov",
    "who.int", "ipcc.ch", "arxiv.org", "reuters.com", "bbc.com",
)


def extract_keywords(text: str) -> List[str]:
    """Extract lowercased alphanumeric keywords excluding common stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def generate_search_queries(claim: str) -> List[str]:
    """Deterministically generate 2-3 focused search queries from a claim."""
    cleaned = claim.strip()
    keywords = extract_keywords(cleaned)
    kw_str = " ".join(keywords)

    queries: List[str] = [cleaned]

    if kw_str and kw_str.lower() != cleaned.lower():
        queries.append(f"{kw_str} study evidence")
    else:
        queries.append(f"{cleaned} research data")

    # Add contrast/lifecycle query for comparative claims
    if any(term in keywords for term in ("more", "less", "better", "worse", "than", "versus", "vs")):
        queries.append(f"{kw_str} comparison facts")
    else:
        queries.append(f"{kw_str} analysis report")

    unique_queries: List[str] = []
    for q in queries:
        if q and q not in unique_queries:
            unique_queries.append(q)
    return unique_queries[:3]


def canonicalize_url(url: str) -> str:
    """Normalize URL by stripping tracking parameters, lowercase domain, and trailing slash."""
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        tracking_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "fbclid"}
        filtered_query = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in tracking_params
        ]
        query_str = urlencode(filtered_query)

        return urlunparse((parsed.scheme.lower() or "https", netloc, path, "", query_str, ""))
    except Exception:
        return url.strip()


def extract_domain(url: str) -> str:
    """Extract clean domain name from URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def score_candidate(title: str, snippet: str, domain: str, claim_keywords: List[str]) -> float:
    """Score a candidate based on keyword overlap with title/snippet and slight domain preference."""
    if not claim_keywords:
        return 0.0

    title_words = set(extract_keywords(title))
    snippet_words = set(extract_keywords(snippet))

    score = 0.0
    for kw in claim_keywords:
        if kw in title_words:
            score += 3.0
        if kw in snippet_words:
            score += 1.5

    if any(domain.endswith(tld) for tld in PREFERRED_TLDS):
        score += 1.0
    if any(pref in domain for pref in PREFERRED_DOMAINS):
        score += 1.0

    return round(score, 2)


def search_web_candidates(
    claim: str,
    max_candidates: int = 5,
    ddgs_client: Any = None,
    queries: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Execute search discovery across queries, deduplicate, rank, and capture diagnostics."""
    if queries is None:
        queries = generate_search_queries(claim)
    claim_keywords = extract_keywords(claim)

    client = ddgs_client if ddgs_client is not None else (DDGS() if DDGS else None)
    if client is None:
        if warnings is not None:
            warnings.append("Search provider is not available or not installed.")
        return []

    discovered_by_canonical: Dict[str, Dict[str, Any]] = {}

    for query in queries:
        try:
            results = client.text(query, max_results=8)
            if not results:
                if warnings is not None:
                    warnings.append(f"Search provider returned 0 results for query: '{query}'")
                continue

            query_valid_urls = 0
            for res in results:
                raw_url = res.get("href") or res.get("url") or ""
                if not raw_url:
                    continue

                canonical = canonicalize_url(raw_url)
                if not canonical.startswith("http"):
                    continue

                query_valid_urls += 1
                title = res.get("title") or ""
                snippet = res.get("body") or res.get("snippet") or ""
                domain = extract_domain(canonical)

                if canonical not in discovered_by_canonical:
                    score = score_candidate(title, snippet, domain, claim_keywords)
                    discovered_by_canonical[canonical] = {
                        "url": canonical,
                        "raw_url": raw_url,
                        "title": title,
                        "snippet": snippet,
                        "domain": domain,
                        "score": score,
                    }

            if query_valid_urls == 0 and warnings is not None:
                warnings.append(f"Results returned for '{query}', but no valid HTTP URLs were present.")

        except Exception as exc:
            err_msg = str(exc)
            if "ratelimit" in err_msg.lower() or "202" in err_msg:
                diag = f"Search provider rate limit encountered on query '{query}'."
            elif "timeout" in err_msg.lower():
                diag = f"Search provider timed out on query '{query}'."
            else:
                diag = f"Search provider error on query '{query}': {type(exc).__name__} - {err_msg}"
            if warnings is not None:
                warnings.append(diag)
            continue

    candidates = list(discovered_by_canonical.values())
    candidates.sort(key=lambda c: (-c["score"], c["url"]))
    return candidates[:max_candidates]
