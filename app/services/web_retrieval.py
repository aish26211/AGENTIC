"""Stage 2: Safe document retrieval, HTML extraction, passage splitting, and local file caching."""

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
try:
    import trafilatura
except ImportError:
    trafilatura = None

from app.services.web_search import extract_domain, extract_keywords

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = BASE_DIR / "data" / "cache"

USER_AGENT = "ClaimCheckBot/1.0 (+https://github.com/aish26211/AGENTIC; claim-verification research)"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB limit
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CACHE_TTL = 86400  # 24 hours


class SimpleFileCache:
    """Local JSON cache for retrieved pages with TTL expiration."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, url: str) -> Path:
        url_hash = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()
        return self.cache_dir / f"{url_hash}.json"

    def get(self, url: str, ttl_seconds: int = DEFAULT_CACHE_TTL) -> Optional[Dict[str, Any]]:
        file_path = self._get_path(url)
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            cached_at = entry.get("cached_at", 0)
            if time.time() - cached_at > ttl_seconds:
                return None  # Expired
            return entry.get("data")
        except Exception:
            return None

    def set(self, url: str, data: Dict[str, Any]) -> None:
        file_path = self._get_path(url)
        try:
            payload = {
                "url": url,
                "cached_at": time.time(),
                "data": data,
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass  # Non-blocking if cache write fails


def fetch_page(
    url: str,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch HTML content safely, respecting redirects, size limits, and access codes."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    own_client = False
    if client is None:
        client = httpx.Client(follow_redirects=True, timeout=timeout)
        own_client = True

    try:
        response = client.get(url, headers=headers)
        if response.status_code in (401, 403):
            return None, f"Access restricted (HTTP {response.status_code})"
        if response.status_code == 429:
            return None, "Rate limited by host (HTTP 429)"
        if response.status_code >= 400:
            return None, f"HTTP error {response.status_code}"

        # Validate content type
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return None, f"Unsupported content-type: {content_type.split(';')[0]}"

        # Validate response size limit
        content_bytes = response.content
        if len(content_bytes) > MAX_RESPONSE_BYTES:
            return None, f"Response exceeded size limit ({len(content_bytes)} bytes)"

        return response.text, None
    except httpx.TimeoutException:
        return None, "Request timed out"
    except httpx.RequestError as exc:
        return None, f"Network error: {str(exc)}"
    except Exception as exc:
        return None, f"Unexpected error: {str(exc)}"
    finally:
        if own_client:
            client.close()


def extract_document_text(
    html: str,
    url: str,
    fallback_title: str = "",
) -> Dict[str, Any]:
    """Extract readable body text and metadata using trafilatura."""
    title = fallback_title
    date = None
    extracted_text = ""

    if trafilatura:
        try:
            metadata = trafilatura.extract_metadata(html, default_url=url)
            if metadata and metadata.title:
                title = metadata.title
            if metadata and metadata.date:
                date = metadata.date
            extracted_text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=False,
                output_format="txt",
            ) or ""
        except Exception:
            extracted_text = ""

    # Fallback cleanup if trafilatura produced empty output
    if not extracted_text:
        # Strip script/style tags and basic HTML
        clean = re.sub(r"<(script|style).*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        extracted_text = re.sub(r"\s+", " ", clean).strip()

    return {
        "url": url,
        "title": title,
        "date": date,
        "domain": extract_domain(url),
        "text": extracted_text,
    }


def split_text_into_passages(text: str, min_words: int = 15, max_words: int = 120) -> List[str]:
    """Break extracted text into clean, coherent passages."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        # Fallback to sentence splitting
        paragraphs = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]

    passages: List[str] = []
    current_passage: List[str] = []
    current_word_count = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

        if current_word_count + len(words) <= max_words:
            current_passage.append(paragraph)
            current_word_count += len(words)
        else:
            if current_word_count >= min_words:
                passages.append(" ".join(current_passage))
            current_passage = [paragraph]
            current_word_count = len(words)

    if current_word_count >= min_words:
        passages.append(" ".join(current_passage))

    return passages


def score_passage(passage: str, claim_keywords: List[str]) -> float:
    """Score passage relevance by counting occurrences of claim keywords."""
    if not claim_keywords:
        return 0.0

    passage_tokens = extract_keywords(passage)
    if not passage_tokens:
        return 0.0

    score = 0.0
    for kw in claim_keywords:
        score += passage_tokens.count(kw)

    # Slight density normalizer
    density = score / (len(passage_tokens) ** 0.5) if passage_tokens else 0.0
    return round(score + density, 2)


def retrieve_and_extract_candidate(
    candidate: Dict[str, Any],
    claim_keywords: List[str],
    cache: Optional[SimpleFileCache] = None,
    client: Optional[httpx.Client] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieve, extract, passage-split, and cache a candidate page."""
    url = candidate["url"]
    fallback_title = candidate.get("title", "")

    # Check cache first
    if cache is not None:
        cached_doc = cache.get(url)
        if cached_doc:
            return cached_doc, None

    # Fetch page
    html, error = fetch_page(url, client=client)
    if error or not html:
        return None, f"Failed retrieving {url}: {error}"

    # Extract text
    doc_data = extract_document_text(html, url, fallback_title=fallback_title)
    if not doc_data["text"] or len(doc_data["text"].split()) < 5:
        return None, f"Extracted text too sparse or empty from {url}"

    # Split passages and attach attribution
    passages = split_text_into_passages(doc_data["text"], min_words=5)
    scored_passages: List[Dict[str, Any]] = []
    for passage in passages:
        score = score_passage(passage, claim_keywords)
        if score > 0:
            scored_passages.append({
                "url": url,
                "title": doc_data["title"] or fallback_title,
                "domain": doc_data["domain"],
                "text": passage,
                "score": score,
            })

    doc_data["passages"] = scored_passages

    # Save to cache
    if cache is not None:
        cache.set(url, doc_data)

    return doc_data, None
