"""Local deterministic evidence search service."""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Set

# Base directory computed relative to this source file, not CWD
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVIDENCE_PATH = BASE_DIR / "data" / "sources" / "evidence.json"

STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "should", "it",
    "its", "this", "that", "these", "those", "they", "them", "their", "we",
    "us", "our", "you", "your", "he", "him", "his", "she", "her", "i", "me",
    "my", "as", "than", "so", "no", "not", "too", "very",
}


def load_evidence(file_path: Path = DEFAULT_EVIDENCE_PATH) -> List[Dict[str, Any]]:
    """Load evidence documents from local JSON storage safely."""
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    """Normalize text to lowercase, extract alphanumeric tokens, and remove stop words."""
    if not text:
        return []
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in raw_tokens if token not in STOP_WORDS and len(token) > 1]


def search_evidence(
    query: str,
    documents: List[Dict[str, Any]] | None = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search local documents deterministically based on query-term overlap.

    Matches query terms against document titles (weight 3.0), tags (weight 2.0),
    and content (weight 1.0). Returns documents sorted by score descending,
    then by document ID.
    """
    if documents is None:
        documents = load_evidence()

    query_tokens = tokenize(query)
    if not query_tokens or not documents:
        return []

    scored_results: List[Dict[str, Any]] = []

    for doc in documents:
        title_tokens = tokenize(doc.get("title", ""))
        content_tokens = tokenize(doc.get("content", ""))

        raw_tags = doc.get("tags", [])
        tag_tokens: List[str] = []
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                tag_tokens.extend(tokenize(str(tag)))
        else:
            tag_tokens = tokenize(str(raw_tags))

        score = 0.0
        for q_term in query_tokens:
            title_count = title_tokens.count(q_term)
            tag_count = tag_tokens.count(q_term)
            content_count = content_tokens.count(q_term)

            term_score = (title_count * 3.0) + (tag_count * 2.0) + (content_count * 1.0)
            score += term_score

        if score > 0:
            scored_doc = {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "tags": doc.get("tags", []),
                "score": round(score, 2),
            }
            scored_results.append(scored_doc)

    scored_results.sort(key=lambda item: (-item["score"], item["id"]))
    return scored_results[:limit]
