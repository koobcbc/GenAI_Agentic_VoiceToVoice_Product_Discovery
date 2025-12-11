# web_search.py
# MCP tool for performing web search via an external API
# Defines MCP tool entrypoint for web search
# Serper.dev API is used for web and shopping search

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import re
import httpx
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")  
SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_SHOPPING_URL = "https://google.serper.dev/shopping"

def _quality_sort_key(item: Dict[str, Any]):
    """
    Sort key for shopping results:
      - Prefer items with rating
      - Then higher rating
      - Then higher rating_count
    """
    rating = item.get("rating")
    rating_count = item.get("rating_count")

    has_rating = 1 if rating is not None else 0
    return (
        has_rating,    
        int(rating_count or 0),  # more reviews is better          
        float(rating or 0.0),    # higher rating is better
    )

def _call_serper_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Call Serper.dev generic web search endpoint.
    Returns a normalized list of results with title, url, snippet.
    """
    if not SERPER_API_KEY:
        return []

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "q": query,
        "num": max_results,
    }

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(SERPER_SEARCH_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results: List[Dict[str, Any]] = []
    for item in data.get("organic", [])[:max_results]:
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "price": None,
                "availability": None,
            }
        )
    return results


def _clean_title_for_search(title: str) -> str:
    """
    Make product title less over-specific for search:
      - keep left side of '-' or '|' if present
      - remove some very specific patterns (scale 1/43, size units etc.)
    """
    if not title:
        return ""

    parts = re.split(r"[|\-\(\)\:]", title)
    base = parts[0].strip() if parts else title.strip()

    noise_patterns = [
        r"\b\d+\s*pcs?\b",
        r"\b\d+\s*pc\b",
        r"\b\d+\s*piece(s)?\b",
        r"\b\d+\s*(inch|inches|in)\b",
        r"\b\d+\s*(cm|mm|oz|g|ml)\b",
        r"\b1/\d+\s*scale\b",
        r"\bfor ages?\s*\d+\+?\b",
    ]
    for pat in noise_patterns:
        base = re.sub(pat, "", base, flags=re.IGNORECASE)

    # normalize whitespace
    base = re.sub(r"\s+", " ", base).strip()
    return base

def _call_serper_shopping(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Call Serper.dev shopping search endpoint.
    Useful when you care about price & availability.
    """
    if not SERPER_API_KEY:
        return []

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "q": query,
        "num": max_results,
    }

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(SERPER_SHOPPING_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results: List[Dict[str, Any]] = []
    for item in data.get("shopping", [])[:max_results]:
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "price": item.get("price"),           
                "availability": item.get("availability") or item.get("condition"),
                "rating": item.get("rating"),
                "rating_count": item.get("ratingCount"),
            }
        )

    cleaned_results = []
    for item in results:
        url = (item.get("url") or "").strip()

        if not url:
            continue

        if "q=nan" in url.lower():
            continue

        cleaned_results.append(item)

    if not cleaned_results:
        cleaned_results = results

    sorted_results = sorted(
        cleaned_results,
        key=_quality_sort_key,
        reverse=True,  
    )

    return sorted_results[:max_results]

def web_search_tool( query: str, max_results: int = 5, mode: str = "shopping") -> Dict[str, Any]:
    raw_query = query
    cleaned_query = _clean_title_for_search(raw_query)
    query = cleaned_query if cleaned_query else raw_query

    if not SERPER_API_KEY:
        return {
            "results": [],
            "note": "SERPER_API_KEY not set."
        }

    if mode == "shopping":
        results = _call_serper_shopping(query, max_results=max_results)
    else:
        results = _call_serper_search(query, max_results=max_results)

    return {"results": results, "note": None}

WEB_SEARCH_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "WebSearchInput",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural language product query to send to the web search API.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 5,
            "minimum": 1,
            "maximum": 10,
        },
        "mode": {
            "type": "string",
            "enum": ["web", "shopping"],
            "description": "Use 'shopping' for product/price search, 'web' for general info.",
            "default": "shopping",
        },
    },
    "required": ["query"],
}

WEB_SEARCH_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "WebSearchOutput",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                    "snippet": {"type": ["string", "null"]},
                    "price": {"type": ["number", "string", "null"]},
                    "availability": {"type": ["string", "null"]},
                    "rating": {"type": ["number", "null"]},
                    "rating_count": {"type": ["integer", "null"]},
                },
            },
        },
        "note": {"type": ["string", "null"]},
    },
    "required": ["results"],
}
