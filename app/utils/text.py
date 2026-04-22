from __future__ import annotations

import re
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")
BOILERPLATE_PATTERNS = [
    re.compile(r"\bthank you for contacting paypal support\b", re.I),
    re.compile(r"\bwe appreciate your patience\b", re.I),
    re.compile(r"\bplease let us know if you need anything else\b", re.I),
    re.compile(r"\bkind regards\b", re.I),
]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    value = normalize_text(text)
    for pattern in BOILERPLATE_PATTERNS:
        value = pattern.sub("", value)
    return normalize_text(value)


def truncate_chars(text: str, limit: int) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def join_lines(lines: Iterable[str]) -> str:
    return "\n".join([line for line in lines if line])


def safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and str(value) == "nan":
        return ""
    return str(value)


def keyword_overlap_score(query: str, document: str) -> float:
    q_tokens = set(re.findall(r"[a-zA-Z0-9_]{3,}", query.lower()))
    d_tokens = set(re.findall(r"[a-zA-Z0-9_]{3,}", document.lower()))
    if not q_tokens or not d_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / max(len(q_tokens), 1)
