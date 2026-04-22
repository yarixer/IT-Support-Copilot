from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DistilledQuery:
    text: str
    signals: dict[str, Any] = field(default_factory=dict)
    strategy: str = "heuristic"


@dataclass
class RetrievedCase:
    thread_id: str
    thread_subject: str
    case_text: str
    issue_family: str
    issue_type: str
    final_action: str
    resolution_summary: str
    channel: str = ""
    country: str = ""
    currency: str = ""
    payment_method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0
    reranker_score: float = 0.0
    hybrid_score: float = 0.0


@dataclass
class AssistResult:
    answer: str
    confidence: str = "medium"
    likely_issue_family: str = "unknown"
    likely_issue_type: str = "unknown"
    probable_root_cause: str = "unknown"
    internal_resolution_hypothesis: str = ""
