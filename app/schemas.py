from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.utils.text import join_lines


class TurnMessage(BaseModel):
    role: Literal["customer", "support", "system", "assistant", "user", "agent"]
    message: str = Field(min_length=1)


class AssistRequest(BaseModel):
    query: str | None = None
    debug: bool = False

    thread_subject: str | None = None
    transcript: str | None = None
    messages: list[TurnMessage] | None = None

    channel: str | None = None
    country: str | None = None
    currency: str | None = None
    amount: float | None = None
    payment_method: str | None = None
    merchant_name: str | None = None
    plugin_or_stack: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "AssistRequest":
        if not self.query and not self.transcript and not self.messages:
            raise ValueError("Provide query, transcript, or messages.")
        return self

    def render_transcript(self) -> str:
        if self.query:
            return self.query.strip()
        if self.transcript:
            return self.transcript
        assert self.messages is not None
        return join_lines(f"[{item.role}] {item.message}" for item in self.messages)

    def effective_subject(self) -> str:
        if self.thread_subject:
            return self.thread_subject
        if self.query:
            return self.query[:120]
        return ""

    def metadata_dict(self) -> dict[str, str | float | None]:
        return {
            "channel": self.channel,
            "country": self.country,
            "currency": self.currency,
            "amount": self.amount,
            "payment_method": self.payment_method,
            "merchant_name": self.merchant_name,
            "plugin_or_stack": self.plugin_or_stack,
        }


class DistilledQueryResponse(BaseModel):
    text: str
    signals: dict[str, str | float | int | None]
    strategy: str


class SimilarCaseLiteResponse(BaseModel):
    thread_id: str
    similarity_score: float
    reranker_score: float


class SimilarCaseDebugResponse(SimilarCaseLiteResponse):
    thread_subject: str
    issue_family: str
    issue_type: str
    final_action: str
    resolution_summary: str
    channel: str | None = None
    country: str | None = None
    currency: str | None = None
    payment_method: str | None = None


class DebugInfoResponse(BaseModel):
    distilled_query: DistilledQueryResponse
    retrieval_query: str
    likely_issue_family: str
    likely_issue_type: str
    probable_root_cause: str
    internal_resolution_hypothesis: str


class AssistResponse(BaseModel):
    answer: str
    confidence: str
    similar_cases: list[SimilarCaseLiteResponse]
    debug_info: DebugInfoResponse | None = None
    similar_cases_debug: list[SimilarCaseDebugResponse] | None = None


class RetrieveResponse(BaseModel):
    distilled_query: DistilledQueryResponse
    retrieval_query: str
    similar_cases: list[SimilarCaseDebugResponse]
