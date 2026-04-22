from __future__ import annotations

import json
import re
from collections import Counter
from typing import Callable, Sequence

from app.domain import AssistResult, DistilledQuery, RetrievedCase
from app.schemas import AssistRequest
from app.utils.text import join_lines, keyword_overlap_score, normalize_text

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
EN_SUPPORT_RE = re.compile(r"\b(customer|buyer|what should i do|what to do|charged twice|double charge|refund|how to help)\b", re.I)
INTERNAL_FIX_RE = re.compile(
    r"\b(disable duplicate submissions|de-bounce|debounce|centralise capture|centralize capture|capture logic|idempotency|api request|operator and automated job|subscription or stored billing|product logic|checkout ui)\b",
    re.I,
)


class GroundedSummaryGenerator:
    def __init__(self, generator_factory: Callable[[], object], settings):
        self._generator_factory = generator_factory
        self._generator = None
        self.settings = settings

    def _get_generator(self):
        if self._generator is None:
            self._generator = self._generator_factory()
        return self._generator

    def preload(self) -> None:
        _ = self._get_generator()

    def _generate_kwargs(self, max_new_tokens: int, *, temperature: float | None = None, top_p: float | None = None) -> dict:
        kwargs = {"max_new_tokens": max_new_tokens}
        if temperature is not None and temperature > 0:
            kwargs["temperature"] = temperature
            if top_p is not None and top_p < 1.0:
                kwargs["top_p"] = top_p
        return kwargs

    def warmup(self) -> None:
        generator = self._get_generator()
        _ = generator.generate(
            'Return exactly this JSON: {"answer":"ok","confidence":"low","likely_issue_family":"unknown","likely_issue_type":"unknown","probable_root_cause":"unknown","internal_resolution_hypothesis":"warmup"}',
            **self._generate_kwargs(24),
        )

    def _max_overlap(self, distilled_query: DistilledQuery, cases: Sequence[RetrievedCase]) -> float:
        scores = [
            keyword_overlap_score(distilled_query.text, f"{case.thread_subject}\n{case.resolution_summary}\n{case.final_action}")
            for case in cases
        ]
        return max(scores) if scores else 0.0

    def _agreement_count(self, values: Sequence[str]) -> int:
        cleaned = [value.strip().lower() for value in values if value and value.strip()]
        if not cleaned:
            return 0
        return Counter(cleaned).most_common(1)[0][1]

    def _evidence_confidence(self, distilled_query: DistilledQuery, cases: Sequence[RetrievedCase]) -> str:
        if not cases:
            return "low"
        max_overlap = self._max_overlap(distilled_query, cases)
        top_reranker = max((case.reranker_score for case in cases), default=0.0)
        issue_type_agreement = self._agreement_count([case.issue_type for case in cases])
        final_action_agreement = self._agreement_count([case.final_action for case in cases])
        if max_overlap < self.settings.low_confidence_overlap_threshold:
            return "low"
        if top_reranker < self.settings.low_confidence_reranker_threshold:
            return "low"
        if top_reranker >= self.settings.high_confidence_reranker_threshold and max(issue_type_agreement, final_action_agreement) >= 2:
            return "high"
        if top_reranker >= self.settings.medium_confidence_reranker_threshold:
            return "medium"
        return "low"

    def _detect_support_intent(self, ticket: AssistRequest) -> bool:
        text = normalize_text(ticket.query or ticket.render_transcript()).lower()
        return bool(EN_SUPPORT_RE.search(text))

    def _dominant_issue_type(self, cases: Sequence[RetrievedCase]) -> str:
        values = [c.issue_type.strip().lower() for c in cases if c.issue_type and c.issue_type.strip()]
        return Counter(values).most_common(1)[0][0] if values else "unknown"

    def _has_internal_fix_cases(self, cases: Sequence[RetrievedCase]) -> bool:
        haystack = "\n".join(f"{c.final_action}\n{c.resolution_summary}" for c in cases)
        return bool(INTERNAL_FIX_RE.search(haystack))

    def _customer_safe_answer(self, issue_type: str, cases: Sequence[RetrievedCase]) -> str | None:
        issue_type = (issue_type or "").lower().strip()
        if issue_type == "duplicate_charge":
            if self._has_internal_fix_cases(cases):
                return (
                    "First, check the payment history and confirm whether this is truly two completed charges or one completed payment plus a temporary authorization hold. "
                    "If both entries are completed charges for the same purchase, open the payment and follow the duplicate charge reporting or escalation path; if the second entry is still pending, wait for the status to update before taking further action."
                )
            return (
                "Check the payment history and confirm that this is truly two matching completed charges. "
                "If yes, open the payment and follow the duplicate charge reporting path; if one entry is still pending, verify whether it is only a temporary authorization hold before escalating."
            )
        if issue_type == "dispute_evidence":
            return (
                "Ask the customer to verify the file format and contents, then upload the evidence again in a supported format. "
                "If the error still appears even for PDF or PNG files, escalate the case as an evidence upload flow issue."
            )
        if issue_type in {"authorization_hold", "auth_hold"}:
            return (
                "Check whether the second entry is a temporary authorization hold or a completed charge. "
                "If it is only a hold, it usually disappears automatically; if there is a second completed charge, treat it as a duplicate charge case."
            )
        return None

    def _fallback_result(self, distilled_query: DistilledQuery, cases: Sequence[RetrievedCase], ticket: AssistRequest | None = None) -> AssistResult:
        if not cases:
            return AssistResult(
                answer="No similar resolved cases were found. Manual review is required.",
                confidence="low",
                internal_resolution_hypothesis="No retrieved evidence was available.",
            )
        top = cases[0]
        support_answer = None
        if ticket is not None and self._detect_support_intent(ticket):
            support_answer = self._customer_safe_answer(self._dominant_issue_type(cases), cases)
        return AssistResult(
            answer=support_answer or f"This is closest to case '{normalize_text(top.thread_subject)}'. Start with this action: {top.final_action or 'manual review'}.",
            confidence=self._evidence_confidence(distilled_query, cases),
            likely_issue_family=(top.issue_family or "unknown"),
            likely_issue_type=(top.issue_type or "unknown"),
            probable_root_cause="unknown",
            internal_resolution_hypothesis=normalize_text(top.resolution_summary),
        )

    def _parse_json_result(self, raw_text: str) -> AssistResult | None:
        match = JSON_BLOCK_RE.search(raw_text)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if "answer" not in payload:
            return None
        return AssistResult(
            answer=str(payload.get("answer", "")).strip(),
            confidence=str(payload.get("confidence", "medium")).strip() or "medium",
            likely_issue_family=str(payload.get("likely_issue_family", "unknown")).strip() or "unknown",
            likely_issue_type=str(payload.get("likely_issue_type", "unknown")).strip() or "unknown",
            probable_root_cause=str(payload.get("probable_root_cause", "unknown")).strip() or "unknown",
            internal_resolution_hypothesis=str(payload.get("internal_resolution_hypothesis", "")).strip(),
        )

    def _calibrate_result(self, parsed: AssistResult, distilled_query: DistilledQuery, cases: Sequence[RetrievedCase], ticket: AssistRequest) -> AssistResult:
        evidence_confidence = self._evidence_confidence(distilled_query, cases)
        allowed = {"low": 0, "medium": 1, "high": 2}
        parsed_confidence = (parsed.confidence or "medium").strip().lower()
        if parsed_confidence not in allowed:
            parsed_confidence = evidence_confidence
        if allowed.get(parsed_confidence, 1) > allowed.get(evidence_confidence, 1):
            parsed.confidence = evidence_confidence
        else:
            parsed.confidence = parsed_confidence

        if self._detect_support_intent(ticket):
            dominant_issue_type = self._dominant_issue_type(cases)
            support_answer = self._customer_safe_answer(dominant_issue_type, cases)
            if support_answer and (self._has_internal_fix_cases(cases) or INTERNAL_FIX_RE.search(parsed.answer or "")):
                parsed.answer = support_answer
                if parsed.confidence == "high":
                    parsed.confidence = "medium"
        return parsed

    def generate(
        self,
        ticket: AssistRequest,
        distilled_query: DistilledQuery,
        cases: Sequence[RetrievedCase],
        max_new_tokens: int = 240,
    ) -> AssistResult:
        if not cases:
            return self._fallback_result(distilled_query, cases, ticket=ticket)

        if self._max_overlap(distilled_query, cases) < self.settings.low_confidence_overlap_threshold:
            return AssistResult(
                answer="Similar cases were found with low topical agreement. Manual review is recommended before taking action.",
                confidence="low",
                internal_resolution_hypothesis="Retrieved evidence is too weak to support a grounded action.",
            )

        support_intent = self._detect_support_intent(ticket)
        dominant_issue_type = self._dominant_issue_type(cases)
        support_answer = self._customer_safe_answer(dominant_issue_type, cases) if support_intent else None

        case_blocks = []
        for idx, case in enumerate(cases, start=1):
            case_blocks.append(
                join_lines(
                    [
                        f"Case {idx}",
                        f"thread_id: {case.thread_id}",
                        f"subject: {case.thread_subject}",
                        f"issue_family: {case.issue_family}",
                        f"issue_type: {case.issue_type}",
                        f"final_action: {case.final_action}",
                        f"resolution_summary: {case.resolution_summary}",
                    ]
                )
            )

        audience_policy = ""
        if support_intent:
            audience_policy = (
                "Audience policy: the user is asking how customer support should help a buyer. "
                "Do NOT return merchant integration fixes such as debounce, idempotency, centralise capture logic, or checkout UI changes as the main answer. "
                "Prefer a support-facing next step: verify whether this is a true duplicate or only a temporary hold, check payment history, then advise the report/refund/escalation path."
            )

        prompt = f"""
You are a payment-support copilot.
Use only the query and the retrieved resolved cases.
Return one compact JSON object with exactly these keys:
answer, confidence, likely_issue_family, likely_issue_type, probable_root_cause, internal_resolution_hypothesis

Rules:
- answer must be short, practical, and action-oriented.
- answer should say what to do next, not explain your reasoning.
- do not mention issue_family or issue_type in answer.
- confidence must be one of: low, medium, high.
- do not invent steps unsupported by retrieved cases.
- if the retrieved cases are merchant-technical but the user is asking how to help a customer, answer with a customer-support step, not an engineering remediation.
{audience_policy}

User query:
{ticket.query or ticket.render_transcript()}

Distilled query:
{distilled_query.text}

Retrieved cases:
{chr(10).join(case_blocks)}
"""
        generator = self._get_generator()
        raw = generator.generate(
            prompt,
            **self._generate_kwargs(
                max_new_tokens,
                temperature=self.settings.generation_temperature,
                top_p=self.settings.generation_top_p,
            ),
        ).strip()
        parsed = self._parse_json_result(raw)
        if parsed is not None:
            calibrated = self._calibrate_result(parsed, distilled_query, cases, ticket)
            if support_answer and self._has_internal_fix_cases(cases) and calibrated.confidence == "high":
                calibrated.confidence = "medium"
            return calibrated
        if support_answer:
            top = cases[0]
            return AssistResult(
                answer=support_answer,
                confidence=min(self._evidence_confidence(distilled_query, cases), "medium", key=lambda x: {"low": 0, "medium": 1, "high": 2}[x]),
                likely_issue_family=(top.issue_family or "unknown"),
                likely_issue_type=(top.issue_type or "unknown"),
                probable_root_cause=normalize_text(top.resolution_summary),
                internal_resolution_hypothesis=normalize_text(top.resolution_summary),
            )
        return self._fallback_result(distilled_query, cases, ticket=ticket)
