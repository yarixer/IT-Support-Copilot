from __future__ import annotations

import re

from app.domain import DistilledQuery
from app.schemas import AssistRequest
from app.utils.text import join_lines, normalize_text, strip_boilerplate, truncate_chars


MONEY_RE = re.compile(r"\b\d+(?:[.,]\d{1,2})?\b")
SYMPTOM_RE = re.compile(
    r"\b(fail(?:ed|s|ing)?|error|invalid|declin(?:ed|e)|missing|stuck|pending|timeout|timed out|cannot|can't|unable|blocked|not delivered|not received|upload|refund|chargeback|dispute|subscription|hold|payout|capture|webhook|authorization|auth|evidence|reversal|duplicate|billing|vault|token|invoice|double|twice)\b",
    re.I,
)
REFERENCE_RE = re.compile(r"\s*[\[(][A-Z]{2,}-[A-Z0-9]{4,}[\])]\s*$")


class HeuristicQueryDistiller:
    def __init__(self, max_chars: int = 850):
        self.max_chars = max_chars

    def _line_priority(self, line: str) -> tuple[int, int, int]:
        low = line.lower()
        customer_bonus = 0 if low.startswith("[customer]") or low.startswith("[user]") else 1
        symptom_bonus = 0 if SYMPTOM_RE.search(low) else 1
        short_bonus = 0 if len(low) < 240 else 1
        return (customer_bonus, symptom_bonus, short_bonus)

    def _clean_subject(self, subject: str) -> str:
        value = normalize_text(subject)
        value = REFERENCE_RE.sub("", value)
        return normalize_text(value)

    def distill(self, ticket: AssistRequest) -> DistilledQuery:
        subject = self._clean_subject(ticket.effective_subject() or "")
        transcript = ticket.render_transcript()
        cleaned_lines = [strip_boilerplate(line) for line in transcript.splitlines() if line.strip()]

        if len(cleaned_lines) <= 1 and ticket.query:
            cleaned_lines = [ticket.query.strip()]

        ranked_lines = sorted(cleaned_lines, key=self._line_priority)
        chosen: list[str] = []
        seen = set()
        for line in ranked_lines:
            normalized = normalize_text(line)
            if not normalized or normalized in seen:
                continue
            chosen.append(normalized)
            seen.add(normalized)
            if len(chosen) >= 3:
                break

        if not chosen:
            chosen = [truncate_chars(normalize_text(transcript), self.max_chars)]

        issue_text = truncate_chars(join_lines(chosen), self.max_chars)
        signals = {
            "channel": ticket.channel,
            "country": ticket.country,
            "currency": ticket.currency,
            "amount": ticket.amount,
            "payment_method": ticket.payment_method,
            "plugin_or_stack": ticket.plugin_or_stack,
            "numbers_detected": ",".join(MONEY_RE.findall(issue_text)[:3]) or None,
        }

        context_parts = [
            f"payment_method={ticket.payment_method}" if ticket.payment_method else "",
            f"plugin_or_stack={ticket.plugin_or_stack}" if ticket.plugin_or_stack else "",
        ]

        distilled = join_lines(
            [
                f"Subject: {subject}" if subject else "",
                "Likely issue:",
                issue_text,
                f"Light context: {', '.join([part for part in context_parts if part])}" if any(context_parts) else "",
            ]
        ).strip()
        return DistilledQuery(text=distilled, signals=signals, strategy="heuristic")
