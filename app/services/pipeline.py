from __future__ import annotations

import math

from app.domain import AssistResult, DistilledQuery, RetrievedCase
from app.schemas import (
    AssistRequest,
    AssistResponse,
    DebugInfoResponse,
    DistilledQueryResponse,
    RetrieveResponse,
    SimilarCaseDebugResponse,
    SimilarCaseLiteResponse,
)
from app.utils.text import truncate_chars


class AssistancePipeline:
    def __init__(
        self,
        settings,
        store,
        embedder,
        reranker,
        distiller,
        summary_generator,
    ):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.distiller = distiller
        self.summary_generator = summary_generator

    def build_retrieval_query(self, ticket: AssistRequest, distilled_text: str) -> str:
        return distilled_text.strip()

    def preload(self, preload_generator: bool = True) -> None:
        _ = self.embedder.embed_query("warmup", instruction=self.settings.query_instruction)
        _ = self.reranker.score("warmup query", ["warmup candidate"])
        if preload_generator:
            self.summary_generator.preload()

    def warmup(self) -> None:
        self.preload(preload_generator=True)
        self.summary_generator.warmup()


    def _serialize_cases_lite(self, cases: list[RetrievedCase]) -> list[SimilarCaseLiteResponse]:
        return [
            SimilarCaseLiteResponse(
                thread_id=case.thread_id,
                similarity_score=round(float(case.similarity_score), 6),
                reranker_score=round(float(case.reranker_score), 6),
            )
            for case in cases
        ]

    def _serialize_cases_debug(self, cases: list[RetrievedCase]) -> list[SimilarCaseDebugResponse]:
        return [
            SimilarCaseDebugResponse(
                thread_id=case.thread_id,
                thread_subject=case.thread_subject,
                issue_family=case.issue_family,
                issue_type=case.issue_type,
                final_action=case.final_action,
                resolution_summary=case.resolution_summary,
                similarity_score=round(float(case.similarity_score), 6),
                reranker_score=round(float(case.reranker_score), 6),
                channel=case.channel or None,
                country=case.country or None,
                currency=case.currency or None,
                payment_method=case.payment_method or None,
            )
            for case in cases
        ]

    def _diversity_key(self, case: RetrievedCase) -> str:
        left = case.issue_type or ""
        right = case.final_action or case.thread_subject or ""
        return f"{left}::{right}".strip(":")

    def _normalize_scores(self, values: list[float]) -> list[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if math.isclose(minimum, maximum):
            return [1.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def _apply_hybrid_ranking(self, candidates: list[RetrievedCase]) -> list[RetrievedCase]:
        if not candidates:
            return candidates
        rerank_norm = self._normalize_scores([c.reranker_score for c in candidates])
        sim_norm = self._normalize_scores([c.similarity_score for c in candidates])
        for case, rerank_score, sim_score in zip(candidates, rerank_norm, sim_norm):
            case.hybrid_score = (
                self.settings.rerank_weight * rerank_score
                + self.settings.similarity_weight * sim_score
            )
        return sorted(
            candidates,
            key=lambda item: (item.hybrid_score, item.reranker_score, item.similarity_score),
            reverse=True,
        )

    def _select_diverse_cases(self, candidates: list[RetrievedCase]) -> list[RetrievedCase]:
        selected: list[RetrievedCase] = []
        seen_keys: set[str] = set()
        for candidate in candidates:
            key = self._diversity_key(candidate)
            if key in seen_keys:
                continue
            selected.append(candidate)
            seen_keys.add(key)
            if len(selected) >= self.settings.top_k_final:
                return selected
        for candidate in candidates:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= self.settings.top_k_final:
                break
        return selected

    def _retrieve_internal(self, ticket: AssistRequest) -> tuple[DistilledQuery, str, list[RetrievedCase]]:
        distilled = self.distiller.distill(ticket)
        retrieval_query = self.build_retrieval_query(ticket, distilled.text)
        query_embedding = self.embedder.embed_query(retrieval_query, instruction=self.settings.query_instruction)
        candidates = self.store.search(query_embedding.tolist(), top_k=self.settings.top_k_retriever)
        if candidates:
            reranker_query = (
                f"{distilled.text}\n\n"
                f"Ticket excerpt:\n{truncate_chars(ticket.render_transcript(), self.settings.reranker_query_chars)}"
            )
            reranker_scores = self.reranker.score(reranker_query, [c.case_text for c in candidates])
            for candidate, score in zip(candidates, reranker_scores):
                candidate.reranker_score = float(score)
            candidates = self._apply_hybrid_ranking(candidates)
        final_cases = self._select_diverse_cases(candidates)
        return distilled, retrieval_query, final_cases

    def retrieve(self, ticket: AssistRequest) -> RetrieveResponse:
        distilled, retrieval_query, final_cases = self._retrieve_internal(ticket)
        return RetrieveResponse(
            distilled_query=DistilledQueryResponse(
                text=distilled.text,
                signals=distilled.signals,
                strategy=distilled.strategy,
            ),
            retrieval_query=retrieval_query,
            similar_cases=self._serialize_cases_debug(final_cases),
        )

    def assist(self, ticket: AssistRequest) -> AssistResponse:
        distilled, retrieval_query, final_cases = self._retrieve_internal(ticket)
        result: AssistResult = self.summary_generator.generate(
            ticket=ticket,
            distilled_query=distilled,
            cases=final_cases,
            max_new_tokens=self.settings.generation_max_new_tokens,
        )
        debug_info = None
        similar_cases_debug = None
        if ticket.debug:
            debug_info = DebugInfoResponse(
                distilled_query=DistilledQueryResponse(
                    text=distilled.text,
                    signals=distilled.signals,
                    strategy=distilled.strategy,
                ),
                retrieval_query=retrieval_query,
                likely_issue_family=result.likely_issue_family,
                likely_issue_type=result.likely_issue_type,
                probable_root_cause=result.probable_root_cause,
                internal_resolution_hypothesis=result.internal_resolution_hypothesis,
            )
            similar_cases_debug = self._serialize_cases_debug(final_cases)
        return AssistResponse(
            answer=result.answer,
            confidence=result.confidence,
            similar_cases=self._serialize_cases_lite(final_cases),
            debug_info=debug_info,
            similar_cases_debug=similar_cases_debug,
        )
