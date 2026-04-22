from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config import get_settings
from app.repository.pgvector_store import PGVectorStore
from app.services.model_backends import TransformersEmbedder, TransformersGenerator, TransformersReranker
from app.services.pipeline import AssistancePipeline
from app.services.query_distiller import HeuristicQueryDistiller
from app.services.summary_generator import GroundedSummaryGenerator

logger = logging.getLogger(__name__)
_RUNTIME_STATUS = "starting"
_RUNTIME_ERROR: str | None = None


def get_runtime_status() -> dict[str, str | None]:
    return {"status": _RUNTIME_STATUS, "error": _RUNTIME_ERROR}


def _set_runtime_status(status: str, error: str | None = None) -> None:
    global _RUNTIME_STATUS, _RUNTIME_ERROR
    _RUNTIME_STATUS = status
    _RUNTIME_ERROR = error


@lru_cache(maxsize=1)
def get_runtime() -> AssistancePipeline:
    settings = get_settings()
    store = PGVectorStore(settings.postgres_dsn, settings.pgvector_table)

    embedder = TransformersEmbedder(
        settings.embedding_model_id,
        device=settings.device,
        torch_dtype=settings.torch_dtype,
        batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
    )
    reranker = TransformersReranker(
        settings.reranker_model_id,
        device=settings.device,
        batch_size=settings.reranker_batch_size,
        max_length=settings.reranker_max_length,
    )

    def build_generator() -> TransformersGenerator:
        return TransformersGenerator(
            settings.generator_model_id,
            device=settings.device,
            torch_dtype=settings.torch_dtype,
        )

    distiller = HeuristicQueryDistiller(max_chars=850)
    summary_generator = GroundedSummaryGenerator(
        generator_factory=build_generator,
        settings=settings,
    )

    return AssistancePipeline(
        settings=settings,
        store=store,
        embedder=embedder,
        reranker=reranker,
        distiller=distiller,
        summary_generator=summary_generator,
    )


def preload_runtime() -> AssistancePipeline:
    settings = get_settings()
    try:
        _set_runtime_status(settings.health_status_starting)
        runtime = get_runtime()
        if settings.preload_models_on_startup or settings.warmup_on_startup:
            runtime.preload(preload_generator=settings.preload_models_on_startup)
        if settings.warmup_on_startup:
            runtime.warmup()
        _set_runtime_status("ok")
        logger.info("Runtime preload/warmup completed.")
        return runtime
    except Exception as exc:
        logger.exception("Runtime preload/warmup failed")
        _set_runtime_status("error", str(exc))
        raise
