from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.runtime import get_runtime, get_runtime_status, preload_runtime
from app.schemas import AssistRequest, AssistResponse, RetrieveResponse

setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_runtime()
    yield


app = FastAPI(title=settings.app_name, version="1.9.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | None]:
    runtime_status = get_runtime_status()
    return {
        "status": runtime_status["status"],
        "error": runtime_status["error"],
        "backend": "transformers",
        "embedder": settings.embedding_model_id,
        "reranker": settings.reranker_model_id,
        "generator": settings.generator_model_id,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: AssistRequest) -> RetrieveResponse:
    pipeline = get_runtime()
    return pipeline.retrieve(payload)


@app.post("/assist", response_model=AssistResponse)
def assist(payload: AssistRequest) -> AssistResponse:
    pipeline = get_runtime()
    return pipeline.assist(payload)
