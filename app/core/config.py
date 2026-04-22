from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "support-ai-mvp-v1.8"
    app_env: str = "dev"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    device: str = "cpu"
    torch_dtype: str = "auto"

    preload_models_on_startup: bool = True
    warmup_on_startup: bool = True
    health_status_starting: str = "starting"

    generator_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model_id: str = "Qwen/Qwen3-Reranker-0.6B"

    embedding_batch_size: int = 4
    embedding_max_length: int = 512
    reranker_batch_size: int = 4
    reranker_max_length: int = 768

    query_instruction: str = Field(
        default="Represent this noisy payment support ticket for retrieving similar resolved cases. Focus on the core failure mode, customer-visible symptoms, and likely topic. Ignore superficial metadata unless it changes the root cause."
    )
    doc_instruction: str = Field(
        default="Represent this resolved payment support case for retrieval. Focus on the failure mode, root cause, and final resolution path."
    )
    max_query_turns: int = 4
    max_transcript_chars: int = 4500
    reranker_query_chars: int = 1800
    top_k_retriever: int = 20
    top_k_final: int = 3
    rerank_weight: float = 0.72
    similarity_weight: float = 0.28
    generation_max_new_tokens: int = 220
    generation_temperature: float = 0.0
    generation_top_p: float = 1.0
    low_confidence_overlap_threshold: float = 0.08
    low_confidence_reranker_threshold: float = 1.5
    medium_confidence_reranker_threshold: float = 4.0
    high_confidence_reranker_threshold: float = 6.5
    confidence_agreement_min_cases: int = 2

    postgres_dsn: str = "postgresql://postgres:postgres@postgres:5432/support_ai"
    pgvector_table: str = "support_cases"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
