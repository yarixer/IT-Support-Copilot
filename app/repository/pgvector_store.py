from __future__ import annotations

import json
import logging
from typing import Iterable

import psycopg
from psycopg.rows import dict_row

from app.domain import RetrievedCase

logger = logging.getLogger(__name__)


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"


class PGVectorStore:
    def __init__(self, dsn: str, table_name: str):
        self.dsn = dsn
        self.table_name = table_name

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def ensure_extension(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

    def recreate_table(self, embedding_dim: int) -> None:
        logger.info("Recreating pgvector table %s with dim=%s", self.table_name, embedding_dim)
        ddl = f"""
        DROP TABLE IF EXISTS {self.table_name};
        CREATE TABLE {self.table_name} (
            thread_id TEXT PRIMARY KEY,
            thread_subject TEXT NOT NULL,
            case_text TEXT NOT NULL,
            issue_family TEXT,
            issue_type TEXT,
            final_action TEXT,
            resolution_summary TEXT,
            channel TEXT,
            country TEXT,
            currency TEXT,
            payment_method TEXT,
            metadata JSONB,
            embedding VECTOR({embedding_dim}) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw
        ON {self.table_name}
        USING hnsw (embedding vector_cosine_ops);
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()

    def upsert_cases(self, rows: Iterable[dict]) -> None:
        sql = f"""
        INSERT INTO {self.table_name} (
            thread_id, thread_subject, case_text, issue_family, issue_type,
            final_action, resolution_summary, channel, country, currency,
            payment_method, metadata, embedding
        ) VALUES (
            %(thread_id)s, %(thread_subject)s, %(case_text)s, %(issue_family)s, %(issue_type)s,
            %(final_action)s, %(resolution_summary)s, %(channel)s, %(country)s, %(currency)s,
            %(payment_method)s, %(metadata)s::jsonb, %(embedding)s::vector
        )
        ON CONFLICT (thread_id) DO UPDATE SET
            thread_subject = EXCLUDED.thread_subject,
            case_text = EXCLUDED.case_text,
            issue_family = EXCLUDED.issue_family,
            issue_type = EXCLUDED.issue_type,
            final_action = EXCLUDED.final_action,
            resolution_summary = EXCLUDED.resolution_summary,
            channel = EXCLUDED.channel,
            country = EXCLUDED.country,
            currency = EXCLUDED.currency,
            payment_method = EXCLUDED.payment_method,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding;
        """
        payload = []
        for row in rows:
            row = dict(row)
            row["metadata"] = json.dumps(row.get("metadata", {}), ensure_ascii=False)
            row["embedding"] = vector_literal(row["embedding"])
            payload.append(row)

        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(sql, payload)
            conn.commit()

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[RetrievedCase]:
        sql = f"""
        SELECT
            thread_id,
            thread_subject,
            case_text,
            issue_family,
            issue_type,
            final_action,
            resolution_summary,
            channel,
            country,
            currency,
            payment_method,
            metadata,
            1 - (embedding <=> %(embedding)s::vector) AS similarity_score
        FROM {self.table_name}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(top_k)s;
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, {"embedding": vector_literal(query_embedding), "top_k": top_k})
            rows = cur.fetchall()

        results: list[RetrievedCase] = []
        for row in rows:
            metadata = row.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            results.append(
                RetrievedCase(
                    thread_id=row["thread_id"],
                    thread_subject=row["thread_subject"],
                    case_text=row["case_text"],
                    issue_family=row.get("issue_family") or "",
                    issue_type=row.get("issue_type") or "",
                    final_action=row.get("final_action") or "",
                    resolution_summary=row.get("resolution_summary") or "",
                    channel=row.get("channel") or "",
                    country=row.get("country") or "",
                    currency=row.get("currency") or "",
                    payment_method=row.get("payment_method") or "",
                    metadata=metadata,
                    similarity_score=float(row.get("similarity_score") or 0.0),
                )
            )
        return results
