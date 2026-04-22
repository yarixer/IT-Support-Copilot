from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.core.config import get_settings
from app.repository.pgvector_store import PGVectorStore
from app.services.model_backends import TransformersEmbedder

ALLOWED_SPLITS = {"train", "val", "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    df = pd.read_csv(Path(args.input))
    splits = [split for split in args.splits if split in ALLOWED_SPLITS]
    df = df[df["split"].isin(splits)].copy()
    df = df[df["resolution_status"].str.lower().eq("resolved")].reset_index(drop=True)

    embedder = TransformersEmbedder(
        settings.embedding_model_id,
        device=settings.device,
        torch_dtype=settings.torch_dtype,
        batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
    )
    store = PGVectorStore(settings.postgres_dsn, settings.pgvector_table)
    store.ensure_extension()

    rows = []
    embedding_dim = None
    for start in tqdm(range(0, len(df), args.batch_size), desc="embedding"):
        batch_df = df.iloc[start : start + args.batch_size]
        embeddings = embedder.embed_documents(batch_df["case_text"].tolist(), instruction=settings.doc_instruction)
        if embedding_dim is None:
            embedding_dim = int(embeddings.shape[1])
            if args.recreate:
                store.recreate_table(embedding_dim)
        for row, emb in zip(batch_df.to_dict(orient="records"), embeddings.tolist()):
            rows.append(
                {
                    "thread_id": row["thread_id"],
                    "thread_subject": row["thread_subject"],
                    "case_text": row["case_text"],
                    "issue_family": row.get("issue_family") or "",
                    "issue_type": row.get("issue_type") or "",
                    "final_action": row.get("final_action") or "",
                    "resolution_summary": row.get("resolution_summary") or "",
                    "channel": row.get("channel") or "",
                    "country": row.get("country") or "",
                    "currency": row.get("currency") or "",
                    "payment_method": row.get("payment_method") or "",
                    "metadata": {
                        "split": row.get("split"),
                        "plugin_or_stack": row.get("plugin_or_stack"),
                        "risk_flag": row.get("risk_flag"),
                        "tags": row.get("tags"),
                    },
                    "embedding": emb,
                }
            )

    if rows:
        store.upsert_cases(rows)
    print(f"Indexed {len(rows)} cases into {settings.pgvector_table}")


if __name__ == "__main__":
    main()
