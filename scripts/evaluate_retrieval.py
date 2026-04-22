from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.core.config import get_settings
from app.services.model_backends import TransformersEmbedder, TransformersReranker


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def dcg(relevances: list[int]) -> float:
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def mrr_for_hits(hits: list[int]) -> float:
    for idx, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / idx
    return 0.0


def dedupe_ranked(ranked: pd.DataFrame, top_k: int) -> pd.DataFrame:
    selected_rows = []
    selected_ids = set()
    seen = set()
    for _, row in ranked.iterrows():
        key = row["issue_type"] or row["final_action"] or row["thread_subject"]
        if key in seen:
            continue
        selected_rows.append(row)
        selected_ids.add(row["thread_id"])
        seen.add(key)
        if len(selected_rows) >= top_k:
            break
    if len(selected_rows) < top_k:
        for _, row in ranked.iterrows():
            if len(selected_rows) >= top_k:
                break
            if row["thread_id"] in selected_ids:
                continue
            selected_rows.append(row)
            selected_ids.add(row["thread_id"])
    return pd.DataFrame(selected_rows)


def evaluate(
    df: pd.DataFrame,
    use_reranker: bool,
    top_k_retriever: int,
    top_k_eval: int,
    embedding_batch_size: int | None = None,
    embedding_max_length: int | None = None,
    reranker_batch_size: int | None = None,
    reranker_max_length: int | None = None,
) -> dict:
    settings = get_settings()
    corpus = df[(df["split"].isin(["train", "val"])) & (df["resolution_status"].str.lower() == "resolved")].reset_index(drop=True)
    queries = df[df["split"] == "test"].reset_index(drop=True)

    embedder = TransformersEmbedder(
        settings.embedding_model_id,
        device=settings.device,
        torch_dtype=settings.torch_dtype,
        batch_size=embedding_batch_size or settings.embedding_batch_size,
        max_length=embedding_max_length or settings.embedding_max_length,
    )
    reranker = (
        TransformersReranker(
            settings.reranker_model_id,
            device=settings.device,
            batch_size=reranker_batch_size or settings.reranker_batch_size,
            max_length=reranker_max_length or settings.reranker_max_length,
        )
        if use_reranker
        else None
    )

    corpus_embeddings = embedder.embed_documents(corpus["case_text"].tolist(), instruction=settings.doc_instruction)
    query_embeddings = embedder.embed_documents(queries["query_text"].tolist(), instruction=settings.query_instruction)
    sims = cosine_matrix(query_embeddings, corpus_embeddings)

    strict_recall_at_3 = []
    strict_recall_at_10 = []
    practical_recall_at_3 = []
    practical_recall_at_10 = []
    strict_mrr = []
    practical_mrr = []
    strict_ndcg = []
    practical_ndcg = []
    evaluated = 0

    for q_idx in tqdm(range(len(queries)), desc="eval"):
        q = queries.iloc[q_idx]
        ranking = np.argsort(-sims[q_idx])[:top_k_retriever]
        ranked = corpus.iloc[ranking].copy().reset_index(drop=True)

        if reranker is not None and len(ranked) > 0:
            scores = reranker.score(str(q["query_text"]), ranked["case_text"].tolist())
            ranked["reranker_score"] = scores
            ranked = ranked.sort_values(["reranker_score", "thread_id"], ascending=[False, True]).reset_index(drop=True)

        ranked = dedupe_ranked(ranked, top_k_eval)
        strict_hits = (ranked["issue_type"] == q["issue_type"]).astype(int).tolist()[:top_k_eval]
        practical_hits = ((ranked["issue_type"] == q["issue_type"]) | (ranked["final_action"] == q["final_action"])).astype(int).tolist()[:top_k_eval]

        if sum(practical_hits) == 0:
            continue

        evaluated += 1
        strict_recall_at_3.append(1.0 if any(strict_hits[:3]) else 0.0)
        strict_recall_at_10.append(1.0 if any(strict_hits[:10]) else 0.0)
        practical_recall_at_3.append(1.0 if any(practical_hits[:3]) else 0.0)
        practical_recall_at_10.append(1.0 if any(practical_hits[:10]) else 0.0)
        strict_mrr.append(mrr_for_hits(strict_hits))
        practical_mrr.append(mrr_for_hits(practical_hits))
        strict_ndcg.append(dcg(strict_hits[:10]) / max(dcg(sorted(strict_hits[:10], reverse=True)), 1e-9))
        practical_ndcg.append(dcg(practical_hits[:10]) / max(dcg(sorted(practical_hits[:10], reverse=True)), 1e-9))

    def avg(values: list[float]) -> float:
        return round(float(np.mean(values)) if values else 0.0, 4)

    return {
        "backend": "transformers",
        "reranker_enabled": use_reranker,
        "queries_total": int(len(queries)),
        "queries_evaluated": int(evaluated),
        "embedding_batch_size": int(embedding_batch_size or settings.embedding_batch_size),
        "embedding_max_length": int(embedding_max_length or settings.embedding_max_length),
        "reranker_batch_size": int(reranker_batch_size or settings.reranker_batch_size),
        "reranker_max_length": int(reranker_max_length or settings.reranker_max_length),
        "strict": {
            "recall_at_3": avg(strict_recall_at_3),
            "recall_at_10": avg(strict_recall_at_10),
            "mrr": avg(strict_mrr),
            "ndcg_at_10": avg(strict_ndcg),
        },
        "practical": {
            "recall_at_3": avg(practical_recall_at_3),
            "recall_at_10": avg(practical_recall_at_10),
            "mrr": avg(practical_mrr),
            "ndcg_at_10": avg(practical_ndcg),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--top-k-retriever", type=int, default=20)
    parser.add_argument("--top-k-eval", type=int, default=10)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-max-length", type=int, default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=None)
    parser.add_argument("--reranker-max-length", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input))
    metrics = evaluate(
        df=df,
        use_reranker=args.use_reranker,
        top_k_retriever=args.top_k_retriever,
        top_k_eval=args.top_k_eval,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
