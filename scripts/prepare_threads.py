from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_RE = re.compile(r"\s*[\[(][A-Z]{2,}-[A-Z0-9]{4,}[\])]\s*$")


def safe(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def clean_subject_for_query(subject: str) -> str:
    value = safe(subject)
    value = REFERENCE_RE.sub("", value)
    return value.strip()


def render_turns(group: pd.DataFrame) -> str:
    lines = []
    for _, row in group.iterrows():
        role = safe(row.get("speaker_role") or row.get("role") or "unknown").lower()
        message = safe(row.get("message_text") or row.get("message") or row.get("body"))
        if message:
            lines.append(f"[{role}] {message}")
    return "\n".join(lines)


def build_case_text(row: pd.Series) -> str:
    metadata = ", ".join(
        f"{k}={row[k]}"
        for k in ["channel", "country", "currency", "payment_method", "plugin_or_stack"]
        if safe(row.get(k))
    )
    parts = [
        f"Subject: {safe(row.get('thread_subject'))}",
        f"Operational context: {metadata}" if metadata else "",
        "Conversation:",
        safe(row.get("raw_transcript")),
        f"Resolution summary: {safe(row.get('resolution_summary'))}",
        f"Final action: {safe(row.get('final_action'))}",
    ]
    return "\n".join([part for part in parts if part])


def build_query_text(row: pd.Series) -> str:
    parts = [
        f"Subject: {clean_subject_for_query(row.get('thread_subject'))}",
        "Visible early conversation:",
        safe(row.get("early_transcript")),
    ]
    return "\n".join([part for part in parts if part])


def aggregate_threads(df: pd.DataFrame, max_query_turns: int) -> pd.DataFrame:
    rows = []
    for thread_id, group in df.groupby("thread_id", sort=False):
        group = group.sort_values("turn_index").reset_index(drop=True)
        first = group.iloc[0]
        early = group.iloc[:max_query_turns]
        row = {
            "thread_id": thread_id,
            "thread_subject": safe(first.get("thread_subject")),
            "message_count": int(len(group)),
            "turn_count": int(first.get("turn_count", len(group))) if not pd.isna(first.get("turn_count", np.nan)) else int(len(group)),
            "channel": safe(first.get("channel")),
            "country": safe(first.get("country")),
            "currency": safe(first.get("currency")),
            "amount": float(first.get("amount")) if not pd.isna(first.get("amount", np.nan)) else np.nan,
            "merchant_vertical": safe(first.get("merchant_vertical")),
            "merchant_name": safe(first.get("merchant_name")),
            "plugin_or_stack": safe(first.get("plugin_or_stack")),
            "payment_method": safe(first.get("payment_method")),
            "risk_flag": safe(first.get("risk_flag")),
            "issue_family": safe(first.get("issue_family")),
            "issue_type": safe(first.get("issue_type")),
            "micro_topic": safe(first.get("micro_topic")),
            "root_cause": safe(first.get("root_cause")),
            "resolution_status": safe(first.get("resolution_status")),
            "final_action": safe(first.get("final_action")),
            "resolution_summary": safe(first.get("resolution_summary")),
            "tags": safe(first.get("tags")),
            "raw_transcript": render_turns(group),
            "early_transcript": render_turns(early),
        }
        rows.append(row)

    threads = pd.DataFrame(rows)
    threads["case_text"] = threads.apply(build_case_text, axis=1)
    threads["query_text"] = threads.apply(build_query_text, axis=1)
    threads["strict_label"] = threads["issue_type"]
    threads["practical_label"] = threads["issue_type"] + " || " + threads["final_action"]
    return threads


def add_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    ids = df["thread_id"].to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    train_cut = int(n * 0.8)
    val_cut = int(n * 0.9)
    train_ids = set(ids[:train_cut])
    val_ids = set(ids[train_cut:val_cut])

    def resolve_split(thread_id: str) -> str:
        if thread_id in train_ids:
            return "train"
        if thread_id in val_ids:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["thread_id"].map(resolve_split)
    return df


def add_splits_from_source(df: pd.DataFrame, split_source_path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(split_source_path)
    id_col = "thread_id" if "thread_id" in split_df.columns else "query_id"
    if "split" not in split_df.columns:
        raise ValueError("Split source must contain a 'split' column.")
    mapping = split_df[[id_col, "split"]].drop_duplicates().rename(columns={id_col: "thread_id"})
    merged = df.merge(mapping, on="thread_id", how="left")
    if merged["split"].isna().any():
        missing = int(merged["split"].isna().sum())
        raise ValueError(f"Split source did not provide splits for {missing} thread ids.")
    merged["split"] = merged["split"].astype(str)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-query-turns", type=int, default=3)
    parser.add_argument("--split-source", default=None, help="Optional CSV with thread_id/query_id and split columns.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".csv":
        raise ValueError("prepare_threads.py writes CSV only. Use an output path ending with .csv")

    df = pd.read_csv(input_path)
    threads = aggregate_threads(df, max_query_turns=args.max_query_turns)
    if args.split_source:
        threads = add_splits_from_source(threads, Path(args.split_source))
    else:
        threads = add_splits(threads)
    threads.to_csv(output_path, index=False)
    print(f"Prepared {len(threads)} threads -> {output_path}")


if __name__ == "__main__":
    main()
