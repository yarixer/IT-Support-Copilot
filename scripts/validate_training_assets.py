from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

import pandas as pd


def describe_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "file": path.name,
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "split_counts": df["split"].value_counts(dropna=False).to_dict() if "split" in df.columns else {},
        "null_counts": {k: int(v) for k, v in df.isna().sum().to_dict().items() if int(v) > 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--reranker", required=True)
    parser.add_argument("--generator", required=True)
    args = parser.parse_args()

    report = {
        "retriever": describe_csv(Path(args.retriever)),
        "reranker": describe_csv(Path(args.reranker)),
        "generator": describe_csv(Path(args.generator)),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
