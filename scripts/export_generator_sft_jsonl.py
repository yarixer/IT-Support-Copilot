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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input))
    if args.split:
        df = df[df["split"] == args.split].copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            prompt = (
                f"{row['task_instruction']}\n\n"
                f"Visible conversation:\n{row['context_text']}\n\n"
                f"Latest customer message:\n{row['latest_customer_message']}\n\n"
                "Return only JSON."
            )
            record = {
                "id": row["sample_id"],
                "messages": [
                    {"role": "system", "content": "You write internal payment-support assist notes as strict JSON."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(row["target_assist_note_json"])},
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} records -> {out_path}")


if __name__ == "__main__":
    main()
