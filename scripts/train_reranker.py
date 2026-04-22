from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


def build_examples(df: pd.DataFrame) -> list[InputExample]:
    examples: list[InputExample] = []
    for row in df.itertuples(index=False):
        examples.append(
            InputExample(
                texts=[str(row.query_text), str(row.candidate_text)],
                label=float(row.label),
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model-id', default='Qwen/Qwen3-Reranker-0.6B')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    model = CrossEncoder(
        args.model_id,
        num_labels=1,
        max_length=args.max_length,
        trust_remote_code=True,
        device=args.device,
    )

    train_examples = build_examples(train_df)
    val_samples = list(zip(val_df['query_text'].astype(str), val_df['candidate_text'].astype(str), val_df['label'].astype(float)))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = max(1, int(len(train_dataloader) * args.epochs * args.warmup_ratio))

    def evaluator(model_obj, output_path=None, epoch=-1, steps=-1):
        preds = model_obj.predict([[q, c] for q, c, _ in val_samples], show_progress_bar=False)
        labels = [label for _, _, label in val_samples]
        paired = list(zip(preds, labels))
        pos = [score for score, label in paired if label >= 0.5]
        neg = [score for score, label in paired if label < 0.5]
        gap = (sum(pos) / max(len(pos), 1)) - (sum(neg) / max(len(neg), 1))
        print({'epoch': epoch, 'steps': steps, 'validation_score_gap': round(float(gap), 4)})
        return float(gap)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        optimizer_params={'lr': args.lr},
        save_best_model=True,
        show_progress_bar=True,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f'Saved fine-tuned reranker to {args.output_dir}')


if __name__ == '__main__':
    main()
