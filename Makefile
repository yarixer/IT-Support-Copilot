PYTHON ?= python

prepare:
	$(PYTHON) scripts/prepare_threads.py --input data/raw/paypal_support_threads_5000_diverse_v3.csv --output data/processed/threads.csv --split-source data/training/paypal_retriever_dataset.csv

index:
	$(PYTHON) scripts/build_index.py --input data/processed/threads.csv --recreate

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

eval:
	$(PYTHON) scripts/evaluate_retrieval.py --input data/processed/threads.csv --output reports/eval_metrics_retriever.json

eval-rerank:
	$(PYTHON) scripts/evaluate_retrieval.py --input data/processed/threads.csv --use-reranker --output reports/eval_metrics_rerank.json

validate-assets:
	$(PYTHON) scripts/validate_training_assets.py --retriever data/training/paypal_retriever_dataset.csv --reranker data/training/paypal_reranker_dataset.csv --generator data/training/paypal_generator_assist_dataset_v3_no_leakage.csv

train-reranker:
	$(PYTHON) scripts/train_reranker.py --input data/training/paypal_reranker_dataset.csv --output-dir artifacts/reranker-v1.4


preload-models:
	python scripts/preload_models.py
