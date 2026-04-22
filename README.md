# AI Support Assistant for Payment Support

An applied AI backend for payment-support workflows. Provides a brief response based on resolved tickets, plus the three most similar tickets(IDs) for reference. Designed for employees, not for customers.

## Stack
- PyTorch
- FastAPI
- PostgreSQL + pgvector
- Qwen embedding / reranker / generator models
- Docker Compose for PostgreSQL

## Main API
### `POST /assist`
Input:
```json
{
  "query": "A customer says the payment amount was charged twice. What should I do?",
  "debug": false
}
```

Output:
```json
{
  "answer": "First, check the payment history and confirm whether this is truly two completed charges or one completed payment plus a temporary authorization hold.",
  "confidence": "medium",
  "similar_cases": [
    {"thread_id": "PPD-03256", "similarity_score": 0.73, "reranker_score": 4.56},
    {"thread_id": "PPD-03730", "similarity_score": 0.73, "reranker_score": 4.00},
    {"thread_id": "PPD-03015", "similarity_score": 0.72, "reranker_score": 4.56}
  ]
}
```

Set `debug=true` to include internal diagnostics such as the distilled query, retrieval query, issue hints, and detailed similar-case metadata.

## Repository layout
- `app/` — FastAPI app, runtime wiring, retrieval pipeline, generator policy
- `scripts/prepare_threads.py` — builds `data/processed/threads.csv`
- `scripts/build_index.py` — builds the pgvector index from `threads.csv`
- `scripts/evaluate_retrieval.py` — offline retrieval evaluation
- `scripts/preload_models.py` — downloads and warms up models before serving traffic
- `data/raw/` — source datasets
- `data/processed/threads.csv` — thread-level runtime/eval dataset
- `data/training/` — retriever, reranker, and generator training assets

## Quick start
```bash
unzip support_ai_mvp_v1_9.zip
cd support_ai_mvp_v1_9

docker compose up -d postgres

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Recommended `.env` values:
```env
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/support_ai
DEVICE=cuda
TORCH_DTYPE=float16
GENERATOR_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL_ID=Qwen/Qwen3-Reranker-0.6B
EMBEDDING_BATCH_SIZE=4
EMBEDDING_MAX_LENGTH=512
RERANKER_BATCH_SIZE=4
RERANKER_MAX_LENGTH=768
PRELOAD_MODELS_ON_STARTUP=true
WARMUP_ON_STARTUP=true
```

Build the runtime dataset and index:
```bash
python scripts/prepare_threads.py   --input data/raw/paypal_support_threads_5000_diverse_v3.csv   --output data/processed/threads.csv   --max-query-turns 3   --split-source data/training/paypal_retriever_dataset.csv

python scripts/build_index.py   --input data/processed/threads.csv   --recreate
```

Preload models and start the API:
```bash
python scripts/preload_models.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
```bash
curl http://localhost:8000/health
```

Example request:
```bash
curl -X POST http://localhost:8000/assist   -H "Content-Type: application/json"   -d '{
    "query": "A customer says the payment amount was charged twice. What should I do?",
    "debug": true
  }'
```

## Notes
- The datasets included here are synthetic.
- Evaluation metrics on synthetic datasets can be optimistic and should be interpreted carefully.
- The project is positioned as an MVP / portfolio project, not as a production certification.

## TODO
- Replace the synthetic dataset with more realistic support data.
- Strengthen retrieval evaluation with harder and more realistic benchmarks.
- Fine-tune reranking and answer generation for more stable support responses.