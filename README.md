# Portfolio Chatbot Evaluation (Ragas)

This branch (`ragas`) contains tools to evaluate the RAG pipeline using the [Ragas](https://docs.ragas.io/) framework.

## Setup

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    pip install -r backend/requirements.txt
    ```
3.  Set up your `.env` file (see `.env.example`). You now only need the `GROQ_API_KEY`. Mistral is no longer required as embeddings are now local.
4.  **Re-ingest data** (Required after switching embeddings):
    ```bash
    python backend/ingest.py
    ```

## Running Evaluation

### 1. Evaluate with Predefined Test Data
Run the existing evaluation script:
```bash
python evaluate_ragas.py
```
This will:
- Run 4 test questions through the RAG pipeline.
- Evaluate the responses using Ragas metrics (faithfulness, relevancy, precision, recall).
- Save a report to `ragas_report.csv`.

### 2. Generate New Test Data
To automatically generate a larger test set from your documents in `data/`:
```bash
python generate_testset.py
```
This will create `generated_testset.csv` which you can then use for more comprehensive evaluations.

## Files Updated/Added
- `backend/evaluate_ragas.py`: Refactored for modularity and updated metrics.
- `backend/generate_testset.py`: Added script for automated test data generation.
- `backend/requirements.txt`: Added `tqdm`.
- `backend/.env.example`: Added template for environment variables.
