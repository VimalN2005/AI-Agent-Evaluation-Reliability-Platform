# 🧪 AI Agent Evaluation & Reliability Platform

A production-oriented platform to **evaluate, monitor, compare, and improve AI agents and GenAI applications** — correctness, faithfulness/hallucination, RAG retrieval quality, tool-call accuracy, latency, token cost, safety, and regression between versions.

This implementation is scoped to run as a **single Streamlit app + SQLite database**, so it deploys in minutes on Streamlit Community Cloud with zero external infrastructure, while keeping the evaluation logic modular enough to swap in Postgres/pgvector/Redis for a heavier production deployment later.

## Features

- **Datasets**: load the bundled sample benchmark or upload your own JSON eval set (question, expected answer, context, relevant doc ids, expected tool calls).
- **Run evaluation** against three agent modes:
  - Mock agent (no API key needed — great for demoing the dashboard)
  - Your own HTTP endpoint (`POST {question, context}` → `{answer, retrieved_doc_ids, tool_calls, ...}`)
  - A minimal built-in Anthropic-model agent (bring your own API key)
- **Evaluators** (`evaluators/`): correctness, faithfulness/hallucination, RAG precision/recall, tool-call accuracy, cost, latency, rule-based safety — each usable standalone or via LLM-as-judge.
- **Results dashboard**: per-item scorecards, safety flags, trace inspection.
- **Compare Runs**: side-by-side metric deltas between two runs with automatic regression flags — the core "did my new prompt/model make things worse?" workflow.
- **Reports**: exportable Markdown evaluation report per run.
- **SDK** (`sdk/client.py`): log traces directly from your own agent's Python code instead of going through the UI.

## Architecture

```
User / Developer
      |
   AI Agent  (mock / HTTP endpoint / Anthropic model / your own via sdk/client.py)
      |
Evaluation SDK / API  (engine.py)
      |
Trace Collection  (db.py -> SQLite: traces table)
      |
Evaluation Engine  (evaluators/*.py)
      |-- Quality:  correctness, faithfulness
      |-- RAG:      retrieval precision/recall
      |-- Reliability: tool-call accuracy, safety flags
      |-- Cost:     token usage, $ estimate
      |-- Latency:  ms per call
      |
Evaluation Database  (SQLite: evaluations table)
      |
Analytics Dashboard  (app.py — Streamlit)
      |
Reports & Alerts  (Markdown export, regression detection in Compare Runs)
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the app: **Datasets → Load sample dataset**, then **Run Evaluation → Run mock agent**, then check **Results** and **Compare Runs**.

## Project structure

```
app.py                 # Streamlit dashboard (all tabs/UI)
engine.py               # Evaluation orchestration + agent runners
db.py                    # SQLite schema & data access
evaluators/
  correctness.py         # Rule-based + LLM-judge answer correctness
  faithfulness.py         # Hallucination / groundedness check
  rag_quality.py          # Retrieval precision/recall
  tool_calls.py            # Tool-call accuracy
  cost_latency.py           # Token cost estimation + rule-based safety
sdk/client.py              # Python SDK for logging traces from your own agent
data/sample_dataset.json    # Bundled benchmark dataset
```

## Deploying

### 1. Push to GitHub

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git add -A
git commit -m "Initial commit: AI Agent Evaluation & Reliability Platform"
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **New app** → select your repo/branch → set **Main file path** to `app.py`.
3. Deploy. First boot installs `requirements.txt` automatically.
4. (Optional) In **App settings → Secrets**, you can pre-fill an Anthropic key instead of typing it into the sidebar each time:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   and read it in `app.py` via `st.secrets.get("ANTHROPIC_API_KEY")` as a default for the sidebar field.

> Note: Streamlit Community Cloud's filesystem is ephemeral across redeploys/restarts, so the SQLite DB (`data/eval_platform.db`) will reset when the app sleeps/redeploys. For persistent production use, point `db.py` at a hosted Postgres instance (e.g. Supabase/Neon) instead of SQLite — the schema and query layer are isolated in `db.py` specifically to make that swap contained.

## Extending toward the full production architecture

This MVP intentionally keeps the "How It Works" pipeline from the spec but simplifies the infra:

| Spec component        | This implementation      | Production upgrade path |
|---|---|---|
| PostgreSQL + pgvector  | SQLite (`db.py`)          | Swap connection layer in `db.py` for Postgres; add pgvector for embedding-based RAG relevance scoring |
| Redis                  | N/A (single-user dashboard) | Add for caching judge-LLM calls / rate limiting on the HTTP agent runner |
| React/Next.js           | Streamlit                | Same evaluation engine (`engine.py`, `evaluators/`) can back a FastAPI service consumed by a separate frontend |
| OpenTelemetry            | `raw_trace` JSON column on traces | Replace manual trace dict with OTel spans; export to the `traces` table via an OTel collector |
| LangGraph agent workflows | `run_http_agent` / `sdk/client.py` | Instrument a LangGraph agent to call `EvalLogger.log_trace()` per node/tool call |
