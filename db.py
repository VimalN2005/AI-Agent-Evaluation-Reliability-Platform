"""
Database layer for the AI Agent Evaluation & Reliability Platform.
Uses SQLite for zero-config local + Streamlit Cloud deployment.
Swap DB_PATH / connection logic for Postgres in production if needed.
"""
import sqlite3
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "eval_platform.db"
DB_PATH.parent.mkdir(exist_ok=True)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_items (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                question TEXT NOT NULL,
                expected_answer TEXT,
                context TEXT,             -- JSON list of reference/context strings
                relevant_doc_ids TEXT,    -- JSON list of doc ids considered relevant (for RAG eval)
                expected_tool_calls TEXT, -- JSON list of tool names expected to be called
                tags TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                model_name TEXT,
                notes TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                question TEXT,
                answer TEXT,
                retrieved_doc_ids TEXT,   -- JSON list
                tool_calls TEXT,          -- JSON list of tool names actually called
                latency_ms REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                cost_usd REAL,
                raw_trace TEXT,           -- JSON blob for anything extra
                created_at REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                correctness REAL,
                faithfulness REAL,
                rag_precision REAL,
                rag_recall REAL,
                tool_call_accuracy REAL,
                safety_flag INTEGER,
                safety_reason TEXT,
                overall_score REAL,
                details TEXT,             -- JSON blob with explanations
                created_at REAL NOT NULL,
                FOREIGN KEY (trace_id) REFERENCES traces(id) ON DELETE CASCADE,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            );
            """
        )


def new_id():
    return str(uuid.uuid4())


def now():
    return time.time()


# ---------- Datasets ----------

def create_dataset(name):
    with get_conn() as conn:
        did = new_id()
        conn.execute(
            "INSERT INTO datasets (id, name, created_at) VALUES (?, ?, ?)",
            (did, name, now()),
        )
        return did


def add_dataset_item(dataset_id, question, expected_answer="", context=None,
                      relevant_doc_ids=None, expected_tool_calls=None, tags=None):
    with get_conn() as conn:
        iid = new_id()
        conn.execute(
            """INSERT INTO dataset_items
               (id, dataset_id, question, expected_answer, context, relevant_doc_ids, expected_tool_calls, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                iid, dataset_id, question, expected_answer,
                json.dumps(context or []),
                json.dumps(relevant_doc_ids or []),
                json.dumps(expected_tool_calls or []),
                json.dumps(tags or []),
            ),
        )
        return iid


def list_datasets():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")]


def get_dataset_items(dataset_id):
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM dataset_items WHERE dataset_id = ?", (dataset_id,)
        )]
        for r in rows:
            r["context"] = json.loads(r["context"] or "[]")
            r["relevant_doc_ids"] = json.loads(r["relevant_doc_ids"] or "[]")
            r["expected_tool_calls"] = json.loads(r["expected_tool_calls"] or "[]")
            r["tags"] = json.loads(r["tags"] or "[]")
        return rows


# ---------- Runs / Traces / Evaluations ----------

def create_run(dataset_id, agent_name, model_name="", notes=""):
    with get_conn() as conn:
        rid = new_id()
        conn.execute(
            "INSERT INTO runs (id, dataset_id, agent_name, model_name, notes, created_at) VALUES (?,?,?,?,?,?)",
            (rid, dataset_id, agent_name, model_name, notes, now()),
        )
        return rid


def add_trace(run_id, item_id, question, answer, retrieved_doc_ids=None, tool_calls=None,
              latency_ms=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0, raw_trace=None):
    with get_conn() as conn:
        tid = new_id()
        conn.execute(
            """INSERT INTO traces
               (id, run_id, item_id, question, answer, retrieved_doc_ids, tool_calls,
                latency_ms, prompt_tokens, completion_tokens, cost_usd, raw_trace, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                tid, run_id, item_id, question, answer,
                json.dumps(retrieved_doc_ids or []),
                json.dumps(tool_calls or []),
                latency_ms, prompt_tokens, completion_tokens, cost_usd,
                json.dumps(raw_trace or {}), now(),
            ),
        )
        return tid


def add_evaluation(trace_id, run_id, correctness, faithfulness, rag_precision, rag_recall,
                    tool_call_accuracy, safety_flag, safety_reason, overall_score, details=None):
    with get_conn() as conn:
        eid = new_id()
        conn.execute(
            """INSERT INTO evaluations
               (id, trace_id, run_id, correctness, faithfulness, rag_precision, rag_recall,
                tool_call_accuracy, safety_flag, safety_reason, overall_score, details, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eid, trace_id, run_id, correctness, faithfulness, rag_precision, rag_recall,
                tool_call_accuracy, int(safety_flag), safety_reason, overall_score,
                json.dumps(details or {}), now(),
            ),
        )
        return eid


def list_runs():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT runs.*, datasets.name as dataset_name FROM runs
               JOIN datasets ON runs.dataset_id = datasets.id
               ORDER BY runs.created_at DESC"""
        )]


def get_run_results(run_id):
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            """SELECT traces.*, evaluations.correctness, evaluations.faithfulness,
                      evaluations.rag_precision, evaluations.rag_recall,
                      evaluations.tool_call_accuracy, evaluations.safety_flag,
                      evaluations.safety_reason, evaluations.overall_score, evaluations.details
               FROM traces
               LEFT JOIN evaluations ON evaluations.trace_id = traces.id
               WHERE traces.run_id = ?
               ORDER BY traces.created_at ASC""",
            (run_id,),
        )]
        for r in rows:
            r["retrieved_doc_ids"] = json.loads(r["retrieved_doc_ids"] or "[]")
            r["tool_calls"] = json.loads(r["tool_calls"] or "[]")
            r["raw_trace"] = json.loads(r["raw_trace"] or "{}")
            r["details"] = json.loads(r["details"] or "{}") if r.get("details") else {}
        return rows


def delete_run(run_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
