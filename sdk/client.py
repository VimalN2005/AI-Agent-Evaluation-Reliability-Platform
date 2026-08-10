"""
Minimal evaluation SDK: import this from your own agent code to log traces
directly into the platform's database (bypassing the HTTP/mock runners in the UI).

Usage:
    from sdk.client import EvalLogger

    logger = EvalLogger(dataset_id="...", agent_name="my-agent-v2", model_name="claude-sonnet-5")
    run_id = logger.start_run()
    for item in logger.items():
        answer, retrieved_docs, tool_calls, usage = my_agent.run(item["question"])
        logger.log_trace(item, answer, retrieved_docs, tool_calls,
                          prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)
    logger.finish()
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import db
from engine import evaluate_trace
from evaluators.cost_latency import estimate_cost


class EvalLogger:
    def __init__(self, dataset_id, agent_name, model_name="", notes=""):
        db.init_db()
        self.dataset_id = dataset_id
        self.agent_name = agent_name
        self.model_name = model_name
        self.notes = notes
        self.run_id = None
        self._items = db.get_dataset_items(dataset_id)
        self._items_by_id = {i["id"]: i for i in self._items}

    def items(self):
        """Iterate the dataset items you should run your agent against."""
        return self._items

    def start_run(self):
        self.run_id = db.create_run(self.dataset_id, self.agent_name, self.model_name, self.notes)
        return self.run_id

    def log_trace(self, item, answer, retrieved_doc_ids=None, tool_calls=None,
                  latency_ms=0, prompt_tokens=0, completion_tokens=0):
        if self.run_id is None:
            self.start_run()
        cost = estimate_cost(self.model_name, prompt_tokens, completion_tokens)
        trace_id = db.add_trace(
            self.run_id, item["id"], item["question"], answer,
            retrieved_doc_ids or [], tool_calls or [], latency_ms,
            prompt_tokens, completion_tokens, cost,
        )
        trace = {
            "answer": answer, "retrieved_doc_ids": retrieved_doc_ids or [],
            "tool_calls": tool_calls or [],
        }
        evald = evaluate_trace(item, trace)
        db.add_evaluation(
            trace_id, self.run_id, evald["correctness"], evald["faithfulness"],
            evald["rag_precision"], evald["rag_recall"], evald["tool_call_accuracy"],
            evald["safety_flag"], evald["safety_reason"], evald["overall_score"], evald["details"],
        )
        return trace_id

    def finish(self):
        return self.run_id
