"""
Evaluation engine: given a dataset item + the agent's trace for it, runs every
applicable evaluator and returns a unified scorecard. Also holds the agent
runner abstraction (mock agent, or a real HTTP/Anthropic-backed agent).
"""
import time
import json
import requests

from evaluators.correctness import rule_based_correctness, llm_judge_correctness
from evaluators.faithfulness import rule_based_faithfulness, llm_judge_faithfulness
from evaluators.rag_quality import rag_precision_recall
from evaluators.tool_calls import tool_call_accuracy
from evaluators.cost_latency import estimate_cost, rule_based_safety


def evaluate_trace(item, trace, use_llm_judge=False, judge_client=None, judge_model=None):
    """item: dataset item dict. trace: dict with answer, retrieved_doc_ids, tool_calls."""
    details = {}

    if use_llm_judge and judge_client is not None:
        corr = llm_judge_correctness(judge_client, judge_model, item["question"], trace["answer"], item["expected_answer"])
        faith = llm_judge_faithfulness(judge_client, judge_model, trace["answer"], item["context"])
    else:
        corr = rule_based_correctness(trace["answer"], item["expected_answer"])
        faith = rule_based_faithfulness(trace["answer"], item["context"])

    rag = rag_precision_recall(trace.get("retrieved_doc_ids", []), item["relevant_doc_ids"])
    tools = tool_call_accuracy(trace.get("tool_calls", []), item["expected_tool_calls"])
    safety = rule_based_safety(trace["answer"])

    details["correctness"] = corr
    details["faithfulness"] = faith
    details["rag"] = rag
    details["tool_calls"] = tools
    details["safety"] = safety

    # Overall score: average of whichever applicable sub-scores exist (0-1 scale).
    applicable = []
    if corr["score"] is not None:
        applicable.append(corr["score"])
    if faith["score"] is not None:
        applicable.append(faith["score"])
    if rag["precision"] is not None:
        applicable.append((rag["precision"] + rag["recall"]) / 2)
    if tools["score"] is not None:
        applicable.append(tools["score"])

    overall = round(sum(applicable) / len(applicable), 3) if applicable else None
    if safety["flagged"]:
        overall = 0.0

    return {
        "correctness": corr["score"],
        "faithfulness": faith["score"],
        "rag_precision": rag["precision"],
        "rag_recall": rag["recall"],
        "tool_call_accuracy": tools["score"],
        "safety_flag": safety["flagged"],
        "safety_reason": safety["reason"],
        "overall_score": overall,
        "details": details,
    }


# ---------------- Agent runners ----------------

def run_mock_agent(item):
    """A deterministic mock agent for demo purposes: perturbs the expected
    answer slightly and 'retrieves' a plausible subset of relevant docs, so
    the dashboard has realistic-looking variance to explore."""
    import random
    start = time.time()
    expected = item["expected_answer"] or ""
    words = expected.split()
    if words and random.random() < 0.3:
        # simulate a partially wrong / truncated answer
        cutoff = max(1, int(len(words) * 0.6))
        answer = " ".join(words[:cutoff]) + " (unverified addition not in source material)"
    else:
        answer = expected

    relevant = item["relevant_doc_ids"]
    retrieved = list(relevant)
    if relevant and random.random() < 0.35:
        retrieved = retrieved[:-1]  # simulate missing a doc
    if random.random() < 0.2:
        retrieved.append("doc_irrelevant_99")

    tool_calls = list(item["expected_tool_calls"])
    if tool_calls and random.random() < 0.25:
        tool_calls = tool_calls[:-1]

    latency_ms = round((time.time() - start) * 1000 + random.uniform(200, 1800), 1)
    prompt_tokens = len(item["question"].split()) * 4 + sum(len(c.split()) for c in item["context"]) * 4
    completion_tokens = len(answer.split()) * 4

    return {
        "answer": answer,
        "retrieved_doc_ids": retrieved,
        "tool_calls": tool_calls,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "raw_trace": {"mode": "mock"},
    }


def run_http_agent(item, endpoint_url, headers=None, timeout=30):
    """Calls a user-supplied HTTP endpoint that accepts {"question": ..., "context": [...]}.
    Expects JSON back: {"answer": str, "retrieved_doc_ids": [...], "tool_calls": [...],
    "prompt_tokens": int, "completion_tokens": int}"""
    start = time.time()
    resp = requests.post(
        endpoint_url,
        json={"question": item["question"], "context": item["context"]},
        headers=headers or {},
        timeout=timeout,
    )
    latency_ms = round((time.time() - start) * 1000, 1)
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data.get("answer", ""),
        "retrieved_doc_ids": data.get("retrieved_doc_ids", []),
        "tool_calls": data.get("tool_calls", []),
        "latency_ms": latency_ms,
        "prompt_tokens": data.get("prompt_tokens", 0),
        "completion_tokens": data.get("completion_tokens", 0),
        "raw_trace": data,
    }


def run_anthropic_agent(item, client, model):
    """A minimal single-turn Anthropic-backed agent (no tools/RAG execution —
    just answers from provided context) for quick real-model comparisons."""
    start = time.time()
    ctx = "\n".join(item["context"])
    prompt = f"Context:\n{ctx}\n\nQuestion: {item['question']}\n\nAnswer concisely using only the context above."
    resp = client.messages.create(model=model, max_tokens=400, messages=[{"role": "user", "content": prompt}])
    latency_ms = round((time.time() - start) * 1000, 1)
    answer = "".join(b.text for b in resp.content if hasattr(b, "text"))
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    return {
        "answer": answer,
        "retrieved_doc_ids": item["relevant_doc_ids"],  # assumed perfect retrieval in this minimal harness
        "tool_calls": [],
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "raw_trace": {"mode": "anthropic", "model": model},
    }
