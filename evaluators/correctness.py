"""
Correctness evaluation: how well the agent's answer matches the expected answer.

Two modes:
- rule_based: fast, free, deterministic (difflib similarity + keyword overlap)
- llm_judge: uses an LLM to grade correctness on a 0-1 scale with reasoning
  (requires an Anthropic API key to be configured)
"""
import difflib
import re


def _normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _keyword_overlap(a, b):
    wa = set(re.findall(r"[a-z0-9]+", _normalize(a)))
    wb = set(re.findall(r"[a-z0-9]+", _normalize(b)))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def rule_based_correctness(answer: str, expected_answer: str) -> dict:
    if not expected_answer:
        return {"score": None, "method": "rule_based", "reason": "No expected answer provided; skipped."}

    a, e = _normalize(answer), _normalize(expected_answer)
    seq_sim = difflib.SequenceMatcher(None, a, e).ratio()
    kw_sim = _keyword_overlap(answer, expected_answer)
    score = round(0.5 * seq_sim + 0.5 * kw_sim, 3)
    return {
        "score": score,
        "method": "rule_based",
        "reason": f"sequence similarity={seq_sim:.2f}, keyword overlap={kw_sim:.2f}",
    }


def llm_judge_correctness(client, model, question, answer, expected_answer) -> dict:
    """client is an anthropic.Anthropic() instance. Falls back to rule-based on any error."""
    if not expected_answer:
        return {"score": None, "method": "llm_judge", "reason": "No expected answer provided; skipped."}
    try:
        prompt = f"""You are grading an AI agent's answer for correctness against a reference answer.

Question: {question}
Reference (expected) answer: {expected_answer}
Agent's answer: {answer}

Score correctness from 0.0 to 1.0 where 1.0 means fully correct and equivalent in meaning,
0.5 means partially correct, and 0.0 means wrong or irrelevant.
Respond ONLY with JSON: {{"score": <float>, "reason": "<short reason>"}}"""
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        import json as _json
        text = text.strip().strip("`").replace("json", "", 1) if text.strip().startswith("```") else text
        data = _json.loads(text)
        return {"score": float(data["score"]), "method": "llm_judge", "reason": data.get("reason", "")}
    except Exception as e:
        fallback = rule_based_correctness(answer, expected_answer)
        fallback["reason"] += f" (LLM judge failed, used fallback: {e})"
        return fallback
