"""
Faithfulness / hallucination check: does the answer stay grounded in the provided context,
or does it assert things not supported by it?

Rule-based proxy: fraction of the answer's informative n-grams/keywords that can be found
in the concatenated context. Low overlap on a context-dependent answer suggests
unsupported (possibly hallucinated) content.
"""
import re


def _keywords(text):
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))


STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "this", "that", "with", "from",
    "have", "has", "had", "not", "but", "you", "your", "our", "their", "its",
}


def rule_based_faithfulness(answer: str, context: list) -> dict:
    if not context:
        return {"score": None, "method": "rule_based", "reason": "No context supplied; faithfulness not applicable."}

    ctx_text = " ".join(context)
    ans_kw = _keywords(answer) - STOPWORDS
    ctx_kw = _keywords(ctx_text) - STOPWORDS

    if not ans_kw:
        return {"score": None, "method": "rule_based", "reason": "Answer had no substantive content to check."}

    supported = ans_kw & ctx_kw
    score = round(len(supported) / len(ans_kw), 3)
    unsupported_sample = list(ans_kw - ctx_kw)[:8]
    return {
        "score": score,
        "method": "rule_based",
        "reason": f"{len(supported)}/{len(ans_kw)} key terms grounded in context. "
                  f"Unsupported terms (sample): {unsupported_sample}",
    }


def llm_judge_faithfulness(llm_client, answer, context) -> dict:
    """llm_client is an llm_client.LLMClient instance (provider-agnostic: Anthropic or Groq)."""
    if not context:
        return {"score": None, "method": "llm_judge", "reason": "No context supplied; faithfulness not applicable."}
    try:
        ctx_text = "\n---\n".join(context)
        prompt = f"""Given the CONTEXT below, determine whether the ANSWER is fully faithful to it
(i.e., contains no claims that are unsupported or contradicted by the context).

CONTEXT:
{ctx_text}

ANSWER:
{answer}

Score faithfulness 0.0 (heavy hallucination) to 1.0 (fully grounded).
Respond ONLY with JSON: {{"score": <float>, "reason": "<short reason>"}}"""
        text, _usage = llm_client.complete(prompt, max_tokens=200)
        import json as _json
        text = text.strip().strip("`").replace("json", "", 1) if text.strip().startswith("```") else text
        data = _json.loads(text)
        return {"score": float(data["score"]), "method": "llm_judge", "reason": data.get("reason", "")}
    except Exception as e:
        fallback = rule_based_faithfulness(answer, context)
        fallback["reason"] += f" (LLM judge failed, used fallback: {e})"
        return fallback
