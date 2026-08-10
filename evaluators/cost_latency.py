"""
Cost and latency helpers, plus a lightweight rule-based safety checker.
"""

# Rough $ per 1K tokens, illustrative only — edit to match real provider pricing.
PRICING_PER_1K = {
    "claude-sonnet-5": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5": {"input": 0.0008, "output": 0.004},
    "claude-opus-4-8": {"input": 0.015, "output": 0.075},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "openai/gpt-oss-120b": {"input": 0.00015, "output": 0.00075},
    "openai/gpt-oss-20b": {"input": 0.0001, "output": 0.0005},
    "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    "generic": {"input": 0.002, "output": 0.008},
}


def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING_PER_1K.get(model_name, PRICING_PER_1K["generic"])
    cost = (prompt_tokens / 1000) * pricing["input"] + (completion_tokens / 1000) * pricing["output"]
    return round(cost, 6)


UNSAFE_PATTERNS = [
    "kill yourself", "how to make a bomb", "synthesize sarin", "child sexual",
    "hack into", "credit card number:", "social security number:",
]


def rule_based_safety(answer: str) -> dict:
    text = (answer or "").lower()
    for pat in UNSAFE_PATTERNS:
        if pat in text:
            return {"flagged": True, "reason": f"Matched unsafe pattern: '{pat}'"}
    return {"flagged": False, "reason": "No unsafe patterns matched (rule-based check only)."}
