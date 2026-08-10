"""
Tool-call accuracy: did the agent call the expected tools (right tools, not
extraneous ones), regardless of order.
"""


def tool_call_accuracy(actual_tool_calls: list, expected_tool_calls: list) -> dict:
    if not expected_tool_calls:
        return {
            "score": None,
            "reason": "No expected tool calls labeled for this item; not applicable.",
        }
    actual = set(actual_tool_calls or [])
    expected = set(expected_tool_calls)

    correct = actual & expected
    missing = expected - actual
    extra = actual - expected

    if not expected:
        score = 1.0 if not extra else 0.0
    else:
        score = round(len(correct) / len(expected | actual) if (expected | actual) else 1.0, 3)

    reason = f"Correct: {sorted(correct)}. Missing: {sorted(missing)}. Unexpected: {sorted(extra)}."
    return {"score": score, "reason": reason}
