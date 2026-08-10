"""
Unified LLM client so the rest of the platform (judge evaluators, built-in
agent runner) doesn't care whether you're using Anthropic or Groq.

Usage:
    client = LLMClient(provider="groq", api_key="gsk_...", model="llama-3.3-70b-versatile")
    text, usage = client.complete("Say hi in 3 words")
"""


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self._client = None

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def complete(self, prompt: str, max_tokens: int = 400):
        """Returns (text, {"prompt_tokens": int, "completion_tokens": int})."""
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "input_tokens", 0) if usage else 0
            ct = getattr(usage, "output_tokens", 0) if usage else 0
            return text, {"prompt_tokens": pt, "completion_tokens": ct}

        elif self.provider == "groq":
            resp = self._client.chat.completions.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) if usage else 0
            ct = getattr(usage, "completion_tokens", 0) if usage else 0
            return text, {"prompt_tokens": pt, "completion_tokens": ct}

        raise ValueError(f"Unknown provider: {self.provider}")


# Common Groq model options (as of this build — check console.groq.com/docs/models for latest)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "mixtral-8x7b-32768",
]

ANTHROPIC_MODELS = [
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "claude-opus-4-8",
]
