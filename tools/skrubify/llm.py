"""Provider-agnostic chat completion through litellm, plus dot-file key loading.

litellm is imported lazily so the rest of the tool (`--check`, `--print-prompt`)
works without it installed:

    uv pip install litellm

Model strings are litellm's ``<provider>/<model>``. Pass ``--provider`` and
``--model`` separately (``--provider gemini --model gemini-2.5-pro``), or a single
fully-qualified ``--model gemini/gemini-2.5-pro``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# provider -> (litellm prefix, a reasonable default model, required env keys)
PROVIDERS: dict[str, tuple[str, str | None, tuple[str, ...]]] = {
    "openai":      ("openai",      "gpt-5",                  ("OPENAI_API_KEY",)),
    "anthropic":   ("anthropic",   "claude-sonnet-4-5",      ("ANTHROPIC_API_KEY",)),
    "gemini":      ("gemini",      "gemini-2.5-pro",         ("GEMINI_API_KEY",)),
    "vertex_ai":   ("vertex_ai",   "gemini-2.5-pro",         ()),
    "azure":       ("azure",       None,                     ("AZURE_API_KEY", "AZURE_API_BASE")),
    "openrouter":  ("openrouter",  None,                     ("OPENROUTER_API_KEY",)),
    "deepseek":    ("deepseek",    "deepseek-chat",          ("DEEPSEEK_API_KEY",)),
    "mistral":     ("mistral",     "mistral-large-latest",   ("MISTRAL_API_KEY",)),
    "groq":        ("groq",        None,                     ("GROQ_API_KEY",)),
    "together_ai": ("together_ai", None,                     ("TOGETHER_API_KEY",)),
    "xai":         ("xai",         "grok-4",                 ("XAI_API_KEY",)),
    "ollama":      ("ollama",      None,                     ()),
}

ENV_FILENAMES = (".env", ".env.local", ".skrubify.env")


def find_env_files(start: Path | None = None) -> list[Path]:
    """Dot files to try: $SKRUBIFY_ENV_FILE, then cwd and its parents, then the package."""
    out = []
    explicit = os.environ.get("SKRUBIFY_ENV_FILE")
    if explicit:
        out.append(Path(explicit).expanduser())
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        out += [d / n for n in ENV_FILENAMES]
        if (d / ".git").exists():
            break
    out += [Path(__file__).parent / n for n in ENV_FILENAMES]
    return out


def load_env(path: Path | str | None = None, *, override: bool = False) -> list[Path]:
    """Load ``KEY=value`` lines into os.environ. Returns the files actually read.

    Existing environment variables win unless ``override``. Values may be quoted;
    ``export KEY=value``, blank lines and ``#`` comments are handled.
    """
    candidates = [Path(path).expanduser()] if path else find_env_files()
    loaded = []
    for f in candidates:
        if not f.is_file():
            continue
        for raw in f.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ").strip()
            key, _, value = line.partition("=")
            if not _:
                continue
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value
        loaded.append(f)
        if path:            # an explicit file is the only one we read
            break
    return loaded


def resolve_model(model: str | None, provider: str | None) -> str:
    """Combine --provider/--model into one litellm model string."""
    if model and "/" in model:
        return model
    if not provider:
        if not model:
            raise ValueError("pass --model (and usually --provider), e.g. "
                             "--provider openai --model gpt-5")
        return model                     # bare name: litellm defaults to openai
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; known: "
                         f"{', '.join(sorted(PROVIDERS))} (or pass a fully "
                         "qualified --model like 'openrouter/qwen/qwen3-coder')")
    prefix, default_model, _ = PROVIDERS[provider]
    name = model or default_model
    if not name:
        raise ValueError(f"provider {provider!r} has no default model -- pass --model")
    return f"{prefix}/{name}"


def missing_keys(model: str) -> list[str]:
    provider = model.split("/", 1)[0]
    for _, (prefix, _d, keys) in PROVIDERS.items():
        if prefix == provider:
            return [k for k in keys if not os.environ.get(k)]
    return []


@dataclass
class LLM:
    """Thin litellm wrapper: a chat call plus token/cost bookkeeping."""

    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    num_retries: int = 2
    timeout: int | None = 600
    extra: dict = field(default_factory=dict)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    def complete(self, messages: list[dict]) -> str:
        import litellm

        kwargs = dict(model=self.model, messages=messages,
                      num_retries=self.num_retries, **self.extra)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        response = litellm.completion(**kwargs)
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        try:
            self.cost += litellm.completion_cost(completion_response=response) or 0.0
        except Exception:       # unknown model pricing -- cost stays 0
            pass
        return response.choices[0].message.content or ""

    def stats(self) -> dict:
        return {"model": self.model, "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost, 6)}
