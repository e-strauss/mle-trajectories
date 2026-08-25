"""Orchestration: prompt -> LLM -> write -> validate -> repair (up to N rounds)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import prompt as P
from .llm import LLM
from .validate import Validation, validate


@dataclass
class SkrubifyConfig:
    llm: LLM | None = None                  # None => dry run (prompt only)
    guide_path: Path | None = None
    examples_dir: Path | None = None
    n_examples: int | None = None
    extra_instructions: str | None = None
    max_repairs: int = 2
    validate_build: bool = True
    strict: bool = False
    python: str | None = None               # interpreter used to build the plan
    timeout: int = 300                      # per plan build
    keep_attempts: bool = False             # also write <stem>.attemptN.py
    verbose: bool = True


@dataclass
class Attempt:
    index: int
    code: str
    validation: Validation
    seconds: float = 0.0

    def as_dict(self) -> dict:
        v = self.validation
        return {"attempt": self.index, "ok": v.ok, "seconds": round(self.seconds, 1),
                "check_errors": v.checks.errors, "warnings": v.checks.warnings,
                "build_ok": v.build_ok, "build_error": v.build_error,
                "structural": v.structural, "info": v.info}


@dataclass
class Result:
    source: Path
    out_path: Path | None
    attempts: list[Attempt] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    prompt_chars: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].validation.ok

    @property
    def final(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    def as_dict(self) -> dict:
        return {"source": str(self.source),
                "out": str(self.out_path) if self.out_path else None,
                "ok": self.ok, "prompt_chars": self.prompt_chars,
                "attempts": [a.as_dict() for a in self.attempts],
                "llm": self.stats}


def default_out_path(source: Path) -> Path:
    """``<dir>/foo.py`` -> ``<dir>/skrubify/foo.py`` (the layout already in use)."""
    source = Path(source)
    return source.parent / "skrubify" / source.name


def build_messages(source: Path, cfg: SkrubifyConfig) -> list[dict]:
    guide = P.load_guide(cfg.guide_path)
    examples = P.load_examples(cfg.examples_dir, limit=cfg.n_examples)
    user = P.build_user_prompt(Path(source).read_text(), source_name=Path(source).name,
                               guide=guide, examples=examples,
                               extra_instructions=cfg.extra_instructions)
    return [{"role": "system", "content": P.SYSTEM},
            {"role": "user", "content": user}]


def skrubify_file(source: Path, out_path: Path | None = None, *,
                  cfg: SkrubifyConfig | None = None) -> Result:
    """Convert one script. Writes the best candidate to ``out_path`` and validates it."""
    cfg = cfg or SkrubifyConfig()
    source = Path(source).resolve()
    out_path = Path(out_path) if out_path else default_out_path(source)
    messages = build_messages(source, cfg)
    result = Result(source=source, out_path=out_path,
                    prompt_chars=sum(len(m["content"]) for m in messages))

    if cfg.llm is None:
        raise ValueError("SkrubifyConfig.llm is None -- nothing to call "
                         "(use build_messages() for a dry run)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(cfg.max_repairs + 1):
        label = "generating" if i == 0 else f"repair round {i}"
        _log(cfg, f"  {label} …")
        t0 = time.monotonic()
        code = P.extract_code(cfg.llm.complete(messages))
        out_path.write_text(code)
        if cfg.keep_attempts:
            out_path.with_suffix(f".attempt{i}.py").write_text(code)
        v = validate(out_path, python=cfg.python, strict=cfg.strict,
                     build=cfg.validate_build, timeout=cfg.timeout)
        if v.build_timed_out and v.checks.ok:
            # Out of clock, not out of correctness: give a big plan more time
            # rather than spending a repair round telling the model to fix a
            # stopwatch (measured: 0020 burned 3 attempts that way).
            _log(cfg, f"    build timed out after {cfg.timeout}s -- retrying "
                      f"with {cfg.timeout * 3}s")
            v = validate(out_path, python=cfg.python, strict=cfg.strict,
                         build=cfg.validate_build, timeout=cfg.timeout * 3)
        attempt = Attempt(index=i, code=code, validation=v,
                          seconds=time.monotonic() - t0)
        result.attempts.append(attempt)
        _log(cfg, f"    {v.summary()}")
        if v.ok:
            break
        if i == cfg.max_repairs:
            break
        for line in v.feedback().splitlines():
            _log(cfg, f"    | {line}")
        messages += [{"role": "assistant", "content": f"```python\n{code}```"},
                     {"role": "user", "content": P.build_repair_prompt(v.feedback())}]

    result.stats = cfg.llm.stats()
    return result


def _log(cfg: SkrubifyConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg, file=sys.stderr, flush=True)
