"""skrubify -- LLM-driven conversion of a plain sklearn/pandas ML script into a
valid skrub DataOps pipeline.

Public entry point: :func:`skrubify.core.skrubify_file` (or the CLI,
``python -m skrubify``).
"""
from __future__ import annotations

__all__ = ["skrubify_file", "SkrubifyConfig", "Result"]


def __getattr__(name):  # lazy so `--check` works without litellm installed
    if name in __all__:
        from . import core
        return getattr(core, name)
    raise AttributeError(name)
