"""Caminhos e configuração lidos do ambiente (e de um `.env`, se existir)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def cases(self) -> Path:
        return self.root / "evals" / "cases"


def get_paths() -> Paths:
    """Raiz do projeto: `REPERTORIO_HOME` ou o diretório atual."""
    return Paths(Path(os.getenv("REPERTORIO_HOME", Path.cwd())).resolve())


@dataclass(frozen=True)
class LLMSettings:
    anthropic_model: str
    anthropic_effort: str
    openai_model: str
    openai_reasoning_effort: str | None


def get_llm_settings() -> LLMSettings:
    return LLMSettings(
        anthropic_model=os.getenv("ANTHROPIC_MODEL") or "claude-opus-5-5",
        anthropic_effort=os.getenv("ANTHROPIC_EFFORT") or "medium",
        openai_model=os.getenv("OPENAI_MODEL") or "gpt-5",
        openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT") or None,
    )
