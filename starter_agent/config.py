"""Settings. Each has a default; put the ones you change in agent.json (see the README)."""
import json
import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    db: str = "agent.db"              # the one file that holds everything
    backend: str = "fake"             # fake | ollama | openai | anthropic
    model: str = ""                   # e.g. "llama3.1:8b" for ollama; blank = the backend's default
    base_url: str = ""                # blank = the backend's usual address
    api_key_env: str = ""             # the NAME of the env var holding your key; blank = the usual one
    embedder: str = "hash"            # hash (needs nothing) | bge-small (needs sentence-transformers)
    recall_budget_tokens: int = 8000  # plan section 5
    idle_minutes: int = 30            # a session ends after this long without a message (section 5)
    question_queue: int = 20          # the most open questions kept (section 6)
    dedupe_similarity: float = 0.85   # a new question this close to an open or dropped one is skipped
    auto_adopt: bool = False          # section 6: questions become desires only on `curious adopt`


def load_config(path="agent.json", **overrides):
    values = {}
    if path and os.path.exists(path):
        with open(path) as f:
            values = json.load(f)
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise SystemExit(f"{path}: unknown settings {sorted(unknown)}")
    values.update({name: value for name, value in overrides.items() if value is not None})
    return Config(**values)
