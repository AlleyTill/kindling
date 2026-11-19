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


KINDS = {bool: "true or false", int: "a whole number", float: "a number", str: "text in quotes"}


def right_kind(kind, value):
    if isinstance(value, bool):          # in Python, True is also an int, so check for it first
        return kind is bool
    return isinstance(value, (int, float)) if kind is float else isinstance(value, kind)


def load_config(path="agent.json", **overrides):
    """Your settings from agent.json over the defaults. A mistake in the file stops with a message."""
    values = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                values = json.load(f)
            except ValueError as error:
                raise SystemExit(f"{path} isn't valid JSON ({error}). It should look like this: "
                                 '{"backend": "ollama", "model": "llama3.1:8b"}')
    if not isinstance(values, dict):
        raise SystemExit(f'{path} should hold one JSON object, like {{"backend": "ollama"}}.')
    defaults = {f.name: f for f in fields(Config)}
    unknown = set(values) - set(defaults)
    if unknown:
        raise SystemExit(f"{path}: unknown settings {sorted(unknown)}")
    for name, value in values.items():
        if not right_kind(defaults[name].type, value):
            raise SystemExit(f'{path}: "{name}" should be {KINDS[defaults[name].type]} (the default is '
                             f"{json.dumps(defaults[name].default)}), not {json.dumps(value)}.")
    values.update({name: value for name, value in overrides.items() if value is not None})
    return Config(**values)
