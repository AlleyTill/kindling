"""Settings. Each has a default; put the ones you change in agent.json (see the README)."""
import json
import os
import re
from dataclasses import dataclass, fields


@dataclass
class Config:
    db: str = "memory/agent.db"       # the one file that holds everything, in this agent's own folder
    backend: str = "fake"             # fake | ollama | openai | anthropic
    model: str = ""                   # e.g. "llama3.1:8b" for ollama; blank = the backend's default
    base_url: str = ""                # blank = the backend's usual address
    api_key_env: str = ""             # the NAME of the env var holding your key; blank = the usual one
    embedder: str = "hash"            # hash (needs nothing) | bge-small (needs sentence-transformers)
    recall_budget_tokens: int = 8000  # plan section 5: the most recall may use; less if the context is short
    context_tokens: int = 0           # the model's window; 0 = the backend's usual one (models.CONTEXT)
    idle_minutes: int = 30            # a session ends after this long without a message (section 5)
    question_queue: int = 20          # the most open questions kept (section 6)
    dedupe_similarity: float = 0.85   # a new question this close to an open or dropped one is skipped
    auto_adopt: bool = False          # section 6: questions become desires only on `curious adopt`


KINDS = {bool: "true or false", int: "a whole number", float: "a number", str: "text in quotes"}
RANGES = {"recall_budget_tokens": (0, 10**7), "context_tokens": (0, 10**8), "idle_minutes": (0, 10**6),
          "question_queue": (1, 10**4), "dedupe_similarity": (0, 1)}


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
        low, high = RANGES.get(name, (value, value))
        if not low <= value <= high:                     # NaN fails this too
            raise SystemExit(f'{path}: "{name}" should be from {low} to {high} (the default is '
                             f"{json.dumps(defaults[name].default)}), not {json.dumps(value)}.")
    if not re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)?", values.get("api_key_env", "")):   # never echo it back
        raise SystemExit(f'{path}: "api_key_env" should be the NAME of a variable, like ANTHROPIC_API_KEY. '
                         "What's there isn't one (if it's a key, take it out of the file and replace that key).")
    values.update({name: value for name, value in overrides.items() if value is not None})
    db = os.path.expanduser(values.get("db", Config.db))
    if not db.strip():
        raise SystemExit(f'{path}: "db" is empty: give a file name, like "memory/agent.db".')
    if overrides.get("db") is None and db != ":memory:" and not os.path.isabs(db):
        db = os.path.join(os.path.dirname(path or ""), db)    # beside this agent's agent.json
    return Config(**{**values, "db": db})
