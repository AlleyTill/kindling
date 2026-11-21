"""Shared test helpers. Every test uses FakeModel and HashEmbedder: no network, no downloads."""
import json
import math
import re
import socket
from datetime import datetime, timedelta, timezone

import pytest

from starter_agent import curiosity as curiosity_module
from starter_agent import models
from starter_agent.agent import Agent
from starter_agent.cli import main
from starter_agent.config import Config
from starter_agent.embed import HashEmbedder
from starter_agent.models import FakeModel
from starter_agent.store import Store


class Clock:
    """A clock that moves one second per reading, and jumps when told to."""

    def __init__(self):
        self.t = datetime(2026, 11, 19, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t

    def advance(self, minutes):
        self.t += timedelta(minutes=minutes)


def say(text, beliefs=(), done=(), **extra):
    """A scripted Act reply: prose plus the JSON block of belief changes (and any extra keys)."""
    block = {"beliefs": list(beliefs), "intention_done": list(done), **extra}
    return f"{text}\n```json\n{json.dumps(block)}\n```"


def scripted(act_replies, curiosity=lambda prompt: '{"questions": []}'):
    """A fake model: Act calls get the next scripted reply; the curiosity call gets curiosity(prompt)."""
    queue = list(act_replies)

    def fn(messages):
        if messages[0]["content"] == curiosity_module.SYSTEM:
            return curiosity(messages[-1]["content"])
        assert queue, f"scripted {len(act_replies)} replies, and the agent asked for more"
        return queue.pop(0)
    return fn


def agent_turns(prompt):
    """{turn id: text} for the agent turns shown in a curiosity prompt."""
    return {int(t): text for t, text in re.findall(r"\[turn (\d+)\] agent: (.*)", prompt)}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The tests never reach a real model: any attempt to connect, or to look a name up, fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach a model over the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    """Each test starts as a new run of Kindling would: no server asked anything yet, no note printed yet, no
    risk of a cut known yet, and no count kept for the log."""
    monkeypatch.setattr(models, "ASKED", {})
    monkeypatch.setattr(models, "TOLD", set())
    monkeypatch.setattr(models, "RISKS", [])
    monkeypatch.setattr(models, "COUNTS", [])


@pytest.fixture(autouse=True)
def fast_files(monkeypatch):
    """Test databases skip the wait for the disk after each write (synchronous=OFF): same bytes in the file,
    a few times faster. The agent itself keeps SQLite's safe default."""
    real = Store.__init__

    def init(self, *args, **kwargs):
        real(self, *args, **kwargs)
        self.db.execute("PRAGMA synchronous=OFF")
    monkeypatch.setattr(Store, "__init__", init)


@pytest.fixture
def make_agent(tmp_path):
    def make(replies=None, fn=None, **settings):
        config = Config(**{"db": str(tmp_path / "agent.db"), **settings})
        return Agent(Store(config.db, Clock()), FakeModel(replies, fn), HashEmbedder(), config)
    return make


@pytest.fixture
def cli(tmp_path):
    """Run the command line on a fresh database, ignoring any agent.json in the working folder."""
    clock = Clock()
    settings = tmp_path / "agent.json"
    if not settings.exists():
        settings.write_text("{}")

    def run(*argv, model=None):
        return main(["--config", str(settings), "--db", str(tmp_path / "agent.db"), *argv], model=model, clock=clock)
    run.clock = clock
    return run


class StubEmbedder:
    """Dense similarity chosen by the test: each listed text gets that cosine with the query."""
    name = "stub"

    def __init__(self, cosine_with_query):
        self.cosine = cosine_with_query

    def embed(self, texts):
        return [[c, math.sqrt(1 - c * c)] if (c := self.cosine.get(t)) is not None else [1.0, 0.0]
                for t in texts]


class PointEmbedder:
    """Vectors chosen by the test, the query's included; any other text points somewhere else."""
    name = "points"

    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        return [self.vectors.get(text, [0.0, 0.0, 1.0]) for text in texts]


def only(rows):
    """The one row expected: none or several fails by assertion, so a mutant counts as caught, not as a crash."""
    rows = list(rows)
    assert len(rows) == 1, rows
    return rows[0]


def add_session(store, turns):
    """Write a finished session straight into the store: turns is a list of (role, text)."""
    sid = store.start_session()
    ids = [store.add_turn(sid, role, text) for role, text in turns]
    store.end_session(sid)
    return sid, ids
