"""Shared test helpers. Every test uses FakeModel and HashEmbedder: no network, no downloads."""
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from starter_agent import curiosity as curiosity_module
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
        return queue.pop(0)
    return fn


def agent_turns(prompt):
    """{turn id: text} for the agent turns shown in a curiosity prompt."""
    return {int(t): text for t, text in re.findall(r"\[turn (\d+)\] agent: (.*)", prompt)}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The tests never reach a real model: any attempt fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach a model over the network")
    monkeypatch.setattr("starter_agent.models.urllib.request.urlopen", refuse)


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

    def run(*argv):
        return main(["--config", str(tmp_path / "agent.json"), "--db", str(tmp_path / "agent.db"), *argv],
                    clock=clock)
    run.clock = clock
    return run


def add_session(store, turns):
    """Write a finished session straight into the store: turns is a list of (role, text)."""
    sid = store.start_session()
    ids = [store.add_turn(sid, role, text) for role, text in turns]
    store.end_session(sid)
    return sid, ids
