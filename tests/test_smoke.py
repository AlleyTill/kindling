"""The whole thing, the way a student first meets it, plus two guarantees the README makes."""
import os
import subprocess
import sys
from pathlib import Path

from conftest import Clock
from starter_agent.cli import main
from starter_agent.config import Config
from starter_agent.models import FakeModel
from starter_agent.store import Store


def test_demo_runs_with_no_key(cli, capsys):
    assert cli("chat", "remember: my experiment uses 40 mice") == 0
    assert cli("chat", "why do the mice lose weight?") == 0
    assert cli("session", "end") == 0
    assert cli("chat", "tell me about the mice") == 0
    out = capsys.readouterr().out
    assert "I'm not sure about that yet" in out and "I recalled session 1" in out
    assert cli("beliefs") == 0
    assert "0.60  my experiment uses 40 mice  (evidence: turns 1)" in capsys.readouterr().out
    assert cli("curious", "list") == 0
    assert '"I\'m not sure about that yet" (turn 4, session 1)' in capsys.readouterr().out
    assert cli("session", "delete", "1", "--yes") == 0
    assert '"belief_versions_removed": 1' in capsys.readouterr().out
    assert cli("beliefs") == 0 and capsys.readouterr().out == ""
    assert cli("curious", "list") == 0 and capsys.readouterr().out == ""


def test_idle_gap_ends_the_session_at_the_next_chat_and_curious_list_only_says_so(cli, capsys):
    cli("chat", "why is it raining?")
    capsys.readouterr()
    cli.clock.advance(29)
    assert cli("curious", "list") == 0 and capsys.readouterr().out == ""      # not idle long enough
    cli.clock.advance(2)
    assert cli("curious", "list") == 0
    out = capsys.readouterr().out
    assert "raining" not in out                                               # not ended, not read
    assert "Session 1 has been idle" in out and "python -m starter_agent session end" in out
    cli("chat", "hello again")                     # the next chat ends the idle session, and curiosity reads it
    capsys.readouterr()
    cli("session", "list")
    listing = capsys.readouterr().out.splitlines()
    assert len(listing) == 2 and "(open)" not in listing[0] and "(open)" in listing[1]
    cli("curious", "list")
    assert "why is it raining?" in capsys.readouterr().out


def test_curious_list_and_intentions_never_call_the_model_or_write(tmp_path, capsys):
    clock, model = Clock(), FakeModel()            # the demo model, recording every call
    run = lambda *argv: main(["--config", str(tmp_path / "agent.json"), "--db", str(tmp_path / "agent.db"),
                              *argv], model=model, clock=clock)
    run("desire", "add", "Weigh the mice")         # desire #1
    run("chat", "why do the mice lose weight?")    # writes intention #1, serving desire #1
    run("desire", "drop", "1")
    run("desire", "add", "Replicate figure 3")     # desire #3: no intention written yet
    clock.advance(31)                              # the session is now idle, and still open
    capsys.readouterr()
    calls, before = len(model.calls), list(Store(str(tmp_path / "agent.db"), clock).db.iterdump())
    assert run("intentions") == 0
    assert capsys.readouterr().out == (
        "(new)  active  Replicate figure 3  (current; your next chat turn writes it down)\n")   # not #1
    assert run("curious", "list") == 0 and "is still open" in capsys.readouterr().out
    assert len(model.calls) == calls                                   # no model call
    assert list(Store(str(tmp_path / "agent.db"), clock).db.iterdump()) == before   # no row written
    run("chat", "hello")                           # the next chat ends the idle session and writes it down
    capsys.readouterr()
    assert run("intentions") == 0 and capsys.readouterr().out == "#2  active  Replicate figure 3  (current)\n"


def test_default_settings_need_nothing_installed():
    """The defaults are the fake model and the hash embedder, and building the agent imports nothing
    beyond Python itself (numpy is optional and unused by default)."""
    c = Config()
    assert (c.backend, c.embedder, c.auto_adopt) == ("fake", "hash", False)
    # the plan's numbers: 8,000-token recall, 30-minute idle gap, 20 open questions, 0.85 dedupe
    assert (c.recall_budget_tokens, c.idle_minutes, c.question_queue, c.dedupe_similarity) == (8000, 30, 20, 0.85)
    code = ("import sys; from starter_agent.cli import build; from starter_agent.config import Config;"
            "build(Config(db=':memory:')); print(sorted({'sentence_transformers', 'numpy'} & set(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).parent.parent, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert (out.returncode, out.stdout.strip()) == (0, "[]"), out.stderr


def test_one_model_call_per_turn_and_one_file(make_agent, tmp_path):
    agent = make_agent(fn=lambda messages: "ok")
    for i in range(3):
        agent.handle(f"message {i}")
    assert len(agent.model.calls) == 3
    agent.end_session()
    assert len(agent.model.calls) == 4                    # plus curiosity, once
    assert sorted(p.name for p in tmp_path.iterdir()) == ["agent.db"]
