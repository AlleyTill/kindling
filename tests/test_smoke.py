"""The whole thing, the way a student first meets it, plus two guarantees the README makes."""
import os
import subprocess
import sys
from pathlib import Path

from starter_agent.config import Config


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


def test_idle_gap_ends_the_session_and_runs_curiosity(cli, capsys):
    cli("chat", "why is it raining?")
    capsys.readouterr()
    cli.clock.advance(29)
    cli("curious", "list")
    assert "raining" not in capsys.readouterr().out       # not idle long enough: still open
    cli.clock.advance(2)
    cli("curious", "list")
    assert "why is it raining?" in capsys.readouterr().out    # idle: ended and read, without a new message
    cli("chat", "hello again")
    capsys.readouterr()
    cli("session", "list")
    listing = capsys.readouterr().out.splitlines()
    assert len(listing) == 2 and "(open)" not in listing[0] and "(open)" in listing[1]
    cli("curious", "list")
    assert "why is it raining?" in capsys.readouterr().out


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
