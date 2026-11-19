"""The command line does everything a student needs, and the README's walk-through runs as written."""
import re
import shlex
import time
from pathlib import Path

import pytest

from conftest import Clock
from starter_agent.cli import main
from starter_agent.models import FakeModel
from starter_agent.store import Store


def test_readme_walkthrough_runs(cli, capsys):
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    block = readme.split("runs every line of the next block -->")[1].split("```")[1]
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    assert len(lines) == 11 and all(line.startswith("python -m starter_agent ") for line in lines)
    for line in lines:
        assert cli(*shlex.split(line)[3:]) == 0, line
    assert capsys.readouterr().out.endswith('"turns_deleted": 4\n}\n')   # the last `beliefs` printed nothing


def test_every_command(cli, capsys, monkeypatch, tmp_path):
    out = lambda: capsys.readouterr().out
    # desires: add, list, rank, drop
    assert cli("desire", "add", "Replicate figure 3") == 0 and out() == "Desire #1 added.\n"
    assert cli("desire", "add", "Write the methods") == 0 and out() == "Desire #2 added.\n"
    assert cli("desire", "rank", "2", "1") == 0 and out() == "Desire is now #3, rank 1.\n"
    assert cli("desire", "rank", "1", "--rank", "3") == 0 and out() == "Desire is now #4, rank 3.\n"
    assert cli("desire", "list") == 0
    assert out() == "#3  rank 1  Write the methods  (student)\n#4  rank 3  Replicate figure 3  (student)\n"
    # chat, one message and the interactive loop; intentions
    assert cli("chat", "remember: the sample is 12 rats") == 0 and out() == "You said: remember: the sample is 12 rats\n"
    answers = iter(["why", "/end", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli("chat") == 0
    assert "agent> You said: why\nSession ended; curiosity queued 0 question(s)." in out()
    assert cli("intentions") == 0 and out() == "#1  active  Write the methods  (current)\n"
    # beliefs and their history
    assert cli("beliefs") == 0 and out() == "#1  0.60  the sample is 12 rats  (evidence: turns 1)\n"
    store = Store(str(tmp_path / "agent.db"), Clock())     # a second version, as a model would propose
    store.append("beliefs", {"text": "the sample is 14 rats", "confidence": 0.8, "evidence": "[1, 3]",
                             "flag": None}, store.get("beliefs", 1))
    assert cli("beliefs", "history", "2") == 0
    assert out() == ("#1  0.60  the sample is 12 rats  (evidence: turns 1)\n"
                     "#2  0.80  the sample is 14 rats  (evidence: turns 1, 3)  supersedes #1\n")
    # curiosity: list, drop, adopt
    assert cli("chat", "why are rats nocturnal?") == 0 and cli("session", "end") == 0
    assert "queued 1 question(s)" in out()
    assert cli("curious", "list") == 0
    assert re.fullmatch(r'#1  why are rats nocturnal\?\n      "I\'m not sure about that yet" \(turn 6, session 2\)\n',
                        out())
    assert cli("curious", "drop", "1") == 0 and "won't come back" in out()
    assert cli("curious", "list") == 0 and out() == ""
    assert cli("curious", "adopt", "1") == 1 and "open question" in out()      # dropped is not open
    assert cli("chat", "how do rats sleep?") == 0 and cli("session", "end") == 0
    assert cli("curious", "adopt", "3", "--rank", "2") == 0 and out().endswith("Adopted as desire #5.\n")
    assert cli("desire", "list") == 0 and "#5  rank 2  how do rats sleep?  (curiosity)" in out()
    assert cli("desire", "drop", "5") == 0 and cli("desire", "list") == 0
    assert "(curiosity)" not in out()
    # recall, sessions, log
    assert cli("recall", "rats", "sample") == 0 and out().startswith("sessions: [1, 2, 3]\n")
    assert cli("session", "list") == 0 and len(out().splitlines()) == 3
    assert cli("session", "show", "1") == 0 and "[turn 1] user: remember: the sample is 12 rats" in out()
    assert cli("log") == 0
    # delete asks first; anything but "yes" deletes nothing
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert cli("session", "delete", "1") == 1 and "Nothing deleted." in out()
    assert cli("session", "delete", "1", "--yes") == 0 and cli("beliefs") == 0
    assert out().endswith("}\n")                                               # no beliefs left
    # mistakes get a message and exit code 1, not a traceback
    for argv in (["desire", "add"], ["desire", "rank", "3"], ["desire", "rank", "99", "1"], ["desire", "drop", "x"],
                 ["beliefs", "history", "99"], ["curious", "drop", "99"], ["curious", "adopt"],
                 ["session", "show", "99"], ["session", "delete"]):
        assert cli(*argv) == 1, argv
        assert out().strip()


def test_a_mistake_in_agent_json_gets_a_message_not_a_traceback(cli, tmp_path):
    for text, message in (('{"backend": "ollama",}', "agent.json isn't valid JSON"),
                          ('["ollama"]', "should hold one JSON object"),
                          ('{"recall_budget_tokens": "8000"}', '"recall_budget_tokens" should be a whole number'),
                          ('{"recall_budget_tokens": 8000.5}', '"recall_budget_tokens" should be a whole number'),
                          ('{"auto_adopt": "no"}', '"auto_adopt" should be true or false'),
                          ('{"dedupe_similarity": true}', '"dedupe_similarity" should be a number'),
                          ('{"backend": 3}', '"backend" should be text in quotes'),
                          ('{"bakend": "ollama"}', "unknown settings ['bakend']")):
        (tmp_path / "agent.json").write_text(text)
        with pytest.raises(SystemExit) as stop:
            cli("beliefs")
        assert message in str(stop.value.code), text
    (tmp_path / "agent.json").write_text('{"dedupe_similarity": 1, "recall_budget_tokens": 400, "auto_adopt": false}')
    assert cli("beliefs") == 0                                       # a whole number is a number too


def test_an_unreachable_model_points_to_a_readme_section_that_exists(tmp_path, capsys):
    def unreachable(messages):
        raise OSError("connection refused")
    argv = ["--config", str(tmp_path / "agent.json"), "--db", str(tmp_path / "agent.db"), "chat", "hello"]
    assert main(argv, model=FakeModel(fn=unreachable), clock=Clock()) == 1
    (section,) = re.findall(r"See '([^']+)' in the README", capsys.readouterr().out)
    assert section in (Path(__file__).parent.parent / "README.md").read_text()


@pytest.fixture
def pacific(monkeypatch):
    """Your computer's clock set to California in winter (UTC-8), for the length of one test."""
    monkeypatch.setenv("TZ", "PST8")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_times_are_shown_in_local_time(cli, capsys, pacific):
    assert cli("chat", "hello") == 0 and cli("session", "end") == 0           # the test clock: 09:00 UTC
    capsys.readouterr()
    assert cli("session", "list") == 0
    assert capsys.readouterr().out == "#1  2026-11-19 01:00 to 2026-11-19 01:00  hello\n"
    assert cli("session", "show", "1") == 0
    assert capsys.readouterr().out.startswith("=== Session 1, 2026-11-19 01:00 to 2026-11-19 01:00 ===")
