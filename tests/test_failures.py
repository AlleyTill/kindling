"""When something goes wrong: a second terminal, a full disk, a failed write, Ctrl-C, a reply the agent can't
use. The rule each test holds: your word on deletion still wins, and no message is half-kept or kept twice.
(Round 3: the Agent Army's red-hat-tester and security-auditor found each of these.)"""
import json
import sqlite3

import pytest

from conftest import Clock, add_session, say
from starter_agent import bdi, curiosity
from starter_agent.agent import Agent
from starter_agent.config import Config
from starter_agent.embed import HashEmbedder
from starter_agent.models import FakeModel, ModelError, demo_brain
from starter_agent.store import Store


def file_bytes(tmp_path):
    return b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("agent.db"))


def outcome(call, *args):
    try:
        return call(*args)
    except BaseException as error:              # KeyboardInterrupt too: a crash must fail by assertion
        return error


def roles(store):
    return [row["role"] for row in store.db.execute("SELECT role FROM turns ORDER BY id")]


class Fails:
    """A connection whose statements starting with `prefix` fail the way a full disk does; the rest pass."""

    def __init__(self, db, prefix, error=sqlite3.OperationalError("database or disk is full (planted)")):
        self.real, self.prefix, self.error = db, prefix, error

    def execute(self, sql, *args):
        if sql.startswith(self.prefix):
            raise self.error
        return self.real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.real, name)


# --- A second terminal deletes the session while the model is answering ----------------------------------

def test_a_session_deleted_while_the_model_answers_keeps_none_of_its_text(make_agent, tmp_path):
    second_terminal = make_agent()                       # the same file, another connection

    def slow_model(messages):
        second_terminal.delete_session(1)
        return demo_brain(messages)
    agent = make_agent(fn=slow_model)
    reply = agent.handle("remember: my landlord is ZEPHYRINE-BRAND")
    assert "deleted while the model was answering" in reply
    assert agent.store.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0
    assert bdi.current_beliefs(agent.store) == [] and agent.store.db.execute("SELECT * FROM log").fetchall()[-1][
        "kind"] == "deleted"
    assert b"ZEPHYRINE" not in file_bytes(tmp_path)


def test_a_session_deleted_while_curiosity_reads_it_gets_no_questions(make_agent, tmp_path):
    second_terminal = make_agent()

    def slow_model(messages):
        if messages[0]["content"] == curiosity.SYSTEM:
            second_terminal.delete_session(1)
        return demo_brain(messages)
    agent = make_agent(fn=slow_model)
    agent.handle("is my passphrase MARIGOLD-ORCHARD?")
    assert agent.end_session() == []
    assert agent.store.db.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 0
    assert b"MARIGOLD" not in file_bytes(tmp_path)


def test_a_delete_also_clears_what_an_older_version_left_behind(make_agent, tmp_path):
    agent = make_agent()
    store = agent.store
    add_session(store, [("user", "first"), ("agent", "ok")])
    store.db.execute("INSERT INTO turns (session_id, n, role, text, at) VALUES (99, 1, 'agent', "
                     "'You said: HEMLOCK-ORPHAN', '2026-11-19T09:00:00+00:00')")     # a session that's gone
    store.db.execute("INSERT INTO log (session_id, at, kind, reason, payload) VALUES "
                     "(99, '2026-11-19T09:00:00+00:00', 'rejected belief', 'x', 'HEMLOCK-ORPHAN')")
    agent.delete_session(1)
    assert store.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0
    assert b"HEMLOCK" not in file_bytes(tmp_path)


# --- The delete is all or nothing, and the text is gone even if compacting fails ---------------------------

def test_a_delete_that_fails_halfway_changes_nothing(make_agent, monkeypatch):
    agent = make_agent()
    store = agent.store
    s1, (t1, _) = add_session(store, [("user", "The code word is HEMLOCK."), ("agent", "ok")])
    store.append("beliefs", {"text": "The code word is HEMLOCK", "confidence": 0.9, "evidence": json.dumps([t1]),
                             "session_id": s1, "flag": None})
    before = list(store.db.iterdump())

    def disk_full(session_id):
        raise sqlite3.OperationalError("disk full (planted by the test)")
    monkeypatch.setattr(store, "forget_session", disk_full)
    assert isinstance(outcome(agent.delete_session, s1), sqlite3.OperationalError)
    assert list(store.db.iterdump()) == before and not store.db.in_transaction


def test_deleted_text_is_overwritten_even_where_sqlite_would_not_and_vacuum_fails(tmp_path, monkeypatch):
    real_connect = sqlite3.connect

    def secure_delete_off(*args, **kwargs):          # a SQLite build whose default leaves deleted text in place
        db = real_connect(*args, **kwargs)
        db.execute("PRAGMA secure_delete=OFF")
        return db
    monkeypatch.setattr("starter_agent.store.sqlite3.connect", secure_delete_off)
    store = Store(str(tmp_path / "agent.db"), Clock())
    assert store.db.execute("PRAGMA secure_delete").fetchone()[0] == 1
    agent = Agent(store, FakeModel(), HashEmbedder(), Config(db=store.path))
    add_session(store, [("user", "My locker code is MARIGOLD-ORCHARD-9917."), ("agent", "ok")])
    store.db = Fails(store.db, "VACUUM")              # the disk fills up as the file is compacted
    counts = outcome(agent.delete_session, 1)
    assert isinstance(counts, dict) and counts["turns_deleted"] == 2
    assert "couldn't be compacted" in agent.notes[-1]
    assert b"MARIGOLD" not in file_bytes(tmp_path)


# --- A failed write leaves no transaction open, so later writes still land ---------------------------------

@pytest.mark.parametrize("step", ["COMMIT", "BEGIN IMMEDIATE"])
def test_a_failed_write_leaves_nothing_open_and_the_next_writes_land(make_agent, tmp_path, step):
    agent = make_agent(fn=lambda messages: "ok")
    store = agent.store
    real, store.db = store.db, Fails(store.db, step, sqlite3.OperationalError("database is locked (planted)"))
    assert isinstance(outcome(bdi.add_desire, store, "Weigh the mice"), sqlite3.OperationalError)
    store.db = real
    assert not store.db.in_transaction and store.depth == 0
    agent.handle("hello")
    again = Store(str(tmp_path / "agent.db"), Clock())          # what another program sees in the file
    assert [r["role"] for r in again.db.execute("SELECT role FROM turns")] == ["user", "agent"]


def test_the_outermost_transaction_takes_the_write_lock_before_it_reads(make_agent, tmp_path):
    """So another program can't change the file between what a transaction reads and what it writes."""
    store = make_agent().store
    other = sqlite3.connect(tmp_path / "agent.db", timeout=0)
    with store.transaction():
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.execute("BEGIN IMMEDIATE")
    other.execute("BEGIN IMMEDIATE")                   # and lets go of it when it's done
    other.rollback()


def test_ctrl_c_inside_a_transaction_rolls_it_back(make_agent):
    store = make_agent().store
    with pytest.raises(KeyboardInterrupt):
        with store.transaction():
            store.add_turn(store.start_session(), "user", "half")
            raise KeyboardInterrupt
    assert not store.db.in_transaction and store.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0


# --- A reply that can't be stored isn't half-kept, and a retry stores the message once ----------------------

@pytest.mark.parametrize("bad", [None, 42, ["a", "list"]])
def test_a_reply_that_isnt_text_keeps_nothing_and_the_retry_is_stored_once(make_agent, bad):
    replies = [bad, say("Hello.")]
    agent = make_agent(fn=lambda messages: replies.pop(0))
    assert isinstance(outcome(agent.handle, "Is anyone there?"), ModelError)
    assert roles(agent.store) == []
    agent.handle("Is anyone there?")
    assert roles(agent.store) == ["user", "agent"]


def test_a_block_nested_too_deep_is_logged_and_the_reply_stands(make_agent):
    agent = make_agent(replies=["Hi.\n```json\n" + "[" * 100000 + "\n```"])
    assert str(outcome(agent.handle, "hello")).startswith("Hi.")
    assert roles(agent.store) == ["user", "agent"]
    assert [r["kind"] for r in agent.store.db.execute("SELECT kind FROM log")] == ["rejected"]


def test_ctrl_c_after_the_reply_arrives_keeps_nothing(make_agent, monkeypatch):
    agent = make_agent(replies=[say("Hello.", beliefs=[{"text": "x", "confidence": 0.5, "evidence": [1]}])])

    def interrupted(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(bdi, "record_changes", interrupted)
    assert isinstance(outcome(agent.handle, "hi"), KeyboardInterrupt)
    assert roles(agent.store) == [] and bdi.current_beliefs(agent.store) == []


def test_a_broken_character_in_a_message_or_reply_is_repaired_not_a_crash(make_agent):
    agent = make_agent(replies=[say("half an emoji \ud83d", beliefs=[{"text": "odd \udcff byte", "confidence": 0.5,
                                                                     "evidence": [1]}])])
    assert outcome(agent.handle, "caf\udce9") == "half an emoji ?"
    assert [t["text"] for t in agent.store.turns(1)] == ["caf?", "half an emoji ?"]
    assert [b["text"] for b in bdi.current_beliefs(agent.store)] == ["odd ? byte"]


# --- Curiosity: a failed read is logged, and tried again until it reads ------------------------------------

@pytest.mark.parametrize("failure", [OSError("connection refused"), ValueError("not JSON"), KeyboardInterrupt()])
def test_curiosity_that_fails_is_logged_and_tried_again_at_the_next_session_end(make_agent, failure):
    up = []

    def model(messages):
        if messages[0]["content"] == curiosity.SYSTEM and not up:
            raise failure
        return demo_brain(messages)
    agent = make_agent(fn=model)
    agent.handle("why do the mice lose weight?")
    first = outcome(agent.end_session)
    assert agent.store.open_session() is None                          # the session ends anyway
    assert curiosity.open_questions(agent.store) == []
    if not isinstance(failure, KeyboardInterrupt):
        assert first == [] and "tries again" in agent.notes[-1]
        assert [r["kind"] for r in agent.store.db.execute("SELECT kind FROM log")] == ["curiosity failed"]
    up.append(True)
    assert len(agent.end_session()) == 1                               # no open session: it reads the old one
    assert [q["text"] for q in curiosity.open_questions(agent.store)] == ["why do the mice lose weight?"]
    assert agent.end_session() == []                                   # and only once


def test_a_session_with_no_agent_turn_is_marked_read_without_a_model_call(make_agent):
    agent = make_agent(fn=lambda messages: '{"questions": []}')
    sid = agent.store.start_session()
    agent.store.add_turn(sid, "user", "hello")                         # the reply never came
    agent.end_session()
    assert agent.model.calls == [] and agent.store.unread_sessions() == []


def test_a_file_from_the_first_release_gets_the_new_column_and_no_old_session_is_read_again(tmp_path):
    db = sqlite3.connect(tmp_path / "agent.db")
    db.executescript("CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, ended_at TEXT, "
                     "title TEXT); CREATE TABLE turns (id INTEGER PRIMARY KEY); CREATE TABLE beliefs (id INTEGER);"
                     "INSERT INTO sessions (started_at, ended_at) VALUES ('2026-11-19T09:00:00+00:00', "
                     "'2026-11-19T09:30:00+00:00');")
    db.commit()
    db.close()
    store = Store(str(tmp_path / "agent.db"), Clock())
    assert store.unread_sessions() == [] and store.session(1)["read_at"] == "2026-11-19T09:30:00+00:00"


# --- A file that isn't Kindling's is left alone ---------------------------------------------------------------

def test_a_file_that_isnt_kindlings_memory_is_left_alone(tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("my notes")
    notes.chmod(0o644)
    other = tmp_path / "library.sqlite"
    db = sqlite3.connect(other)
    db.execute("CREATE TABLE books (title TEXT)")
    db.commit()
    db.close()
    for path in (notes, other, tmp_path):
        before = (path.stat().st_mode, path.read_bytes() if path.is_file() else None)
        stopped = outcome(Store, str(path), Clock())
        assert isinstance(stopped, SystemExit), path
        assert (path.stat().st_mode, path.read_bytes() if path.is_file() else None) == before
