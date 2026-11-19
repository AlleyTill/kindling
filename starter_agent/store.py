"""One SQLite file holds everything the agent knows (plan section 2). Open it with any SQLite viewer.

The rule across parts (plan section 3): a record is changed by appending a new row that points at the
old one (`supersedes`), never by editing it in place, so you can see how the agent got where it is.
All versions of one record share a `key`: the id of its first version.

The exceptions, said out loud:
- a session's `title` (its first message) and `ended_at` are filled in place;
- deleting a session removes what it taught the agent (see Agent.delete_session). Your word wins;
- a message the model never answered is taken back (take_back), so a retry isn't stored twice.
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (        -- AUTOINCREMENT: a deleted id is never handed out again,
    id INTEGER PRIMARY KEY AUTOINCREMENT,     -- so an old record never points at a new, unrelated turn
    started_at TEXT, ended_at TEXT, title TEXT);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, n INTEGER, role TEXT, text TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS vectors (          -- one embedding per user turn, per embedder
    turn_id INTEGER, embedder TEXT, vec BLOB, PRIMARY KEY (turn_id, embedder));
CREATE TABLE IF NOT EXISTS beliefs (         -- evidence: JSON list of turn ids
    id INTEGER PRIMARY KEY, key INTEGER, supersedes INTEGER, at TEXT,
    text TEXT, confidence REAL, evidence TEXT,
    session_id INTEGER, flag TEXT);           -- session_id: the session this version was written in
CREATE TABLE IF NOT EXISTS desires (
    id INTEGER PRIMARY KEY, key INTEGER, supersedes INTEGER, at TEXT,
    text TEXT, rank INTEGER, source TEXT, live INTEGER,
    question_id INTEGER, source_turn INTEGER, flag TEXT);
CREATE TABLE IF NOT EXISTS intentions (
    id INTEGER PRIMARY KEY, key INTEGER, supersedes INTEGER, at TEXT,
    text TEXT, serves INTEGER, status TEXT, evidence TEXT, flag TEXT);  -- serves: a desire's key
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY, key INTEGER, supersedes INTEGER, at TEXT,
    text TEXT, quote TEXT, source_turn INTEGER, session_id INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS log (              -- what was dropped, and why
    id INTEGER PRIMARY KEY, session_id INTEGER, at TEXT, kind TEXT, reason TEXT, payload TEXT);
"""

VERSIONED = ("beliefs", "desires", "intentions", "questions")
META = ("id", "key", "supersedes", "at")


def make_private(path):
    """Your agent's memory is yours. Where the system has file modes (Linux, macOS), a folder created here
    is readable by you alone (700), and so is the file (600). On Windows your user folder's permissions apply."""
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder, mode=0o700)
        if os.name == "posix":
            os.chmod(folder, 0o700)
    os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))     # created private, never briefly open
    if os.name == "posix":
        os.chmod(path, 0o600)


def local(timestamp):
    """A stored time (UTC) as your computer's local time, for showing: '2026-11-19 01:00'."""
    return datetime.fromisoformat(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")


class Store:
    def __init__(self, path, clock):
        self.path = path
        self.clock = clock                     # a function returning the time now, as a datetime
        if path != ":memory:":
            make_private(path)
        self.db = sqlite3.connect(path, isolation_level=None)   # each statement commits by itself
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def now(self):
        return self.clock().isoformat(timespec="seconds")

    @contextmanager
    def transaction(self):
        """Everything inside is written together or not at all. One inside another is a savepoint."""
        self.db.execute("SAVEPOINT tx")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK TO tx")
            self.db.execute("RELEASE tx")
            raise
        self.db.execute("RELEASE tx")

    # --- sessions and turns -------------------------------------------------------------------

    def start_session(self):
        return self.db.execute("INSERT INTO sessions (started_at) VALUES (?)", (self.now(),)).lastrowid

    def open_session(self):
        return self.db.execute(
            "SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1").fetchone()

    def end_session(self, session_id):
        self.db.execute("UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL",
                        (self.now(), session_id))

    def sessions(self):
        return self.db.execute("SELECT * FROM sessions ORDER BY id").fetchall()

    def session(self, session_id):
        return self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()

    def add_turn(self, session_id, role, text):
        n = self.db.execute("SELECT COUNT(*) FROM turns WHERE session_id=?", (session_id,)).fetchone()[0]
        if role == "user":
            self.db.execute("UPDATE sessions SET title=? WHERE id=? AND title IS NULL",
                            (text[:60], session_id))
        return self.db.execute(
            "INSERT INTO turns (session_id, n, role, text, at) VALUES (?, ?, ?, ?, ?)",
            (session_id, n + 1, role, text, self.now())).lastrowid

    def take_back(self, turn_id):
        """Remove your newest turn, which got no reply, and its embedding (and the title it gave)."""
        session_id = self.db.execute("SELECT session_id FROM turns WHERE id=?", (turn_id,)).fetchone()[0]
        with self.transaction():
            self.db.execute("DELETE FROM vectors WHERE turn_id=?", (turn_id,))
            self.db.execute("DELETE FROM turns WHERE id=?", (turn_id,))
            self.db.execute("UPDATE sessions SET title=NULL WHERE id=? AND NOT EXISTS "
                            "(SELECT 1 FROM turns WHERE session_id=? AND role='user')", (session_id, session_id))

    def turns(self, session_id):
        return self.db.execute("SELECT * FROM turns WHERE session_id=? ORDER BY n", (session_id,)).fetchall()

    def turn_ids(self, session_ids):
        marks = ",".join("?" * len(session_ids))
        rows = self.db.execute(f"SELECT id FROM turns WHERE session_id IN ({marks})", list(session_ids))
        return {row[0] for row in rows}

    def forget_session(self, session_id):
        """Delete a session's turns, their embeddings, its log rows and the session itself."""
        self.db.execute("DELETE FROM vectors WHERE turn_id IN (SELECT id FROM turns WHERE session_id=?)",
                        (session_id,))
        self.db.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        self.db.execute("DELETE FROM log WHERE session_id=?", (session_id,))
        self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # --- versioned records: beliefs, desires, intentions, questions ---------------------------

    def append(self, table, fields, old=None):
        """Write a new version. With `old`, the new row supersedes it and shares its key. The row and its
        key are written in one transaction, so no row is ever left without a key."""
        assert table in VERSIONED
        fields = dict(fields, supersedes=old["id"] if old else None, at=self.now())
        names = ", ".join(fields)
        marks = ", ".join("?" * len(fields))
        with self.transaction():
            new_id = self.db.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})",
                                     list(fields.values())).lastrowid
            self.db.execute(f"UPDATE {table} SET key=? WHERE id=?", (old["key"] if old else new_id, new_id))
        return new_id

    def revise(self, table, old, **changes):
        """Append a copy of `old` with some fields changed."""
        fields = {name: old[name] for name in old.keys() if name not in META}
        fields.update(changes)
        return self.append(table, fields, old)

    def get(self, table, row_id):
        return self.db.execute(f"SELECT * FROM {table} WHERE id=?", (row_id,)).fetchone()

    def current(self, table, where="1", args=()):
        """The latest version of every record (that also matches `where`), oldest record first."""
        return self.db.execute(
            f"SELECT * FROM {table} WHERE id IN (SELECT MAX(id) FROM {table} GROUP BY key) "
            f"AND ({where}) ORDER BY key", args).fetchall()

    def history(self, table, row_id):
        """Every version of the record that row `row_id` belongs to, oldest first."""
        return self.db.execute(
            f"SELECT * FROM {table} WHERE key=(SELECT key FROM {table} WHERE id=?) ORDER BY id",
            (row_id,)).fetchall()

    def latest(self, table, row_id):
        versions = self.history(table, row_id)
        return versions[-1] if versions else None

    # --- the log ------------------------------------------------------------------------------

    def log(self, session_id, kind, reason, payload=None):
        self.db.execute("INSERT INTO log (session_id, at, kind, reason, payload) VALUES (?, ?, ?, ?, ?)",
                        (session_id, self.now(), kind, reason, payload))
