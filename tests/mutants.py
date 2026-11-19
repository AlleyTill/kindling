"""How we know the tests can fail: plant one bug at a time and check that a test catches it.

    python tests/mutants.py [WORK_FOLDER]

The code is copied once to WORK_FOLDER (a frozen snapshot; default: a new temporary folder). The
unchanged snapshot must pass every test first. Then each mutant below gets its own copy of the
snapshot, with one promise broken by an exact text replacement, and the tests run on it. A mutant
counts as KILLED only when at least one test fails with an AssertionError; a crash proves nothing
about the test, so it counts as INVALID. Anything else SURVIVED, and needs a new test.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (the promise, the file, [(exact text, its replacement), ...])
MUTANTS = [
    # --- The shape (plan sections 2 and 3) ---------------------------------------------------------
    ("Act makes ONE model call per turn", "starter_agent/agent.py", [(
        "        raw = self.model.complete(",
        "        self.model.complete([{'role': 'user', 'content': message}])\n        raw = self.model.complete(")]),
    ("the tests never reach a real model", "starter_agent/models.py", [(
        'return FakeModel()\n    if config.backend == "ollama"',
        'return Ollama("", "http://127.0.0.1:9")\n    if config.backend == "ollama"')]),
    ("everything is in one SQLite file", "starter_agent/recall.py", [(
        "    if missing:\n",
        "    open(store.path + '.cache', 'a').close()\n    if missing:\n")]),
    ("a change appends a new row; it never edits one in place", "starter_agent/store.py", [(
        "        return self.append(table, fields, old)",
        "        self.db.execute(f\"UPDATE {table} SET \" + \", \".join(f\"{k}=?\" for k in changes) + \" WHERE id=?\",\n"
        "                        [*changes.values(), old['id']])\n        return old['id']")]),
    ("all versions of a record share its key", "starter_agent/store.py", [(
        "(old[\"key\"] if old else new_id, new_id)", "(new_id, new_id)")]),
    ("the current version is the latest", "starter_agent/store.py", [(
        "SELECT MAX(id) FROM {table} GROUP BY key", "SELECT MIN(id) FROM {table} GROUP BY key")]),
    ("a session ends after the idle gap", "starter_agent/agent.py", [(
        "timedelta(minutes=self.config.idle_minutes)", "timedelta(minutes=self.config.idle_minutes * 2)")]),
    ("the default recall budget is 8,000 tokens", "starter_agent/config.py", [(
        "recall_budget_tokens: int = 8000", "recall_budget_tokens: int = 24000")]),
    ("the default embedder needs nothing installed", "starter_agent/config.py", [(
        'embedder: str = "hash"', 'embedder: str = "bge-small"')]),
    # --- Recall (section 5) --------------------------------------------------------------------------
    ("RRF: dense weighted 2", "starter_agent/recall.py", [(
        "fused[i] += 2 / (k + rank)", "fused[i] += 1 / (k + rank)")]),
    ("RRF: BM25 weighted 1", "starter_agent/recall.py", [(
        "fused[i] += 1 / (k + rank)", "fused[i] += 2 / (k + rank)")]),
    ("RRF: k = 60", "starter_agent/recall.py", [("def fuse(dense, lexical, k=60):", "def fuse(dense, lexical, k=1):")]),
    ("RRF: BM25 is part of the fusion", "starter_agent/recall.py", [(
        "    for i, rank in ranks(lexical, positive_only=True).items():\n        fused[i] += 1 / (k + rank)\n", "")]),
    ("RRF: the two rankings go in the right places", "starter_agent/recall.py", [(
        "fuse(dense, lexical).items()", "fuse(lexical, dense).items()")]),
    ("a session scores its best two turns, not its best one", "starter_agent/recall.py", [(
        "sorted(s, reverse=True)[:2]", "sorted(s, reverse=True)[:1]")]),
    ("a session scores its best two turns, not all of them", "starter_agent/recall.py", [(
        "sorted(s, reverse=True)[:2]", "sorted(s, reverse=True)")]),
    ("only the user's turns are indexed", "starter_agent/recall.py", [
        ("WHERE role='user' AND id NOT IN", "WHERE role IN ('user', 'agent') AND id NOT IN"),
        ("WHERE t.role = 'user' AND", "WHERE t.role IN ('user', 'agent') AND")]),
    ("the pack never exceeds the budget", "starter_agent/recall.py", [("if used + cost <= budget:", "if used <= budget:")]),
    ("a session is never cut in half", "starter_agent/recall.py", [(
        "        block = render_session(store, sid)\n",
        "        block = render_session(store, sid)[:max(0, budget - used) * 4 - 8]\n")]),
    ("the pack runs oldest to newest", "starter_agent/recall.py", [("    chosen.sort()", "    pass")]),
    # --- BDI (section 4) -----------------------------------------------------------------------------
    ("the intention serves the highest-ranked desire", "starter_agent/bdi.py", [(
        'key=lambda d: (d["rank"], d["key"])', 'key=lambda d: (-d["rank"], d["key"])')]),
    ("a done intention is passed over", "starter_agent/bdi.py", [(
        '        if found and found[0]["status"] == "done":\n            continue\n', "")]),
    ("only live desires are served", "starter_agent/bdi.py", [(
        "\"live=1 AND flag IS NOT 'removed'\"", "\"flag IS NOT 'removed'\"")]),
    ("only the student creates desires; the model can't", "starter_agent/bdi.py", [(
        "    done = changes.get(\"intention_done\")",
        "    for text in changes.get('desires') or []:\n        add_desire(store, text, source='model')\n"
        "    done = changes.get(\"intention_done\")")]),
    ("evidence must be turns that exist", "starter_agent/bdi.py", [(
        "type(t) is not int or t not in allowed", "type(t) is not int")]),
    ("evidence must be turns the model was shown", "starter_agent/agent.py", [(
        "allowed = self.store.turn_ids(recalled + [sid])",
        "allowed = {row[0] for row in self.store.db.execute('SELECT id FROM turns')}")]),
    ("a change with no evidence is rejected", "starter_agent/bdi.py", [(
        "if not isinstance(evidence, list) or not evidence:", "if not isinstance(evidence, list):")]),
    ("true is not turn 1", "starter_agent/bdi.py", [("type(t) is not int or", "not isinstance(t, int) or")]),
    ("confidence is at most 1", "starter_agent/bdi.py", [("not 0 <= confidence <= 1", "not 0 <= confidence")]),
    ("confidence true is not 1", "starter_agent/bdi.py", [(
        "if isinstance(confidence, bool) or not isinstance", "if not isinstance")]),
    ("a rejected change is logged", "starter_agent/bdi.py", [(
        '            store.log(session_id, "rejected belief", problem, json.dumps(change))\n', "")]),
    ("a change supersedes only the current version", "starter_agent/bdi.py", [(
        'if latest is None or latest["id"] != old or', "if latest is None or")]),
    ("supersedes true is not belief 1", "starter_agent/bdi.py", [(
        "if type(old) is int else None", "if isinstance(old, int) else None")]),
    ("a belief row names its evidence", "starter_agent/bdi.py", [(
        '"evidence": json.dumps(change["evidence"])', '"evidence": "[]"')]),
    ("done needs real evidence", "starter_agent/bdi.py", [(
        "problem = check_evidence(done, allowed) or (None", "problem = (None")]),
    # --- Deleting a session (section 4) ------------------------------------------------------------
    ("a belief resting only on the session is removed", "starter_agent/bdi.py", [(
        "if evidence and all(t in gone for t in evidence):", "if False:")]),
    ("the marker keeps no text", "starter_agent/bdi.py", [(
        '(REMOVED.format(what="a belief", date=date), row["id"])',
        '(REMOVED.format(what="a belief", date=date) + ": " + row["text"], row["id"])')]),
    ("a belief with other evidence keeps only that evidence", "starter_agent/bdi.py", [(
        'kept = [t for t in json.loads(survivors[-1]["evidence"]) if t not in gone]',
        'kept = json.loads(survivors[-1]["evidence"])')]),
    ("a belief that lost evidence is flagged for review", "starter_agent/bdi.py", [(
        '"evidence": json.dumps(kept), "flag": "review"}', '"evidence": json.dumps(kept), "flag": None}')]),
    ("a change made only in the session is undone", "starter_agent/bdi.py", [(
        'survivors = [v for v in store.history("beliefs", row["id"]) if v["flag"] != "removed"]',
        'survivors = [v for v in [store.latest("beliefs", row["id"])] if v["flag"] != "removed"]')]),
    ("a done intention loses the session's evidence", "starter_agent/bdi.py", [("        if kept != evidence:\n", "        if False:\n")]),
    ("a desire adopted from the session is removed", "starter_agent/bdi.py", [(
        "WHERE source_turn IN ({marks})", "WHERE 0 AND source_turn IN ({marks})")]),
    ("the intention serving it is removed", "starter_agent/bdi.py", [(
        "UPDATE intentions SET text=?, flag='removed' WHERE serves=?", "SELECT ?, ?")]),
    ("questions from the session are removed", "starter_agent/curiosity.py", [(
        "    if not gone:\n        return 0\n", "    return 0\n")]),
    ("the session's log rows are deleted", "starter_agent/store.py", [(
        '        self.db.execute("DELETE FROM log WHERE session_id=?", (session_id,))\n', "")]),
    ("the session's embeddings are deleted", "starter_agent/store.py", [(
        "DELETE FROM vectors WHERE turn_id IN", "SELECT * FROM vectors WHERE turn_id IN")]),
    ("deleted text doesn't linger in free space", "starter_agent/agent.py", [(
        'self.store.db.execute("VACUUM")', "pass")]),
    ("a deleted id is never handed out again", "starter_agent/store.py", [(
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id", "id INTEGER PRIMARY KEY, session_id")]),
    # --- Curiosity (section 6) -----------------------------------------------------------------------
    ("curiosity never runs mid-conversation", "starter_agent/agent.py", [(
        "        return reply\n",
        "        curiosity.run(self.store, self.model, self.embedder, self.config, sid)\n        return reply\n")]),
    ("an idle session is ended and read before `curious list`", "starter_agent/cli.py", [(
        "        agent.end_if_idle()                  # so yesterday's session has been read\n", "")]),
    ("a session is read once", "starter_agent/agent.py", [(
        "        session = self.store.open_session()\n        if not session:\n            return []",
        "        session = self.store.open_session() or self.store.sessions()[-1]\n        if not session:\n"
        "            return []")]),
    ("curiosity reads only the agent's turns", "starter_agent/curiosity.py", [(
        'mine = [t for t in store.turns(session_id) if t["role"] == "agent"]', "mine = list(store.turns(session_id))")]),
    ("a quote is at most 20 words", "starter_agent/curiosity.py", [(
        "2 <= len(quote.split()) <= 20", "2 <= len(quote.split()) <= 21")]),
    ("a quote is at least 2 words", "starter_agent/curiosity.py", [(
        "2 <= len(quote.split()) <= 20", "1 <= len(quote.split()) <= 20")]),
    ("a quote is whole words", "starter_agent/curiosity.py", [(
        'if not re.search(r"(?<!\\w)" + re.escape(quote) + r"(?!\\w)", my_turns[turn]):',
        "if quote not in my_turns[turn]:")]),
    ("a quote is found verbatim", "starter_agent/curiosity.py", [(
        'if not re.search(r"(?<!\\w)" + re.escape(quote) + r"(?!\\w)", my_turns[turn]):', "if False:")]),
    ("dedupe against dropped questions", "starter_agent/curiosity.py", [(
        "status IN ('open', 'dropped')", "status IN ('open')")]),
    ("dedupe against open questions", "starter_agent/curiosity.py", [(
        "status IN ('open', 'dropped')", "status IN ('dropped')")]),
    ("at most 20 open questions", "starter_agent/curiosity.py", [(
        "while len(waiting) >= limit:", "while len(waiting) > limit:")]),
    ("an expired question is kept, not deleted", "starter_agent/curiosity.py", [(
        'store.revise("questions", waiting.pop(0), status="expired")',
        'store.db.execute("DELETE FROM questions WHERE key=?", (waiting.pop(0)["key"],))')]),
    ("the OLDEST open question expires", "starter_agent/curiosity.py", [("waiting.pop(0)", "waiting.pop()")]),
    ("automatic adoption is off by default", "starter_agent/config.py", [(
        "auto_adopt: bool = False", "auto_adopt: bool = True")]),
    ("the auto_adopt switch works", "starter_agent/curiosity.py", [("        if config.auto_adopt:", "        if False:")]),
    ("an adopted desire names its question and turn", "starter_agent/curiosity.py", [(
        'question_id=question["key"], source_turn=question["source_turn"]', "question_id=None, source_turn=None")]),
    ("only an open question can be adopted", "starter_agent/curiosity.py", [(
        '    question = store.latest("questions", question_id)\n    if question is None or question["status"] != "open":\n'
        '        return None\n    desire_id',
        '    question = store.latest("questions", question_id)\n    if question is None:\n        return None\n    desire_id')]),
    # --- The command line --------------------------------------------------------------------------------
    ("`desire rank ID N` sets rank N", "starter_agent/cli.py", [(
        "bdi.rerank_desire(store, desire_id, new_rank)", "bdi.rerank_desire(store, desire_id, 1)")]),
    ("`intentions` marks the current pick", "starter_agent/cli.py", [(
        "current = bdi.pick_intention(agent.store)", "current = None")]),
    ("`session delete` asks first", "starter_agent/cli.py", [("if not a.yes and ask(", "if False and ask(")]),
    ("`beliefs history` shows every version", "starter_agent/cli.py", [(
        'rows = agent.store.history("beliefs", a.id) if a.id is not None else []',
        'rows = [agent.store.latest("beliefs", a.id)] if a.id is not None else []')]),
    ("`curious drop` drops", "starter_agent/cli.py", [(
        'act = curiosity.drop if a.action == "drop"', 'act = curiosity.adopt if a.action == "drop"')]),
]

PLUGIN = '''
import json, os
FAILED = []
def pytest_runtest_makereport(item, call):
    if call.excinfo is not None:
        FAILED.append([item.nodeid, call.when, call.excinfo.typename])
def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["MUTANT_REPORT"], "w") as f:
        json.dump({"exit": int(exitstatus), "ran": session.testscollected, "failed": FAILED}, f)
'''


def run_tests(folder):
    """{"exit": pytest's exit code, "ran": tests collected, "failed": [(test, phase, exception name)]}."""
    report = folder / "failures.json"
    report.unlink(missing_ok=True)                 # never read an old report as this run's
    env = {**os.environ, "MUTANT_REPORT": str(report), "PYTHONPATH": str(folder.parent / "plugin"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "mutant_plugin"],
                   cwd=folder, env=env, capture_output=True,
                   timeout=600)    # the suite takes ~20 s; a mutant that loops forever stops here
    return json.loads(report.read_text()) if report.exists() else {"exit": -1, "ran": 0, "failed": []}


def mutate(snapshot, number, file, swaps):
    folder = snapshot.parent / f"mutant_{number:02d}"
    shutil.copytree(snapshot, folder, ignore=shutil.ignore_patterns("failures.json"))
    path = folder / file
    text = path.read_text()
    for old, new in swaps:
        if text.count(old) != 1:
            return None, f"the text to replace occurs {text.count(old)} times in {file}"
        text = text.replace(old, new)
    path.write_text(text)
    return folder, None


def verdict(run):
    """KILLED only if pytest finished normally and a test failed by assertion."""
    failures = run["failed"]
    killers = sorted({test.split("::")[-1] for test, phase, kind in failures
                      if phase == "call" and kind == "AssertionError"})
    if run["exit"] == 1 and killers:
        return "KILLED", killers
    if run["exit"] == 0:
        return "SURVIVED", []
    return "INVALID", [f"pytest exit {run['exit']}"] + sorted({f"{t.split('::')[-1]} ({k})" for t, _, k in failures})


def main(work=None):
    work = Path(work or tempfile.mkdtemp(prefix="mutants-"))
    work.mkdir(parents=True, exist_ok=True)
    snapshot = work / "snapshot"
    shutil.copytree(ROOT, snapshot, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.db"))
    (work / "plugin").mkdir(exist_ok=True)
    (work / "plugin" / "mutant_plugin.py").write_text(PLUGIN)
    baseline = run_tests(snapshot)
    if baseline["exit"] != 0 or not baseline["ran"]:
        print("The unchanged code doesn't pass cleanly, so no mutant can be judged:", baseline)
        return 2
    print(f"Baseline: all {baseline['ran']} tests pass on the unchanged copy.")

    def one(numbered):
        number, (promise, file, swaps) = numbered
        folder, problem = mutate(snapshot, number, file, swaps)
        return (number, promise, file, "NOT APPLIED", [problem]) if problem else \
            (number, promise, file, *verdict(run_tests(folder)))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, enumerate(MUTANTS, start=1)))
    print("| # | The promise the bug breaks | File | Result | Failed by assertion |\n|---|---|---|---|---|")
    for number, promise, file, result, detail in results:
        print(f"| {number} | {promise} | {file.split('/')[-1]} | {result} | {', '.join(detail)} |")
    killed = sum(r[3] == "KILLED" for r in results)
    print(f"\n{killed} of {len(results)} mutants killed by an assertion. Work folder: {work}")
    return 0 if killed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
