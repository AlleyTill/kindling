"""Beliefs, desires and intentions (plan section 4, CC).

Kintsugi's BDI cut to the smallest version that runs; Kintsugi's design began in Thomas's Orion.
- Desires come from you (`desire add`), or from curiosity only when you say `curious adopt`.
- Each turn, the intention is picked in plain code: it serves your highest-ranked live desire.
- The model's one call returns a reply plus a JSON block of belief changes. A change is stored only
  if it cites real turns (from the recalled sessions or this session) and a confidence from 0 to 1;
  otherwise it is dropped and logged.
- A changed belief is a new row that supersedes the old one and names its evidence. There is no
  free-text "reason" field: the evidence is the reason, so nothing is ever recorded as "I decided".
"""
import json

from .models import parse_json

ACT_SYSTEM = """You are a research assistant with a memory of past sessions. Answer the student's last \
message, using what you recall, your current beliefs and your current intention where they help.

After your reply, add one fenced JSON block exactly like this:
```json
{"beliefs": [{"text": "...", "confidence": 0.7, "evidence": [12, 15], "supersedes": null}], "intention_done": []}
```
- "beliefs": only beliefs this turn creates or changes. "evidence" lists the turn ids (the numbers in \
[turn N]) the belief rests on. "confidence" is from 0 to 1. To change a belief, put its id in "supersedes".
- "intention_done": the turn ids that show your current intention is done, or [].
- If nothing changed, write {"beliefs": [], "intention_done": []}.
You may suggest a new ranking of the student's desires in your reply; only the student can change it."""

REMOVED = "{what} derived from a deleted session was removed on {date}"


# --- Desires: yours to write and rank ------------------------------------------------------------

def live_desires(store):
    """Your live desires, highest-ranked (rank 1) first."""
    rows = store.current("desires", "live=1 AND flag IS NOT 'removed'")
    return sorted(rows, key=lambda d: (d["rank"], d["key"]))


def next_rank(store):
    return max([d["rank"] for d in live_desires(store)], default=0) + 1


def add_desire(store, text, rank=None, source="student", question_id=None, source_turn=None):
    return store.append("desires", {
        "text": text, "rank": next_rank(store) if rank is None else rank, "source": source, "live": 1,
        "question_id": question_id, "source_turn": source_turn, "flag": None})


def rerank_desire(store, desire_id, rank):
    return store.revise("desires", store.latest("desires", desire_id), rank=rank)


def drop_desire(store, desire_id):
    return store.revise("desires", store.latest("desires", desire_id), live=0)


# --- Deliberate, step 1: pick the intention in plain code ----------------------------------------

def choose(store):
    """Read-only: (your highest-ranked live desire whose intention isn't done, its intention or None
    if none is written yet), or (None, None) when no desire is live."""
    for desire in live_desires(store):
        found = store.current("intentions", "serves=? AND flag IS NOT 'removed'", (desire["key"],))
        if found and found[0]["status"] == "done":
            continue
        return desire, (found[0] if found else None)
    return None, None


def pick_intention(store):
    """The current intention, written down first if it's new. None: no live desire, so just answer."""
    desire, intention = choose(store)
    if desire is None or intention is not None:
        return intention
    new_id = store.append("intentions", {"text": desire["text"], "serves": desire["key"],
                                         "status": "active", "evidence": "[]", "flag": None})
    return store.get("intentions", new_id)


def current_beliefs(store):
    return store.current("beliefs", "flag IS NOT 'removed'")


# --- Deliberate, step 2: what the model is handed ------------------------------------------------

def act_messages(pack, beliefs, intention, this_session):
    belief_lines = [f"#{b['id']} (confidence {b['confidence']:.2f}"
                    + (", flagged for the student's review" if b["flag"] == "review" else "")
                    + f"): {b['text']}" for b in beliefs]
    parts = [
        "RECALLED PAST SESSIONS (oldest first):\n" + (pack or "(none)"),
        "YOUR CURRENT BELIEFS:\n" + ("\n".join(belief_lines) or "(none)"),
        "YOUR CURRENT INTENTION:\n" + (f"#{intention['id']}: {intention['text']}" if intention
                                       else "(none: just answer)"),
        "THIS SESSION SO FAR:\n" + this_session,
    ]
    return [{"role": "system", "content": ACT_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# --- Deliberate, steps 3 and 4: check every change, record it honestly ---------------------------

def check_evidence(evidence, allowed):
    if not isinstance(evidence, list) or not evidence:
        return "no evidence: a change must cite the turn ids it rests on"
    bad = [t for t in evidence if type(t) is not int or t not in allowed]   # type(): True is not turn 1
    if bad:
        return f"evidence {bad} is not a turn in the recalled sessions or this session"
    return None


def check_belief(store, change, allowed):
    """None if the proposed change can be stored, else what's wrong with it."""
    if not isinstance(change, dict):
        return "not a JSON object"
    text, confidence = change.get("text"), change.get("confidence")
    if not isinstance(text, str) or not text.strip():
        return "no text"
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return f"confidence {confidence!r} is not a number from 0 to 1"
    problem = check_evidence(change.get("evidence"), allowed)
    if problem:
        return problem
    old = change.get("supersedes")
    if old is not None:
        latest = store.latest("beliefs", old) if type(old) is int else None
        if latest is None or latest["id"] != old or latest["flag"] == "removed":
            return f"supersedes {old!r}, which is not a current belief"
    return None


def record_changes(store, block, allowed, session_id, intention):
    """Store the valid belief changes from Act's JSON block; log the rest. Returns new belief ids."""
    if block is None:
        return []
    changes = parse_json(block)
    if not isinstance(changes, dict):
        store.log(session_id, "rejected", "the JSON block did not parse as an object", block)
        return []
    proposed = changes.get("beliefs") or []
    if not isinstance(proposed, list):
        store.log(session_id, "rejected", '"beliefs" is not a list', block)
        proposed = []
    stored = []
    for change in proposed:
        problem = check_belief(store, change, allowed)
        if problem:
            store.log(session_id, "rejected belief", problem, json.dumps(change))
            continue
        old = store.get("beliefs", change["supersedes"]) if change.get("supersedes") is not None else None
        stored.append(store.append("beliefs", {
            "text": change["text"].strip(), "confidence": float(change["confidence"]),
            "evidence": json.dumps(change["evidence"]), "session_id": session_id, "flag": None}, old))
    done = changes.get("intention_done")
    if done:
        problem = check_evidence(done, allowed) or (None if intention else "there is no current intention")
        if problem:
            store.log(session_id, "rejected intention_done", problem, json.dumps(done))
        else:
            store.revise("intentions", intention, status="done", evidence=json.dumps(done))
    return stored


# --- When you delete a session --------------------------------------------------------------------

def forget_turns(store, session_id, gone, date):
    """Remove what the BDI learned from session `session_id`, whose turns are `gone` (a set of turn ids).
    A cited turn that no longer exists (an earlier delete took it) counts as gone too.

    - Every belief version WRITTEN in the session becomes a content-free marker, whatever it cites: its
      words may come from that session. So does every version whose evidence is all gone.
    - A current belief that lost a version or some evidence falls back to its latest version that isn't
      a marker, keeping only the evidence that still exists, flagged for your review: the code can't tell
      whether its confidence still holds. With no such version left, the belief is just a marker.
    - A done intention whose evidence is gone keeps only its other evidence; with none, it's active again.
    - A desire adopted from a question raised in the session, and the intention serving it, become markers.
    """
    alive = {row[0] for row in store.db.execute("SELECT id FROM turns")} - set(gone)
    counts = {"belief_versions_removed": 0, "beliefs_flagged_for_review": 0, "desires_removed": 0}
    before = current_beliefs(store)
    for row in store.db.execute("SELECT * FROM beliefs WHERE flag IS NOT 'removed'").fetchall():
        evidence = json.loads(row["evidence"])
        if row["session_id"] == session_id or (evidence and not any(t in alive for t in evidence)):
            store.db.execute("UPDATE beliefs SET text=?, confidence=NULL, evidence='[]', session_id=NULL, "
                             "flag='removed' WHERE id=?", (REMOVED.format(what="a belief", date=date), row["id"]))
            counts["belief_versions_removed"] += 1
    for row in before:
        if row["session_id"] != session_id and all(t in alive for t in json.loads(row["evidence"])):
            continue                      # untouched by this delete
        survivors = [v for v in store.history("beliefs", row["id"]) if v["flag"] != "removed"]
        if survivors:                     # its latest version that isn't a marker
            back = survivors[-1]
            kept = [t for t in json.loads(back["evidence"]) if t in alive]
            store.append("beliefs", {"text": back["text"], "confidence": back["confidence"],
                                     "evidence": json.dumps(kept), "session_id": back["session_id"],
                                     "flag": "review"},     # its text came from back's session: say so
                         store.latest("beliefs", row["id"]))
            counts["beliefs_flagged_for_review"] += 1
    for row in store.current("intentions", "status='done' AND flag IS NOT 'removed'"):
        evidence = json.loads(row["evidence"])
        kept = [t for t in evidence if t in alive]
        if kept != evidence:
            store.revise("intentions", row, evidence=json.dumps(kept), status="done" if kept else "active")
    marks = ",".join("?" * len(gone))
    keys = [r[0] for r in store.db.execute(
        f"SELECT DISTINCT key FROM desires WHERE source_turn IN ({marks})", list(gone))] if gone else []
    for key in keys:
        store.db.execute("UPDATE desires SET text=?, live=0, flag='removed' WHERE key=?",
                         (REMOVED.format(what="a desire", date=date), key))
        store.db.execute("UPDATE intentions SET text=?, flag='removed' WHERE serves=?",
                         (REMOVED.format(what="an intention", date=date), key))
    counts["desires_removed"] = len(keys)
    return counts
