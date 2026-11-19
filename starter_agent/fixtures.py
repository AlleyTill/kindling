"""Small made-up data for each part, so each one runs alone: offline, in memory, no real data.

    python -m starter_agent.fixtures              # all three parts
    python -m starter_agent.fixtures recall       # or: bdi, curiosity

Each function builds a fresh in-memory store from its made-up sessions, runs ONE part on it, and returns
what that part did. tests/test_fixtures.py checks each part this way. Change the data here to try a part
on a case of your own before you wire a change into the loop.
"""
import json
import sys
from datetime import datetime, timedelta, timezone

from . import bdi, curiosity
from .config import Config
from .embed import HashEmbedder
from .models import FakeModel
from .recall import Memory
from .store import Store


def step_clock():
    """A clock that moves one second per reading, so every run gives the same times."""
    now = [datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)]

    def clock():
        now[0] += timedelta(seconds=1)
        return now[0]
    return clock


def fresh_store(sessions):
    """An in-memory store holding these finished sessions, each a list of (role, text)."""
    store = Store(":memory:", step_clock())
    for turns in sessions:
        sid = store.start_session()
        for role, text in turns:
            store.add_turn(sid, role, text)
        store.end_session(sid)
    return store


# --- Recall: four made-up sessions, and one of them holds the answer -----------------------------------
RECALL_SESSIONS = [
    [("user", "My sourdough starter smells of nail polish."), ("agent", "That's acetone: feed it more often.")],
    [("user", "The greenhouse reads 31 degrees at noon."), ("agent", "Open the vents before noon.")],
    [("user", "My cat is called Biscuit and she hates the vacuum."), ("agent", "Noted: Biscuit, no vacuum.")],
    [("user", "Which bicycle gear is best for hills?"), ("agent", "A low one: the small chainring.")],
]
RECALL_QUERY = "What is my cat called?"


def recall_alone(budget_tokens=60):
    """Recall alone: room for one session, so it has to pick the one about the cat (session 3)."""
    return Memory(fresh_store(RECALL_SESSIONS), HashEmbedder()).recall(RECALL_QUERY, budget_tokens)


# --- BDI: one session, two desires, and the JSON block a model might send ------------------------------
BDI_SESSIONS = [[("user", "We weigh the mice on Mondays."), ("agent", "Noted.")]]          # turns 1 and 2
BDI_DESIRES = [("Write the methods section", 2), ("Replicate figure 3", 1)]
BDI_BLOCK = json.dumps({"beliefs": [
    {"text": "The mice are weighed on Mondays", "confidence": 0.8, "evidence": [1]},        # kept
    {"text": "The mice are weighed daily", "confidence": 0.9, "evidence": [99]},            # no turn 99
    {"text": "The mice are fine", "confidence": 1.5, "evidence": [1]}],                     # over 1
    "intention_done": []})


def bdi_alone():
    """Deliberate and Record alone, with no model: pick the intention, then check a block of changes."""
    store = fresh_store(BDI_SESSIONS)
    for text, rank in BDI_DESIRES:
        bdi.add_desire(store, text, rank)
    intention = bdi.pick_intention(store)
    bdi.record_changes(store, BDI_BLOCK, store.turn_ids([1]), None, intention)
    return {"intention": intention["text"],
            "beliefs": [b["text"] for b in bdi.current_beliefs(store)],
            "rejected": [row[0] for row in store.db.execute("SELECT reason FROM log ORDER BY id")]}


# --- Curiosity: one session, and the reply a model might give when it ends ------------------------------
CURIOSITY_SESSIONS = [[("user", "The loss spiked at step 400."),                       # turn 1
                       ("agent", "I'm not sure why the loss spikes at step 400."),     # turn 2
                       ("user", "And the learning rate?"),                             # turn 3
                       ("agent", "The warm-up ends at step 400.")]]                    # turn 4
CURIOSITY_REPLY = json.dumps({"questions": [
    {"text": "Why does the loss spike at step 400?", "quote": "not sure why the loss spikes", "turn": 2},
    {"text": "Is the data shuffled?", "quote": "the data is not shuffled", "turn": 2},  # not in turn 2
    {"text": "What did the student see?", "quote": "The loss spiked", "turn": 1}]})    # the student's words


def curiosity_alone():
    """Curiosity alone: its one call (a fake model with a scripted reply) over a finished session."""
    store = fresh_store(CURIOSITY_SESSIONS)
    curiosity.run(store, FakeModel([CURIOSITY_REPLY]), HashEmbedder(), Config(db=":memory:"), 1)
    return {"queued": [(q["text"], q["quote"], q["source_turn"]) for q in curiosity.open_questions(store)],
            "rejected": [row[0] for row in store.db.execute("SELECT reason FROM log ORDER BY id")]}


PARTS = {"recall": recall_alone, "bdi": bdi_alone, "curiosity": curiosity_alone}


def main(names):
    for name in names or PARTS:
        if name not in PARTS:
            print(f"No fixture called {name!r}: use {', '.join(PARTS)}.")
            return 1
        print(f"--- {name}, alone, on made-up data ---")
        result = PARTS[name]()
        if name == "recall":
            for session, score in result.sessions:
                print(f"session {session}  score {score:.4f}  ({result.scored_by})")
            print(result.text)
        else:
            print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
