"""The loop (plan section 2):

    event (your message) -> RECALL -> DELIBERATE -> ACT -> RECORD
                              ^                              |
                              +---- CURIOSITY (between sessions)

RECALL      recall.py   the past sessions that matter, whole
DELIBERATE  bdi.py      pick the intention; gather the current beliefs
ACT         models.py   ONE model call: a reply plus a JSON block of belief changes
RECORD      store.py    the turns, and every belief change that passes bdi's checks
CURIOSITY   curiosity.py   one call when a session ends
"""
from datetime import datetime, timedelta, timezone

from . import bdi, curiosity
from .models import split_json
from .recall import recall, render_session


def utc_now():
    return datetime.now(timezone.utc)


class Agent:
    def __init__(self, store, model, embedder, config):
        self.store, self.model, self.embedder, self.config = store, model, embedder, config

    def end_if_idle(self):
        """Nothing runs in the background, so an idle session is ended the next time you use the agent."""
        session = self.store.open_session()
        if session:
            turns = self.store.turns(session["id"])
            last = datetime.fromisoformat(turns[-1]["at"] if turns else session["started_at"])
            if self.store.clock() - last > timedelta(minutes=self.config.idle_minutes):
                self.end_session()

    def session_id(self):
        """The open session; a new one if there's none, or if the open one sat idle too long."""
        self.end_if_idle()
        session = self.store.open_session()
        return session["id"] if session else self.store.start_session()

    def handle(self, message):
        """One turn of the loop. Returns the agent's reply."""
        sid = self.session_id()
        self.store.add_turn(sid, "user", message)

        # RECALL: past sessions only; this one is already in context below.
        pack, recalled = recall(self.store, self.embedder, message,
                                self.config.recall_budget_tokens, exclude=sid)

        # DELIBERATE: the intention is picked in code, never by the model.
        intention = bdi.pick_intention(self.store)
        beliefs = bdi.current_beliefs(self.store)

        # ACT: one call.
        raw = self.model.complete(bdi.act_messages(pack, beliefs, intention, render_session(self.store, sid)))
        reply, block = split_json(raw)

        # RECORD: evidence may cite only turns the model was shown.
        allowed = self.store.turn_ids(recalled + [sid])
        self.store.add_turn(sid, "agent", reply)
        bdi.record_changes(self.store, block, allowed, sid, intention)
        return reply

    def end_session(self):
        """End the open session and run curiosity once over it. Returns the new question ids."""
        session = self.store.open_session()
        if not session:
            return []
        self.store.end_session(session["id"])
        try:
            return curiosity.run(self.store, self.model, self.embedder, self.config, session["id"])
        except OSError as error:            # the model couldn't be reached: the session still ends
            self.store.log(session["id"], "curiosity failed", repr(error))
            return []

    def delete_session(self, session_id):
        """Your word wins (plan section 4): remove the session and what it alone taught the agent."""
        if self.store.session(session_id) is None:
            return None
        gone = self.store.turn_ids([session_id])
        date = self.store.clock().astimezone().date().isoformat()      # your local date
        with self.store.transaction():
            counts = bdi.forget_turns(self.store, session_id, gone, date)
            counts["question_rows_removed"] = curiosity.forget_turns(self.store, gone, date)
            self.store.forget_session(session_id)
            counts["turns_deleted"] = len(gone)
            self.store.log(None, "deleted", f"session {session_id} deleted on {date}: " +
                           ", ".join(f"{k} {v}" for k, v in counts.items()))
        # Rebuild the file from what's left, so no deleted text survives in its free space.
        self.store.db.execute("VACUUM")
        return counts
