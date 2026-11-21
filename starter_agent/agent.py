"""The loop (plan section 2):

    event (your message) -> RECALL -> DELIBERATE -> ACT -> RECORD
                              ^                              |
                              +---- CURIOSITY (between sessions)

RECALL      recall.py   the past sessions that matter, whole (Memory.recall: swap in your own)
DELIBERATE  bdi.py      pick the intention; gather the current beliefs
ACT         models.py   ONE model call: a reply plus a JSON block of belief changes
RECORD      store.py    the turns, and every belief change that passes bdi's checks
CURIOSITY   curiosity.py   one call when a session ends
"""
import sqlite3
from datetime import datetime, timedelta, timezone

from . import bdi, curiosity
from .models import check_fits, clean, context_of, estimate_tokens, split_json, take_counts, text_of
from .recall import Memory, Recalled, render_session


def utc_now():
    return datetime.now(timezone.utc)


class Agent:
    def __init__(self, store, model, embedder, config, memory=None):
        self.store, self.model, self.embedder, self.config = store, model, embedder, config
        self.memory = memory or Memory(store, embedder)      # anything with recall(query, budget_tokens)
        self.notes = []                  # what you should know after a command; the command line prints them

    def idle_session(self):
        """The open session if it has sat idle past the gap, else None. Read-only."""
        session = self.store.open_session()
        if session:
            turns = self.store.turns(session["id"])
            last = datetime.fromisoformat(turns[-1]["at"] if turns else session["started_at"])
            if self.store.clock() - last > timedelta(minutes=self.config.idle_minutes):
                return session
        return None

    def end_if_idle(self):
        """Nothing runs in the background, so an idle session ends at your next chat (or `session end`)."""
        if self.idle_session():
            self.end_session()

    def session_id(self):
        """The open session; a new one if there's none, or if the open one sat idle too long."""
        self.end_if_idle()
        session = self.store.open_session()
        return session["id"] if session else self.store.start_session()

    def recall(self, query, bare):
        """RECALL in the room the model's context leaves (Nexus): the budget is the smaller of
        recall_budget_tokens and context - this prompt without recall - room for the answer. With under
        about 1,000 tokens of room, recall is skipped, and you and the model are told."""
        context = context_of(self.config, self.model)
        room = context - estimate_tokens("".join(m["content"] for m in bare)) - max(context // 4, 1024)
        if room < 1000:
            self.notes.append(f"recall skipped: this session leaves ~{max(room, 0):,} tokens of the model's "
                              f"{context:,}-token context")
            return Recalled("(none: this session has filled the context, so recall was skipped)", [], "")
        budget = min(self.config.recall_budget_tokens, room)
        pack = self.memory.recall(query, budget)
        for sid, score, tokens in getattr(pack, "skipped", []):
            self.notes.append(f"recall skipped session {sid} (score {score:.4f}): ~{tokens:,} tokens, more than "
                              f"was left of this turn's {budget:,}-token budget, and a session is never cut")
        return pack

    def handle(self, message):
        """One turn of the loop. Returns the agent's reply."""
        message = clean(message)
        sid = self.session_id()
        turn = self.store.add_turn(sid, "user", message)
        try:
            # DELIBERATE: the intention is picked in code, never by the model.
            intention = bdi.pick_intention(self.store)
            beliefs = bdi.current_beliefs(self.store)
            this = render_session(self.store, sid)

            # RECALL: past sessions only; this one is already in context below.
            pack = self.recall(message, bdi.act_messages("", beliefs, intention, this))
            recalled = [session for session, _ in pack.sessions]

            # ACT: one call, refused before it's sent if it can't fit the model's context.
            messages = bdi.act_messages(pack.text, beliefs, intention, this)
            check_fits(messages, context_of(self.config, self.model))
            reply, changes = split_json(clean(text_of(self.model.complete(messages))))
            counts = take_counts()      # the server's counts far under the estimate: for the log only

            # RECORD: the reply and its changes together, only while the session exists (you may have
            # deleted it from another terminal while the model answered). Evidence: turns the model saw.
            with self.store.transaction():
                if self.store.session(sid) is None:
                    return "(This session was deleted while the model was answering, so nothing was kept.)"
                self.store.add_turn(sid, "agent", reply)
                for count in counts:
                    self.store.log(sid, "count", count)
                self.notes += bdi.record_changes(self.store, changes, self.store.turn_ids(recalled + [sid]),
                                                 sid, intention)
        except BaseException:              # no usable reply (unreachable, unreadable, or you pressed Ctrl-C):
            self.store.take_back(turn)     # a turn is kept only with its reply, so a retry isn't stored twice
            raise
        return reply

    def end_session(self):
        """End the open session; then curiosity reads every ended session it hasn't read yet: this one, and
        any whose read failed before. A session too long for one call is read in part, and you're told
        once (curiosity.run). Returns the new question ids."""
        session = self.store.open_session()
        if session:
            self.store.end_session(session["id"])
        queued = []
        for unread in self.store.unread_sessions():
            try:
                queued += curiosity.run(self.store, self.model, self.embedder, self.config, unread["id"], self.notes)
            except Exception as error:     # the model can't be reached: it stays unread, and the next end tries again
                self.store.log(unread["id"], "curiosity failed", repr(error))
                self.notes.append(f"curiosity couldn't read session {unread['id']} ({error}); the next "
                                  "session end tries again (see: log)")
                break
        return queued

    def delete_session(self, session_id):
        """Your word wins (plan section 4): remove the session and everything written in it."""
        date = self.store.clock().astimezone().date().isoformat()      # your local date
        with self.store.transaction():    # the write lock first: no reply can land in it meanwhile
            if self.store.session(session_id) is None:
                return None
            gone = self.store.turn_ids([session_id]) | self.store.orphan_turn_ids()
            counts = bdi.forget_turns(self.store, session_id, gone, date)
            counts["question_rows_removed"] = curiosity.forget_turns(self.store, gone, date)
            self.store.forget_session(session_id)
            counts["turns_deleted"] = len(gone)
            self.store.log(None, "deleted", f"session {session_id} deleted on {date}: " +
                           ", ".join(f"{k} {v}" for k, v in counts.items()))
        try:   # secure_delete has already overwritten the deleted text; VACUUM also shrinks the file
            self.store.db.execute("VACUUM")
        except sqlite3.Error as error:
            self.notes.append(f"session {session_id} is deleted and its text overwritten; the file couldn't be "
                              f"compacted ({error}), which is safe to leave")
        return counts
