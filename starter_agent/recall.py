"""Recall (plan section 5, Nexus's Mnemosyne wiring): find the past sessions that matter, hand them over whole.

1. Score every user turn against the event with two indexes, BM25 (words) and dense (embeddings),
   and fuse the two rankings with reciprocal rank fusion: k = 60, dense weighted 2, BM25 1.
2. Score each session by its best two turns.
3. Pack whole sessions, best first, into a token budget (at most 8,000, less when the model's context
   has less room: see Agent.recall); never cut one in half. A session too big for the budget is skipped,
   and named, so you know. Inside the pack, put them oldest to newest so time reads forward.
4. Return the pack and the session ids, so Record can check what was cited.

Only the user's turns are indexed. No model is called here except the embedder.
The agent sees only Memory.recall(query, budget_tokens) at the bottom: swap in your own retriever there.
"""
import json
import math
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .embed import cosine, from_blob, to_blob, words
from .models import estimate_tokens
from .store import local

PROBE = "A fixed sentence whose embedding fingerprints the embedder."


def bm25(query, docs, k1=1.5, b=0.75):
    """Okapi BM25: one score per doc (each a list of words) for the query (a list of words)."""
    if not docs:
        return []
    avg_len = sum(len(d) for d in docs) / len(docs) or 1.0
    df = Counter(word for d in docs for word in set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for word in set(query):
            if tf[word]:
                idf = math.log(1 + (len(docs) - df[word] + 0.5) / (df[word] + 0.5))
                score += idf * tf[word] * (k1 + 1) / (tf[word] + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(score)
    return scores


def ranks(scores, positive_only=False):
    """{position: rank}, rank 1 = best score."""
    order = sorted((i for i, s in enumerate(scores) if s > 0 or not positive_only), key=lambda i: -scores[i])
    return {i: r for r, i in enumerate(order, start=1)}


def render_session(store, session_id):
    session = store.session(session_id)
    ended = local(session["ended_at"]) if session["ended_at"] else "now"
    lines = [f"=== Session {session_id}, {local(session['started_at'])} to {ended} ==="]
    return "\n".join(lines + [turn_line(t) for t in store.turns(session_id)])


def turn_line(t):
    """'[turn N] role: text', with the text's own new lines indented, so nothing written in a turn can
    pose as another turn or as a section of the prompt."""
    return f"[turn {t['id']}] {t['role']}: " + t["text"].replace("\n", "\n    ")


def vector_key(embedder):
    """What the stored vectors are filed under: the embedder's name plus a fingerprint of what it does
    (Nexus). Change how an embedder works and the fingerprint changes, so every turn is embedded again
    instead of scores mixing old vectors with new ones."""
    probe = [round(x, 4) for x in embedder.embed([PROBE])[0]]
    return f"{embedder.name}:{zlib.crc32(json.dumps(probe).encode()):08x}"


def ensure_vectors(store, embedder):
    """Embed any user turn that has no vector from this embedder yet. Returns the key they're filed under."""
    key = vector_key(embedder)
    missing = store.db.execute(
        "SELECT id, text FROM turns WHERE role='user' AND id NOT IN "
        "(SELECT turn_id FROM vectors WHERE embedder=?)", (key,)).fetchall()
    if missing:
        vecs = embedder.embed([row["text"] for row in missing])
        store.db.executemany("INSERT INTO vectors (turn_id, embedder, vec) VALUES (?, ?, ?)",
                             [(row["id"], key, to_blob(v)) for row, v in zip(missing, vecs)])
    return key


def fuse(dense, lexical, k=60):
    """Reciprocal rank fusion of two score lists (one score per turn): dense weighted 2, BM25 1.
    A turn with no word in common with the query gets nothing from BM25."""
    fused = defaultdict(float)
    for i, rank in ranks(dense).items():
        fused[i] += 2 / (k + rank)
    for i, rank in ranks(lexical, positive_only=True).items():
        fused[i] += 1 / (k + rank)
    return fused


def rank_sessions(store, embedder, query, exclude=None):
    """[(session id, score)], best first. A session scores the sum of its best two user turns."""
    key = ensure_vectors(store, embedder)
    rows = store.db.execute(
        "SELECT t.id, t.session_id, t.text, v.vec FROM turns t JOIN vectors v "
        "ON v.turn_id = t.id AND v.embedder = ? WHERE t.role = 'user' AND t.session_id IS NOT ?",
        (key, exclude)).fetchall()
    if not rows:
        return []
    q = embedder.embed([query])[0]
    dense = [cosine(q, from_blob(row["vec"])) for row in rows]
    lexical = bm25(words(query), [words(row["text"]) for row in rows])
    per_session = defaultdict(list)
    for i, score in fuse(dense, lexical).items():
        per_session[rows[i]["session_id"]].append(score)
    scores = {sid: sum(sorted(s, reverse=True)[:2]) for sid, s in per_session.items()}
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def pack_sessions(store, ranked, budget):
    """Whole sessions from `ranked` (best first) that fit the budget, oldest first: [(session id, its text)];
    and the sessions too big for the budget on their own, which no pack can hold: [(id, score, tokens)]."""
    chosen, too_big, used = [], [], 0
    for sid, score in ranked:
        block = render_session(store, sid)
        cost = estimate_tokens(block + "\n\n")
        if used + cost <= budget:        # a session that doesn't fit is skipped, never cut
            chosen.append((sid, block))
            used += cost
        elif cost > budget:
            too_big.append((sid, score, cost))
    chosen.sort()                        # session ids grow with time, so this is oldest to newest
    return chosen, too_big


def recall(store, embedder, query, budget=8000, exclude=None):
    """Return (pack, session_ids): the most relevant past sessions, whole, oldest first."""
    chosen, _ = pack_sessions(store, rank_sessions(store, embedder, query, exclude), budget)
    return "\n\n".join(block for _, block in chosen), [sid for sid, _ in chosen]


@dataclass
class Recalled:
    text: str              # the sessions, whole and oldest first: what the model is shown
    sessions: list         # [(session id, score)], oldest first
    scored_by: str         # what made the scores; print it beside them
    skipped: list = field(default_factory=list)   # [(id, score, tokens)]: would have been packed, too big


class Memory:
    """The one question the agent asks its memory (Nexus's narrow interface): recall(query, budget_tokens).
    To try another retriever, write a class with this method and pass it as Agent(..., memory=yours)."""

    def __init__(self, store, embedder):
        self.store, self.embedder = store, embedder

    def recall(self, query, budget_tokens):
        """The past sessions that matter for `query`, whole, within the budget. The open session is left
        out: it's already in the model's context."""
        current = self.store.open_session()
        ranked = rank_sessions(self.store, self.embedder, query, exclude=current["id"] if current else None)
        scores = dict(ranked)
        chosen, too_big = pack_sessions(self.store, ranked, budget_tokens)
        lowest = min((scores[sid] for sid, _ in chosen), default=-1)
        return Recalled("\n\n".join(block for _, block in chosen), [(sid, scores[sid]) for sid, _ in chosen],
                        f"RRF of BM25 and the {self.embedder.name} embedder",
                        [s for s in too_big if s[1] > lowest])       # it outscored a session that was packed
