"""Recall (plan section 5, Nexus's Mnemosyne wiring): find the past sessions that matter, hand them over whole.

1. Score every user turn against the event with two indexes, BM25 (words) and dense (embeddings),
   and fuse the two rankings with reciprocal rank fusion: k = 60, dense weighted 2, BM25 1.
2. Score each session by its best two turns.
3. Pack whole sessions, best first, into a token budget (default 8,000); never cut one in half.
   Inside the pack, put them oldest to newest so time reads forward.
4. Return the pack and the session ids, so Record can check what was cited.

Only the user's turns are indexed. No model is called here except the embedder.
"""
import math
from collections import Counter, defaultdict

from .embed import cosine, from_blob, to_blob, words


def estimate_tokens(text):
    return (len(text) + 3) // 4         # about four characters a token; no tokenizer needed


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
    lines = [f"=== Session {session_id}, {session['started_at']} to {session['ended_at'] or 'now'} ==="]
    lines += [f"[turn {t['id']}] {t['role']}: {t['text']}" for t in store.turns(session_id)]
    return "\n".join(lines)


def ensure_vectors(store, embedder):
    """Embed any user turn that has no vector from this embedder yet."""
    missing = store.db.execute(
        "SELECT id, text FROM turns WHERE role='user' AND id NOT IN "
        "(SELECT turn_id FROM vectors WHERE embedder=?)", (embedder.name,)).fetchall()
    if missing:
        vecs = embedder.embed([row["text"] for row in missing])
        store.db.executemany("INSERT INTO vectors (turn_id, embedder, vec) VALUES (?, ?, ?)",
                             [(row["id"], embedder.name, to_blob(v)) for row, v in zip(missing, vecs)])


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
    ensure_vectors(store, embedder)
    rows = store.db.execute(
        "SELECT t.id, t.session_id, t.text, v.vec FROM turns t JOIN vectors v "
        "ON v.turn_id = t.id AND v.embedder = ? WHERE t.role = 'user' AND t.session_id IS NOT ?",
        (embedder.name, exclude)).fetchall()
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


def recall(store, embedder, query, budget=8000, exclude=None):
    """Return (pack, session_ids): the most relevant past sessions, whole, oldest first."""
    chosen, used = [], 0
    for sid, _ in rank_sessions(store, embedder, query, exclude):
        block = render_session(store, sid)
        cost = estimate_tokens(block + "\n\n")
        if used + cost <= budget:        # a session that doesn't fit is skipped, never cut
            chosen.append((sid, block))
            used += cost
    chosen.sort()                        # session ids grow with time, so this is oldest to newest
    return "\n\n".join(block for _, block in chosen), [sid for sid, _ in chosen]
