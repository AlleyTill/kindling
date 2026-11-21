"""Plan section 5's three tests (Nexus), plus the fusion and session scoring it specifies."""
import pytest

from conftest import Clock, PointEmbedder, StubEmbedder, add_session
from starter_agent.agent import Agent
from starter_agent.config import Config
from starter_agent.embed import HashEmbedder
from starter_agent.models import FakeModel
from starter_agent.recall import (Memory, Recalled, ensure_vectors, estimate_tokens, fuse, rank_sessions, recall,
                                 render_session)
from starter_agent.store import Store

TOPICS = ["sourdough starters", "bicycle gearing", "tidal pools", "jazz chord voicings", "glacier retreat",
          "medieval bookbinding", "solar panel inverters", "beekeeping in winter", "orbital mechanics",
          "fermenting kimchi", "chess openings", "river delta sediment", "violin varnish", "coral bleaching",
          "typesetting with LaTeX", "volcanic soils", "migratory warblers", "sailing knots", "moss gardens"]


def filler(topic):
    return [("user", f"Today I want to talk about {topic} and how {topic} work in practice. " * 4),
            ("agent", f"Here is an overview of {topic}, with the main ideas and common mistakes. " * 4)]


# 1. A planted fact from session 1 comes back when asked about in session 20.
def test_recall_1_fact_from_session_1_returns_in_session_20(make_agent):
    store = make_agent().store
    first, _ = add_session(store, [("user", "My cat is called Biscuit and she hates the vacuum cleaner."),
                                   ("agent", "Noted: Biscuit, no vacuum.")])
    for topic in TOPICS[:18]:                              # sessions 2 to 19
        add_session(store, filler(topic))
    twentieth = store.start_session()
    assert twentieth == 20
    # A filler session costs about 175 tokens, so a 400-token budget holds two or three: recall must choose.
    pack, ids = recall(store, HashEmbedder(), "What is my cat called?", budget=400, exclude=twentieth)
    assert first in ids and "Biscuit" in pack
    assert len(ids) < 5


# 2. A session with the right words in the agent's turns only does not outrank one with them in the user's.
def test_recall_2_agent_only_words_do_not_outrank_user_words(make_agent):
    store = make_agent().store
    agent_side, _ = add_session(store, [
        ("user", "Can you help with my thesis plan?"),
        ("agent", "Quantum annealing schedule, quantum annealing schedule: the annealing schedule matters.")])
    user_side, _ = add_session(store, [
        ("user", "I need a quantum annealing schedule for my solver."),
        ("agent", "Here is a plan.")])
    costs = [estimate_tokens(render_session(store, sid) + "\n\n") for sid in (agent_side, user_side)]
    budget = max(costs)                                    # room for either session, not for both
    assert budget < sum(costs)
    pack, ids = recall(store, HashEmbedder(), "quantum annealing schedule", budget=budget)
    assert ids == [user_side]


# 3. The pack never exceeds the budget, and never cuts a session in half.
def test_recall_3_pack_within_budget_and_sessions_whole(make_agent):
    store = make_agent().store
    sessions = {}
    for i, topic in enumerate(TOPICS):
        turns = filler(topic) * (1 + i % 4)                # sessions of different lengths
        sid, _ = add_session(store, turns)
        sessions[sid] = turns
    for budget in (50, 120, 300, 700, 1500):
        pack, ids = recall(store, HashEmbedder(), "how do glacier and river sediment work", budget=budget)
        assert estimate_tokens(pack) <= budget
        for sid in ids:
            for role, text in sessions[sid]:
                assert f"{role}: {text}" in pack
        headers = pack.count("=== Session ")
        assert headers == len(ids) and ids == sorted(ids)
    assert recall(store, HashEmbedder(), "glacier", budget=10**6)[1] == sorted(sessions)


# 3b. The budget's exact edge, counting the blank line between sessions: two sessions that fill it exactly
#     both fit, and with one token less only one does.
def test_recall_3b_exact_budget_edge_counts_the_separators(make_agent):
    store = make_agent().store
    sids = []
    for text in ("first notes on soil", "second notes on soil"):
        sid, (turn,) = add_session(store, [("user", text)])
        while estimate_tokens(render_session(store, sid) + "\n\n") == estimate_tokens(render_session(store, sid)):
            store.db.execute("UPDATE turns SET text = text || 'a' WHERE id=?", (turn,))   # a letter at a time...
        sids.append(sid)
    blocks = [render_session(store, sid) for sid in sids]
    costs = [estimate_tokens(block + "\n\n") for block in blocks]
    assert costs == [estimate_tokens(block) + 1 for block in blocks]   # ...so each separator costs a token
    assert recall(store, HashEmbedder(), "soil", budget=sum(costs))[1] == sids
    assert len(recall(store, HashEmbedder(), "soil", budget=sum(costs) - 1)[1]) == 1


# 4. Reciprocal rank fusion with k = 60, dense weighted 2 and BM25 1; BM25 ranks only turns it matched.
def test_recall_4_fusion_is_rrf_k60_dense_2_bm25_1():
    fused = fuse(dense=[0.9, 0.5, 0.1], lexical=[0.0, 2.0, 1.0])
    assert fused[0] == pytest.approx(2 / 61)
    assert fused[1] == pytest.approx(2 / 62 + 1 / 61)
    assert fused[2] == pytest.approx(2 / 63 + 1 / 62)


def test_recall_4b_dense_outweighs_bm25_in_recall(make_agent):
    store = make_agent().store
    words_win, _ = add_session(store, [("user", "zebra zebra")])                         # BM25's first
    meaning_wins, _ = add_session(store, [("user", "zebra seen near the river at dawn, with many others")])
    embedder = StubEmbedder({"zebra zebra": 0.5, "zebra seen near the river at dawn, with many others": 0.9})
    assert [sid for sid, _ in rank_sessions(store, embedder, "zebra")] == [meaning_wins, words_win]


# 5. A session scores the sum of its best two turns: not its best one, not all of them.
def test_recall_5_session_scored_by_its_best_two_turns(make_agent):
    store = make_agent().store
    one, _ = add_session(store, [("user", "a1")])
    two, _ = add_session(store, [("user", "b1"), ("user", "b2")])
    five, _ = add_session(store, [("user", f"c{i}") for i in range(1, 6)])
    embedder = StubEmbedder({"a1": .99, "b1": .98, "b2": .97, "c1": .96, "c2": .95, "c3": .94, "c4": .93,
                             "c5": .92})
    ranked = rank_sessions(store, embedder, "nothing in common")      # no shared words: dense only
    assert [sid for sid, _ in ranked] == [two, five, one]
    assert ranked[0][1] == pytest.approx(2 / 62 + 2 / 63)


# 4c. The dense index scores each turn against the query itself. The query has its own vector and shares
#     no word with either turn, so BM25 can't carry the ranking.
def test_recall_4c_the_dense_index_scores_the_query(make_agent):
    store = make_agent().store
    far, _ = add_session(store, [("user", "bravo")])
    near, _ = add_session(store, [("user", "alpha")])
    embedder = PointEmbedder({"where is my cat": [1.0, 0.0, 0.0], "alpha": [0.9, 0.0, 0.1], "bravo": [0.1, 0.0, 0.9]})
    assert [sid for sid, _ in rank_sessions(store, embedder, "where is my cat")] == [near, far]


# 6. The narrow interface (Nexus): Memory.recall(query, budget_tokens) hands back the packed sessions with
#    their scores and says what made the scores. The open session is never recalled.
def test_recall_6_memory_recall_is_the_narrow_interface(make_agent):
    store = make_agent().store
    for topic in TOPICS[:4]:
        add_session(store, filler(topic))
    now = store.start_session()
    store.add_turn(now, "user", "tidal pools again")                  # open: already in the model's context
    got = Memory(store, HashEmbedder()).recall("tidal pools", 400)
    pack, ids = recall(store, HashEmbedder(), "tidal pools", budget=400, exclude=now)
    assert (got.text, [sid for sid, _ in got.sessions]) == (pack, ids) and now not in ids
    scores = dict(rank_sessions(store, HashEmbedder(), "tidal pools", exclude=now))
    assert got.sessions == [(sid, scores[sid]) for sid in ids]
    assert got.scored_by == "RRF of BM25 and the hash-256 embedder"


def test_recall_6b_another_retriever_can_be_swapped_in(tmp_path):
    class OwnRetriever:
        asked = None

        def recall(self, query, budget_tokens):
            self.asked = (query, budget_tokens)
            return Recalled("=== Session 7 ===\n[turn 70] user: from my own retriever", [(7, 1.0)], "by hand")
    own = OwnRetriever()
    config = Config(db=str(tmp_path / "agent.db"))
    agent = Agent(Store(config.db, Clock()), FakeModel(fn=lambda messages: "ok"), HashEmbedder(), config, memory=own)
    agent.handle("hello")
    assert own.asked == ("hello", 8000)                         # the agent asked the retriever it was given
    assert "[turn 70] user: from my own retriever" in agent.model.calls[-1][-1]["content"]


# --- Round 3: Nexus's two offered tests (19 Nov, 05:25), the first verbatim ---------------------------------

def test_best_two_not_worst_two(make_agent):
    store = make_agent().store
    x, _ = add_session(store, [("user", f"x{i}") for i in range(1, 6)])
    y, _ = add_session(store, [("user", "y1"), ("user", "y2")])
    embedder = StubEmbedder({"x1": .99, "x2": .98, "x3": .50, "x4": .49, "x5": .48, "y1": .90, "y2": .89})
    assert [sid for sid, _ in rank_sessions(store, embedder, "nothing in common")] == [x, y]


def test_changed_embedder_same_name_is_embedded_again(make_agent):   # Nexus's trap test, inverted after the fix
    store = make_agent().store
    add_session(store, [("user", "tidal pools at low tide")])
    ensure_vectors(store, HashEmbedder())

    class Tweaked(HashEmbedder):
        def _one(self, text):
            return list(reversed(super()._one(text)))
    tweaked = Tweaked()
    assert tweaked.name == "hash-256"
    stored = store.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    ensure_vectors(store, tweaked)
    assert store.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 2 * stored      # was: == stored


# --- Round 3: scores from one embedder only, and named after it (the r2 verifier's W11 and W12) ------------

def test_recall_7_scores_never_mix_two_embedders(make_agent):
    store, fresh = make_agent().store, make_agent(db=":memory:").store
    for s in (store, fresh):
        add_session(s, [("user", "alpha"), ("user", "charlie")])
        add_session(s, [("user", "bravo")])
    points = PointEmbedder({"query": [1.0, 0.0, 0.0], "alpha": [0.9, 0.1, 0.0], "bravo": [0.5, 0.5, 0.0]})
    ensure_vectors(store, StubEmbedder({"alpha": 0.1, "bravo": 0.99}))      # another embedder was used before
    assert rank_sessions(store, points, "query") == rank_sessions(fresh, points, "query")


def test_recall_7b_memory_names_the_embedder_it_used(make_agent):
    store = make_agent().store
    add_session(store, [("user", "alpha")])
    assert Memory(store, StubEmbedder({})).recall("alpha", 1000).scored_by == "RRF of BM25 and the stub embedder"


# --- Round 3: a session too big for the budget is named, never skipped in silence (army, r3) ---------------

def test_recall_8_a_session_too_big_for_the_budget_is_named_and_packing_goes_on(make_agent):
    store = make_agent().store
    big, _ = add_session(store, [("user", "my cat is called Biscuit. " * 100)])      # the best match, ~650 tokens
    small, _ = add_session(store, [("user", "notes on my cat")])
    got = Memory(store, HashEmbedder()).recall("what is my cat called", 400)
    assert [sid for sid, _ in got.sessions] == [small]
    cost = estimate_tokens(render_session(store, big) + "\n\n")
    assert [(sid, tokens) for sid, _, tokens in got.skipped] == [(big, cost)] and cost > 400
    assert recall(store, HashEmbedder(), "what is my cat called", budget=400)[1] == [small]


def test_recall_8b_chat_and_the_recall_command_both_say_so(make_agent, cli, capsys, tmp_path):
    agent = make_agent(fn=lambda messages: "ok", recall_budget_tokens=400, db=str(tmp_path / "chat.db"))
    add_session(agent.store, [("user", "my cat is called Biscuit. " * 100)])
    agent.handle("what is my cat called?")
    assert any(note.startswith("recall skipped session 1") and "was left of this turn's 400-token budget" in note
               for note in agent.notes)
    (tmp_path / "agent.json").write_text('{"recall_budget_tokens": 400}')
    cli("chat", "my cat is called Biscuit. " * 100)
    cli("session", "end")
    capsys.readouterr()
    assert cli("recall", "cat", "Biscuit") == 0
    assert "session 1  score" in (out := capsys.readouterr().out) and "skipped: ~" in out and "Nothing recalled" in out


# --- Round 4: a session is named when its size, not its rank, kept it out (Nexus, 20 Nov: V23) --------------
# Each session: (name, its cosine with the query, about how many tokens its text is), best first.

V23 = {
    "a: too big, above a packed one: named": ([("big", .9, 700), ("small", .8, 100)], 400, ["small"], ["big"]),
    "b: too big, below every packed one: not named": ([("small", .9, 100), ("big", .8, 700)], 400, ["small"], []),
    "c: the only session, too big: named": ([("big", .9, 700)], 400, [], ["big"]),
    "B: fits the budget, not what was left, above one that got in: named": (
        [("A", .9, 500), ("B", .8, 400), ("C", .7, 100)], 720, ["A", "C"], ["B"]),
    "V5: two too big, each above a packed one: both named": (            # the round-4 verifier's V5
        [("big1", .9, 700), ("s1", .8, 100), ("big2", .7, 700), ("s2", .6, 100)], 400, ["s1", "s2"], ["big1", "big2"]),
    "V6: nothing fits among three: all three named": (                    # the round-4 verifier's V6
        [("x", .9, 700), ("y", .8, 600), ("z", .7, 500)], 400, [], ["x", "y", "z"]),
}


@pytest.mark.parametrize("case", V23)
def test_recall_11_a_session_is_named_when_its_size_not_its_rank_kept_it_out(make_agent, case):
    sessions, budget, packed, named = V23[case]
    store, texts = make_agent().store, {name: f"{name} " + "z" * (4 * tokens) for name, _, tokens in sessions}
    ids = {name: add_session(store, [("user", text)])[0] for name, text in texts.items()}
    cost = {name: estimate_tokens(render_session(store, sid) + "\n\n") for name, sid in ids.items()}
    if case.startswith("B"):        # B alone fits the budget, so the old rule (bigger than it all) never named it
        assert cost["B"] <= budget < cost["A"] + cost["B"] and cost["A"] + cost["C"] <= budget, cost
    embedder = StubEmbedder({texts[name]: cosine for name, cosine, _ in sessions})    # no shared words: dense only
    got = Memory(store, embedder).recall("q", budget)
    name_of = {sid: name for name, sid in ids.items()}
    assert sorted(name_of[sid] for sid, _ in got.sessions) == packed
    assert [(name_of[sid], tokens) for sid, _, tokens in got.skipped] == [(name, cost[name]) for name in named]


# --- Round 3: recall's budget is sized from the model's context (Nexus, 19 Nov, 02:33) ---------------------

class Asked:
    """A retriever that records the budget it was given."""
    budget = None

    def recall(self, query, budget_tokens):
        self.budget = budget_tokens
        return Recalled("", [], "by hand")


@pytest.mark.parametrize("context, prompt, cap, budget", [
    (8000, 2000, 8000, 4000),          # 8,000 - 2,000 - a quarter of 8,000 for the answer
    (100000, 100, 8000, 8000),         # never more than the cap
    (100000, 100, 500, 500),           # the cap is yours to set
    (3000, 100, 8000, 1876),           # the answer's room is at least 1,024 tokens
    (8000, 5100, 8000, None),          # under 1,000 tokens of room: recall is skipped, and says so
])
def test_recall_9_the_budget_is_what_the_context_leaves(tmp_path, context, prompt, cap, budget):
    config = Config(db=str(tmp_path / "agent.db"), context_tokens=context, recall_budget_tokens=cap)
    asked = Asked()
    agent = Agent(Store(config.db, Clock()), FakeModel(), HashEmbedder(), config, memory=asked)
    pack = agent.recall("q", [{"role": "user", "content": "x" * (4 * prompt)}])
    assert asked.budget == budget
    if budget is None:
        assert pack.sessions == [] and "recall was skipped" in pack.text and "recall skipped" in agent.notes[0]


def test_recall_10_a_prompt_too_long_for_a_tiny_context_is_refused_not_cut(make_agent):
    """Nexus's tiny num_ctx fixture: the session alone outgrows a 1,000-token context."""
    from starter_agent.models import ModelError
    agent = make_agent(fn=lambda messages: "ok", context_tokens=1000)
    agent.handle("short")
    with pytest.raises(ModelError, match="too long"):
        agent.handle("a long paste " * 300)
    assert len(agent.model.calls) == 1                                    # never sent
    assert [t["text"] for t in agent.store.turns(1)] == ["short", "ok"]   # and not kept


def test_recall_10b_as_the_session_grows_the_best_session_that_fits_is_kept_not_the_newest(make_agent):
    agent = make_agent(fn=lambda messages: "ok", context_tokens=4000)
    add_session(agent.store, [("user", "the warblers migrate in April " * 107)])        # relevant, ~800 tokens
    for topic in ("sourdough", "bicycles"):
        add_session(agent.store, [("user", f"notes about {topic} " * 145)])            # newer, ~800 tokens each
    agent.handle("when do the warblers migrate? " + "padding " * 600)                  # room for one session
    recalled = agent.model.calls[-1][-1]["content"].split("YOUR CURRENT BELIEFS")[0]
    assert recalled.count("=== Session ") == 1 and "=== Session 1," in recalled
