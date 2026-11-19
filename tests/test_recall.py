"""Plan section 5's three tests (Nexus)."""
from conftest import add_session
from starter_agent.embed import HashEmbedder
from starter_agent.recall import estimate_tokens, recall, render_session

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
