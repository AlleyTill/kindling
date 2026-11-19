"""Plan section 4's six tests (CC), plus one on where evidence may come from."""
import json
import re

from conftest import add_session, agent_turns, say, scripted
from starter_agent import bdi, curiosity


def logged(store, kind):
    return [row["reason"] for row in store.db.execute("SELECT reason FROM log WHERE kind=? ORDER BY id", (kind,))]


# 1. A belief change appends a superseding row, and the old row survives.
def test_bdi_1_belief_change_appends_and_old_row_survives(make_agent):
    agent = make_agent(replies=[
        say("Noted.", beliefs=[{"text": "The mice lose weight in week 3", "confidence": 0.6, "evidence": [1]}]),
        say("Revised.", beliefs=[{"text": "The mice lose weight in week 4", "confidence": 0.8,
                                  "evidence": [1, 3], "supersedes": 1}]),
    ])
    agent.handle("The mice lost weight in week 3.")     # turn 1; the reply is turn 2
    agent.handle("Correction: it was week 4.")          # turn 3
    old, new = agent.store.db.execute("SELECT * FROM beliefs ORDER BY id").fetchall()
    assert (old["text"], old["confidence"], json.loads(old["evidence"])) == ("The mice lose weight in week 3", 0.6, [1])
    assert new["supersedes"] == old["id"] and new["key"] == old["key"]
    assert (new["text"], json.loads(new["evidence"])) == ("The mice lose weight in week 4", [1, 3])
    assert [b["id"] for b in bdi.current_beliefs(agent.store)] == [new["id"]]
    assert [b["id"] for b in agent.store.history("beliefs", new["id"])] == [old["id"], new["id"]]


# 2. A change that cites a turn id that doesn't exist is rejected (and so is one citing nothing).
def test_bdi_2_change_citing_missing_turn_is_rejected(make_agent):
    agent = make_agent(replies=[say("ok", beliefs=[
        {"text": "Cites a ghost", "confidence": 0.5, "evidence": [999]},
        {"text": "Cites nothing", "confidence": 0.5, "evidence": []},
        {"text": "Cites turn 1", "confidence": 0.5, "evidence": [1]},
    ])])
    agent.handle("hello")
    assert [b["text"] for b in bdi.current_beliefs(agent.store)] == ["Cites turn 1"]
    reasons = logged(agent.store, "rejected belief")
    assert len(reasons) == 2 and "999" in reasons[0] and "no evidence" in reasons[1]


def test_bdi_2b_evidence_must_come_from_the_recall_pack_or_this_session(make_agent):
    agent = make_agent(replies=[say("ok", beliefs=[{"text": "From an unrecalled session", "confidence": 0.5,
                                                    "evidence": [1]}])],
                       recall_budget_tokens=0)          # nothing fits, so nothing is recalled
    add_session(agent.store, [("user", "an old fact"), ("agent", "noted")])   # turns 1 and 2
    agent.handle("hello")
    assert bdi.current_beliefs(agent.store) == []
    assert "[1]" in logged(agent.store, "rejected belief")[0]


# 3. A confidence outside 0-1 is rejected.
def test_bdi_3_confidence_outside_0_1_is_rejected(make_agent):
    proposals = [(1.5, False), (-0.1, False), ("0.9", False), (True, False), (None, False), (0, True), (1, True)]
    agent = make_agent(replies=[say("ok", beliefs=[
        {"text": f"belief {i}", "confidence": c, "evidence": [1]} for i, (c, _) in enumerate(proposals)])])
    agent.handle("hello")
    kept = {b["text"] for b in bdi.current_beliefs(agent.store)}
    assert kept == {f"belief {i}" for i, (_, ok) in enumerate(proposals) if ok}
    assert len(logged(agent.store, "rejected belief")) == 5


# 4. The intention always serves the highest-ranked live desire, including after a re-rank.
def test_bdi_4_intention_serves_top_live_desire_after_rerank(make_agent):
    agent = make_agent(replies=[say("a"), say("Marked done.", done=[3])])
    store = agent.store
    assert bdi.pick_intention(store) is None
    methods = bdi.add_desire(store, "Write the methods section", rank=2)
    figure = bdi.add_desire(store, "Replicate figure 3", rank=1)
    assert bdi.pick_intention(store)["serves"] == figure
    bdi.rerank_desire(store, methods, 1)
    bdi.rerank_desire(store, figure, 3)
    assert bdi.pick_intention(store)["serves"] == methods
    agent.handle("What next?")
    shown = agent.model.calls[-1][-1]["content"].split("YOUR CURRENT INTENTION:")[1]
    assert shown.split("THIS SESSION")[0].strip().endswith("Write the methods section")
    agent.handle("Done with methods.")                  # turn 3, which the reply cites as the evidence
    assert bdi.pick_intention(store)["serves"] == figure        # a done intention is passed over
    bdi.drop_desire(store, figure)
    assert bdi.pick_intention(store) is None


# 5. With automatic adoption off, nothing from curiosity becomes a desire without `curious adopt`.
def test_bdi_5_no_desire_from_curiosity_without_adopt(make_agent):
    def reader(prompt):
        turn = next(t for t, text in agent_turns(prompt).items() if "not sure why" in text)
        return json.dumps({"questions": [{"text": "Why does the loss spike at step 400?",
                                          "quote": "not sure why the loss spikes", "turn": turn}]})
    agent = make_agent(fn=scripted([
        say("I'm not sure why the loss spikes at step 400.", desires=["Fix the loss"]),   # not the model's call
    ], reader))
    agent.handle("The loss spiked again.")
    agent.end_session()
    (question,) = curiosity.open_questions(agent.store)
    assert agent.store.db.execute("SELECT COUNT(*) FROM desires").fetchone()[0] == 0
    desire_id = curiosity.adopt(agent.store, question["id"])
    desire = agent.store.get("desires", desire_id)
    assert (desire["source"], desire["question_id"], desire["source_turn"]) == ("curiosity", question["key"], 2)
    assert desire["text"] == "Why does the loss spike at step 400?"
    assert agent.store.latest("questions", question["id"])["status"] == "adopted"


# 6. Deleting a session deletes the beliefs whose only evidence was in it, leaving content-free markers;
#    beliefs with other evidence keep only that evidence; no text from the session survives in the file.
def test_bdi_6_delete_session_removes_what_it_alone_taught(make_agent, tmp_path):
    secret = "violet harbour cipher"
    agent = make_agent(fn=scripted([say("Noted."), say(f"I'm not sure why the {secret} matters.")],
                                   lambda prompt: json.dumps({"questions": [
                                       {"text": f"What is the {secret} for?", "quote": f"the {secret} matters",
                                        "turn": max(agent_turns(prompt))}]})))
    store = agent.store
    store.db.execute("PRAGMA secure_delete=OFF")   # some SQLite builds zero deleted data; test our own cleanup
    s1 = store.start_session()
    t1 = store.add_turn(s1, "user", f"My lab code word is {secret}.")
    agent.handle(f"Remember the {secret}.")        # appends to the open session s1
    agent.handle("Why?")
    agent.end_session()                            # curiosity quotes the secret from an agent turn
    curiosity.adopt(store, curiosity.open_questions(store)[0]["id"])
    bdi.pick_intention(store)                      # an intention now serves the adopted desire
    store.log(s1, "rejected belief", "a log row from s1", json.dumps({"text": secret}))
    s2, (t2, _) = add_session(store, [("user", "The mice are weighed on Mondays."), ("agent", "Noted.")])
    only = store.append("beliefs", {"text": f"The code word is {secret}", "confidence": 0.9,
                                    "evidence": json.dumps([t1]), "flag": None})
    mixed = store.append("beliefs", {"text": "The student runs a mouse study", "confidence": 0.7,
                                     "evidence": json.dumps([t1, t2]), "flag": None})
    other = store.append("beliefs", {"text": "Weighing is on Mondays", "confidence": 0.8,
                                     "evidence": json.dumps([t2]), "flag": None})
    files = lambda: b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("agent.db"))
    assert secret.encode() in files()              # the grep can see it before the delete

    counts = agent.delete_session(s1)

    removed = store.latest("beliefs", only)
    assert removed["flag"] == "removed" and removed["confidence"] is None and json.loads(removed["evidence"]) == []
    assert re.fullmatch(r"a belief derived from a deleted session was removed on \d{4}-\d\d-\d\d", removed["text"])
    reduced = store.latest("beliefs", mixed)
    assert reduced["id"] != mixed and json.loads(reduced["evidence"]) == [t2] and reduced["flag"] == "review"
    assert store.latest("beliefs", other)["id"] == other
    assert [b["key"] for b in bdi.current_beliefs(store)] == [mixed, other]
    assert store.turn_ids([s1]) == set() and store.session(s1) is None
    assert store.db.execute("SELECT COUNT(*) FROM vectors WHERE turn_id=?", (t1,)).fetchone()[0] == 0
    assert bdi.live_desires(store) == [] and bdi.pick_intention(store) is None
    assert counts["belief_versions_removed"] == 1 and counts["beliefs_flagged_for_review"] == 1
    assert secret.encode() not in files()


# 1b. A change may only supersede the CURRENT version of a belief that exists.
def test_bdi_1b_supersedes_must_name_a_current_belief(make_agent):
    agent = make_agent(replies=[
        say("a", beliefs=[{"text": "keep", "confidence": 0.5, "evidence": [1]},          # belief #1
                          {"text": "v1", "confidence": 0.5, "evidence": [1]}]),           # belief #2
        say("b", beliefs=[{"text": "v2", "confidence": 0.6, "evidence": [3], "supersedes": 2}]),
        say("c", beliefs=[{"text": "from an old version", "confidence": 0.7, "evidence": [5], "supersedes": 2},
                          {"text": "from nowhere", "confidence": 0.7, "evidence": [5], "supersedes": 99},
                          {"text": "true is not #1", "confidence": 0.7, "evidence": [5], "supersedes": True},
                          {"text": "true is not turn 1", "confidence": 0.7, "evidence": [True]},
                          {"text": "a list is not a turn", "confidence": 0.7, "evidence": [[5]]}]),
    ])
    for message in ("one", "two", "three"):             # turns 1, 3, 5
        agent.handle(message)
    assert [b["text"] for b in bdi.current_beliefs(agent.store)] == ["keep", "v2"]
    assert len(logged(agent.store, "rejected belief")) == 5


# 4b. An intention is marked done only with real turns as evidence.
def test_bdi_4b_intention_done_needs_real_evidence(make_agent):
    agent = make_agent(replies=[say("Done!", done=[999]), say("Done!", done=[]), say("Done.", done=[5])])
    bdi.add_desire(agent.store, "Replicate figure 3")
    agent.handle("Is it done?")                          # turn 1: cites a ghost turn
    agent.handle("Now?")                                 # turn 3: cites nothing
    assert bdi.pick_intention(agent.store)["status"] == "active"
    assert "999" in logged(agent.store, "rejected intention_done")[0]
    agent.handle("It is done.")                          # turn 5, cited as the evidence
    (done,) = agent.store.current("intentions")
    assert (done["status"], json.loads(done["evidence"])) == ("done", [5])
    assert bdi.pick_intention(agent.store) is None       # the only desire's intention is done


# 6b. A belief that changed only in the deleted session goes back to its earlier view, flagged;
#     an intention done only by the deleted session's evidence is active again.
def test_bdi_6b_delete_undoes_a_change_made_only_in_that_session(make_agent):
    secret = "amber lantern protocol"
    agent = make_agent(replies=[say("ok", done=[3])])
    store = agent.store
    s1, (t1, _) = add_session(store, [("user", "The mice are weighed on Mondays."), ("agent", "Noted.")])
    old = store.append("beliefs", {"text": "Weighing is on Mondays", "confidence": 0.8,
                                   "evidence": json.dumps([t1]), "flag": None})
    bdi.add_desire(store, "Weigh the mice")
    agent.handle(f"Switch to Fridays, per the {secret}.")      # turn 3 opens session 2
    s2 = store.open_session()["id"]
    changed = store.append("beliefs", {"text": f"Weighing is on Fridays ({secret})", "confidence": 0.9,
                                       "evidence": json.dumps([3]), "flag": None}, store.get("beliefs", old))
    assert bdi.pick_intention(store) is None                   # done, citing turn 3
    agent.delete_session(s2)
    versions = store.history("beliefs", old)
    assert [v["id"] for v in versions[:2]] == [old, changed] and versions[1]["flag"] == "removed"
    (now,) = bdi.current_beliefs(store)
    assert (now["text"], now["confidence"], json.loads(now["evidence"])) == ("Weighing is on Mondays", 0.8, [t1])
    assert now["flag"] == "review" and now["supersedes"] == changed
    intention = bdi.pick_intention(store)
    assert (intention["status"], json.loads(intention["evidence"])) == ("active", [])
    dump = "\n".join(store.db.iterdump())
    assert secret not in dump and "Fridays" not in dump


# 6c. Ids of deleted turns and sessions are never handed out again, so no record can come to point
#     at a new, unrelated turn.
def test_bdi_6c_deleted_ids_are_never_reused(make_agent):
    agent = make_agent()
    store = agent.store
    add_session(store, [("user", "first"), ("agent", "ok")])
    s2, ids = add_session(store, [("user", "second"), ("agent", "ok")])
    agent.delete_session(s2)
    s3, (t,) = add_session(store, [("user", "third")])
    assert s3 > s2 and t > max(ids)
