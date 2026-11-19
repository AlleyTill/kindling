"""Plan section 6's six tests (Lyra)."""
import json

from conftest import agent_turns, say, scripted
from starter_agent import curiosity

PUZZLE = "I'm not sure why the loss spikes at step 400"


def honest_reader(prompt):
    """Stands in for a model that reads the whole session: a phrase "I'm not sure why X" is an open
    question unless a later agent turn in the same prompt explains X with "because"."""
    turns = agent_turns(prompt)
    asks = [t for t, text in turns.items() if PUZZLE in text]
    answered = any("loss spikes at step 400 because" in text for text in turns.values())
    return json.dumps({"questions": [] if answered or not asks else [
        {"text": "Why does the loss spike at step 400?", "quote": PUZZLE, "turn": asks[0]}]})


def questions(store, where="1"):
    return store.db.execute(f"SELECT * FROM questions WHERE {where} ORDER BY id").fetchall()


# 1. The agent says "I'm not sure why X happens" and never resolves it: the question appears,
#    with that phrase and turn id.
def test_curiosity_1_unresolved_question_appears_with_phrase_and_turn(make_agent):
    agent = make_agent(fn=scripted([say(PUZZLE + "."), say("Let me check the data loader."),
                                    say("The logs look normal.")], honest_reader))
    agent.handle("The loss spiked again.")              # turn 1; the reply is turn 2
    agent.handle("Any ideas?")
    agent.handle("Thanks.")
    queued = agent.end_session()
    (q,) = curiosity.open_questions(agent.store)
    assert queued == [q["id"]]
    assert (q["quote"], q["source_turn"], q["session_id"]) == (PUZZLE, 2, 1)
    assert len(agent.model.calls) == 4                   # three turns plus ONE curiosity call
    assert agent.end_session() == [] and len(agent.model.calls) == 4    # an ended session isn't read twice
    assert "The loss spiked again" not in agent.model.calls[-1][-1]["content"]   # only the agent's turns


# 2. The same, but the agent answers it two turns later: no question appears.
def test_curiosity_2_answered_question_does_not_appear(make_agent):
    agent = make_agent(fn=scripted([say(PUZZLE + "."), say("Let me check the data loader."),
                                    say("Found it: the loss spikes at step 400 because the warm-up ends.")],
                                   honest_reader))
    for message in ("The loss spiked again.", "Any ideas?", "And now?"):
        agent.handle(message)
    assert agent.end_session() == []
    assert questions(agent.store) == []
    assert "because the warm-up ends" in agent.model.calls[-1][-1]["content"]   # the reader saw the answer


# 3. The model returns a question whose phrase isn't in the cited turn: it's dropped.
def test_curiosity_3_question_with_unfound_phrase_is_dropped(make_agent):
    def inventive(prompt):
        first, second = sorted(agent_turns(prompt))
        return json.dumps({"questions": [
            {"text": "Real one", "quote": "not sure why the loss spikes", "turn": first},
            {"text": "Wrong turn", "quote": "not sure why the loss spikes", "turn": second},
            {"text": "Paraphrased", "quote": "unsure about the loss spike", "turn": first},
            {"text": "One word", "quote": "loss", "turn": first},
            {"text": "Made-up turn", "quote": "not sure why the loss spikes", "turn": 999},
        ]})
    agent = make_agent(fn=scripted([say(PUZZLE + "."), say("The data loader looks fine.")], inventive))
    agent.handle("The loss spiked.")
    agent.handle("Hmm.")
    agent.end_session()
    assert [q["text"] for q in questions(agent.store)] == ["Real one"]
    reasons = [r[0] for r in agent.store.db.execute("SELECT reason FROM log WHERE kind='rejected question'")]
    assert len(reasons) == 4 and all(reasons)


# 3b. The quote is 2 to 20 whole words, from one of the AGENT's turns in this session.
def test_curiosity_3b_quote_limits(make_agent):
    long_turn = ("I am not sure why the loss spikes at step four hundred when the learning rate warm up "
                 "ends and the batch size doubles at once")              # 24 words
    words = long_turn.split()

    def reader(prompt):
        turn = max(agent_turns(prompt))
        return json.dumps({"questions": [
            {"text": "twenty words", "quote": " ".join(words[:20]), "turn": turn},
            {"text": "twenty-one words", "quote": " ".join(words[:21]), "turn": turn},
            {"text": "two words", "quote": "loss spikes", "turn": turn},
            {"text": "cut words", "quote": "ot sure wh", "turn": turn},
            {"text": "the user's words", "quote": "my training run", "turn": turn - 1},
        ]})
    agent = make_agent(fn=scripted([say(long_turn)], reader), dedupe_similarity=1.01)
    agent.handle("Look at my training run.")               # turn 1; the agent's reply is turn 2
    agent.end_session()
    assert [q["text"] for q in questions(agent.store)] == ["twenty words", "two words"]
    assert len(agent.store.db.execute("SELECT * FROM log WHERE kind='rejected question'").fetchall()) == 3


# 4. A dropped question, reworded in a later session: it doesn't come back (nor does an open one).
def test_curiosity_4_dropped_question_reworded_does_not_return(make_agent):
    asked = iter([
        ["Why does the training loss spike at step 400?", "Is the validation set too small?"],
        ["Why does the training loss spike around step 400?",     # the dropped one, reworded
         "Is the validation set too small for this?",             # the open one, reworded
         "Should the mice be weighed twice a week?"],             # new
    ])

    def reader(prompt):
        turn = max(agent_turns(prompt))
        return json.dumps({"questions": [{"text": text, "quote": "not sure", "turn": turn} for text in next(asked)]})
    agent = make_agent(fn=scripted([say("I'm not sure."), say("I'm not sure.")], reader))
    agent.handle("Session one.")
    agent.end_session()
    spike, small = curiosity.open_questions(agent.store)
    curiosity.drop(agent.store, spike["id"])
    agent.handle("Session two.")
    agent.end_session()
    assert [q["text"] for q in curiosity.open_questions(agent.store)] == [
        "Is the validation set too small?", "Should the mice be weighed twice a week?"]
    assert agent.store.latest("questions", spike["id"])["status"] == "dropped"


# 5. The queue at 20: the oldest open question becomes `expired`, and nothing is deleted.
def test_curiosity_5_full_queue_expires_oldest_and_deletes_nothing(make_agent):
    animals = ["zebra", "falcon", "otter", "lynx", "heron", "bison", "gecko", "walrus", "ibis", "koala", "moose",
               "newt", "okapi", "panda", "quail", "raven", "skunk", "tapir", "viper", "wombat", "yak"]
    texts = [f"Why does the {a} {verb}?" for a, verb in zip(animals, ["sleep", "migrate", "whistle", "dig", "glow",
             "molt", "swim", "climb", "hum", "sneeze", "wander", "blink", "shiver", "drift", "nest", "sing",
             "hide", "graze", "coil", "burrow", "roam"])]
    reader = lambda prompt: json.dumps({"questions": [
        {"text": t, "quote": "not sure", "turn": max(agent_turns(prompt))} for t in texts]})
    agent = make_agent(fn=scripted([say("I'm not sure.")], reader))
    agent.handle("Go.")
    queued = agent.end_session()
    assert len(queued) == 21
    waiting = curiosity.open_questions(agent.store)
    assert len(waiting) == 20 and waiting[0]["text"] == texts[1]
    assert agent.store.latest("questions", queued[0])["status"] == "expired"
    assert [q["id"] for q in questions(agent.store, "supersedes IS NULL")] == queued   # every original row is there


# 6. Adoption with automatic adoption off: nothing becomes a desire without `curious adopt`.
def test_curiosity_6_adoption_needs_curious_adopt(make_agent, cli, tmp_path, capsys):
    for message in ("why is the sky blue?", "how do magnets work?"):
        assert cli("chat", message) == 0 and cli("session", "end") == 0
    assert cli("curious", "list") == 0 and capsys.readouterr().out.count("#") == 2
    assert cli("desire", "list") == 0 and capsys.readouterr().out == ""      # two questions, no desires
    assert cli("curious", "adopt", "1") == 0
    capsys.readouterr()
    assert cli("desire", "list") == 0
    assert capsys.readouterr().out == "#1  rank 1  why is the sky blue?  (curiosity)\n"
    # The switch exists and works: with auto_adopt on, a new question becomes a desire at once.
    agent = make_agent(fn=scripted([say(PUZZLE + ".")], honest_reader), db=str(tmp_path / "auto.db"),
                       auto_adopt=True)
    agent.handle("The loss spiked.")
    agent.end_session()
    assert agent.store.db.execute("SELECT source FROM desires").fetchall()[0][0] == "curiosity"
