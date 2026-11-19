# A starter agent of your own

A small agent you run on your own computer. It:

- **remembers your past sessions** and brings back the ones that matter, whole;
- **keeps beliefs it can change**, each with how sure it is (0 to 1) and the turns that are its evidence.
  When a belief changes, the new version names its evidence; it never just says "I decided";
- **notices questions it raised and didn't answer**, and keeps a short list you can read and prune.

**Status: a rough first skeleton** (19 November 2026). It runs and its tests pass, but no student has set
it up from this README yet, so expect rough edges. **Its licence is still to be decided.**

It's deliberately small (about 1,100 lines of Python, standard library only) so you can read all of it in
an afternoon and change any of it.

## The loop

```
your message ─► RECALL ─► DELIBERATE ─► ACT ─► RECORD
                past       beliefs,     ONE    the turns, and each belief change
                sessions   desires,     model  that cites real evidence
                           intention    call
                   ▲                               │
                   └──── CURIOSITY (once, when a session ends) ◄┘
```

Every part calls a model only through one function, `complete(messages) -> text`, so nothing depends on
which model you pick. Everything it knows is in one SQLite file, `agent.db`. Nothing runs as a service.

## Set it up (an afternoon)

Run every command from inside this folder.

1. **Python 3.10 or newer** (tested on 3.12). Nothing else is needed to run it.
2. **Run the tests** (pytest is needed only for the tests):
   ```
   python -m pip install pytest
   python -m pytest
   ```
3. **Try it with the fake model first.** It needs no key and downloads nothing. The "model" is three rules
   (in `models.py`, `demo_brain`): a message starting `remember:` becomes a belief; a message ending in `?`
   gets "I'm not sure about that yet", which curiosity notices when the session ends.
   <!-- tests/test_cli.py runs every line of the next block -->
   ```
   python -m starter_agent chat "remember: my experiment uses 40 mice"
   python -m starter_agent chat "why do the mice lose weight?"
   python -m starter_agent session end
   python -m starter_agent chat "tell me about the mice"
   python -m starter_agent beliefs
   python -m starter_agent curious list
   python -m starter_agent desire add "Find out why the mice lose weight" --rank 1
   python -m starter_agent intentions
   python -m starter_agent recall mice
   python -m starter_agent session delete 1 --yes
   python -m starter_agent beliefs
   ```
   The last `beliefs` prints nothing: the belief's only evidence was in session 1, so it went with it.
   Start over any time by deleting `agent.db`.
4. **Choose a real model.** Put your settings in a file called `agent.json` next to this README.
   - **A free local model with [Ollama](https://ollama.com).** Install Ollama, then `ollama pull llama3.1:8b`
     (about 5 GB, the longest step on a slow connection). Then:
     ```json
     {"backend": "ollama", "model": "llama3.1:8b"}
     ```
     On a laptop without a GPU a reply can take 20 to 60 seconds.
   - **Your own API key.** Put the key in an environment variable, never in `agent.json`:
     `agent.json` holds only the variable's *name*.
     ```json
     {"backend": "anthropic"}
     ```
     reads `ANTHROPIC_API_KEY` (Claude Opus 5.5 by default; set `"model"` to change it). It asks
     Anthropic to retry a message on a recommended model if a safety check declines it
     (`"fallbacks"` in `models.py`; delete it there if your setup doesn't accept it).
     ```json
     {"backend": "openai", "model": "THE-MODEL-NAME", "api_key_env": "OPENAI_API_KEY"}
     ```
     works with any server that speaks the OpenAI chat API (set `"base_url"` for vLLM, llama.cpp or LM Studio).
5. **Optional: better recall.** The default embedder matches shared words and spellings, not meaning.
   `pip install sentence-transformers` and add `"embedder": "bge-small"` to `agent.json`; the first use
   downloads the bge-small-en-v1.5 model (about 130 MB).

## Commands

`python -m starter_agent COMMAND`, or `python -m starter_agent --help`.

| Command | What it does |
|---|---|
| `chat "message"` | one turn; `chat` alone talks until `/quit` (`/end` ends the session) |
| `session end` | end the session now; curiosity reads it once |
| `session list` / `session show ID` | your sessions, or one whole |
| `session delete ID` | delete a session and what it alone taught the agent (asks first; `--yes` skips) |
| `desire add "TEXT" [--rank N]` | something you want the agent to work toward; rank 1 is the top |
| `desire list` / `desire rank ID N` / `desire drop ID` | your live desires; change a rank; stop one |
| `intentions` | what the agent is working on: each serves a live desire, and one is current |
| `beliefs` / `beliefs history ID` | current beliefs; every version of one, with its evidence |
| `curious list` | open questions, each with the agent's own words it came from |
| `curious drop ID` / `curious adopt ID [--rank N]` | turn a question down (it won't come back); make it a desire |
| `recall WORDS` | exactly what recall would hand the model for these words |
| `log` | what was dropped, and why |

**Who decides what.** You write and rank the desires. Each turn, plain code (not the model) picks the
intention that serves your highest-ranked live desire whose intention isn't done. The model can suggest a
re-ranking in its reply; only your command changes a rank. A question becomes a desire only when you say
`curious adopt` (you can switch on `auto_adopt`; it's off by default).

## What it remembers, and how to delete it

Everything is in `agent.db`: sessions, every turn, an embedding of each of your messages, beliefs, desires,
intentions, curiosity's questions, and a log of what was rejected. Open it with any SQLite viewer
(`sqlite3 agent.db .tables`). With Ollama nothing leaves your machine. With an API key, each turn sends the
recalled sessions, the current beliefs and this session to that provider, and curiosity sends the agent's
own turns once when a session ends.

Records are never edited in place: a change appends a new row that points at the old one (`supersedes`),
so `beliefs history ID` shows how a view moved and why. Deleting is the one exception, because your word
wins.

**When you delete a session, no text written in it stays in the file.** `session delete ID` removes:

- the session, its turns, their embeddings and its log rows;
- **every belief version written during that session, whatever it cites**, even one that cites only an
  older session's turns: its words were written there. Also every belief version whose evidence is all
  gone (in this session, or in one you deleted before). A content-free marker stays in each one's place
  ("a belief derived from a deleted session was removed on DATE"), so the history is honest about the gap;
- a belief that lost a version this way **falls back to its latest version written outside the deleted
  sessions whose evidence still exists**, flagged for your review. If there is none, the belief is just
  the marker;
- that session's turns from the evidence of beliefs written in *other* sessions that also cite other
  evidence. Those beliefs stay, keeping only the evidence that's left, flagged for your review, because
  the agent can't tell whether the confidence still holds;
- that session's turns from the evidence of a done intention (with no evidence left, it's active again);
- the questions curiosity found in it, and any desire adopted from them, which become markers too.

Then the file is rebuilt (SQLite's `VACUUM`) so the deleted text doesn't linger in its free space.

What it can't reach, plainly: text written in a *later* session stays, even when it repeats the deleted
session. That covers the agent's replies there, and a belief written there that cited the deleted session
alongside other evidence (its wording stays, flagged for review). Delete that later session too if you
want its words gone. Copies outside the file (your backups, an API provider's logs) are outside its reach.
To delete everything, delete `agent.db`.

## Settings (`agent.json`)

| Setting | Default | |
|---|---|---|
| `db` | `agent.db` | the one file |
| `backend` | `fake` | `fake`, `ollama`, `openai` or `anthropic` |
| `model`, `base_url` | blank | blank means the backend's usual one |
| `api_key_env` | blank | the NAME of the variable holding your key |
| `embedder` | `hash` | `hash` (needs nothing) or `bge-small` |
| `recall_budget_tokens` | 8000 | recalled sessions per turn, counted as about 4 characters a token |
| `idle_minutes` | 30 | a session ends after this long without a message. Nothing runs in the background, so it ends at your next `chat` or `session end`; `curious list` tells you when one is waiting |
| `question_queue` | 20 | the most open questions; past it, the oldest expires (it isn't deleted) |
| `dedupe_similarity` | 0.85 | a new question this close to an open or dropped one is skipped |
| `auto_adopt` | false | turn new questions into desires without asking you |

## Reading the code

In this order: `agent.py` (the loop, 90 lines), `recall.py`, `bdi.py`, `curiosity.py`, `models.py`,
`store.py`, `embed.py`, `cli.py`. Each file starts with a note on what it does.

- **Recall** scores each of your past messages two ways, BM25 (shared words) and the embedder (similar
  meaning), fuses the two rankings (reciprocal rank fusion, k = 60, embeddings weighted 2 and words 1),
  scores each session by its best two messages, and hands over whole sessions, best first, within the
  budget, never half of one. Inside the pack they run oldest to newest. Only your messages are indexed, so
  the agent's own words can't pull a session up.
- **Beliefs** change only through the JSON block at the end of the model's one reply per turn. A change is
  stored only if it cites turns the model was shown (from the recalled sessions or this one) and a
  confidence from 0 to 1; otherwise it's dropped and logged (`log`).
- **Curiosity** runs once when a session ends: one model call reads the agent's own turns and lists the
  questions it raised and didn't answer. Each must quote 2 to 20 of the agent's words exactly, from the
  turn it names, or it's dropped. It never searches outside and never talks to you on its own.

**How we know the tests can fail.** `python tests/mutants.py` copies the code to a scratch folder, plants
one bug at a time (each listed with the promise it breaks), and checks that a test fails by assertion.

## Credits

- The memory design is from Nexus's Mnemosyne.
- The beliefs-and-intentions core is from Kintsugi, whose design began in Thomas's Orion.
- Curiosity is Lyra's design.
