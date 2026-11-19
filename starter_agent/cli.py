"""The command line. Run `python -m starter_agent --help`."""
import argparse
import json
import sys

from . import bdi, curiosity
from .agent import Agent, utc_now
from .config import load_config
from .embed import HashEmbedder, make_embedder
from .models import make_model
from .recall import render_session
from .store import Store, local


HASH_NOTE = ("Note: the hash embedder is in use (the default). It matches shared words and spellings, not "
             "meaning, so its recall and similarity scores don't measure the memory design. "
             "See 'Optional: better recall' in the README.")


def build(config, model=None, clock=utc_now):
    store = Store(config.db, clock)
    embedder = make_embedder(config.embedder)
    if isinstance(embedder, HashEmbedder):
        print(HASH_NOTE, file=sys.stderr)                 # once, at startup
    return Agent(store, model or make_model(config), embedder, config)


def ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def show_belief(b):
    if b["flag"] == "removed":
        return f"#{b['id']}  [removed] {b['text']}"
    why = ", ".join(str(t) for t in json.loads(b["evidence"]))
    line = f"#{b['id']}  {b['confidence']:.2f}  {b['text']}  (evidence: turns {why})"
    if b["supersedes"]:
        line += f"  supersedes #{b['supersedes']}"
    return line + ("  [flagged for your review]" if b["flag"] == "review" else "")


def cmd_chat(agent, a):
    if a.message:
        print(agent.handle(" ".join(a.message)))
        return 0
    print("Type a message. /end ends the session, /quit leaves (the session stays open).")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if text == "/quit":
            return 0
        if text == "/end":
            print(f"Session ended; curiosity queued {len(agent.end_session())} question(s).")
        elif text:
            print("agent>", agent.handle(text))


def cmd_session(agent, a):
    store = agent.store
    if a.action == "end":
        queued = agent.end_session()
        print(f"Session ended; curiosity queued {len(queued)} question(s). See: curious list")
    elif a.action == "list":
        for s in store.sessions():
            ended = local(s["ended_at"]) if s["ended_at"] else "(open)"
            print(f"#{s['id']}  {local(s['started_at'])} to {ended}  {s['title'] or ''}")
    elif a.id is None or store.session(a.id) is None:
        print("Give the id of an existing session (see: session list).")
        return 1
    elif a.action == "show":
        print(render_session(store, a.id))
    elif a.action == "delete":
        if not a.yes and ask(f"Delete session {a.id} and what it alone taught the agent? "
                             "This can't be undone. Type yes: ") != "yes":
            print("Nothing deleted.")
            return 1
        print(json.dumps(agent.delete_session(a.id), indent=1))
    return 0


def cmd_desire(agent, a):
    store = agent.store
    if a.action == "add":
        if not " ".join(a.args).strip():
            print('Say what the agent should want, e.g.: desire add "Replicate figure 3" --rank 1')
            return 1
        print(f"Desire #{bdi.add_desire(store, ' '.join(a.args), a.rank)} added.")
    elif a.action == "list":
        for d in bdi.live_desires(store):
            print(f"#{d['id']}  rank {d['rank']}  {d['text']}  ({d['source']})")
    else:
        try:
            desire_id = int(a.args[0])
            desire = store.latest("desires", desire_id)
            if desire is None or desire["flag"] == "removed":
                raise ValueError
            new_rank = a.rank if a.rank is not None or a.action == "drop" else int(a.args[1])
        except (IndexError, ValueError):
            print("Give the id of a desire (see: desire list), and for rank the new rank: desire rank ID N")
            return 1
        if a.action == "rank":
            print(f"Desire is now #{bdi.rerank_desire(store, desire_id, new_rank)}, rank {new_rank}.")
        else:
            print(f"Desire dropped (#{bdi.drop_desire(store, desire_id)}).")
    return 0


def cmd_beliefs(agent, a):
    if a.action == "history":
        rows = agent.store.history("beliefs", a.id) if a.id is not None else []
        if not rows:
            print("Give the id of a belief (see: beliefs).")
            return 1
    else:
        rows = bdi.current_beliefs(agent.store)
    for b in rows:
        print(show_belief(b))
    return 0


def cmd_intentions(agent, a):
    """Read-only, like every list: no model call, nothing written."""
    desire, current = bdi.choose(agent.store)          # plain code, the same pick a chat turn makes
    live = {d["key"] for d in bdi.live_desires(agent.store)}
    for i in agent.store.current("intentions", "flag IS NOT 'removed'"):
        if i["serves"] in live:                         # not one whose desire you dropped
            mark = "  (current)" if current and i["id"] == current["id"] else ""
            print(f"#{i['id']}  {i['status']}  {i['text']}{mark}")
    if desire is not None and current is None:
        print(f"(new)  active  {desire['text']}  (current; your next chat turn writes it down)")
    return 0


def cmd_curious(agent, a):
    store = agent.store
    if a.action == "list":                   # read-only: it never ends a session or calls the model
        for q in curiosity.open_questions(store):
            print(f"#{q['id']}  {q['text']}\n      \"{q['quote']}\" (turn {q['source_turn']}, "
                  f"session {q['session_id']})")
        idle = agent.idle_session()
        if idle:
            print(f"Session {idle['id']} has been idle over {agent.config.idle_minutes} minutes but is still "
                  "open; curiosity reads it once it ends.\nTo end it now: python -m starter_agent session end")
        return 0
    act = curiosity.drop if a.action == "drop" else (lambda s, i: curiosity.adopt(s, i, a.rank))
    result = act(store, a.id) if a.id is not None else None
    if result is None:
        print("Give the id of an open question (see: curious list).")
        return 1
    print("Dropped. It stays in the file, and a new question in nearly the same words won't be queued."
          if a.action == "drop"
          else f"Adopted as desire #{result}.")
    return 0


def cmd_recall(agent, a):
    pack = agent.memory.recall(" ".join(a.query), agent.config.recall_budget_tokens)
    for session, score in pack.sessions:
        print(f"session {session}  score {score:.4f}  ({pack.scored_by})")
    print(f"\n{pack.text}" if pack.sessions else "Nothing recalled.")
    return 0


def cmd_log(agent, a):
    for row in agent.store.db.execute("SELECT * FROM log ORDER BY id"):
        print(f"#{row['id']}  {local(row['at'])}  session {row['session_id']}  {row['kind']}: {row['reason']}")
    return 0


def main(argv=None, model=None, clock=utc_now):
    p = argparse.ArgumentParser(prog="python -m starter_agent",
                                description="A small agent that remembers, keeps beliefs, and wonders.")
    p.add_argument("--config", default="agent.json", help="settings file (default: agent.json)")
    p.add_argument("--db", help="the database file (default: agent.db)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("chat", help="send one message, or none to talk until /quit")
    s.add_argument("message", nargs="*")
    s = sub.add_parser("session", help="end | list | show ID | delete ID [--yes]")
    s.add_argument("action", choices=["end", "list", "show", "delete"])
    s.add_argument("id", nargs="?", type=int)
    s.add_argument("--yes", action="store_true", help="delete without asking")
    s = sub.add_parser("desire", help='add "TEXT" [--rank N] | list | rank ID N | drop ID')
    s.add_argument("action", choices=["add", "list", "rank", "drop"])
    s.add_argument("args", nargs="*")
    s.add_argument("--rank", type=int, help="1 is the top; default: the bottom of the list")
    s = sub.add_parser("beliefs", help="the current beliefs, or: history ID")
    s.add_argument("action", nargs="?", default="list", choices=["list", "history"])
    s.add_argument("id", nargs="?", type=int)
    sub.add_parser("intentions", help="the intentions and their status")
    s = sub.add_parser("curious", help="list | drop ID | adopt ID [--rank N]")
    s.add_argument("action", choices=["list", "drop", "adopt"])
    s.add_argument("id", nargs="?", type=int)
    s.add_argument("--rank", type=int)
    s = sub.add_parser("recall", help="what recall would hand the model for this query")
    s.add_argument("query", nargs="+")
    sub.add_parser("log", help="what was dropped, and why")
    a = p.parse_args(argv)
    agent = build(load_config(a.config, db=a.db), model, clock)
    try:
        return globals()[f"cmd_{a.command}"](agent, a)
    except OSError as error:
        print(f"Could not reach the model: {error}. See 'Choose a real model' in the README.")
        return 1
