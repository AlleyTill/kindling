"""The one model function: complete(messages) -> text (plan section 2).

`messages` is a list of {"role": "system" | "user" | "assistant", "content": text}. Every part of the
agent calls a model only through `.complete`, so no part depends on which backend you pick.

The backends talk plain HTTP with Python's own urllib, so nothing needs installing:
- FakeModel: scripted and deterministic. The tests use it; with no script it runs the demo below.
- Ollama: a local model at http://localhost:11434 (https://ollama.com).
- OpenAICompatible: any server that speaks the OpenAI chat API (OpenAI, vLLM, llama.cpp, LM Studio...).
- Anthropic: Claude, with your own key in an environment variable.
"""
import http.client
import ipaddress
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


class ModelError(Exception):
    """The model's server answered with something the agent can't use, or the prompt can't be sent."""


class FakeModel:
    """Give it a list of replies (used in order) or a function of the messages."""

    def __init__(self, replies=None, fn=None):
        self.replies = list(replies or [])
        self.fn = demo_brain if fn is None and replies is None else fn
        self.calls = []                       # every list of messages it was sent, for tests to read

    def complete(self, messages):
        self.calls.append(messages)
        return self.fn(messages) if self.fn else self.replies.pop(0)


def demo_brain(messages):
    """The no-key demo: a 'model' made of three rules, so you can watch every part work.
    - a message starting "remember:" becomes a belief, citing your turn as its evidence;
    - a message ending in "?" gets "I'm not sure about that yet", which curiosity will notice;
    - at the end of a session, curiosity quotes that phrase back as an open question."""
    system, prompt = messages[0]["content"], messages[-1]["content"]
    if '"questions"' in system:               # the curiosity call
        found = re.findall(r"\[turn (\d+)\] agent: You said: (.*?) I'm not sure about that yet\.", prompt)
        return json.dumps({"questions": [
            {"text": said, "quote": "I'm not sure about that yet", "turn": int(turn)} for turn, said in found]})
    earlier, _, now = prompt.partition("THIS SESSION SO FAR:")
    turn, said = re.findall(r"\[turn (\d+)\] user: (.*)", now)[-1]
    reply = f"You said: {said}"
    if said.rstrip().endswith("?"):
        reply += " I'm not sure about that yet."
    recalled = re.findall(r"=== Session (\d+)", earlier)
    if recalled:
        reply += f" (I recalled session {', '.join(recalled)}.)"
    beliefs = []
    if said.lower().startswith("remember:"):
        beliefs.append({"text": said[len("remember:"):].strip(), "confidence": 0.6, "evidence": [int(turn)]})
    return reply + "\n```json\n" + json.dumps({"beliefs": beliefs, "intention_done": []}) + "\n```"


# --- The context window: one number, read by the request and by recall's budget (Nexus) ---------------

CONTEXT = {"fake": 12288, "ollama": 12288, "openai": 16384, "anthropic": 1_000_000}   # each backend's usual


def context_of(config):
    """The model's context window in tokens: "context_tokens" in agent.json, or the backend's usual one.
    Ollama is asked for exactly this window (num_ctx), and recall's budget is sized from the same number."""
    return config.context_tokens or CONTEXT.get(config.backend, 12288)


def estimate_tokens(text):
    return (len(text) + 3) // 4         # about four characters a token; no tokenizer needed


def check_fits(messages, context):
    """Refuse, loudly, a prompt too long for the context, before it's sent: a server like Ollama would cut
    it without telling the model. Four characters a token undercounts code, numbers and non-English text,
    so the prompt must fit with a quarter to spare."""
    size = estimate_tokens("".join(m["content"] for m in messages))
    if size * 5 // 4 > context:
        raise ModelError(f"The prompt is about {size:,} tokens: too long for the model's {context:,}-token "
                         "context, so it wasn't sent (it would be cut without the model knowing). End this "
                         'session (/end, or `session end`), or raise "context_tokens" in agent.json')


# --- Talking to a server -----------------------------------------------------------------------------

def is_local(url):
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def post_json(url, body, headers=None, timeout=600):
    """POST `body` as JSON; return the reply's JSON. A local address never goes through a proxy set in
    your environment (http_proxy), so a local model stays on your machine."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    opener = urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if is_local(url) else []))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:              # reached, but it said no: show what it said
        raise ModelError(f"The model server answered HTTP {error.code}: "
                         f"{error.read().decode('utf-8', 'replace')[:300]}") from None
    except http.client.HTTPException as error:          # e.g. the reply was cut off part way
        raise ModelError(f"The model server's reply broke off: {error!r}") from None
    try:
        return json.loads(raw)
    except ValueError:
        start = raw[:200].decode("utf-8", "replace")
        raise ModelError(f"The model server's reply wasn't JSON: {start!r}") from None


def text_of(reply):
    """The model's reply, if it is text."""
    if not isinstance(reply, str):
        raise ModelError(f"The model's reply wasn't text: {reply!r:.200}")
    return reply


def reply_at(out, *path):
    """The text at out[path[0]][path[1]]..., or a ModelError saying the reply wasn't the expected shape."""
    try:
        for step in path:
            out = out[step]
        return text_of(out)
    except (KeyError, IndexError, TypeError):
        raise ModelError(f"The model server's reply had no {'/'.join(map(str, path))}: {out!r:.200}") from None


def key_for(name, usual, base_url):
    """Your key, from the environment variable named in "api_key_env". With no name, the usual variable is
    read only when "base_url" is blank (the provider's own address): a key never goes to a server you didn't
    name it for, and never over plain http to another machine."""
    key = os.environ.get(name or usual, "") if name or not base_url else ""
    if key and base_url.startswith("http://") and not is_local(base_url):
        raise SystemExit(f"Refusing to send your key over plain http to {base_url}: use https.")
    return key


class Ollama:
    def __init__(self, model="", base_url="", num_ctx=CONTEXT["ollama"]):
        self.model = model or "llama3.1:8b"
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.num_ctx = num_ctx    # Ollama's own default window is smaller than an 8,000-token recall pack

    def complete(self, messages):
        out = post_json(f"{self.base_url}/api/chat", {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"num_ctx": self.num_ctx}})
        used = out.get("prompt_eval_count") if isinstance(out, dict) else None
        if isinstance(used, int) and used >= self.num_ctx * 0.95:     # the model's own count: probably cut
            print(f"WARNING: Ollama read {used:,} prompt tokens, at its {self.num_ctx:,}-token window: the "
                  "prompt was probably cut, and the model never saw what was cut. End this session, or raise "
                  '"context_tokens" in agent.json.', file=sys.stderr)
        return reply_at(out, "message", "content")


class OpenAICompatible:
    def __init__(self, model="", base_url="", api_key_env=""):
        if not model:
            raise SystemExit('The openai backend needs "model" in agent.json.')
        self.key = key_for(api_key_env, "OPENAI_API_KEY", base_url)
        if not base_url and not self.key:      # nothing would stop it going to OpenAI, without a key
            raise SystemExit('The openai backend with no "base_url" sends to OpenAI: set "base_url" for a '
                             "local server, or put your OpenAI key in OPENAI_API_KEY.")
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def complete(self, messages):
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        out = post_json(f"{self.base_url}/chat/completions",
                        {"model": self.model, "messages": messages}, headers)
        return reply_at(out, "choices", 0, "message", "content")


class Anthropic:
    def __init__(self, model="", base_url="", api_key_env=""):
        self.key = key_for(api_key_env, "ANTHROPIC_API_KEY", base_url)
        if not self.key:
            raise SystemExit("The anthropic backend reads your key from the environment variable named in "
                             '"api_key_env" (ANTHROPIC_API_KEY when "base_url" is blank).')
        self.model = model or "claude-opus-5-5"
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")

    def complete(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        out = post_json(f"{self.base_url}/v1/messages",
                        # max_tokens is required. A reply here is ~300 tokens (plan section 7), but
                        # current Claude models think first and thinking counts against it; ~16,000 is
                        # the usual ceiling for a request that isn't streamed.
                        # "fallbacks": if a safety check declines a harmless message, Anthropic retries it
                        # on the model it recommends instead of failing. Delete it and the beta header
                        # if your model or provider doesn't accept them.
                        {"model": self.model, "max_tokens": 16000, "system": system, "messages": rest,
                         "fallbacks": "default"},
                        {"x-api-key": self.key, "anthropic-version": "2026-10-21",
                         "anthropic-beta": "server-side-fallback-2026-07-01"})
        if isinstance(out, dict) and out.get("stop_reason") == "refusal":     # declined: there's no reply
            return "(The model declined to answer this message.)"
        blocks = out.get("content") if isinstance(out, dict) else None
        if not isinstance(blocks, list):
            raise ModelError(f"The model server's reply had no content: {out!r:.200}")
        return "".join(text_of(b.get("text")) for b in blocks if isinstance(b, dict) and b.get("type") == "text")


def make_model(config):
    if config.backend == "fake":
        return FakeModel()
    if config.backend == "ollama":
        return Ollama(config.model, config.base_url, context_of(config))
    if config.backend == "openai":
        return OpenAICompatible(config.model, config.base_url, config.api_key_env)
    if config.backend == "anthropic":
        return Anthropic(config.model, config.base_url, config.api_key_env)
    raise SystemExit(f'Unknown backend "{config.backend}": use fake, ollama, openai or anthropic.')


# --- Reading what the model wrote --------------------------------------------------------------------

CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean(text):
    """Text as it's stored and shown: valid UTF-8 (a broken character becomes ?), and none of the terminal
    control characters that could hide or rewrite what your screen shows. Newlines and tabs stay."""
    return CONTROL.sub("", text.encode("utf-8", "replace").decode("utf-8"))


def one_line(text):
    """A belief or a question: clean and on one line, so it can't pose as another part of the prompt."""
    return " ".join(clean(text).split())


def split_json(text, keys=("beliefs", "intention_done")):
    """Split a model's output into (the prose, its last fenced JSON block as an object), or (text, None).
    The block counts only if it is an object whose keys are all in `keys`, so an example block you asked
    for stays in the reply. The fence may be ```json, ```JSON or bare ```. Found with str.rfind, not a
    regular expression, so any reply is split in linear time."""
    close = text.rfind("```")
    start = text.rfind("```", 0, close) if close > 0 else -1
    if start < 0:
        return text.strip(), None
    inner = text[start + 3:close]
    found = parse_json(inner[4:] if inner[:4].lower() == "json" else inner)
    if not isinstance(found, dict) or not found or set(found) - set(keys):
        return text.strip(), None
    return (text[:start] + text[close + 3:]).strip(), found


def parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return None
