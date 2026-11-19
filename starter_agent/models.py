"""The one model function: complete(messages) -> text (plan section 2).

`messages` is a list of {"role": "system" | "user" | "assistant", "content": text}. Every part of the
agent calls a model only through `.complete`, so no part depends on which backend you pick.

The backends talk plain HTTP with Python's own urllib, so nothing needs installing:
- FakeModel: scripted and deterministic. The tests use it; with no script it runs the demo below.
- Ollama: a local model at http://localhost:11434 (https://ollama.com).
- OpenAICompatible: any server that speaks the OpenAI chat API (OpenAI, vLLM, llama.cpp, LM Studio...).
- Anthropic: Claude, with your own key in an environment variable.
"""
import json
import os
import re
import urllib.request


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


def post_json(url, body, headers=None, timeout=600):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Ollama:
    def __init__(self, model="", base_url="", num_ctx=12288):
        self.model = model or "llama3.1:8b"
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.num_ctx = num_ctx    # Ollama's default window is smaller than an 8,000-token recall pack

    def complete(self, messages):
        out = post_json(f"{self.base_url}/api/chat", {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"num_ctx": self.num_ctx}})
        return out["message"]["content"]


class OpenAICompatible:
    def __init__(self, model="", base_url="", api_key_env=""):
        if not model:
            raise SystemExit('The openai backend needs "model" in agent.json.')
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get(api_key_env or "OPENAI_API_KEY", "")   # local servers need none

    def complete(self, messages):
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        out = post_json(f"{self.base_url}/chat/completions",
                        {"model": self.model, "messages": messages}, headers)
        return out["choices"][0]["message"]["content"]


class Anthropic:
    def __init__(self, model="", base_url="", api_key_env=""):
        name = api_key_env or "ANTHROPIC_API_KEY"
        self.key = os.environ.get(name)
        if not self.key:
            raise SystemExit(f"The anthropic backend reads your key from the {name} environment variable.")
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
        if out.get("stop_reason") == "refusal":     # declined: there is no reply to read
            return "(The model declined to answer this message.)"
        return "".join(block.get("text", "") for block in out["content"] if block.get("type") == "text")


def make_model(config):
    if config.backend == "fake":
        return FakeModel()
    if config.backend == "ollama":
        return Ollama(config.model, config.base_url)
    if config.backend == "openai":
        return OpenAICompatible(config.model, config.base_url, config.api_key_env)
    if config.backend == "anthropic":
        return Anthropic(config.model, config.base_url, config.api_key_env)
    raise SystemExit(f'Unknown backend "{config.backend}": use fake, ollama, openai or anthropic.')


FENCE = re.compile(r"```json\s*(.*?)\s*```", re.S)


def split_json(text):
    """Split a model's output into (the prose, the text of its last ```json block, or None)."""
    blocks = list(FENCE.finditer(text))
    if not blocks:
        return text.strip(), None
    last = blocks[-1]
    return (text[:last.start()] + text[last.end():]).strip(), last.group(1)


def parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None
