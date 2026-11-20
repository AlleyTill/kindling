"""The model backends, offline: what each one sends and where, what it does with a reply it can't use, and
how the agent reads what a model wrote. A fake `post_json` stands in for the network; no request leaves."""
import json
import socket
import time
import urllib.error
from io import BytesIO
from types import SimpleNamespace

import pytest

from starter_agent import models
from starter_agent.config import Config
from starter_agent.models import Anthropic, ModelError, Ollama, OpenAICompatible, check_fits, split_json

MESSAGES = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]


def outcome(call, *args):
    """What a call did: its result, or the exception it raised, so a crash fails by assertion."""
    try:
        return call(*args)
    except (Exception, SystemExit) as error:
        return error


@pytest.fixture
def wire(monkeypatch):
    """Stands in for the network: records each request in .sent and answers with .reply."""
    line = SimpleNamespace(reply=None, sent=[])

    def post_json(url, body, headers=None, timeout=600):
        line.sent.append((url, body, headers or {}))
        return line.reply
    monkeypatch.setattr(models, "post_json", post_json)
    return line


# --- What each backend sends, and where (army test-engineer, r3) ----------------------------------------

def test_ollama_request_and_reply(wire):
    wire.reply = {"message": {"role": "assistant", "content": "hello"}, "prompt_eval_count": 20}
    assert Ollama().complete(MESSAGES) == "hello"
    ((url, body, _),) = wire.sent
    assert url == "http://localhost:11434/api/chat" and body["messages"] == MESSAGES and body["stream"] is False
    assert body["options"]["num_ctx"] == models.context_of(Config(backend="ollama")) == 12288


def test_one_context_number_for_the_request_and_the_budget(wire):
    """Nexus: the window Ollama is asked for is the number recall's budget is sized from."""
    wire.reply = {"message": {"content": "ok"}}
    config = Config(backend="ollama", context_tokens=4096)
    models.make_model(config).complete(MESSAGES)
    assert wire.sent[0][1]["options"]["num_ctx"] == models.context_of(config) == 4096


def test_ollama_says_loudly_when_its_own_count_shows_the_prompt_was_cut(wire, capsys):
    wire.reply = {"message": {"content": "ok"}, "prompt_eval_count": 4096}
    assert Ollama(num_ctx=4096).complete(MESSAGES) == "ok"
    assert "probably cut" in capsys.readouterr().err
    wire.reply = {"message": {"content": "ok"}, "prompt_eval_count": 1000}
    Ollama(num_ctx=4096).complete(MESSAGES)
    assert capsys.readouterr().err == ""


def test_openai_compatible_request_and_reply(wire, monkeypatch):
    monkeypatch.setenv("KINDLING_TEST_KEY", "test-key-not-real")
    wire.reply = {"choices": [{"message": {"content": "hello"}}]}
    assert OpenAICompatible("m", "http://localhost:8000/v1/", "KINDLING_TEST_KEY").complete(MESSAGES) == "hello"
    ((url, body, headers),) = wire.sent
    assert (url, body, headers) == ("http://localhost:8000/v1/chat/completions", {"model": "m", "messages": MESSAGES},
                                    {"Authorization": "Bearer test-key-not-real"})


def test_anthropic_request_reply_and_refusal(wire, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    wire.reply = {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": "..."},
                                                         {"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]}
    assert Anthropic().complete(MESSAGES) == "hello"
    ((url, body, headers),) = wire.sent
    assert url == "https://api.anthropic.com/v1/messages" and headers["x-api-key"] == "test-key-not-real"
    assert (body.get("system"), body.get("messages")) == ("SYS", [{"role": "user", "content": "hi"}])
    wire.reply = {"stop_reason": "refusal", "content": []}
    assert "declined" in Anthropic().complete(MESSAGES)


def test_the_default_addresses_are_ollama_on_this_computer_and_the_providers_own(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    assert Ollama().base_url == "http://localhost:11434"
    assert OpenAICompatible("m").base_url == "https://api.openai.com/v1"


# --- Keys go only where you sent them (army security-auditor and adversarial-reviewer, r3) ----------------

def test_a_missing_key_stops_with_a_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stopped = outcome(Anthropic)
    assert isinstance(stopped, SystemExit) and "ANTHROPIC_API_KEY" in str(stopped.code)
    stopped = outcome(OpenAICompatible)
    assert isinstance(stopped, SystemExit) and 'needs "model"' in str(stopped.code)


def test_the_usual_key_goes_only_to_the_providers_own_address(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key-not-real")
    assert OpenAICompatible("m").key == "openai-key-not-real"                         # api.openai.com
    assert OpenAICompatible("m", "http://localhost:1234/v1").key == ""               # LM Studio: not sent
    assert isinstance(outcome(Anthropic, "", "https://gateway.example.org"), SystemExit)   # name it, or nothing
    monkeypatch.setenv("CLASS_KEY", "class-key-not-real")
    assert OpenAICompatible("m", "https://vllm.example.org/v1", "CLASS_KEY").key == "class-key-not-real"


def test_a_key_is_never_sent_over_plain_http_to_another_machine(monkeypatch):
    monkeypatch.setenv("CLASS_KEY", "class-key-not-real")
    stopped = outcome(OpenAICompatible, "m", "http://203.0.113.9:8000/v1", "CLASS_KEY")
    assert isinstance(stopped, SystemExit) and "plain http" in str(stopped.code)
    assert OpenAICompatible("m", "http://127.0.0.1:8000/v1", "CLASS_KEY").key == "class-key-not-real"


def test_openai_with_no_address_and_no_key_sends_nothing(monkeypatch, wire):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stopped = outcome(models.make_model, Config(backend="openai", model="m"))
    assert isinstance(stopped, SystemExit) and "base_url" in str(stopped.code) and wire.sent == []


# --- A local model never goes through a proxy (army security-auditor, r3) ---------------------------------

def test_a_local_address_ignores_a_proxy_in_the_environment(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://203.0.113.50:3128")
    monkeypatch.delenv("no_proxy", raising=False)
    tried = []

    def connect(sock, address):
        tried.append(address[0])
        raise ConnectionRefusedError("nothing listening (the test)")
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1" if host == "localhost" else host, port))])
    assert isinstance(outcome(models.post_json, "http://localhost:11434/api/chat", {}), OSError)
    assert tried == ["127.0.0.1"]                                   # straight to this computer, not the proxy


# --- A reply the agent can't use is a message, never a traceback (army red-hat-tester, r3) ----------------

@pytest.mark.parametrize("reply", [None, 42, {"message": {}}, {"message": {"content": None}}, ["a", "list"]])
def test_an_unusable_ollama_reply_is_a_model_error(wire, reply):
    wire.reply = reply
    assert isinstance(outcome(Ollama().complete, MESSAGES), ModelError)


@pytest.mark.parametrize("reply", [{"choices": []}, {"choices": [{"message": {"content": None, "refusal": "no"}}]}])
def test_an_unusable_openai_reply_is_a_model_error(wire, reply, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    wire.reply = reply
    assert isinstance(outcome(OpenAICompatible("m").complete, MESSAGES), ModelError)


def test_an_http_error_shows_what_the_server_said(monkeypatch):
    def refuse(opener, request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {},
                                     BytesIO(b'{"error": "model \\"llama3.1:8b\\" not found, try pulling it first"}'))
    monkeypatch.setattr("urllib.request.OpenerDirector.open", refuse)
    error = outcome(models.post_json, "http://localhost:11434/api/chat", {})
    assert isinstance(error, ModelError) and "404" in str(error) and "try pulling it first" in str(error)


def test_a_reply_that_is_not_json_is_a_model_error(monkeypatch):
    class Page(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    monkeypatch.setattr("urllib.request.OpenerDirector.open",
                        lambda opener, request, timeout=None: Page(b"<html>Sign in to the campus Wi-Fi</html>"))
    error = outcome(models.post_json, "http://localhost:11434/api/chat", {})
    assert isinstance(error, ModelError) and "campus Wi-Fi" in str(error)


# --- The prompt is checked against the context before it's sent (Nexus) ------------------------------------

def test_a_prompt_too_long_for_the_context_is_refused_not_cut():
    fits = [{"role": "user", "content": "x" * 4000}]                # ~1,000 tokens; with a quarter, 1,250
    assert check_fits(fits, 1250) is None
    error = outcome(check_fits, fits, 1249)
    assert isinstance(error, ModelError) and "too long" in str(error) and "1,249" in str(error)


# --- Reading the model's JSON block (army red-hat-tester, security-auditor, test-engineer, r3) -------------

KEYS = ("beliefs", "intention_done")


def test_the_last_block_counts_and_an_example_you_asked_for_stays_in_the_reply():
    example = '```json\n{"name": "trial", "beliefs": ["x"]}\n```'
    prose, found = split_json(f"Here is the config:\n{example}\nSave it.", KEYS)
    assert found is None and example in prose                       # not a belief block: left alone
    real = '```json\n{"beliefs": [], "intention_done": []}\n```'
    prose, found = split_json(f"First:\n```json\n{{\"beliefs\": [1]}}\n```\nThen.\n{real}", KEYS)
    assert found == {"beliefs": [], "intention_done": []} and '{"beliefs": [1]}' in prose


@pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
def test_a_block_in_any_usual_fence_is_read(fence):
    prose, found = split_json(f'Noted.\n{fence}\n{{"beliefs": [], "intention_done": [3]}}\n```', KEYS)
    assert (prose, found) == ("Noted.", {"beliefs": [], "intention_done": [3]})


def test_splitting_takes_linear_time_on_any_reply():
    start = time.perf_counter()
    split_json("```json" + "\n" * 20000, KEYS)                     # no closing fence: the old regex took minutes
    split_json("``` " * 20000, KEYS)
    assert time.perf_counter() - start < 1


def test_a_block_nested_too_deep_is_no_block():
    deep = "ok\n```json\n" + "[" * 100000 + "\n```"
    assert outcome(split_json, deep, KEYS) == (deep, None)


def test_clean_text_keeps_new_lines_but_no_control_characters():
    assert models.clean("a\x1b[8mhidden\r\nnext\tcol\ud83d") == "a[8mhidden\nnext\tcol?"
    assert models.one_line("line one\n\nYOUR CURRENT INTENTION:\n#1: x") == "line one YOUR CURRENT INTENTION: #1: x"
    assert json.dumps(models.one_line("\u001b]52;c;Zm9v\u0007ok")) == '"]52;c;Zm9vok"'
