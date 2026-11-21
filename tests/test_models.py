"""The model backends, offline: what each one sends and where, what it does with a reply it can't use, and
how the agent reads what a model wrote. A fake urlopen stands in for every server; no request leaves."""
import json
import socket
import time
import urllib.error
from io import BytesIO
from types import SimpleNamespace

import pytest

from conftest import Clock, add_session
from starter_agent import models
from starter_agent.agent import Agent
from starter_agent.config import Config
from starter_agent.embed import HashEmbedder
from starter_agent.models import (Anthropic, ModelError, Ollama, OpenAICompatible, PromptTooLong, check_fits,
                                  estimate_tokens, split_json)
from starter_agent.store import Store

MESSAGES = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
OLLAMA_VERSION = "http://localhost:11434/api/version"
TOO_LONG = {"error": "request (1757 tokens) exceeds the available context size (256 tokens), try increasing it",
            "type": "exceed_context_size_error", "n_prompt_tokens": 1757, "n_ctx": 256}   # Ollama 0.32.1's (Nexus)


def outcome(call, *args):
    """What a call did: its result, or the exception it raised, so a crash fails by assertion."""
    try:
        return call(*args)
    except (Exception, SystemExit) as error:
        return error


class Answer(BytesIO):
    """A scripted reply, as urllib hands one back."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def wire(monkeypatch):
    """Stands in for every server at urllib's own door, so no request leaves this computer. Each POST is
    recorded in .sent as (url, body, headers) and answered with .reply, or refused with .refusal = (HTTP
    status, body), or by .refusal(body) when it is a function (None: not refused). Each GET is recorded in .asked as (url, headers) and answered with .facts[url] (raised, if
    it is an exception), or with HTTP 404 if there's none. Every method is recorded in .methods, read from
    the request itself: a GET carries no body and a POST always does. Header names are lower case."""
    line = SimpleNamespace(reply=None, refusal=None, facts={}, sent=[], asked=[], methods=[])

    def open_(opener, request, timeout=None):
        url, headers = request.full_url, {k.lower(): v for k, v in request.header_items()}
        method = request.get_method()
        line.methods.append(method)
        if method == "GET":
            assert request.data is None, f"a GET with a body: {url}"
            line.asked.append((url, headers))
            if url not in line.facts:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, BytesIO(b"404 page not found"))
            if isinstance(line.facts[url], BaseException):
                raise line.facts[url]
            return Answer(json.dumps(line.facts[url]).encode())
        assert method == "POST" and request.data is not None, f"a {method} with no body: {url}"
        line.sent.append((url, json.loads(request.data), headers))
        refused = line.refusal(line.sent[-1][1]) if callable(line.refusal) else line.refusal
        if refused:
            status, body = refused
            raise urllib.error.HTTPError(url, status, "Refused", {}, BytesIO(json.dumps(body).encode()))
        return Answer(json.dumps(line.reply).encode())
    monkeypatch.setattr("urllib.request.OpenerDirector.open", open_)
    return line


# --- What each backend sends, and where (army test-engineer, r3) ----------------------------------------

def test_ollama_request_and_reply(wire):
    wire.reply = {"message": {"role": "assistant", "content": "hello"}, "prompt_eval_count": 20}
    assert Ollama().complete(MESSAGES) == "hello"
    ((url, body, _),) = wire.sent
    assert url == "http://localhost:11434/api/chat" and body["messages"] == MESSAGES and body["stream"] is False
    assert body.get("truncate") is False                             # too long: refuse, never cut
    assert body["options"]["num_ctx"] == models.context_of(Config(backend="ollama")) == 12288


def test_one_context_number_for_the_request_and_the_budget(wire):
    """Nexus: the window Ollama is asked for is the number recall's budget is sized from."""
    wire.reply = {"message": {"content": "ok"}}
    config = Config(backend="ollama", context_tokens=4096)
    models.make_model(config).complete(MESSAGES)
    assert wire.sent[0][1]["options"]["num_ctx"] == models.context_of(config) == 4096


# --- Ollama refuses a prompt too long for its window instead of cutting it (Nexus, 20 Nov) -----------------

def test_ollamas_count_near_its_window_means_little_room_for_the_reply_never_a_cut(wire, capsys):
    """Nexus measured a cut prompt reporting LESS, about half the window, so a count near the window only
    means the prompt fit and left little room to answer."""
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    near = [{"role": "user", "content": "x" * 16000}]                # estimated 4,000 tokens
    wire.reply = {"message": {"content": "ok"}, "prompt_eval_count": 3900}
    assert Ollama(num_ctx=4096).complete(near) == "ok"
    err = capsys.readouterr().err
    assert "almost no room for the reply" in err and "cut" not in err
    wire.reply = {"message": {"content": "ok"}, "prompt_eval_count": 3800}    # under 95%: nothing to say
    assert Ollama(num_ctx=4096).complete(near) == "ok"
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("refusal", [TOO_LONG, {"error": "too long", "n_prompt_tokens": 1757, "n_ctx": 256}],
                         ids=["as Ollama 0.32.1 said it", "the numbers alone"])
def test_ollamas_400_for_a_prompt_too_long_is_prompt_too_long_in_its_own_numbers(wire, refusal):
    wire.refusal = (400, refusal)
    error = outcome(Ollama(num_ctx=256).complete, MESSAGES)
    assert isinstance(error, PromptTooLong), error
    assert "1757 tokens by its own count, for its 256-token window" in str(error) and "/end" in str(error)


def test_ollamas_error_type_alone_makes_it_prompt_too_long(wire):
    wire.refusal = (400, {"error": "the prompt is too long", "type": "exceed_context_size_error"})
    error = outcome(Ollama().complete, MESSAGES)
    assert isinstance(error, PromptTooLong) and "exceed_context_size_error" in str(error), error


@pytest.mark.parametrize("status, body", [(400, {"error": "invalid option: num_ctx"}), (500, TOO_LONG),
                                          (404, {"error": 'model "llama3.1:8b" not found, try pulling it first'})])
def test_any_other_refusal_stays_a_plain_model_error(wire, status, body):
    wire.refusal = (status, body)
    error = outcome(Ollama().complete, MESSAGES)
    assert isinstance(error, ModelError) and not isinstance(error, PromptTooLong), error
    assert f"HTTP {status}" in str(error)


def test_ollamas_refusal_mid_chat_keeps_nothing_and_says_why(wire, tmp_path):
    config = Config(db=str(tmp_path / "agent.db"), backend="ollama", context_tokens=4000)
    agent = Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config)
    wire.refusal = (400, {**TOO_LONG, "n_prompt_tokens": 4100, "n_ctx": 4000})
    error = outcome(agent.handle, "hello")
    assert isinstance(error, PromptTooLong) and "4100 tokens by its own count" in str(error), error
    assert agent.store.turns(1) == []                                  # your message wasn't kept: send it again


@pytest.mark.parametrize("facts, said", [
    ({"version": "0.32.1"}, None), ({"version": "0.40.0"}, None),     # the version tested, or newer: silent
    ({"version": "0.31.9"}, "this Ollama is 0.31.9, older than 0.32.1"),
    ({"version": "0.32.0"}, "this Ollama is 0.32.0, older than 0.32.1"),   # the patch number counts (r4 V3)
    ({"version": "0.5.7-rc1"}, "this Ollama is 0.5.7-rc1, older than 0.32.1"),
    (None, "couldn't read this Ollama's version"),                     # no /api/version: HTTP 404
    ({"version": 7}, "couldn't read this Ollama's version")])
def test_an_ollama_older_than_the_one_tested_is_named_once_per_run(wire, capsys, facts, said):
    if facts is not None:
        wire.facts[OLLAMA_VERSION] = facts
    wire.reply = {"message": {"content": "ok"}}
    ollama = Ollama()
    assert ollama.complete(MESSAGES) == ollama.complete(MESSAGES) == "ok"
    assert [url for url, _ in wire.asked] == [OLLAMA_VERSION]          # asked once in this run
    err = capsys.readouterr().err
    if said is None:
        assert err == ""
    else:
        assert said in err and err.count("0.32.1") == 1               # said once
        assert 'may ignore "truncate": false and cut a prompt too long' in err


# --- The window comes from the server where it can say; agent.json always wins (Nexus, 20 Nov) -------------

VLLM = "http://localhost:8000/v1"


def window_of(wire, facts, **settings):
    """The window the agent would use with these settings, when the servers say `facts` ({url: JSON})."""
    wire.facts.update(facts)
    config = Config(**{"backend": "openai", "model": "m", "base_url": VLLM, **settings})
    return models.context_of(config, models.make_model(config))


def test_vllm_says_its_window_on_v1_models(wire):
    listed = {"data": [{"id": "other", "max_model_len": 99999}, {"id": "m", "max_model_len": 4096}]}
    assert window_of(wire, {f"{VLLM}/models": listed}) == 4096
    assert [url for url, _ in wire.asked] == [f"{VLLM}/models"]                  # /props wasn't needed


@pytest.mark.parametrize("props", [{"n_ctx": 3072}, {"default_generation_settings": {"n_ctx": 3072}}])
def test_llama_cpp_says_its_window_on_props(wire, props):
    listed = {"data": [{"id": "m", "meta": {"n_ctx_train": 131072}}]}           # its training length, not its window
    assert window_of(wire, {f"{VLLM}/models": listed, "http://localhost:8000/props": props}) == 3072


@pytest.mark.parametrize("facts", [{}, {f"{VLLM}/models": {"data": [{"id": "other", "max_model_len": 4096}]}},
                                   {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": "4096"}]}},
                                   {f"{VLLM}/models": {"data": "m"}, "http://localhost:8000/props": [3072]},
                                   {"http://localhost:8000/props": {"n_ctx": 0}}])
def test_a_server_that_cant_say_gets_the_usual_window_said_once(wire, capsys, facts):
    assert window_of(wire, facts) == window_of(wire, {}) == 16384
    err = capsys.readouterr().err
    assert err.count("assuming a 16,384-token window. A server with a smaller one may cut a longer prompt without "
                     "saying so: set context_tokens in agent.json to its real window.") == 1
    assert models.RISKS == ["Kindling had to assume this server's window: set context_tokens in agent.json to its "
                            "real one"]                                 # kept once, as a risk of a cut
    assert [url for url, _ in wire.asked] == [f"{VLLM}/models", "http://localhost:8000/props"]   # once each


def test_anthropic_says_its_window_asked_with_the_same_headers_as_a_message(wire, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    wire.facts["https://api.anthropic.com/v1/models/claude-opus-5-5"] = {"id": "claude-opus-5-5",
                                                                         "max_input_tokens": 200000}
    config = Config(backend="anthropic")
    model = models.make_model(config)
    assert models.context_of(config, model) == 200000
    wire.reply = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}
    assert model.complete(MESSAGES) == "ok"
    ((_, asked),), ((_, _, sent),) = wire.asked, wire.sent
    assert asked == sent and asked.get("x-api-key") == "test-key-not-real" and "anthropic-version" in asked
    assert "assuming" not in capsys.readouterr().err


def test_anthropic_that_cant_say_gets_the_usual_window_said_once(wire, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert window_of(wire, {}, backend="anthropic", model="", base_url="") == 1_000_000
    assert window_of(wire, {}, backend="anthropic", model="", base_url="") == 1_000_000
    assert capsys.readouterr().err.count("assuming a 1,000,000-token window. A server with a smaller one may cut") == 1


def test_ollamas_window_is_the_num_ctx_kindling_sends_and_needs_no_question(wire):
    config = Config(backend="ollama")
    model = models.make_model(config)
    assert models.context_of(config, model) == model.num_ctx == 12288 and wire.asked == []


@pytest.mark.parametrize("settings", [{}, {"backend": "anthropic", "base_url": ""}, {"backend": "ollama"}],
                         ids=["openai", "anthropic", "ollama"])
def test_context_tokens_in_agent_json_always_wins_and_nothing_is_asked(wire, monkeypatch, capsys, settings):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    says_4096 = {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": 4096}]},
                 "http://localhost:8000/props": {"n_ctx": 4096},
                 "https://api.anthropic.com/v1/models/m": {"max_input_tokens": 4096}}
    assert window_of(wire, says_4096, context_tokens=5000, **settings) == 5000
    assert wire.asked == [] and capsys.readouterr().err == ""


def test_the_servers_window_sizes_the_length_check(wire, tmp_path):
    """llama.cpp started with a 2,048-token window: the usual 16,384 would let prompts eight times too big through."""
    wire.facts["http://localhost:8000/props"] = {"n_ctx": 2048}
    config = Config(db=str(tmp_path / "agent.db"), backend="openai", model="m", base_url=VLLM)
    agent = Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config)
    error = outcome(agent.handle, "a long paste " * 600)               # ~2,000 tokens: under 16,384, over 2,048
    assert isinstance(error, PromptTooLong) and "2,048-token" in str(error), error
    assert wire.sent == []                                             # refused before it was sent


# --- Round 5: a test for each mutant the round-4 verifier found surviving (V1 to V12; V9 was caught) ---------

def openai_agent(wire, tmp_path, facts, **settings):
    wire.facts.update(facts)
    config = Config(**{"db": str(tmp_path / "agent.db"), "backend": "openai", "model": "m", "base_url": VLLM,
                       **settings})
    return Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config)


def test_recalls_budget_is_sized_from_the_servers_window(wire, tmp_path):
    """V7. vLLM at 4,096: a past session of ~3,000 tokens fits the 8,000 budget but not the room 4,096 leaves,
    so it is skipped and named and the turn goes through. Sized from the usual 16,384, it would be packed,
    and the whole turn refused."""
    agent = openai_agent(wire, tmp_path, {f"{VLLM}/models": {"data": [{"id": "m", "max_model_len": 4096}]}})
    big, _ = add_session(agent.store, [("user", "tell me about mice " + "z" * 12000)])
    wire.reply = {"choices": [{"message": {"content": "ok"}}]}
    assert outcome(agent.handle, "mice?") == "ok"
    assert any(f"recall skipped session {big}" in note for note in agent.notes), agent.notes


def test_curiosity_never_sends_a_prompt_too_long_for_the_servers_window(wire, tmp_path):
    """V8. llama.cpp at 2,048: a session of ~3,000 tokens of agent turns fits the usual 16,384, not 2,048."""
    agent = openai_agent(wire, tmp_path, {"http://localhost:8000/props": {"n_ctx": 2048}})
    add_session(agent.store, [("user", "go"), ("agent", "word " * 400)] * 6)
    wire.reply = {"choices": [{"message": {"content": '{"questions": []}'}}]}
    agent.end_session()
    sizes = [models.prompt_size(body["messages"]) for _, body, _ in wire.sent]
    assert all(size * 5 // 4 <= 2048 for size in sizes), sizes


@pytest.mark.parametrize("failure", [urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
                                     TimeoutError("timed out")], ids=["refused", "timed out"])
def test_a_server_that_cant_be_reached_for_a_question_gives_no_answer_never_a_crash(wire, failure):
    """V10: a question about the server (its version, its window) that can't connect is optional, like one
    the server can't answer; the turn goes on."""
    for url in (OLLAMA_VERSION, f"{VLLM}/models", "http://localhost:8000/props"):
        wire.facts[url] = failure
    wire.reply = {"message": {"content": "ok"}}
    assert outcome(Ollama().complete, MESSAGES) == "ok"
    assert outcome(window_of, wire, {}) == 16384


def test_every_question_to_a_server_is_a_get_and_every_message_a_post(wire):
    """V11: sent as a POST, each question would get HTTP 405 from a real server and quietly fall back."""
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    wire.reply = {"message": {"content": "ok"}}
    assert outcome(Ollama().complete, MESSAGES) == "ok"
    assert outcome(window_of, wire, {"http://localhost:8000/props": {"n_ctx": 3072}}) == 3072
    assert wire.methods == ["GET", "POST", "GET", "GET"]               # version, chat, /models, /props


def test_the_length_check_counts_the_system_prompt_too():
    """V12: the instructions take room in the window like everything else."""
    long_system = [{"role": "system", "content": "x" * 4000}, {"role": "user", "content": "hi"}]   # ~1,000
    assert check_fits(long_system, 1300) is None
    assert isinstance(outcome(check_fits, long_system, 1000), PromptTooLong)


# --- After each call, the server's own count checks the estimate (r4 verifier's MEDIUM 1, CC's rule, 20 Nov) -----
# Over the estimate by more than a quarter (it undercounted: the dangerous side) is printed. Under it by more than
# a quarter is logged only: usual with Ollama's prompt cache, but also what a cut looks like (Nexus), so where a cut
# is possible (an assumed window, an Ollama older than 0.32.1) it is printed once, as a possible cut.

PROSE = [{"role": "user", "content": "x" * 4000}]                       # estimated 1,000 tokens
COUNTED = {"ollama": lambda n: {"message": {"content": "ok"}, "prompt_eval_count": n},
           "openai": lambda n: {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": n}},
           "anthropic": lambda n: {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": n}}}


def backend(wire, monkeypatch, name):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    return models.make_model(Config(backend=name, model="m", base_url=VLLM if name == "openai" else ""))


@pytest.mark.parametrize("name", COUNTED)
def test_a_count_far_over_the_estimate_is_printed_and_one_far_under_only_logged(wire, capsys, monkeypatch, name):
    """The two sides are the right way round (the round-4 verifier's V4), and neither says "not a cut"."""
    model = backend(wire, monkeypatch, name)
    for counted, printed, kept in ((1200, False, False), (800, False, False), (1300, True, False), (700, False, True)):
        wire.reply = COUNTED[name](counted)
        assert model.complete(PROSE) == "ok"
        err = capsys.readouterr().err
        assert ("counted" in err) is printed, (counted, err)
        assert models.take_counts() == ([f"{err_name(name)} counted 700 prompt tokens; Kindling estimated 1,000, over a "
                                         "quarter fewer"] if kept else []), counted
        if printed:
            assert f"counted {counted:,} prompt tokens; Kindling estimated 1,000, over a quarter more" in err
            assert "the length check may let through a prompt too long for the window" in err
        assert "mean the prompt was cut" not in err and "not a cut" not in err


def err_name(name):
    return {"ollama": "Ollama", "openai": "The server", "anthropic": "Anthropic"}[name]


@pytest.mark.parametrize("risk", ["an assumed window", "an Ollama older than 0.32.1"])
def test_a_count_far_under_the_estimate_is_printed_once_as_a_possible_cut_where_a_cut_is_possible(wire, capsys,
                                                                                                risk):
    """The round-4 verifier's path: Ollama's own /v1 says no window, Kindling assumes 16,384, and the server
    counts far fewer tokens than the estimate. That can be a cut, so it's said, once per run, and logged each time."""
    if risk == "an assumed window":
        config = Config(backend="openai", model="m", base_url=VLLM)
        model, reply = models.make_model(config), COUNTED["openai"]
        assert models.context_of(config, model) == 16384               # nothing says: assumed
    else:
        wire.facts[OLLAMA_VERSION] = {"version": "0.31.9"}
        model, reply = Ollama(), COUNTED["ollama"]
    for counted in (700, 600, 500):
        wire.reply = reply(counted)
        assert model.complete(PROSE) == "ok"
    err = capsys.readouterr().err
    assert err.count("which can mean it cut the prompt without saying so") == 1, err   # once per run
    assert "counted 700 prompt tokens; Kindling estimated 1,000, over a quarter fewer" in err
    assert ("set context_tokens in agent.json to its real one" in err) is (risk == "an assumed window")
    assert ("This Ollama (0.31.9) is older than 0.32.1" in err) is (risk != "an assumed window")
    assert len(models.take_counts()) == 3                              # and every one is in the log


def test_without_a_known_risk_a_count_far_under_is_never_printed(wire, capsys):
    """An Ollama whose version can't be read is not a known risk (CC's rule names two): logged, not printed."""
    model = Ollama()                                                    # no /api/version: HTTP 404
    wire.reply = COUNTED["ollama"](500)
    assert model.complete(PROSE) == "ok"
    assert "counted" not in capsys.readouterr().err and len(models.take_counts()) == 1


def test_the_agent_and_curiosity_write_a_count_far_under_the_estimate_to_the_log(wire, tmp_path, capsys):
    wire.facts[OLLAMA_VERSION] = {"version": "0.32.1"}
    config = Config(db=str(tmp_path / "agent.db"), backend="ollama")
    agent = Agent(Store(config.db, Clock()), models.make_model(config), HashEmbedder(), config)
    wire.reply = {"message": {"content": "ok"}, "prompt_eval_count": 5}     # far under: a prompt cache, say
    assert agent.handle("hello") == "ok"
    wire.reply = {"message": {"content": '{"questions": []}'}, "prompt_eval_count": 6}
    agent.end_session()
    rows = agent.store.db.execute("SELECT session_id, reason FROM log WHERE kind='count' ORDER BY id").fetchall()
    assert [tuple(row) for row in rows] == [
        (1, f"Ollama counted 5 prompt tokens; Kindling estimated {models.prompt_size(wire.sent[0][1]['messages']):,}, "
            "over a quarter fewer"),
        (1, f"Ollama counted 6 prompt tokens; Kindling estimated {models.prompt_size(wire.sent[1][1]['messages']):,}, "
            "over a quarter fewer")]
    assert "counted" not in capsys.readouterr().err


@pytest.mark.parametrize("name", COUNTED)
@pytest.mark.parametrize("count", [None, "lots", 1.5, True, -3, 0, [1200], {"n": 1}])
def test_the_after_call_compare_never_raises_whatever_the_count(wire, capsys, monkeypatch, name, count):
    model = backend(wire, monkeypatch, name)
    wire.reply = COUNTED[name](count)
    assert outcome(model.complete, PROSE) == "ok"
    assert capsys.readouterr().err == ""                                # no count it can read: nothing to say


@pytest.mark.parametrize("usage", ["n/a", [1200], None, 7])
def test_the_after_call_compare_never_raises_whatever_the_usage(wire, monkeypatch, usage):
    for name, path in (("openai", "prompt_tokens"), ("anthropic", "input_tokens")):
        model = backend(wire, monkeypatch, name)
        wire.reply = {**COUNTED[name](1000), "usage": usage}
        assert outcome(model.complete, PROSE) == "ok", (name, path)


# --- An OpenAI-compatible server's refusal of a prompt too long is PromptTooLong too (r4 verifier's LOW) ------

LLAMA_TOO_LONG = {"error": {"code": 400, "message": "the request exceeds the available context size, try increasing "
                            "it", "type": "exceed_context_size_error", "n_prompt_tokens": 5120, "n_ctx": 4096}}


@pytest.mark.parametrize("body, quoted", [
    (LLAMA_TOO_LONG, "5120 tokens by its own count, for its 4096-token window"),          # llama.cpp: under "error"
    (TOO_LONG, "1757 tokens by its own count, for its 256-token window"),                 # Ollama's, passed on by /v1
    ({"error": {"type": "exceed_context_size_error", "message": "too long"}}, "exceed_context_size_error")],
    ids=["llama.cpp", "Ollama's /v1", "the error type alone"])
def test_an_openai_compatible_servers_400_for_a_prompt_too_long_is_prompt_too_long(wire, body, quoted):
    wire.refusal = (400, body)
    error = outcome(OpenAICompatible("m", VLLM).complete, MESSAGES)
    assert isinstance(error, PromptTooLong) and quoted in str(error), error
    assert str(error).startswith("The server refused the prompt as too long") and "bigger window" in str(error)


@pytest.mark.parametrize("status, body", [
    (400, {"object": "error", "message": "This model's maximum context length is 4096 tokens.", "code": 400}),
    (500, LLAMA_TOO_LONG)])
def test_any_other_openai_compatible_refusal_stays_a_plain_model_error(wire, status, body):
    wire.refusal = (status, body)
    error = outcome(OpenAICompatible("m", VLLM).complete, MESSAGES)
    assert isinstance(error, ModelError) and not isinstance(error, PromptTooLong), error


def test_a_servers_refusal_during_curiosity_skips_that_session_and_reads_the_rest(wire, tmp_path):
    """Before, llama.cpp's refusal was a plain error: curiosity stopped there, left every later session unread,
    and stopped there again at each session end."""
    agent = openai_agent(wire, tmp_path, {}, context_tokens=4096)
    add_session(agent.store, [("user", "go"), ("agent", "the long one")])
    small, _ = add_session(agent.store, [("user", "hi"), ("agent", "why is the sky blue?")])
    wire.refusal = lambda body: (400, LLAMA_TOO_LONG) if "the long one" in body["messages"][-1]["content"] else None
    wire.reply = {"choices": [{"message": {"content": '{"questions": []}'}}]}
    agent.end_session()
    read = [body["messages"][-1]["content"] for _, body, _ in wire.sent]
    assert len(read) == 2 and "why is the sky blue?" in read[1], read          # the next session was still read
    assert small not in [s["id"] for s in agent.store.unread_sessions()]


def test_openai_compatible_request_and_reply(wire, monkeypatch):
    monkeypatch.setenv("KINDLING_TEST_KEY", "test-key-not-real")
    wire.reply = {"choices": [{"message": {"content": "hello"}}]}
    assert OpenAICompatible("m", "http://localhost:8000/v1/", "KINDLING_TEST_KEY").complete(MESSAGES) == "hello"
    ((url, body, headers),) = wire.sent
    assert (url, body) == ("http://localhost:8000/v1/chat/completions", {"model": "m", "messages": MESSAGES})
    assert headers.get("authorization") == "Bearer test-key-not-real"


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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real-2")
    assert OpenAICompatible("m").key == "test-key-not-real-1"                         # api.openai.com
    assert OpenAICompatible("m", "http://localhost:1234/v1").key == ""               # LM Studio: not sent
    assert isinstance(outcome(Anthropic, "", "https://gateway.example.org"), SystemExit)   # name it, or nothing
    monkeypatch.setenv("CLASS_KEY", "test-key-not-real-3")
    assert OpenAICompatible("m", "https://vllm.example.org/v1", "CLASS_KEY").key == "test-key-not-real-3"


def test_a_key_is_never_sent_over_plain_http_to_another_machine(monkeypatch):
    monkeypatch.setenv("CLASS_KEY", "test-key-not-real-3")
    stopped = outcome(OpenAICompatible, "m", "http://203.0.113.9:8000/v1", "CLASS_KEY")
    assert isinstance(stopped, SystemExit) and "plain http" in str(stopped.code)
    assert OpenAICompatible("m", "http://127.0.0.1:8000/v1", "CLASS_KEY").key == "test-key-not-real-3"


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
    assert isinstance(outcome(models.fetch_json, "http://localhost:11434/api/chat", {}), OSError)
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
    error = outcome(models.fetch_json, "http://localhost:11434/api/chat", {})
    assert isinstance(error, ModelError) and "404" in str(error) and "try pulling it first" in str(error)


def test_a_reply_that_is_not_json_is_a_model_error(monkeypatch):
    monkeypatch.setattr("urllib.request.OpenerDirector.open",
                        lambda opener, request, timeout=None: Answer(b"<html>Sign in to the campus Wi-Fi</html>"))
    error = outcome(models.fetch_json, "http://localhost:11434/api/chat", {})
    assert isinstance(error, ModelError) and "campus Wi-Fi" in str(error)


# --- The prompt is checked against the context before it's sent (Nexus) ------------------------------------

def test_a_prompt_too_long_for_the_context_is_refused_not_cut():
    fits = [{"role": "user", "content": "x" * 4000}]                # ~1,000 tokens; with a quarter, 1,250
    assert check_fits(fits, 1250) is None
    error = outcome(check_fits, fits, 1249)
    assert isinstance(error, ModelError) and "too long" in str(error) and "1,249" in str(error)


# --- The token estimate (Nexus, 20 Nov): it errs high where four characters a token undercounted ------------
# Nexus measured real tokens against the old estimate, (len + 3) // 4, on texts like these (Mistral 7B v0.3 /
# Qwen3.5-27B): prose 1.14 / 1.04, code 1.20 / 1.03, Spanish 1.16 / 0.88, mixed languages 1.91 / 1.24,
# JSON 2.08 / 2.08, numbers 4.00 / 4.00, Japanese 4.50 / 1.92. A test can't call a tokenizer, so this pins the
# estimate's shape instead: never below the old one, and a digit or a non-ASCII character is a whole token.

ESTIMATED = [
    ("prose", "The mice were weighed every Monday morning, and the heavier ones moved to the second cage."),
    ("code", "def mean(xs):\n    return sum(xs) / len(xs) if xs else 0.0\n"),
    ("Spanish", "Los ratones se pesaron cada lunes por la mañana, y los más pesados pasaron a la segunda jaula."),
    ("mixed languages", "Los ratones, les souris, die Mäuse, マウス, мыши: weighed on Mondays."),
    ("JSON", '[{"id": 1, "mass_g": 21.4, "cage": "B"}, {"id": 2, "mass_g": 19.8, "cage": "A"}]'),
    ("numbers", "3.14159 2.71828 1.41421 0.57721 1.61803 4.66920"),
    ("Japanese", "マウスは毎週月曜日の朝に体重を測定した。"),
]


@pytest.mark.parametrize("kind, text", ESTIMATED, ids=[kind for kind, _ in ESTIMATED])
def test_the_estimate_is_never_below_four_characters_a_token(kind, text):
    new, old = estimate_tokens(text), (len(text) + 3) // 4
    whole = sum(c.isdigit() or ord(c) > 127 for c in text)          # each a token of its own
    assert new >= old and new >= whole, (kind, new, old, whole)
    assert estimate_tokens(text + "7") == new + 1                     # one more digit, one more token
    assert estimate_tokens(text + "語") == new + 1                    # one more non-ASCII character, too


def test_a_lone_punctuation_mark_costs_and_an_accented_letter_is_a_whole_token():
    """The round-4 verifier's V1 and V2: half a token per mark rounds UP, and Latin-1 (128 to 255) is non-ASCII."""
    assert estimate_tokens(".") == 1 and estimate_tokens("...") == 2
    assert estimate_tokens("é" * 100) == estimate_tokens("ñ" * 100) == 100


def test_digits_and_non_ascii_characters_are_whole_tokens_and_letters_four_a_token():
    assert estimate_tokens("1234567890" * 10) == 100                  # the old estimate said 25
    assert estimate_tokens("語" * 100) == estimate_tokens("ж" * 100) == 100
    assert estimate_tokens("." * 100) == 50                           # ASCII punctuation: half a token
    assert estimate_tokens("word " * 100) == 125                      # prose: as before


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
