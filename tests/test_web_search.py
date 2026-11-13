"""Web search + SearXNG Docker setup + researcher integration.

Everything external is mocked: HTTP, Docker, the terminal prompt, the LLM.
"""

import json
import subprocess

import pytest
import requests
import yaml

import main as pipeline
from agents import researcher
from shared import llm_client, prompts, web_search
from shared.context import context, reset_context, update_context
from shared.llm_client import load_config


def _config(tmp_path, ws="  enabled: true\n"):
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n'
                    f"web_search:\n{ws}")
    load_config(path)
    reset_context()
    web_search.reset_status()
    return tmp_path / "out"


class _Resp:
    def __init__(self, status=200, data=None, bad_json=False):
        self.status_code, self._data, self._bad = status, data, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._data


# --------------------------------------------------------------------- search

def test_search_parses_cleans_and_caps_results(tmp_path, monkeypatch):
    _config(tmp_path, "  enabled: true\n  results_per_query: 2\n"
                      "  snippet_chars: 20\n")
    items = [{"title": "  A\n title ", "url": "https://a.example",
              "content": "x" * 100},
             {"title": "bad", "url": "javascript:alert(1)", "content": "no"},
             {"title": "B", "url": "https://b.example", "content": "b"},
             {"title": "C", "url": "https://c.example", "content": "c"}]
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return _Resp(data={"results": items})

    monkeypatch.setattr(web_search.requests, "get", fake_get)
    results = web_search.search("real thing")
    assert [r["url"] for r in results] == ["https://a.example",
                                           "https://b.example"]
    assert results[0]["title"] == "A title"
    assert len(results[0]["snippet"]) == 20
    assert seen["url"] == "http://localhost:8888/search"
    assert seen["params"]["format"] == "json"


@pytest.mark.parametrize("resp,fragment", [
    (_Resp(403), "json"),
    (_Resp(500), "HTTP 500"),
    (_Resp(bad_json=True), "JSON"),
])
def test_search_errors_are_explained(tmp_path, monkeypatch, resp, fragment):
    _config(tmp_path)
    monkeypatch.setattr(web_search.requests, "get", lambda *a, **k: resp)
    with pytest.raises(web_search.SearchError, match=fragment):
        web_search.search("q")


def test_search_unreachable_raises_search_error(tmp_path, monkeypatch):
    _config(tmp_path)

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(web_search.requests, "get", boom)
    with pytest.raises(web_search.SearchError, match="could not reach"):
        web_search.search("q")


# ------------------------------------------------------------ query privacy

def test_safe_queries_drop_names_dupes_and_junk():
    raw = ["history of the Pennine Way", "Aria Vance hospital", "aria",
           "history of the pennine way", 42, "", "x" * 200,
           "how paramedics triage", "tide tables", "fourth query"]
    out = web_search.safe_queries(raw, forbidden_terms=["Aria Vance", "Aria"],
                                  limit=3)
    assert out == ["history of the Pennine Way", "how paramedics triage",
                   "tide tables"]


def test_safe_queries_name_match_is_whole_word():
    # a name inside another word must not block an innocent query
    assert web_search.safe_queries(["Mariana Trench depth"],
                                   forbidden_terms=["Ana"]) == \
        ["Mariana Trench depth"]
    assert web_search.safe_queries("not a list") == []


# ---------------------------------------------------------- settings file

def test_searxng_settings_enable_json_and_get_unique_secrets(tmp_path):
    a = web_search.write_searxng_settings(tmp_path / "a")
    b = web_search.write_searxng_settings(tmp_path / "b")
    cfg = yaml.safe_load(a.read_text())
    assert cfg["use_default_settings"] is True
    assert "json" in cfg["search"]["formats"]
    assert cfg["server"]["limiter"] is False
    assert cfg["server"]["secret_key"] != \
        yaml.safe_load(b.read_text())["server"]["secret_key"]
    before = a.read_text()
    web_search.write_searxng_settings(tmp_path / "a")
    assert a.read_text() == before                  # never overwritten


# ------------------------------------------------------------ docker offer

class FakeDocker:
    """Stands in for web_search._docker; records the commands run."""

    def __init__(self, state=None, info_rc=0, run_error=False):
        self.state, self.info_rc, self.run_error = state, info_rc, run_error
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        if args[0] == "info":
            return subprocess.CompletedProcess(args, self.info_rc, "", "")
        if args[0] == "ps":
            return subprocess.CompletedProcess(args, 0, self.state or "", "")
        if args[0] == "run" and self.run_error:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0, "", "")

    def ran(self, verb):
        return [c for c in self.calls if c[0] == verb]


@pytest.fixture
def docker_env(tmp_path, monkeypatch):
    """Docker present, SearXNG down until `docker run/start` was called."""
    _config(tmp_path)
    fake = FakeDocker()
    monkeypatch.setattr(web_search, "_docker", fake)
    monkeypatch.setattr(web_search.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(web_search, "config_dir", lambda: tmp_path / "sx")
    monkeypatch.setattr(web_search.time, "sleep", lambda s: None)

    def check(only=None):
        started = fake.ran("run") or fake.ran("start")
        return (True, "") if started else (False, "connection refused")

    monkeypatch.setattr(web_search, "check", check)
    monkeypatch.setattr(web_search, "reachable", lambda timeout=5: True)
    monkeypatch.setattr(web_search, "_ask", lambda q: True)
    return fake


def test_already_running_never_touches_docker(docker_env, monkeypatch):
    monkeypatch.setattr(web_search, "check", lambda only=None: (True, ""))
    assert web_search.offer_docker_setup() is True
    assert docker_env.calls == []


def test_yes_pulls_and_runs_the_container_on_loopback(docker_env, tmp_path):
    assert web_search.offer_docker_setup() is True
    (run,) = docker_env.ran("run")
    assert run[:4] == ["run", "-d", "--name", web_search.CONTAINER_NAME]
    assert "127.0.0.1:8888:8080" in run           # loopback only
    assert f"{tmp_path / 'sx'}:/etc/searxng" in run
    assert run[-1] == "searxng/searxng"
    assert (tmp_path / "sx" / "settings.yml").exists()


def test_declining_the_prompt_skips_docker(docker_env, monkeypatch):
    monkeypatch.setattr(web_search, "_ask", lambda q: False)
    assert web_search.offer_docker_setup() is False
    assert not docker_env.ran("run")


def test_declining_says_without_search_only_when_nothing_works(
        docker_env, tmp_path, monkeypatch, capsys):
    # "the run continues without web search" was printed on declining
    # SearXNG, before the other providers had even been tried
    _config(tmp_path, "  enabled: true\n  providers: [searxng, parallel]\n")
    monkeypatch.setattr(web_search, "_ask", lambda q: False)
    monkeypatch.setattr(web_search, "check", lambda only=None: (
        (True, "") if only == "parallel" else (False, "connection refused")))
    assert web_search.offer_docker_setup() is True
    out = capsys.readouterr().out
    assert "Searching with parallel" in out and "without web search" not in out

    monkeypatch.setattr(web_search, "check", lambda only=None: (False, "down"))
    assert web_search.offer_docker_setup() is False
    out = capsys.readouterr().out
    assert "continues without web search" in out
    assert "BOOKWRITER_SEARXNG_URL" in out


def test_auto_start_yes_does_not_prompt(docker_env, tmp_path, monkeypatch):
    _config(tmp_path, "  enabled: true\n  auto_start: yes\n")
    monkeypatch.setattr(web_search, "_ask", lambda q: pytest.fail("prompted"))
    assert web_search.offer_docker_setup() is True
    assert docker_env.ran("run")


def test_auto_start_no_never_touches_docker(docker_env, tmp_path):
    _config(tmp_path, "  enabled: true\n  auto_start: no\n")
    assert web_search.offer_docker_setup() is False
    assert docker_env.calls == []


def test_non_local_url_is_never_started(docker_env, tmp_path):
    _config(tmp_path, '  enabled: true\n  searxng_url: "https://search.example"\n')
    assert web_search.offer_docker_setup() is False
    assert docker_env.calls == []


def test_missing_docker_is_reported(docker_env, monkeypatch):
    monkeypatch.setattr(web_search.shutil, "which", lambda name: None)
    assert web_search.offer_docker_setup() is False
    assert docker_env.calls == []


def test_unreachable_docker_daemon_is_reported(docker_env, monkeypatch):
    monkeypatch.setattr(web_search, "_docker", FakeDocker(info_rc=1))
    assert web_search.offer_docker_setup() is False


def test_stopped_container_is_started_not_recreated(docker_env, monkeypatch):
    fake = FakeDocker(state="exited")
    monkeypatch.setattr(web_search, "_docker", fake)
    monkeypatch.setattr(web_search, "check",
                        lambda only=None: (True, "") if fake.ran("start")
                        else (False, "down"))
    assert web_search.offer_docker_setup() is True
    assert fake.ran("start") == [["start", web_search.CONTAINER_NAME]]
    assert not fake.ran("run")


def test_waiting_for_the_container_polls_health_not_search(docker_env,
                                                         monkeypatch):
    """Polling with a real search sent a query upstream every few seconds."""
    answers = iter([False, False, True])
    searches = []
    monkeypatch.setattr(web_search, "reachable",
                        lambda timeout=5: next(answers))
    monkeypatch.setattr(web_search, "check",
                        lambda only=None: searches.append(1) or (True, ""))
    assert web_search._wait_until_ready() is True
    assert len(searches) == 1


def test_waiting_gives_up_when_nothing_answers(docker_env, monkeypatch):
    monkeypatch.setattr(web_search, "reachable", lambda timeout=5: False)
    monkeypatch.setattr(web_search, "check",
                        lambda only=None: pytest.fail("searched an unreachable server"))
    ticks = iter(range(0, 1000, 50))
    monkeypatch.setattr(web_search.time, "time", lambda: next(ticks))
    assert web_search._wait_until_ready(seconds=90) is False


def test_a_reachable_server_that_cannot_search_is_reported(
        docker_env, monkeypatch, capsys):
    monkeypatch.setattr(web_search, "check",
                        lambda only=None: (False, "HTTP 403: enable json"))
    assert web_search._wait_until_ready() is False
    assert "cannot search: HTTP 403" in capsys.readouterr().out


def test_reachable_reads_healthz_and_treats_404_as_up(tmp_path, monkeypatch):
    _config(tmp_path)
    urls = []

    class Resp:
        status_code = 404

    monkeypatch.setattr(web_search.requests, "get",
                        lambda url, **k: urls.append(url) or Resp())
    assert web_search.reachable() is True
    assert urls[0].endswith("/healthz")
    Resp.status_code = 502
    assert web_search.reachable() is False


def test_reachable_is_false_when_the_connection_fails(tmp_path, monkeypatch):
    _config(tmp_path)

    def refuse(*a, **k):
        raise web_search.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(web_search.requests, "get", refuse)
    assert web_search.reachable() is False


def test_docker_failure_is_swallowed(docker_env, monkeypatch):
    monkeypatch.setattr(web_search, "_docker", FakeDocker(run_error=True))
    assert web_search.offer_docker_setup() is False   # no exception


def test_ask_never_blocks_without_a_terminal(monkeypatch):
    class NoTty:
        def isatty(self):
            return False

    monkeypatch.setattr(web_search.sys, "stdin", NoTty())
    monkeypatch.setattr("builtins.input",
                        lambda *a: pytest.fail("must not prompt"))
    assert web_search._ask("start?") is False


def test_ask_accepts_yes_on_a_terminal(monkeypatch):
    class Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(web_search.sys, "stdin", Tty())
    for answer, expected in (("y", True), ("YES", True), ("n", False),
                             ("", False)):
        monkeypatch.setattr("builtins.input", lambda *a, a_=answer: a_)
        assert web_search._ask("start?") is expected


# -------------------------------------------------------- researcher wiring

CHAPTER = {"number": 1, "title": "The Ward", "summary": "Aria visits a ward."}
BIBLE = {"title": "T", "world": "w",
         "characters": [{"name": "Aria Vance", "role": "", "description": ""}]}
TRIAGE = {"title": "Triage 101", "url": "https://n.example/t",
          "snippet": "Nurses sort patients by urgency using a triage scale "
                     "on arrival."}
PROPOSED = [{"claim": "Hospital wards triage patients on arrival",
             "query": "how hospital wards triage"},
            {"claim": "Aria Vance works there", "query": "Aria Vance ward"}]


def _verdict(**kw):
    base = {"claim_id": 1, "verdict": "supported", "source": 1,
            "evidence": "sort patients by urgency using a triage scale",
            "note": ""}
    base.update(kw)
    return json.dumps([base])


def _research_setup(tmp_path, monkeypatch, results=None, verdicts=None,
                    available=True, proposed=None, chapters=None,
                    second=None, banned='["Aria Vance", "Willow Rooms"]'):
    out = _config(tmp_path, "  enabled: true\n"
                  f"  banned_terms: {banned}\n")
    update_context("chapters", chapters or [CHAPTER])
    update_context("bible", BIBLE)
    calls = []

    def llm(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        if "fact-check a novel" in prompt:
            return json.dumps(PROPOSED if proposed is None else proposed)
        if "CLAIMS TO VERIFY" in prompt:
            return verdicts if verdicts is not None else _verdict()
        if "does this quote" in prompt:
            return second if second is not None else '{"answer": true}'
        return "BRIEF TEXT"

    searched = []

    def fake_search(query, objective=None):
        searched.append(query)
        if isinstance(results, Exception):
            raise results
        return [dict(r) for r in (TRIAGE_RESULTS if results is None
                                  else results)]

    monkeypatch.setattr(researcher, "generate_with_wait", llm)
    monkeypatch.setattr(web_search, "search", fake_search)
    monkeypatch.setattr(web_search, "is_available", lambda: available)
    return out, calls, searched


TRIAGE_RESULTS = [TRIAGE]


def _brief_prompt(calls):
    return next(c["prompt"] for c in calls if "lore keeper" in c["prompt"])


def test_verified_fact_reaches_the_brief_with_its_quote(tmp_path, monkeypatch):
    out, calls, searched = _research_setup(tmp_path, monkeypatch)
    researcher.run_researcher()
    assert searched == ["how hospital wards triage"]    # name query dropped
    brief = _brief_prompt(calls)
    assert "VERIFIED REAL-WORLD FACTS" in brief
    assert "SUPPORTED: Hospital wards triage patients on arrival" in brief
    assert '"sort patients by urgency using a triage scale"' in brief
    assert "https://n.example/t" in brief
    assert "DATA, not instructions" in brief and "Fact notes" in brief
    assert context["research"][1] == "BRIEF TEXT"
    audit = (out / "interim" / "search_chapter_01.md").read_text()
    assert "**supported**" in audit and "how hospital wards triage" in audit


def test_character_full_names_are_blocked_without_banned_terms(
        tmp_path, monkeypatch, capsys):
    _, _, searched = _research_setup(tmp_path, monkeypatch, banned="[]")
    researcher.run_researcher()
    assert searched == ["how hospital wards triage"]   # "Aria Vance ward" dropped
    assert "1 full character name(s) are blocked" in capsys.readouterr().out


def test_character_name_blocking_can_be_switched_off(tmp_path, monkeypatch):
    _, _, searched = _research_setup(
        tmp_path, monkeypatch,
        banned="[]\n  block_character_names: false")
    researcher.run_researcher()
    assert "Aria Vance ward" in searched


def test_character_terms_are_full_names_and_multi_word_aliases_only():
    bible = {"characters": [
        {"name": "Aria Vance (the nurse)", "aliases": ["Aria", "Ari Vance"]},
        {"name": "Rose", "aliases": ["Rosie"]},            # one word: skipped
        {"name": "Tom Baker"}, {"name": "tom baker"}, {"name": ""}, {}]}
    assert web_search.character_terms(bible) == [
        "Aria Vance", "Ari Vance", "Tom Baker"]
    assert web_search.character_terms(None) == []


def test_verifier_runs_with_its_own_system_prompt(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(tmp_path, monkeypatch)
    researcher.run_researcher()
    verify = next(c for c in calls if "CLAIMS TO VERIFY" in c["prompt"])
    assert verify["system"] == prompts.VERIFIER
    assert "[1] Triage 101" in verify["prompt"]


def test_contradicted_claim_is_flagged_with_the_correct_fact(tmp_path,
                                                             monkeypatch):
    _, calls, _ = _research_setup(
        tmp_path, monkeypatch,
        verdicts=_verdict(verdict="contradicted",
                          note="Triage is by urgency, not arrival order."))
    researcher.run_researcher()
    brief = _brief_prompt(calls)
    assert "CONTRADICTED:" in brief
    assert "Correct fact: Triage is by urgency, not arrival order." in brief


def test_a_fabricated_quote_is_rejected(tmp_path, monkeypatch):
    out, calls, _ = _research_setup(
        tmp_path, monkeypatch,
        verdicts=_verdict(evidence="Every ward uses the Manchester scale"))
    researcher.run_researcher()
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)
    audit = (out / "interim" / "search_chapter_01.md").read_text()
    assert "quote not found" in audit


def test_a_verdict_citing_the_wrong_source_is_rejected(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(tmp_path, monkeypatch,
                                  verdicts=_verdict(source=7))
    researcher.run_researcher()
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)


def test_unclear_claims_are_left_out_of_the_brief(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(
        tmp_path, monkeypatch, verdicts=_verdict(verdict="unclear",
                                                 source=None, evidence=""))
    researcher.run_researcher()
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)
    assert context["research"][1] == "BRIEF TEXT"


def test_no_search_results_means_no_verifier_call(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(tmp_path, monkeypatch, results=[])
    researcher.run_researcher()
    assert not any("CLAIMS TO VERIFY" in c["prompt"] for c in calls)
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)


def test_garbage_verifier_output_still_produces_a_brief(tmp_path,
                                                        monkeypatch):
    _, calls, _ = _research_setup(tmp_path, monkeypatch,
                                  verdicts="I cannot verify these.")
    researcher.run_researcher()
    assert context["research"][1] == "BRIEF TEXT"
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)


def test_search_failure_still_produces_a_brief(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(tmp_path, monkeypatch,
                                  results=web_search.SearchError("boom"))
    researcher.run_researcher()
    assert context["research"][1] == "BRIEF TEXT"
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)


def test_repeated_queries_are_skipped_across_chapters(tmp_path, monkeypatch):
    chapters = [CHAPTER, {**CHAPTER, "number": 2, "title": "Again"}]
    proposed = [{"claim": "Pampas grass is hardy in Hampshire",
                 "query": "pampas grass hardiness zone UK"}]
    _, calls, searched = _research_setup(
        tmp_path, monkeypatch, chapters=chapters, proposed=proposed)
    researcher.run_researcher()
    assert searched == ["pampas grass hardiness zone UK"]       # once only
    second = [c for c in calls if "fact-check a novel" in c["prompt"]][1]
    assert "Already checked in other chapters" in second["prompt"]
    assert "pampas grass hardiness zone UK" in second["prompt"]


def test_reworded_repeat_is_skipped_too(tmp_path, monkeypatch):
    chapters = [CHAPTER, {**CHAPTER, "number": 2, "title": "Again"}]
    replies = iter([
        [{"claim": "c1", "query": "pampas grass hardiness zone UK"}],
        [{"claim": "c2", "query": "pampas grass hardiness zones UK Hampshire"}]])
    _, calls, searched = _research_setup(tmp_path, monkeypatch,
                                         chapters=chapters)
    plan = lambda p, **k: (json.dumps(next(replies))   # noqa: E731
                           if "fact-check a novel" in p else
                           (_verdict() if "CLAIMS TO VERIFY" in p else "B"))
    monkeypatch.setattr(researcher, "generate_with_wait", plan)
    researcher.run_researcher()
    assert searched == ["pampas grass hardiness zone UK"]


def test_researcher_without_web_search_is_unchanged(tmp_path, monkeypatch):
    _, calls, searched = _research_setup(tmp_path, monkeypatch,
                                         available=False)
    researcher.run_researcher()
    assert searched == []
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)
    assert not any("fact-check a novel" in c["prompt"] for c in calls)


def test_researcher_disabled_in_config_never_searches(tmp_path, monkeypatch):
    _, _, searched = _research_setup(tmp_path, monkeypatch)
    _config(tmp_path, "  enabled: false\n")
    update_context("chapters", [CHAPTER])
    update_context("bible", BIBLE)
    researcher.run_researcher()
    assert searched == []


def test_researcher_never_searches_a_banned_term(tmp_path, monkeypatch, capsys):
    _, _, searched = _research_setup(tmp_path, monkeypatch, proposed=[
        {"claim": "A clinic exists", "query": "Willow Rooms clinic location"},
        {"claim": "Paramedics triage", "query": "how paramedics triage"}])
    researcher.run_researcher()
    assert searched == ["how paramedics triage"]
    out = capsys.readouterr().out
    assert "Dropped query 'Willow Rooms clinic location'" in out
    assert "banned term 'Willow Rooms'" in out


def test_researcher_searches_anything_when_nothing_is_blocked(tmp_path,
                                                              monkeypatch):
    _, _, searched = _research_setup(
        tmp_path, monkeypatch, banned="[]\n  block_character_names: false")
    researcher.run_researcher()
    assert searched == ["how hospital wards triage", "Aria Vance ward"]


# ------------------------------------------- verification helpers (no LLM)

@pytest.mark.parametrize("quote,text,ok", [
    ("Nurses sort patients by urgency", TRIAGE["snippet"], True),
    ("nurses  SORT patients, by urgency!", TRIAGE["snippet"], True),
    ("Nurses sort ... triage scale on arrival", TRIAGE["snippet"], True),
    ("Doctors rank patients by wealth", TRIAGE["snippet"], False),
    ("urgency", TRIAGE["snippet"], False),             # too short to trust
    ("", TRIAGE["snippet"], False),
])
def test_quote_in_text(quote, text, ok):
    assert web_search.quote_in_text(quote, text) is ok


@pytest.mark.parametrize("a,b,same", [
    ("pampas grass hardiness zone UK", "pampas grass hardiness zones UK", True),
    ("pampas grass hardiness UK", "Pampas grass hardiness and cultivation UK",
     True),
    ("village green fete customs", "yoga class schedule village halls", False),
    ("how hospital wards triage", "hospital ward triage process", True),
])
def test_similar_query(a, b, same):
    assert bool(web_search.similar_query(a, [b])) is same


def test_validate_verdicts_downgrades_untrustworthy_ones():
    claims = [{"claim": "Nurses sort patients by urgency", "query": "q1",
               "results": [{"n": 1, **TRIAGE}]},
              {"claim": "Nurses sort patients by urgency", "query": "q2",
               "results": [{"n": 2, "title": "T", "url": "https://x.example",
                            "snippet": "Totally unrelated text here today."}]},
              {"claim": "Ward doors are blue", "query": "q3", "results": []}]
    raw = [{"claim_id": 1, "verdict": "supported", "source": 1,
            "evidence": "sort patients by urgency using a triage scale"},
           {"claim_id": 2, "verdict": "supported", "source": 1,   # not c2's
            "evidence": "Nurses sort patients by urgency"},
           {"claim_id": "nonsense", "verdict": "supported"}]
    out = web_search.validate_verdicts(claims, raw)
    assert [v["verdict"] for v in out] == ["supported", "unclear", "unclear"]
    assert out[0]["url"] == "https://n.example/t"
    assert out[1]["reason"] == "no valid source cited"
    assert web_search.validate_verdicts(claims, "not a list")[0][
        "verdict"] == "unclear"


# ------------------------------------------------------------------- main

def test_research_complete_detection():
    chapters = [{"number": 1}, {"number": 2}]
    assert pipeline._research_complete(
        {"chapters": chapters, "research": {1: "a", 2: "b"}})
    assert not pipeline._research_complete(
        {"chapters": chapters, "research": {1: "a"}})
    assert not pipeline._research_complete(None)
    assert not pipeline._research_complete({"chapters": None, "research": {}})


@pytest.mark.parametrize("value,expected", [
    (True, "yes"), (False, "no"), ("yes", "yes"), ("No", "no"),
    ("ask", "ask"), (None, "ask"), ("sometimes", "ask")])
def test_auto_start_accepts_yaml_booleans(value, expected):
    assert web_search._auto_start_mode(value) == expected


# ------------------------------------------------ story-term privacy filter

WORLD_BIBLE = {
    "premise": "Stuart treats a sprained ankle at the Willow Rooms in "
               "Hollowby. Everyone in the village watches.",
    "world": "A village in Hampshire. The Willow Rooms is a private clinic "
             "run by Lisa. Calm, bubbly neighbours gather on the green. "
             "Carole lives at No.12 and runs the raffle.",
    "characters": [
        {"name": "Stuart", "description": "Divorced. Very protective."},
        {"name": "Carole (from No.12)", "description": "Tolerant, watches."},
    ],
}
OUTLINE = [{"title": "The Fall", "summary": "Stuart falls near Hollowby."}]


def test_nothing_is_banned_by_default(monkeypatch):
    assert web_search.banned_terms() == []
    queries = ["Willow Rooms private clinic location", "Aria Vance"]
    assert web_search.safe_queries(queries, web_search.banned_terms()) == queries


def test_banned_terms_are_read_from_config_and_cleaned(monkeypatch):
    cfg = llm_client.get_config()
    monkeypatch.setitem(cfg["web_search"], "banned_terms",
                        ["  Willow   Rooms ", "", None, "Hollowby", 7])
    assert web_search.banned_terms() == ["Willow Rooms", "Hollowby", "7"]
    monkeypatch.setitem(cfg["web_search"], "banned_terms", "Hollowby")
    assert web_search.banned_terms() == ["Hollowby"]


def test_blocked_term_is_whole_word_and_case_insensitive():
    terms = ["Willow Rooms", "Tom"]
    assert web_search.blocked_term("WILLOW ROOMS clinic location", terms) == \
        "Willow Rooms"
    assert web_search.blocked_term("customs of a village fete", terms) is None
    assert web_search.blocked_term("tom cats", terms) == "Tom"


def test_a_banned_term_is_blocked_and_logged_everything_else_allowed():
    terms = ["Willow Rooms", "Stuart"]
    dropped = []
    queries = ["Willow Rooms private clinic location and services UK",
               "private physiotherapy self-referral process UK",
               "glass fronted barn conversions in Hampshire"]
    out = web_search.safe_queries(queries, terms, limit=5,
                                  on_drop=lambda q, t: dropped.append((q, t)))
    assert out == queries[1:]
    assert dropped == [(queries[0], "Willow Rooms")]


# ------------------------------------- the verdict check: relevance + 2nd pass

@pytest.mark.parametrize("claim,quote,query,ok", [
    ("Hospital wards triage patients on arrival",
     "sort patients by urgency using a triage scale",
     "how hospital wards triage", True),
    # shares only the TOPIC words that the query itself contained
    ("Hospital wards triage patients on arrival",
     "Nurses wear blue uniforms on most hospital wards",
     "how hospital wards triage", False),
    ("Hospital wards triage patients on arrival", "triage", "", False),
    ("Pampas grass is hardy", "pampas grass survives frost", "", True),
    ("Pampas", "pampas grass survives frost", "", True),         # 1-word claim
    ("", "anything at all", "", False),                         # nothing to judge
])
def test_addresses_claim(claim, quote, query, ok):
    assert web_search.addresses_claim(claim, quote, query) is ok


def test_a_real_quote_about_something_else_is_rejected(tmp_path, monkeypatch):
    results = [{"title": "Uniforms", "url": "https://u.example",
                "snippet": "Nurses wear blue uniforms on most hospital wards "
                           "in the north."}]
    out, calls, _ = _research_setup(
        tmp_path, monkeypatch, results=results,
        verdicts=_verdict(evidence="wear blue uniforms on most hospital wards"))
    researcher.run_researcher()
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)
    audit = (out / "interim" / "search_chapter_01.md").read_text()
    assert "quote does not address the claim" in audit


def test_second_check_confirms_a_good_verdict(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(tmp_path, monkeypatch)
    researcher.run_researcher()
    second = next(c for c in calls if "does this quote" in c["prompt"])
    assert second["system"] == prompts.VERIFIER
    assert 'Quote from https://n.example/t: "sort patients' in second["prompt"]
    assert "directly support the claim" in second["prompt"]
    assert "SUPPORTED:" in _brief_prompt(calls)


@pytest.mark.parametrize("reply", ['{"answer": false}', "no idea",
                                   '{"answer": "yes"}', '{"agrees": false}'])
def test_second_check_disagreement_or_garbage_drops_the_fact(
        tmp_path, monkeypatch, reply):
    out, calls, _ = _research_setup(tmp_path, monkeypatch, second=reply)
    researcher.run_researcher()
    assert "VERIFIED REAL-WORLD FACTS" not in _brief_prompt(calls)
    audit = (out / "interim" / "search_chapter_01.md").read_text()
    assert "second check did not confirm it" in audit


def test_second_check_asks_the_contradiction_question(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(
        tmp_path, monkeypatch,
        verdicts=_verdict(verdict="contradicted", note="Use urgency."))
    researcher.run_researcher()
    second = next(c for c in calls if "does this quote" in c["prompt"])
    assert "directly contradict the claim" in second["prompt"]


def test_second_check_can_be_switched_off(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(tmp_path, monkeypatch)
    (tmp_path / "config.yaml").write_text(
        f'output:\n  directory: "{tmp_path / "out"}"\nweb_search:\n'
        "  enabled: true\n  double_check: false\n"
        "agents:\n  reviewer:\n    enabled: false\n")
    load_config(tmp_path / "config.yaml")
    update_context("chapters", [CHAPTER])
    update_context("bible", BIBLE)
    researcher.run_researcher()
    assert not any("does this quote" in c["prompt"] for c in calls)
    assert "SUPPORTED:" in _brief_prompt(calls)


def test_unclear_verdicts_are_never_double_checked(tmp_path, monkeypatch):
    _, calls, _ = _research_setup(
        tmp_path, monkeypatch,
        verdicts=_verdict(verdict="unclear", source=None, evidence=""))
    researcher.run_researcher()
    assert not any("does this quote" in c["prompt"] for c in calls)


def test_verification_calls_use_the_verifier_agent(tmp_path, monkeypatch):
    """Judging results and the second check run as `verifier` (so they can be
    tuned or sent to another model separately); the claim proposal and the
    lore brief stay with the researcher."""
    _, calls, _ = _research_setup(tmp_path, monkeypatch)
    researcher.run_researcher()
    agent_of = {}
    for c in calls:
        kind = ("propose" if "fact-check a novel" in c["prompt"] else
                "verify" if "CLAIMS TO VERIFY" in c["prompt"] else
                "second" if "does this quote" in c["prompt"] else "brief")
        agent_of[kind] = c["agent"]
    assert agent_of == {"propose": "researcher", "verify": "verifier",
                        "second": "verifier", "brief": "researcher"}


def test_queries_are_spaced_by_min_interval(monkeypatch):
    """Regression: a chapter's queries went out back to back, Brave's Search
    API answered 429 and SearXNG suspended the engine for three minutes."""
    monkeypatch.setattr(web_search, "PACING", True)    # off in conftest
    real_pace = web_search._pace
    clock, slept = [100.0], []
    monkeypatch.setattr(web_search.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(web_search.time, "sleep",
                        lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(web_search, "_last_query", [None])
    real_pace(1.5)                      # the first query goes straight out
    clock[0] += 0.5
    real_pace(1.5)                      # 0.5 s later: waits the other 1.0 s
    clock[0] += 3.0
    real_pace(1.5)                      # long after: no wait
    real_pace(0)                        # pacing off
    assert slept == [1.0]


# ------------------------------------------------- Parallel (MCP) provider

class _MCP:
    """A fake Parallel MCP server: requests.post -> JSON-RPC over SSE."""

    def __init__(self, results=None, status=None, expire_once=False):
        self.calls, self.sessions = [], 0
        self.results = results if results is not None else [{
            "url": "https://en.wikipedia.org/wiki/Hampshire",
            "title": "Hampshire - Wikipedia",
            "excerpts": ["The remainder of the county is rural.\n\n...\n\n"
                         "Average rainfall is 640–1,060 mm per year "
                         "and 9.8 °C on average.\n\n...\n\nIBM has a lab."]}]
        self.status, self.expire_once = status, expire_once

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}), "body": json})
        method = json["method"]
        if method == "initialize":
            self.sessions += 1
            return _SSE({"jsonrpc": "2.0", "id": json["id"], "result": {}},
                        session=f"s{self.sessions}")
        if method == "notifications/initialized":
            return _SSE(None, status=202)
        if self.expire_once and self.sessions == 1:
            return _SSE(None, status=404)
        if self.status:
            return _SSE(None, status=self.status)
        data = {"search_id": "x", "results": self.results}
        return _SSE({"jsonrpc": "2.0", "id": json["id"], "result": {
            "content": [{"type": "text", "text": "..."}], "isError": False,
            "structuredContent": data}})


class _SSE:
    """An event-stream reply WITHOUT a charset, as Parallel sends it."""

    def __init__(self, message, status=200, session=None):
        self.status_code = status
        self.headers = {"mcp-session-id": session} if session else {}
        body = "" if message is None else (
            "event: message\ndata: " + json.dumps(message, ensure_ascii=False) + "\n\n")
        self.content = body.encode("utf-8")
        self.text = self.content.decode("latin-1")   # what requests would guess


def _parallel(tmp_path, monkeypatch, server=None, extra=""):
    _config(tmp_path, "  enabled: true\n  providers: [parallel]\n" + extra)
    monkeypatch.setattr(web_search, "_PARALLEL", {"session": None, "next_id": 1})
    server = server or _MCP()
    monkeypatch.setattr(web_search.requests, "post", server)
    return server


def test_parallel_search_speaks_mcp_and_keeps_the_answering_passage(
        tmp_path, monkeypatch):
    server = _parallel(tmp_path, monkeypatch, extra="  snippet_chars: 120\n")
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    found = web_search.search("average rainfall Hampshire",
                              objective="Typical annual rainfall in Hampshire")
    methods = [c["body"]["method"] for c in server.calls]
    assert methods == ["initialize", "notifications/initialized", "tools/call"]
    call = server.calls[-1]
    assert call["headers"]["Mcp-Session-Id"] == "s1"
    assert call["body"]["params"] == {"name": "web_search", "arguments": {
        "objective": "Typical annual rainfall in Hampshire",
        "search_queries": ["average rainfall Hampshire"]}}
    (result,) = found
    assert result["url"] == "https://en.wikipedia.org/wiki/Hampshire"
    # the passage that answers the claim, not the page's opening
    assert result["snippet"].startswith("Average rainfall is 640")
    # decoded as UTF-8 although the stream names no charset
    assert "–" in result["snippet"] and "°C" in result["snippet"]
    assert "Â" not in result["snippet"]
    # a second search reuses the session
    web_search.search("rainfall")
    assert [c["body"]["method"] for c in server.calls].count("initialize") == 1


def test_parallel_never_gets_the_llm_key(tmp_path, monkeypatch):
    server = _parallel(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    web_search.search("q")
    assert all("Authorization" not in c["headers"] for c in server.calls)
    monkeypatch.setenv("PARALLEL_API_KEY", "par-key")
    web_search.search("q")
    assert server.calls[-1]["headers"]["Authorization"] == "Bearer par-key"


def test_parallel_key_from_config_env_reference(tmp_path, monkeypatch):
    server = _parallel(tmp_path, monkeypatch,
                       extra='  parallel_api_key: "env:MY_PAR"\n')
    monkeypatch.setenv("MY_PAR", "from-env")
    web_search.search("q")
    assert server.calls[-1]["headers"]["Authorization"] == "Bearer from-env"


def test_an_expired_parallel_session_is_reopened_once(tmp_path, monkeypatch):
    server = _parallel(tmp_path, monkeypatch, _MCP(expire_once=True))
    assert web_search.search("q")
    assert [c["body"]["method"] for c in server.calls].count("initialize") == 2


def test_parallel_rate_limit_is_explained(tmp_path, monkeypatch):
    _parallel(tmp_path, monkeypatch, _MCP(status=429))
    with pytest.raises(web_search.SearchError, match="parallel_api_key"):
        web_search.search("q")


def test_providers_are_merged_and_one_failing_leaves_the_other(
        tmp_path, monkeypatch, capsys):
    server = _parallel(tmp_path, monkeypatch)
    _config(tmp_path, "  enabled: true\n  providers: [searxng, parallel]\n")
    monkeypatch.setattr(web_search, "_warned", set())
    monkeypatch.setattr(web_search.requests, "get", lambda *a, **k: _Resp(data={
        "results": [{"url": "https://en.wikipedia.org/wiki/Hampshire",
                     "title": "dupe", "content": "c"},
                    {"url": "https://example.org/a", "title": "A",
                     "content": "a"}]}))
    found = web_search.search("q")
    # in provider order, one entry per URL
    assert [r["url"] for r in found] == ["https://en.wikipedia.org/wiki/Hampshire",
                                         "https://example.org/a"]
    assert found[0]["title"] == "dupe"
    # SearXNG down: Parallel still answers, and the failure is said once
    def down(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(web_search.requests, "get", down)
    assert web_search.search("q")
    web_search.search("q")
    assert capsys.readouterr().out.count("searxng failed") == 1
    # both down: an error
    server.status = 503
    with pytest.raises(web_search.SearchError, match="searxng: .*parallel: "):
        web_search.search("q")


def test_parallel_only_never_offers_docker(tmp_path, monkeypatch):
    _parallel(tmp_path, monkeypatch)
    monkeypatch.setattr(web_search, "_docker",
                        lambda *a, **k: pytest.fail("touched docker"))
    assert web_search.offer_docker_setup() is True


def test_searxng_failing_falls_back_to_parallel(tmp_path, monkeypatch):
    _parallel(tmp_path, monkeypatch)
    _config(tmp_path, "  enabled: true\n  providers: [searxng, parallel]\n"
                      "  auto_start: \"no\"\n")
    monkeypatch.setattr(web_search.requests, "get", lambda *a, **k: _Resp(503))
    assert web_search.offer_docker_setup() is True


def test_unknown_providers_are_warned_about_and_dropped(tmp_path):
    from shared.config_schema import problems
    warnings = problems({"web_search": {"providers": ["searxng", "parralel"]}})
    assert any("parralel" in w and "parallel" in w for w in warnings)
    _config(tmp_path, "  providers: [parralel]\n")
    assert web_search.providers() == ["searxng"]


def test_best_passages_prefers_the_claims_words():
    excerpts = ["Opening text about nothing.\n\n...\n\nThe ferry runs hourly "
                "from the quay.\n\n...\n\nClosing remarks."]
    text = web_search.best_passages(excerpts, {"ferry", "hourly", "quay"}, 40)
    assert text.startswith("The ferry runs hourly")
    assert web_search.best_passages([], {"x"}, 100) == ""


def test_the_objective_passes_the_same_screen_as_the_query(tmp_path, monkeypatch):
    """Parallel is sent the claim as well as the query: a claim that names a
    character (its query being generic) must not reach it."""
    objectives = []
    proposed = [{"claim": "Hospital wards triage patients on arrival",
                 "query": "how hospital wards triage"},
                {"claim": "Willow Rooms hold the night shift's handover",
                 "query": "night shift handover"}]
    _research_setup(tmp_path, monkeypatch, proposed=proposed)
    monkeypatch.setattr(web_search, "search", lambda q, objective=None: (
        objectives.append(objective) or [dict(r) for r in TRIAGE_RESULTS]))
    researcher.run_researcher()
    assert objectives == ["Hospital wards triage patients on arrival",
                          "night shift handover"]       # the name never sent


# ------------------------------------------------------ TinyFish provider

def _tinyfish(tmp_path, monkeypatch, reply=None, status=200, extra=""):
    _config(tmp_path, "  enabled: true\n  providers: [tinyfish]\n" + extra)
    calls = []

    def get(url, timeout=None, params=None, headers=None):
        calls.append({"url": url, "params": dict(params or {}),
                      "headers": dict(headers or {})})
        return _Resp(status, data=reply if reply is not None else {"results": [
            {"position": 1, "title": "Solent averages",
             "url": "https://www.metoffice.gov.uk/x", "snippet": "Annual 676.84"},
            {"title": "no link"}]})
    monkeypatch.setattr(web_search.requests, "get", get)
    return calls


def test_tinyfish_search_sends_the_key_and_the_purpose(tmp_path, monkeypatch):
    calls = _tinyfish(tmp_path, monkeypatch)
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-key")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    found = web_search.search("rainfall Hampshire", objective="Hampshire's rainfall")
    assert found == [{"title": "Solent averages", "url": "https://www.metoffice.gov.uk/x",
                      "snippet": "Annual 676.84"}]
    (call,) = calls
    assert call["url"] == "https://api.search.tinyfish.ai/"
    assert call["headers"] == {"X-API-Key": "tf-key"}          # never the LLM key
    assert call["params"] == {"query": "rainfall Hampshire",
                              "purpose": "Hampshire's rainfall"}


def test_tinyfish_without_a_key_says_how_to_set_one(tmp_path, monkeypatch):
    calls = _tinyfish(tmp_path, monkeypatch)
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    with pytest.raises(web_search.SearchError, match="TINYFISH_API_KEY"):
        web_search.search("q")
    assert calls == []                                      # nothing sent


def test_tinyfish_key_from_an_env_reference(tmp_path, monkeypatch):
    calls = _tinyfish(tmp_path, monkeypatch, extra='  tinyfish_api_key: "env:TF2"\n')
    monkeypatch.setenv("TF2", "other")
    web_search.search("q")
    assert calls[0]["headers"]["X-API-Key"] == "other"
    assert "purpose" not in calls[0]["params"]             # no objective given


@pytest.mark.parametrize("status, fragment", [(401, "refused the key"),
                                              (429, "rate limit"),
                                              (500, "HTTP 500")])
def test_tinyfish_errors_are_explained(tmp_path, monkeypatch, status, fragment):
    _tinyfish(tmp_path, monkeypatch, status=status)
    monkeypatch.setenv("TINYFISH_API_KEY", "k")
    with pytest.raises(web_search.SearchError, match=fragment):
        web_search.search("q")
