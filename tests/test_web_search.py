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

    def check():
        started = fake.ran("run") or fake.ran("start")
        return (True, "") if started else (False, "connection refused")

    monkeypatch.setattr(web_search, "check", check)
    monkeypatch.setattr(web_search, "_ask", lambda q: True)
    return fake


def test_already_running_never_touches_docker(docker_env, monkeypatch):
    monkeypatch.setattr(web_search, "check", lambda: (True, ""))
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
                        lambda: (True, "") if fake.ran("start")
                        else (False, "down"))
    assert web_search.offer_docker_setup() is True
    assert fake.ran("start") == [["start", web_search.CONTAINER_NAME]]
    assert not fake.ran("run")


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

    def fake_search(query):
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


def test_researcher_searches_anything_when_nothing_is_banned(tmp_path,
                                                             monkeypatch):
    _, _, searched = _research_setup(tmp_path, monkeypatch, banned="[]")
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
