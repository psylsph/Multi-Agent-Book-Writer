"""The reviewer evaluation harness (tools/reviewer_eval.py) and review_chapter."""

import json as jsonlib
import re

import pytest

from agents import reviewer
from shared import prompts
from shared.context import reset_context, update_context
from shared.llm_client import EndpointUnavailable, load_config
from shared.story_state import merge_states
from tests.fake_llm import FakeResponse
from tools import reviewer_eval as ev
from tools import reviewer_prompts
from tools.reviewer_cases import (BIBLE, CASES, CHAPTERS, NUMBER, TITLE,
                                  chronology)

KNOWN_TYPES = set(reviewer.CONTINUITY_TYPES) | {"outline_gap", "constraint"}


# ----------------------------------------------------------------- the cases

def test_cases_are_well_formed():
    ids = [c[0] for c in CASES]
    assert len(ids) == len(set(ids)) == 14
    assert sum(1 for c in CASES if c[1] is None) == 4
    assert {c[1] for c in CASES if c[1]} <= KNOWN_TYPES
    assert len({c[2] for c in CASES}) == len(CASES)            # all different
    for case_id, planted, draft in CASES:
        assert len(draft.split()) > 80, case_id


def test_the_story_facts_the_cases_rely_on_are_recorded():
    facts = merge_states(chronology(), upto=NUMBER)
    assert facts["dead"] == {"Marcus Reed": 4}
    assert set(facts["met_pairs"]) == {"elizabeth hale+tom baker",
                                       "elizabeth hale+marcus reed"}
    assert facts["relationships"]["elizabeth hale+tom baker"][1] == 3
    assert [s["who"] for s in facts["secrets"]] == [["Elizabeth Hale"]]
    assert facts["injuries"][0]["who"] == ["Tom Baker"]
    assert [c["number"] for c in CHAPTERS][-1] == NUMBER


def test_each_planted_error_is_really_in_its_chapter():
    by = {c[0]: c[2] for c in CASES}
    assert "Marcus Reed stood" in by["dead"]
    assert "Pleased to meet you" in by["regress"]
    assert "husband of ten years" in by["leap"]
    assert "The letter names you heir" in by["knowledge"]
    assert "crushed on Wednesday night" in by["heal"]
    assert "Tuesday dusk" in by["weekday"]
    assert "London" in by["london"]
    assert "winch" not in by["outline"].lower()
    assert "Tom thought she was" in by["pov"]
    assert "comes down the jetty" in by["tense"]
    # the tempting clean chapters really contain the tempting material
    assert "dreamed of Marcus" in by["clean-b"]
    assert "good hand found hers" in by["clean-c"]


# ------------------------------------------------------------------ scoring

def issue(kind):
    return {"type": kind, "description": "d", "fix": ""}


@pytest.mark.parametrize("planted,issues,outcome", [
    (None, [], "ok"), (None, [issue("timeline")], "false_alarm"),
    ("knowledge", [issue("knowledge")], "typed"),
    ("knowledge", [issue("timeline"), issue("knowledge")], "typed"),
    ("knowledge", [issue("timeline")], "found"),
    ("knowledge", [], "missed")])
def test_score_run(planted, issues, outcome):
    assert ev.score_run(planted, issues) == outcome


def _run(case_id, planted, outcome, types=()):
    return {"id": case_id, "planted": planted, "outcome": outcome,
            "types": set(types)}


def test_summarize_rates_and_breakdown():
    runs = [_run("clean-a", None, "ok"), _run("clean-b", None, "false_alarm",
                                              ["timeline"]),
            _run("dead", "dead_resurrection", "typed"),
            _run("dead", "dead_resurrection", "missed"),
            _run("pov", "constraint", "found", ["timeline"]),
            _run("tense", "constraint", "unreadable")]
    s = ev.summarize(runs, seconds=60, tokens=900)
    assert s["false_alarm_rate"] == 0.5 and s["false_alarms"] == 1
    assert s["recall_any"] == pytest.approx(2 / 4)       # typed + found
    assert s["recall_typed"] == pytest.approx(1 / 4)
    assert s["unreadable"] == 1 and s["sec_per_chapter"] == 10.0
    assert s["by_type"]["dead_resurrection"] == "1/2 (any 1)"
    assert s["by_type"]["constraint"] == "0/2 (any 1)"
    assert ev.summarize([], 0, 0)["recall_typed"] == 0.0


def test_problem_lines_group_repeats():
    runs = [_run("dead", "dead_resurrection", "missed")] * 2 + [
        _run("pov", "constraint", "found", ["timeline"]),
        _run("clean-b", None, "false_alarm", ["knowledge", "timeline"]),
        _run("dead", "dead_resurrection", "typed"),
        _run("tense", "constraint", "unreadable")]
    lines = ev.problem_lines(runs)
    assert "dead: 2x MISSED" in lines
    assert any(line.startswith("pov: found but not typed 'constraint'")
               and "timeline" in line for line in lines)
    assert "clean-b: FALSE ALARM (knowledge, timeline)" in lines
    assert "tense: UNREADABLE reply" in lines


# ------------------------------------- review_chapter vs run_reviewer (real)

def _setup_context(tmp_path, interim=True):
    path = tmp_path / "config.yaml"
    path.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n'
                    f"  interim: {str(interim).lower()}\n")
    load_config(path)
    reset_context()
    update_context("bible", BIBLE)
    update_context("chapters", CHAPTERS)
    update_context("chronology", chronology())


def test_review_chapter_raises_but_run_reviewer_forgives(tmp_path,
                                                         monkeypatch, capsys):
    _setup_context(tmp_path)
    monkeypatch.setattr(reviewer, "generate_with_wait",
                        lambda p, **k: "not json at all")
    with pytest.raises(ValueError):
        reviewer.review_chapter(NUMBER, TITLE, "A draft.")
    assert reviewer.run_reviewer(NUMBER, TITLE, "A draft.") == ("pass", [])
    assert "Error reviewing chapter" in capsys.readouterr().out


def test_an_outage_is_never_forgiven(tmp_path, monkeypatch):
    _setup_context(tmp_path)

    def down(*a, **k):
        raise EndpointUnavailable("down")

    monkeypatch.setattr(reviewer, "generate_with_wait", down)
    for call in (reviewer.review_chapter, reviewer.run_reviewer):
        with pytest.raises(EndpointUnavailable):
            call(NUMBER, TITLE, "A draft.")


# --------------------------------------------- a whole run against stub models

def _case_for(prompt):
    return next(c for c in CASES if c[2] in prompt)


@pytest.fixture
def stub_servers(monkeypatch):
    modes = {"7001": "perfect", "7002": "silent", "7003": "paranoid",
             "7004": "garbage", "7005": "mistyped"}
    seen = []

    def post(url, json=None, **kwargs):
        mode = modes[re.search(r":(\d+)/", url).group(1)]
        seen.append((mode, json))
        prompt = json["messages"][-1]["content"]
        _, planted, _ = _case_for(prompt)
        kind = {"perfect": planted, "paranoid": "timeline",
                "mistyped": "setting" if planted else None}.get(mode)
        issues = ([{"type": kind, "description": "d", "fix": "f"}]
                  if kind else [])
        text = ("sorry, no JSON" if mode == "garbage"
                else jsonlib.dumps({"verdict": "revise" if issues else "pass",
                                    "issues": issues}))
        return FakeResponse(data={
            "choices": [{"message": {"content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 40}})

    monkeypatch.setattr(ev.llm_client.requests, "post", post)
    return seen


def _eval(tmp_path, *targets, extra=()):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f'output:\n  directory: "{tmp_path / "out"}"\n'
                   "  interim: true\n")
    argv = ["--config", str(cfg)]
    for label, port in targets:
        argv += ["--target", label, f"http://h:{port}", f"model-{label}"]
    return ev.main(argv + list(extra))


def test_a_perfect_reviewer(tmp_path, stub_servers, capsys):
    r = _eval(tmp_path, ("good", 7001))["good"]
    assert r["false_alarm_rate"] == 0.0 and r["recall_typed"] == 1.0
    assert r["unreadable"] == 0 and r["problems"] == []
    assert set(r["by_type"]) == {"constraint", "dead_resurrection",
                                 "knowledge", "outline_gap",
                                 "relationship_leap",
                                 "relationship_regression", "setting",
                                 "timeline"}
    out = capsys.readouterr().out
    assert "FALSE ALARM" in out and "nothing" in out


def test_a_silent_reviewer_misses_everything(tmp_path, stub_servers):
    r = _eval(tmp_path, ("quiet", 7002))["quiet"]
    assert r["recall_any"] == 0.0 and r["false_alarm_rate"] == 0.0
    assert any(p.startswith("dead: MISSED") for p in r["problems"])


def test_a_paranoid_reviewer_flags_every_clean_chapter(tmp_path, stub_servers):
    r = _eval(tmp_path, ("jumpy", 7003))["jumpy"]
    assert r["false_alarm_rate"] == 1.0 and r["false_alarms"] == 4
    assert r["recall_any"] == 1.0                    # it "finds" everything
    assert r["recall_typed"] == pytest.approx(2 / 10)  # but only 2 are timeline


def test_a_reviewer_that_mislabels_is_not_credited_with_the_right_type(
        tmp_path, stub_servers):
    r = _eval(tmp_path, ("wrongtype", 7005))["wrongtype"]
    assert r["recall_any"] == 1.0
    assert r["recall_typed"] == pytest.approx(1 / 10)        # only 'london'
    assert any("found but not typed 'dead_resurrection'" in p
               for p in r["problems"])


def test_unreadable_replies_are_counted_not_passed(tmp_path, stub_servers):
    r = _eval(tmp_path, ("junk", 7004), extra=["--runs", "2"])["junk"]
    assert r["unreadable"] == 28                     # every reply, 14 x 2
    assert r["recall_any"] == 0.0 and r["false_alarm_rate"] == 0.0


def test_nothing_is_written_to_the_output_directory(tmp_path, stub_servers):
    _eval(tmp_path, ("good", 7001), extra=["--cases", "dead,clean-a"])
    assert not (tmp_path / "out").exists()           # interim switched off


def test_case_and_check_filters(tmp_path, stub_servers):
    r = _eval(tmp_path, ("good", 7001), extra=["--cases", "dead,pov,clean-a"])
    assert sum(1 for _ in r["good"]["by_type"]) == 2
    _eval(tmp_path, ("only", 7001),
          extra=["--cases", "outline", "--checks", "outline"])
    prompt = stub_servers[-1][1]["messages"][-1]["content"]
    assert "CHAPTER OUTLINE" in prompt
    assert "STORY FACTS" not in prompt and "AUTHOR CONSTRAINTS" not in prompt


def test_options_apply_only_to_the_labelled_target(tmp_path, stub_servers):
    cfg_text = ("llm:\n  reasoning_effort: xhigh\n"
                f'output:\n  directory: "{tmp_path / "out"}"\n')
    (tmp_path / "config.yaml").write_text(cfg_text)
    ev.main(["--config", str(tmp_path / "config.yaml"),
             "--target", "a", "http://h:7001", "ma",
             "--target", "b", "http://h:7001", "mb", "--cases", "clean-a",
             "--temperature", "b=0.2", "--no-thinking", "b"])
    by_model = {p["model"]: p for _, p in stub_servers}
    assert by_model["ma"].get("temperature") is None
    assert by_model["ma"]["chat_template_kwargs"] == \
        {"reasoning_effort": "medium"}               # the reviewer default
    assert by_model["mb"]["temperature"] == 0.2
    assert by_model["mb"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_system_prompt_variants_are_used_and_restored(tmp_path, stub_servers):
    _eval(tmp_path, ("plain", 7001), ("sure", 7001),
          extra=["--system", "sure=classic", "--cases", "clean-a"])
    systems = {p["model"]: p["messages"][0]["content"]
               for _, p in stub_servers}
    assert systems["model-plain"].startswith(reviewer_prompts.VARIANTS[
        "default"])
    assert "quote the exact sentence" in systems["model-plain"]  # the shipped one
    assert systems["model-sure"].startswith(reviewer_prompts.VARIANTS[
        "classic"])
    assert "quote the exact sentence" not in systems["model-sure"]
    assert "STORY BIBLE" in systems["model-sure"]       # the bible still rides along
    assert prompts.REVIEWER == reviewer_prompts.VARIANTS["default"]


def test_bad_options_are_reported(tmp_path, stub_servers):
    for extra in (["--system", "a=nope"], ["--system", "ghost=classic"],
                  ["--temperature", "ghost=0.1"], ["--temperature", "a"],
                  ["--checks", "vibes"], ["--cases", "nonexistent"],
                  ["--system", "a=@/no/such/file.txt"]):
        with pytest.raises(SystemExit):
            _eval(tmp_path, ("a", 7001), extra=extra)


def test_the_shipped_reviewer_variants():
    assert set(reviewer_prompts.VARIANTS) >= {"default", "classic",
                                              "sceptical"}
    assert all("JSON" in v for v in reviewer_prompts.VARIANTS.values())
    assert reviewer_prompts.resolve("sceptical") == \
        reviewer_prompts.VARIANTS["sceptical"]


def test_a_token_cap_applies_only_to_the_labelled_target(tmp_path,
                                                         stub_servers):
    _eval(tmp_path, ("free", 7001), ("capped", 7001),
          extra=["--max-tokens", "capped=1500", "--cases", "clean-a"])
    by_model = {p["model"]: p for _, p in stub_servers}
    assert by_model["model-free"]["max_tokens"] == 8000     # built-in ceiling
    assert by_model["model-capped"]["max_tokens"] == 1500
    with pytest.raises(SystemExit):
        _eval(tmp_path, ("a", 7001), extra=["--max-tokens", "ghost=100"])
    with pytest.raises(SystemExit):
        _eval(tmp_path, ("a", 7001), extra=["--max-tokens", "a"])
