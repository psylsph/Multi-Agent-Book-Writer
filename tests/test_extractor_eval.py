"""The extractor evaluation harness (tools/extractor_eval.py)."""

import json
import re

import pytest

from tests.fake_llm import FakeResponse
from tools import extractor_eval as ev
from tools import extractor_prompts
from tools.extractor_cases import BIBLE_CHARACTERS, CASES

TYPES = {"first_meeting", "relationship_change", "death", "injury",
         "secret_revealed"}


def state(present, events):
    return {"present": present,
            "events": [{"type": t, "who": w, "detail": ""} for t, w in events]}


# ----------------------------------------------------------- the test cases

def test_cases_are_well_formed():
    names = {c["name"] for c in BIBLE_CHARACTERS}
    assert [c["number"] for c in CASES] == list(range(1, 11))
    for case in CASES:
        assert set(case["present"]) <= names
        assert set(case.get("present_optional", [])) <= names
        assert set(case.get("prior_deaths", [])) <= names
        for kind, who in case["events"]:
            assert kind in TYPES and set(who) <= names and who
        assert len(case["text"].split()) > 60


def test_the_cases_contain_the_traps_they_claim_to():
    by = {c["number"]: c for c in CASES}
    assert "Marcus Reed" in by[1]["text"] and \
        "Marcus Reed" not in by[1]["present"]            # mentioned only
    assert "Liz" in by[3]["text"] and "Liz" not in by[3]["present"]  # alias
    assert by[5]["events"] == []                          # nothing happens
    assert not any(k == "first_meeting" for k, _ in by[3]["events"])
    assert ("death", ["Marcus Reed"]) in by[4]["events"]
    assert "Marcus Reed" not in by[4]["present"]          # dead and absent
    # the false-death traps: a death only in backstory, a near-death, an attack
    assert "drowned" in by[6]["text"] and by[6]["events"] == []
    assert "nearly drowned" in by[7]["text"] and by[7]["events"] == []
    assert all(kind != "death" for kind, _ in by[10]["events"])
    assert by[10]["events"] == [("injury", ["Tom Baker"])]
    # an implicit death with no death word in it that the model must still find
    assert "died" not in by[8]["text"] and ("death", ["Marcus Reed"]) in \
        by[8]["events"]
    # a funeral after the death is on record is not a new death
    assert by[9]["prior_deaths"] == ["Marcus Reed"] and by[9]["events"] == []


# ------------------------------------------------------------------ scoring

TRUTH_PRESENT = ["Elizabeth Hale", "Tom Baker"]
TRUTH_EVENTS = [("first_meeting", ["Elizabeth Hale", "Tom Baker"])]


def test_a_perfect_extraction_scores_perfectly():
    c = ev.score_state(state(["elizabeth hale", "Tom Baker"], TRUTH_EVENTS),
                       TRUTH_PRESENT, TRUTH_EVENTS)
    assert c["present_hit"] == 2 and c["strict"] == 1 and c["lenient"] == 1
    assert c["spurious"] == 0 and c["false_deaths"] == 0


def test_strict_versus_lenient_matching():
    partial = state(TRUTH_PRESENT, [("first_meeting", ["Tom Baker"])])
    c = ev.score_state(partial, TRUTH_PRESENT, TRUTH_EVENTS)
    assert c["strict"] == 0 and c["lenient"] == 1 and c["spurious"] == 0
    wrong_type = state(TRUTH_PRESENT,
                       [("injury", ["Elizabeth Hale", "Tom Baker"])])
    c = ev.score_state(wrong_type, TRUTH_PRESENT, TRUTH_EVENTS)
    assert c["lenient"] == 0 and c["spurious"] == 1


def test_an_invented_death_is_counted_as_the_worst_kind_of_spurious():
    c = ev.score_state(
        state(TRUTH_PRESENT, TRUTH_EVENTS + [("death", ["Tom Baker"])]),
        TRUTH_PRESENT, TRUTH_EVENTS)
    assert c["spurious"] == 1 and c["false_deaths"] == 1
    # a death that really happened is not spurious
    c = ev.score_state(state([], [("death", ["Marcus Reed"])]),
                       [], [("death", ["Marcus Reed"])])
    assert c["false_deaths"] == 0 and c["lenient"] == 1


def test_a_character_who_is_only_mentioned_hurts_precision():
    c = ev.score_state(state(TRUTH_PRESENT + ["Marcus Reed"], TRUTH_EVENTS),
                       TRUTH_PRESENT, TRUTH_EVENTS)
    assert (c["present_hit"], c["present_got"], c["present_expected"]) == \
        (2, 3, 2)


def test_summarize_rates_and_empty_cases():
    counts = [ev.score_state(state(TRUTH_PRESENT, TRUTH_EVENTS),
                             TRUTH_PRESENT, TRUTH_EVENTS),
              ev.score_state(state([], []), TRUTH_PRESENT, TRUTH_EVENTS)]
    s = ev.summarize(counts, failures=1, total=2, seconds=10, tokens=500)
    assert s["valid_json"] == "1/2"
    assert s["present_recall"] == 0.5 and s["present_precision"] == 1.0
    assert s["event_recall_strict"] == 0.5 and s["sec_per_chapter"] == 5.0
    # nothing expected and nothing found is a perfect score, not a divide error
    quiet = ev.score_state(state([], []), [], [])
    assert ev.summarize([quiet], 0, 1, 1, 0)["event_recall_strict"] == 1.0
    assert ev.summarize([], 0, 0, 0, 0)["present_recall"] == 1.0


def test_agreement_between_two_extractions():
    a = {1: state(["Tom"], [("death", ["Marcus"])]),
         2: state(["Tom", "Liz"], [])}
    b = {1: state(["Tom"], [("death", ["Marcus"])]),
         2: state(["Tom"], [("injury", ["Tom"])]), 3: state([], [])}
    got = ev.agreement(a, b)
    assert got["chapters"] == 2                           # only shared ones
    assert got["present"] == pytest.approx((1.0 + 0.5) / 2)
    assert got["events"] == pytest.approx((1.0 + 0.0) / 2)


def test_real_chapters_are_loaded_from_a_directory(tmp_path):
    (tmp_path / "chapter_02.md").write_text("# Chapter 2: Dusk\n\nBody text.")
    (tmp_path / "chapter_01.md").write_text("# Chapter 1: Dawn\n\nFirst.")
    (tmp_path / "notes.md").write_text("ignored")
    cases = ev.load_real_chapters(tmp_path)
    assert [(c["number"], c["title"], c["text"]) for c in cases] == [
        (1, "Dawn", "First."), (2, "Dusk", "Body text.")]


# ------------------------------------------- a whole run against stub servers

def _case_for_text(evidence):
    """The case whose chapter text contains this snippet."""
    probe = " ".join(evidence.split())[:50]
    return next(c for c in CASES if probe in " ".join(c["text"].split()))


def _answer_death_question(prompt):
    """A model that answers the focused death/alive questions correctly."""
    evidence = prompt.split('"""')[1]
    candidates = prompt.split("Candidates: ")[1].split("\n")[0].split(", ")
    case = _case_for_text(evidence)
    truth = {n for kind, who in case["events"] if kind == "death"
             for n in who}
    if "still alive" in prompt:
        return json.dumps({"alive": [c for c in candidates if c not in truth]})
    return json.dumps({"dead": [c for c in candidates if c in truth]})


def _extraction_for(prompt, mode):
    number = int(re.search(r"\nChapter (\d+):", prompt).group(1))
    case = CASES[number - 1]
    events = [{"type": t, "who": w, "detail": ""} for t, w in case["events"]]
    present = list(case["present"])
    if mode == "hallucinate" and number == 5:
        events.append({"type": "death", "who": ["Tom Baker"], "detail": "x"})
    if mode == "blind":                                  # misses everything
        events = []
    if mode == "nickname":
        present = ["Liz" if n == "Elizabeth Hale" else n for n in present]
    return json.dumps({"summary": "S.", "time": "day", "location": "marsh",
                       "present": present, "events": events})


@pytest.fixture
def stub_servers(monkeypatch):
    """Fake servers: the port picks the model's behaviour."""
    modes = {"7001": "perfect", "7002": "hallucinate", "7003": "garbage",
             "7004": "nickname", "7005": "blind"}
    seen = []

    def post(url, json=None, **kwargs):
        mode = modes[re.search(r":(\d+)/", url).group(1)]
        seen.append((mode, json))
        prompt = json["messages"][-1]["content"]
        if "Candidates: " in prompt:                     # the death question
            text = "no idea" if mode == "garbage" else \
                _answer_death_question(prompt)
        else:
            text = "no idea" if mode == "garbage" else \
                _extraction_for(prompt, mode)
        return FakeResponse(data={
            "choices": [{"message": {"content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    monkeypatch.setattr(ev.llm_client.requests, "post", post)
    return seen


def _run(tmp_path, *targets, extra=()):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("agents:\n  reviewer:\n    enabled: false\n")
    argv = ["--config", str(cfg)]
    for label, port in targets:
        argv += ["--target", label, f"http://h:{port}", f"model-{label}"]
    return ev.main(argv + list(extra))


def test_a_perfect_model_scores_perfectly(tmp_path, stub_servers, capsys):
    r = _run(tmp_path, ("good", 7001))["good"]
    assert r["valid_json"] == "10/10" and r["false_deaths"] == 0
    assert r["present_recall"] == r["present_precision"] == 1.0
    assert r["event_recall_strict"] == 1.0 and r["spurious_events"] == 0
    assert r["deaths_found"] == "2/2"
    out = capsys.readouterr().out
    assert "good" in out and "FALSE DTH" in out and "calls/ch" in out


def test_the_checks_cost_extra_calls_only_where_death_language_is(
        tmp_path, stub_servers):
    r = _run(tmp_path, ("on", 7001), ("off", 7001),
             extra=["--no-checks", "off"])
    assert r["off"]["calls_per_chapter"] == 1.0           # extraction only
    assert 1.0 < r["on"]["calls_per_chapter"] < 2.0       # + a few questions


def test_a_hallucinated_death_is_caught_by_the_checks_not_by_luck(
        tmp_path, stub_servers, capsys):
    r = _run(tmp_path, ("checked", 7002), ("unchecked", 7002),
             extra=["--no-checks", "unchecked"])
    assert r["unchecked"]["false_deaths"] == 1            # the model's lie
    assert r["unchecked"]["spurious_events"] == 1
    assert r["checked"]["false_deaths"] == 0              # ...caught
    assert r["checked"]["spurious_events"] == 0
    assert "does not confirm that Tom Baker dies" in capsys.readouterr().out


def test_a_model_that_misses_deaths_is_rescued_only_with_the_checks(
        tmp_path, stub_servers):
    r = _run(tmp_path, ("checked", 7005), ("unchecked", 7005),
             extra=["--no-checks", "unchecked"])
    assert r["unchecked"]["deaths_found"] == "0/2"
    assert r["checked"]["deaths_found"] == "2/2"          # found by the check
    assert r["checked"]["false_deaths"] == 0


def test_a_model_that_cannot_produce_json_fails_every_chapter(tmp_path,
                                                              stub_servers):
    r = _run(tmp_path, ("junk", 7003))["junk"]
    assert r["valid_json"] == "0/10" and r["present_recall"] == 0.0
    assert r["event_recall_lenient"] == 0.0 and r["deaths_found"] == "0/2"


def test_nicknames_are_snapped_by_the_real_pipeline_code(tmp_path,
                                                         stub_servers):
    """The model says 'Liz'; the extraction step must report Elizabeth Hale,
    so the harness measures the pipeline, not just the raw model."""
    r = _run(tmp_path, ("nick", 7004))["nick"]
    assert r["present_recall"] == 1.0 and r["present_precision"] == 1.0


def test_a_chapter_filter_runs_only_those_cases(tmp_path, stub_servers):
    r = _run(tmp_path, ("good", 7001), extra=["--cases", "4,8"])["good"]
    assert r["valid_json"] == "2/2" and r["deaths_found"] == "2/2"
    assert r["event_recall_strict"] == 1.0


def test_already_dead_characters_are_set_up_from_prior_deaths(tmp_path,
                                                              stub_servers):
    """Case 9 (a funeral) starts with Marcus on record as dead; the pipeline
    must not record his death again."""
    r = _run(tmp_path, ("good", 7001), extra=["--cases", "9"])["good"]
    assert r["false_deaths"] == 0 and r["spurious_events"] == 0
    asked = [p for _, p in stub_servers
             if "Candidates: " in p["messages"][-1]["content"]]
    for payload in asked:
        line = payload["messages"][-1]["content"].split("Candidates: ")[1]
        assert "Marcus Reed" not in line.split("\n")[0]


def test_no_thinking_targets_switch_thinking_off(tmp_path, stub_servers):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  reasoning_effort: xhigh\n"
                   "agents:\n  reviewer:\n    enabled: false\n")
    ev.main(["--config", str(cfg), "--target", "think", "http://h:7001", "m1",
             "--target", "plain", "http://h:7001", "m2",
             "--no-thinking", "plain", "--cases", "1"])
    kwargs = {p["model"]: p.get("chat_template_kwargs")
              for _, p in stub_servers}
    assert kwargs["m1"] == {"reasoning_effort": "low"}      # extractor default
    # no reasoning effort at all, and the switch for hybrid models turned off
    assert kwargs["m2"] == {"enable_thinking": False}


def test_temperature_applies_only_to_the_labelled_target(tmp_path,
                                                         stub_servers):
    _run(tmp_path, ("warm", 7001), ("cold", 7001),
         extra=["--temperature", "cold=0.1", "--cases", "1"])
    temps = {p["model"]: p.get("temperature") for _, p in stub_servers}
    assert temps == {"model-warm": None, "model-cold": 0.1}


def test_system_prompt_variants_are_used_per_target(tmp_path, stub_servers):
    _run(tmp_path, ("plain", 7001), ("tailored", 7001),
         extra=["--system", "tailored=strict", "--cases", "1"])
    systems = {p["model"]: p["messages"][0]["content"]
               for _, p in stub_servers if p["messages"][0]["role"] == "system"
               and "Candidates" not in p["messages"][-1]["content"]}
    assert systems["model-plain"] == extractor_prompts.VARIANTS["default"]
    assert systems["model-tailored"] == extractor_prompts.VARIANTS["strict"]
    assert "conservative" in systems["model-tailored"]
    # and the shipped prompt is restored afterwards
    from shared import prompts
    assert prompts.EXTRACTOR == extractor_prompts.VARIANTS["default"]


def test_a_custom_system_prompt_can_come_from_a_file(tmp_path, stub_servers):
    custom = tmp_path / "mine.txt"
    custom.write_text("  You are my custom extractor.  \n")
    _run(tmp_path, ("mine", 7001),
         extra=["--system", f"mine=@{custom}", "--cases", "1"])
    (payload,) = [p for _, p in stub_servers][:1]
    assert payload["messages"][0]["content"] == "You are my custom extractor."


def test_bad_option_values_are_reported(tmp_path, stub_servers):
    with pytest.raises(SystemExit, match="unknown system prompt 'nope'"):
        _run(tmp_path, ("a", 7001), extra=["--system", "a=nope", "--cases", "1"])
    with pytest.raises(SystemExit, match="--system"):
        _run(tmp_path, ("a", 7001),
             extra=["--system", "a=@/no/such/file.txt", "--cases", "1"])
    for extra in (["--system", "ghost=strict"], ["--temperature", "ghost=0.1"],
                  ["--temperature", "a"]):
        with pytest.raises(SystemExit):
            _run(tmp_path, ("a", 7001), extra=extra)


def test_ambiguous_names_are_neither_rewarded_nor_penalised():
    with_marcus = ev.score_state(state(["Elizabeth Hale", "Marcus Reed"], []),
                                 ["Elizabeth Hale"], [], ["Marcus Reed"])
    without = ev.score_state(state(["Elizabeth Hale"], []),
                             ["Elizabeth Hale"], [], ["Marcus Reed"])
    assert with_marcus == without
    assert with_marcus["present_got"] == with_marcus["present_hit"] == 1


def test_death_metrics_count_real_and_invented_deaths():
    truth = [("death", ["Marcus Reed"])]
    caught = ev.score_state(state([], [("death", ["Marcus Reed"])]), [], truth)
    assert (caught["deaths_expected"], caught["deaths_found"]) == (1, 1)
    missed = ev.score_state(state([], []), [], truth)
    assert (missed["deaths_expected"], missed["deaths_found"]) == (1, 0)
    invented = ev.score_state(state([], [("death", ["Tom Baker"])]), [], [])
    assert invented["false_deaths"] == 1 and invented["deaths_expected"] == 0


def test_the_shipped_prompt_variants():
    assert set(extractor_prompts.VARIANTS) >= {"default", "minimal", "strict"}
    assert all("JSON" in v for v in extractor_prompts.VARIANTS.values())
    assert extractor_prompts.resolve("strict") == \
        extractor_prompts.VARIANTS["strict"]


def test_json_mode_flag_is_passed_through(tmp_path, stub_servers):
    _run(tmp_path, ("good", 7001), extra=["--json-mode", "--cases", "1,4"])
    assert all(p.get("response_format") == {"type": "json_object"}
               for _, p in stub_servers)


def test_real_chapter_mode_reports_agreement(tmp_path, stub_servers, capsys):
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    for case in CASES[:2]:
        (chapters / f"chapter_{case['number']:02d}.md").write_text(
            f"# Chapter {case['number']}: {case['title']}\n\n{case['text']}")
    results = _run(tmp_path, ("ref", 7001), ("liar", 7002),
                   extra=["--chapters-dir", str(chapters)])
    out = capsys.readouterr().out
    assert "Agreement with the reference 'ref'" in out
    assert "liar: present names 1.00, events 1.00 over 2 chapters" in out
    assert set(results["ref"]) == {"valid_json", "sec_per_chapter", "tokens",
                                   "calls_per_chapter"}


# ------------------------------------------------- where the errors are

def _counts(**kw):
    base = {k: 0 for k in ev.COUNT_KEYS}
    base.update(kw)
    return base


def test_problem_cases_describes_each_kind_of_error():
    cases = [{"number": 6, "title": "Pale Water"},
             {"number": 7, "title": "Thin Ice"}, {"number": 8, "title": "X"}]
    per_case = {
        6: _counts(spurious=3, false_deaths=2, present_got=3, present_hit=2,
                   present_expected=3, events_expected=1, lenient=0),
        7: _counts(),                                   # clean: not listed
        8: _counts(spurious=1, events_expected=2, lenient=2),
    }
    out = ev.problem_cases(per_case, cases)
    assert set(out) == {6, 8}
    assert out[6] == ("case 6 'Pale Water': 2 FALSE DEATH(S), 1 invented "
                      "event(s), 1 event(s) missed, 1 wrong name(s) in "
                      "'present', 1 name(s) missing from 'present'")
    assert out[8] == "case 8 'X': 1 invented event(s)"


def test_the_table_is_followed_by_where_the_errors_are(tmp_path, stub_servers,
                                                      capsys):
    r = _run(tmp_path, ("liar", 7002), ("good", 7001),
             extra=["--no-checks", "liar"])
    out = capsys.readouterr().out
    assert "Where the errors are" in out
    assert "case 5 'Ropes': 1 FALSE DEATH(S)" in out        # the lie, located
    assert r["liar"]["problem_cases"][5].startswith("case 5 'Ropes'")
    good = out.split("  good\n")[1]
    assert good.strip().startswith("none")                  # nothing to report
