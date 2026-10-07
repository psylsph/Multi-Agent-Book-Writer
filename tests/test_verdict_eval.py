"""The verdict second check: confirms() and tools/verdict_eval.py."""

import json as jsonlib
import re

import pytest

from agents import researcher
from shared import prompts
from shared.llm_client import EndpointUnavailable
from tests.fake_llm import FakeResponse
from tools import verdict_eval as ve
from tools import verdict_prompts
from tools.verdict_cases import CASES, KINDS


# ----------------------------------------------------------------- the cases

def test_cases_are_well_formed_and_balanced():
    assert len(CASES) == 34
    assert {c[0] for c in CASES} == set(KINDS)
    assert len({c[:4] for c in CASES}) == len(CASES)          # no duplicates
    for kind, verdict, claim, quote, agrees in CASES:
        assert verdict in ("supported", "contradicted")
        assert claim.strip() and quote.strip()
        # the answer follows from the kind
        assert agrees is (kind in ("support", "contradict"))
        if kind == "support":
            assert verdict == "supported"
        if kind == "contradict":
            assert verdict == "contradicted"
    yes = sum(c[4] for c in CASES)
    assert 10 <= yes <= 20 and len(CASES) - yes >= 15         # both matter


def test_every_pairing_has_a_sibling_with_the_opposite_answer():
    """The same quote must appear with answers that differ, so a model cannot
    score well by reacting to the quote alone."""
    by_quote = {}
    for kind, verdict, claim, quote, agrees in CASES:
        by_quote.setdefault(quote, set()).add(agrees)
    assert any(len(answers) == 2 for answers in by_quote.values())


# ------------------------------------------------------------------ scoring

@pytest.mark.parametrize("expected,answer,outcome", [
    (True, True, "ok"), (False, False, "ok"),
    (False, True, "false_agree"), (True, False, "false_reject"),
    (True, None, "unreadable"), (False, None, "unreadable")])
def test_classify(expected, answer, outcome):
    assert ve.classify(expected, answer) == outcome


def test_summarize_rates():
    outcomes = [("support", True, "ok"), ("support", True, "false_reject"),
                ("unrelated", False, "false_agree"),
                ("unrelated", False, "ok"), ("opposite", False, "unreadable")]
    s = ve.summarize(outcomes, seconds=10, tokens=500)
    assert s["accuracy"] == pytest.approx(2 / 5)
    assert s["false_agree_rate"] == pytest.approx(1 / 3)     # of the 3 "no"s
    assert s["false_reject_rate"] == pytest.approx(1 / 2)    # of the 2 "yes"s
    assert s["false_agrees"] == 1 and s["unreadable"] == 1
    assert s["by_kind"] == {"support": "1/2", "unrelated": "1/2",
                            "opposite": "0/1"}
    assert s["sec_per_check"] == 2.0 and s["tokens"] == 500
    assert ve.summarize([], 0, 0)["accuracy"] == 0.0


# ------------------------------------------------- confirms(): the real code

def _reply(monkeypatch, text):
    monkeypatch.setattr(researcher, "generate_with_wait",
                        lambda prompt, **kw: text)


VERDICT = {"verdict": "supported", "claim": "c", "evidence": "e", "url": "u"}


@pytest.mark.parametrize("reply,expected", [
    ('{"answer": true}', True), ('{"answer": false}', False),
    ('{"answer": "yes"}', None), ('{"answer": 1}', None),
    ('{"agrees": true}', True), ('{"agrees": false}', False),   # old key
    ('{"agrees": "yes"}', None),
    ('{"other": true}', None), ("not json", None), ("[true]", None)])
def test_confirms_reads_only_a_real_boolean(monkeypatch, reply, expected):
    _reply(monkeypatch, reply)
    assert researcher.confirms(VERDICT) is expected


def test_confirms_does_not_swallow_an_outage(monkeypatch):
    def down(*a, **k):
        raise EndpointUnavailable("down")

    monkeypatch.setattr(researcher, "generate_with_wait", down)
    with pytest.raises(EndpointUnavailable):
        researcher.confirms(VERDICT)


def test_unreadable_answers_still_drop_the_fact(monkeypatch):
    _reply(monkeypatch, "no idea")
    verdicts = [{**VERDICT, "note": "", "reason": ""}]
    researcher._double_check(1, verdicts)
    assert verdicts[0]["verdict"] == "unclear"


# --------------------------------------------- a whole run against stub models

def _case_from_prompt(prompt):
    claim = prompt.split("Claim: ")[1].split("\n")[0]
    quote = re.search(r'Quote from \S+: "(.*)"', prompt).group(1)
    word = re.search(r"directly (support|contradict) the claim", prompt).group(1)
    verdict = "supported" if word == "support" else "contradicted"
    return next(c for c in CASES if c[1:4] == (verdict, claim, quote))


def _words(text):
    return set(re.findall(r"[a-z]{4,}", text.lower()))


@pytest.fixture
def stub_servers(monkeypatch):
    modes = {"7001": "perfect", "7002": "yes", "7003": "no",
             "7004": "garbage", "7005": "sycophant"}
    seen = []

    def post(url, json=None, **kwargs):
        mode = modes[re.search(r":(\d+)/", url).group(1)]
        seen.append((mode, json))
        prompt = json["messages"][-1]["content"]
        case = _case_from_prompt(prompt)
        answer = {"perfect": case[4], "yes": True, "no": False,
                  "sycophant": len(_words(case[2]) & _words(case[3])) >= 2,
                  }.get(mode)
        text = ("I think so" if mode == "garbage"
                else jsonlib.dumps({"answer": answer}))
        return FakeResponse(data={
            "choices": [{"message": {"content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 6}})

    monkeypatch.setattr(ve.llm_client.requests, "post", post)
    return seen


def _run(tmp_path, *targets, extra=(), config=""):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(config or "agents:\n  reviewer:\n    enabled: false\n")
    argv = ["--config", str(cfg)]
    for label, port in targets:
        argv += ["--target", label, f"http://h:{port}", f"model-{label}"]
    return ve.main(argv + list(extra))


def test_a_perfect_model_has_no_false_yes(tmp_path, stub_servers, capsys):
    r = _run(tmp_path, ("good", 7001))["good"]
    assert r["accuracy"] == 1.0 and r["false_agrees"] == 0
    assert r["false_reject_rate"] == 0.0 and r["unreadable"] == 0
    assert all(v == f"{n}/{n}" for v in r["by_kind"].values()
               for n in [int(v.split("/")[1])])
    out = capsys.readouterr().out
    assert "FALSE YES" in out and "accuracy by kind" in out


def test_a_yes_man_is_exposed_by_false_yes(tmp_path, stub_servers):
    r = _run(tmp_path, ("yes", 7002))["yes"]
    assert r["false_agree_rate"] == 1.0 and r["false_reject_rate"] == 0.0
    assert r["accuracy"] == pytest.approx(15 / 34)


def test_a_no_man_drops_every_good_fact_but_lets_nothing_through(
        tmp_path, stub_servers):
    r = _run(tmp_path, ("no", 7003))["no"]
    assert r["false_agree_rate"] == 0.0 and r["false_reject_rate"] == 1.0


def test_a_model_that_cannot_answer_in_json_is_unreadable(tmp_path,
                                                          stub_servers):
    r = _run(tmp_path, ("junk", 7004), extra=["--runs", "2"])["junk"]
    assert r["unreadable"] == 68 and r["accuracy"] == 0.0


def test_topic_matching_is_not_judgement(tmp_path, stub_servers):
    """A model that says yes whenever claim and quote share words does well on
    easy cases and badly on the opposite/overstated ones."""
    r = _run(tmp_path, ("syc", 7005))["syc"]
    assert r["false_agrees"] > 0
    assert r["by_kind"]["opposite"] != "8/8"


def test_the_kind_filter_runs_only_those_pairs(tmp_path, stub_servers):
    r = _run(tmp_path, ("good", 7001), extra=["--kinds", "opposite,overstated"])
    assert set(r["good"]["by_kind"]) == {"opposite", "overstated"}
    assert sum(int(v.split("/")[1]) for v in r["good"]["by_kind"].values()) == 11


def test_options_apply_only_to_the_labelled_target(tmp_path, stub_servers):
    _run(tmp_path, ("a", 7001), ("b", 7001), extra=[
        "--temperature", "b=0.2", "--no-thinking", "b", "--kinds", "support"],
        config="llm:\n  reasoning_effort: xhigh\n"
               "agents:\n  reviewer:\n    enabled: false\n")
    by_model = {}
    for _, payload in stub_servers:
        by_model.setdefault(payload["model"], payload)
    assert by_model["model-a"].get("temperature") is None
    assert by_model["model-a"]["chat_template_kwargs"] == \
        {"reasoning_effort": "low"}                  # the researcher default
    assert by_model["model-b"]["temperature"] == 0.2
    assert by_model["model-b"]["chat_template_kwargs"] == \
        {"enable_thinking": False}


def test_system_prompt_variants_are_used_and_restored(tmp_path, stub_servers):
    _run(tmp_path, ("plain", 7001), ("tough", 7001),
         extra=["--system", "tough=strict", "--kinds", "support"])
    systems = {}
    for _, payload in stub_servers:
        systems.setdefault(payload["model"], payload["messages"][0]["content"])
    assert systems["model-plain"] == verdict_prompts.VARIANTS["default"]
    assert systems["model-tough"] == verdict_prompts.VARIANTS["strict"]
    assert "When in doubt, answer false" in systems["model-tough"]
    assert prompts.VERIFIER == verdict_prompts.VARIANTS["default"]


def test_a_custom_prompt_file_and_bad_options(tmp_path, stub_servers):
    mine = tmp_path / "mine.txt"
    mine.write_text(" Judge carefully. \n")
    _run(tmp_path, ("a", 7001), extra=["--system", f"a=@{mine}",
                                       "--kinds", "support"])
    assert stub_servers[0][1]["messages"][0]["content"] == "Judge carefully."
    for extra in (["--system", "a=nope"], ["--system", "ghost=strict"],
                  ["--temperature", "ghost=0.1"], ["--temperature", "a"],
                  ["--kinds", "nonsense"]):
        with pytest.raises(SystemExit):
            _run(tmp_path, ("a", 7001), extra=extra)


def test_the_shipped_verifier_variants():
    assert set(verdict_prompts.VARIANTS) >= {"default", "strict"}
    assert all("JSON" in v for v in verdict_prompts.VARIANTS.values())
    assert verdict_prompts.resolve("strict") == verdict_prompts.VARIANTS["strict"]


def test_the_most_missed_pairs_are_listed(tmp_path, stub_servers, capsys):
    r = _run(tmp_path, ("yes", 7002), extra=["--runs", "2"])["yes"]
    assert len(r["missed_pairs"]) == 19                  # every "no" pairing
    assert all(m.startswith("2x FALSE_AGREE") for m in r["missed_pairs"])
    out = capsys.readouterr().out
    assert "most-missed pairs" in out and "2x FALSE_AGREE" in out
    assert out.count("2x FALSE_AGREE") == 6              # capped per target


def test_a_perfect_model_has_no_missed_pairs(tmp_path, stub_servers, capsys):
    r = _run(tmp_path, ("good", 7001))["good"]
    assert r["missed_pairs"] == []
    assert "      none" in capsys.readouterr().out


def test_the_second_check_never_asks_a_contradiction_as_agreement():
    """The key must not read as 'does the quote agree with the claim' when the
    question is whether it CONTRADICTS the claim (the bug this fixes)."""
    from shared import web_search
    contradicted = web_search.second_check_prompt(
        {"verdict": "contradicted", "claim": "C", "evidence": "E", "url": "U"})
    supported = web_search.second_check_prompt(
        {"verdict": "supported", "claim": "C", "evidence": "E", "url": "U"})
    assert "directly contradict the claim" in contradicted
    assert "directly support the claim" in supported
    for prompt in (contradicted, supported):
        assert "agrees" not in prompt
        assert '{"answer": true} if YES' in prompt
        assert '{"answer": false} if NO' in prompt
