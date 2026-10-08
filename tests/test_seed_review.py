"""The seed review: does the seed hold enough story for the book's length?"""

import json

import pytest

import main as pipeline
from agents import architect, planner, seed_review
from shared import llm_client
from shared.context import reset_context
from shared.llm_client import EndpointUnavailable, load_config
from shared.llm_utils import render_bible
from shared.output import format_bible_markdown

SEED = """# Snowbound

**Target Length:** Friday evening through Sunday afternoon, approximately 50,000 words

Friday evening: the guests arrive and dinner is served.
Saturday morning: the snow traps everyone in the house.
Sunday afternoon: the final sauna scene, and everyone leaves.
"""


def _config(tmp_path, book=""):
    path = tmp_path / "config.yaml"
    path.write_text("book:\n  words_per_chapter: 2000\n  min_chapters: 3\n"
                    f"  max_chapters: 30\n  num_chapters: 5\n{book}"
                    f'output:\n  directory: "{tmp_path / "out"}"\n')
    load_config(path)
    reset_context()


def _assessment(**over):
    data = {
        "stated": {"total_words": 50000, "words_per_chapter": None,
                   "chapters_min": None, "chapters_max": None,
                   "quote": "approximately 50,000 words"},
        "timespan": "Friday evening to Sunday afternoon (about 48 hours)",
        "scenes": [
            {"what": "Arrival and dinner", "quote": "the guests arrive and dinner is served"},
            {"what": "Snowed in", "quote": "the snow traps everyone in the house"},
            {"what": "Sauna finale", "quote": "the final sauna scene"},
            {"what": "Invented", "quote": "a duel at dawn on the frozen lake"}],
        "threads": ["Stuart and Kirsty's marriage"],
        "natural_words": {"low": 15000, "high": 25000},
        "verdict": "too_long",
        "reasons": ["Three scenes cannot fill 50,000 words."],
        "recommended": {"chapters": 10, "words_per_chapter": 2000},
        "to_fill_requested": ["more Saturday scenes"],
        "questions": [{"question": "Is there a subplot for the hosts?",
                       "why": "it could carry Saturday",
                       "default": "No hosts subplot."},
                      {"question": "POV?", "why": "voice",
                       "default": "Alternating Stuart and Kirsty."}],
    }
    data.update(over)
    return data


def _llm(monkeypatch, *replies):
    calls = []
    queue = list(replies)

    def fake(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    monkeypatch.setattr(seed_review, "generate_with_wait", fake)
    return calls


def _answers(monkeypatch, *answers):
    queue = list(answers)
    asked = []

    def fake_input(prompt):
        asked.append(prompt)
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr(seed_review, "_input", fake_input)
    monkeypatch.setattr(seed_review, "interactive", lambda: True)
    return asked


# ------------------------------------------------------------- parsing

def test_parse_verifies_quotes_against_the_seed():
    a = seed_review.parse_assessment(json.dumps(_assessment()), SEED, 5)
    assert [s["found"] for s in a["scenes"]] == [True, True, True, False]
    assert a["stated"]["total_words"] == 50000
    assert a["verdict"] == "too_long"
    assert len(a["questions"]) == 2


def test_a_stated_length_whose_quote_is_not_in_the_seed_is_ignored():
    raw = _assessment(stated={"total_words": 90000, "quote": "about ninety thousand words"})
    a = seed_review.parse_assessment(json.dumps(raw), SEED, 5)
    assert a["stated"]["total_words"] is None and a["stated"]["quote"] == ""


def test_parse_is_forgiving_about_shapes_and_caps_questions():
    raw = {"scenes": ["just a string"], "verdict": "nonsense",
           "questions": ["q1", "q2", "q3"], "recommended": {"chapters": "12"}}
    a = seed_review.parse_assessment(json.dumps(raw), SEED, 2)
    assert a["verdict"] == "about_right"
    assert a["scenes"] == [{"what": "just a string", "quote": "", "found": False}]
    assert [q["question"] for q in a["questions"]] == ["q1", "q2"]
    assert a["recommended"]["chapters"] == 12


def test_unreadable_reply_raises():
    with pytest.raises(ValueError):
        seed_review.parse_assessment("not json at all", SEED, 5)


# ------------------------------------------------------ sizes and arithmetic

def test_requested_size_priority():
    stated = {"total_words": 50000, "words_per_chapter": None}
    assert seed_review.requested_size(8, 4, stated, 2000)["source"] == \
        "your --chapters option"
    assert seed_review.requested_size(None, 4, stated, 2000)["chapters"] == 4
    from_stated = seed_review.requested_size(None, 0, stated, 2000)
    assert (from_stated["chapters"], from_stated["words_per_chapter"]) == (25, 2000)
    assert seed_review.requested_size(None, 0, {"total_words": None}, 2000) is None
    per = seed_review.requested_size(None, 0, {"total_words": 30000,
                                               "words_per_chapter": 3000}, 2000)
    assert (per["chapters"], per["words_per_chapter"]) == (10, 3000)


def test_recommendation_stays_inside_config_and_stated_ranges(tmp_path):
    _config(tmp_path)
    a = seed_review.parse_assessment(json.dumps(_assessment(
        recommended={"chapters": 50, "words_per_chapter": 2000})), SEED, 5)
    assert seed_review.recommended_size(a, 2000)["chapters"] == 30
    a["stated"].update(chapters_min=5, chapters_max=8)
    assert seed_review.recommended_size(a, 2000)["chapters"] == 8
    a["recommended"]["chapters"] = 2
    assert seed_review.recommended_size(a, 2000)["chapters"] == 5


def test_recommendation_falls_back_to_the_natural_length(tmp_path):
    _config(tmp_path)
    a = seed_review.parse_assessment(json.dumps(_assessment(
        recommended={})), SEED, 5)
    assert seed_review.recommended_size(a, 2000)["chapters"] == 10   # 20k / 2k


def test_arithmetic_flags_a_stretched_book():
    a = seed_review.parse_assessment(json.dumps(_assessment()), SEED, 5)
    lines, flag = seed_review.arithmetic(
        a, {"chapters": 25, "words_per_chapter": 2000})
    assert flag == "stretched"
    assert "3 distinct scenes" in lines[0] and "16,667 words per scene" in lines[0]
    assert any("25 chapters for 3 scenes" in x for x in lines)
    _, flag = seed_review.arithmetic(a, {"chapters": 3, "words_per_chapter": 2000})
    assert flag is None
    _, flag = seed_review.arithmetic(a, {"chapters": 1, "words_per_chapter": 600})
    assert flag == "cramped"


def test_arithmetic_without_found_scenes_says_so():
    a = seed_review.parse_assessment(json.dumps(_assessment(scenes=[])), SEED, 5)
    lines, flag = seed_review.arithmetic(a, {"chapters": 5, "words_per_chapter": 2000})
    assert flag is None and "could not be checked" in lines[0]


def test_report_shows_the_verdict_and_unfound_scenes():
    a = seed_review.parse_assessment(json.dumps(_assessment()), SEED, 5)
    req = {"chapters": 25, "words_per_chapter": 2000, "source": "the seed"}
    text = seed_review.report(a, req, None, ["note"], "stretched")
    assert "TOO LONG" in text and "about 50,000" in text
    assert "*(quote not found in the seed)*" in text
    assert "more Saturday scenes" in text


# ---------------------------------------------------------------- the step

def test_off_does_nothing(tmp_path, monkeypatch):
    _config(tmp_path, "  seed_review: off\n")
    calls = _llm(monkeypatch, _assessment())
    out = seed_review.run_seed_review(SEED)
    assert out["chapters"] is None and calls == []


def test_ask_without_a_terminal_only_warns(tmp_path, monkeypatch, capsys):
    _config(tmp_path)
    monkeypatch.setattr(seed_review, "interactive", lambda: False)
    calls = _llm(monkeypatch, _assessment())
    monkeypatch.setattr(seed_review, "_input",
                        lambda p: pytest.fail("must not prompt"))
    out = seed_review.run_seed_review(SEED)
    assert out["chapters"] == 25 and not out["stop"]   # the request stands
    assert "at most 0" in calls[0]["prompt"]           # no questions asked for
    text = capsys.readouterr().out
    assert "TOO LONG" in text and "Warning: the requested size does not fit" in text
    plan = json.loads((tmp_path / "out" / "interim" / "plan.json").read_text())
    assert plan["chapters"] == 25


def test_answers_trigger_a_recheck_and_the_recommendation_can_be_taken(
        tmp_path, monkeypatch):
    _config(tmp_path)
    calls = _llm(monkeypatch, _assessment(),
                 _assessment(recommended={"chapters": 14,
                                          "words_per_chapter": 2500},
                             questions=[]))
    _answers(monkeypatch, "Yes: the hosts are divorcing", "", "r")
    out = seed_review.run_seed_review(SEED)
    assert (out["chapters"], out["words_per_chapter"]) == (14, 2500)
    assert out["clarifications"] == [
        {"question": "Is there a subplot for the hosts?",
         "answer": "Yes: the hosts are divorcing", "answered": True},
        {"question": "POV?", "answer": "Alternating Stuart and Kirsty.",
         "answered": False}]
    assert len(calls) == 2
    assert "the hosts are divorcing" in calls[1]["prompt"]
    assert "at most 0" in calls[1]["prompt"]           # no second round
    saved = (tmp_path / "out" / "interim" / "seed_review.md").read_text()
    assert "Clarifications" in saved and "*(assumed)*" in saved


def test_unanswered_questions_need_no_recheck_and_enter_keeps_the_request(
        tmp_path, monkeypatch):
    _config(tmp_path)
    calls = _llm(monkeypatch, _assessment())
    _answers(monkeypatch, "skip", "")
    out = seed_review.run_seed_review(SEED)
    assert len(calls) == 1
    assert out["chapters"] == 25
    assert all(not c["answered"] for c in out["clarifications"])


def test_custom_size(tmp_path, monkeypatch):
    _config(tmp_path)
    _llm(monkeypatch, _assessment(questions=[]))
    _answers(monkeypatch, "c", "12", "1800")
    out = seed_review.run_seed_review(SEED)
    assert (out["chapters"], out["words_per_chapter"]) == (12, 1800)


def test_stop(tmp_path, monkeypatch):
    _config(tmp_path)
    _llm(monkeypatch, _assessment(questions=[]))
    _answers(monkeypatch, "x", "s")                     # a bad letter, then stop
    out = seed_review.run_seed_review(SEED)
    assert out["stop"]
    assert not (tmp_path / "out" / "interim" / "plan.json").exists()


def test_a_fitting_request_is_not_questioned(tmp_path, monkeypatch):
    _config(tmp_path)
    _llm(monkeypatch, _assessment(verdict="about_right", questions=[]))
    _answers(monkeypatch)                               # any prompt -> EOF
    out = seed_review.run_seed_review(SEED, num_chapters=3)
    assert out["chapters"] == 3


def test_no_request_uses_the_recommendation(tmp_path, monkeypatch):
    _config(tmp_path, "  seed_review: warn\n")
    _llm(monkeypatch, _assessment(stated={}, verdict="about_right"))
    out = seed_review.run_seed_review("A premise only.")
    assert out["chapters"] == 10


def test_a_failed_review_falls_back_to_the_old_planning(tmp_path, monkeypatch,
                                                        capsys):
    _config(tmp_path, "  seed_review: warn\n")
    _llm(monkeypatch, "garbage")
    out = seed_review.run_seed_review(SEED)
    assert out["chapters"] is None and not out["stop"]
    assert "Could not review the seed" in capsys.readouterr().out


def test_an_outage_is_not_swallowed(tmp_path, monkeypatch):
    _config(tmp_path, "  seed_review: warn\n")
    _llm(monkeypatch, EndpointUnavailable("down"))
    with pytest.raises(EndpointUnavailable):
        seed_review.run_seed_review(SEED)


def test_unknown_mode_falls_back_to_ask(tmp_path, capsys):
    _config(tmp_path, "  seed_review: maybe\n")
    assert seed_review.mode() in ("ask", "warn")
    assert "Unknown book.seed_review" in capsys.readouterr().out


# ------------------------------------------- clarifications reach the book

def test_clarifications_go_into_the_bible_and_every_prompt(tmp_path,
                                                           monkeypatch):
    _config(tmp_path)
    seen = []

    def fake(prompt, **kw):
        seen.append(prompt)
        return json.dumps({"title": "Snowbound", "characters": []})

    monkeypatch.setattr(architect, "generate_with_wait", fake)
    clar = [{"question": "Hosts subplot?", "answer": "They are divorcing",
             "answered": True}]
    bible = architect.run_architect(SEED, clar)
    assert "They are divorcing" in seen[0]
    assert bible["clarifications"][0]["answer"] == "They are divorcing"
    assert bible["seed"] == SEED                    # resume compares this
    assert "Author's clarifications" in render_bible(bible)
    assert "## Clarifications" in format_bible_markdown(bible)


def test_the_planner_sees_the_seed_and_the_clarifications(tmp_path,
                                                          monkeypatch):
    _config(tmp_path)
    seen = []

    def fake(prompt, **kw):
        seen.append(prompt)
        return json.dumps([{"title": "A", "summary": "a"},
                           {"title": "B", "summary": "b"}])

    monkeypatch.setattr(planner, "generate_with_wait", fake)
    bible = {"title": "T", "seed": SEED, "clarifications": [
        {"question": "Hosts?", "answer": "Divorcing"}]}
    planner._plan_with_llm(bible, 2)
    assert "the snow traps everyone" in seen[0] and "Divorcing" in seen[0]
    assert "do not split one scene across chapters" in seen[0]


def test_apply_plan_restores_the_chosen_chapter_length(tmp_path):
    _config(tmp_path)
    assert pipeline.apply_plan({"chapters": 14, "words_per_chapter": 2500}) == 14
    assert llm_client.get_config()["book"]["words_per_chapter"] == 2500
    assert pipeline.apply_plan(None) is None
    assert pipeline.apply_plan({"chapters": None, "words_per_chapter": "x"}) is None


def test_stopping_writes_nothing_to_resume(tmp_path, monkeypatch):
    _config(tmp_path)
    monkeypatch.setattr(pipeline, "run_seed_review", lambda *a: {
        "chapters": None, "words_per_chapter": 2000, "clarifications": [],
        "stop": True})
    monkeypatch.setattr(pipeline, "run_architect",
                        lambda *a: pytest.fail("must not run"))
    assert pipeline.run_pipeline(SEED) == 0
    assert not (tmp_path / "out" / "interim" / "bible.json").exists()


@pytest.fixture(autouse=True)
def _clean():
    yield
    reset_context()
