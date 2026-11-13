"""Unit tests for the pure scoring code in tools/style_eval.py.

The tool itself needs a live model; these pin the measurements so a threshold
picked from its output means the same thing next month.
"""

from tools import style_eval


def test_word_count_and_target_ratio():
    prompt = {"target": 100}
    scored = style_eval.score(prompt, "one two three " * 40)
    assert scored["words"] == 120
    assert scored["target_ratio"] == 1.2


def test_distinct_n_falls_when_text_repeats():
    varied = ("the rain fell on the roof and the wind moved through the "
              "broken windows while somewhere a dog barked at nothing")
    repeated = "the rain fell on the roof " * 8
    assert style_eval.distinct_n(varied, 2) > style_eval.distinct_n(repeated, 2)
    assert style_eval.distinct_n("one", 2) == 1.0


def test_stock_phrase_density_scales_with_length():
    one = "A coil of smoke. " + "The street was empty. " * 90
    many = "A coil of smoke. " * 10
    assert style_eval.stock_per_1k(many) > style_eval.stock_per_1k(one)


def test_hedge_density():
    hedged = ("It seemed to matter. She somehow knew. A sense of dread. "
              + "The clock ticked. " * 20)
    plain = "The clock ticked. " * 20
    assert style_eval.hedge_per_1k(hedged) > style_eval.hedge_per_1k(plain)


def test_loop_score_finds_a_repeated_run():
    looped = "and then the door opened " * 10
    normal = ("the door opened and she stepped through into a hall that "
              "smelled of coal smoke and old carpet")
    assert style_eval.loop_score(looped) > style_eval.loop_score(normal)


def test_sentence_cv_separates_flat_from_varied():
    flat = ("He walked in. She looked up. He sat down. She said nothing. "
            "He waited. She turned away.")
    varied = ("He walked in. She looked up from the ledger, and for a moment "
              "neither of them said anything at all. He sat down. The chair "
              "creaked. Outside, a cart went past, and the sound of it faded "
              "slowly into the rain and did not come back.")
    assert style_eval.sentence_cv(flat) < style_eval.sentence_cv(varied)


def test_dialogue_ratio():
    text = '"Hello," she said.\n"I am here," he said.\nThe room was quiet.'
    assert style_eval.dialogue_ratio(text) == round(2 / 3, 3)


def test_pov_and_tense():
    first = ("I opened the door. I saw the body. My hands were wet. "
             "I had met the man. I took the letter. I left.")
    pov, tense = style_eval.pov_tense(first)
    assert pov == "first"
    assert tense == "past"


def test_lore_recall_counts_planted_facts():
    kept = ("Sera kept the lighthouse. The brass key on a red cord stayed "
            "on her neck. Ianthe wrote about the storm glass.")
    recall, kept_labels = style_eval.lore_recall(kept)
    assert recall == 5
    assert "ten" not in kept_labels


def test_summarise_reports_the_expected_rates():
    results = [
        {"words": 100, "refusals": 0, "target_ratio": 1.0,
         "stock_per_1k": 1.0, "hedge_per_1k": 2.0, "loop": 0.01,
         "d2": 0.9, "d3": 0.95, "sentence_cv": 0.4,
         "explicit_per_1k": 3.0, "lore_recall": 4},
        {"words": 100, "refusals": 2, "target_ratio": 0.6,
         "stock_per_1k": 3.0, "hedge_per_1k": 4.0, "loop": 0.03,
         "d2": 0.7, "d3": 0.8, "sentence_cv": 0.2,
         "explicit_per_1k": 1.0, "lore_recall": 2},
    ]
    summary = style_eval.summarise(results)
    assert summary["n"] == 2
    assert summary["refusal_rate"] == 0.5
    assert summary["within_20pct"] == 0.5
    assert summary["mean_stock"] == 2.0
    assert summary["mean_lore_recall"] == 3.0


def test_deltas_are_direction_aware():
    base = {"mean_stock": 3.0, "mean_d2": 0.80}
    current = {"mean_stock": 1.0, "mean_d2": 0.75}
    verdicts = {key: verdict
                for key, _, _, _, verdict in style_eval.deltas(current, base)}
    assert verdicts["mean_stock"] == "better"
    assert verdicts["mean_d2"] == "worse"


def test_every_prompt_has_a_target_and_tags():
    for prompt in style_eval.PROMPTS:
        assert prompt["target"] > 0
        assert prompt["tags"], prompt["id"]
        assert prompt["text"].strip()
