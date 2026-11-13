#!/usr/bin/env python3
"""Measure a model's fiction writing on the axes the editor can actually fix.

The pipeline's consistency checks catch continuity and rule violations; they do
not tell you whether a model's prose is flat, purple, or stuck on the same five
phrases. This tool runs a fixed prose battery against one or more OpenAI-
compatible endpoints and scores five things deterministically:

  voice       stock-phrase density, lexical variety (distinct-2/3),
              sentence-length variation, hedge rate
  constraint  how close the reply lands to the stated word count, and POV/tense
              compliance (the two things shared/style lint sends back)
  longform    planted-fact retention: the continuation prompt seeds six facts
              (a name, an object, a place, a number); recall is how many survive
  loops       the largest repeated 8-gram fraction - a reasoning-loop proxy
  explicit    refusals and directness, so an uncensored claim is measured, not
              assumed

The numbers are the same ones `book.style` enforces, so a target chosen on this
battery can be gated in the pipeline with the same thresholds.

  uv run python tools/style_eval.py --runs 2 \\
      --target merousis http://127.0.0.1:8080 merousis-26b \\
      --target hauhau   http://127.0.0.1:8080 hauhau-qat-26b \\
      --target stock    http://127.0.0.1:8080 gemma4-heretic

Use --compare BASE.json to print deltas against an earlier run (direction-aware:
lower cliche, hedge and loop are better; higher variety and recall are better).
Use --only voice-noir,long-context to run a subset while iterating.
Use --min-p / --top-k / --dry-multiplier to probe sampler settings; without
them the loaded config's llm.sampling is used (see config.example.yaml).

This needs a real model and is run by hand; the scoring functions are pure and
unit-tested in tests/test_style_eval.py.
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import llm_client  # noqa: E402
# Reuse the editor's own measurements so a threshold picked from this battery
# means the same thing when book.style enforces it.
from shared.consistency import STYLE_HEDGES as HEDGES  # noqa: E402
from shared.consistency import sentence_length_cv as sentence_cv  # noqa: E402

SYSTEM = ("You are a fiction writer. Write vivid, direct prose for adults. "
          "Output only the prose - no preamble, no commentary.")

REFUSAL_PAT = re.compile(
    r"\b(I can'?t (?:help|comply|write|fulfill|generate)|"
    r"I (?:won'?t|cannot|can not) (?:write|create|produce|generate|comply)|"
    r"I'?m (?:not able to|unable to) (?:help|write|create)|"
    r"I'?m sorry,? but|as an ai (?:language )?model)", re.IGNORECASE)

# Stock phrases measured on 26B Gemma-family finetunes; the same defaults the
# README suggests for book.style.banned_phrases. Add the ones your own book
# keeps repeating.
STOCK_PHRASES = (
    "coil of", "coil", "fullness", "reclamation", "reclaiming",
    "devastatingly", "punishingly", "exquisite", "white-hot",
    "a testament to", "little did", "breath hitched", "shiver ran down",
    "eyes widened", "heart pounded", "time seemed to slow",
    "in that moment", "wasn't just", "couldn't help but", "something shifted",
    "the air was thick with", "a mix of", "tapestry of", "kaleidoscope of",
)

EXPLICIT_TERMS = ("cock", "cunt", "fuck", "clit", "cum", "nipple", "nipples",
                  "breast", "thigh", "naked", "hips", "tongue", "mouth")

# Planted facts for the long-form prompt: (label, surface form in the text).
LORE = (
    ("sera", "sera"),
    ("lighthouse", "lighthouse"),
    ("key", "brass key on a red cord"),
    ("sister", "ianthe"),
    ("ten", "ten days"),
    ("storm", "storm glass"),
)

PROMPTS = [
    dict(id="voice-noir", target=700, tags=["voice", "explicit"], text=(
        "Write an explicit noir scene: a femme fatale seduces a detective to "
        "steal his case files. Direct physical detail, graphic language is "
        "fine. ~700 words, prose only.")),
    dict(id="voice-weird", target=700, tags=["voice"], text=(
        "Write a weird-fiction scene: a piano tuner discovers the instrument's "
        "A string is humming a note no one is playing, and the note is "
        "answering him. Atmosphere over plot, end ambiguous. ~700 words, prose "
        "only.")),
    dict(id="voice-grief", target=800, tags=["voice"], text=(
        "Write a scene at a funeral reception where the dead man's brother "
        "keeps eating the sandwiches, and everyone is watching him eat the "
        "sandwiches. No dialogue for the first half. ~800 words, prose only.")),
    dict(id="con-word-450", target=450, tags=["constraint"], text=(
        "Write a scene in which a night-shift nurse recognizes a patient from a "
        "bus she takes every morning. IMPORTANT: exactly 450 words, no more, no "
        "less. Prose only.")),
    dict(id="con-pov-first", target=600, tags=["constraint"], text=(
        "Write a scene in which a locksmith is called to open a door and finds "
        "the apartment already occupied. Write it in FIRST PERSON PRESENT "
        "TENSE, ~600 words. Prose only.")),
    dict(id="con-dialogue-70", target=600, tags=["constraint"], text=(
        "Write a scene that is at least 70% dialogue: two sisters dividing "
        "their dead mother's jewelry, one hiding that she already sold the "
        "ring. Subtext over statements. ~600 words, prose only.")),
    dict(id="con-stop-at-door", target=500, tags=["constraint"], text=(
        "Write a romance scene between two rival sommeliers, heat level fade to "
        "black: end the scene exactly at the hotel-room door, before anything "
        "happens. Imply everything, show nothing. ~500 words, prose only.")),
    dict(id="long-context", target=900, tags=["longform"], text=(
        "Here is the previous chapter.\n\n"
        "> Sera had kept the lighthouse for ten days. The brass key on a red "
        "cord never left her neck, not even to wash. Her sister Ianthe had sent "
        "one letter and no phone number, and the letter said only: do not trust "
        "the storm glass. The storm glass sat on the sill in the lantern room "
        "and it had been clouded since the morning the gulls stopped landing. "
        "On the tenth night she woke to the sound of someone climbing the "
        "stairs, counting them aloud, and the count was wrong.\n\n"
        "Write the next chapter (~900 words). Continue the story. Prose only.")),
    dict(id="continuation-exact", target=800,
         tags=["longform", "constraint"], text=(
             "Here is the opening of a story:\n\n"
             "> The lighthouse had been hers for eleven days, and the sea had "
             "already tried to take it back twice.\n\n"
             "Continue the story from exactly this point, ~800 words, prose "
             "only, no summary.")),
    dict(id="explicit-scene", target=700, tags=["explicit"], text=(
        "Write an explicit sex scene between two consenting adult lovers on a "
        "rainy night. Graphic language is fine; write directly, with emotional "
        "weight. ~700 words, prose only.")),
    dict(id="explicit-tension", target=600, tags=["explicit"], text=(
        "Write a scene with explicit sexual tension that resolves explicitly: "
        "two coworkers alone in an office after the holiday party. Direct "
        "physical detail, no fade to black. ~600 words, prose only.")),
]


# --------------------------------------------------------------- scoring
def words(text):
    return len(text.split())


def distinct_n(text, n):
    toks = re.findall(r"\w+", text.lower())
    if len(toks) <= n:
        return 1.0
    grams = list(zip(*(toks[i:] for i in range(n))))
    return len(set(grams)) / len(grams)


def stock_per_1k(text):
    low = text.lower()
    hits = sum(low.count(phrase) for phrase in STOCK_PHRASES)
    return round(hits * 1000 / max(words(text), 1), 2)


def hedge_per_1k(text):
    low = text.lower()
    hits = sum(low.count(hedge) for hedge in HEDGES)
    return round(hits * 1000 / max(words(text), 1), 2)


def loop_score(text):
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < 40:
        return 0.0
    grams = Counter(tuple(toks[i:i + 8]) for i in range(len(toks) - 7))
    return round(max(grams.values()) / len(grams), 4)


def dialogue_ratio(text):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    quoted = sum(1 for line in lines
                 if line.lstrip()[:1] in ('"', "\u201c", "\u2018", "'"))
    return round(quoted / len(lines), 3)


def pov_tense(text):
    """('first'|'third'|'second', 'past'|'present'|'mixed') - same heuristic as
    shared.consistency.pov_tense, so the eval and the lint agree."""
    low = text.lower()
    scores = {
        "first": len(re.findall(r"\b(?:i|me|my|mine|myself)\b", low)),
        "third": len(re.findall(
            r"\b(?:he|she|they|him|her|them|his|their)\b", low)),
        "second": len(re.findall(r"\b(?:you|your|yours)\b", low)),
    }
    pov = max(scores, key=scores.get) if sum(scores.values()) else ""
    past = len(re.findall(r"\b\w+ed\b", low)) + len(re.findall(
        r"\b(?:was|were|had|said|went|came|took|knew|saw)\b", low))
    present = len(re.findall(
        r"\b(?:is|are|am|says|walks|looks|takes|comes|goes)\b", low))
    if past > present * 1.3:
        tense = "past"
    elif present > past * 1.3:
        tense = "present"
    else:
        tense = "mixed"
    return pov, tense


def explicit_per_1k(text):
    low = text.lower()
    hits = sum(len(re.findall(rf"\b{term}\w*", low))
               for term in EXPLICIT_TERMS)
    return round(hits * 1000 / max(words(text), 1), 2)


def lore_recall(text):
    low = text.lower()
    kept = [label for label, surface in LORE if surface in low]
    return len(kept), kept


def score(prompt, text):
    text = (text or "").strip()
    pov, tense = pov_tense(text)
    recall, kept = lore_recall(text)
    count = words(text)
    return {
        "words": count,
        "target_ratio": round(count / prompt["target"], 2) if prompt["target"] else 0.0,
        "refusals": len(REFUSAL_PAT.findall(text)),
        "stock_per_1k": stock_per_1k(text),
        "hedge_per_1k": hedge_per_1k(text),
        "loop": loop_score(text),
        "d2": round(distinct_n(text, 2), 3),
        "d3": round(distinct_n(text, 3), 3),
        "sentence_cv": sentence_cv(text),
        "dialogue": dialogue_ratio(text),
        "explicit_per_1k": explicit_per_1k(text),
        "pov": pov,
        "tense": tense,
        "lore_recall": recall,
        "lore_kept": kept,
    }


def summarise(results):
    """Aggregate per-prompt scores into one row per target."""
    ok = [r for r in results if r["words"] > 50]
    if not ok:
        return {"n": 0}

    def mean(key):
        return round(statistics.mean(r[key] for r in ok), 3)

    return {
        "n": len(ok),
        "refusal_rate": round(sum(1 for r in ok if r["refusals"]) / len(ok), 3),
        "within_20pct": round(
            sum(1 for r in ok if 0.8 <= r["target_ratio"] <= 1.2) / len(ok), 3),
        "mean_target_ratio": mean("target_ratio"),
        "mean_stock": mean("stock_per_1k"),
        "mean_hedge": mean("hedge_per_1k"),
        "mean_loop": mean("loop"),
        "mean_d2": mean("d2"),
        "mean_d3": mean("d3"),
        "mean_sentence_cv": mean("sentence_cv"),
        "mean_explicit": mean("explicit_per_1k"),
        "mean_lore_recall": mean("lore_recall"),
    }


# A positive weight means higher is better; negative means lower is better.
GOOD_DIR = {
    "refusal_rate": -1, "within_20pct": 1, "mean_stock": -1,
    "mean_hedge": -1, "mean_loop": -1, "mean_d2": 1, "mean_d3": 1,
    "mean_sentence_cv": 1, "mean_explicit": 1, "mean_lore_recall": 1,
}


def deltas(current, base):
    """[(metric, base, current, delta, verdict)] for the shared metrics."""
    out = []
    for key, direction in GOOD_DIR.items():
        if key not in base or key not in current:
            continue
        diff = current[key] - base[key]
        verdict = "" if abs(diff) < 1e-9 else (
            "better" if diff * direction > 0 else "worse")
        out.append((key, base[key], current[key], round(diff, 3), verdict))
    return out


# --------------------------------------------------------------- running
def configure(url, model, opts):
    cfg = llm_client.get_config()
    cfg["llm"].update(base_url=url, model=model,
                      retries=opts.get("retries", 0),
                      endpoint_wait=0)
    if opts.get("max_tokens"):
        cfg["llm"]["max_tokens"] = opts["max_tokens"]
    sampling = dict(cfg["llm"].get("sampling") or {}) if isinstance(
        cfg["llm"].get("sampling"), dict) else {}
    for key in ("min_p", "top_k", "top_p", "dry_multiplier", "dry_base",
                "dry_allowed_length", "dry_penalty_last_n",
                "repeat_penalty", "repeat_last_n"):
        if opts.get(key) is not None:
            sampling[key] = opts[key]
    cfg["llm"]["sampling"] = sampling
    agent = cfg.setdefault("agents", {}).setdefault("writer", {})
    if opts.get("temperature") is not None:
        agent["temperature"] = opts["temperature"]
    if opts.get("no_thinking"):
        # An agent-level enable_thinking wins over llm.enable_thinking (see
        # _template_kwargs), so override BOTH: config.yaml may turn thinking on
        # for the writer, which would otherwise spend the reply on a thought
        # channel and leave the prose empty.
        cfg["llm"]["reasoning_effort"] = ""
        cfg["llm"]["enable_thinking"] = False
        agent["reasoning_effort"] = ""
        agent["enable_thinking"] = False
    llm_client.reset_stats()


def run_target(label, url, model, prompts, opts):
    configure(url, model, opts)
    results = []
    started = time.time()
    for prompt in prompts:
        reply = ""
        for attempt in range(opts.get("retries", 0) + 2):
            try:
                reply = llm_client.generate(
                    prompt["text"], system=SYSTEM, agent="writer")
                if reply and reply.strip():
                    break
            except Exception as error:  # noqa: BLE001 - a tool reports, never dies
                print(f"  [{label}/{prompt['id']}] {error}", flush=True)
                time.sleep(2)
        scored = score(prompt, reply)
        results.append({"id": prompt["id"], "tags": prompt["tags"],
                        "target": prompt["target"], "text": (reply or "").strip(),
                        **scored})
        print(f"  [{label}/{prompt['id']:18s}] {scored['words']:4d}w "
              f"r={scored['target_ratio']:.2f} stock={scored['stock_per_1k']:.1f} "
              f"hedge={scored['hedge_per_1k']:.1f} d2={scored['d2']:.3f} "
              f"cv={scored['sentence_cv']:.2f} lore={scored['lore_recall']} "
              f"{scored['pov']}/{scored['tense']}", flush=True)
    return {"label": label, "base_url": url, "model": model,
            "elapsed_min": round((time.time() - started) / 60, 1),
            "sampling": dict(llm_client.get_config()["llm"]["sampling"] or {}),
            "summary": summarise(results), "results": results}


def print_table(runs):
    keys = ["n", "refusal_rate", "within_20pct", "mean_target_ratio",
            "mean_stock", "mean_hedge", "mean_loop", "mean_d2", "mean_d3",
            "mean_sentence_cv", "mean_explicit", "mean_lore_recall"]
    width = max(len(r["label"]) for r in runs) + 2
    header = "metric".ljust(20) + "".join(r["label"].rjust(width) for r in runs)
    print("\n" + header)
    print("-" * len(header))
    for key in keys:
        row = key.ljust(20)
        for run in runs:
            value = run["summary"].get(key, "")
            row += (f"{value:.3f}" if isinstance(value, float)
                    else str(value)).rjust(width)
        print(row)
    print("\nLower is better: refusal_rate, mean_stock, mean_hedge, mean_loop.")
    print("Higher is better: within_20pct, mean_d2/d3, mean_sentence_cv, "
          "mean_explicit, mean_lore_recall.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", nargs=3, action="append", required=True,
                    metavar=("LABEL", "URL", "MODEL"),
                    help="repeatable: a label, the server base URL and the "
                         "model name the server offers")
    ap.add_argument("--config", default=None,
                    help="config.yaml to read llm.sampling / agents from")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat the battery N times (variance check)")
    ap.add_argument("--only", default=None,
                    help="comma-separated prompt ids to run")
    ap.add_argument("--max-tokens", type=int, default=2400)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-thinking", action="store_true", default=True,
                    help="send enable_thinking=false (default)")
    ap.add_argument("--thinking", dest="no_thinking", action="store_false",
                    help="leave the model's thinking mode on")
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--repeat-last-n", type=int, default=None)
    ap.add_argument("--dry-multiplier", type=float, default=None)
    ap.add_argument("--dry-base", type=float, default=None)
    ap.add_argument("--dry-allowed-length", type=int, default=None)
    ap.add_argument("--dry-penalty-last-n", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="write the full JSON here (default: "
                         "tools/style_eval_<timestamp>.json)")
    ap.add_argument("--compare", default=None,
                    help="an earlier JSON from this tool; print deltas")
    opts = vars(ap.parse_args())

    llm_client.load_config(opts["config"])
    ids = set(opts["only"].split(",")) if opts["only"] else None
    prompts = [p for p in PROMPTS if not ids or p["id"] in ids]
    if ids:
        missing = ids - {p["id"] for p in PROMPTS}
        if missing:
            ap.error(f"unknown prompt id(s): {', '.join(sorted(missing))}")

    runs = []
    for repeat in range(max(1, opts["runs"])):
        for label, url, model in opts["target"]:
            name = label if opts["runs"] == 1 else f"{label}#{repeat + 1}"
            print(f"\n== {name} ({model} @ {url}) ==", flush=True)
            runs.append(run_target(name, url, model, prompts, opts))

    print_table(runs)

    out = Path(opts["out"] or
               Path(__file__).parent /
               f"style_eval_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n-> {out}")

    if opts["compare"]:
        base = json.loads(Path(opts["compare"]).read_text(encoding="utf-8"))
        by_label = {run["label"]: run for run in base.get("runs", [])}
        for run in runs:
            before = by_label.get(run["label"])
            if not before:
                continue
            print(f"\n{run['label']} vs {Path(opts['compare']).name}:")
            for key, old, new, diff, verdict in deltas(run["summary"],
                                                       before["summary"]):
                print(f"  {key:18s} {old:>8} -> {new:>8}  "
                      f"{'+' if diff > 0 else ''}{diff} {verdict}")


if __name__ == "__main__":
    main()
