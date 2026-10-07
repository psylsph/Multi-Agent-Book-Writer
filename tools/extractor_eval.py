#!/usr/bin/env python3
"""Compare extractor setups on the pipeline's real extraction step.

The extractor reads a finished chapter and records who is on the page and
what happened (first meetings, deaths, injuries...). A wrong event poisons
every later continuity check, so before changing the model, its temperature,
its prompt or the safety checks around it, measure.

A "target" is a server + model, and a LABEL for the setup. The same server and
model can appear under several labels with different options, which is how you
compare temperatures, system prompts or checks on/off:

  uv run python tools/extractor_eval.py --runs 3 \\
      --target main      http://127.0.0.1:8080 my-model \\
      --target main-cold http://127.0.0.1:8080 my-model \\
      --target main-nochk http://127.0.0.1:8080 my-model \\
      --temperature main-cold=0.1 --no-checks main-nochk \\
      --system main-cold=strict

Per-target options (repeatable, LABEL or LABEL=VALUE):
  --temperature LABEL=0.1    extractor sampling temperature
  --system LABEL=NAME        system prompt: a variant from
                             tools/extractor_prompts.py or @file.txt
  --no-thinking LABEL        thinking off (a non-reasoning/hybrid model)
  --no-checks LABEL          switch off book.extraction_checks

Mode 1 (default): synthetic chapters with KNOWN answers (tools/extractor_cases)
Mode 2 (--chapters-dir DIR): your real chapters; there is no ground truth, so
    each target is compared with the FIRST target (the reference).

The extraction runs through agents.writer._summarize_and_extract, so the
prompt, name snapping, per-agent reasoning defaults, json_mode and the
extraction checks are exactly what a real run uses.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import writer  # noqa: E402
from shared import llm_client, prompts  # noqa: E402
from shared.context import reset_context, update_context  # noqa: E402
from shared.story_state import normalize_state  # noqa: E402
from tools import extractor_prompts  # noqa: E402
from tools.extractor_cases import BIBLE_CHARACTERS, CASES  # noqa: E402


# ------------------------------------------------------------------ scoring

def _norm(name):
    return " ".join(str(name).lower().split())


def event_key(event):
    return event["type"], frozenset(_norm(w) for w in event["who"])


def score_state(state, expected_present, expected_events, optional=()):
    """Score one extracted state against the truth. Returns raw counts so
    they can be summed over cases:

    present_hit / present_expected / present_got
        (names in `optional` are ambiguous and count neither way)
    strict:   expected events found with exactly the right people
    lenient:  expected events found with the right type and >=1 right person
    spurious: extracted events matching no expected event (even leniently)
    false_deaths: extracted deaths matching no expected death, the most
        damaging invention (a living character is declared dead)
    deaths_expected / deaths_found: real deaths, and how many were caught
    """
    skip = {_norm(n) for n in optional}
    got_present = {_norm(n) for n in state["present"]} - skip
    want_present = {_norm(n) for n in expected_present} - skip
    got = [(e["type"], {_norm(w) for w in e["who"]}) for e in state["events"]]
    want = [(t, {_norm(w) for w in who}) for t, who in expected_events]

    def lenient(ev, other):
        return ev[0] == other[0] and bool(ev[1] & other[1])

    strict_hits = sum(1 for t, who in want
                      if any(g[0] == t and g[1] == who for g in got))
    lenient_hits = sum(1 for w in want if any(lenient(g, w) for g in got))
    spurious = [g for g in got if not any(lenient(g, w) for w in want)]
    deaths = [w for w in want if w[0] == "death"]
    return {
        "present_hit": len(got_present & want_present),
        "present_expected": len(want_present),
        "present_got": len(got_present),
        "events_expected": len(want),
        "strict": strict_hits,
        "lenient": lenient_hits,
        "events_got": len(got),
        "spurious": len(spurious),
        "false_deaths": sum(1 for g in spurious if g[0] == "death"),
        "deaths_expected": len(deaths),
        "deaths_found": sum(1 for w in deaths
                            if any(lenient(g, w) for g in got)),
    }


def _ratio(num, den, empty=1.0):
    return num / den if den else empty


COUNT_KEYS = ("present_hit", "present_expected", "present_got",
              "events_expected", "strict", "lenient", "events_got",
              "spurious", "false_deaths", "deaths_expected", "deaths_found")


def problem_cases(per_case, cases):
    """Where the errors are: {number: 'title: 2 false deaths, 1 invented ...'}
    for every case that had any, from {number: summed counts}."""
    titles = {c["number"]: c["title"] for c in cases}
    out = {}
    for number, c in sorted(per_case.items()):
        bits = []
        if c["false_deaths"]:
            bits.append(f"{c['false_deaths']} FALSE DEATH(S)")
        invented = c["spurious"] - c["false_deaths"]
        if invented:
            bits.append(f"{invented} invented event(s)")
        missed = c["events_expected"] - c["lenient"]
        if missed:
            bits.append(f"{missed} event(s) missed")
        wrong_names = c["present_got"] - c["present_hit"]
        if wrong_names:
            bits.append(f"{wrong_names} wrong name(s) in 'present'")
        if c["present_expected"] - c["present_hit"]:
            bits.append(f"{c['present_expected'] - c['present_hit']} name(s) "
                        "missing from 'present'")
        if bits:
            out[number] = f"case {number} '{titles.get(number, '?')}': " \
                          + ", ".join(bits)
    return out


def summarize(counts, failures, total, seconds, tokens, calls=0):
    """Aggregate per-case counts into the rates shown in the table."""
    s = {k: sum(c[k] for c in counts) for k in COUNT_KEYS}
    return {
        "valid_json": f"{total - failures}/{total}",
        "present_recall": _ratio(s["present_hit"], s["present_expected"]),
        "present_precision": _ratio(s["present_hit"], s["present_got"]),
        "event_recall_strict": _ratio(s["strict"], s["events_expected"]),
        "event_recall_lenient": _ratio(s["lenient"], s["events_expected"]),
        "spurious_events": s["spurious"],
        "false_deaths": s["false_deaths"],
        "deaths_found": f"{s['deaths_found']}/{s['deaths_expected']}",
        "calls_per_chapter": calls / total if total else 0.0,
        "sec_per_chapter": seconds / total if total else 0.0,
        "tokens": tokens,
    }


# ------------------------------------------------------------------ running

def configure(url, model, opts, json_mode):
    """Point the shared client at one target, as a real run would be."""
    cfg = llm_client.get_config()
    cfg["llm"].update(base_url=url, model=model, retries=0, endpoint_wait=0,
                      json_mode=json_mode)
    cfg.setdefault("book", {})["extraction_checks"] = opts.get("checks", True)
    extractor = cfg.setdefault("agents", {}).setdefault("extractor", {})
    for key in ("model", "reasoning_effort", "enable_thinking", "temperature"):
        extractor.pop(key, None)
    if opts.get("no_thinking"):
        extractor["reasoning_effort"] = ""          # no effort; thinking off
        extractor["enable_thinking"] = False
    if opts.get("temperature") is not None:
        extractor["temperature"] = opts["temperature"]
    reset_context()
    update_context("bible", {"characters": BIBLE_CHARACTERS})
    llm_client.reset_stats()


def _prior_chronology(case):
    """Earlier-chapter state for cases that start with someone already dead."""
    names = case.get("prior_deaths")
    if not names:
        return {}
    canon = [c["name"] for c in BIBLE_CHARACTERS]
    return {0: normalize_state(0, "earlier", {"events": [
        {"type": "death", "who": names}]}, canon)}


def run_target(label, url, model, cases, opts, json_mode=False, runs=1,
               verbose=False):
    configure(url, model, opts, json_mode)
    original_system = prompts.EXTRACTOR
    if opts.get("system"):
        prompts.EXTRACTOR = extractor_prompts.resolve(opts["system"])
    counts, failures, total, started = [], 0, 0, time.time()
    states, per_case = {}, {}

    def tally(case, counted):
        sums = per_case.setdefault(case["number"], {k: 0 for k in COUNT_KEYS})
        for key in COUNT_KEYS:
            sums[key] += counted[key]
    try:
        for case in cases:
            for _ in range(runs):
                total += 1
                update_context("chronology", _prior_chronology(case))
                result = writer._summarize_and_extract(
                    case["number"], case["title"], case["text"],
                    fallback=False)
                if result is None:
                    failures += 1
                    counts.append(score_state(
                        {"present": [], "events": []}, case["present"],
                        case["events"], case.get("present_optional", ())))
                    tally(case, counts[-1])
                    if verbose:
                        print(f"  [{label}] chapter {case['number']}: FAILED "
                              "(no valid JSON)")
                    continue
                _, state = result
                states[case["number"]] = state
                counts.append(score_state(
                    state, case["present"], case["events"],
                    case.get("present_optional", ())))
                tally(case, counts[-1])
                if verbose:
                    print(f"  [{label}] chapter {case['number']}: present="
                          f"{state['present']} events="
                          f"{[(e['type'], e['who']) for e in state['events']]}")
    finally:
        prompts.EXTRACTOR = original_system
    stats = llm_client.STATS.get("extractor", {})
    tokens = stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0)
    summary = summarize(counts, failures, total, time.time() - started, tokens,
                        stats.get("calls", 0))
    summary["problem_cases"] = problem_cases(per_case, cases)
    return summary, states


def load_real_chapters(directory):
    """(number, title, text) for chapter_NN.md files in `directory`."""
    cases = []
    for path in sorted(Path(directory).glob("chapter_*.md")):
        number = int(re.search(r"(\d+)", path.stem).group(1))
        text = path.read_text(encoding="utf-8")
        title = (re.match(r"#+\s*Chapter \d+:\s*(.+)", text) or
                 [None, f"Chapter {number}"])[1]
        body = text.split("\n\n", 1)[1] if text.startswith("#") else text
        cases.append({"number": number, "title": title.strip(), "text": body,
                      "present": [], "events": []})
    return cases


def agreement(reference, other):
    """How closely `other`'s extraction matches the reference's, per chapter
    set: Jaccard over present names and over (type, who) events."""
    def jac(a, b):
        return len(a & b) / len(a | b) if a | b else 1.0
    chapters = sorted(set(reference) & set(other))
    present = [jac({_norm(n) for n in reference[c]["present"]},
                   {_norm(n) for n in other[c]["present"]}) for c in chapters]
    events = [jac({event_key(e) for e in reference[c]["events"]},
                  {event_key(e) for e in other[c]["events"]})
              for c in chapters]
    n = len(chapters) or 1
    return {"chapters": len(chapters), "present": sum(present) / n,
            "events": sum(events) / n}


# --------------------------------------------------------------------- CLI

COLUMNS = [("valid_json", "JSON ok", "{}"),
           ("present_recall", "pres R", "{:.2f}"),
           ("present_precision", "pres P", "{:.2f}"),
           ("event_recall_strict", "evt R str", "{:.2f}"),
           ("event_recall_lenient", "evt R len", "{:.2f}"),
           ("spurious_events", "invented", "{}"),
           ("false_deaths", "FALSE DTH", "{}"),
           ("deaths_found", "deaths", "{}"),
           ("calls_per_chapter", "calls/ch", "{:.1f}"),
           ("sec_per_chapter", "s/ch", "{:.1f}"),
           ("tokens", "tokens", "{:,}")]


def format_table(results):
    labels = list(results)
    width = max(len(label) for label in labels + ["target"]) + 2
    head = "target".ljust(width) + "".join(h.rjust(11) for _, h, _ in COLUMNS)
    lines = [head, "-" * len(head)]
    for label in labels:
        row = label.ljust(width)
        for key, _, fmt in COLUMNS:
            row += fmt.format(results[label][key]).rjust(11)
        lines.append(row)
    return "\n".join(lines)


def _label_values(items, cast=str):
    """['a=1', 'b=2'] -> {'a': cast('1'), 'b': cast('2')}"""
    out = {}
    for item in items or []:
        label, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"expected LABEL=VALUE, got '{item}'")
        out[label] = cast(value)
    return out


def build_options(args):
    """Per-label options from the command line."""
    temps = _label_values(args.temperature, float)
    systems = _label_values(args.system)
    options = {}
    for label, _, _ in args.target:
        options[label] = {
            "no_thinking": label in args.no_thinking,
            "checks": label not in args.no_checks,
            "temperature": temps.get(label),
            "system": systems.get(label),
        }
    for spec in systems.values():             # fail early, with a clear message
        try:
            extractor_prompts.resolve(spec)
        except (KeyError, OSError) as e:
            raise SystemExit(f"--system: {e}")
    known = {label for label, _, _ in args.target}
    for name, mapping in (("--temperature", temps), ("--system", systems)):
        for label in mapping:
            if label not in known:
                raise SystemExit(f"{name} names unknown target '{label}'")
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score extractor setups on chapters with known answers.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", nargs=3, action="append", required=True,
                        metavar=("LABEL", "URL", "MODEL"),
                        help="a server/model setup to test (repeatable; the "
                             "first is the reference in --chapters-dir mode)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json-mode", action="store_true",
                        help="ask servers for JSON-object output")
    parser.add_argument("--no-thinking", action="append", default=[],
                        metavar="LABEL",
                        help="thinking off for this target: no reasoning "
                             "effort, enable_thinking=false (plain models "
                             "ignore the flag)")
    parser.add_argument("--temperature", action="append", default=[],
                        metavar="LABEL=T", help="extractor temperature")
    parser.add_argument("--system", action="append", default=[],
                        metavar="LABEL=NAME",
                        help="system prompt: "
                             + ", ".join(extractor_prompts.VARIANTS)
                             + ", or @file.txt")
    parser.add_argument("--no-checks", action="append", default=[],
                        metavar="LABEL",
                        help="switch off book.extraction_checks for a target")
    parser.add_argument("--cases", metavar="N,N",
                        help="only these case numbers, e.g. 4,6,8")
    parser.add_argument("--runs", type=int, default=1,
                        help="repeat each chapter N times (models are "
                             "non-deterministic)")
    parser.add_argument("--chapters-dir", metavar="DIR",
                        help="compare targets on your own chapter_NN.md files "
                             "instead (no ground truth)")
    parser.add_argument("--json", metavar="FILE", help="also write results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    llm_client.load_config(args.config)
    options = build_options(args)
    real = bool(args.chapters_dir)
    cases = load_real_chapters(args.chapters_dir) if real else CASES
    if args.cases and not real:
        wanted = {int(n) for n in args.cases.split(",")}
        cases = [c for c in cases if c["number"] in wanted]
    if not cases:
        sys.exit("No chapters found.")

    results, extracted = {}, {}
    for label, url, model in args.target:
        print(f"== {label}: {model} @ {url} ({len(cases)} chapters)")
        try:
            results[label], extracted[label] = run_target(
                label, url, model, cases, options[label], args.json_mode,
                args.runs, args.verbose)
        except Exception as e:                  # an unreachable server
            print(f"   could not run: {e}")

    if not results:
        sys.exit("No target produced results.")
    if real:
        reference = args.target[0][0]
        print(f"\nAgreement with the reference '{reference}' on your "
              "chapters (1.00 = identical):")
        for label in extracted:
            if label != reference and reference in extracted:
                a = agreement(extracted[reference], extracted[label])
                print(f"  {label}: present names {a['present']:.2f}, events "
                      f"{a['events']:.2f} over {a['chapters']} chapters")
        results = {k: {kk: vv for kk, vv in v.items()
                       if kk in ("valid_json", "sec_per_chapter", "tokens",
                                 "calls_per_chapter")}
                   for k, v in results.items()}
    else:
        print("\n" + format_table(results))
        print("\nWhere the errors are (summed over runs):")
        for label, r in results.items():
            lines = list(r["problem_cases"].values()) or ["none"]
            print(f"  {label}")
            for line in lines:
                print(f"      {line}")
        print("\nR = recall, P = precision; str/len = strict/lenient match. "
              "'invented' = events that match nothing real. 'FALSE DTH' = "
              "living characters declared dead (should be 0); 'deaths' = real "
              "deaths caught.")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
