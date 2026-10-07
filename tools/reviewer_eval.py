#!/usr/bin/env python3
"""Score reviewers on chapters with planted errors.

The reviewer reads a chapter plus the recorded story facts and reports
continuity errors, missing outline beats and broken constraints; the editor
then revises the chapter. A MISS lets an error through. A FALSE ALARM costs a
wasted revision round (and a revision can introduce new errors). Before using
a different or smaller model as the reviewer, or trusting the main model to
review its own writing, measure both.

tools/reviewer_cases.py holds 14 versions of one chapter: 4 clean (two
deliberately tempting) and 10 each with one planted error of a kind the
reviewer claims to check.

  uv run python tools/reviewer_eval.py --runs 2 \\
      --target main http://127.0.0.1:8080 my-big-model \\
      --target small http://127.0.0.1:9090 my-small-model \\
      --system small=classic

Per-target options (repeatable, LABEL or LABEL=VALUE): --temperature,
--system (tools/reviewer_prompts.py variants or @file), --no-thinking and
--max-tokens (a cap on each reply: without one a model that loops can generate
for an hour; a reply cut off by the cap counts as unreadable).
It runs agents.reviewer.review_chapter(), the pipeline's own code. Interim
output is switched off, so nothing is written to your real output directory.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import reviewer  # noqa: E402
from shared import llm_client, prompts  # noqa: E402
from shared.context import reset_context, update_context  # noqa: E402
from tools import reviewer_prompts  # noqa: E402
from tools.reviewer_cases import (BIBLE, BRIEF, CASES, CHAPTERS,  # noqa: E402
                                  NUMBER, PLANTED_TYPES, TITLE, chronology)


# ------------------------------------------------------------------ scoring

def score_run(planted, issues):
    """Outcome of one review. planted is the expected issue type or None.

    clean chapter:   'ok' (no issues) or 'false_alarm'
    planted error:   'typed' (an issue of the planted type was reported),
                     'found' (issues, but none of the planted type: the error
                     was noticed but mislabelled, or other things were
                     reported) or 'missed'
    """
    if planted is None:
        return "false_alarm" if issues else "ok"
    types = {i["type"] for i in issues}
    if planted in types:
        return "typed"
    return "found" if issues else "missed"


def summarize(runs, seconds, tokens):
    """Rates from [{id, planted, outcome, types}] (outcome 'unreadable' for
    replies that could not be parsed)."""
    clean = [r for r in runs if r["planted"] is None]
    errors = [r for r in runs if r["planted"] is not None]
    unreadable = sum(r["outcome"] == "unreadable" for r in runs)

    def rate(num, den):
        return num / den if den else 0.0

    by_type = {}
    for r in errors:
        n, typed, found = by_type.get(r["planted"], (0, 0, 0))
        by_type[r["planted"]] = (
            n + 1, typed + (r["outcome"] == "typed"),
            found + (r["outcome"] in ("typed", "found")))
    return {
        "unreadable": unreadable,
        "false_alarm_rate": rate(sum(r["outcome"] == "false_alarm"
                                     for r in clean), len(clean)),
        "false_alarms": sum(r["outcome"] == "false_alarm" for r in clean),
        "recall_any": rate(sum(r["outcome"] in ("typed", "found")
                               for r in errors), len(errors)),
        "recall_typed": rate(sum(r["outcome"] == "typed" for r in errors),
                             len(errors)),
        "by_type": {k: f"{typed}/{n} (any {found})"
                    for k, (n, typed, found) in by_type.items()},
        "sec_per_chapter": seconds / len(runs) if runs else 0.0,
        "tokens": tokens,
    }


def problem_lines(runs):
    """Every miss, mislabel and false alarm, grouped by case."""
    out = {}
    for r in runs:
        if r["outcome"] in ("typed", "ok"):
            continue
        what = {"missed": "MISSED", "found": "found but not typed "
                f"'{r['planted']}' (reported: {', '.join(sorted(r['types']))})",
                "false_alarm": "FALSE ALARM ("
                + ", ".join(sorted(r["types"])) + ")",
                "unreadable": "UNREADABLE reply"}[r["outcome"]]
        out.setdefault(r["id"], []).append(what)
    lines = []
    for case_id, items in out.items():
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        lines.append(f"{case_id}: " + "; ".join(
            f"{n}x {item}" if n > 1 else item for item, n in counts.items()))
    return lines


# ------------------------------------------------------------------ running

def configure(url, model, opts):
    cfg = llm_client.get_config()
    cfg["llm"].update(base_url=url, model=model, retries=0, endpoint_wait=0,
                      json_mode=opts.get("json_mode", False))
    # never write review files into the user's real output directory
    cfg.setdefault("output", {})["interim"] = False
    cfg.setdefault("book", {})["review_checks"] = opts.get("checks") or list(
        reviewer.ALL_CHECKS)
    agent = cfg.setdefault("agents", {}).setdefault("reviewer", {})
    for key in ("model", "reasoning_effort", "enable_thinking", "temperature",
                "max_tokens"):
        agent.pop(key, None)
    if opts.get("max_tokens"):
        agent["max_tokens"] = int(opts["max_tokens"])
    if opts.get("no_thinking"):
        agent["reasoning_effort"] = ""
        agent["enable_thinking"] = False
    if opts.get("temperature") is not None:
        agent["temperature"] = opts["temperature"]
    reset_context()
    update_context("bible", BIBLE)
    update_context("chapters", CHAPTERS)
    update_context("research", {NUMBER: BRIEF})
    update_context("chronology", chronology())
    llm_client.reset_stats()


def run_target(label, url, model, cases, opts, runs=1, verbose=False):
    configure(url, model, opts)
    original = prompts.REVIEWER
    if opts.get("system"):
        prompts.REVIEWER = reviewer_prompts.resolve(opts["system"])
    results, started = [], time.time()
    try:
        for case_id, planted, draft in cases:
            for _ in range(runs):
                try:
                    _, issues = reviewer.review_chapter(NUMBER, TITLE, draft)
                    outcome = score_run(planted, issues)
                except llm_client.EndpointUnavailable:
                    raise
                except Exception:
                    issues, outcome = [], "unreadable"
                results.append({"id": case_id, "planted": planted,
                                "outcome": outcome,
                                "types": {i["type"] for i in issues}})
                if verbose:
                    shown = ", ".join(sorted(results[-1]["types"])) or "-"
                    print(f"  [{label}] {case_id:10} planted="
                          f"{planted or 'none':24} -> {outcome:11} ({shown})")
    finally:
        prompts.REVIEWER = original
    stats = llm_client.STATS.get("reviewer", {})
    tokens = stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0)
    summary = summarize(results, time.time() - started, tokens)
    summary["problems"] = problem_lines(results)
    return summary


COLUMNS = [("unreadable", "unreadable", "{}"),
           ("recall_any", "found any", "{:.2f}"),
           ("recall_typed", "right type", "{:.2f}"),
           ("false_alarm_rate", "FALSE ALARM", "{:.2f}"),
           ("false_alarms", "(count)", "{}"),
           ("sec_per_chapter", "s/chapter", "{:.1f}"),
           ("tokens", "tokens", "{:,}")]


def format_table(results):
    width = max(len(label) for label in list(results) + ["target"]) + 2
    head = "target".ljust(width) + "".join(h.rjust(13) for _, h, _ in COLUMNS)
    lines = [head, "-" * len(head)]
    for label, r in results.items():
        lines.append(label.ljust(width) + "".join(
            fmt.format(r[key]).rjust(13) for key, _, fmt in COLUMNS))
    lines += ["", "planted errors found (right type / runs, any issue):"]
    for label, r in results.items():
        lines.append(f"  {label}")
        for kind in PLANTED_TYPES:
            if kind in r["by_type"]:
                lines.append(f"      {kind:24} {r['by_type'][kind]}")
    lines += ["", "where it went wrong:"]
    for label, r in results.items():
        lines.append(f"  {label}")
        lines += [f"      {p}" for p in r["problems"]] or ["      nothing"]
    return "\n".join(lines)


def _label_values(items, cast=str):
    out = {}
    for item in items or []:
        label, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"expected LABEL=VALUE, got '{item}'")
        out[label] = cast(value)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score reviewers on chapters with planted errors.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", nargs=3, action="append", required=True,
                        metavar=("LABEL", "URL", "MODEL"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--no-thinking", action="append", default=[],
                        metavar="LABEL")
    parser.add_argument("--temperature", action="append", default=[],
                        metavar="LABEL=T")
    parser.add_argument("--max-tokens", action="append", default=[],
                        metavar="LABEL=N", help="cap each reply at N tokens")
    parser.add_argument("--system", action="append", default=[],
                        metavar="LABEL=NAME",
                        help="one of " + ", ".join(reviewer_prompts.VARIANTS)
                             + ", or @file.txt")
    parser.add_argument("--checks", metavar="C,C",
                        help="review checks to run (default all): "
                             + ", ".join(reviewer.ALL_CHECKS))
    parser.add_argument("--cases", metavar="ID,ID",
                        help="only these case ids, e.g. clean-a,dead")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json", metavar="FILE")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every review as it happens")
    args = parser.parse_args(argv)

    llm_client.load_config(args.config)
    temps = _label_values(args.temperature, float)
    systems = _label_values(args.system)
    caps = _label_values(args.max_tokens, int)
    known = {label for label, _, _ in args.target}
    for name, mapping in (("--temperature", temps), ("--system", systems),
                          ("--max-tokens", caps)):
        for label in mapping:
            if label not in known:
                raise SystemExit(f"{name} names unknown target '{label}'")
    for spec in systems.values():
        try:
            reviewer_prompts.resolve(spec)
        except (KeyError, OSError) as e:
            raise SystemExit(f"--system: {e}")
    checks = args.checks.split(",") if args.checks else None
    if checks and not set(checks) <= set(reviewer.ALL_CHECKS):
        raise SystemExit(f"--checks: choose from {', '.join(reviewer.ALL_CHECKS)}")
    cases = CASES
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - {c[0] for c in CASES}
        if unknown:
            raise SystemExit(f"unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in CASES if c[0] in wanted]

    results = {}
    for label, url, model in args.target:
        print(f"== {label}: {model} @ {url} ({len(cases)} chapters x "
              f"{args.runs})")
        opts = {"no_thinking": label in args.no_thinking,
                "temperature": temps.get(label), "system": systems.get(label),
                "max_tokens": caps.get(label),
                "json_mode": args.json_mode, "checks": checks}
        try:
            results[label] = run_target(label, url, model, cases, opts,
                                        args.runs, args.verbose)
        except Exception as e:
            print(f"   could not run: {e}")
    if not results:
        sys.exit("No target produced results.")
    print("\n" + format_table(results))
    print("\nfound any = a planted error produced at least one issue; right "
          "type = it was labelled as planted. FALSE ALARM = a clean chapter "
          "was sent back for revision (wasted round, risk of new errors).")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
