#!/usr/bin/env python3
"""Score models on the verdict second check (web_search.double_check).

After the researcher decides a search result SUPPORTS or CONTRADICTS a claim,
the pipeline asks the model one independent question: does that quote, by
itself, directly support/contradict the claim? A wrong "yes" lets a misread
fact reach the writer; a wrong "no" merely drops a fact. This is a short
yes/no task, so a small model on a CPU can be a different-family second
opinion. Measure before trusting one.

  uv run python tools/verdict_eval.py --runs 3 \\
      --target main  http://127.0.0.1:8080 my-big-model \\
      --target small http://127.0.0.1:9090 my-small-model --no-thinking small \\
      --system small=strict

Options (repeatable, LABEL or LABEL=VALUE) are the same as the extractor tool:
--temperature, --system (tools/verdict_prompts.py variants or @file),
--no-thinking. It runs agents.researcher.confirms(), the pipeline's own code, as the
`verifier` agent (agents.verifier.* settings apply).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import researcher  # noqa: E402
from shared import llm_client, prompts  # noqa: E402
from shared.context import reset_context  # noqa: E402
from tools import verdict_prompts  # noqa: E402
from tools.verdict_cases import CASES, KINDS  # noqa: E402


def classify(expected, answer):
    """'ok', 'false_agree' (a wrong yes: the costly error), 'false_reject' (a
    wrong no: drops a good fact) or 'unreadable' (no boolean answer)."""
    if answer is None:
        return "unreadable"
    if answer is expected:
        return "ok"
    return "false_agree" if answer else "false_reject"


def summarize(outcomes, seconds, tokens):
    """Rates from [(kind, expected, outcome)] triples."""
    total = len(outcomes)
    should_no = [o for _, exp, o in outcomes if not exp]
    should_yes = [o for _, exp, o in outcomes if exp]
    by_kind = {}
    for kind, _, outcome in outcomes:
        n, ok = by_kind.get(kind, (0, 0))
        by_kind[kind] = (n + 1, ok + (outcome == "ok"))
    return {
        "accuracy": sum(o == "ok" for _, _, o in outcomes) / total
        if total else 0.0,
        "false_agree_rate": (should_no.count("false_agree") / len(should_no))
        if should_no else 0.0,
        "false_reject_rate": (should_yes.count("false_reject")
                              / len(should_yes)) if should_yes else 0.0,
        "false_agrees": should_no.count("false_agree"),
        "unreadable": sum(o == "unreadable" for _, _, o in outcomes),
        "by_kind": {k: f"{ok}/{n}" for k, (n, ok) in by_kind.items()},
        "sec_per_check": seconds / total if total else 0.0,
        "tokens": tokens,
    }


def configure(url, model, opts):
    cfg = llm_client.get_config()
    cfg["llm"].update(base_url=url, model=model, retries=0, endpoint_wait=0,
                      json_mode=opts.get("json_mode", False))
    agent = cfg.setdefault("agents", {}).setdefault("verifier", {})
    for key in ("model", "reasoning_effort", "enable_thinking", "temperature"):
        agent.pop(key, None)
    if opts.get("no_thinking"):
        agent["reasoning_effort"] = ""
        agent["enable_thinking"] = False
    if opts.get("temperature") is not None:
        agent["temperature"] = opts["temperature"]
    reset_context()
    llm_client.reset_stats()


def run_target(label, url, model, cases, opts, runs=1, verbose=False):
    configure(url, model, opts)
    original = prompts.VERIFIER
    if opts.get("system"):
        prompts.VERIFIER = verdict_prompts.resolve(opts["system"])
    outcomes, started, misses = [], time.time(), {}
    try:
        for kind, verdict, claim, quote, expected in cases:
            for _ in range(runs):
                answer = researcher.confirms({
                    "verdict": verdict, "claim": claim, "evidence": quote,
                    "url": "https://example.org/page"})
                outcome = classify(expected, answer)
                outcomes.append((kind, expected, outcome))
                if outcome != "ok":
                    key = (kind, verdict, claim, quote, outcome)
                    misses[key] = misses.get(key, 0) + 1
                if verbose and outcome != "ok":
                    print(f"  [{label}] {outcome.upper():12} {kind:10} "
                          f"({verdict}) {claim[:48]!r} / {quote[:40]!r}")
    finally:
        prompts.VERIFIER = original
    stats = llm_client.STATS.get("verifier", {})
    tokens = stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0)
    summary = summarize(outcomes, time.time() - started, tokens)
    summary["missed_pairs"] = [
        f"{count}x {outcome.upper()} {kind} ({verdict}): {claim[:46]!r} / "
        f"{quote[:38]!r}" for (kind, verdict, claim, quote, outcome), count
        in sorted(misses.items(), key=lambda kv: -kv[1])]
    return summary


COLUMNS = [("accuracy", "accuracy", "{:.2f}"),
           ("false_agree_rate", "FALSE YES", "{:.2f}"),
           ("false_agrees", "(count)", "{}"),
           ("false_reject_rate", "false no", "{:.2f}"),
           ("unreadable", "unreadable", "{}"),
           ("sec_per_check", "s/check", "{:.1f}"),
           ("tokens", "tokens", "{:,}")]


def format_table(results):
    width = max(len(label) for label in list(results) + ["target"]) + 2
    head = "target".ljust(width) + "".join(h.rjust(12) for _, h, _ in COLUMNS)
    lines = [head, "-" * len(head)]
    for label, r in results.items():
        lines.append(label.ljust(width) + "".join(
            fmt.format(r[key]).rjust(12) for key, _, fmt in COLUMNS))
    lines += ["", "accuracy by kind (right answers / asked):"]
    for label, r in results.items():
        lines.append("  " + label.ljust(width - 2) + "  ".join(
            f"{k} {r['by_kind'].get(k, '-')}" for k in KINDS))
    lines += ["", "most-missed pairs (up to 6 per target):"]
    for label, r in results.items():
        lines.append(f"  {label}")
        lines += [f"      {m}" for m in r["missed_pairs"][:6]] or [
            "      none"]
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
        description="Score models on the verdict second check.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", nargs=3, action="append", required=True,
                        metavar=("LABEL", "URL", "MODEL"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--no-thinking", action="append", default=[],
                        metavar="LABEL")
    parser.add_argument("--temperature", action="append", default=[],
                        metavar="LABEL=T")
    parser.add_argument("--system", action="append", default=[],
                        metavar="LABEL=NAME",
                        help="one of " + ", ".join(verdict_prompts.VARIANTS)
                             + ", or @file.txt")
    parser.add_argument("--kinds", metavar="K,K",
                        help="only these kinds: " + ", ".join(KINDS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json", metavar="FILE")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every wrong answer")
    args = parser.parse_args(argv)

    llm_client.load_config(args.config)
    temps = _label_values(args.temperature, float)
    systems = _label_values(args.system)
    known = {label for label, _, _ in args.target}
    for name, mapping in (("--temperature", temps), ("--system", systems)):
        for label in mapping:
            if label not in known:
                raise SystemExit(f"{name} names unknown target '{label}'")
    for spec in systems.values():
        try:
            verdict_prompts.resolve(spec)
        except (KeyError, OSError) as e:
            raise SystemExit(f"--system: {e}")
    cases = CASES
    if args.kinds:
        wanted = set(args.kinds.split(","))
        cases = [c for c in CASES if c[0] in wanted]
    if not cases:
        sys.exit("No cases selected.")

    results = {}
    for label, url, model in args.target:
        print(f"== {label}: {model} @ {url} ({len(cases)} pairs x "
              f"{args.runs})")
        opts = {"no_thinking": label in args.no_thinking,
                "temperature": temps.get(label), "system": systems.get(label),
                "json_mode": args.json_mode}
        try:
            results[label] = run_target(label, url, model, cases, opts,
                                        args.runs, args.verbose)
        except Exception as e:
            print(f"   could not run: {e}")
    if not results:
        sys.exit("No target produced results.")
    print("\n" + format_table(results))
    print("\nFALSE YES = a wrong 'yes' on a pairing that does not hold: the "
          "costly error (a misread fact reaches the writer). false no = a "
          "good fact dropped.")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
