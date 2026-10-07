"""System-prompt variants for the reviewer (prompts.REVIEWER), for
tools/reviewer_eval.py (`--system LABEL=NAME` or `--system LABEL=@file`)."""

from shared import prompts

_JSON = (" You reply with a single valid JSON value and nothing else: no "
         "markdown fences, no commentary before or after it.")

VARIANTS = {
    # what the pipeline ships with: every report must quote the sentence AND
    # name the fact it breaks, and silence is a normal answer
    "default": prompts.REVIEWER,

    # the prompt used before the evidence rule (for comparison)
    "classic": prompts.REVIEWER_CLASSIC,

    # the opposite bias: read as if the author slipped (aimed at misses)
    "sceptical": (
        "You are a sceptical continuity editor who has seen authors slip "
        "many times. Check the chapter sentence by sentence against every "
        "recorded fact (who is dead, who has met, what each person knows, "
        "injuries, the day and place, the author's constraints and the "
        "outline) and report each contradiction you find, quoting the "
        "sentence. You judge only from the material given, never from "
        "style preferences." + _JSON),
}


def resolve(spec):
    """A system prompt from a variant name or '@path/to/file.txt'."""
    from pathlib import Path
    if spec.startswith("@"):
        return Path(spec[1:]).expanduser().read_text(encoding="utf-8").strip()
    if spec not in VARIANTS:
        raise KeyError(f"unknown system prompt '{spec}' "
                       f"(choose from {', '.join(VARIANTS)} or @file)")
    return VARIANTS[spec]
