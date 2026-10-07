"""System-prompt variants for the extractor, for tools/extractor_eval.py.

Prompts change model behaviour as much as a different model does, and cheaply.
Each variant replaces prompts.EXTRACTOR for one evaluation target
(`--system LABEL=NAME`, or `--system LABEL=@path/to/file.txt` to try your own).
Whichever wins on your model can be copied into shared/prompts.py.
"""

from shared import prompts

_JSON = (" You reply with a single valid JSON value and nothing else: no "
         "markdown fences, no commentary before or after it.")

VARIANTS = {
    # what the pipeline ships with
    "default": prompts.EXTRACTOR,

    # a bare instruction: does a short prompt help a small model?
    "minimal": ("You extract facts from fiction into JSON. Record only what "
                "the text explicitly states." + _JSON),

    # explains why precision matters, gives a procedure, and states the base
    # rate so that "no events" is an acceptable answer
    "strict": (
        "You are the continuity analyst for a novel. Your records are the "
        "book's only memory: a wrong entry corrupts every later chapter and "
        "nobody will check it, so you are conservative.\n"
        "- Before you record an event, find the sentence in the text that "
        "shows it. If you cannot point to one, leave the event out.\n"
        "- Most chapters hold few events and some hold none. An empty events "
        "list is a normal, correct answer.\n"
        "- Cooperation, conversation, kindness and shared danger are not "
        "relationship changes. A relationship change needs a clear turning "
        "point: new intimacy, a break, a betrayal, a declaration.\n"
        "- A first meeting needs the text to show the two have not met "
        "before.\n"
        "- Record a death only when the text states or unmistakably shows "
        "that someone has died (died, drowned, found dead, 'he's gone', a "
        "body covered). A near-death, an injury, a threat, a dream, and a "
        "death long ago in someone's past are NOT deaths. But never miss a "
        "real death: it is the most important thing you record.\n"
        "- A person is present only if they are in the scene, speaking or "
        "acting. People who are merely mentioned, remembered, expected or "
        "dead are not present." + _JSON),
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
