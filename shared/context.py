"""Shared context: the single source of truth passed between agents.

Agents mutate this dict in place via update_context(). reset_context()
clears and refills the SAME dict (never rebinds it), so modules that did
`from shared.context import context` always see current state.
"""

DEFAULTS = {
    "seed": "",        # raw seed prompt text (characters, world, outline, ...)
    "title": "",       # book title (set by the architect from the bible)
    "bible": {},       # structured story bible built by the architect agent
    "chapters": [],    # [{"number": int, "title": str, "summary": str}]
    "research": {},    # chapter number -> lore brief
    "drafts": {},      # chapter number -> draft text
    "summaries": {},   # chapter number -> rolling plot summary
    "final": [],       # edited chapters in order
}

context = dict(DEFAULTS)


def reset_context():
    """Reset the context to its initial state (in place)."""
    context.clear()
    for key, value in DEFAULTS.items():
        context[key] = value.copy() if isinstance(value, (dict, list)) else value


def update_context(key, value):
    """Update a specific key in the context."""
    context[key] = value


def get_context(key=None):
    """Get a specific key or the entire context."""
    if key:
        return context.get(key)
    return context
