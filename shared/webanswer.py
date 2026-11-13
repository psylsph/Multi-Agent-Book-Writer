"""Ask the browser instead of the terminal: the web GUI's question channel.

The pipeline has exactly two interactive prompts -- the seed review
(agents/seed_review.py, via the module's `_ask(question, default)` seam) and the
SearXNG Docker offer (shared/web_search.py, via `_ask`). Both read stdin, which
a web GUI has not got. When the GUI starts the pipeline it sets
BOOKWRITER_ASK_DIR, and install_if_enabled() swaps both prompts for a file
handshake:

    <dir>/question.json   {id, kind, prompt, default, asked_at}
                          + optional details (text to read before answering),
                          choices ([[key, label]]: a menu, shown as buttons),
                          fields ([{question, why, default, note?}]: a form;
                          its answer is a JSON list, one string per field)
    <dir>/answer.json     {id, answer, at}

The child writes a question and polls for the matching answer (removing both
files when it gets one). The parent -- web/app.py -- serves the question to
the page and writes the answer. That is the whole protocol: no sockets, no
shared memory, so a pipeline in a container can be answered by a browser on
the host, and either side can be restarted without losing the run (the
review's progress is also in state/seed.json's `pending` list).

If nobody answers within the timeout (default 900s, BOOKWRITER_ASK_TIMEOUT),
ask() raises EOFError, which is how a closed stdin already behaves: the seed
review treats it as "done, keep the seed as it is" instead of looping.
"""

import json
import os
import time
import uuid
from pathlib import Path

from shared.output import atomic_write_text

ENV_DIR = "BOOKWRITER_ASK_DIR"
ENV_TIMEOUT = "BOOKWRITER_ASK_TIMEOUT"
DEFAULT_TIMEOUT = 900.0     # a review can sit unanswered for a while
QUESTION = "question.json"
ANSWER = "answer.json"


def question_path(directory):
    return Path(directory) / QUESTION


def answer_path(directory):
    return Path(directory) / ANSWER


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def _read_json(path):
    """The file's JSON, or None when it is missing or unreadable (a half
    written file must not crash the pipeline)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def new_id():
    return uuid.uuid4().hex


def write_question(directory, prompt, default="", kind="seed_review", qid=None,
                   details="", choices=(), fields=()):
    """Publish a question and return it (a dict)."""
    question = {
        "id": qid or new_id(),
        "kind": kind,
        "prompt": prompt or "",
        "default": default or "",
        "asked_at": time.time(),
    }
    if details:
        question["details"] = str(details)
    if choices:
        question["choices"] = [[str(k), str(label)] for k, label in choices]
    if fields:
        question["fields"] = [dict(f) for f in fields]
    _write_json(question_path(directory), question)
    return question


def read_question(directory):
    """The unanswered question, or None."""
    return _read_json(question_path(directory))


def write_answer(directory, qid, answer):
    """Answer the question `qid`. Returns the record written."""
    record = {"id": qid, "answer": str(answer), "at": time.time()}
    _write_json(answer_path(directory), record)
    return record


def read_answer(directory, qid):
    """Consume the answer to `qid`, or None if it has not arrived.

    An answer with a different id belongs to an earlier question (the page
    was left open across a rerun): it is discarded, not handed to this one.
    """
    record = _read_json(answer_path(directory))
    if record is None:
        return None
    if record.get("id") != qid:
        try:
            answer_path(directory).unlink()
        except OSError:
            pass
        return None
    try:
        answer_path(directory).unlink()
    except OSError:
        pass
    question = read_question(directory)
    if question and question.get("id") == qid:
        try:
            question_path(directory).unlink()
        except OSError:
            pass
    return str(record.get("answer", ""))


def ask(prompt, default="", directory=None, timeout=None, poll=0.25,
        kind="seed_review", **extra):
    """Ask one question and block until it is answered.

    Raises EOFError on timeout, like a closed stdin: callers already handle
    that as "use the default and stop asking".
    """
    directory = Path(directory or os.environ.get(ENV_DIR) or "")
    if not str(directory):
        raise ValueError(f"{ENV_DIR} is not set")
    if timeout is None:
        timeout = float(os.environ.get(ENV_TIMEOUT) or DEFAULT_TIMEOUT)
    question = write_question(directory, prompt, default, kind, **extra)
    deadline = time.time() + timeout
    while time.time() < deadline:
        answer = read_answer(directory, question["id"])
        if answer is not None:
            return answer
        time.sleep(poll)
    try:
        question_path(directory).unlink()   # leave no stale question on the page
    except OSError:
        pass
    raise EOFError(f"no answer to {question['id']} within {timeout:.0f}s")


def form_replies(answer, count):
    """A form's answer (a JSON list of strings) as exactly `count` strings.
    Anything else - a plain string from an old page, broken JSON - fills
    the first field and leaves the rest to their defaults."""
    try:
        replies = json.loads(answer)
    except ValueError:
        replies = [answer]
    if not isinstance(replies, list):
        replies = [replies]
    replies = ["" if r is None else str(r) for r in replies][:count]
    return replies + [""] * (count - len(replies))


def _yes(text):
    return str(text).strip().lower() in ("y", "yes")


def install(directory=None, timeout=None):
    """Swap the terminal prompts for the file handshake. Returns the dir."""
    directory = Path(directory or os.environ.get(ENV_DIR) or "")
    if not str(directory):
        raise ValueError(f"{ENV_DIR} is not set")
    directory.mkdir(parents=True, exist_ok=True)
    if timeout is None:
        timeout = float(os.environ.get(ENV_TIMEOUT) or DEFAULT_TIMEOUT)

    # imported here: this module must stay importable without the pipeline
    from agents import seed_review
    from shared import web_search

    def seed_ask(prompt, default="", details="", choices=()):
        try:
            answer = ask(prompt, default=default, directory=directory,
                         timeout=timeout, kind="seed_review",
                         details=details, choices=choices)
        except EOFError:                   # what a closed stdin does: take
            print()                        # the default and stop asking
            seed_review._INPUT["closed"] = True
            return default
        return answer or default           # an empty answer means Enter

    def seed_form(fields):                 # seed_review._ask_form(fields)
        try:
            answer = ask("Answer what you can; an empty field takes its "
                         "suggestion.", directory=directory, timeout=timeout,
                         kind="seed_questions", fields=fields)
        except EOFError:
            return None
        return form_replies(answer, len(fields))

    def web_ask(prompt):                   # web_search._ask(question) -> bool
        try:
            return _yes(ask(prompt, directory=directory, timeout=timeout,
                            kind="web_search"))
        except EOFError:
            return False

    seed_review._ask = seed_ask
    seed_review._ask_form = seed_form
    seed_review._INPUT["closed"] = False   # a browser is there to answer
    # without a tty, mode() downgrades seed_review: ask to warn and the whole
    # review loop disappears -- say a terminal is present and answer via file
    seed_review.interactive = lambda: True
    web_search._ask = web_ask
    return directory


def install_if_enabled():
    """Install the browser prompts when BOOKWRITER_ASK_DIR is set.

    Called at the top of main.main(). Returns the directory, or None when the
    pipeline is running in a terminal as usual.
    """
    if not os.environ.get(ENV_DIR):
        return None
    try:
        return install()
    except (OSError, ValueError) as e:
        print(f"[WEB] Could not open the browser question channel ({e}); "
              "using the terminal prompts.")
        return None
