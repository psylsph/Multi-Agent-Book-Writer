"""An in-process fake LLM server for end-to-end tests.

install() replaces requests.get/post so the whole pipeline (main.main) runs
against scripted, deterministic replies with no network. The fake recognises
each agent by a phrase in its prompt, counts calls per kind, can drop the
connection after N chat calls, and can stream its replies (SSE).
"""

import json

import requests

CHAPTER_WORDS = 130

SEED = """# The Marsh Light

## Premise
A surveyor finds a hidden letter in a flooded chapel.

## Characters
- **Aria** - protagonist. A careful surveyor.
- **Tom** - supporting. The ferryman.

## Outline
- Chapter 1: The Letter - Aria finds the hidden letter.
- Chapter 2: The Ferry - Tom takes her across the marsh.
"""


def _letters(i):
    """0 -> a, 25 -> z, 26 -> aa ... (letters only: the repetition lint's
    tokenizer ignores digits, so numbered fake words would all look alike)."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


def words(count, prefix):
    """`count` distinct alphabetic fake words that start with `prefix`."""
    prefix = "".join(chr(97 + int(c)) if c.isdigit() else c for c in prefix)
    return " ".join(f"{prefix}{_letters(i)}" for i in range(count))


def marker(chapter):
    """The first word of the fake text written for `chapter`."""
    return words(1, f"c{chapter}w")


class FakeResponse:
    def __init__(self, status=200, data=None, lines=None, text=""):
        self.status_code = status
        self._data, self._lines = data, lines
        self.text = text or (json.dumps(data) if data is not None else "")
        self.encoding = None
        self.closed = False

    def json(self):
        return self._data

    def iter_lines(self, decode_unicode=False):
        yield from self._lines or []

    def close(self):
        self.closed = True


class FakeLLM:
    """Scripted server. `kinds` counts chat calls by recognised agent."""

    def __init__(self, models=("fake",), fail_after=None, stream=False,
                 usage=True, refuse_chapter=None, reviewer_issues=None,
                 reject_json_mode=False):
        self.models = list(models)
        self.fail_after = fail_after        # drop the connection after N calls
        self.stream = stream
        self.usage = usage
        self.refuse_chapter = refuse_chapter  # writer refuses this chapter
        self.reviewer_issues = reviewer_issues  # issues the reviewer reports
        self.reject_json_mode = reject_json_mode  # 400 on response_format
        self.kinds = {}
        self.calls = []                      # [{"kind", "payload"}]
        self.chat_calls = 0

    # ------------------------------------------------------------- dispatch
    def kind(self, prompt):
        for phrase, kind in (
                ("Decide how many chapters", "suggest"),
                ("You are a story architect", "architect"),
                ("planning the chapter outline", "planner"),
                ("lore keeper", "researcher"),
                ("Analyse this chapter", "extractor"),
                ("Review this chapter draft", "reviewer"),
                ("You are revising", "reviser"),
                ("professional fiction editor", "polisher"),
                ("Write Chapter", "writer")):
            if phrase in prompt:
                return kind
        return "other"

    def reply(self, kind, prompt):
        if kind == "suggest":
            return json.dumps({"chapters": 2, "reason": "short"})
        if kind == "architect":
            return json.dumps({
                "title": "The Marsh Light", "genre": "gothic", "tone": "dry",
                "premise": "p", "world": "A marsh", "constraints": [],
                "notes": "", "characters": [
                    {"name": "Aria", "role": "protagonist",
                     "description": "surveyor"}]})
        if kind == "planner":
            return json.dumps([{"title": "One", "summary": "a"},
                               {"title": "Two", "summary": "b"}])
        if kind == "researcher":
            return "Brief: Aria finds the letter."
        if kind == "extractor":
            return json.dumps({"summary": "It happened.", "time": "dusk",
                               "location": "marsh", "present": ["Aria"],
                               "events": []})
        if kind == "reviewer":
            issues = self.reviewer_issues or []
            return json.dumps({"verdict": "revise" if issues else "pass",
                               "issues": issues})
        if kind == "reviser":
            return words(CHAPTER_WORDS, "revised")
        if kind == "polisher":
            start = prompt.index("## Chapter")
            return prompt[start:prompt.index("\n\nReturn ONLY")]
        if kind == "writer":
            number = int(prompt.split("Write Chapter ")[1].split()[0])
            if number == self.refuse_chapter:
                return "I can't write that."
            return words(CHAPTER_WORDS, f"c{number}w")
        return "OK"

    # ----------------------------------------------------------------- HTTP
    def _outage(self):
        return self.fail_after is not None and self.chat_calls > self.fail_after

    def get(self, url, **kwargs):
        if self._outage():
            raise requests.exceptions.ConnectionError("fake outage")
        if url.endswith("/v1/models"):
            return FakeResponse(data={"data": [{"id": m} for m in self.models]})
        return FakeResponse(404, text="not found")

    def post(self, url, json=None, **kwargs):
        payload = json
        self.chat_calls += 1
        if self._outage():
            raise requests.exceptions.ConnectionError("fake outage")
        if self.reject_json_mode and "response_format" in payload:
            return FakeResponse(400, text="unsupported field: response_format")
        prompt = payload["messages"][-1]["content"]
        kind = self.kind(prompt)
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        self.calls.append({"kind": kind, "payload": payload})
        text = self.reply(kind, prompt)
        usage = ({"prompt_tokens": max(len(prompt) // 4, 1),
                  "completion_tokens": max(len(text) // 4, 1)}
                 if self.usage else None)
        if payload.get("stream"):
            pieces = [text[i:i + 40] for i in range(0, len(text), 40)] or [""]
            lines = [_sse({"choices": [{"delta": {"content": p}}]})
                     for p in pieces]
            lines.append(_sse({"choices": [{"delta": {},
                                            "finish_reason": "stop"}],
                               **({"usage": usage} if usage else {})}))
            lines.append("data: [DONE]")
            return FakeResponse(lines=lines)
        data = {"choices": [{"message": {"content": text},
                             "finish_reason": "stop"}]}
        if usage:
            data["usage"] = usage
        return FakeResponse(data=data)

    def install(self, monkeypatch):
        monkeypatch.setattr(requests, "get", self.get)
        monkeypatch.setattr(requests, "post", self.post)
        return self


def _sse(obj):
    return "data: " + json.dumps(obj)
