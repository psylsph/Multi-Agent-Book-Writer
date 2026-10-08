# AGENTS.md

Guidance for coding agents working in this repository. Human-facing docs live in
`README.md` (usage, every option), `QUICKSTART.md`, `CONTRIBUTING.md` and
`ROADMAP.md` (planned work, measurements, known limitations). Read the relevant
section there before changing user-visible behaviour.

## What this is

A CLI that turns a seed prompt (premise, characters, world, optional outline) into
a drafted, reviewed and polished book, using any OpenAI-compatible chat endpoint
(llama.cpp, LM Studio, vLLM, Ollama, OpenAI, OpenRouter). Plain Python 3.11+,
sequential pipeline, two runtime dependencies (`requests`, `PyYAML`).

## Commands

```sh
uv sync                    # install (creates .venv)
uv run pytest -q           # full suite: offline, ~1s, no LLM/network/Docker
uv run ruff check .        # lint (bug-catching rules only; CI runs it)
uv run main.py --help      # the pipeline CLI
uv run make_epub.py --help # finished chapters -> EPUB (no LLM)
```

CI (`.github/workflows/tests.yml`) runs `uv sync --locked`, ruff, then pytest on
Python 3.11, 3.12 and 3.13. Keep `uv.lock` in sync: add dependencies with
`uv add` and prefer the standard library.

## Layout and pipeline

```
main.py            CLI + orchestration: STAGES (the pipeline steps), run_pipeline,
                   run_interleaved, resume checks
agents/            one module per pipeline stage
  seed_review.py   step 0: critique the seed, expand it with the author's answers
                   (a loop), settle the size; saves the MASTER seed
  architect.py     step 1: seed -> story bible (JSON); seed characters/constraints taken verbatim
  planner.py       step 2: outline; a seed's own outline is parsed deterministically and wins
  researcher.py    step 3: per-chapter lore brief (+ optional SearXNG fact-check)
  writer.py        step 4: drafts + per-chapter summary/story-state extraction and checks
  reviewer.py      LLM review: continuity / outline / constraints -> JSON verdict
  editor.py        step 5: lint -> review -> bounded revise loop -> polish -> save book
shared/
  context.py       the shared in-memory state dict (see invariants)
  llm_client.py    config loading + the ONLY HTTP path to the LLM (retries, waits, stats)
  llm_utils.py     JSON extraction, output cleanup, deterministic seed parsing
  prompts.py       system prompts for every agent; with_bible() appends the bible
  story_state.py   chronology merge/render, name canonicalisation (pure)
  consistency.py   deterministic lint: banned words, quotas, names, word count, repetition
  extraction_checks.py  grounding/death checks on extracted story state (pure)
  web_search.py    SearXNG client, query filtering, verdict validation, Docker helper
  resume.py        the run's state store (output/state/): save_state, load_state
  output.py        interim artifacts, durable chapters/, archiving, atomic writes
  config_schema.py known config keys; warns about typos
  runlog.py, epub.py
tools/             offline-from-the-pipeline evaluation harnesses (reviewer, extractor,
                   verdict) run against real models; not part of a book run
tests/             pytest; tests/fake_llm.py is an in-process fake OpenAI server
seeds/             SEED_SCHEMA.md (format) and example_seed.md
```

Flow: `seed_review -> architect -> planner -> researcher -> writer -> editor`,
declared as the `STAGES` tuple in `main.py`. Each `Stage` has a run function
(returns an exit code to stop, or None), a `wanted` predicate (agent switched
off, review-as-you-go branch) and a `done` check for resume. Add or reorder steps
there, not with new branches in `run_pipeline`.
With `book.review_as_you_go: true` (`run_interleaved` in `main.py`) each chapter is
written, edited and has its story state re-extracted before the next is drafted.

## Invariants - do not break these

- **All LLM calls go through `shared/llm_client.py`** (`generate`,
  `generate_with_wait`, `generate_prose`). Pass the right `agent=` name: it selects
  per-agent model, temperature, max_tokens, reasoning effort and stats bucket.
  Use `generate_prose` for long text (it stitches continuations on
  `finish_reason=length`), `json_mode=True` only for replies that are a JSON
  *object*.
- **`AbortRun` must propagate.** Agents catch broad `Exception` to degrade
  gracefully, but always re-raise `AbortRun` first (`except AbortRun: raise`).
  Its subclasses are `EndpointUnavailable` (server down) and `StateWriteError`
  (state can't be saved). Swallowing one would silently ship degraded output or
  lose resumability; letting it escape lets the user fix the cause and rerun.
- **Shared context is mutated in place.** Modules do `from shared.context import
  context`; `reset_context()` clears and refills the same dict. Never rebind it.
  Chapter-keyed dicts (`drafts`, `research`, `summaries`, `chronology`) use `int`
  keys; JSON round-trips turn them into strings, so `resume.py` converts back.
- **Chapter text is stored without headings.** `drafts` and `final` map chapter
  number -> body (`final` holds the edited chapters; its keys are the chapters the
  editor has finished). Headings are added only when writing files, via
  `shared/output.py` (`chapter_heading`, `render_chapter`, `strip_heading`).
  Never parse model output for a heading: strip it and render ours.
- **Resume reads `output/state/`, not `interim/`.** Anything a later stage or a
  rerun needs goes through `shared/resume.py:save_state(key, value)` (one JSON
  file per key in `STATE_KEYS`; chapter-keyed dicts get their int keys back on
  load). It is always written and raises `StateWriteError` on failure.
  `interim/` (`save_interim`) is the optional, best-effort human-readable view:
  never read it back. A new piece of state means a new `STATE_KEYS` entry, a
  `load_state()` field, `main._hydrate_resume`, and a test in
  `tests/test_resume.py`. Every stage must stay idempotent: skip work whose
  result is already in context.
- **Never lose generated text.** Writes use `atomic_write_text`; a fresh run
  archives the previous one (`archive_previous_run`) instead of deleting it;
  revisions/polishes that shrink a chapter or add findings are rejected and the
  previous version kept.
- **The master seed.** When the seed review expands the seed, the expansion is
  the seed for everything after it and for any restart (`state/seed.json`:
  `original`, `current`, `pending`). `main.check_resume` accepts the original
  or the master as "the same run". `lost_material()` must keep rejecting an
  expansion that drops a character, constraint or outline chapter.
- **Verbatim over LLM transcription.** Seed characters, constraints, author notes
  and outlines are parsed deterministically (`llm_utils.py`, `planner.py`) and
  override what the model returns. Don't route them through the model.
- **Deterministic code checks the model.** Arithmetic, quote verification
  (`quote_in_text`), name grounding and lint are done in code; prefer adding a
  deterministic check over trusting another prompt.

## Config

- `config.example.yaml` (tracked) documents every option; `config.yaml` is each
  user's local, untracked copy (`cp config.example.yaml config.yaml`). Never
  commit `config.yaml`. Adding or renaming an option means updating the example,
  `KNOWN_KEYS`/`AGENT_KEYS` in `shared/config_schema.py`, and the README's
  configuration section. `tests/test_config_files.py` enforces the sync (its
  checks on `config.yaml` run only when a local copy exists).
- Defaults are applied partly in `load_config()` and partly at call sites
  (`cfg["book"].get("words_per_chapter", 800)` etc.). If you change a default,
  grep for every occurrence.
- Only send optional request fields (temperature, max_tokens, thinking controls)
  when the user configured them; servers have their own defaults.

## Tests

- Tests must be offline and deterministic. Stub the LLM (monkeypatch
  `generate_with_wait` / `generate_prose` in the agent module, or use
  `tests/fake_llm.py` for end-to-end runs through `main.main`), HTTP, Docker,
  `input()` and prompts. Use `tmp_path` for files and point
  `output.directory` there - a test must never write to the real `output/`.
- `tests/conftest.py` blocks real `requests.get/post` (a test that forgets a stub
  fails fast instead of hanging). It does **not** reset the shared context,
  config or LLM stats: reset what your test touches (`reset_context()`,
  `load_config(path)`, `llm_client.reset_stats()`).
- New behaviour needs a test. Bug fixes need a regression test.
- `tools/*_eval.py` need a real model and are run by hand; their pure scoring code
  is unit-tested in `tests/test_*_eval.py`.

## Style

- Match the surrounding code: plain functions, module-level docstrings that
  explain the stage, comments that give the *why* (often with the measured
  failure that motivated the code). No classes or frameworks where a function
  will do.
- Line length 100 (ruff). Console output uses a `[STAGE]` prefix
  (`[WRITER]`, `[EDITOR]`, `[LLM]` ...) and should tell the user what to do next
  when something fails.
- User-visible changes update README/QUICKSTART; gaps and future work go in
  ROADMAP.md. Commits are focused; the git history is the changelog.

## Don'ts

- Don't commit anything under `output*/`, `.coverage`, or API keys
  (`llm.api_key` supports `env:VAR`).
- Don't add a dependency for something the standard library does.
- Don't make a stage fail hard on a bad model reply: fall back, log it, continue
  (except on `AbortRun`).
