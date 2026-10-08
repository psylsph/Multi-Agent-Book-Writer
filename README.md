# Multi-Agent Book Writer

A collaborative AI book writing system using multiple specialized agents. Feed it a **seed prompt** — your premise, characters, world, tone, and (optionally) a chapter outline — and the agent team drafts a complete, polished book from it.

<img src="/images/writer.gif" alt="writer demo" style="width:100%; height:auto;" />

## Project Overview

The writing team consists of five agents:

- **Architect**: reads your seed prompt and distills it into a *story bible* (title, genre, tone, characters, world, constraints)
- **Planner**: builds the chapter outline — **honoring your outline if you provided one**, only expanding it if you ask for more chapters
- **Researcher (Lore Keeper)**: writes a per-chapter brief: which characters are on page, setting details, plot beats, continuity notes
- **Writer**: drafts each chapter with continuity — it sees the story bible, the chapter brief, a rolling summary, and structured **story facts** (who is alive, who has met whom, relationship states, timeline), and records a structured story state after each chapter
- **Reviewer**: checks every draft against the story state for dead-character resurrection, relationship regression/leaps (characters acting like strangers after they've met or become lovers), knowledge errors, and timeline/setting contradictions
- **Editor**: runs bounded revision rounds until deterministic lint + reviewer findings are fixed, then a final polish pass with a lint guard

All agents share one context object; every LLM call goes through a single configured client with timeouts and retries.

## Tech Stack

| Component | Tool/Library |
|-----------|-------------|
| LLMs | Any OpenAI-compatible endpoint (LM Studio, llama.cpp server, vLLM, Ollama, OpenAI, OpenRouter, ...) |
| Agent Orchestration | Python functions in a sequential pipeline |
| Context Sharing | In-memory shared dict |
| HTTP Client | requests (with timeout + retry) |
| Configuration | YAML (actually loaded and applied) |

## Project Structure

```
multi-agent-book-writer/
├── main.py                 # CLI entry point & pipeline orchestrator
├── make_epub.py            # finished chapters -> one EPUB (no LLM)
├── agents/
│   ├── architect.py        # seed prompt -> story bible
│   ├── planner.py          # chapter outline (JSON + seed-outline aware)
│   ├── researcher.py       # per-chapter lore briefs
│   ├── writer.py           # continuity-aware drafts + story state
│   ├── reviewer.py         # continuity review vs story state (JSON verdict)
│   └── editor.py           # lint -> review -> revise loop, polish, save
├── shared/
│   ├── context.py          # shared state (in-place reset)
│   ├── llm_client.py      # single LLM client: config, timeout, retries
│   ├── llm_utils.py        # JSON extraction / output cleanup helpers
│   ├── story_state.py      # chronology: merge/render story facts
│   ├── consistency.py      # deterministic lint: bans, quotas, timeline
│   ├── web_search.py       # SearXNG search + optional Docker setup
│   ├── prompts.py          # every agent's system prompt (role + rules)
│   ├── config_schema.py    # known config options; warns about typos
│   ├── runlog.py           # copies console output to a log file
│   ├── epub.py             # the EPUB builder behind make_epub.py
│   └── output.py           # interim artifacts + bible formatting
├── seeds/
│   ├── SEED_SCHEMA.md      # seed format reference
│   └── example_seed.md     # example seed prompt (a fantasy mystery)
├── tests/                  # offline tests; fake_llm.py is an in-process fake server
├── ROADMAP.md              # what is planned and what is known to be missing
├── output/                 # generated books + interim progress artifacts
├── config.yaml             # your settings (copy of the example; not tracked)
├── config.example.yaml     # every option documented, with defaults
├── pyproject.toml          # dependencies (managed with uv)
├── uv.lock
└── README.md
```

## Installation

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible LLM server (LM Studio, llama.cpp server, vLLM,
  Ollama, OpenAI, OpenRouter, ...) reachable over HTTP

### Setup Steps

1. **Clone the project**
   ```bash
   git clone <this-repo>
   cd Multi-Agent-Book-Writer
   ```

2. **Point it at your LLM server**

   Copy the example config (`config.yaml` is your local file and is not
   tracked by git), then edit it:

   ```bash
   cp config.example.yaml config.yaml
   ```

   `config.example.yaml` documents every option. Set
   `llm.base_url` to your server (the OpenAI-compatible
   `/v1/chat/completions` path is appended automatically).
   Set `llm.model` to a model the server offers, and `llm.api_key` if the
   server requires one (`env:MY_VAR` reads it from an environment variable;
   `LLM_API_KEY` is the fallback).

   Examples:
   - LM Studio / llama.cpp server: `base_url: "http://localhost:1234"`
   - Ollama: `base_url: "http://localhost:11434"` (its `/v1` API is used)
   - OpenAI: `base_url: "https://api.openai.com"`, `model: "gpt-4o"`, plus an API key
   - OpenRouter: `base_url: "https://openrouter.ai"`, `model: "<vendor>/<model>"`, plus an API key

3. **Install Python dependencies**
   ```bash
   uv sync    # install uv first: https://docs.astral.sh/uv/
   ```

## Usage

### Write a book from a seed prompt

```bash
uv run main.py --seed seeds/example_seed.md
uv run main.py --prompt "A noir thriller set on Mars. Detective Rya Cole ... "
uv run main.py --seed my_story.md --chapters 7 --model llama3 --out my_book.md
```

Run the bundled example seed with no preparation:

```bash
uv run main.py            # uses seeds/example_seed.md
uv run main.py --demo 2   # same, but only the first 2 chapters
```

### Command-line reference

| Option | Meaning |
|---|---|
| `--seed FILE` | Read the seed prompt from a `.md`/`.txt` file. |
| `--prompt TEXT` | Use inline text as the seed prompt. |
| `--demo` | Use the bundled example seed (`seeds/example_seed.md`). Cannot be combined with `--seed`/`--prompt`. |
| `-c N`, `--chapters N` | Chapter count. Beats the seed's outline, the LLM's suggestion and the config. The legacy positional form `main.py N` does the same. |
| `--config FILE` | Config file to use (default `config.yaml`). |
| `--model NAME` | Override `llm.model` for this run. |
| `--out FILE` | Override the output filename (written under `output.directory`). |
| `--plan-only` | Stop after the seed review, story bible and outline. Rerun without it to write the book; it continues from that plan. |
| `--seed-review MODE` | Override `book.seed_review` for this run: `ask`, `warn` or `off`. See [Seed review](#seed-review). |
| `--no-resume` | Ignore saved interim files and start over from the seed. See [Resume](#resume). |
| `-h`, `--help` | Show the built-in help. |

With no seed option the bundled example is used — unless an interrupted run is
waiting in `output/interim/`, in which case its saved seed is reused.

There is no `--resume` flag: resuming is automatic whenever
`output/interim/bible.json` exists. Rerun the same command after a crash or
Ctrl-C and the run continues from the last saved chapter. Add `--no-resume`
to discard that state and start fresh. A different `--seed`/`--prompt`, or a
different `-c`, than the saved run is refused unless you add `--no-resume`.

### The seed prompt format

Freeform text works, but a structured seed gives the best results. See
[`seeds/SEED_SCHEMA.md`](seeds/SEED_SCHEMA.md) for the full schema (what is
parsed verbatim vs. LLM-extracted) and
[`seeds/example_seed.md`](seeds/example_seed.md) for a complete example:

```markdown
# My Book Title

## Premise
1-3 sentences: who wants what, and what stands in the way.

## Characters
- **Name** - role. Appearance, personality, motivation.

## World
Setting, rules (magic/tech), background the story depends on.

## Tone & Style
Point of view, pacing, atmosphere.

## Constraints
- Facts that must stay consistent (ages, dates, rules, names).

## Outline            # optional - the planner will honor this directly
- Chapter 1: Title - what happens
- Chapter 2: Title - what happens
```

If the seed has no outline, the planner generates one from the story bible.
The chapter count comes from `-c N`, otherwise the LLM reads the seed and
suggests one (honoring any length the seed states, e.g. "5 to 10 chapters"). If the seed's
outline has fewer chapters than you request, the planner expands it; if it has
more, the first N are used (with a notice).

### Chapter count priority

1. `--chapters N` / positional `N` if given
2. the number of chapters in the seed's own outline
3. a total length stated in the seed ("approximately 50,000 words"), divided
   by the chapter length
4. the seed review's recommendation (see [Seed review](#seed-review))
5. the LLM's suggestion after reading the seed (`book.auto_chapters: true`,
   clamped to `book.min_chapters`–`book.max_chapters`), used only when the
   seed review is off or fails
6. `book.num_chapters` from config.yaml (default 5) — also the fallback if
   `auto_chapters` is off or the suggestion call fails

With the seed review on (`ask`, the default) you see 1–3 checked against the
seed's material and can choose a different size before anything is written.

When resuming, the saved outline is reused so chapter numbers keep matching
the drafts on disk.

## Seed review

Before the story bible is built, the seed review checks that the seed holds
enough story for the length asked of it. A seed covering one weekend with a
handful of scenes can't fill 50,000 words without padding, however the chapters
are divided.

1. The model lists every scene the seed supplies, each with a short quote. Code
   checks each quote really is in the seed; scenes it can't find are shown but
   not counted. It also reports the story's timespan, its subplots, the length
   the material would naturally fill, and a verdict on the requested size:
   *too long* (it would need padding), *about right* or *too short* (it would
   be rushed).
2. Code, not the model, does the arithmetic: words per scene at the requested
   size (a scene usually runs 1,000–3,000 words) and whether there are more
   chapters than scenes.
3. On a terminal it asks up to `book.seed_questions` questions about gaps that
   would change the book: a missing subplot, the ending, point of view. Press
   Enter to accept the assumption shown, or type `skip` to accept all the rest.
   If you answered any, the size is checked again with your answers.
4. If the size doesn't fit, it asks which to use:

   ```
   [k] keep the requested 25 chapters x 2,000 words (about 50,000)  (Enter)
   [r] use the recommended 10 chapters x 2,000 words (about 20,000)
   [c] choose your own
   [s] stop here: nothing is written, so you can edit the seed and rerun
   ```

Your answers become part of the story bible, so every later stage sees them. The
review, scene list and answers are saved to `output/interim/seed_review.md`;
copy the answers into your seed if you want them for a fresh run. The chosen
size is saved in `output/interim/plan.json`, so a resumed run keeps it. The
planner now also reads the seed itself, not just the bible's summary, and is told
to give each chapter its own material rather than spreading one scene across
several.

`book.seed_review` sets the mode: `ask` (the default), `warn` (print the review
and carry on with the requested size, never prompting; also what `ask` does
without a terminal) or `off`. `--plan-only` stops after the outline so you can
read the plan cheaply, then rerun without it to write the book.

## Making an EPUB

Once the chapters are finished, `make_epub.py` turns them into one EPUB 3 book.
It needs no LLM and no extra packages, and sends nothing anywhere.

```bash
uv run python make_epub.py --author "Your Name"            # output/chapters -> output/<title>.epub
uv run python make_epub.py --author "Your Name" --cover cover.jpg
uv run python make_epub.py output/draft.md -o my-book.epub  # from one assembled file instead
```

It reads `<output.directory>/chapters/chapter_NN.md` (the edited chapters once a
run has finished) or a single book file such as `draft.md`, takes the title from
the book's own files (override with `--title`), and writes the EPUB next to the
`chapters/` directory unless you pass `-o`.

What it does to the book:

- A cover (a generated typographic one, or your own image with `--cover`), a title
  page, a contents page, and a navigable table of contents.
- Typography: curly quotes and apostrophes, real ellipses and em dashes, small caps
  on each chapter's opening line, indented paragraphs, justified text. Pass
  `--plain-quotes` to leave the text exactly as written.
- Chapter headings read "CHAPTER ONE" over the title. Planning notes the pipeline
  leaves in headings, such as `(Stuart POV, heat 3)`, are dropped from the
  displayed titles (`--keep-annotations` keeps them). `--numbers digits|none`
  changes the "Chapter One" label.
- Markdown emphasis (`*italic*`, `**bold**`) becomes real italics and bold, and a
  line of `***` or `---` becomes a scene-break ornament.
- `--language` (default `en-GB`) sets the language for hyphenation and reading apps.

The files validate cleanly with the W3C's EPUBCheck. Reading apps differ: the
generated cover is an SVG, which Kindle and some other apps handle poorly, so
supply a JPEG or PNG with `--cover` if the cover matters. To send a book to a Kindle,
use Amazon's Send to Kindle, which accepts EPUB. Review the whole book once before
publishing it; the pipeline's checks reduce errors but cannot remove them.

## Resume

If the app or the LLM crashes mid-run, re-running it picks up where it left
off rather than starting over. Each agent writes structured JSON snapshots
(bible.json, outline.json, summaries.json, chronology.json) plus per-chapter
text files; the next run loads them and skips already-completed work.

- Architect: skipped once bible.json exists
- Planner: skipped once outline.json exists
- Researcher/Writer/Editor: per-chapter skip when the corresponding draft or
  edited artifact is on disk

The resume is automatic when `output/interim/bible.json` exists. Use
`--no-resume` to force a clean restart: the previous run (interim files,
chapters and the assembled book) is **moved** to
`output/archive/<timestamp>/`, never deleted, and everything is rebuilt from
the seed. Delete old archives yourself when you no longer need them.

Safety rules for resuming:

- The interim files belong to one seed. With no `--seed`/`--prompt`/`--demo`
  argument, the seed saved with them is reused. Passing a *different* seed
  (or a different `-c` count) is an error — it would reuse the old book's
  chapters — unless you add `--no-resume`.
- A chapter whose draft survived but whose story state did not (crash, corrupt
  snapshot) has its state rebuilt from the draft; it is never redrafted.
- Files are written atomically, so a crash mid-write cannot leave a truncated
  snapshot.
- If any chapter cannot be finished the run ends with a non-zero exit code
  and lists the missing chapters; rerun to resume.

### Server outages mid-run

If the LLM server drops mid-run (timeout, connection refused, or HTTP 503
"Loading model" while it reloads), the pipeline does not skip the remaining
chapters: it waits up to `llm.endpoint_wait` seconds (default 300) for the
server to come back and retries. If the server never returns, the phase
aborts with a clear message and a non-zero exit code instead of silently
producing an empty book. Rerun the same command when the server is up —
resume picks up from the last saved chapter.

## How It Works

### Pipeline Flow

1. **Architect**: seed prompt → structured story bible (JSON, with a
   no-LLM fallback that keeps the raw seed)
2. **Planner**: seed outline (if any) or LLM-generated JSON outline, validated
   and normalized
3. **Researcher**: a lore brief per chapter, keyed by chapter number
4. **Writer**: drafts each chapter from bible + brief + story facts + rolling
   summary; after each chapter one call produces both the next-chapter summary
   and a structured **story state** entry (time, location, who's present,
   first meetings, relationship changes, deaths, injuries, secrets revealed)
5. **Reviewer + Editor**: per chapter, a deterministic lint (banned words,
   countable quotas, `wc -w` word count vs the configured minimum, name
   near-misses, dead characters acting alive, stranger-language between
   characters who've met) plus an LLM continuity review (relationship
   regression/leaps, knowledge and timeline errors) — then bounded revision
   rounds until the findings are fixed (extra rounds for length while the
   chapter keeps growing ≥15% per round), a final polish pass, and a lint
   guard that keeps whichever version is cleaner
6. **Save**: `output/draft.md` (created automatically) plus
   `output/story_bible.md` and `output/interim/lint_report.md` (includes
   per-chapter word counts with **SHORT** flags)

### Interim output

Every artifact is written to `output/interim/` the moment it is produced, so
you can follow a run as it happens (the previous run is archived to
`output/archive/` at the start of a fresh run):

```
output/interim/
├── story_bible.md        # as soon as the architect finishes
├── outline.md            # as soon as the planner finishes
├── lore_chapter_NN.md    # per-chapter briefs, one by one
├── draft_chapter_NN.md   # each chapter draft, right after writing
├── summaries.md          # rolling chapter summaries (rewritten per chapter)
├── story_state.md        # rolling chronology: meetings, deaths, relationships
├── search_chapter_NN.md  # web fact-check claims, results, verdicts (if enabled)
├── review_chapter_NN.md  # reviewer verdicts + issues per chapter
├── diff_chapter_NN.md    # what the editor changed: draft -> final, with its decisions
├── edited_chapter_NN.md  # each edited chapter
├── lint_report.md        # final deterministic lint across the book
├── run_stats.md/.json    # LLM calls, tokens, time per agent and phase
├── bible.json            # structured bible (resume)
├── outline.json          # structured plan (resume)
├── summaries.json        # rolling summaries (resume)
└── chronology.json       # rolling story state (resume)
```

Disable with `output.interim: false` in config.yaml (resume reads these
files, so runs can't resume with it off).

### Durable per-chapter files

Every chapter is also saved on its own the moment it exists, in a directory
that is never deleted (a fresh start archives it) and is written even with
`output.interim: false`:

```
output/chapters/
└── chapter_NN.md         # draft as soon as it's written; replaced by the
                          # edited version when polishing finishes
```

If the app crashes, every finished chapter is still here. A forced clean
restart moves them to `output/archive/` rather than deleting them.

### Agent Communication

All agents share a central `context` dict: `seed`, `title`, `bible`,
`chapters`, `research`, `drafts`, `summaries`, `final`.

## Configuration

`config.yaml` is loaded and applied; every option, with its default, is
documented in [`config.example.yaml`](config.example.yaml). Every key is
optional. Keys:

- **book**: `auto_chapters`/`min_chapters`/`max_chapters` (LLM-suggested chapter count and its clamp), `num_chapters` (fallback), `words_per_chapter` (target length), `word_count_tolerance` (enforced minimum as a fraction of the target — short chapters are lint findings and get sent back for substantive expansion, never padding), `extra_length_rounds` (additional revision rounds granted for length only, while each round still adds ≥15%), `revision_rounds` (review/revise passes per chapter; 0 disables revision), `summary_window` (how many recent chapter summaries the writer/reviewer see in full; older ones are cut to a sentence), `review_as_you_go` (see [Review as you go](#review-as-you-go)), `review_checks` and `repetition_lint` (see [Review quality](#review-quality)), `name_lint_ignore` (words the name-typo lint must never flag), `extraction_checks` (see [Review quality](#review-quality))
- **llm**: `base_url`, `api_key` (`""`, plain value, or `env:VAR`; `LLM_API_KEY` env var is the fallback), `model`, `timeout` (per request), `retries` (with backoff), `max_tokens` (optional per-request cap; replies cut off by a length limit are continued automatically), `endpoint_wait` (seconds to wait for a downed/reloading server before aborting a phase), `reasoning_effort` (`""`, `low`, `medium`, `xhigh` = the maximum; sent as `chat_template_kwargs.reasoning_effort` for Qwen3.8-style thinking models. When set, the researcher and extractor default to `low` and the reviewer to `medium`; the other stages use the value as set. Empty sends no thinking controls to anyone), `enable_thinking` (`true`/`false`; sent as `chat_template_kwargs.enable_thinking` when set; reportedly unsupported on Qwen3.8), `stream`, `json_mode` and `context_window` (see [Observability](#observability))
- **agents**: per-agent `model` (default `llm.model`; each must exist on the same server and is checked at startup), `reasoning_effort` / `enable_thinking` / `max_tokens` (override the `llm.*` value; an empty `reasoning_effort:` sends nothing for that agent). The researcher, extractor, verifier and reviewer, which only produce short structured replies, have a built-in output ceiling (6000 / 6000 / 2000 / 8000 tokens) unless you set `max_tokens`: a model that loops while writing JSON otherwise never stops and `temperature` for `architect`, `planner`, `researcher`, `verifier` (judging search results and the second check), `writer`, `extractor` (per-chapter JSON continuity extraction), `reviewer`, `editor` — only sent when set; otherwise the server/model default applies — and `enabled` for `researcher`/`reviewer`/`editor` (disable them to speed things up)
- **web_search**: `enabled`, `searxng_url`, `auto_start`, `docker_image`, `queries_per_chapter`, `results_per_query`, `snippet_chars`, `categories`, `timeout`, `double_check`, `banned_terms` (see [Web fact-checking](#web-fact-checking-optional))
- **output**: `directory`, `filename`, `overwrite` (`false` appends `-1`, `-2`, ... instead of clobbering), `interim` (progress artifacts under `<directory>/interim/`), `log` (copy console output to `<directory>/logs/run-<time>.log`)

Misspelled or unknown options are not silently ignored: at startup each one is reported with a suggestion, e.g. `[CONFIG] unknown key 'book.revision_round' is ignored (did you mean 'revision_rounds'?)`.

Command-line options (`--model`, `--out`, `--config`, ...) are listed in the
[command-line reference](#command-line-reference).

## Review as you go

By default the pipeline writes **all** chapters, then reviews and polishes them
all. A continuity slip in chapter 3 is therefore built on by chapters 4 to 10
before review sees it, and the review of chapter 3 fixes only chapter 3.

Set `book.review_as_you_go: true` to work one chapter at a time instead:

1. write chapter N
2. lint, review, revise and polish it
3. re-extract its summary and story facts from the **final** text
4. only then write chapter N+1 (its prompt uses the edited chapter's facts,
   summary and ending)

Later chapters build on corrected earlier ones, so a drift is caught right
after it happens. The cost is one extra LLM call per chapter and no complete
draft until the end. It needs the editor enabled (otherwise it is ignored with
a note), and resumes like any run: rerun the same command. The partial book is
rewritten after every chapter, so it is always on disk. The two modes use the
same files, so you can resume a run in either mode.

## Review quality

Each chapter goes through the lint, the reviewer and the editor. Beyond the
continuity checks (deaths, who has met, timeline) there are:

- **Reviewer checks** (`book.review_checks`, default all three):
  `continuity`; `outline` (does the chapter deliver the beats its outline entry
  and lore brief require? only entirely missing beats count); `constraints`
  (your seed's rules a word-counter can't check: POV, tense, content and style).
  Findings feed the same bounded revise loop; for an outline gap the reviser
  adds the missing beat as a short passage.
- **Judging a revision**: a revision (or polish) is kept only if it does not
  make the chapter worse by *severity*, not by a raw count of findings. A
  continuity error weighs 5, a missed outline beat, broken constraint or short
  chapter 3, a name typo 2, a banned word, quota or repeated phrase 1. So a
  revision that fixes a dead character walking about but trips two word quotas
  is accepted. A revision that grows the chapter by more than 10% (a length
  expansion, an added beat) is reviewed again even if the draft passed, because
  new material can contradict the story.
- **Unreadable reviews**: if the reviewer's reply can't be read, the chapter is
  marked **not reviewed** rather than passed: `review_chapter_NN.md`, the
  edit diff, `lint_report.md` and the end-of-run summary all say so. If a
  revision's re-review can't be read, its earlier issues are assumed unfixed.
- **Repetition lint** (`book.repetition_lint`): models reuse imagery and
  phrasing across chapters ("a shiver ran down her spine as..."). A phrase of six
  or more words already used twice in earlier chapters is reported and the
  reviser rewords it; `lint_report.md` also lists phrases repeated across the
  whole book. Character names and function words don't count.
- **Edit diffs**: `output/interim/diff_chapter_NN.md` shows, per chapter, the
  sentence-level diff from draft to final, the lint findings it started with and
  every decision the editor took (revision accepted or rejected and why, polish
  accepted or rejected), so you can audit what was changed and catch over-editing.
- **Extraction checks** (`book.extraction_checks`, on by default): the story
  facts extracted after each chapter are cross-checked against the chapter text.
  Names the text never mentions are dropped (invented characters, and a meeting
  or relationship change that loses one of its two people); a character who
  died in an *earlier* chapter is not killed again (that would move the death);
  and deaths, the costliest error either way, are verified with **one** focused
  question per chapter. The candidates are every death the model claimed plus
  every bible character named near death language (including euphemisms such as
  "he's gone" or a hand on a pulse), and the model says which really die: a
  claimed death that isn't confirmed is dropped, and a confirmed one the
  extractor missed is added. A chapter with no death language and no claimed
  death costs no extra call. The extractor also tolerates loosely typed output
  (`"who": "Tom and Liz"` instead of a list).
- **Character names**: a nickname resolves to the bible character, either one
  declared in the seed (`**Elizabeth (Liz)**`, `aka Liz`, `known as Liz`) or a
  common short form (Liz/Beth for Elizabeth, Bob for Robert, Stu for Stuart...),
  so a death or meeting is tracked for one person however the prose names them.
  A nickname that fits two characters is left alone.

## Observability

- **Run statistics**: at the end of every run (also a failed or interrupted
  one) a table of LLM calls, prompt and output tokens, minutes and output
  tokens/s per agent, plus time per phase, is printed and saved to
  `output/interim/run_stats.md` and `run_stats.json`. If the server doesn't
  report token usage the report says so rather than inventing numbers.
- **Log file**: everything printed, including tracebacks, is also written to
  `output/logs/run-<time>.log` (`output.log: false` turns it off).
- **Streaming** (`llm.stream: true`): replies are streamed, a progress line
  (tokens so far, tokens/s) is printed every ~20 s, and `llm.timeout` becomes
  the time allowed *without any data*, so a stalled server is noticed in
  seconds. Off by default.
- **Context-window guard** (`llm.context_window: <tokens>`): a warning, once per
  agent, when a prompt reaches 90% of the window (characters per token is
  calibrated from the usage the server reports). A context-overflow error from
  the server also gets a hint pointing at `book.summary_window`.
- **JSON mode** (`llm.json_mode: true`): the calls that return a JSON object
  (architect, extractor, reviewer, chapter-count suggestion, verdict check) ask
  the server to enforce it. Array replies and prose are never constrained. If
  the server rejects it, the run carries on without it.

## Prompts

Every agent has a system prompt (role and rules) in
[`shared/prompts.py`](shared/prompts.py): architect, planner, researcher,
writer, extractor, reviewer, reviser and polisher. Agents that need the story
bible get it in the *system* message rather than the user message. The bible
is identical for every call, so servers with prompt caching (llama.cpp, vLLM)
reuse it instead of re-reading it for each chapter. Each chapter's prompt also
carries the last ~180 words of the previous chapter so voice and transitions
carry over. To change how a stage behaves (voice, POV, tone rules), edit its
prompt there.

The writer also guards against unusable replies: an empty, far-too-short, or
refusal-style response is retried once, and if it fails again the writing
phase stops (the reply is never saved as a chapter); rerun to resume.

## Web fact-checking (optional)

The researcher can check **real-world** details (places, professions,
procedures, history) against the web. It is off by default and uses a
[SearXNG](https://github.com/searxng/searxng) metasearch instance, so there is
no API key and no third-party account.

```yaml
web_search:
  enabled: true
  searxng_url: "http://localhost:8888"
  auto_start: ask        # ask | yes | no
```

**Setup.** Nothing to install if you have Docker. With `enabled: true`, if
SearXNG isn't reachable at a local URL the app offers to download and start
it before the run:

```
[SEARCH] SearXNG is not available: could not reach SearXNG at http://localhost:8888/search (...)
Download the 'searxng/searxng' Docker image and start SearXNG on 127.0.0.1:8888? [y/N]
```

It runs a container named `book-writer-searxng`, bound to `127.0.0.1` only,
with a generated settings file (JSON output on, limiter off, random secret key)
in `~/.config/multi-agent-book-writer/searxng/`. An existing stopped
container is restarted instead. `auto_start: yes` skips the question; `no`
never touches Docker. Without a terminal (cron, CI) it never prompts. To stop
it later: `docker stop book-writer-searxng` (`docker rm` to remove it). If
you'd rather run your own SearXNG, point `searxng_url` at it and enable
`json` under `search.formats` in its `settings.yml`.

**What gets sent.** For each chapter the model proposes a few concrete
real-world *claims* (how a profession, procedure or place really works), each
with a short generic query (the model is told never to put story details in
it). Every query is allowed by default. A query is dropped, and logged, if it
mentions a term in `web_search.banned_terms` (whole word, any case; list
character names, an invented place such as "Willow Rooms", anything you don't
want sent to a search engine), or if it essentially repeats a query already run
for an earlier chapter. Nothing is detected automatically: a name you don't
list can be searched. Everything searched, and every verdict, is saved to `output/interim/search_chapter_NN.md` so you can audit it.
Queries still leave your machine through SearXNG's upstream search engines.

**How claims are verified.** After searching, the model judges each claim from
its search results alone: *supported*, *contradicted* or *unclear*. A verdict
only counts if it cites one of that claim's own results **and** its quote
really appears in that result's text (checked in code, not by the model);
otherwise it is downgraded to *unclear*. Only supported and contradicted facts
reach the lore brief, as quoted data (the model is told to ignore any
instructions inside it), with the correct fact for anything contradicted. Your
invented world always stands; only real-world errors are corrected. If search
or verification fails the brief is written without web facts; the run never
stops for a search problem. Resumed runs skip chapters that already have a
brief, and skip the setup offer when every brief exists.

## Evaluating a model for a stage

`tools/extractor_eval.py` scores models on the **extractor** (the step that
records who is on the page and what happened; a wrong event poisons every later
continuity check). It runs the pipeline's real extraction code against five
invented chapters whose correct answers are known, including traps (a character
who is only mentioned, a nickname, a chapter where nothing happens), and reports
valid-JSON rate, recall/precision, invented events, **false deaths** and speed:

```bash
uv run python tools/extractor_eval.py \
    --target main http://127.0.0.1:8080 my-main-model \
    --target small http://127.0.0.1:9090 my-small-model --no-thinking small \
    --runs 3                    # models are non-deterministic: repeat
```

Each target can carry its own `--temperature LABEL=T`, `--system LABEL=NAME`
(a prompt variant from `tools/extractor_prompts.py`, or `@file.txt`),
`--no-thinking LABEL` and `--no-checks LABEL`, so the *same* model can be
compared under different temperatures, system prompts and with the extraction
checks on or off. `--cases 4,8` runs a subset (ten cases cover backstory deaths,
near-deaths, an implicit death, a funeral after a recorded death and an attack
nobody dies in).

`--json-mode` tries grammar-constrained JSON; `--chapters-dir output/chapters`
instead compares targets on your own chapters (agreement with the first target,
no ground truth).

### The verdict second check

`tools/verdict_eval.py` does the same for `web_search.double_check` (34
claim/quote pairs with known answers: supported, contradicted, same-topic but
unrelated, sign backwards, claim stronger than the quote). It reports accuracy
and, separately, the **false yes** rate, the costly error (a misread fact reaches
the writer), against the false no rate (a good fact dropped). A short yes/no
task like this is where a small model on a CPU can be a useful second opinion
from another model family:

```bash
REPO=LiquidAI/LFM2.5-2.6B-GGUF ALIAS=lfm tools/run-small-model.sh -d   # CPU, port 9090
uv run python tools/verdict_eval.py --runs 3 \
    --target main http://127.0.0.1:8080 my-main-model \
    --target lfm  http://127.0.0.1:9090 lfm --no-thinking lfm --system lfm=strict
```

### The reviewer

`tools/reviewer_eval.py` scores reviewers on 14 versions of one chapter: four
clean (two deliberately tempting: a dream about the dead harbourmaster, and a
warm scene that *is* consistent with the recorded relationship) and ten with
one planted error each (a dead character acting alive, a relationship
regression and leap, a secret someone could not know, an injury that healed
overnight, a weekday that contradicts the timeline, a character in the wrong
country, a missing outline beat, a point-of-view slip, a tense slip). It
reports how many errors are found, how many are labelled with the right type,
how many clean chapters are wrongly sent back for revision (a wasted round and a
risk of new errors), and every miss and false alarm. `--system LABEL=classic`
compares the previous reviewer prompt; `--max-tokens LABEL=N` bounds a model that
loops. Nothing is written to your output directory.

```bash
uv run python tools/reviewer_eval.py --runs 2 -v \
    --target main http://127.0.0.1:8080 my-main-model \
    --target small http://127.0.0.1:9090 my-small-model --max-tokens small=8000
```

`tools/run-small-model.sh` starts any small GGUF model from Hugging Face as a
CPU-only llama.cpp server on port 9090 (loopback only, memory-guarded, resumable
checksummed download): `REPO=owner/model-GGUF ALIAS=name QUANT=Q6_K
tools/run-small-model.sh -d`, then `status` / `stop`. It needs `llama-server` on
`PATH` or `LLAMA_SERVER=/path/to/llama-server`.

## Development

```bash
uv run pytest
```

The tests cover the pure parsing helpers (JSON extraction, LLM output cleanup,
seed-outline extraction) — no model required.

## Troubleshooting

### Connection error to the LLM server
The pipeline preflights the endpoint at startup and exits with the available
models listed if the configured model is missing. Make sure the server is
running and `llm.base_url` in config.yaml is correct (a trailing `/v1` is
fine; it is not doubled).

### Slow generation
- Use `--chapters 2` for a quick test before a full run
- Set `agents.researcher.enabled: false` and/or `agents.editor.enabled: false`
- Use a smaller/faster model (`--model`)

### Out of memory
- Try a smaller model
- Reduce chapter count

## Roadmap

Planned work and known limitations are in [ROADMAP.md](ROADMAP.md).

## Requirements

Dependencies are declared in `pyproject.toml` (Python 3.11+):

- `requests`: HTTP client for the OpenAI-compatible chat API
- `PyYAML`: config loading

## License

This project is open source and available for educational and commercial use.

## Notes

- Works with any OpenAI-compatible endpoint (`llm.base_url` in config.yaml;
  default `http://localhost:11434`, i.e. a local Ollama server)
- Content quality depends on the selected model; a 7B model is fine for
  structure but a larger model gives noticeably better prose
- A full 5-chapter run is roughly 4N+2 LLM calls (~22) — expect several
  minutes to an hour depending on hardware
