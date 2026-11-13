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
│   ├── resume.py           # the run's state store (output/state/) + resume
│   ├── llm_client.py      # single LLM client: config, timeout, retries
│   ├── llm_utils.py        # JSON extraction / output cleanup helpers
│   ├── story_state.py      # chronology: merge/render story facts
│   ├── consistency.py      # deterministic lint: bans, quotas, timeline
│   ├── web_search.py       # SearXNG search + optional Docker setup
│   ├── prompts.py          # every agent's system prompt (role + rules)
│   ├── config_schema.py    # known config options; warns about typos
│   ├── runlog.py           # copies console output to a log file
│   ├── epub.py             # the EPUB builder behind make_epub.py
│   ├── book_html.py        # the styled HTML page behind the GUI's book link
│   └── output.py           # interim artifacts + bible formatting
├── seeds/
│   ├── SEED_SCHEMA.md      # seed format reference
│   └── example_seed.md     # example seed prompt (a fantasy mystery)
├── web/                    # optional web GUI (extra: fastapi + uvicorn)
│   ├── app.py              # HTTP API: one main.py subprocess per book
│   ├── static/index.html   # the page (no build step)
│   └── __init__.py
├── Dockerfile              # CLI + web GUI image
├── docker-compose.yml      # app (+ optional SearXNG sidecar), books on the host
├── tests/                  # offline tests; fake_llm.py is an in-process fake server
├── ROADMAP.md              # what is planned and what is known to be missing
├── output/                 # generated books, run state, interim progress artifacts
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

   Add `--extra web` (`uv sync --extra web`) if you want the optional
   [web GUI](#web-gui-optional). The CLI does not need it.

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
| `--seed-review MODE` | Override `book.seed_review` for this run: `ask`, `warn`, `auto` or `off`. See [Seed review](#seed-review). |
| `--no-resume` | Ignore the saved run and start over from the seed. See [Resume](#resume). |
| `-h`, `--help` | Show the built-in help. |

With no seed option the bundled example is used — unless an interrupted run is
waiting in `output/state/`, in which case its saved seed is reused.

There is no `--resume` flag: resuming is automatic whenever
`output/state/bible.json` exists. Rerun the same command after a crash or
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
3. On a terminal the review becomes a loop that grows the seed with you. Each
   round shows the critique and asks what next:

   ```
   [a] answer the questions and/or add your own notes; the seed is then expanded with them  (Enter)
   [d] done: the seed is ready; go on to the book's size
   [s] stop here: anything expanded so far is saved; rerun to continue
   ```

   With `a` it asks up to `book.seed_questions` questions about gaps that would
   change the book (a missing subplot, the ending, point of view). The round's
   questions are listed first; then type an answer to each, or press Enter to
   accept the suggestion shown (`back` returns to the previous question, `skip`
   accepts all the rest); then add notes of your own if you like. The model then **expands
   the seed** with your answers and the suggestions you accepted: it keeps
   every word you wrote and adds the new material where it belongs (a character
   under Characters, an event in the Outline). You see what changed and keep it
   (`k`) or undo it (`u`; your answers are then passed on as clarifications).
   The expanded seed is critiqued again, and so on until you choose `d`.
   Code checks every expansion: if it dropped a character, reworded a
   constraint, lost an outline chapter or got shorter, the model is told what
   it lost and tries once more; if that fails too, it is not used (the
   rejected text is saved as `interim/seed_round_NN_rejected.md`).
4. If the size doesn't fit, it asks which to use:

   ```
   [k] keep the requested 25 chapters x 2,000 words (about 50,000)  (Enter)
   [r] use the recommended 10 chapters x 2,000 words (about 20,000)
   [c] choose your own
   [s] stop here: nothing is written, so you can edit the seed and rerun
   ```

   A chapter length your seed states ("1,500-1,700 words per chapter", "2000
   words a chapter", "chapters of about 1.8k words", "(about 2,000 words)" on
   an outline line) is read by code and overrides `book.words_per_chapter`: a
   range is aimed at its middle (1,600), and the recommendation is kept inside
   it. The statement is then **taken out of the seed** (the sentence that
   states it; the rest of the line stays), and out of any expansion or answer
   that writes one in, so every later prompt sees one length: the number the
   run settled on. The log lists what was removed. Your seed file is not
   changed, and rerunning with it is still recognised as the same run. A
   total book length ("about 50,000 words") is not per-chapter and stays.

**The expanded seed is the master seed.** Each kept expansion is saved at once
to `output/state/seed.json` and, readable, to `output/seed.md`; every later stage
writes the book from it. A restart uses it too: rerunning the same command (with
your original `--seed` file, or none) continues from the expanded seed, and a
review you stopped or interrupted picks up where it left off. Each round's
version is also kept as `output/interim/seed_round_NN.md`. To use the expanded
seed for a brand-new run, pass `--seed output/seed.md --no-resume` (copy it out
first: a fresh run archives `output/`).

Answers that did not go into the seed (an undone expansion, or the review
without a terminal) become part of the story bible, so every later stage sees
them. The
review, scene list and answers are saved to `output/interim/seed_review.md`;
copy the answers into your seed if you want them for a fresh run. The chosen
size is saved in `output/state/plan.json`, so a resumed run keeps it. The
planner now also reads the seed itself, not just the bible's summary, and is told
to give each chapter its own material rather than spreading one scene across
several.

`book.seed_review` sets the mode: `ask` (the default), `warn` (print the review
and carry on with the requested size, never prompting; also what `ask` does
without a terminal), `auto` (accept every suggested answer, expand the seed with
it, and review the expanded seed again until the reviewer asks nothing more —
then take the recommended size if the request still does not fit; no terminal
needed, and it stops after `book.seed_auto_rounds` expansion rounds, default 5,
as a guard against a model that always asks; 0 reviews without expanding), or
`off`. `--plan-only` stops after the outline so you can
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

## Web GUI (optional)

`web/` is a small FastAPI page for starting a book, watching the chapters and
log arrive, downloading the result, and **answering the seed review's questions
in the browser**. It is an optional extra (`fastapi` + `uvicorn`); the CLI never
imports it.

```bash
uv sync --extra web
uv run --extra web uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

- **Each book is its own process.** The GUI starts `main.py` as a subprocess
  with a directory of its own (`output/web/<run>/`: `config.yaml`, `state/`,
  `chapters/`, and the web layer's `job/` with its seed, journal, log and
  questions). A crash or a stop cannot take the server down, the pipeline's
  global config/context/stats stay per-run, and the CLI remains the only
  implementation of a run.
- **Per-book settings.** Besides chapters, model and seed review, the
  *Settings* panel overrides a curated set of `config.yaml` options for the
  book you start: words per chapter, the length minimum and maximum, revision and
  length rounds, review as you go, auto seed rounds, point of view, spelling, reasoning
  effort and max tokens (all agents, and the reviewer's own), writer
  temperature, and the reviewer, researcher and web fact-check switches. Each
  field shows the value from `config.yaml`; leave it blank to keep it. The
  server checks every value (`SETTINGS` in `web/app.py`) and writes it into
  that run's own `config.yaml` only, so `config.yaml` itself and other books
  are untouched, and a resumed book keeps the settings it started with. The
  run's status line lists what was changed.
- **Persistent, so any device can pick the book up.** The run's state is its
  directory on disk, not the server's memory: `job/job.json` says whether it is
  running (pid + its command line), `job/exit.json` records how it ended, and
  the log is a file. Reload the page on a phone, or restart the server, and the
  run list, chapters and log are all there. Each book has a URL
  (`/?run=<id>`) you can send to yourself. A run whose server went away shows
  as **interrupted** and continues with *Resume book*; set
  `BOOKWRITER_AUTO_RESUME=1` to resume the newest one automatically at startup.
- **Built for a phone.** Under 780 px: one column, the book list becomes a
  native picker, 44 px+ tap targets, safe-area insets for the notch and home
  bar, 16 px fields so iOS does not zoom on focus, a bottom-sheet question
  dialog, long paths and JSON allowed to break inside the log instead of
  scrolling sideways, and a log pane capped at 2 000 lines so a long book
  cannot freeze Safari.
- **The seed review works in the browser.** The prompts the CLI would print in
  the terminal are published as files (`<run>/ask/question.json`) and answered
  from the page. A round's questions arrive together as one form (an empty
  field takes its suggestion, and there is a box for your own notes); menus
  are buttons; and keep-or-undo shows every line the expansion changed, not
  just the log's last lines. Answers are text areas, so a paragraph is easy to
  type on a phone. With
  `book.seed_review: ask` a run really does wait for you. An unanswered
  question falls back to its suggestion after `BOOKWRITER_ASK_TIMEOUT` seconds
  (default 900), the same way a closed terminal does.
- **Progress you can read at a glance.** Once two chapters are finished the
  status line estimates the time left, from the median time a chapter has
  taken (so a night spent stopped does not skew it); while every chapter is
  still being drafted it estimates the drafting only. Each chapter row also
  says what the editor had to give up on: a rejected revision or polish, a
  missing review, findings left (`state/chapter_notes.json`; the full story is
  in `interim/diff_chapter_NN.md`).
- **Stop saves the run.** It sends SIGINT — what Ctrl-C sends — so the partial
  book is saved and *Resume this book* continues from the last finished
  chapter. *Build EPUB* runs `make_epub.py` for the selected book.
- **Delete book** (tap twice) removes a finished or stopped book from the list.
  It is moved to `output/web/.trash/<run>`, not erased: move it back to
  restore it, or empty the folder to free the space. A book being written
  must be stopped first.
- **The seed review's work is in Files.** The kept expanded seed is `seed.md`;
  the review report (`interim/seed_review.md`) and each round's expansion,
  including one the code check rejected (`seed_round_NN_rejected.md`), are
  listed too, as is each chapter's web fact-check
  (`interim/search_chapter_NN.md`: the claims, queries, verdicts and results).
- **Read a book in the browser.** Once chapters exist, the Files box gains an
  *open book (live HTML)* link: the whole book as one self-contained page
  (`GET /api/runs/<id>/book`), re-rendered from the current chapters on every
  load — so it works mid-run, in light or dark. The type is set like a printed
  book: justified and hyphenated, drop caps, scene-break ornaments, and A−/A+
  buttons in the corner resize the text live (remembered per browser).
- **One book at a time, no accounts.** Bind it to `127.0.0.1`, or set
  `BOOKWRITER_TOKEN=<secret>` to require a shared token: open
  `http://<host>:8000/?token=<secret>` once per browser (it is kept in a
  cookie; scripts can send an `X-Bookwriter-Token` header instead). More than
  that needs real job isolation and auth, not this page.
- **Your API key stays out of the run folders.** A literal `llm.api_key` is
  removed from each run's `config.yaml` and handed to the run through its
  environment (`LLM_API_KEY`), and the page only downloads the files it lists
  (the book, seed, bible, EPUB and chapters), never a run's config or state.

## Docker

The image contains the CLI and the web GUI. `config.yaml`, `seeds/` and
`output/` stay on the host.

```bash
cp config.example.yaml config.yaml   # once, then edit llm.base_url / llm.model
docker compose up --build            # http://localhost:8000
```

**Port 8000 is published on every interface** by default, so a phone on the same
network or tailnet (Tailscale) can open `http://<host>:8000`. The GUI has no
accounts; on a network you don't trust, set a token, or keep it on this machine:

```bash
BOOKWRITER_TOKEN=some-long-secret docker compose up
# then open http://<host>:8000/?token=some-long-secret once per browser
BOOKWRITER_BIND=127.0.0.1 docker compose up   # this machine only
```

(or put the variables in `.env`). A reverse proxy with its own auth works too.

Two things differ from running on the host:

- **The LLM endpoint must be reachable from inside the container.** A server on
  the host (llama.cpp, LM Studio, vLLM, Ollama) is not on `localhost` there.
  Either set `llm.base_url: "http://host.docker.internal:8080"` in `config.yaml`,
  or leave your config alone and override it for every run:

  ```bash
  BOOKWRITER_LLM_BASE_URL=http://host.docker.internal:8080 docker compose up --build
  ```

  `host.docker.internal` is mapped to the host by `extra_hosts` in
  `docker-compose.yml`. To make it stick, put
  `BOOKWRITER_LLM_BASE_URL=http://host.docker.internal:8080` in a `.env` file
  next to `docker-compose.yml` (compose reads it, and `.env` is gitignored). A hosted API needs neither, and `llm.api_key:
  "env:LLM_API_KEY"` with `export LLM_API_KEY=...` before `up` works as usual.
- **SearXNG: reuse the one you already run, or use the bundled sidecar.** The
  app never mounts the Docker socket. Either:
  - **reuse an existing container.** `docker-compose.yml` attaches the app to
    the external `searxng-net` network and `.env` sets
    `BOOKWRITER_SEARXNG_URL=http://searxng:8080`. SearXNG must serve the JSON
    API, so its `settings.yml` needs `search.formats: [html, json]`; without it
    every search fails with HTTP 403 (the log says so). If that network does not
    exist on your machine, either `docker network create searxng-net` or delete
    the `networks:` block from `docker-compose.yml`.
  - **or run the bundled sidecar.** Set `web_search.auto_start: "no"`, keep
    `BOOKWRITER_SEARXNG_URL=http://searxng:8080` (the sidecar's service name on
    the project network) and start it with
    `docker compose --profile search up --build`.

The same image runs the CLI:

```bash
docker compose run --rm app python main.py --demo
```

Books land in `./output` (each web run in `./output/web/<run>/`) owned by you:
compose runs the container as `${UID}:${GID}`.

## Resume

If the app or the LLM crashes mid-run, re-running it picks up where it left
off rather than starting over. Everything a later stage needs is saved to
`output/state/` the moment it exists (`bible.json`, `plan.json`,
`outline.json`, `research.json`, `drafts.json`, `summaries.json`,
`chronology.json`, `final.json`); the next run loads it and skips
already-completed work.

- Architect: skipped once the bible is saved
- Planner: skipped once the outline is saved
- Researcher/Writer/Editor: per-chapter skip when that chapter's brief, draft
  or edited text is saved

`output/state/` is always written, whatever `output.interim` says, and a write
that fails (disk full, permissions) **stops the run** with a message rather
than carrying on with work a crash could no longer resume from. Runs saved by
older versions, which kept their state in `output/interim/`, still resume.

The resume is automatic when `output/state/bible.json` exists. Use
`--no-resume` to force a clean restart: the previous run (state, interim
files, chapters and the assembled book) is **moved** to
`output/archive/<timestamp>/`, never deleted, and everything is rebuilt from
the seed. Delete old archives yourself when you no longer need them.

Safety rules for resuming:

- A saved run belongs to one seed. With no `--seed`/`--prompt`/`--demo`
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
└── run_stats.md/.json    # LLM calls, tokens, time per agent and phase
```

Disable with `output.interim: false` in config.yaml. Resume does not depend
on these files: it reads `output/state/` (see [Resume](#resume)).

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
optional. A misspelled key is reported at startup, and so is a value of the
wrong type (`revision_rounds: two`, `stream: "maybe"`): it falls back to its
default with a `[CONFIG]` warning instead of crashing the run later. Keys:

- **book**: `auto_chapters`/`min_chapters`/`max_chapters` (LLM-suggested chapter count and its clamp), `num_chapters` (fallback), `words_per_chapter` (target length), `word_count_tolerance` (enforced minimum as a fraction of the target — short chapters are lint findings and get sent back for substantive expansion, never padding), `extra_length_rounds` (additional revision rounds granted for length only, while each round still adds ≥15%), `max_length_ratio` (chapters over this multiple of the target get one tightening pass; 0 = never; see [Review quality](#review-quality)), `revision_rounds` (review/revise passes per chapter; 0 disables revision), `summary_window` (how many recent chapter summaries the writer/reviewer see in full; older ones are cut to a sentence), `review_as_you_go` (see [Review as you go](#review-as-you-go)), `review_checks` and `repetition_lint` (see [Review quality](#review-quality)), `name_lint_ignore` (words the name-typo lint must never flag), `extraction_checks` (see [Review quality](#review-quality)), `style` (deterministic style lint: banned stock phrases, hedging, flat rhythm, POV/tense slips, British spelling; see [Review quality](#review-quality))
- **llm**: `base_url`, `api_key` (`""`, plain value, or `env:VAR`; `LLM_API_KEY` env var is the fallback), `model`, `timeout` (per request), `retries` (with backoff), `max_tokens` (optional per-request cap; replies cut off by a length limit are continued automatically), `endpoint_wait` (seconds to wait for a downed/reloading server before aborting a phase), `reasoning_effort` (`""`, `low`, `medium`, `xhigh` = the maximum; sent as `chat_template_kwargs.reasoning_effort` for Qwen3.8-style thinking models. When set, the researcher and extractor default to `low` and the reviewer to `medium`; the other stages use the value as set. Empty sends no thinking controls to anyone), `enable_thinking` (`true`/`false`; sent as `chat_template_kwargs.enable_thinking` when set; reportedly unsupported on Qwen3.8), `stream`, `stream_max_seconds`, `json_mode` and `context_window` (see [Observability](#observability)), `sampling` (extra sampler fields sent verbatim; llama.cpp/Ollama only, see [Observability](#observability))
- **agents**: per-agent `model` (default `llm.model`; each must exist on the same server and is checked at startup), `reasoning_effort` / `enable_thinking` / `max_tokens` / `sampling` (override the `llm.*` value; an empty `reasoning_effort:` sends nothing for that agent). The researcher, extractor, verifier and reviewer, which only produce short structured replies, have a built-in output ceiling (6000 / 6000 / 2000 / 8000 tokens) unless you set `max_tokens`: a model that loops while writing JSON otherwise never stops and `temperature` for `seed_reviewer` (the seed review), `architect`, `planner`, `researcher`, `verifier` (judging search results and the second check), `writer`, `extractor` (per-chapter JSON continuity extraction), `reviewer`, `editor` — only sent when set; otherwise the server/model default applies — and `enabled` for `researcher`/`reviewer`/`editor` (disable them to speed things up)
- **Endpoint variables**: `BOOKWRITER_LLM_BASE_URL` and `BOOKWRITER_SEARXNG_URL`, when set, replace `llm.base_url` and `web_search.searxng_url` for every run, CLI and web GUI alike (`load_config` prints a `[CONFIG]` line saying so). Where the servers are is a fact about the machine, not the book. Example: a SearXNG container on a Docker network publishes no port, so on the host run `BOOKWRITER_SEARXNG_URL=http://$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' searxng):8080 uv run main.py --seed ...`. The CLI also reads `.env` next to `main.py` (the file `docker compose` reads), so API keys such as `TINYFISH_API_KEY`, `PARALLEL_API_KEY` and `LLM_API_KEY` put there once work for both; variables already in the environment win, and only names, never values, are printed. An endpoint variable in `.env` whose host does not resolve on the host (`host.docker.internal`, a compose service name like `searxng`) is skipped for the CLI with a `[CONFIG]` line, and `config.yaml`'s address is used. An address that works on both sides, such as the SearXNG container's IP on its Docker network, serves the CLI and the container alike. `.env` is in `.dockerignore`, so keys are never baked into the image.
- **web_search**: `enabled`, `searxng_url`, `auto_start`, `docker_image`, `queries_per_chapter`, `results_per_query`, `snippet_chars`, `categories`, `timeout`, `double_check`, `banned_terms`, `block_character_names`, `min_interval`, `providers`, `parallel_url`, `parallel_api_key`, `tinyfish_url`, `tinyfish_api_key` (see [Web fact-checking](#web-fact-checking-optional))
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
  continuity error weighs 5, a missed outline beat or broken constraint 3, a
  name typo or point-of-view slip 2, a banned word, quota or repeated phrase 1.
  Length weighs nothing here (quality over quantity): a short chapter can
  still earn expansion rounds, but a tighter revision that fixes a real
  finding is kept even under the minimum. A shorter revision that fixes
  nothing is not, and a revision or polish that loses a quarter (a polish:
  15%) of the chapter *and* goes under the length the book asks for (a
  revision: the minimum; a polish: the target) is treated as truncated and
  thrown away. So a 3,189-word chapter may be tightened to 2,200, but a
  1,806-word chapter polished down to 438 is refused. A revision that fixes a
  dead character walking about but trips two word quotas is accepted. A
  revision that changes the chapter's length by more than 10% either way (a
  length expansion, an added beat, a tightening) is reviewed again even if the
  draft passed, because new material can contradict the story and cut material
  can be a missing outline beat.
- **Over-long chapters**: some models ignore the length they are asked for (a
  27B model asked for 1,400-1,700 words wrote 1,803-3,189). A chapter over
  `book.max_length_ratio` times the target (default 1.5; 0 = never) gets one
  tightening pass inside the normal revision rounds: cut repetition and
  over-described beats, keep every scene. The tightened version is rejected
  if it drops a character the chapter had, an outline beat (the re-review) or
  anything else the lint or reviewer catches; then the long chapter is kept.
  The lint report marks such chapters **LONG**. Setting a low `max_tokens`
  does not shorten chapters: the writer continues a reply cut at the limit. A revision that fixes nothing (same
  severity, and no growth while expanding for length) is kept but ends the
  rounds: another try on the same findings rarely does better.
- **Checking the reviewer**: an issue whose description quotes the chapter
  (15+ characters in quotes) is dropped when none of its quotes is really in
  the chapter, so invented evidence can't send a sound chapter back. Issues
  without quotes (a missing outline beat) are kept.
- **Checking the polish**: the polish pass is not reviewed, so a polish that
  uses a name mid-sentence that the chapter never contained (a renamed
  character, an invented place) is rejected in code and the pre-polish text
  kept. The polish is a copy-edit of the chapter body, and it is shown the
  banned words, quotas and `book.style` rules the guard will measure it by.
- **Unreadable reviews**: if the reviewer's reply can't be read, the chapter is
  marked **not reviewed** rather than passed: `review_chapter_NN.md`, the
  edit diff, `lint_report.md` and the end-of-run summary all say so. If a
  revision's re-review can't be read, its earlier issues are assumed unfixed.
- **Repetition lint** (`book.repetition_lint`): models reuse imagery and
  phrasing across chapters ("a shiver ran down her spine as..."). A phrase of six
  or more words already used twice in earlier chapters is reported and the
  reviser rewords it; `lint_report.md` also lists phrases repeated across the
  whole book. Character names and function words don't count.
- **Scene checks** (`shared/scene_checks.py`, always on): two outline facts
  a reviewer let through, checked in code. *Who is in the chapter*: when the
  outline lists a chapter's people ("Couples: Angela & Pete visiting Stuart &
  Kirsty", "Characters: ..."), a story-bible character outside that list who
  acts in the narration three or more times (speaks, nods, reaches for the
  wine) is a `cast` finding, and the chapter goes back to the reviser.
  Mentions, memories and talk about someone are fine. *Where the book is*:
  when the seed or the bible's setting names a UK county ("rural Hampshire"),
  narration that names another ("a heavy, Somerset rain") is a `place`
  finding; dialogue is left alone. The writer is told both before drafting.
  An outline with no cast field, or one that names nobody ("All three
  couples"), and a seed with no county, are not checked.
- **Style lint** (`book.style`): the same deterministic loop, pointed at prose
  rather than continuity. Every rule is opt-in, so an empty `book.style` block
  (the default) adds nothing. `banned_phrases` catches the stock phrases a model
  returns to (measured on 26B Gemma-family finetunes: "coil of", "fullness",
  "reclamation", "devastatingly"); `max_hedges_per_1000` catches vague
  narration ("seemed to", "a sense of"); `min_sentence_cv` catches metronomic
  sentence rhythm; `pov` and `tense` catch a chapter that slips out of the
  book's person or tense; both judge the narration only (quoted dialogue and
  *italic thoughts* are left out, since characters speak in the first person
  and present tense). Books are narrated in the **third person** by default,
  and that is checked on every chapter even without `pov`: a chapter whose
  narration turns to "I" is sent back, and the writer is told the person
  before drafting. For a first-person book, set `pov: first` or say so in the
  seed ("first-person narrator", "POV: first person").
  `spelling: british` flags American spellings and words ("color",
  "neighbor", "realize", "toward", "sidewalk", "gotten") and tells the writer
  and polish pass to use British ones. It uses -ise, not Oxford -ize, and
  leaves alone words with a different British meaning (pants, fall, check).
  Phrases match whole words ("sigh" does not count "sight"). Findings are
  advisory (severity 1-2), the reviser is
  told exactly what to change, and the polish guard still keeps whichever
  version measures better. `tools/style_eval.py` measures the same numbers on a
  model before you pick thresholds.
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
  seconds. Off by default. `llm.stream_max_seconds` caps one whole streamed
  reply (a model that loops while trickling tokens otherwise never times out);
  a reply over the cap is retried like any failed call.
- **Context-window guard** (`llm.context_window: <tokens>`): a warning, once per
  agent, when a prompt reaches 90% of the window (characters per token is
  calibrated from the usage the server reports). A context-overflow error from
  the server also gets a hint pointing at `book.summary_window`.
- **JSON mode** (`llm.json_mode: true`): the calls that return a JSON object
  (architect, extractor, reviewer, chapter-count suggestion, verdict check) ask
  the server to enforce it. Array replies and prose are never constrained. If
  the server rejects it, the run carries on without it.
- **Sampler controls** (`llm.sampling`): extra sampler fields (`min_p`, `top_k`,
  `top_p`, `repeat_penalty`, `repeat_last_n`, the `dry_*` family, `xtc_*`,
  `typ_p`, `dynatemp_*`) sent **verbatim** on prose requests. They are not
  sent to the JSON-only agents (`extractor`, `verifier`, `reviewer`) or to any
  JSON-object call, since repetition penalties fight JSON's repeated keys.
  The `editor` (revisions, the polish) and the `seed_reviewer` (expanding the
  seed) get the block **without** its repetition penalties (`repeat_*`,
  `presence/frequency_penalty`, `dry_*`, `xtc_*`): they reproduce text that is
  already in their context, which is what those penalties punish. With DRY on,
  the polish fused and garbled words the draft never had ("againsther",
  "unremarkableRE"). As a backstop, the editor splits a fused word back when
  the text it was given has the two words side by side, and rejects a
  revision or polish that brings in mid-word capitals.
  `agents.<name>.sampling` replaces the block for one agent (`{}` sends
  nothing, or give the reviewer a block of its own). Only the keys you
  write are sent, and only llama.cpp / Ollama family servers understand them;
  OpenAI rejects unknown request fields, so leave the block out for a hosted
  API. `temperature` stays per agent (`agents.<name>.temperature`). Measured on
  a local fiction model, temp 1.0 was purple and cliché-heavy; temp 0.85-0.9,
  `min_p: 0.1` and DRY cut the stock phrasing without touching the weights.
  `tools/style_eval.py` can sweep these settings against a model.

## Prompts

Every agent has a system prompt (role and rules) in
[`shared/prompts.py`](shared/prompts.py): architect, planner, researcher,
writer, extractor, reviewer, reviser and polisher. Agents that need the story
bible get it in the *system* message rather than the user message. The bible
is identical for every call, so servers with prompt caching (llama.cpp, vLLM)
reuse it instead of re-reading it for each chapter. The same goes for the
writer's `book.style` rules (the writer is told what each chapter will be
linted for) and the researcher's copy of the whole outline (so a brief can say
what to set up for later chapters). Each chapter's prompt also carries the last
~180 words of the previous chapter so voice and transitions carry over, and the
next chapter's outline entry as a boundary: the writer is told to stop where
its own chapter ends and to hit every beat of its lore brief. Material the
model works on (a chapter, a seed) is wrapped in `<chapter>`/`<brief>` tags
rather than triple quotes, which broke when the text itself contained them. To
change how a stage behaves (voice, POV, tone rules), edit its prompt there.

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

**Parallel as well (or instead).** `providers: [searxng, parallel]` also
asks [Parallel](https://parallel.ai)'s search, through its MCP server
(`https://mcp.parallel.ai/mcp`, spoken directly over HTTP: no extra
dependency). It is sent the claim being checked as well as the query, and
returns the passages of each page that address it, which is what the
verifier needs to quote. It works without a key at a free rate limit;
`parallel_api_key` (or `PARALLEL_API_KEY`) raises it. The claim passes the
same screen as a query (character names, `banned_terms`), or the query is
sent in its place. `tinyfish` adds [TinyFish](https://tinyfish.ai) Search
the same way (the claim goes as its `purpose`); it needs a key, read from
`TINYFISH_API_KEY` (or `tinyfish_api_key`). Results from every provider are
merged, one per URL, and a provider failing leaves the others.

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
for an earlier chapter. The full name of every story character (two or more
words, such as "Aria Vance") and any such nickname is blocked too
(`block_character_names`, on by default). Nothing else is detected: a
single-word name, an invented place or any other term you don't list can be
searched. Everything searched, and every verdict, is saved to `output/interim/search_chapter_NN.md` so you can audit it.
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
- optional `web` extra: `fastapi` + `uvicorn` for the web GUI

## License

This project is open source and available for educational and commercial use.

## Notes

- Works with any OpenAI-compatible endpoint (`llm.base_url` in config.yaml;
  default `http://localhost:11434`, i.e. a local Ollama server)
- Content quality depends on the selected model; a 7B model is fine for
  structure but a larger model gives noticeably better prose
- A full 5-chapter run is roughly 4N+2 LLM calls (~22) — expect several
  minutes to an hour depending on hardware
