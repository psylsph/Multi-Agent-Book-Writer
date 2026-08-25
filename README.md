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
│   └── output.py           # interim artifacts + bible formatting
├── seeds/
│   ├── SEED_SCHEMA.md      # seed format reference
│   └── example_seed.md     # example seed prompt (a fantasy mystery)
├── tests/                  # 54 unit tests (no model needed)
├── output/                 # generated books + interim progress artifacts
├── config.yaml             # model, temperatures, timeouts, output settings
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- An OpenAI-compatible LLM server (LM Studio, llama.cpp server, vLLM,
  Ollama, OpenAI, OpenRouter, ...) reachable over HTTP

### Setup Steps

1. **Clone the project**
   ```bash
   git clone <this-repo>
   cd Multi-Agent-Book-Writer
   ```

2. **Point it at your LLM server**

   Edit `config.yaml` and set `llm.base_url` to your server (the
   OpenAI-compatible `/v1/chat/completions` path is appended automatically).
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
   pip install -r requirements.txt
   ```

## Usage

### Write a book from a seed prompt

```bash
python main.py --seed seeds/example_seed.md
python main.py --prompt "A noir thriller set on Mars. Detective Rya Cole ... "
python main.py --seed my_story.md --chapters 7 --model llama3 --out my_book.md
```

Run the bundled example seed with no preparation:

```bash
python main.py            # uses seeds/example_seed.md
python main.py --demo 2   # same, but only the first 2 chapters
```

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

If the seed has no outline, the planner generates one from the story bible
(config `book.num_chapters` or `-c N` controls the count). If the seed's
outline has fewer chapters than you request, the planner expands it; if it has
more, the first N are used (with a notice).

### Chapter count priority

1. `--chapters N` / positional `N` if given
2. the number of chapters in the seed's own outline
3. `book.num_chapters` from config.yaml (default 5)

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
`--no-resume` to force a clean restart (clears the interim directory and
rebuilds everything from the seed).

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
you can follow a run as it happens (the directory is cleared at the start of
each run):

```
output/interim/
├── story_bible.md        # as soon as the architect finishes
├── outline.md            # as soon as the planner finishes
├── lore_chapter_NN.md    # per-chapter briefs, one by one
├── draft_chapter_NN.md   # each chapter draft, right after writing
├── summaries.md          # rolling chapter summaries (rewritten per chapter)
├── story_state.md        # rolling chronology: meetings, deaths, relationships
├── review_chapter_NN.md  # reviewer verdicts + issues per chapter
├── edited_chapter_NN.md  # each edited chapter
├── lint_report.md        # final deterministic lint across the book
├── bible.json            # structured bible (resume)
├── outline.json          # structured plan (resume)
├── summaries.json        # rolling summaries (resume)
└── chronology.json       # rolling story state (resume)
```

Disable with `output.interim: false` in config.yaml.

### Agent Communication

All agents share a central `context` dict: `seed`, `title`, `bible`,
`chapters`, `research`, `drafts`, `summaries`, `final`.

## Configuration

`config.yaml` is actually loaded and applied:

- **book**: `num_chapters` (fallback), `words_per_chapter` (target length), `word_count_tolerance` (enforced minimum as a fraction of the target — short chapters are lint findings and get sent back for substantive expansion, never padding), `extra_length_rounds` (additional revision rounds granted for length only, while each round still adds ≥15%), `revision_rounds` (review/revise passes per chapter; 0 disables revision)
- **llm**: `base_url`, `api_key` (`""`, plain value, or `env:VAR`; `LLM_API_KEY` env var is the fallback), `model`, `timeout` (per request), `retries` (with backoff)
- **agents**: per-agent `temperature` and `enabled` (disable `researcher`/`reviewer`/`editor` to speed things up)
- **output**: `directory`, `filename`, `overwrite` (`false` appends `-1`, `-2`, ... instead of clobbering), `interim` (progress artifacts under `<directory>/interim/`)

CLI overrides: `--model`, `--out`, `--config`.

## Development

```bash
pip install pytest
python -m pytest tests/ -q
```

The tests cover the pure parsing helpers (JSON extraction, LLM output cleanup,
seed-outline extraction) — no model required.

## Troubleshooting

### Connection error to the LLM server
The pipeline preflights the endpoint at startup and exits with the available
models listed if the configured model is missing. Make sure the server is
running and `llm.base_url` in config.yaml is correct (no trailing
`/v1` — it is appended automatically).

### Slow generation
- Use `--chapters 2` for a quick test before a full run
- Set `agents.researcher.enabled: false` and/or `agents.editor.enabled: false`
- Use a smaller/faster model (`--model`)

### Out of memory
- Try a smaller model
- Reduce chapter count

## Extensions & Improvements

- [ ] Parallelize per-chapter research/writing
- [ ] Word-count enforcement / regeneration passes
- [ ] Web search integration for research-grounded nonfiction
- [ ] PDF/EPUB export
- [ ] Streamlit UI for monitoring and editing seeds
- [ ] Character-voice conditioning per POV chapter

## Requirements

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
