# Multi-Agent Book Writer

A collaborative AI book writing system using multiple specialized agents. Feed it a **seed prompt** — your premise, characters, world, tone, and (optionally) a chapter outline — and the agent team drafts a complete, polished book from it.

<img src="/images/writer.gif" alt="writer demo" style="width:100%; height:auto;" />

## Project Overview

The writing team consists of five agents:

- **Architect**: reads your seed prompt and distills it into a *story bible* (title, genre, tone, characters, world, constraints)
- **Planner**: builds the chapter outline — **honoring your outline if you provided one**, only expanding it if you ask for more chapters
- **Researcher (Lore Keeper)**: writes a per-chapter brief: which characters are on page, setting details, plot beats, continuity notes
- **Writer**: drafts each chapter with continuity — it sees the story bible, the chapter brief, and a rolling summary of every previous chapter
- **Editor**: polishes each chapter and checks it against the story bible (character names, world facts, constraints)

All agents share one context object; every LLM call goes through a single configured client with timeouts and retries.

## Tech Stack

| Component | Tool/Library |
|-----------|-------------|
| LLMs | Ollama (mistral, llama3, deepseek, ...) |
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
│   ├── writer.py           # continuity-aware chapter drafts + summaries
│   └── editor.py           # polish, consistency check, save
├── shared/
│   ├── context.py          # shared state (in-place reset)
│   ├── ollama_client.py    # single LLM client: config, timeout, retries
│   └── llm_utils.py        # JSON extraction / output cleanup helpers
├── seeds/
│   └── example_seed.md     # example seed prompt (a fantasy mystery)
├── tests/
│   ├── test_parsing.py     # unit tests for parsing helpers
│   └── test_output.py      # unit tests for interim output helpers
├── output/                 # generated books + interim progress artifacts
├── config.yaml             # model, temperatures, timeouts, output settings
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running

### Setup Steps

1. **Clone the project**
   ```bash
   git clone <this-repo>
   cd Multi-Agent-Book-Writer
   ```

2. **Pull an LLM model with Ollama**
   ```bash
   ollama pull mistral
   # Alternatives: ollama pull llama3, ollama pull deepseek-r1
   ```

3. **Start Ollama** (keep it running in a separate terminal)
   ```bash
   ollama serve
   ```

4. **Install Python dependencies**
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

## How It Works

### Pipeline Flow

1. **Architect**: seed prompt → structured story bible (JSON, with a
   no-LLM fallback that keeps the raw seed)
2. **Planner**: seed outline (if any) or LLM-generated JSON outline, validated
   and normalized
3. **Researcher**: a lore brief per chapter, keyed by chapter number
4. **Writer**: drafts each chapter from bible + brief + rolling summary of
   previous chapters; generates a short summary after each for continuity
5. **Editor**: polishes each chapter and fixes inconsistencies with the bible;
   on any failure the unedited draft is kept (nothing is ever lost)
6. **Save**: `output/draft.md` (created automatically) plus
   `output/story_bible.md` for inspection

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
└── edited_chapter_NN.md  # each edited chapter
```

Disable with `output.interim: false` in config.yaml.

### Agent Communication

All agents share a central `context` dict: `seed`, `title`, `bible`,
`chapters`, `research`, `drafts`, `summaries`, `final`.

## Configuration

`config.yaml` is actually loaded and applied:

- **book**: `num_chapters` (fallback), `words_per_chapter` (target length)
- **ollama**: `api_url`, `model`, `timeout` (per request), `retries` (with backoff)
- **agents**: per-agent `temperature` and `enabled` (disable `researcher`/`editor` to speed things up)
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

### Connection error to Ollama
The pipeline preflights Ollama at startup and exits with the available models
listed if the configured model is missing. Make sure `ollama serve` is running.

### Slow generation
- Use `--chapters 2` for a quick test before a full run
- Set `agents.researcher.enabled: false` and/or `agents.editor.enabled: false`
- Use a smaller/faster model (`--model`, or `ollama pull orca-mini`)

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

- `requests`: HTTP client for the Ollama API
- `PyYAML`: config loading

## License

This project is open source and available for educational and commercial use.

## Notes

- Ollama must be running locally (default `http://localhost:11434`)
- Content quality depends on the selected model; a 7B model is fine for
  structure but a larger model gives noticeably better prose
- A full 5-chapter run is roughly 4N+2 LLM calls (~22) — expect several
  minutes to an hour depending on hardware
