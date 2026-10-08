# Multi-Agent Book Writer - Quick Start Guide

Write a book from a seed prompt: characters, world, tone, and an optional
outline in - a polished draft out.

## Quick Setup (5 minutes)

### 1. Start an OpenAI-Compatible LLM Server
The pipeline talks the OpenAI chat-completions API, so any compatible
server works. Examples:

- **LM Studio** - start the local server (default `http://localhost:1234`)
- **llama.cpp server** - `llama-server` (default `http://localhost:8080`)
- **Ollama** - `ollama serve` (default `http://localhost:11434`;
  the pipeline uses its OpenAI-compatible `/v1` API)
- **Hosted APIs** - OpenAI, OpenRouter, etc. (set `llm.api_key` too)

Then point `config.yaml` at it (`config.example.yaml` lists every option):

```yaml
llm:
  base_url: "http://localhost:1234"  # /v1/chat/completions is appended
  model: "mistral"                   # any model the server offers
  api_key: ""                        # or "env:MY_VAR" for hosted APIs
```

### 2. Install Dependencies
The project uses [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

### 3. Run the Project
```bash
uv run main.py                 # bundled example seed (fantasy mystery)
uv run main.py --seed my.md    # your own seed prompt
```

## Writing From Your Own Seed

Create a markdown file with your story's ingredients:

```markdown
# My Book

## Premise
Who wants what, and what stands in the way.

## Characters
- **Aria Voss** - protagonist. Stubborn chartmaker, distrusts the sea.

## World
The island, the rules, the background.

## Tone & Style
Third-person limited, atmospheric.

## Constraints
- Aria left the island twelve years ago.

## Outline
- Chapter 1: Homecoming - she returns for the bequeathal.
- Chapter 2: What the Flame Remembers - her first memory.
```

Then run:

```bash
uv run main.py --seed my_book.md
```

See `seeds/example_seed.md` for a complete example and
`seeds/SEED_SCHEMA.md` for the full schema (what is parsed verbatim vs.
LLM-extracted). Sections are optional - the Architect agent fills gaps, and
the Planner generates an outline if you don't provide one.

## Command Examples

- **Your seed**: `uv run main.py --seed story.md`
- **Inline premise**: `uv run main.py --prompt "A noir thriller set on Mars..."`
- **Chapter count**: `uv run main.py --seed story.md -c 3`
- **Check the plan first**: `uv run main.py --seed story.md --plan-only`
- **Different model**: `uv run main.py --model llama3`
- **Custom output name**: `uv run main.py --out my_book.md`
- **View output**: `cat output/draft.md`
- **See what happened**: `output/interim/run_stats.md` (calls, tokens, time),
  `output/interim/diff_chapter_NN.md` (what the editor changed) and
  `output/logs/run-*.log` (the whole console output)

## Command-line Options

| Option | Meaning |
|---|---|
| `--seed FILE` | seed prompt from a file |
| `--prompt TEXT` | seed prompt as inline text |
| `--demo` | bundled example seed |
| `-c N`, `--chapters N` | chapter count (also positional: `main.py N`) |
| `--config FILE` | config file (default `config.yaml`) |
| `--model NAME` | override `llm.model` |
| `--out FILE` | override the output filename |
| `--plan-only` | stop after the seed review and outline; rerun without it to write |
| `--seed-review MODE` | `ask` (default), `warn` or `off` |
| `--no-resume` | start over (the old run is archived to `output/archive/`, not deleted) |

## Resuming

There is no `--resume` flag; it is automatic. If a run crashes, the server
drops, or you hit Ctrl-C, just rerun the **same command** and it continues
from the last saved chapter (progress lives in `output/interim/`).

- Start over instead: add `--no-resume` (the old run is moved to `output/archive/`, not deleted).
- Rerun with no `--seed`/`--prompt`: the saved seed is reused.
- Run with a *different* seed or `-c` than the saved run: refused, so you
  can't accidentally reuse another book's chapters. Add `--no-resume` to
  start the new book.

## Before writing: the seed review

The first thing a run does is check that your seed has enough story for the
length you asked for. On a terminal it asks a few questions about gaps (Enter
accepts the assumption shown) and, if the size doesn't fit, lets you keep it,
take the recommended size, choose your own, or stop and edit the seed. Use
`--plan-only` to stop after the outline and read `output/interim/outline.md`
before committing to the full run.

## Making an EPUB

When the chapters are done, build a single ebook (no LLM needed):

```bash
uv run python make_epub.py --author "Your Name"
```

This reads `output/chapters/` and writes `output/<title>.epub` with a cover,
contents page and tidied typography. Add `--cover cover.jpg` for your own cover;
`make_epub.py --help` lists the rest.

## Optional: web fact-checking

Set `web_search.enabled: true` in `config.yaml` and the researcher checks
real-world details online through SearXNG. If it isn't running and you have
Docker, the app asks `[y/N]` before downloading and starting it (local only,
no API key). Queries never include character names and are logged to
`output/interim/search_chapter_NN.md`. See the README for details.

## Agents

1. **Architect** - seed prompt -> story bible
2. **Planner** - chapter outline (honors yours)
3. **Researcher** - per-chapter lore briefs
4. **Writer** - drafts with continuity (story facts + recent chapter summaries)
5. **Reviewer** - continuity check (deaths, relationships, timeline)
6. **Editor** - lint -> revise -> polish, with a final lint report

## Troubleshooting

**Error: could not connect to the LLM server**
- Make sure the server is running and `llm.base_url` in config.yaml is correct
- If the server needs a key, set `llm.api_key` (or the `LLM_API_KEY` env var)

**Slow generation**
- Try 2 chapters first: `uv run main.py -c 2`
- Disable editing in `config.yaml` (`agents.editor.enabled: false`)
- Use a faster model: `--model <name>`

**Model not available**
- The startup preflight lists the models your server offers; set `llm.model`
  in config.yaml or pass `--model <name>`

## Output

Your finished book is saved in `output/draft.md` (plus
`output/story_bible.md` so you can check what the agents extracted from your
seed):

- Complete chapters with headings
- Character names and world facts kept consistent with your seed
- Edited and polished prose

While the run is in progress, every intermediate artifact lands in
`output/interim/` as soon as it's ready - story bible, outline, per-chapter
lore briefs, chapter drafts, rolling summaries, and edited chapters - so you
can read along instead of waiting:

```bash
watch ls output/interim/    # or just re-run: ls output/interim/
```

---

For detailed documentation, see README.md
