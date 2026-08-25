# Multi-Agent Book Writer - Quick Start Guide

Write a book from a seed prompt: characters, world, tone, and an optional
outline in - a polished draft out.

## Quick Setup (5 minutes)

### 1. Install Ollama
Download from https://ollama.ai

### 2. Pull a Model
```bash
ollama pull mistral
```

### 3. Start Ollama Server
```bash
ollama serve
```
Keep this terminal open!

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Project
```bash
python main.py                 # bundled example seed (fantasy mystery)
python main.py --seed my.md    # your own seed prompt
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
python main.py --seed my_book.md
```

See `seeds/example_seed.md` for a complete example and
`seeds/SEED_SCHEMA.md` for the full schema (what is parsed verbatim vs.
LLM-extracted). Sections are optional - the Architect agent fills gaps, and
the Planner generates an outline if you don't provide one.

## Command Examples

- **Your seed**: `python main.py --seed story.md`
- **Inline premise**: `python main.py --prompt "A noir thriller set on Mars..."`
- **Chapter count**: `python main.py --seed story.md -c 3`
- **Different model**: `python main.py --model llama3`
- **Custom output name**: `python main.py --out my_book.md`
- **View output**: `cat output/draft.md`

## Agents

1. **Architect** - seed prompt -> story bible
2. **Planner** - chapter outline (honors yours)
3. **Researcher** - per-chapter lore briefs
4. **Writer** - drafts with continuity (story facts + all previous summaries)
5. **Reviewer** - continuity check (deaths, relationships, timeline)
6. **Editor** - lint -> revise -> polish, with a final lint report

## Troubleshooting

**Error: could not connect to Ollama**
- Make sure `ollama serve` is running in another terminal

**Slow generation**
- Try 2 chapters first: `python main.py -c 2`
- Disable editing in `config.yaml` (`agents.editor.enabled: false`)
- Use a faster model: `--model orca-mini`

**Model not available**
- The startup preflight lists your installed models; pull one with `ollama pull <name>`

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
