# Seed File Schema

A seed is the creative brief the agent team works from: premise, characters,
world, style rules, and (optionally) a chapter outline. This document is the
authoritative reference for what the parsers recognize.

- **Format**: plain Markdown, UTF-8, `.md` or `.txt`
- **Input**: `--seed FILE` (or the same text inline via `--prompt TEXT`)
- **Inspection**: after a run, `output/interim/story_bible.md` (or
  `output/story_bible.md`) shows exactly what was extracted

## Skeleton

```markdown
# Book Title                      <- H1: the book title
Short preamble prose.             <- optional; becomes author notes

## Premise                        <- LLM-extracted into the bible
...

## Characters                     <- parsed VERBATIM (deterministic)
...

## World                          <- LLM-extracted
...

## Tone & Style                   <- LLM-extracted
...

## Constraints                    <- parsed VERBATIM
...

## Outline                        <- parsed VERBATIM
...

## Anything Else                  <- unrecognized sections become
...                                  author notes (verbatim)
```

All sections are optional. A single freeform paragraph is a valid seed.

## Section recognition

- `#` (H1) and `##` (H2) headings start a new section. Deeper headings
  (`###` and below) are kept **inside** the current section.
- Section names match by **case-insensitive substring**:

| Bible key     | Heading must contain        | How it is parsed              |
|---------------|-----------------------------|-------------------------------|
| `premise`     | `premise`, `synopsis`, `plot` | LLM-extracted (faithful paraphrase) |
| `characters`  | `character`                 | deterministic, verbatim       |
| `world`       | `world`, `setting`          | LLM-extracted                 |
| `tone`        | `tone`, `style`             | LLM-extracted                 |
| `outline`     | heading must be **exactly** `Outline` / `Story Outline` / `The Outline` (bold allowed) | deterministic, verbatim |
| `constraints` | `constraint`                | deterministic, verbatim       |
| (notes)       | anything else               | whole body kept verbatim      |

> Deterministic sections are taken word-for-word from your seed — the LLM
> never rewrites them. LLM-extracted sections are summarized into the bible
> by the architect model, so put anything that must survive byte-for-byte
> into Characters, Constraints, an Outline, or a custom notes section.

## Title

The H1 heading is the book title. If there is no H1, the architect's LLM
titles the book; the last-resort fallback is the first non-empty line.

## Characters

One bullet per character. Grouping sub-headings (`### ...`) are allowed and
ignored. Lines that wrap are joined onto the description.

```markdown
## Characters

### The trio

- **Aria Voss** - protagonist. 29, chartmaker, distrusts the sea.
  Left Hesswick at seventeen. Sharp, stubborn, quietly grieving.
- **Elder Marrow (harbormaster)** - antagonist. Keeper of the island's
  ledgers and its silences.
```

Grammar per bullet: optional `-`/`*`/`1.` marker, `**Name**` in bold,
optional `(role)` in parentheses, a separator (`-`, `–`, `—`, or `:`), then
the description.

- Put the role at the **start of the description** (`- **Aria** -
  protagonist. 29, ...`) — `protagonist` / `antagonist` / `supporting`
  there is picked up as the role. A parenthetical directly after the bold
  name (`**Aria** (protagonist) - ...`) is ignored by the parser.
- Parentheses **inside** the bold (`**Lisa (the third)**`) become part of
  the name — useful for disambiguating epithets, but the name reaches the
  writer exactly as written.
- **Names must be bold** (`**Name**`) to be captured; a plain `- Aria - ...`
  line is ignored by the deterministic parser.

## Constraints

One rule per bullet; wrapped lines join onto their bullet. These go straight
into every writer and editor prompt as "must remain true", so keep them
imperative and checkable.

```markdown
## Constraints

- British English. Never the words "unhurried", "unrushed", or "exactly".
- Em-dash rare — max 8 per scene, never in dialogue (only for a sentence
  cut off mid-word).
- All characters are adults.
```

## Author notes (custom sections)

Any section whose heading contains none of the recognized words is preserved
verbatim under **Author notes** and shown to the writer and editor. Use this
for explicitness levels, banned-word lists, formatting rules, canon notes:

```markdown
## Explicitness

Very explicit. No euphemisms. No fade-to-black for the first time (Ch 9):
on-page but nothing finished inside.

## Canon

- The Beacon shows memories, never predictions.
```

Preamble prose between the H1 and the first `##` also becomes author notes.

## Outline

The planner uses this section **as-is** — it does not invent its own
structure unless you ask for a different chapter count.

Inside `## Outline`:

```markdown
## Outline

### Part One - The Three          <- H3+: part label for the chapters below

- Chapter 1: The Fall (Stuart POV, heat 2) - He rides the woods loop
  and comes off the bike on a rooty descent.                 <- wrapped
- Chapter 2: The Treatment (Lisa POV, heat 4) - First meeting at the clinic.

### Part Two - The Build

- Chapter 3: The Party (heat 6) - A charged evening with the group.
- The pampas beat (folded into the build) - A comedic setup; not its
  own chapter.                          <- non-chapter line: folds into Ch 3
```

Rules, in order of application:

1. **Chapter lines** — anything matching
   `[bullet] [**]Chapter N[**] [:|-|–|—] Title [ - Summary]` becomes a
   chapter. Title and summary split on the *first* of ` - `, ` — `, ` — `,
   or `: `. Everything after the number is kept verbatim in the title, so
   annotations like `(Stuart POV, heat 2)` survive.
2. **Parts** — `###`+ headings inside the outline become `part` labels
   attached to the following chapters.
3. **Bare-bullet mode** — if the section contains *no* `Chapter N:` lines at
   all, plain bullets (`- Title - Summary`) are treated as chapters.
4. **Fold-in beats & wrapped summaries** — any other non-blank line is
   appended to the previous chapter's summary. This is how multi-line
   summaries and non-chapter beats stay in the plan.
5. **Numbering** — explicit numbers are kept when strictly increasing;
   duplicates or gaps are renumbered sequentially.
6. **Section end** — the outline ends at the next H1/H2 heading. Anything
   deeper (H3+) stays inside.
7. **Loose fallback** — if there is *no* Outline section, explicit
   `Chapter N:` lines found anywhere in the seed (at least two) are used.
   Avoid stray sentence-initial "Chapter 3 ..." lines in prose.

### Chapter count resolution

1. `--chapters N` (or the legacy positional `N`) if given — always wins
2. otherwise, the number of chapters in the seed outline
3. otherwise `book.num_chapters` from `config.yaml` (default 5)

If you ask for **more** chapters than the outline has, the planner expands
it via the LLM (your chapters stay fixed as the opening run). If you ask for
**fewer**, the first N are used and a notice is printed.

## Pipeline mapping (what reads what)

| Agent        | Reads from the seed                                        |
|--------------|------------------------------------------------------------|
| Architect    | everything (builds the bible; Characters/Constraints/notes/outline override its LLM output) |
| Planner      | Outline section (or generates one from the bible)           |
| Researcher   | bible → per-chapter lore brief                              |
| Writer       | bible + lore brief + rolling summaries of earlier chapters  |
| Editor       | bible (names, world, constraints, notes) for consistency    |

## Minimal seed

```markdown
A lonely lighthouse keeper on a deep-space beacon discovers the light has
been signaling for someone else. Quiet, melancholic, first person.
```

Everything is LLM-generated from this (title, bible, outline) — fine for a
quick draft, weak on control.

## Copy-paste template

```markdown
# <Book Title>

<1-3 sentence elevator pitch.>

## Premise

<Who wants what, and what stands in the way.>

## Characters

- **<Name>** - <role>. <age/occupation, appearance, personality, motivation,
  speech patterns, what they must never do.>
- **<Name>** - <role>. <...>

## World

<Setting, rules (magic/tech/society), background the plot depends on,
sensory palette.>

## Tone & Style

<POV, tense, pacing, atmosphere, sentence rhythm, dialogue style.>

## Constraints

- <Hard fact that must never contradict.>
- <Banned words / required dialect.>
- <Anything countable (max N per scene) — best-effort, not machine-enforced.>

## <Custom section, e.g. Explicitness>

<Verbatim author notes passed to every writing/editing call.>

## Outline

### <Optional part label>

- Chapter 1: <Title (annotations allowed)> - <what happens; may wrap
  across lines>
- Chapter 2: <Title> - <what happens>
```

## Gotchas

- Section matching is substring-based: a heading like `## Character Arcs`
  would be treated as the Characters section (its bullets just won't parse
  as characters if they aren't bold-named). Prefer plain headings.
- Keep the outline heading exactly `Outline` (optionally `Story Outline` /
  `The Outline`) — `Chapter Outline` will **not** be recognized as the
  outline section (though its `Chapter N:` lines would still be picked up by
  the loose fallback).
- Use one H1 (the title). Extra H1s each start a new section.
- Character names must be `**bold**` to be captured verbatim.
- The `role` field in parentheses and the leading role word are cosmetic;
  they do not affect the pipeline beyond bible display.
