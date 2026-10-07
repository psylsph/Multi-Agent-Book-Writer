# Roadmap

What is planned, and what is known to be missing or imperfect. Nothing here is
promised; it is a list of honest gaps and good next steps, roughly in the order
worth doing them. Tick a box (and move it to the README if it needs
documenting) when it ships.

## Planned

### Pipeline
- [ ] **Parallel research.** Lore briefs are independent per chapter, and so are
  the claim verifications within a chapter. Run them concurrently when the
  server has spare slots (llama.cpp `--parallel`, vLLM). Writing and editing
  stay sequential because each chapter builds on the last.
- [ ] **Per-agent `base_url`.** `agents.<name>.model` already lets an agent use a
  different model, but all models must live on one server. A per-agent URL (and
  key) would allow, say, a hosted reviewer with a local writer. Needs per-URL
  endpoint waiting and preflight.
- [ ] **Token-aware prompt budget.** The recap is windowed by *chapter count*
  (`book.summary_window`). Now that the context guard calibrates characters per
  token, trim the recap, story facts and bible to a token budget instead.
- [ ] **JSON mode for array replies.** `llm.json_mode` only covers calls that
  return an object, because a JSON-object grammar can't produce an array. Wrap
  the planner outline, claim list and verifier output in an object to cover them
  too.
- [ ] **Adaptive chapter length.** One `words_per_chapter` for every chapter. Let
  the outline mark a chapter as short/long, or derive targets from its beats.
- [ ] **Per-agent sampling.** Only `temperature` is configurable per agent. Add
  `top_p`, `seed` and friends, each sent only when set (the same rule as
  temperature).
- [ ] **Reproducible runs.** A `seed` option, sent to servers that honour it, so
  a run can be repeated.
- [ ] **Post-draft fact-check.** The researcher checks real-world claims *before*
  writing. The reviewer could also extract claims from the finished prose and run
  them through the same verify-with-quote machinery.
- [ ] **Cross-run search cache.** Remember verified claims between runs and
  books so the same fact is not searched twice.

### Output
- [ ] **EPUB / PDF export** (pandoc, or `ebooklib`): `--format epub`.
- [ ] **Streamlit UI** for monitoring a run and editing seeds.
- [ ] **Character-voice conditioning** per point-of-view chapter.

### Engineering
- [ ] **Explicit `--resume`.** Resume is automatic. A flag that *fails* when there
  is nothing to resume would make intent visible in `--help` and in scripts.
- [ ] **Remove global state.** The config (`_config`), the web-search status and
  the run statistics are module-level globals, which is why tests must reset
  them. Pass a config object instead.
- [ ] **Prompt evaluation harness.** The system prompts, the refusal guard,
  verdict checking and review-as-you-go are tested with a fake LLM, which proves
  the wiring but not the quality. A small set of golden seeds run against a real
  model, with the outputs diffed after a prompt change, would catch regressions
  the unit tests cannot.
- [ ] **CI hardening.** The workflow has not run yet (action versions are
  unverified). Once it has: a coverage floor (`--cov-fail-under`), and a lockfile
  freshness check.
- [ ] **Estimate tokens when the server hides usage.** The statistics report
  "did not report token usage" and count nothing; a labelled estimate would still
  give a useful tokens/s.

## Measurements

Everything below was measured with the tools in `tools/` (`extractor_eval.py`,
`verdict_eval.py`, `reviewer_eval.py`) on one machine: a Ryzen 5 7600, a
Radeon RX 7700 XT (12 GB), a 35B mixture-of-experts model with thinking on as
the main model, and small models run through `tools/run-small-model.sh`. The
chapters and claims are invented. Samples are small (one to four runs per
chapter): treat the numbers as indicative, and re-measure on your own model.

### The verdict second check (`web_search.double_check`), 34 claim/quote pairs

The check asks one model, per verdict, "does this quote, by itself, directly
support / contradict the claim?". A wrong *yes* lets a misread fact reach the
writer (the costly error); a wrong *no* only drops a good fact.

- **A bug, now fixed:** the answer was requested as `{"agrees": true}`. For a
  *contradiction* verdict a model reads that as "does the quote agree with the
  claim?" and answers false to a quote that does contradict it. The main model
  got **0 of 16** contradictions right (accuracy 0.68, 53% good facts dropped);
  with the key renamed `"answer"` it scores **1.00 accuracy, 0.00 false yes,
  0.00 false no**. Contradicted facts are the ones that correct real-world
  errors, so they were being thrown away.
- **Thinking does not help this task** on the main model: thinking off and low
  both score 1.00 / 0.00, at 0.5 s against 7-10 s per check. The verification
  calls now run as their own `verifier` agent, so
  `agents.verifier.enable_thinking: false` takes the speed-up without touching
  the researcher's brief-writing.
- **Small models** (thinking *disabled* with `--reasoning-budget 0`, accuracy /
  false yes / false no): MiniCPM5-2B 0.85 / 0.24 / 0.03 and Phi-4-mini
  0.82 / 0.26 / 0.07 (both with the `strict` prompt in `tools/verdict_prompts.py`),
  Granite 4.2 3B 0.74 / 0.16 / 0.40, LFM2.5-2.6B 0.68 / 0.16 / 0.53. All are weak
  on the "verdict has the sign backwards" trap.
- **LFM2.5-2.6B with thinking on** scored 1.00 / 0.00 / 0.00 (one run of 34
  pairs, ~12 s per check on a busy CPU). Small *thinking* models can do this
  task; forcing them not to think is what made them look poor. It would add a
  different-family second opinion, but the main model already scores 1.00, so
  the value is unproven without real data.
- K2 Horizon 3.7B, the top tiny model on Artificial Analysis, could not be
  tested: this llama.cpp build reports `unknown model architecture`.

### The reviewer, 14 versions of one chapter (4 clean, 10 with a planted error)

| reviewer | found any | right type | false alarms (of clean) | unreadable | s/chapter |
|---|---|---|---|---|---|
| main model, thinking off, previous prompt | 0.70 | 0.50 | 1 of 4 | 3 | 24 |
| main model, thinking off, **evidence prompt (now the default)** | 0.80 | 0.80 | 1 of 4 | 0 | 8 |
| main model, **thinking on (default; the effort setting is ignored by this model)**, evidence prompt, 2 runs | 0.65 | 0.65 | 0 of 8 | 0 | 69 |
| LFM2.5-2.6B (GPU, thinking on), evidence prompt, 2 runs | 0.45 | 0.25 | 0 of 8 | 0 | 31 |
| LFM2.5-2.6B, previous prompt | 0.50 | 0.50 | 0 of 4 | 1 | 32 |
| LFM2.5-2.6B, `sceptical` prompt | 0.60 | 0.30 | 0 of 4 | 1 | 33 |

- The prompt matters as much as the model: requiring the reviewer to quote the
  sentence *and* name the recorded fact, with silence an acceptable answer,
  removed the runaway replies and raised right-type recall from 0.50 to 0.80.
- **A small model is not a good enough reviewer.** LFM never raised a false
  alarm but missed over half the planted errors under every prompt; the same
  ones slip through regardless of wording (a secret a character could not know,
  an injury that healed overnight, a character in the wrong country, a tense
  slip, usually a relationship leap). Because of that, LFM's extraction was not
  re-checked.
- Even the main model misses the "Tom knows what only Elizabeth was told" and
  "Elizabeth is in London" errors, and sends back one clean chapter in four.
  A reviewer is a useful filter, not a guarantee; the deterministic lints
  (dead-character, first-meeting, names, repetition) do not depend on it.
- **Thinking on is not better than thinking off.** This model's chat template
  has no `reasoning_effort` support (the server reports it, and the effort
  levels are a Qwen3.8 feature), so the "medium" run was the model's ordinary
  thinking; the effort setting only matters for models that support it. It
  raised no false alarm in 8 clean reviews (thinking off: 1 of 4) but found
  fewer planted errors (13 of 20 against 8 of 10) and took about nine times
  as long (69 s against 8 s). It caught every dead-character, relationship
  regression, outline and point-of-view/tense error, and missed the secret
  a character could not know, the relationship leap and the wrong weekday
  every time. The samples are small, so the recall gap is within noise; the
  speed gap is not.

### The extractor (10 chapters with known answers, main model)

| setup | runs | invented events | false deaths | real deaths found | s/chapter |
|---|---|---|---|---|---|
| thinking off, checks on | 40 | 12 | 1 | 8/8 | ~4 |
| thinking off, checks off | 40 | 11 | 3 | 8/8 | ~4 |
| thinking off, temperature 0.1 | 40 | 15 | 4 | 8/8 | ~4 |
| thinking off, `strict` extractor prompt | 40 | 10 | 3 | 8/8 | ~4 |
| thinking low (the default), checks on | 10 | 3 | 0 | 2/2 | ~55 |
| Phi-4-mini on CPU (earlier five-chapter set) | 15 | 12 (0.8 per chapter) | 0 | - | ~14 |

- Lowering the temperature did **not** help; the tailored prompt improved which
  characters are listed as present (precision 0.81 to 0.90) more than anything.
- A small CPU model is not worth it as extractor: it invented about four times
  as many events as the main model.
- Fixed after these runs (the re-measure was stopped before it finished):
  the death-confirmation step could *add* a death for a living character
  standing near death language, so an added death now needs a second,
  differently phrased question ("who is still alive?") to agree; titles ("Dr
  Cole") now match the character; a character who died earlier is no longer
  listed as present.
- Also found and fixed along the way: a model answering `"who": "Tom Baker"`
  (a string) created one single-letter "character" per letter.

### Operational findings

- **Runaway replies are real.** A reviewer reply decoded past 9,800 tokens
  (about ten minutes) before being cancelled; the pipeline would then have
  treated the unparseable reply as a pass. The researcher, extractor, verifier
  and reviewer now have built-in output ceilings (6000 / 6000 / 2000 / 8000
  tokens) unless `max_tokens` is set; the writer, editor, architect and planner
  are deliberately not capped.
- On the GPU, LFM2.5-2.6B reviewed a chapter in about 31 s; on a busy CPU it
  took 4 to 6 minutes.

## Known limitations

### Accuracy
- **Verdicts can still be wrong.** A fact reaches the writer only if its quote is
  really in the cited result, shares the claim's specific words, and survives an
  independent second check (`web_search.double_check`). None of that proves the
  model *understood* the quote. Read the first `search_chapter_NN.md` files of a
  new book.
- **Search queries are only filtered by your list.** Every query is allowed
  unless it mentions a term in `web_search.banned_terms`; story names are not
  detected automatically (an earlier heuristic that did so was removed). The
  model is told not to put story details in a query, but a name you did not
  list can still be searched. Queries leave the machine via SearXNG's upstream
  engines.
- **Nicknames.** Declared aliases and a built-in table of common short forms are
  resolved in code; a nickname in neither relies on the extractor prompt.
- **Repetition lint is fixed-shape.** It looks for repeated runs of six words,
  with "already used twice" as the per-chapter threshold. Neither is
  configurable, and repeated *ideas* in different words are not detected.
- **Name-typo lint is ambiguous by nature.** A near-miss like `Lissa` for `Lisa`
  could be a typo or a different person; it is flagged, and
  `book.name_lint_ignore` silences it.
- **Reviewer false positives.** Outline and constraint checks can ask for a
  revision the chapter did not need. Edits are rolled back if they make things
  worse, and `diff_chapter_NN.md` shows what happened, but a wasted round costs
  time. `book.review_checks` can switch checks off.

### Robustness
- **Crash window in review-as-you-go.** A crash between a chapter's edit and the
  refresh of its story facts leaves that chapter with facts from its first draft
  after resume.
- **Streaming usage.** Not every server reports token usage in a stream; the
  statistics then show those calls as unreported.
- **Language.** Prompts, linting heuristics (stop-words, adverb counting, name
  matching) and the nickname table are English-only.

### Operations
- **SearXNG settings directory ownership.** After the container first starts, the
  Docker image takes ownership of `~/.config/multi-agent-book-writer/searxng/`
  (uid 977), so editing or deleting `settings.yml` needs `sudo`. A named Docker
  volume instead of a bind mount would avoid it. Stop the container with
  `docker stop book-writer-searxng`, remove it with `docker rm`.
- **`config.yaml` is tracked in git** and holds machine-specific settings (model
  name, port). Untracking it, keeping only `config.example.yaml`, is an open
  decision.
