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

## Known limitations

### Accuracy
- **Verdicts can still be wrong.** A fact reaches the writer only if its quote is
  really in the cited result, shares the claim's specific words, and survives an
  independent second check (`web_search.double_check`). None of that proves the
  model *understood* the quote. Read the first `search_chapter_NN.md` files of a
  new book.
- **Search queries are filtered heuristically.** Character names, capitalised
  phrases ("Willow Rooms") and capitalised words that never appear in lower case
  are blocked. An invented single-word place that also occurs in lower case in
  your text will not be recognised. Real places need `web_search.allow_terms`.
  Queries still leave the machine via SearXNG's upstream engines.
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
