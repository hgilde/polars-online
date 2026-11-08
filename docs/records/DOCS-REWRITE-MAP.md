# The documentation rewrite: the map

*Status, 2026-11-05: written for stage 0 (d) of PLAN task 338, before any
page is rewritten. It is a record: each stage that moves a section ticks
it off in PLAN 338, not here, and the map is not kept in step with the
pages. Its section lists and line counts are generated from the pages as
they stood at `965e57c0`, so they show WRITING.md as it stood before
stage 0 changed it. Where each section lands was decided by hand.*

The rewrite turns the README and the guides, about 31,600 lines in 632
sections, into a landing page, two maps, the records, and pages of four
kinds: tutorials, how-tos, references and
explanations ([WRITING.md §9](../WRITING.md#9-four-kinds-of-page)). Every
section of today's text has one destination, the guides stay Markdown on
GitHub, and stage 1 writes ten pieces first, as patterns. Every section
has an address in the map, and a model entry's recipes get theirs
by the rule for model entries, so nothing is lost
([WRITING.md §1](../WRITING.md#1-every-document-has-an-altitude-and-stays-at-it):
depth has an address). `CONTRIBUTING.md`, `llms.txt` and `docs/README.md`
are rewritten in place, so their sections are not mapped.

## The user's decisions

Taken on 2026-11-05. Decisions 1, 2, 4 and 5 were the coordinator's
recommendations, which the user took with "Your recos". Decisions 3, 6
and 7 are in the user's own words, and decision 3 replaced the
coordinator's recommendation.

| | decision | what it means for the map |
|---|---|---|
| 1 | a stable docs link per message, and no error codes | a message's long form is a heading or a page whose address never changes ([Links from messages](#links-from-messages)) |
| 2 | validators report every problem in a spec at once, a code change, in stage 2 | nothing here; the shape of each message is stage 2's |
| 3 | "Guides stay in markdown unless a compelling reason." | the guides stay Markdown, read on GitHub. [Where the guides live](#where-the-guides-live) gives the evidence weighed, and no compelling reason to move was found |
| 4 | Vale is revisited after stage 1, and stage 0's checks do not wait for it | nothing here |
| 5 | errors and warnings first, then docstrings, then the guides | the order of the stages in [The stages](#the-stages), which the page list's stage column names |
| 6 | "Do the full doc rewrite and I will review on GitHub." | stage 1 is the first slice of the rewrite, not a pilot held for a verdict, and each stage is pushed as it lands. Stage 1's pieces still set the pattern the later stages copy, so [The stage 1 set](#the-stage-1-set) chooses them as the best representatives |
| 7 | "For the docs be sure to iterate until there is no more to consider." | every page the rewrite writes is drafted, reviewed by fresh readers with none of the stage's context, revised, and reviewed again, until a round's reviewers raise no *wrong* and no *rule* finding; preferences not applied are logged in PLAN 338 ([WRITING.md §6](../WRITING.md#6-the-pass-itself), step 10) |

### Links from messages

A message links its long form as
`https://github.com/hgilde/polars-online/blob/main/docs/<page>.md#<anchor>`,
which a terminal shows in full, and a test holds every such link to a
heading that exists. A heading a message links stays on its old page, as a
heading that shipped code links does
([Rules for moving text](#rules-for-moving-text)). A message links no heading of the README, whose sections
move.

Stages 1 and 2 come before most pages are written, so their messages
link two kinds of address only. The first is a heading the map leaves in place: the
per-model anchors of OUTPUTS.md, the finding anchors of DATA-QUALITY.md,
and DIAGNOSTICS.md's *Is it ready?* and *One memory per diagnostic: run
once or windowed*. The second is a page stage 1 writes, such as
`docs/how-to/put-rows-in-clock-order.md`, which the step-back error
links. Each later stage adds links to the messages whose long form it
writes. A link once shipped is never changed: the heading it names stays,
and links on to the page that later holds the long form.

## The stages

Each stage lands as its own change, in this order, and runs WRITING
§6's pass with its review loop:

| stage | what it rewrites | the pages it writes |
|---|---|---|
| 0 | the foundations: the rules, the glossary, the checks in report mode (each fails the test gate from the stage *When each check fails the gate* names), and this map | `docs/GLOSSARY.md`, `docs/WRITING.md`, `docs/records/DOCS-REWRITE-MAP.md` |
| 1 | the first slice: at least one piece of each kind of page, docstring and message, which every later stage copies ([The stage 1 set](#the-stage-1-set)) | the stage 1 set |
| 2 | every error, warning and printed message, Python by module and then Rust by crate, and the validators that report every problem in a spec at once | none; the messages link headings that exist when it lands ([Links from messages](#links-from-messages)) |
| 3 | every public docstring, by family, and the Sphinx reference as one page per entry | `docs/reference/` |
| 4 | the reference pages, moving the dated records out of PERFORMANCE.md and STATE-WORKFLOW.md as it rewrites them | `docs/records/PERFORMANCE-LOG.md`, `docs/records/STATE-WORKFLOW-RESEARCH.md`, `docs/MODELS.md`, `docs/OUTPUTS.md`, `docs/RUNNER.md`, `docs/STATE-WORKFLOW.md`, `docs/DIAGNOSTICS.md`, `docs/DATA-QUALITY.md`, `docs/PERFORMANCE.md`, `docs/REGIMES.md`, `docs/VALIDATION.md`, `docs/VERSIONS.md` |
| 5 | the explanations | the rest of `docs/explanation/`, `docs/GRAM.md`, `docs/WINDOWS.md` |
| 6 | the how-tos, one task a page | the rest of `docs/how-to/`, `docs/CALIBRATION.md`, `docs/GBDT.md` |
| 7 | the tutorials | `docs/tutorials/` |
| 8 | the README as the landing page, `llms.txt` and the index of documents, each rewritten in place | `README.md`, `llms.txt`, `docs/README.md` |
| 9 | the contributor pages, and the records moved out of TESTING.md and RELEASE-READINESS.md | `docs/contributing/`, `CONTRIBUTING.md` (rewritten in place), `docs/EXTENDING.md`, `docs/TESTING.md`, `docs/RELEASE-READINESS.md`, `docs/WRITING.md`, `docs/records/TESTING-LOG.md`, `docs/records/RELEASE-HISTORY.md` |
| 10 | American spelling over the pages no earlier stage rewrote, the records included (their words change only in spelling, and a quotation keeps its own), then a cold read of every page and the link checks over all of them | none new |

**When each check fails the gate.** Each check of WRITING §13 runs in
report mode from stage 0. It fails the test gate once the stage that
rewrites what it reads has finished:

| check | fails the gate |
|---|---|
| the message shape | after stage 2 |
| the vocabulary in messages | after stage 2 |
| the docstring sections | after stage 3 |
| the vocabulary, the sentence counts and the page kind on a guide | from the last stage that moves text onto or off that guide, and on every guide (the records apart) after stage 10 |

Stage 1 and stage 8 rewrite the README's prose, so each logs its pass in
the log of README passes, which WRITING §2 and §6 name, with the map's
README rows copied into its entry, as WRITING §2 asks. The stages between move README sections out and leave a
one-line link, which rewrites no prose and so is no pass.

## The pages the rewrite produces

Three of the kinds in the page list sit outside WRITING §9's four: the
landing page, the two maps of the documents, and the records, which are not rewritten
beyond stage 10's spelling.
A *contributor how-to* and a *contributor reference* are §9's how-to and
reference, written for someone changing the library, and declare
themselves how-to and reference for WRITING §13's page-kind check. Six
pages
hold a second kind on purpose:

| page | its second kind | why |
|---|---|---|
| the README | its opening is the first tutorial | the first example is what a new reader opens it for |
| WRITING.md | its pass (§6, a how-to) and its rewrites (§8, explanations) | every rule cites them |
| DATA-QUALITY.md | each finding's entry ends in its remedy | WRITING §11 asks every message to say what to do, and the entry is the long form of `bank.check()`'s message |
| EXTENDING.md | each step says what the layer it touches holds | a contributor needs to know what a layer holds to act on the step that touches it |
| GLOSSARY.md | the reason for each contested word | a writer needs it to choose a new one |
| TESTING.md | the reasons a platform's last bits differ | a contributor reading which machine runs what needs them beside it |

The page list has one row more than there are paths: the README appears
twice, as the landing page and as its opening, the first tutorial. One of
the paths is the Sphinx directory `docs/reference/`. The how-to count is
a floor, since the how-tos for the recipes in the model entries are named
later, by the rule for model entries. The stage column names the stage
that rewrites a page, not every stage that edits it. Every stage adds its
new pages to `docs/README.md`, stage 1 corrects a sentence of WRITING
§2, and stage 8 replaces §2's rule that every guide has a README section.
The rows of the page list,
counted by kind (the README twice, and `docs/reference/` as one):

| kind | rows |
|---|---:|
| landing | 1 |
| map | 2 |
| tutorial | 4 |
| how-to | 68 |
| reference | 12 |
| explanation | 17 |
| contributor how-to | 10 |
| contributor reference | 4 |
| record | 5 |
| all | 123, for 122 paths |

PLAN 338's stages named 7 explanations. The map needs 17, because ten
more topics carry reasons that no how-to or reference entry can hold:

- how diagnostics keep their memory
- how a Gram stands in for the rows
- where the time and memory go
- why asynchronous series bias a correlation
- the comparison with scikit-learn
- the four interfaces into Polars
- how the data checks work
- how `po.gbdt` builds its training data
- how the library is tested
- how a window operator weighs its rows

**Where each page goes.** Every path in the page list is from the
repository's top. In the section tables, a path names a file under
`docs/` unless it is `README.md`, `CONTRIBUTING.md` or `llms.txt`. New
pages go in subdirectories named by kind: `docs/tutorials/`,
`docs/how-to/` and `docs/explanation/` for those kinds, and
`docs/contributing/` for contributor pages of either kind. A new
reference page for users goes at the top of `docs/`, since `docs/reference/` holds
the Sphinx sources, and a record goes under `docs/records/`.

**Every page that exists today keeps its path, and becomes whichever kind
its remaining text is.** That is one rule for every page, since `llms.txt`,
the README, code comments and other guides link them all by path. It
matters most where released docstrings link pages on `main` by path, and a released wheel keeps those links for
as long as it is installed. They link `OUTPUTS.md` with an anchor per
model (24 links), `GRAM.md` (11, none with an anchor), `DIAGNOSTICS.md`
(5, two with an anchor), `DATA-QUALITY.md` (3, one with an anchor, and
`bank.check()` links each finding's anchor), and `RUNNER.md` and
`REGIMES.md` (one each, with no anchor). Every anchor in that list keeps a
heading on its page.

**The README's sections move once their stage comes,** since no shipped
code links an anchor of the README. `llms.txt` links four of them (the
README's top, *Introduction*, *How a bank sees a stream* and *Against
scikit-learn*), and those headings stay until stage 8 rewrites it. The
README also links dated sections of PERFORMANCE.md (§11, §30, §33 and
§46), which stage 4 moves into the records. Stage 4 points those links at
PERFORMANCE.md's reference sections instead.

| page | kind | stage | what it is for |
|---|---|---|---|
| `README.md` | landing | 8 | What the library is and is not, how to install it, the first example, the example data, the guarantees (versions and testing among them, a row each), the license, a trimmed *Terminology* that links the glossary, a table of the four model families, and the README's map: a row for each reader task, linking that task's pages of each kind. |
| `llms.txt` | map | 8 | The map a coding agent reads before it writes against the library, rebuilt from the new pages. |
| `docs/README.md` | map | 8 | The index of documents, grouped by the four kinds, then the contributor pages and the records. |
| `README.md`, its opening | tutorial | 1 | Fit a regression over a made-up trading day and read the coefficients it found, in one example. |
| `docs/tutorials/stream-from-parquet.md` | tutorial | 7 | Fit a bank over a parquet file larger than you would hold, in a query, and write its output to a file. |
| `docs/tutorials/local-fit-with-a-clock.md` | tutorial | 7 | Give a regression a clock and a half-life, and follow its coefficients through a day and across a session boundary, beyond the opening's one local fit. |
| `docs/tutorials/fit-save-and-serve.md` | tutorial | 7 | Fit a bank, save it, load it in a fresh process and predict new rows from it. |
| `docs/how-to/resume-from-a-saved-state.md` | how-to | 1 | Load a saved bank and go on learning from the next row, from Python or the command line, including on input that overlaps what it learned. |
| `docs/how-to/fit-on-factor-residuals.md` | how-to | 6 | Fit factor ridges, then feed their out-of-sample residuals (each factor's actual less its estimate) as features to a ridge whose target is a forward-looking exponentially weighted mean. The second ridge runs under a hard window with `half_life=float("inf")`, `embargo` equal to the target's window, and `emit_robust_se` with `horizon_rows`. It needs task 344's refusal from `.online.unnest` on a name collision and its window-aware `half_life` message. |
| `docs/how-to/save-a-state.md` | how-to | 6 | Save a bank's state from a query, a loop or the command line, to load later. |
| `docs/how-to/serve-from-a-saved-state.md` | how-to | 6 | Predict rows from a saved state without learning from them. |
| `docs/how-to/run-a-bank-from-the-command-line.md` | how-to | 6 | Run a bank as a scheduled job with the `polars-online` program and a TOML file, with no Python. |
| `docs/how-to/run-a-bank-in-a-query.md` | how-to | 6 | Add a bank to a Polars query with `lf.online.fit_predict` and choose its output, linking the how-tos for threads and chunk size. |
| `docs/how-to/run-a-bank-in-a-python-loop.md` | how-to | 6 | Feed a `ModelBank` chunk by chunk from your own loop, and read it between chunks. |
| `docs/how-to/hand-the-output-to-arrow.md` | how-to | 6 | Pass a bank's output to a consumer that reads Arrow, such as pyarrow or DuckDB. |
| `docs/how-to/fit-one-model-per-group.md` | how-to | 6 | Fit a separate model for each value of a column, in one pass. |
| `docs/how-to/close-finished-groups.md` | how-to | 6 | Keep memory bounded when new groups never stop appearing, and read each finished group's row. |
| `docs/how-to/weight-rows.md` | how-to | 6 | Make some rows count more than others with a weight column. |
| `docs/how-to/weight-rows-by-a-robust-scale.md` | how-to | 6 | Weight each row by a robust scale of its target, without reading labels not yet known. |
| `docs/how-to/handle-sessions-gaps-and-steps-back.md` | how-to | 6 | Set a spec's clock policy for session boundaries, quiet gaps and a clock that steps back. |
| `docs/how-to/put-rows-in-clock-order.md` | how-to | 1 | Sort a stream by its clock before the bank, and read the bank's warnings about row order. |
| `docs/how-to/learn-targets-known-late.md` | how-to | 6 | Predict each row where it sits and learn from it only once its target is known, with `embargo` or `po.stream.embargo`. |
| `docs/how-to/hold-a-row-back.md` | how-to | 6 | Choose between a null, a zero weight and `predict` to keep a row from teaching the model. |
| `docs/how-to/make-window-features.md` | how-to | 6 | Make features that look back over the stream, as columns made with `po.ops` and `with_windows` before the bank. |
| `docs/how-to/make-a-target-that-looks-ahead.md` | how-to | 6 | Make a target from the rows after each row, as a column or as a formula target the bank holds until its window closes. |
| `docs/how-to/scale-a-feature-by-its-spread.md` | how-to | 6 | Fit a model on a feature's z-score against its own recent spread. |
| `docs/how-to/resume-a-window-run.md` | how-to | 6 | Run `with_windows` over a stream in two runs, saving and loading its state. |
| `docs/how-to/forget-rows-past-a-window.md` | how-to | 6 | Make rows older than a set span count for nothing, with `window_size`. |
| `docs/how-to/fit-locally-along-a-feature.md` | how-to | 6 | Fit any model locally along a feature by sorting on it and using it as the clock. |
| `docs/how-to/align-series-that-tick-at-their-own-times.md` | how-to | 6 | Put series observed at different instants on one grid with `refresh_time` before fitting them together. |
| `docs/how-to/read-a-state-without-this-library.md` | how-to | 6 | Export a saved state as JSON for a program that does not use this library. |
| `docs/how-to/inspect-a-bank.md` | how-to | 6 | List what a bank holds, its specs, groups and row counts, and drop the groups you no longer need. |
| `docs/how-to/find-an-output-field.md` | how-to | 6 | Find the name of the field you want in a spec's output without running it. |
| `docs/how-to/read-the-coefficients.md` | how-to | 6 | Read a fit's coefficients from the bank or row by row from the output. |
| `docs/how-to/repair-and-score-a-correlation-matrix.md` | how-to | 6 | Repair, shrink, summarize and score a correlation matrix with `po.corr`. |
| `docs/how-to/check-data-quality.md` | how-to | 6 | Run `bank.check()` after a pass and act on what it finds. |
| `docs/how-to/choose-a-model.md` | how-to | 6 | Choose a model family by the question you are asking, then a model by what your data need. |
| `docs/how-to/choose-settings-by-hand.md` | how-to | 6 | Choose a grid point yourself from `po.eval.metrics` and `po.eval.compare_specs`, where `grid_search` does not fit the case. |
| `docs/how-to/choose-a-ridge-and-half-life.md` | how-to | 6 | Choose `ewridge`'s ridge, alone or with its half-life, from the stream's own errors. |
| `docs/how-to/choose-a-kalman-drift-speed.md` | how-to | 6 | Choose `kalman`'s `coef_half_life` from the stream's own errors. |
| `docs/how-to/choose-a-learning-rate.md` | how-to | 6 | Choose `sgd`'s `learning_rate` from the stream's own errors. |
| `docs/how-to/choose-holts-smoothing.md` | how-to | 6 | Choose `holt`'s two half-lives from the stream's own errors. |
| `docs/how-to/choose-a-changepoint-hazard.md` | how-to | 6 | Choose `bocpd`'s hazard by the density it gave each row before learning it. |
| `docs/how-to/check-the-scale.md` | how-to | 6 | Find out whether a model's predictions are the right size, with its calibration diagnostic. |
| `docs/how-to/detect-a-break.md` | how-to | 6 | Find out whether a model's relationship has broken, and when. |
| `docs/how-to/check-standard-errors.md` | how-to | 6 | Find out whether a coefficient's t can be trusted, and use robust errors where it cannot. |
| `docs/how-to/find-what-the-model-misses.md` | how-to | 6 | Test a model's residuals for a lag, a spread or a curvature it has not learned. |
| `docs/how-to/measure-heavy-tails.md` | how-to | 6 | Measure how heavy a model's residual tails are before setting a threshold on them. |
| `docs/how-to/find-an-influential-row.md` | how-to | 6 | Find the rows that moved a fit most, with `influence`. |
| `docs/how-to/check-feature-health.md` | how-to | 6 | Find a feature that has gone stale or drifted while every residual looks fine. |
| `docs/how-to/choose-among-a-grid-online.md` | how-to | 6 | Let a spec choose or average its grid points as the stream runs, with `emit_selected` and `emit_averaged`. |
| `docs/how-to/compare-two-models.md` | how-to | 6 | Test whether two models' errors differ, with Diebold and Mariano's test. |
| `docs/how-to/evaluate-an-output-frame.md` | how-to | 6 | Score a finished run's output per group or over time windows, and unpack it to long form, with `po.eval`. |
| `docs/how-to/evaluate-a-stream-too-large-to-hold.md` | how-to | 6 | Score an output too large to hold, by reducing it in a query first. |
| `docs/how-to/decide-when-a-model-is-ready.md` | how-to | 6 | Read the readiness fields and set the gates so a model reports only once it is informed enough. |
| `docs/how-to/put-an-interval-on-a-prediction.md` | how-to | 6 | Give each prediction an interval, Gaussian or conformal. |
| `docs/how-to/test-a-detector-on-known-truth.md` | how-to | 6 | Test a regime or break detector on `po.sim` streams whose truth is known. |
| `docs/how-to/fit-weekly-grams-in-one-pass.md` | how-to | 6 | Leave one Gram per week from a single pass, for fits chosen later. |
| `docs/how-to/pool-grams.md` | how-to | 6 | Pool blocks' Grams into one, aged to a common row, with or without their levels. |
| `docs/how-to/fit-a-ridge-from-a-gram.md` | how-to | 6 | Fit a ridge and its t statistics from a Gram, for a grid of penalties at once. |
| `docs/how-to/check-a-fit-is-determined.md` | how-to | 6 | Find out whether nearly dependent columns leave a fit undetermined. |
| `docs/how-to/choose-columns-on-the-lasso-path.md` | how-to | 6 | Choose columns on the lasso path from a Gram, by least angle regression or coordinate descent. |
| `docs/how-to/select-columns-stably-over-weeks.md` | how-to | 6 | Select columns that hold across halves of the weeks, with a null by sign flips. |
| `docs/how-to/fit-per-block-coefficients.md` | how-to | 6 | Fit each block's own coefficients and read their spread. |
| `docs/how-to/read-grams-from-saved-files.md` | how-to | 6 | Read Grams back from a saved state or a closed-groups file. |
| `docs/how-to/store-a-compact-gram.md` | how-to | 6 | Store a wide Gram in float32 or as a packed triangle. |
| `docs/how-to/update-a-wide-gram-on-threads.md` | how-to | 6 | Spread one wide Gram's update over several threads with `gram_threads`. |
| `docs/how-to/fit-huber-on-a-gram.md` | how-to | 6 | Fit a Huber regression from one shared Gram, a query per pass. |
| `docs/how-to/feed-a-bank-from-a-database.md` | how-to | 6 | Feed a bank from DuckDB or any ADBC database, in clock order. |
| `docs/how-to/run-a-bank-in-pathway.md` | how-to | 6 | Run a bank inside a Pathway pipeline as a stateful operator. |
| `docs/how-to/use-more-cores.md` | how-to | 6 | Give a bank more tasks so it uses more cores. |
| `docs/how-to/set-the-chunk-size.md` | how-to | 6 | Set how many rows a bank takes at a time. |
| `docs/how-to/bound-memory.md` | how-to | 6 | Keep a run's memory down with Polars' own settings and by filtering after the bank. |
| `docs/CALIBRATION.md` | how-to | 6 | Choose a model's settings from the stream's own out-of-sample errors with `po.eval.grid_search`, and report what the choice earned. |
| `docs/GBDT.md` | how-to | 6 | Train XGBoost or LightGBM on files larger than memory with `po.gbdt`. |
| `docs/MODELS.md` | reference | 1 (`ewridge`), 4 | Every model's entry by family, as the rule for model entries sets it (summary, when to use it, links to its parameters, outputs and refusals, update, one example, See also, and what it is checked against in its Notes), and what every spec takes, in an order of its own (Parameters, Raises, Examples). |
| `docs/OUTPUTS.md` | reference | 4 | Every field each model writes, with its type and when it is null, generated from the code. |
| `docs/GLOSSARY.md` | reference | 0 | Every concept's one word, its meaning, the words it replaces and whether a message may use it, and the reason for each contested word. |
| `docs/RUNNER.md` | reference | 4 | The command line, in the order of its sections today: exit status; the configuration file (the flags and the TOML keys they override, input and output files, specs in TOML, clocks that are times); chunk size; the Rust entry points; and what `--version` prints. |
| `docs/STATE-WORKFLOW.md` | reference | 4 | What a saved state carries, what each step of fit, save, serve and resume guarantees, and the evidence for each rule. |
| `docs/DIAGNOSTICS.md` | reference | 4 | One entry per diagnostic, in a fixed order of its own (summary, switch, fields, definition, threshold, false-alarm rate, cost, Notes), opened by a table from each question to its how-to. |
| `docs/DATA-QUALITY.md` | reference | 4 | One entry per finding `bank.check()` can report, at the anchor the check links, in a fixed order of its own: what it means, what it does to each model, and its remedy. |
| `docs/PERFORMANCE.md` | reference | 4 | The measured costs: throughput, parallelism, how cost grows with size, memory and the window operators, with a link to the diagnostics' costs, and how the numbers are made. |
| `docs/REGIMES.md` | reference | 4 | What each regime and correlation model finds on streams whose truth is known, and what it misses. |
| `docs/VALIDATION.md` | reference | 4 | Where each validated default came from, generated by its script. |
| `docs/VERSIONS.md` | reference | 4 | The version promise, the Polars range and pin, NumPy, state files across releases, what the API is, and names that changed. |
| `docs/reference/` | reference | 3 | The Sphinx API reference built from the docstrings, grouped by family with one page per function and model. |
| `docs/explanation/out-of-sample.md` | explanation | 1 | Why every prediction and diagnostic is made before its row is learned, and what that guarantees. |
| `docs/explanation/time-and-decay.md` | explanation | 5 | How a model forgets along a clock, why a huge half-life is not infinity, and how a hard window differs. |
| `docs/explanation/chunk-invariance-and-streaming.md` | explanation | 5 | Why any chunking gives the same numbers, why memory stays bounded, and why row order is the caller's. |
| `docs/explanation/state-and-resuming.md` | explanation | 5 | What a bank's state is, why a resumed run matches one that never stopped, and the one gap. |
| `docs/explanation/warm-up-and-readiness.md` | explanation | 5 | Why a model withholds its first predictions, what each gate measures, and how seeding differs. |
| `docs/explanation/robust-scale.md` | explanation | 5 | How the robust models measure a residual's scale, refit their first rows, and trade breakdown for speed. |
| `docs/explanation/grids-and-instances.md` | explanation | 5 | How one spec fits a grid in one pass, what an instance and a slot are, and why the best error over a grid is optimistic. |
| `docs/explanation/how-diagnostics-work.md` | explanation | 5 | How a diagnostic keeps its own memory, and how to read a statistic read on every row. |
| `docs/GRAM.md` | explanation | 5 | How a Gram stands in for the rows of a least-squares fit, which rows it holds, and what it cannot answer, opened by a table from each task to its how-to. |
| `docs/WINDOWS.md` | explanation | 5 | How a window operator weighs the rows near each row, at its edges and on an irregular clock. |
| `docs/explanation/where-the-time-and-memory-go.md` | explanation | 5 | Where a bank spends its time and memory, and which settings move each. |
| `docs/explanation/asynchronous-series.md` | explanation | 5 | Why series that tick at their own times bias a correlation toward zero, and the two ways out. |
| `docs/explanation/against-scikit-learn.md` | explanation | 5 | How this library's exact solves differ from scikit-learn's streaming gradient steps, and where each wins. |
| `docs/explanation/polars-interfaces.md` | explanation | 5 | The four ways the library crosses into Polars, which carry a promise, why it carries its own Polars, and what read-ahead does. |
| `docs/explanation/how-data-quality-checks-work.md` | explanation | 5 | How `check()` finds problems from what a bank keeps anyway, at no cost to the run, and why each finding harms a fit. |
| `docs/explanation/how-po-gbdt-builds-training-data.md` | explanation | 5 | How `po.gbdt` builds each library's training object: where it places its bins, its two engines, categorical features, and LightGBM's dataset parameters. |
| `docs/explanation/how-the-library-is-tested.md` | explanation | 5 | How each guarantee is held by a test against an oracle or an invariant, for a user deciding whether to trust it. |
| `docs/EXTENDING.md` | contributor how-to | 9 | Add a model: every layer it touches, with the test that catches each omission. |
| `docs/contributing/add-an-output-or-a-parameter.md` | contributor how-to | 9 | Add an output field or a spec parameter, and what each changes in a saved state. |
| `docs/contributing/iterate-on-a-change.md` | contributor how-to | 9 | Run the tests for what a change touches while working, and the full tier before a push. |
| `docs/contributing/use-a-worktree-slot.md` | contributor how-to | 9 | Take, use and give back a warm worktree slot. |
| `docs/contributing/clear-the-caches.md` | contributor how-to | 9 | Clear the build caches, and know when to. |
| `docs/contributing/update-the-oracle-ledger.md` | contributor how-to | 9 | Rerun an oracle grid and record its error, which only tightens. |
| `docs/contributing/cut-a-release.md` | contributor how-to | 9 | Cut a release: rehearse, compare with the last release, dispatch and approve. |
| `docs/contributing/move-the-polars-range.md` | contributor how-to | 9 | Move the Polars range: cap it when a new Polars breaks the library, and raise the ceiling to a new major in five steps. |
| `docs/contributing/change-the-api.md` | contributor how-to | 9 | Change the public API under the stability policy, with the snapshot test. |
| `CONTRIBUTING.md` | contributor how-to | 9 | Build and test from a checkout, and make a change. |
| `docs/contributing/repository-settings.md` | contributor reference | 9 | Every repository setting, with what it is and why. |
| `docs/TESTING.md` | contributor reference | 9 | What the suite proves, the test files that hold each guarantee and how they run, its two tiers, the oracle ledger, the performance budgets, where it runs, the machines and their last bits, and where it is thin. |
| `docs/RELEASE-READINESS.md` | contributor reference | 9 | What a release is held to: the release gate, the README's links and size row on PyPI, the budgets baseline and the tests that hold the workflow. |
| `docs/WRITING.md` | contributor reference | 0, 9 | The rules every reader-facing page, docstring and message is held to. |
| `docs/records/PERFORMANCE-LOG.md` | record | 4 | Every dated measurement, with its section number kept, since code comments and the records cite them. |
| `docs/records/TESTING-LOG.md` | record | 9 | What the suite found and when, and the test entries by ID. |
| `docs/records/DOCS-REWRITE-MAP.md` | record | 0 | This map: every section of the README and the guides, where it lands, and the pages the rewrite produces. |
| `docs/records/RELEASE-HISTORY.md` | record | 9 | How the release workflow, the Polars range and the public repository came to be as they are. |
| `docs/records/STATE-WORKFLOW-RESEARCH.md` | record | 4 | The research of 2026-09-03 that decided the state workflow, moved whole. |

## Where the guides live

The user decided: "Guides stay in markdown unless a compelling reason."
They stay Markdown, read on GitHub, beside a Sphinx site that holds the
API reference alone. The evidence weighed, measured at `965e57c0`:

| what was weighed | Markdown on GitHub, as now | inside the Sphinx site (MyST or RST) |
|---|---|---|
| the tests that run the guides' code | 55 test and script files read the guides by path, and `tests/doc_recipes.py` holds each recipe to its printed output by reading Markdown fences | each of those tests would read MyST or RST instead, and a recipe's output block would need a directive of its own |
| GitHub's rendering, which `scripts/doc_structure.py` checks | what a reader on GitHub sees is what the checks test | a reader browsing the repository would see the source, with its directives unrendered |
| PyPI's README | `scripts/pypi_readme.py` rewrites the README's relative links at release; unchanged | the same; the README stays Markdown either way |
| links from shipped docstrings | `blob/main/docs/*.md` addresses, which every released wheel carries | every one of those links would break in every installed version |
| cross-links between a guide and the reference | a guide links the reference's HTML pages by URL, and a docstring links a guide by URL; `tests/test_api_links.py` resolves both | Sphinx resolves them by name and fails `-W` on a broken one, a real gain |
| search | GitHub's search over the repository | Sphinx's search over guides and reference together, the other gain |
| the cost of moving | none | a new Sphinx extension (MyST) or every guide rewritten in RST, the 55 test and script files that read the guides moved with them, and a published site whose URLs replace the GitHub ones |

No compelling reason to move was found. Two gains are real: checked
cross-links, and one search over guides and reference. The link check
the tests already run gives the first of them.

## Rules for moving text

Each stage moves sections by these rules, so the map's last column can be
followed without a decision at each row:

- **The destination's stage copies the text, and the source page's stage
  removes it** and leaves a one-line link. Where the two are the same
  stage, the text moves. Where the destination's stage comes after
  the source page's, the source page's stage leaves the text as it
  stands, and the destination's stage moves it. Where the destination's
  stage passed without copying the text, as stage 0's glossary did for
  PERFORMANCE.md's words, the source page's stage moves it. A README section follows the rule *A README section moves* instead.
  A record moves out in the stage that rewrites the page holding it.
- **A README section moves, and is not copied,** in the stage that writes
  its destination, in all but five cases. *Terminology* stays until stage
  8, which trims it to the words a newcomer needs and a link to the
  glossary, since the glossary already holds every term. *Development*
  stays until stage 8 moves its text as it stands into CONTRIBUTING.md,
  which stage 9 rewrites. The headings
  `llms.txt` links stay until stage 8 too. And a README section that leads to an existing guide stays, with
  its recipe, until stage 8, since `test_every_guide_has_a_readme_section`
  holds that guide to it until then. This covers the section's
  own text and recipe, not its subsections. The stage that writes its
  destination copies its text there, and stage 8 removes it from the
  README. The fifth case is a part whose destination stage 0 wrote, the
  glossary: it leaves the README in the last stage that moves the rest of
  its section. Where a README section moves whole in a middle stage, its
  one-line link stays where the section stood, under its parent's opener,
  until stage 8.
- **A stage points every link to a heading at the text's new place
  before it removes the heading:** the README's, `llms.txt`'s, the
  guides', the tests' and the scripts' links alike (WRITING §2, a
  heading's text is its anchor, and §6 step 5).
- **A heading's address matters only where something links it.** Where
  shipped code or a message links a heading, it stays on its old page,
  over a one-line link to each part that has moved. Any other heading
  stays on its old page, over what remains and a one-line link to each
  part that has gone, until its last part has moved, and is then
  removed. The text moves under the destination's own headings, and a
  section that becomes a page of its own takes a title of its kind. A
  page's title stays on its page whatever its section's row says,
  retitled to what the page now is, and only the text under it moves.
- **A model's entry in MODELS.md follows one rule.** Stage 1 writes
  `ewridge`'s entry beside its docstring to test the rule first, and
  corrects the rule from what it finds. The entry follows WRITING §10's
  order. It keeps the model's summary and when to use it. It links its
  parameters and outputs in the docstring and OUTPUTS.md rather than
  repeating them. It keeps its update rule, in the notation the entries
  share, and its first recipe as its one example. It links its refusals
  to the docstring's Raises, gives its See also, and its Notes say what it
  is checked against. Every recipe after the first goes to the how-to its
  row names. Where its row names none, the stage writing the entry sends
  the recipe to a how-to in the page list that covers the task, or else
  names a new how-to for it by the task it solves. That stage is stage 1
  for `ewridge`, and stage 4 for the rest. That stage records each recipe's how-to, listed or new,
  in PLAN 338. The
  page list leaves the new how-tos out, and stage 6 writes them with the
  rest. Until then, a rewritten entry keeps those recipes, and every
  other part bound for a later stage's page, as they stand, under a
  heading of their own before its See also and Notes. Its mechanism goes to
  its docstring's Notes, unless its row names an explanation for it.
- **A page links what exists, and the stage that writes the target
  repoints the link.** Before stage 6, for example, the tables that open
  DIAGNOSTICS.md and GRAM.md link the sections still on their pages, and
  the how-to stage 1 writes on putting rows in clock order links the
  README's *Sessions, gaps and steps back*. Stage 6 points each at its new
  how-to.
- **The stage that moves a record rewrites the citations to it** in code
  comments, tests and other records, as `PERFORMANCE.md §N` becomes the
  log's path with the same section number. It points every guide's link
  to the record at the reference page that remains instead, since a guide
  cites no record (WRITING §4).
- **A stage records each page's plan before drafting it:** the page's
  opener and the subsections its kinds become, with `←` marking what moved
  (WRITING §2 and §6, step 2), in PLAN 338, and for a README pass in the
  log of README passes.
- **A test moves with the text it reads.** The page tests
  (`tests/test_*_page.py`), the README's recipe tests and
  `tests/test_api_links.py` follow a section to its new page in the same
  change.
- **Every new page except a record joins `docs/README.md`'s Guides table
  and `tests/test_no_plan_references.py`** the day it is created, as
  `test_every_guide_is_held` already requires. A record joins the index's
  Records table, and may cite tasks.
- **A reference page whose entries are not functions keeps one fixed
  order of its own,** since WRITING §10's order is for functions and
  models. DIAGNOSTICS.md's and DATA-QUALITY.md's orders are in their rows
  of the page list. Each starts, as §10 does, with what the entry is, and
  ends with its Notes or its remedy.
- **The README's rule that every guide has a README section with a
  recipe** (WRITING §2, held by `test_every_guide_has_a_readme_section`,
  from the user, 2026-10-10) cannot hold for 68 how-tos. So until stage 8,
  each page the rewrite creates goes under a new bold group of
  `docs/README.md`'s Guides table, *New pages*, which that test does not
  read: it reads only *Using the library*, *Changing the library* and
  *Writing the documentation*. The pages are still held to the rule
  against citing the plan, since `test_every_guide_is_held` reads the
  whole table. Stage 1 adds the group, and corrects WRITING §2, which says
  the test holds every guide. `docs/GLOSSARY.md`, which stage 0 wrote,
  stays under *Writing the documentation*, where the README links it. Stage 8 replaces the rule, in WRITING §2 and in the
  test, with the README's map: a row for each reader task, linking that
  task's pages of each kind. Both the interim exception and the
  replacement change a rule of the user's, so the coordinator brings the
  exception to the user before stage 1 starts, and the replacement before
  stage 8 starts.
- **The Sphinx reference's split into one page per entry** (stage 3)
  changes the `hgilde.github.io/polars-online/<module>.html#...`
  addresses, which the README links 36 times (the docstrings link
  entries by name, which Sphinx resolves). The README of every release
  already on PyPI keeps those addresses. So stage 3 keeps each module's
  page, and at each old address puts a raw HTML anchor with the exact old
  id, such as `polars_online.ModelBank.fit_predict`, beside a link to the
  entry's new page. A heading would not do: its id is a slug of its text,
  and a second description of the same object fails Sphinx's `-W`.

## Lines by kind, as they stand

How the 31,600 lines of the README and the guides divide by kind today,
counting each section's own lines (its heading to its first subsection).
A *mixed* section holds more than one kind in parts of some size, and its
row in the map says where each part goes. Its lines are counted under
*mixed*. A section of one kind with a
few sentences of another, an aside, is counted under its main kind, and
its row still says where the aside goes. The records are the dated research and logs that
the guides hold today.

| page | lines | sections | tutorial | how-to | reference | explanation | mixed | record |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| README.md | 4492 | 73 | 159 | 1669 | 479 | 457 | 1728 |  |
| docs/MODELS.md | 3916 | 26 |  |  | 79 |  | 3837 |  |
| docs/WINDOWS.md | 774 | 7 |  | 419 |  |  | 355 |  |
| docs/RUNNER.md | 570 | 19 |  | 123 | 350 | 18 | 79 |  |
| docs/STATE-WORKFLOW.md | 788 | 20 |  | 75 | 119 | 42 | 155 | 397 |
| docs/DIAGNOSTICS.md | 2344 | 25 |  | 340 | 130 | 80 | 1794 |  |
| docs/CALIBRATION.md | 588 | 12 |  | 447 | 13 |  | 128 |  |
| docs/DATA-QUALITY.md | 1650 | 14 |  |  | 23 | 74 | 1553 |  |
| docs/GRAM.md | 2070 | 21 |  | 1778 | 31 | 96 | 165 |  |
| docs/GBDT.md | 347 | 8 |  |  |  | 225 | 122 |  |
| docs/OUTPUTS.md | 459 | 31 |  |  | 459 |  |  |  |
| docs/REGIMES.md | 722 | 13 |  |  | 662 |  | 60 |  |
| docs/VALIDATION.md | 119 | 8 |  |  | 119 |  |  |  |
| docs/PERFORMANCE.md | 6416 | 200 |  |  | 510 | 160 | 56 | 5690 |
| docs/EXTENDING.md | 778 | 33 |  | 778 |  |  |  |  |
| docs/TESTING.md | 2587 | 46 |  | 272 | 690 |  | 490 | 1135 |
| docs/RELEASE-READINESS.md | 1983 | 59 |  | 212 | 751 | 44 | 168 | 808 |
| docs/WRITING.md | 986 | 17 |  | 81 | 828 | 77 |  |  |
| all | 31589 | 632 | 159 | 6194 | 5243 | 1273 | 10690 | 8030 |

## The stage 1 set

Stage 1 is no longer held for a verdict, but its pieces are the first of
their kinds, and every later stage copies their shape. So each is chosen
as the case that shows most of its kind's rules, or as a page another
piece needs. PLAN 338 named six pieces. The map keeps them and adds
four, each for a shape the six do not show or a page one of them needs.

WRITING §10 and §11 tell the faults of the stage 1 pieces in the past tense, as the cases
their rules came from. The faults still stand in the code and the pages
until stages 1 and 2 fix them.

| piece | kind | why it represents its kind |
|---|---|---|
| the clock's step-back error (`crates/online-polars`) | Rust error | the longest message (over a hundred words), with three remedies, an internal setting and a row number: every part of the four-part shape is tested on it |
| `spec "m": ridge must be finite and >= 0, got -1`, raised when a spec is built | Python error | **added.** The package has 354 raise sites in Python, most of them spec checks at build time, beside about 1,090 error returns in Rust, and none of the six pieces is a Python refusal. This one comes from the range check every numeric keyword shares, so its shape reaches every such refusal at once, and its rewritten form is WRITING §11's example |
| the readiness warning | warning | it holds every fault of a warning at once: the internal words *settled* and `lam`, a formula, an unrounded number, and `group ""` for a spec with no groups |
| `ewridge`'s docstring | model docstring | the longest and most used model, whose mechanism (the solve, the condition check) sits before its parameters today |
| `ModelBank.fit_predict`'s docstring | function docstring | **added.** Most of the reference's entries are functions, whose sections (Parameters, Returns, Raises) differ from a model's, and this is the call every reader makes first |
| `ewridge`'s entry in MODELS.md | reference entry | **added.** Stage 4 rewrites the other 21 entries against §10's order, linking the docstring rather than repeating it. This entry, beside its docstring, sets where the two divide |
| how-to: resume a saved bank (`docs/how-to/resume-from-a-saved-state.md`) | how-to | it has a prerequisite (a saved state, whose steps STATE-WORKFLOW.md gives until stage 6 writes `docs/how-to/save-a-state.md`), steps from both Python and the command line, an edge case (input that overlaps the state) and a check (the output equals an unbroken run's) |
| how-to: put rows in clock order (`docs/how-to/put-rows-in-clock-order.md`) | how-to | **added.** It is the long form the step-back error links, so the error cannot take its final shape without it. Its task, sorting before the bank and reading the bank's warnings, is the one most readers meet first |
| explanation: why every prediction is out of sample (`docs/explanation/out-of-sample.md`) | explanation | the library's central guarantee, short enough to finish in the first slice, and cited by nearly every other page |
| the README's opening (its title section, *Introduction*'s opener and *A first fit*), as the first tutorial | tutorial | the page read hardest, and the one the four kinds of page are reached from |

## The map, section by section

Each table lists a page's sections in order, with each section's own lines, its kind
now, and where it lands. The kinds are tutorial, how-to, reference,
explanation and record, and a *mixed* row names its parts. A row that
says *with its parent* moves with the section that contains it, named
in the row, as one piece. Where that parent splits, the child goes where
the part of the parent it belongs to goes. A record moves
whole to `docs/records/` and is not rewritten beyond stage 10's
spelling. Where a row names a
page's section after a colon, as *MODELS.md: What every spec takes*, the
words describe the section and are not its heading. A heading that
shipped code, a message or `llms.txt` links keeps its wording, so its
anchor holds. Any other heading may be reworded once its stage has
pointed every link to it at the new wording (WRITING §2, a heading's text
is its anchor), as the `huber` / `quantile` split is. Stage 0 renamed one
this way, WRITING §11's *Errors, warnings and notices* to *Errors and
warnings*, after checking that only WRITING's own contents table linked
it. [Rules for moving text](#rules-for-moving-text) says how titles
change.


### README.md

73 sections, 4492 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | polars-online | 28 | mixed: reference (badges, the links table), explanation (what it is) | README landing: what the library is, with its badges; part of the README's opening, which stage 1 rewrites. The links table stays until stage 8 replaces it with the README's map |
| 29 | · Introduction | 4 | explanation (opener) | README landing; part of the README's opening, which stage 1 rewrites; its heading stays until stage 8, since `llms.txt` links it |
| 33 | · · A first fit | 159 | tutorial, with explanation asides on the local fit | README landing, the first tutorial, the last part of the README's opening (stage 1). The example keeps its local fit and its look-ahead target. Stage 1 moves the aside on the look-ahead target straight to explanation/out-of-sample.md, which it writes, and the aside on the local fit into *Time and decay*, which carries it on to explanation/time-and-decay.md in stage 5. Stage 7 writes tutorials/local-fit-with-a-clock.md new, starting from this example's local-fit step |
| 192 | · · The idea | 35 | explanation (a summary of the ideas) | README landing, as its summary, each point linking its page: predicting before learning to explanation/out-of-sample.md, a stream larger than memory to explanation/chunk-invariance-and-streaming.md, row order to explanation/chunk-invariance-and-streaming.md, windows to WINDOWS.md |
| 227 | · · Terminology | 14 | reference | GLOSSARY.md, which stage 0 wrote and the README already links; the section stays until stage 8 trims it to the words a newcomer needs and a link to the glossary |
| 241 | · · What you can rely on | 17 | reference (the guarantees) | README landing, one row each; the depth of each to the page that holds it: out-of-sample predictions to explanation/out-of-sample.md, any chunking to explanation/chunk-invariance-and-streaming.md, bounded memory to explanation/chunk-invariance-and-streaming.md, thread counts to explanation/where-the-time-and-memory-go.md, unusable values and zero weights to how-to/hold-a-row-back.md, `weight_sum` to explanation/warm-up-and-readiness.md, the clock policy to explanation/time-and-decay.md, named mistakes and testing to explanation/how-the-library-is-tested.md |
| 258 | · · Install | 25 | how-to | README landing: Install |
| 283 | · · Example data | 70 | reference (builds the two example frames) | README landing: Example data stays where WRITING §3 and its tests put it, and a how-to that reads `df` or `trades` names it as a prerequisite and links it |
| 353 | · How a bank sees a stream | 22 | mixed: explanation (opener), reference (table of shared parameters) | MODELS.md: What every spec takes (new section, in an order of its own: Parameters, Raises, Examples), whose opener its opener becomes; its heading stays until stage 8, since `llms.txt` links it |
| 375 | · · What a spec names | 60 | mixed: reference (what a spec names), how-to (two refusals shown) | MODELS.md: What every spec takes; its refusals become that section's Examples |
| 435 | · · · Relative and look-ahead targets | 133 | how-to | how-to/make-a-target-that-looks-ahead.md; its reasons, and those of the first fit's aside on the look-ahead target, to explanation/out-of-sample.md, which says why a target that looks ahead is learned only once it is known; that part moves in stage 1, which writes the explanation |
| 568 | · · Time and decay | 9 | explanation | explanation/time-and-decay.md |
| 577 | · · · Clock types and units | 101 | mixed: reference (clock types and units), explanation | MODELS.md: What every spec takes (clock rows); the reasons to explanation/time-and-decay.md |
| 678 | · · · Sessions, gaps and steps back | 65 | mixed: how-to, reference (the clock policy table) | how-to/handle-sessions-gaps-and-steps-back.md; its table to MODELS.md: What every spec takes |
| 743 | · · Convergence without a decay | 50 | explanation | explanation/time-and-decay.md: without a decay |
| 793 | · · A hard window | 136 | mixed: how-to, explanation, reference (caveats table) | how-to/forget-rows-past-a-window.md; why to explanation/time-and-decay.md; the caveats table to MODELS.md: What every spec takes, under `window_size` |
| 929 | · · A local fit along any feature | 70 | how-to | how-to/fit-locally-along-a-feature.md |
| 999 | · · Row order and the two guarantees | 58 | explanation | explanation/chunk-invariance-and-streaming.md, with its chunking paragraph and the rest; its out-of-sample paragraph to explanation/out-of-sample.md |
| 1057 | · · · How a bank detects rows out of order | 130 | mixed: explanation (the three checks), how-to | how-to/put-rows-in-clock-order.md; the checks to explanation/chunk-invariance-and-streaming.md; its text on the remedy `restart_after_step_back` merged into how-to/handle-sessions-gaps-and-steps-back.md, which this how-to links, with WRITING §6 step 4's account checking that nothing is lost |
| 1187 | · · Groups | 55 | how-to | how-to/fit-one-model-per-group.md |
| 1242 | · · Weights | 55 | how-to | how-to/weight-rows.md |
| 1297 | · · Warm-up | 111 | mixed: explanation, reference (the gates table) | explanation/warm-up-and-readiness.md; the gates to MODELS.md: What every spec takes; the readiness fields to OUTPUTS.md, through its generator's text (`scripts/outputs_doc.py`) |
| 1408 | · · Labels that arrive late | 11 | explanation (opener) | how-to/learn-targets-known-late.md |
| 1419 | · · · The built-in delay | 46 | how-to | how-to/learn-targets-known-late.md: the built-in delay |
| 1465 | · · · The delay as data | 55 | how-to | how-to/learn-targets-known-late.md: the delay as data |
| 1520 | · · Nulls, and three ways to hold a row back | 73 | mixed: reference (three ways compared), how-to | how-to/hold-a-row-back.md; what a null is to GLOSSARY.md (stage 0 copied it; stage 6 removes it from the README); the rule for a null row's fields to OUTPUTS.md: when a field is null, through its generator's text |
| 1593 | · Preparing a stream | 12 | explanation (opener) | README landing: the map row for preparing a stream |
| 1605 | · · Making features and targets from windows | 58 | how-to | stays in the README, with its recipe, until stage 8, as the section that leads to WINDOWS.md; then how-to/make-window-features.md; its target that looks ahead to how-to/make-a-target-that-looks-ahead.md |
| 1663 | · · Series that tick at their own times | 134 | mixed: how-to, explanation (the Epps effect) | how-to/align-series-that-tick-at-their-own-times.md; the Epps effect to explanation/asynchronous-series.md |
| 1797 | · Running a bank | 13 | reference (opener, three ways) | README landing: the map row for running a bank |
| 1810 | · · As a query: `lf.online.fit_predict` | 137 | mixed: tutorial (the path), how-to (its options) | tutorials/stream-from-parquet.md; the options to how-to/run-a-bank-in-a-query.md |
| 1947 | · · In a loop: `ModelBank` | 68 | how-to | how-to/run-a-bank-in-a-python-loop.md |
| 2015 | · · Running a bank as a job, without Python | 60 | how-to | stays in the README, with its recipe, until stage 8, as the section that leads to RUNNER.md; then how-to/run-a-bank-from-the-command-line.md |
| 2075 | · · Output as Arrow | 46 | how-to | how-to/hand-the-output-to-arrow.md |
| 2121 | · Saving, loading and serving | 55 | explanation (opener) | stays in the README, with its recipe, until stage 8, as the section that leads to STATE-WORKFLOW.md; then explanation/state-and-resuming.md, and its recipe (fit, save, load, serve) to tutorials/fit-save-and-serve.md. *Save and load*'s loading and resuming steps move in stage 1, and the rest in stages 5 and 6; *Save and load*'s own row gives where |
| 2176 | · · Save and load | 110 | mixed: how-to, explanation (what the file guarantees) | how-to/resume-from-a-saved-state.md (stage 1), its loading and resuming steps; its saving steps to how-to/save-a-state.md; the guarantees to explanation/state-and-resuming.md |
| 2286 | · · Serving without learning | 46 | how-to | how-to/serve-from-a-saved-state.md |
| 2332 | · · Reading a state without this library | 31 | how-to | how-to/read-a-state-without-this-library.md |
| 2363 | · Reading the fit | 16 | reference (opener) | README landing: the map row for reading a fit |
| 2379 | · · What a bank holds | 147 | mixed: reference (the methods), how-to | how-to/inspect-a-bank.md; the methods to the API reference's `ModelBank` entries |
| 2526 | · · Output field names | 77 | mixed: how-to, reference (two tables) | stays in the README, with its recipe, until stage 8, as the section that leads to OUTPUTS.md; then how-to/find-an-output-field.md; the tables to OUTPUTS.md, through its generator's text |
| 2603 | · · Coefficients | 123 | how-to | how-to/read-the-coefficients.md |
| 2726 | · · Fitting many features from one pass | 73 | how-to | stays in the README, with its recipe, until stage 8, as the section that leads to GRAM.md; then how-to/fit-weekly-grams-in-one-pass.md; the README keeps a map row linking GRAM.md |
| 2799 | · · One row per finished group | 86 | how-to | how-to/close-finished-groups.md |
| 2885 | · · Reading a correlation matrix | 57 | how-to | how-to/repair-and-score-a-correlation-matrix.md |
| 2942 | · Checking the data and the model | 13 | explanation (opener) | README landing: the map row for checking |
| 2955 | · · Checking your data's quality | 76 | how-to | stays in the README, with its recipe, until stage 8, as the section that leads to DATA-QUALITY.md; then how-to/check-data-quality.md |
| 3031 | · · Checking that a model works | 74 | how-to (an index of the questions) | stays in the README, with its recipe, until stage 8, as the section that leads to DIAGNOSTICS.md; then DIAGNOSTICS.md opener; each question to its how-to; its recipe (a break test) to how-to/detect-a-break.md |
| 3105 | · Models | 52 | mixed: reference (the model table), how-to (choosing a family) | the full per-model table to MODELS.md's opener; choosing a family to how-to/choose-a-model.md; the README keeps a table of the four families, one row each, linking each family's section of MODELS.md |
| 3157 | · · Choosing a model | 106 | how-to | stays in the README, with its recipe, until stage 8, as the section that leads to MODELS.md; then how-to/choose-a-model.md |
| 3263 | · · Detecting changes and regimes | 78 | mixed: explanation, how-to | stays in the README, with its recipe, until stage 8, as the section that leads to REGIMES.md; then how-to/choose-a-model.md, with the question each of the four answers and its recipe; one line on what the four share to MODELS.md's opener of Sequential tests and regimes; the measurements to REGIMES.md |
| 3341 | · · Calibrating a model | 40 | how-to | stays in the README, with its recipe, until stage 8, as the section that leads to CALIBRATION.md; then CALIBRATION.md, the how-to it already leads to |
| 3381 | · Performance | 60 | explanation | stays in the README, with its recipe, until stage 8, as the section that leads to PERFORMANCE.md; then explanation/where-the-time-and-memory-go.md; its timing recipe to PERFORMANCE.md: how the numbers are made, as its example |
| 3441 | · · Throughput | 104 | reference (measurements) | PERFORMANCE.md: throughput |
| 3545 | · · Parallelism | 185 | mixed: how-to, reference (figures) | how-to/use-more-cores.md; the figures to PERFORMANCE.md |
| 3730 | · · Chunk size | 51 | how-to | how-to/set-the-chunk-size.md |
| 3781 | · · Memory | 82 | mixed: explanation, reference (tables) | explanation/where-the-time-and-memory-go.md; the tables to PERFORMANCE.md |
| 3863 | · · Tuning memory with Polars' own settings | 70 | how-to | how-to/bound-memory.md |
| 3933 | · · Window operators | 45 | reference | PERFORMANCE.md: window operators |
| 3978 | · · Against scikit-learn | 65 | explanation | explanation/against-scikit-learn.md; its heading stays until stage 8, since `llms.txt` links it |
| 4043 | · Scope and integrations | 11 | explanation (opener) | README landing: what the library is not |
| 4054 | · · What this is not | 24 | explanation | its first paragraph, the rule to sort by the clock, to the README landing; its reasons and its table to explanation/chunk-invariance-and-streaming.md |
| 4078 | · · Databases: DuckDB and ADBC | 45 | how-to | how-to/feed-a-bank-from-a-database.md |
| 4123 | · · Pathway | 54 | how-to | how-to/run-a-bank-in-pathway.md |
| 4177 | · · Training XGBoost and LightGBM on files larger than memory | 51 | how-to | stays in the README, with its recipe, until stage 8, as the section that leads to GBDT.md; then GBDT.md, the how-to it already leads to |
| 4228 | · Versions, testing and development | 8 | reference (opener) | README landing: versions, one row of the guarantees |
| 4236 | · · Versioning and the Polars pin | 7 | reference | VERSIONS.md |
| 4243 | · · · This package's own versioning | 150 | reference | with its parent: Versioning and the Polars pin |
| 4393 | · · · What is pinned | 31 | reference | with its parent: Versioning and the Polars pin |
| 4424 | · · · Which interfaces carry a promise | 23 | explanation | explanation/polars-interfaces.md |
| 4447 | · · Testing | 27 | explanation | README landing (one row) and explanation/how-the-library-is-tested.md |
| 4474 | · · Development | 15 | how-to | README landing, as a one-line link; stage 8 moves its text into CONTRIBUTING.md as it stands, and stage 9 rewrites it there |
| 4489 | · · License | 4 | reference | README landing: license |

### docs/MODELS.md

26 sections, 3916 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Models | 30 | reference (notation, accumulators) | MODELS.md opener: the notation stays; the kinds of running sums to explanation/time-and-decay.md |
| 31 | · Linear models | 29 | reference (family opener, shared parameters) | MODELS.md: Linear models opener |
| 60 | · · `ewridge` — EW ridge on sufficient statistics | 267 | mixed: reference (API line, when to use it, parameters), how-to (recipes), explanation (the solve, the condition check) | MODELS.md entry in §10's order (stage 1 sets its shape), by the rule for model entries; the reason a grid costs one pass to explanation/grids-and-instances.md |
| 327 | · · `rls` — recursive least squares | 105 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries) |
| 432 | · · `lasso` — lasso path, its penalty chosen as it runs | 108 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries) |
| 540 | · · `kalman` — random-walk-β dynamic linear model | 208 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries); choosing `coef_half_life` to how-to/choose-a-kalman-drift-speed.md |
| 748 | · · `huber` / `quantile` — robust regression | 493 | mixed: reference, how-to, explanation (the scale, the refit) | MODELS.md entries for huber and quantile (by the rule for model entries), each under a heading of its own, as `huber` — robust regression and `quantile` — robust regression, after stage 4 points the old anchor's links at the `huber` heading (in the README, GRAM.md, RELEASE-READINESS.md, OUTPUTS.md through `scripts/outputs_doc.py`, and MODELS.md itself); `tests/test_api_links.py` and `tests/test_model_registry.py`, which find model sections by level, move with the split; the scale and the refit to explanation/robust-scale.md |
| 1241 | · · `sgd` — stochastic gradient descent | 334 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries); the learning rate to how-to/choose-a-learning-rate.md |
| 1575 | · · `pa` — passive-aggressive regression | 141 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries) |
| 1716 | · · `ftrl` — online logistic regression | 147 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries) |
| 1863 | · · `holt` — Holt's linear trend | 139 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries); the smoothing to how-to/choose-holts-smoothing.md |
| 2002 | · Moments and correlation | 7 | reference (family opener) | MODELS.md: Moments and correlation opener |
| 2009 | · · `ew_cov` — exponentially weighted moments | 154 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries) |
| 2163 | · · `marginal` — every pair's moments, kept in the state | 392 | mixed: reference, how-to (five recipes), explanation (lags, bins, shards) | MODELS.md entry (by the rule for model entries); its cost at width to PERFORMANCE.md |
| 2555 | · · `deco` — one correlation for the whole matrix | 102 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries); its measured behavior to REGIMES.md |
| 2657 | · · `rcov` — a block's realized covariance, robust to noise | 144 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries); its measured behavior to REGIMES.md |
| 2801 | · · `audit` — what a stream's columns hold | 90 | mixed: reference, how-to | MODELS.md entry (by the rule for model entries); reading what it counted to how-to/check-data-quality.md |
| 2891 | · Clustering and classification | 7 | reference (family opener) | MODELS.md: Clustering and classification opener |
| 2898 | · · `kmeans` — exponentially weighted k-means | 140 | mixed: reference, how-to, explanation (seeding) | MODELS.md entry (by the rule for model entries); seeding to explanation/warm-up-and-readiness.md |
| 3038 | · · `micro` — density-based clustering, any shape | 130 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries) |
| 3168 | · · `ew_class` — Gaussian classification on `ew_cov` moments | 102 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries) |
| 3270 | · Sequential tests and regimes | 6 | reference (family opener) | MODELS.md: Sequential tests and regimes opener |
| 3276 | · · `seqtest` — a sequential test of a sign, by betting | 121 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries) |
| 3397 | · · `corrchange` — has the correlation structure changed? | 199 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries); measured behavior to REGIMES.md |
| 3596 | · · `bocpd` — how long has this regime lasted? | 181 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries); the hazard to how-to/choose-a-changepoint-hazard.md |
| 3777 | · · `hmm` — which regime are we in | 140 | mixed: reference, how-to, explanation | MODELS.md entry (by the rule for model entries); measured behavior to REGIMES.md |

### docs/WINDOWS.md

7 sections, 774 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Windowed means, looking back or ahead | 265 | mixed: explanation (how a window weighs its rows), reference (the operators), how-to (a first column) | WINDOWS.md, as an explanation (the path stays); each operator's parameters to its API reference entry; the first column to how-to/make-window-features.md |
| 266 | · Windows as columns | 98 | how-to | how-to/make-window-features.md |
| 364 | · Windows as a model's inputs and target | 92 | how-to | how-to/make-a-target-that-looks-ahead.md, with its target; its features that look back to how-to/make-window-features.md |
| 456 | · Features in units of their spread | 74 | how-to | how-to/scale-a-feature-by-its-spread.md |
| 530 | · Quantiles and a robust scale | 90 | mixed: reference, explanation, how-to | WINDOWS.md, with how `po.ewm_quantile` weighs its rows; its parameters to the API reference's `po.ewm_quantile` entry; its recipe to how-to/weight-rows-by-a-robust-scale.md |
| 620 | · · A scale of a target, used as a weight | 68 | how-to | how-to/weight-rows-by-a-robust-scale.md |
| 688 | · Saving and resuming a window run | 87 | how-to | how-to/resume-a-window-run.md |

### docs/RUNNER.md

19 sections, 570 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Running a bank as a job: the `polars-online` command line | 32 | reference (opener) | RUNNER.md, the command line's reference (the path stays) |
| 33 | · Running it | 6 | how-to (opener) | how-to/run-a-bank-from-the-command-line.md |
| 39 | · · A first run | 74 | mixed: tutorial, how-to | how-to/run-a-bank-from-the-command-line.md, whose steps its first run becomes |
| 113 | · · Saving, resuming and scoring | 67 | how-to | how-to/run-a-bank-from-the-command-line.md; saving to how-to/save-a-state.md and resuming to how-to/resume-from-a-saved-state.md (their command-line steps) |
| 180 | · · A run whose product is its state | 20 | how-to | how-to/run-a-bank-from-the-command-line.md |
| 200 | · · Closed groups to a sidecar file | 30 | how-to | how-to/close-finished-groups.md (its command-line steps) |
| 230 | · · Exit status | 31 | reference | RUNNER.md: exit status |
| 261 | · The configuration file | 6 | reference | RUNNER.md: the configuration file |
| 267 | · · Flags and the TOML keys they override | 25 | reference | with its parent: The configuration file |
| 292 | · · Input and output files | 69 | reference | with its parent: The configuration file |
| 361 | · · Specs in TOML | 74 | reference | with its parent: The configuration file |
| 435 | · · Clocks that are times | 24 | reference | with its parent: The configuration file |
| 459 | · Memory, threads and chunk size | 5 | mixed: reference, explanation | RUNNER.md; the pipeline's reasons to explanation/where-the-time-and-memory-go.md |
| 464 | · · Memory | 41 | reference (measurements) | PERFORMANCE.md: the command line's memory |
| 505 | · · The pipeline and its threads | 18 | explanation | explanation/where-the-time-and-memory-go.md |
| 523 | · · Chunk size | 20 | reference | RUNNER.md: chunk size |
| 543 | · From Rust, and the Polars it needs | 6 | reference | RUNNER.md: from Rust |
| 549 | · · The same pipeline from Rust | 8 | reference | with its parent: From Rust, and the Polars it needs |
| 557 | · · Versioning | 14 | reference | RUNNER.md (what `--version` prints); VERSIONS.md (the Rust Polars the program carries) |

### docs/STATE-WORKFLOW.md

20 sections, 788 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | The state workflow: fit, save, serve, learn on | 14 | explanation (opener) | the title stays, retitled as the saved state's reference (the path stays); the opener's text to explanation/state-and-resuming.md |
| 15 | · The workflow in four steps | 19 | reference (opener) | STATE-WORKFLOW.md, the saved state's reference |
| 34 | · · Each step, and what it guarantees | 13 | reference | STATE-WORKFLOW.md: each step and its guarantee |
| 47 | · · Fit, and keep the state | 115 | mixed: tutorial, how-to | tutorials/fit-save-and-serve.md; its steps for keeping the state of a query to how-to/save-a-state.md |
| 162 | · · The one gap: a failure after the bank | 28 | explanation | explanation/state-and-resuming.md: the one gap |
| 190 | · · Resuming on input that overlaps the state | 75 | how-to | how-to/resume-from-a-saved-state.md: input that overlaps the state |
| 265 | · · What a saved bank carries | 65 | reference | STATE-WORKFLOW.md: what a saved bank carries |
| 330 | · · A window run's state | 40 | mixed: reference, how-to | STATE-WORKFLOW.md; the steps to how-to/resume-a-window-run.md |
| 370 | · · The rules, and the evidence for each | 22 | reference | STATE-WORKFLOW.md: the rules and their evidence |
| 392 | · The research behind it, 2026-09-03 | 54 | record (dated research) | records/STATE-WORKFLOW-RESEARCH.md, moved whole and not rewritten beyond stage 10's spelling |
| 446 | · · 0. The ask, and the answer in one paragraph | 31 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 477 | · · 1. What existed before the decision, per step and surface | 39 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 516 | · · 2. How polars executes a Python IO source — measured | 34 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 550 | · · 3. The candidates | 12 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 562 | · · 4. The proposal: A, made exact | 123 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 685 | · · 5. The prototype's checks (real bank, 3 groups × 1,000 rows, chunks of 250) | 20 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 705 | · · 6. Usability, performance, stability — checked, not assumed | 33 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 738 | · · 7. Decisions (taken 2026-09-03) | 33 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 771 | · · 8. Where it landed | 10 | record (dated research) | with its parent: The research behind it, 2026-09-03 |
| 781 | · · 9. Reproduce | 8 | record (dated research) | with its parent: The research behind it, 2026-09-03 |

### docs/DIAGNOSTICS.md

25 sections, 2344 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Is the model working? | 123 | mixed: explanation (opener), reference (questions table) | DIAGNOSTICS.md, the diagnostics' reference (the path stays): a table from each question to its how-to and its entries; the opener's framing of the questions to explanation/how-diagnostics-work.md |
| 124 | · How the diagnostics work | 7 | explanation | explanation/how-diagnostics-work.md |
| 131 | · · Out of sample, by construction | 28 | explanation | explanation/out-of-sample.md (stage 1) |
| 159 | · · One memory per diagnostic: run once or windowed | 135 | mixed: explanation, reference (the half-life keywords) | explanation/how-diagnostics-work.md; the keywords to DIAGNOSTICS.md; the heading stays in DIAGNOSTICS.md, which shipped code links |
| 294 | · · Reading a statistic that is read on every row | 10 | explanation | explanation/how-diagnostics-work.md |
| 304 | · · Every switch at once | 69 | reference | DIAGNOSTICS.md: every switch, one spec |
| 373 | · · What each costs, and which models take it | 32 | reference | DIAGNOSTICS.md: costs, which holds the table; each diagnostic's entry links its row, and PERFORMANCE.md links the table |
| 405 | · Is it scaled right? | 127 | mixed: how-to, reference (definition, threshold), explanation (theory) | how-to/check-the-scale.md; the definition and threshold to DIAGNOSTICS.md's `emit_calibration` entry, and the theory to that entry's Notes |
| 532 | · Has it broken? | 428 | mixed: how-to, reference, explanation | how-to/detect-a-break.md; each break statistic's entry to DIAGNOSTICS.md; the theory to the Notes of each diagnostic's entry in DIAGNOSTICS.md |
| 960 | · Can its t be trusted? | 175 | mixed: how-to, reference, explanation | how-to/check-standard-errors.md; entries to DIAGNOSTICS.md; the theory to the Notes of each diagnostic's entry in DIAGNOSTICS.md |
| 1135 | · What is it missing? | 151 | mixed: how-to, reference, explanation | how-to/find-what-the-model-misses.md; entries to DIAGNOSTICS.md; the theory to the Notes of each diagnostic's entry in DIAGNOSTICS.md |
| 1286 | · Are its tails heavy? | 140 | mixed: how-to, reference, explanation | how-to/measure-heavy-tails.md; entries to DIAGNOSTICS.md; the theory to the Notes of each diagnostic's entry in DIAGNOSTICS.md |
| 1426 | · Which row moved it? | 101 | mixed: how-to, reference, explanation | how-to/find-an-influential-row.md; entry to DIAGNOSTICS.md; the theory to the Notes of each diagnostic's entry in DIAGNOSTICS.md |
| 1527 | · Are its features healthy? | 87 | mixed: how-to, reference, explanation | how-to/check-feature-health.md; entry to DIAGNOSTICS.md; the theory to the Notes of each diagnostic's entry in DIAGNOSTICS.md |
| 1614 | · Is another model better? | 35 | explanation (opener) | DIAGNOSTICS.md: the comparison rows of its questions table; its prose on comparing models to the openings of how-to/compare-two-models.md and how-to/choose-among-a-grid-online.md, and on why the best error is optimistic to explanation/grids-and-instances.md |
| 1649 | · · Choosing among a grid's settings | 76 | how-to | how-to/choose-among-a-grid-online.md |
| 1725 | · · Comparing two models | 95 | mixed: how-to, explanation | how-to/compare-two-models.md; the test's theory to the Notes of `po.eval.compare_specs`'s docstring |
| 1820 | · · Evaluating an output frame | 79 | how-to | how-to/evaluate-an-output-frame.md |
| 1899 | · · Evaluating a stream too large to hold | 54 | how-to | how-to/evaluate-a-stream-too-large-to-hold.md |
| 1953 | · Is it ready? | 82 | mixed: how-to, explanation | how-to/decide-when-a-model-is-ready.md; the reasons to explanation/warm-up-and-readiness.md; the heading stays in DIAGNOSTICS.md, which shipped code links |
| 2035 | · How wide is its error? | 26 | mixed: explanation (opener), reference | how-to/put-an-interval-on-a-prediction.md, whose opening sentence its opener becomes; the four diagnostics' entries to DIAGNOSTICS.md |
| 2061 | · · Conformal intervals | 91 | how-to | how-to/put-an-interval-on-a-prediction.md |
| 2152 | · A worked example: the two-stage reversion beta | 124 | mixed: how-to (a worked two-stage fit), explanation (why each stage is checked) | how-to/fit-on-factor-residuals.md, which generalizes it to several factor ridges and a second-stage ridge under a hard window; why each stage is checked by the diagnostics it asks for to explanation/how-diagnostics-work.md |
| 2276 | · Data whose truth is known | 40 | how-to | how-to/test-a-detector-on-known-truth.md |
| 2316 | · Sources | 29 | reference | DIAGNOSTICS.md keeps its sources, and each new page carries the ones it cites |

### docs/CALIBRATION.md

12 sections, 588 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Calibrating a model | 65 | mixed: explanation (why the best error is optimistic), how-to | CALIBRATION.md, the how-to: choose settings with `grid_search` (the path stays); why to explanation/grids-and-instances.md |
| 66 | · One pattern for every setting | 69 | how-to | CALIBRATION.md |
| 135 | · · Choosing with `po.eval.grid_search` | 63 | mixed: reference, how-to | CALIBRATION.md (the steps); the parameters to the API reference's `grid_search` entry |
| 198 | · · Choosing by hand | 56 | how-to | how-to/choose-settings-by-hand.md |
| 254 | · Recipes | 6 | how-to (opener) | CALIBRATION.md: a table from each setting to its how-to |
| 260 | · · The ridge | 60 | how-to | how-to/choose-a-ridge-and-half-life.md |
| 320 | · · The ridge and the half-life together | 55 | how-to | how-to/choose-a-ridge-and-half-life.md |
| 375 | · · A Kalman filter's drift speed | 51 | how-to | how-to/choose-a-kalman-drift-speed.md |
| 426 | · · The learning rate | 49 | how-to | how-to/choose-a-learning-rate.md |
| 475 | · · Holt's smoothing | 46 | how-to | how-to/choose-holts-smoothing.md |
| 521 | · · The changepoint hazard | 55 | how-to | how-to/choose-a-changepoint-hazard.md |
| 576 | · What this cannot choose | 13 | reference | the Notes of `po.eval.grid_search`'s docstring, with a one-line link left in CALIBRATION.md |

### docs/DATA-QUALITY.md

14 sections, 1650 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Data quality | 111 | mixed: explanation (opener), reference (findings table) | DATA-QUALITY.md, the findings' reference (the path stays); running the check to how-to/check-data-quality.md; the opener's account of what a fit cannot tell you to explanation/how-data-quality-checks-work.md |
| 112 | · How the checks work | 74 | explanation | explanation/how-data-quality-checks-work.md; the heading stays in DATA-QUALITY.md, which shipped code links |
| 186 | · Missing values | 165 | mixed: reference (the finding, what it does to each model), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept (`bank.check()` links them); the fix stays in the entry, as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 351 | · Sentinel values | 97 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 448 | · Constant and frozen features | 179 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 627 | · A level far above its spread, and scales apart | 200 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 827 | · Duplicated and collinear features | 115 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 942 | · Categories stored as numbers | 79 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 1021 | · Random walks | 99 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 1120 | · Heavy tails | 109 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 1229 | · Leakage | 108 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 1337 | · The clock | 144 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 1481 | · Too little data for the fit | 147 | mixed: reference (the finding), how-to (the fix in Polars), explanation (theory) | DATA-QUALITY.md entry, its finding anchors kept; the fix stays as its remedy; the theory to explanation/how-data-quality-checks-work.md |
| 1628 | · Sources | 23 | reference | DATA-QUALITY.md keeps its sources, and each new page carries the ones it cites |

### docs/GRAM.md

21 sections, 2070 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Wide fits from Grams | 165 | mixed: explanation, how-to (a first recipe) | GRAM.md as explanation: how a Gram stands in for the rows (the path stays; shipped docstrings link it); the recipe to how-to/fit-weekly-grams-in-one-pass.md; GRAM.md opens with a table from each task to its how-to |
| 166 | · How a Gram stands in for the rows | 9 | explanation | GRAM.md |
| 175 | · · Every least-squares fit is a function of the Gram | 57 | explanation | with its parent: How a Gram stands in for the rows |
| 232 | · · Which rows a Gram holds, and as of when | 17 | explanation | with its parent: How a Gram stands in for the rows |
| 249 | · · What a Gram cannot answer | 13 | explanation | with its parent: How a Gram stands in for the rows |
| 262 | · Weekly blocks from one pass | 110 | how-to | how-to/fit-weekly-grams-in-one-pass.md |
| 372 | · Pooling blocks | 107 | how-to | how-to/pool-grams.md |
| 479 | · · Aging a block before it is pooled | 83 | how-to | with its parent: Pooling blocks |
| 562 | · · Pooling within blocks | 128 | how-to | with its parent: Pooling blocks |
| 690 | · A ridge and its t | 149 | how-to | how-to/fit-a-ridge-from-a-gram.md |
| 839 | · Is the fit determined? | 83 | how-to | how-to/check-a-fit-is-determined.md |
| 922 | · Choosing columns on the lasso path | 146 | how-to | how-to/choose-columns-on-the-lasso-path.md |
| 1068 | · Stability selection over halves of the weeks | 144 | how-to | how-to/select-columns-stably-over-weeks.md |
| 1212 | · · A null for the selection, by sign flips | 118 | how-to | with its parent: Stability selection over halves of the weeks |
| 1330 | · · A screen of its own for each path | 115 | how-to | with its parent: Stability selection over halves of the weeks |
| 1445 | · Per-block coefficients | 92 | how-to | how-to/fit-per-block-coefficients.md |
| 1537 | · Reading Grams from saved files | 101 | how-to | how-to/read-grams-from-saved-files.md |
| 1638 | · A compact Gram | 114 | how-to | how-to/store-a-compact-gram.md |
| 1752 | · One wide Gram on several threads | 116 | how-to | how-to/update-a-wide-gram-on-threads.md |
| 1868 | · Huber regression on a Gram | 172 | how-to | how-to/fit-huber-on-a-gram.md |
| 2040 | · Sources | 31 | reference | GRAM.md keeps its sources, and each new page carries the ones it cites |

### docs/GBDT.md

8 sections, 347 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Training XGBoost and LightGBM: `po.gbdt` | 122 | mixed: how-to, explanation | GBDT.md, the how-to: train XGBoost or LightGBM on files larger than memory (the path stays); its reasons to explanation/how-po-gbdt-builds-training-data.md |
| 123 | · What each step does | 12 | explanation | explanation/how-po-gbdt-builds-training-data.md |
| 135 | · · XGBoost places its cuts where they are planted | 36 | explanation | with its parent: What each step does |
| 171 | · · The Python engine grows the native engine's trees | 35 | explanation | with its parent: What each step does |
| 206 | · · A categorical feature is a `pl.Enum` | 56 | explanation | with its parent: What each step does |
| 262 | · · A LightGBM dataset takes its binning parameters when it is built | 47 | explanation | with its parent: What each step does |
| 309 | · · LightGBM's route in memory | 9 | explanation | with its parent: What each step does |
| 318 | · Polars' read-ahead | 30 | explanation | explanation/polars-interfaces.md: read-ahead |

### docs/OUTPUTS.md

31 sections, 459 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | What each model writes | 14 | reference (generated) | OUTPUTS.md, unchanged in kind; regenerated by its script |
| 15 | · Reading this page | 5 | reference (generated) | with its parent: What each model writes |
| 20 | · · The fields of your own spec | 30 | reference (generated) | with its parent: Reading this page |
| 50 | · · Field names | 34 | reference (generated) | with its parent: Reading this page |
| 84 | · · Field types | 19 | reference (generated) | with its parent: Reading this page |
| 103 | · · When a field is null | 13 | reference (generated) | with its parent: Reading this page |
| 116 | · · Fields most models write | 11 | reference (generated) | with its parent: Reading this page |
| 127 | · · What is not listed | 21 | reference (generated) | with its parent: Reading this page |
| 148 | · · How this page is made | 10 | reference (generated) | with its parent: Reading this page |
| 158 | · `ewridge` | 14 | reference (generated) | with its parent: What each model writes |
| 172 | · `rls` | 13 | reference (generated) | with its parent: What each model writes |
| 185 | · `lasso` | 16 | reference (generated) | with its parent: What each model writes |
| 201 | · `kalman` | 13 | reference (generated) | with its parent: What each model writes |
| 214 | · `huber` | 14 | reference (generated) | with its parent: What each model writes |
| 228 | · `quantile` | 14 | reference (generated) | with its parent: What each model writes |
| 242 | · `sgd` | 13 | reference (generated) | with its parent: What each model writes |
| 255 | · `pa` | 13 | reference (generated) | with its parent: What each model writes |
| 268 | · `ftrl` | 13 | reference (generated) | with its parent: What each model writes |
| 281 | · `holt` | 13 | reference (generated) | with its parent: What each model writes |
| 294 | · `ew_cov` | 15 | reference (generated) | with its parent: What each model writes |
| 309 | · `marginal` | 10 | reference (generated) | with its parent: What each model writes |
| 319 | · `deco` | 14 | reference (generated) | with its parent: What each model writes |
| 333 | · `rcov` | 10 | reference (generated) | with its parent: What each model writes |
| 343 | · `audit` | 10 | reference (generated) | with its parent: What each model writes |
| 353 | · `kmeans` | 14 | reference (generated) | with its parent: What each model writes |
| 367 | · `micro` | 17 | reference (generated) | with its parent: What each model writes |
| 384 | · `ew_class` | 14 | reference (generated) | with its parent: What each model writes |
| 398 | · `seqtest` | 14 | reference (generated) | with its parent: What each model writes |
| 412 | · `corrchange` | 15 | reference (generated) | with its parent: What each model writes |
| 427 | · `bocpd` | 16 | reference (generated) | with its parent: What each model writes |
| 443 | · `hmm` | 17 | reference (generated) | with its parent: What each model writes |

### docs/REGIMES.md

13 sections, 722 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Regimes: what the detectors find, measured | 45 | reference (measurements) | REGIMES.md, the regime models' measured behavior (the path stays) |
| 46 | · 0. The short version | 72 | reference (a summary of the measurements) | REGIMES.md opener |
| 118 | · 1. Does `hmm` recover the stream that made it? | 59 | reference (measurements) | with its parent: Regimes: what the detectors find, measured |
| 177 | · · Through a switch | 31 | reference (measurements) | with its parent: 1. Does `hmm` recover the stream that made it? |
| 208 | · 2. Is the monitor the size its paper says? | 65 | reference (measurements) | with its parent: Regimes: what the detectors find, measured |
| 273 | · 3. And is it the power its paper says? | 54 | reference (measurements) | with its parent: Regimes: what the detectors find, measured |
| 327 | · 4. Where the heavy tails go: `D-hat` | 59 | reference (measurements) | with its parent: Regimes: what the detectors find, measured |
| 386 | · 5. Two changepoint detectors on the same break | 67 | reference (measurements) | with its parent: Regimes: what the detectors find, measured |
| 453 | · 6. The Epps effect, and the two ways out | 60 | mixed: reference, explanation (the Epps effect) | REGIMES.md; the effect to explanation/asynchronous-series.md |
| 513 | · 7. `deco`: an equicorrelation that moves | 34 | reference (measurements) | with its parent: Regimes: what the detectors find, measured |
| 547 | · 8. `rcov`: three estimators against each block's truth | 55 | reference (measurements) | with its parent: Regimes: what the detectors find, measured |
| 602 | · 9. The sequential detector against its paper | 74 | reference (measurements) | with its parent: Regimes: what the detectors find, measured |
| 676 | · 10. Running them | 47 | reference (how the measurements are made) | REGIMES.md: how the measurements are made |

### docs/VALIDATION.md

8 sections, 119 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Validation results | 12 | reference (generated) | VALIDATION.md, unchanged; regenerated by its script |
| 13 | · How the runs are set | 10 | reference (generated) | with its parent: Validation results |
| 23 | · 1. Solve schedule (`solve_every` default = by weight, half_life/50 in steady state) [validate] | 15 | reference (generated) | with its parent: Validation results |
| 38 | · 2. `standardize` default (false for ridge, true for lasso) | 7 | reference (generated) | with its parent: Validation results |
| 45 | · 3. Elastic net `l1_ratio` [validate] | 16 | reference (generated) | with its parent: Validation results |
| 61 | · 4. Kalman `share_p` approximation [validate] | 11 | reference (generated) | with its parent: Validation results |
| 72 | · 5. Models at matched settings | 21 | reference (generated) | with its parent: Validation results |
| 93 | · 6. Features as z-scores (the per-standard-deviation recipe) | 27 | reference (generated) | with its parent: Validation results |

### docs/PERFORMANCE.md

200 sections, 6416 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Performance: where the time and memory go | 65 | explanation | the title stays, retitled as the measured costs' reference (the path stays); the opener's text to explanation/where-the-time-and-memory-go.md |
| 66 | · Reading this document | 8 | reference (opener) | PERFORMANCE.md, the measured costs' reference |
| 74 | · · The headline | 95 | explanation | explanation/where-the-time-and-memory-go.md |
| 169 | · · Words this document uses | 23 | reference | GLOSSARY.md, which stage 0 wrote without these words; stage 4 adds any the glossary lacks and removes the section |
| 192 | · · Names that changed | 59 | reference | VERSIONS.md: names that changed |
| 251 | · · How the numbers are made | 47 | reference | PERFORMANCE.md: how the numbers are made |
| 298 | · · Budgets that only tighten | 73 | reference (contributor) | TESTING.md: budgets |
| 371 | · How cost grows with size | 15 | reference | PERFORMANCE.md: how cost grows with size |
| 386 | · · How a cell is measured | 55 | reference | with its parent: How cost grows with size |
| 441 | · · The table | 147 | reference | with its parent: How cost grows with size |
| 588 | · · Specs on one stream | 45 | reference | with its parent: How cost grows with size |
| 633 | · · What the table found | 38 | reference | with its parent: How cost grows with size |
| 671 | · The first pass, and the bank's plumbing | 7 | record (dated measurements) | records/PERFORMANCE-LOG.md, moved whole with its section numbers, which code comments and the records cite. A guide that cites one today cites PERFORMANCE.md's reference instead |
| 678 | · · The first pass, in one table | 32 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 710 | · · 1. The measured baseline | 47 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 757 | · · 2. What the numbers say | 42 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 799 | · · 3. The plan | 23 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 822 | · · · P1 — Columnar hot path | 44 | record (dated measurements) | with its parent: 3. The plan |
| 866 | · · · P2 — One flat task pool | 30 | record (dated measurements) | with its parent: 3. The plan |
| 896 | · · · P3 — Extraction and grouping without materialization | 30 | record (dated measurements) | with its parent: 3. The plan |
| 926 | · · · P4 — Assembly into typed builders | 17 | record (dated measurements) | with its parent: 3. The plan |
| 943 | · · · P5 — Expression path parity | 32 | record (dated measurements) | with its parent: 3. The plan |
| 975 | · · · P6 — Runner pipelining | 19 | record (dated measurements) | with its parent: 3. The plan |
| 994 | · · · P7 — Build flags: measured, and both rejected | 26 | record (dated measurements) | with its parent: 3. The plan |
| 1020 | · · · P8 — Re-baseline and lock | 17 | record (dated measurements) | with its parent: 3. The plan |
| 1037 | · · · P9 — Group-contiguous layout (2026-09-04) | 12 | record (dated measurements) | with its parent: 3. The plan |
| 1049 | · · · P10 — Assembly per field (2026-09-04) | 17 | record (dated measurements) | with its parent: 3. The plan |
| 1066 | · · · P11 — Extraction: per column, per arrow chunk, and integer keys as themselves (2026-09-04) | 20 | record (dated measurements) | with its parent: 3. The plan |
| 1086 | · · 4. Where it ended up | 41 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 1127 | · · 5. Rejected, and why | 63 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 1190 | · · · Rejected later, where each was measured | 20 | record (dated measurements) | with its parent: 5. Rejected, and why |
| 1210 | · · 6. The allocator (2026-08-31) | 29 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 1239 | · · 7. Bugs found by this review | 7 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 1246 | · · 8. Refresh (2026-09-02), and what `rls` costs now | 2 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 1248 | · · · The refresh | 40 | record (dated measurements) | with its parent: 8. Refresh (2026-09-02), and what `rls` costs now |
| 1288 | · · · `rls` and its square-root form | 19 | record (dated measurements) | with its parent: 8. Refresh (2026-09-02), and what `rls` costs now |
| 1307 | · · · Where the QR form's time actually went (2026-09-02) | 41 | record (dated measurements) | with its parent: 8. Refresh (2026-09-02), and what `rls` costs now |
| 1348 | · · 12. The chunk plan, revisited (2026-09-04) | 8 | record (dated measurements) | with its parent: The first pass, and the bank's plumbing |
| 1356 | · · · The plan as it stood | 15 | record (dated measurements) | with its parent: 12. The chunk plan, revisited (2026-09-04) |
| 1371 | · · · What the sections said | 8 | record (dated measurements) | with its parent: 12. The chunk plan, revisited (2026-09-04) |
| 1379 | · · · The benchmark artifact | 23 | record (dated measurements) | with its parent: 12. The chunk plan, revisited (2026-09-04) |
| 1402 | · · · Where it ended up | 55 | record (dated measurements) | with its parent: 12. The chunk plan, revisited (2026-09-04) |
| 1457 | · · · `chunk_rows`, swept | 39 | record (dated measurements) | with its parent: 12. The chunk plan, revisited (2026-09-04) |
| 1496 | · · · The README's numbers, regenerated | 43 | record (dated measurements) | with its parent: 12. The chunk plan, revisited (2026-09-04) |
| 1539 | · · · The row floor | 32 | record (dated measurements) | with its parent: 12. The chunk plan, revisited (2026-09-04) |
| 1571 | · · · What is left, and why it stays | 14 | record (dated measurements) | with its parent: 12. The chunk plan, revisited (2026-09-04) |
| 1585 | · Each way of running a bank | 5 | record (dated measurements) | records/PERFORMANCE-LOG.md |
| 1590 | · · 9. `predict` (2026-09-02) | 20 | record (dated measurements) | with its parent: Each way of running a bank |
| 1610 | · · 10. The runner: every format, every source (2026-09-02) | 10 | record (dated measurements) | with its parent: Each way of running a bank |
| 1620 | · · · The pipeline | 11 | record (dated measurements) | with its parent: 10. The runner: every format, every source (2026-09-02) |
| 1631 | · · · Every format | 38 | record (dated measurements) | with its parent: 10. The runner: every format, every source (2026-09-02) |
| 1669 | · · · Why py-polars reads for the Python path | 30 | record (dated measurements) | with its parent: 10. The runner: every format, every source (2026-09-02) |
| 1699 | · · · What the custom parts are worth | 12 | record (dated measurements) | with its parent: 10. The runner: every format, every source (2026-09-02) |
| 1711 | · · · Why the bank is not a polars node | 23 | record (dated measurements) | with its parent: 10. The runner: every format, every source (2026-09-02) |
| 1734 | · · · Before and after: the one-thread runner against the pipeline | 29 | record (dated measurements) | with its parent: 10. The runner: every format, every source (2026-09-02) |
| 1763 | · · · Writers | 11 | record (dated measurements) | with its parent: 10. The runner: every format, every source (2026-09-02) |
| 1774 | · · 11. Memory: which surface is O(data) (2026-09-02) | 32 | record (dated measurements) | with its parent: Each way of running a bank |
| 1806 | · · · The plugin was O(data); nothing else is | 38 | record (dated measurements) | with its parent: 11. Memory: which surface is O(data) (2026-09-02) |
| 1844 | · · · What the constant is | 25 | record (dated measurements) | with its parent: 11. Memory: which surface is O(data) (2026-09-02) |
| 1869 | · · · The query-shaped trap, and the fix | 25 | record (dated measurements) | with its parent: 11. Memory: which surface is O(data) (2026-09-02) |
| 1894 | · · · What comes before the bank | 58 | record (dated measurements) | with its parent: 11. Memory: which surface is O(data) (2026-09-02) |
| 1952 | · · · Where the window comes from | 72 | record (dated measurements) | with its parent: 11. Memory: which surface is O(data) (2026-09-02) |
| 2024 | · · · What shrinks the window | 26 | record (dated measurements) | with its parent: 11. Memory: which surface is O(data) (2026-09-02) |
| 2050 | · · · Filter after the bank | 44 | mixed: record, how-to (one rule) | records/PERFORMANCE-LOG.md, whole; stage 6 copies its rule, filter after the bank, into how-to/bound-memory.md and leaves the record as it is |
| 2094 | · · · Two levers not taken | 26 | record (dated measurements) | with its parent: 11. Memory: which surface is O(data) (2026-09-02) |
| 2120 | · · · Polars' own windowed operations do the same thing | 67 | record (dated measurements) | with its parent: 11. Memory: which surface is O(data) (2026-09-02) |
| 2187 | · · · How it was measured, and why not RSS | 12 | mixed: record, reference (how memory is measured) | records/PERFORMANCE-LOG.md, whole; stage 4 copies its account of how memory is measured into PERFORMANCE.md: how the numbers are made |
| 2199 | · · 21. What inspecting the plan costs `fit(lf)` (2026-09-18) | 5 | record (dated measurements) | with its parent: Each way of running a bank |
| 2204 | · · · The bisect | 23 | record (dated measurements) | with its parent: 21. What inspecting the plan costs `fit(lf)` (2026-09-18) |
| 2227 | · · · What it actually is | 7 | record (dated measurements) | with its parent: 21. What inspecting the plan costs `fit(lf)` (2026-09-18) |
| 2234 | · · · The fix | 32 | record (dated measurements) | with its parent: 21. What inspecting the plan costs `fit(lf)` (2026-09-18) |
| 2266 | · · · Measure the case the user has | 9 | record (dated measurements) | with its parent: 21. What inspecting the plan costs `fit(lf)` (2026-09-18) |
| 2275 | · · 40. An eager call's peak: each run into the output as it ends (2026-10-13) | 67 | record (dated measurements) | with its parent: Each way of running a bank |
| 2342 | · · · Across pool sizes | 73 | record (dated measurements) | with its parent: 40. An eager call's peak: each run into the output as it ends (2026-10-13) |
| 2415 | · Model by model | 7 | record (dated measurements) | records/PERFORMANCE-LOG.md; nothing else moves, since PERFORMANCE.md: how cost grows with size already gives each model's current cost |
| 2422 | · · 13. The new families, and where a wide row goes (2026-09-05) | 10 | record (dated measurements) | with its parent: Model by model |
| 2432 | · · · The contract: bit for bit | 15 | record (dated measurements) | with its parent: 13. The new families, and where a wide row goes (2026-09-05) |
| 2447 | · · · The survey | 42 | record (dated measurements) | with its parent: 13. The new families, and where a wide row goes (2026-09-05) |
| 2489 | · · · Where `ew_cov`'s row went | 74 | record (dated measurements) | with its parent: 13. The new families, and where a wide row goes (2026-09-05) |
| 2563 | · · · Reading the wall clock, not the profile | 27 | record (dated measurements) | with its parent: 13. The new families, and where a wide row goes (2026-09-05) |
| 2590 | · · · `ew_class`: one factorization per learned row | 19 | record (dated measurements) | with its parent: 13. The new families, and where a wide row goes (2026-09-05) |
| 2609 | · · · `kalman`, `sgd`, `pa`: allocations and a closure | 23 | record (dated measurements) | with its parent: 13. The new families, and where a wide row goes (2026-09-05) |
| 2632 | · · · Thread scaling | 41 | record (dated measurements) | with its parent: 13. The new families, and where a wide row goes (2026-09-05) |
| 2673 | · · · What is left, and why | 69 | record (dated measurements) | with its parent: 13. The new families, and where a wide row goes (2026-09-05) |
| 2742 | · · 14. The Gram update, measured (2026-09-05) | 9 | record (dated measurements) | with its parent: Model by model |
| 2751 | · · · The variants | 23 | record (dated measurements) | with its parent: 14. The Gram update, measured (2026-09-05) |
| 2774 | · · · The mirror is not bit-identical | 14 | record (dated measurements) | with its parent: 14. The Gram update, measured (2026-09-05) |
| 2788 | · · · And it is slower, by a lot | 9 | record (dated measurements) | with its parent: 14. The Gram update, measured (2026-09-05) |
| 2797 | · · · What is worth having | 23 | record (dated measurements) | with its parent: 14. The Gram update, measured (2026-09-05) |
| 2820 | · · 18. The blocked Gram update (2026-09-08) | 17 | record (dated measurements) | with its parent: Model by model |
| 2837 | · · · The measurement | 17 | record (dated measurements) | with its parent: 18. The blocked Gram update (2026-09-08) |
| 2854 | · · · Three things to read off it | 44 | record (dated measurements) | with its parent: 18. The blocked Gram update (2026-09-08) |
| 2898 | · · · What blocking does not change | 12 | record (dated measurements) | with its parent: 18. The blocked Gram update (2026-09-08) |
| 2910 | · · 37. One wide Gram on several threads (2026-10-08) | 15 | record (dated measurements) | with its parent: Model by model |
| 2925 | · · · Where one core's time goes | 27 | record (dated measurements) | with its parent: 37. One wide Gram on several threads (2026-10-08) |
| 2952 | · · · The design: pieces fixed by `k` | 30 | record (dated measurements) | with its parent: 37. One wide Gram on several threads (2026-10-08) |
| 2982 | · · · The measurement | 55 | record (dated measurements) | with its parent: 37. One wide Gram on several threads (2026-10-08) |
| 3037 | · · · Many groups share the threads | 19 | record (dated measurements) | with its parent: 37. One wide Gram on several threads (2026-10-08) |
| 3056 | · · · The solve | 13 | record (dated measurements) | with its parent: 37. One wide Gram on several threads (2026-10-08) |
| 3069 | · · 38. A solve's bits, whatever the pool (2026-10-09) | 13 | record (dated measurements) | with its parent: Model by model |
| 3082 | · · · What followed the pool | 20 | record (dated measurements) | with its parent: 38. A solve's bits, whatever the pool (2026-10-09) |
| 3102 | · · · The fix: cut by shape, or a fixed degree | 20 | record (dated measurements) | with its parent: 38. A solve's bits, whatever the pool (2026-10-09) |
| 3122 | · · · The measurement | 21 | record (dated measurements) | with its parent: 38. A solve's bits, whatever the pool (2026-10-09) |
| 3143 | · · · What moved, once | 21 | record (dated measurements) | with its parent: 38. A solve's bits, whatever the pool (2026-10-09) |
| 3164 | · · 41. A product's memory on each thread (2026-10-13) | 33 | record (dated measurements) | with its parent: Model by model |
| 3197 | · · · What a thread allocates | 19 | record (dated measurements) | with its parent: 41. A product's memory on each thread (2026-10-13) |
| 3216 | · · · What moved | 19 | record (dated measurements) | with its parent: 41. A product's memory on each thread (2026-10-13) |
| 3235 | · · · Speed | 34 | record (dated measurements) | with its parent: 41. A product's memory on each thread (2026-10-13) |
| 3269 | · · 42. A blocked Gram's targets, and a bank's own share (2026-10-13) | 9 | record (dated measurements) | with its parent: Model by model |
| 3278 | · · · Where a target's time went | 21 | record (dated measurements) | with its parent: 42. A blocked Gram's targets, and a bank's own share (2026-10-13) |
| 3299 | · · · The fold | 37 | record (dated measurements) | with its parent: 42. A blocked Gram's targets, and a bank's own share (2026-10-13) |
| 3336 | · · · The copy into rows | 23 | record (dated measurements) | with its parent: 42. A blocked Gram's targets, and a bank's own share (2026-10-13) |
| 3359 | · · · The copy, a run at a time | 67 | record (dated measurements) | with its parent: 42. A blocked Gram's targets, and a bank's own share (2026-10-13) |
| 3426 | · · · The cast, in the copy | 111 | record (dated measurements) | with its parent: 42. A blocked Gram's targets, and a bank's own share (2026-10-13) |
| 3537 | · · 43. faer's x86 kernel on every x86 build (2026-10-15) | 23 | record (dated measurements) | with its parent: Model by model |
| 3560 | · · · Speed on Linux, with the kernel against without it | 33 | record (dated measurements) | with its parent: 43. faer's x86 kernel on every x86 build (2026-10-15) |
| 3593 | · · · Windows: the build without the kernel, against 1.0.1 | 27 | record (dated measurements) | with its parent: 43. faer's x86 kernel on every x86 build (2026-10-15) |
| 3620 | · · · The kernel's wrong products, and the version that fixes them | 26 | record (dated measurements) | with its parent: 43. faer's x86 kernel on every x86 build (2026-10-15) |
| 3646 | · · 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) | 6 | record (dated measurements) | with its parent: Model by model |
| 3652 | · · · The protocol | 18 | record (dated measurements) | with its parent: 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) |
| 3670 | · · · Accuracy | 49 | record (dated measurements) | with its parent: 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) |
| 3719 | · · · The first hundred rows of every group | 26 | record (dated measurements) | with its parent: 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) |
| 3745 | · · · Where `sgd` lost, and the scaler's fix | 45 | record (dated measurements) | with its parent: 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) |
| 3790 | · · · Throughput | 27 | record (dated measurements) | with its parent: 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) |
| 3817 | · · · A grid | 15 | record (dated measurements) | with its parent: 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) |
| 3832 | · · · A wide row | 153 | record (dated measurements) | with its parent: 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) |
| 3985 | · · · What the comparison gave back | 14 | record (dated measurements) | with its parent: 19. Against `sklearn.linear_model.SGDRegressor` (2026-09-08) |
| 3999 | · · 20. `sgd`'s per-feature cost (2026-09-08) | 9 | record (dated measurements) | with its parent: Model by model |
| 4008 | · · · Where the 14 ns went | 50 | record (dated measurements) | with its parent: 20. `sgd`'s per-feature cost (2026-09-08) |
| 4058 | · · · The result | 29 | record (dated measurements) | with its parent: 20. `sgd`'s per-feature cost (2026-09-08) |
| 4087 | · · · What is left, at `k = 10,000` | 26 | record (dated measurements) | with its parent: 20. `sgd`'s per-feature cost (2026-09-08) |
| 4113 | · · · The scaler | 12 | record (dated measurements) | with its parent: 20. `sgd`'s per-feature cost (2026-09-08) |
| 4125 | · · 15. The correlation families: `bocpd`'s `prune_below` keeps it finite, and `rcov`'s estimator sets its cost (2026-09-06) | 9 | record (dated measurements) | with its parent: Model by model |
| 4134 | · · · `bocpd`: `prune_below` is what makes it finite | 39 | record (dated measurements) | with its parent: 15. The correlation families: `bocpd`'s `prune_below` keeps it finite, and `rcov`'s estimator sets its cost (2026-09-06) |
| 4173 | · · · `rcov`: the cost is at the close, and the kernel chooses it | 28 | record (dated measurements) | with its parent: 15. The correlation families: `bocpd`'s `prune_below` keeps it finite, and `rcov`'s estimator sets its cost (2026-09-06) |
| 4201 | · · · The other three | 20 | record (dated measurements) | with its parent: 15. The correlation families: `bocpd`'s `prune_below` keeps it finite, and `rcov`'s estimator sets its cost (2026-09-06) |
| 4221 | · · · And they go wide | 29 | record (dated measurements) | with its parent: 15. The correlation families: `bocpd`'s `prune_below` keeps it finite, and `rcov`'s estimator sets its cost (2026-09-06) |
| 4250 | · · 16. What a `window` costs (2026-09-07) | 5 | record (dated measurements) | with its parent: Model by model |
| 4255 | · · · The windowless path is unchanged | 24 | record (dated measurements) | with its parent: 16. What a `window` costs (2026-09-07) |
| 4279 | · · · What the window itself costs | 45 | record (dated measurements) | with its parent: 16. What a `window` costs (2026-09-07) |
| 4324 | · · 17. What `marginal`'s two views cost, and two costs that were not theirs (2026-09-07) | 6 | record (dated measurements) | with its parent: Model by model |
| 4330 | · · · The plain path, and two regressions no test could see | 46 | record (dated measurements) | with its parent: 17. What `marginal`'s two views cost, and two costs that were not theirs (2026-09-07) |
| 4376 | · · · What the views themselves cost | 29 | record (dated measurements) | with its parent: 17. What `marginal`'s two views cost, and two costs that were not theirs (2026-09-07) |
| 4405 | · · 22. `marginal`'s bins at several targets (2026-09-25) | 40 | record (dated measurements) | with its parent: Model by model |
| 4445 | · · 23. `marginal`'s cross terms on request (2026-09-25) | 42 | record (dated measurements) | with its parent: Model by model |
| 4487 | · · 24. A wide `marginal` through the bank: the cost that was not the model's (2026-09-25) | 74 | record (dated measurements) | with its parent: Model by model |
| 4561 | · · 25. A wide `marginal` split across the pool (2026-09-25) | 12 | record (dated measurements) | with its parent: Model by model |
| 4573 | · · · A fork-join per row does not pay | 32 | record (dated measurements) | with its parent: 25. A wide `marginal` split across the pool (2026-09-25) |
| 4605 | · · · The unsplit row got faster first | 22 | record (dated measurements) | with its parent: 25. A wide `marginal` split across the pool (2026-09-25) |
| 4627 | · · · What the split buys | 49 | record (dated measurements) | with its parent: 25. A wide `marginal` split across the pool (2026-09-25) |
| 4676 | · · · `"auto"` | 25 | record (dated measurements) | with its parent: 25. A wide `marginal` split across the pool (2026-09-25) |
| 4701 | · · 27. `marginal`'s shared feature moments (2026-09-29) | 40 | record (dated measurements) | with its parent: Model by model |
| 4741 | · · 39. The state's traffic behind `marginal`'s slope (2026-10-10) | 19 | record (dated measurements) | with its parent: Model by model |
| 4760 | · · · The model's slope and the stream's intercept | 51 | record (dated measurements) | with its parent: 39. The state's traffic behind `marginal`'s slope (2026-10-10) |
| 4811 | · · · The pair's state against the caches | 15 | record (dated measurements) | with its parent: 39. The state's traffic behind `marginal`'s slope (2026-10-10) |
| 4826 | · · · How much of the slope is the traffic | 49 | record (dated measurements) | with its parent: 39. The state's traffic behind `marginal`'s slope (2026-10-10) |
| 4875 | · · · Three loop orders on the same arithmetic | 28 | record (dated measurements) | with its parent: 39. The state's traffic behind `marginal`'s slope (2026-10-10) |
| 4903 | · · · Built: a sharded `marginal` steps one target at a time | 94 | record (dated measurements) | with its parent: 39. The state's traffic behind `marginal`'s slope (2026-10-10) |
| 4997 | · · · Not built: a pair-major run for the unsplit model | 14 | record (dated measurements) | with its parent: 39. The state's traffic behind `marginal`'s slope (2026-10-10) |
| 5011 | · Releases compared, and slowdowns bisected | 5 | record (dated measurements) | records/PERFORMANCE-LOG.md |
| 5016 | · · 26. 0.11.1 against 0.10.0, two slowdowns fixed, and what the readers hold (2026-09-27) | 11 | record (dated measurements) | with its parent: Releases compared, and slowdowns bisected |
| 5027 | · · · The benchmark, three builds | 47 | record (dated measurements) | with its parent: 26. 0.11.1 against 0.10.0, two slowdowns fixed, and what the readers hold (2026-09-27) |
| 5074 | · · · Thread scaling | 15 | record (dated measurements) | with its parent: 26. 0.11.1 against 0.10.0, two slowdowns fixed, and what the readers hold (2026-09-27) |
| 5089 | · · · Peak memory, re-measured | 19 | record (dated measurements) | with its parent: 26. 0.11.1 against 0.10.0, two slowdowns fixed, and what the readers hold (2026-09-27) |
| 5108 | · · · The CSV and NDJSON read-ahead | 28 | record (dated measurements) | with its parent: 26. 0.11.1 against 0.10.0, two slowdowns fixed, and what the readers hold (2026-09-27) |
| 5136 | · · · The plan inspection, re-measured | 22 | record (dated measurements) | with its parent: 26. 0.11.1 against 0.10.0, two slowdowns fixed, and what the readers hold (2026-09-27) |
| 5158 | · · 28. The README's numbers, re-measured (2026-09-29) | 15 | record (dated measurements) | with its parent: Releases compared, and slowdowns bisected |
| 5173 | · · · The benchmark | 63 | record (dated measurements) | with its parent: 28. The README's numbers, re-measured (2026-09-29) |
| 5236 | · · · The Parallelism workloads | 26 | record (dated measurements) | with its parent: 28. The README's numbers, re-measured (2026-09-29) |
| 5262 | · · · The wheel's size | 16 | record (dated measurements) | with its parent: 28. The README's numbers, re-measured (2026-09-29) |
| 5278 | · · 29. The drop since 0.2.0, bisected (2026-09-29) | 10 | record (dated measurements) | with its parent: Releases compared, and slowdowns bisected |
| 5288 | · · · Method | 25 | record (dated measurements) | with its parent: 29. The drop since 0.2.0, bisected (2026-09-29) |
| 5313 | · · · The causes | 14 | record (dated measurements) | with its parent: 29. The drop since 0.2.0, bisected (2026-09-29) |
| 5327 | · · · The readiness cost: the inverse's diagonal on every solve | 33 | record (dated measurements) | with its parent: 29. The drop since 0.2.0, bisected (2026-09-29) |
| 5360 | · · · The fix: the shares when they are read | 77 | record (dated measurements) | with its parent: 29. The drop since 0.2.0, bisected (2026-09-29) |
| 5437 | · · 30. Where every row solves (2026-09-29) | 11 | record (dated measurements) | with its parent: Releases compared, and slowdowns bisected |
| 5448 | · · · Bisecting the solve | 25 | record (dated measurements) | with its parent: 30. Where every row solves (2026-09-29) |
| 5473 | · · · Where the time went | 17 | record (dated measurements) | with its parent: 30. Where every row solves (2026-09-29) |
| 5490 | · · · The change | 49 | record (dated measurements) | with its parent: 30. Where every row solves (2026-09-29) |
| 5539 | · · 31. 0.13.0 against 0.12.0 (2026-09-29) | 52 | record (dated measurements) | with its parent: Releases compared, and slowdowns bisected |
| 5591 | · · · The Parallelism figures, re-measured | 31 | record (dated measurements) | with its parent: 31. 0.13.0 against 0.12.0 (2026-09-29) |
| 5622 | · · 46. 1.4.0 against 1.3.0 (2026-11-03) | 20 | record (dated measurements) | with its parent: Releases compared, and slowdowns bisected |
| 5642 | · · · The benchmark | 115 | record (dated measurements) | with its parent: 46. 1.4.0 against 1.3.0 (2026-11-03) |
| 5757 | · · · The wide benchmark | 41 | record (dated measurements) | with its parent: 46. 1.4.0 against 1.3.0 (2026-11-03) |
| 5798 | · · · The bank's share by width | 22 | record (dated measurements) | with its parent: 46. 1.4.0 against 1.3.0 (2026-11-03) |
| 5820 | · · · On x86-64 | 20 | record (dated measurements) | with its parent: 46. 1.4.0 against 1.3.0 (2026-11-03) |
| 5840 | · · 45. 1.3.0 against 1.2.0 (2026-10-24) | 15 | record (dated measurements) | with its parent: Releases compared, and slowdowns bisected |
| 5855 | · · · The benchmark | 90 | record (dated measurements) | with its parent: 45. 1.3.0 against 1.2.0 (2026-10-24) |
| 5945 | · · · The wide benchmark | 23 | record (dated measurements) | with its parent: 45. 1.3.0 against 1.2.0 (2026-10-24) |
| 5968 | · · · Diagnostics, the query path and the bank's share | 42 | record (dated measurements) | with its parent: 45. 1.3.0 against 1.2.0 (2026-10-24) |
| 6010 | · · 44. The README's numbers, re-measured (2026-10-24) | 13 | record (dated measurements) | with its parent: Releases compared, and slowdowns bisected |
| 6023 | · · · The benchmark | 78 | record (dated measurements) | with its parent: 44. The README's numbers, re-measured (2026-10-24) |
| 6101 | · · · The recipes | 8 | record (dated measurements) | with its parent: 44. The README's numbers, re-measured (2026-10-24) |
| 6109 | · · · The bank's own share, by width | 46 | record (dated measurements) | with its parent: 44. The README's numbers, re-measured (2026-10-24) |
| 6155 | · The window operators | 7 | record (dated measurements) | records/PERFORMANCE-LOG.md; the operators' current cost is in PERFORMANCE.md: window operators, moved there from the README |
| 6162 | · · 32. Windowed means against the `rolling` recipe (2026-09-30) | 47 | record (dated measurements) | with its parent: The window operators |
| 6209 | · · 33. Window operators (2026-10-03) | 72 | record (dated measurements) | with its parent: The window operators |
| 6281 | · · 34. A window expression as a target (2026-10-03) | 55 | record (dated measurements) | with its parent: The window operators |
| 6336 | · · 35. Two costs named, one since removed (2026-10-03) | 39 | record (dated measurements) | with its parent: The window operators |
| 6375 | · · 36. The window core's edges from raw clocks (2026-10-03) | 42 | record (dated measurements) | with its parent: The window operators |

### docs/EXTENDING.md

33 sections, 778 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Adding a model | 54 | how-to, each step with what it touches | EXTENDING.md, the contributor how-to: add a model (the path stays) |
| 55 | · The recursion — `crates/online-core` | 2 | how-to, each step with what it touches | with its parent: Adding a model |
| 57 | · · Step 1 — `src/<model>.rs` | 37 | how-to, each step with what it touches | with its parent: The recursion — `crates/online-core` |
| 94 | · · · When the model holds more than a recursion | 66 | how-to, each step with what it touches | with its parent: Step 1 — `src/<model>.rs` |
| 160 | · · · When the model keeps a running mean | 18 | how-to, each step with what it touches | with its parent: Step 1 — `src/<model>.rs` |
| 178 | · · · Restoring | 25 | how-to, each step with what it touches | with its parent: Step 1 — `src/<model>.rs` |
| 203 | · · Step 2 — `src/lib.rs` | 6 | how-to, each step with what it touches | with its parent: The recursion — `crates/online-core` |
| 209 | · · Step 3 — `src/model.rs` | 11 | how-to, each step with what it touches | with its parent: The recursion — `crates/online-core` |
| 220 | · · Step 4 — `tests/model_contract.rs` | 15 | how-to, each step with what it touches | with its parent: The recursion — `crates/online-core` |
| 235 | · · Step 5 — `tests/golden.rs` | 21 | how-to, each step with what it touches | with its parent: The recursion — `crates/online-core` |
| 256 | · The bank — `crates/online-polars` | 2 | how-to, each step with what it touches | with its parent: Adding a model |
| 258 | · · Step 6 — `src/spec.rs` | 13 | how-to, each step with what it touches | with its parent: The bank — `crates/online-polars` |
| 271 | · · · A model that is not a regression | 19 | how-to, each step with what it touches | with its parent: Step 6 — `src/spec.rs` |
| 290 | · · · What a model reads for the diagnostics | 25 | how-to, each step with what it touches | with its parent: Step 6 — `src/spec.rs` |
| 315 | · · · Three arms nothing is exhaustive over | 14 | how-to, each step with what it touches | with its parent: Step 6 — `src/spec.rs` |
| 329 | · · · A block rather than a row | 8 | how-to, each step with what it touches | with its parent: Step 6 — `src/spec.rs` |
| 337 | · · · A parameter measured in clock units | 22 | how-to, each step with what it touches | with its parent: Step 6 — `src/spec.rs` |
| 359 | · · Step 7 — `src/stream.rs` | 25 | how-to, each step with what it touches | with its parent: The bank — `crates/online-polars` |
| 384 | · · Step 8 — `src/bank.rs` | 77 | how-to, each step with what it touches | with its parent: The bank — `crates/online-polars` |
| 461 | · The Python surface — `python/polars_online` | 2 | how-to, each step with what it touches | with its parent: Adding a model |
| 463 | · · Step 9 — `_spec.py` and `spec.py` | 29 | how-to, each step with what it touches | with its parent: The Python surface — `python/polars_online` |
| 492 | · · Step 10 — `tests/api_surface.txt` | 23 | how-to, each step with what it touches | with its parent: The Python surface — `python/polars_online` |
| 515 | · Tests — `tests/` | 2 | how-to, each step with what it touches | with its parent: Adding a model |
| 517 | · · Step 11 — `tests/test_<model>.py` | 14 | how-to, each step with what it touches | with its parent: Tests — `tests/` |
| 531 | · · Step 12 — Per-model sweeps | 125 | how-to, each step with what it touches | with its parent: Tests — `tests/` |
| 656 | · · Step 13 — `tests/test_model_registry.py` | 7 | how-to, each step with what it touches | with its parent: Tests — `tests/` |
| 663 | · · Step 14 — `tests/test_golden_pipeline.py` | 13 | how-to, each step with what it touches | with its parent: Tests — `tests/` |
| 676 | · Docs | 2 | how-to, each step with what it touches | with its parent: Adding a model |
| 678 | · · Step 15 — `docs/MODELS.md` and `README.md` | 30 | how-to, each step with what it touches | with its parent: Docs |
| 708 | · · Step 16 — `CHANGELOG.md` and the design note | 9 | how-to, each step with what it touches | with its parent: Docs |
| 717 | · · Step 17 — `crates/online-bench/src/scaling.rs` | 14 | how-to, each step with what it touches | with its parent: Docs |
| 731 | · · Then the gate | 12 | how-to, each step with what it touches | with its parent: Docs |
| 743 | · Adding an output or a parameter instead | 36 | how-to | contributing/add-an-output-or-a-parameter.md |

### docs/TESTING.md

46 sections, 2587 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Test coverage and testing improvements | 82 | reference | TESTING.md, the contributor reference: what the suite proves (the path stays) |
| 83 | · What the suite proves | 11 | reference | with its parent: Test coverage and testing improvements |
| 94 | · · The eight test classes | 52 | reference | with its parent: What the suite proves |
| 146 | · · Against reference implementations | 72 | reference | with its parent: What the suite proves |
| 218 | · · An oracle, not a golden number | 33 | reference | with its parent: What the suite proves |
| 251 | · · Libraries the package does not depend on | 90 | reference | with its parent: What the suite proves |
| 341 | · · The window operators and formula targets | 35 | reference | with its parent: What the suite proves |
| 376 | · · Resuming a run | 26 | reference | with its parent: What the suite proves |
| 402 | · · Beyond the eight classes | 41 | record (done items) | records/TESTING-LOG.md |
| 443 | · · · Production hardening round 3 (vs river's own battery) | 37 | record (done items) | with its parent: Beyond the eight classes |
| 480 | · · · Post-refactor hardening (P1–P8 review) | 22 | record (done items) | with its parent: Beyond the eight classes |
| 502 | · · · The validated defaults are still the measured ones | 11 | record (done items) | with its parent: Beyond the eight classes |
| 513 | · · · The declared schema is checked for every model | 14 | record (done items) | with its parent: Beyond the eight classes |
| 527 | · · · Hard rule 1 is enforced, not remembered | 10 | record (done items) | with its parent: Beyond the eight classes |
| 537 | · · · How cost grows is held, not only what it is | 37 | record (done items) | with its parent: Beyond the eight classes |
| 574 | · · · Examples are executed | 18 | record (done items) | with its parent: Beyond the eight classes |
| 592 | · · Two tiers | 97 | reference | TESTING.md: two tiers |
| 689 | · · The oracle ledger | 240 | mixed: reference, how-to | TESTING.md; updating the ledger to contributing/update-the-oracle-ledger.md |
| 929 | · · Iterating | 165 | how-to | contributing/iterate-on-a-change.md |
| 1094 | · · Worktree slots | 66 | how-to | contributing/use-a-worktree-slot.md |
| 1160 | · · Caches and when to clear them | 41 | how-to | contributing/clear-the-caches.md |
| 1201 | · · Where the suite runs | 29 | reference | TESTING.md: where the suite runs |
| 1230 | · · The machines and the CPU paths | 134 | mixed: reference, explanation (its reasons) | TESTING.md: the machines and their last bits; its reasons stay with it, as TESTING.md's second kind |
| 1364 | · · · The math library's last bits | 47 | mixed: reference, explanation (its reasons) | with its parent: The machines and the CPU paths |
| 1411 | · · What the README's guarantees are held to | 69 | mixed: explanation (for users), reference | explanation/how-the-library-is-tested.md takes what each guarantee is held to; TESTING.md keeps the contributor detail (the test files and how they run) |
| 1480 | · What it has found | 5 | record | records/TESTING-LOG.md |
| 1485 | · · Defects, and where each is told | 43 | record | with its parent: What it has found |
| 1528 | · · Differences from river that are not bugs | 11 | record | with its parent: What it has found |
| 1539 | · Where it is thin, and what is left | 7 | reference | TESTING.md: where it is thin |
| 1546 | · · What is left | 23 | reference | with its parent: Where it is thin, and what is left |
| 1569 | · · Measured coverage | 18 | reference | with its parent: Where it is thin, and what is left |
| 1587 | · · Mutation survivors | 115 | reference | with its parent: Where it is thin, and what is left |
| 1702 | · How the suite looks for defects | 7 | record | records/TESTING-LOG.md |
| 1709 | · · What the mutation run actually found | 34 | record | with its parent: How the suite looks for defects |
| 1743 | · · · The mutation blind spot | 12 | record | with its parent: What the mutation run actually found |
| 1755 | · · · Five defects the behavioural tests were happy with | 25 | record | with its parent: What the mutation run actually found |
| 1780 | · · · Do not run anything else while a mutation pass is going | 22 | record | with its parent: What the mutation run actually found |
| 1802 | · · · Where the 217 stood after run 3 | 22 | record | with its parent: What the mutation run actually found |
| 1824 | · · FFI memory and crash safety (2026-08-31) | 84 | record | with its parent: How the suite looks for defects |
| 1908 | · The entries, by ID | 9 | record (entries cited by ID) | records/TESTING-LOG.md, IDs kept |
| 1917 | · · A. Close the oracle gaps (our own references) | 73 | record (entries cited by ID) | with its parent: The entries, by ID |
| 1990 | · · B. Cross-checks against river | 56 | record (entries cited by ID) | with its parent: The entries, by ID |
| 2046 | · · C. Edge-case matrix | 146 | record (entries cited by ID) | with its parent: The entries, by ID |
| 2192 | · · D. Windows and cross-platform | 150 | record (entries cited by ID) | with its parent: The entries, by ID |
| 2342 | · · E. Infrastructure | 209 | record (entries cited by ID) | with its parent: The entries, by ID |
| 2551 | · · What the first CI runs cleared (2026-08-31 to 2026-09-06) | 37 | record (entries cited by ID) | with its parent: The entries, by ID |

### docs/RELEASE-READINESS.md

59 sections, 1983 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | Release readiness and API stability | 21 | reference | RELEASE-READINESS.md, the contributor reference for a release (the path stays) |
| 22 | · Cutting a release | 21 | how-to | contributing/cut-a-release.md |
| 43 | · · The steps of a release | 28 | how-to | with its parent: Cutting a release |
| 71 | · · The release gate | 58 | reference | RELEASE-READINESS.md: the release gate |
| 129 | · · Rehearse before tagging | 20 | how-to | with its parent: Cutting a release |
| 149 | · · Compare with the last release, bit for bit | 70 | how-to | with its parent: Cutting a release |
| 219 | · · · A released build's state files | 82 | reference | VERSIONS.md: state files across releases |
| 301 | · · The README's links on PyPI | 10 | reference | RELEASE-READINESS.md: the README's links on PyPI |
| 311 | · · The README's size row | 17 | reference | RELEASE-READINESS.md: the README's size row |
| 328 | · · The budgets baseline | 24 | reference | RELEASE-READINESS.md: the budgets baseline |
| 352 | · · The tests that hold the workflow | 12 | reference | RELEASE-READINESS.md: the tests that hold the workflow |
| 364 | · · How the release workflow came to this | 5 | record | records/RELEASE-HISTORY.md |
| 369 | · · · Why the tag comes last (2026-09-27) | 14 | record | with its parent: How the release workflow came to this |
| 383 | · · · The six faults of 2026-09-03 | 25 | record | with its parent: How the release workflow came to this |
| 408 | · · · The Linux CLI's glibc floor (2026-09-27) | 32 | record | with its parent: How the release workflow came to this |
| 440 | · Which Polars versions are promised | 16 | reference | VERSIONS.md: the Polars range |
| 456 | · · The floor and the ceiling | 82 | reference | with its parent: Which Polars versions are promised |
| 538 | · · · The suite on the floor, as recorded | 57 | record | records/RELEASE-HISTORY.md |
| 595 | · · How the pin moves | 44 | how-to | contributing/move-the-polars-range.md |
| 639 | · · Which interfaces carry a promise | 44 | explanation | explanation/polars-interfaces.md |
| 683 | · · NumPy, the one optional dependency | 25 | reference | VERSIONS.md: NumPy |
| 708 | · · Raising the ceiling to a new major | 29 | how-to | contributing/move-the-polars-range.md: a new major |
| 737 | · · The Polars pin, and the two copies of Polars (2026-08-31) | 5 | mixed: record, explanation (the reasoning) | records/RELEASE-HISTORY.md, whole; stage 5 copies its reasoning into explanation/polars-interfaces.md |
| 742 | · · · "Statically linking Polars will break users on other versions" | 32 | mixed: record, explanation (the reasoning) | records/RELEASE-HISTORY.md, whole; stage 5 copies its reasoning into explanation/polars-interfaces.md |
| 774 | · · · "A frame allocated by one binary and freed by the other" | 32 | mixed: record, explanation (the reasoning) | records/RELEASE-HISTORY.md, whole; stage 5 copies its reasoning into explanation/polars-interfaces.md |
| 806 | · · · "Why not dynamically link Polars?" | 80 | mixed: record, explanation (the reasoning) | records/RELEASE-HISTORY.md, whole; stage 5 copies its reasoning into explanation/polars-interfaces.md |
| 886 | · · Polars 2.0.0rc1, measured (2026-09-10) | 24 | record | records/RELEASE-HISTORY.md |
| 910 | · · · R6 narrows, as `docs/STATE-WORKFLOW.md` said it might | 31 | record | with its parent: Polars 2.0.0rc1, measured (2026-09-10) |
| 941 | · · · The other failure is the canary's own hygiene | 35 | record | with its parent: Polars 2.0.0rc1, measured (2026-09-10) |
| 976 | · · · What 2.0 adds that this library could use | 27 | record | with its parent: Polars 2.0.0rc1, measured (2026-09-10) |
| 1003 | · · · What this does not settle | 17 | record | with its parent: Polars 2.0.0rc1, measured (2026-09-10) |
| 1020 | · · · Polars 2.0.0 final, and the pin moved (2026-10-08) | 25 | record | with its parent: Polars 2.0.0rc1, measured (2026-09-10) |
| 1045 | · · The expression plugin, as recorded | 24 | record (a removed surface) | records/RELEASE-HISTORY.md |
| 1069 | · Keeping the API stable | 8 | reference | VERSIONS.md: the API promise |
| 1077 | · · The policy | 205 | reference | with its parent: Keeping the API stable |
| 1282 | · · What not to do | 11 | reference | contributing/change-the-api.md |
| 1293 | · · What the API actually is | 102 | reference | VERSIONS.md: what the API is |
| 1395 | · · S — The mechanism: one API snapshot test — **done** | 55 | reference | contributing/change-the-api.md |
| 1450 | · · API usability round (pre-users window) | 31 | record | records/RELEASE-HISTORY.md |
| 1481 | · · What the correlation and regime models added to the surface (2026-09) | 16 | record (dated) | records/RELEASE-HISTORY.md |
| 1497 | · CI cost while the repo was private | 11 | record | records/RELEASE-HISTORY.md, its own lines; its policy is its subsection *The policy, as the test holds it*, which goes to TESTING.md by itself |
| 1508 | · · The policy, as the test holds it | 23 | reference | TESTING.md: where the suite runs |
| 1531 | · · Actions quota: exhausted on day one (2026-08-31) | 40 | record (dated) | records/RELEASE-HISTORY.md |
| 1571 | · · CI cost: where the 80 minutes went | 32 | record (dated) | records/RELEASE-HISTORY.md |
| 1603 | · · What was unexpectedly expensive (2026-08-31) | 32 | record (dated) | records/RELEASE-HISTORY.md |
| 1635 | · · · The ubuntu `test` failure was not infrastructure | 26 | record (dated) | with its parent: What was unexpectedly expensive (2026-08-31) |
| 1661 | · · · What a push cost after the fixes | 33 | record (dated) | with its parent: What was unexpectedly expensive (2026-08-31) |
| 1694 | · Going public, as recorded | 13 | record | records/RELEASE-HISTORY.md |
| 1707 | · · Suggested order | 19 | record | with its parent: Going public, as recorded |
| 1726 | · · Before going public: R1–R6 | 22 | record | with its parent: Going public, as recorded |
| 1748 | · · · R1 — Least-privilege workflow permissions | 15 | record | with its parent: Before going public: R1–R6 |
| 1763 | · · · R2 — Pin actions to commit SHAs | 16 | record | with its parent: Before going public: R1–R6 |
| 1779 | · · · R3 — Rust crates: not published | 20 | record | with its parent: Before going public: R1–R6 |
| 1799 | · · · R4 — Repository settings | 19 | record | with its parent: Before going public: R1–R6 |
| 1818 | · · · R5 — Green CI on all three platforms first | 7 | record | with its parent: Before going public: R1–R6 |
| 1825 | · · · R6 — History scan | 17 | record | with its parent: Before going public: R1–R6 |
| 1842 | · · Open-source preparation: the full sweep (2026-08-31) | 51 | record | with its parent: Going public, as recorded |
| 1893 | · · Going public (2026-08-31) | 72 | record | with its parent: Going public, as recorded |
| 1965 | · · Repository settings, as recorded | 19 | mixed: record, reference (the current settings) | records/RELEASE-HISTORY.md, whole; stage 9 copies the current settings into contributing/repository-settings.md |

### docs/WRITING.md

17 sections, 986 lines.

| line | section | lines | kind now | lands in |
|---:|---|---:|---|---|
| 1 | How the documentation is written | 56 | reference (rules for writers) | WRITING.md, unchanged in kind (the path stays) |
| 57 | · 0. Write for one stated reader | 43 | reference (rules for writers) | with its parent: How the documentation is written |
| 100 | · 1. Every document has an altitude, and stays at it | 26 | reference (rules for writers) | with its parent: How the documentation is written |
| 126 | · 2. Plan the map, then give each section one job | 130 | reference (rules for writers) | with its parent: How the documentation is written |
| 256 | · 3. Code with comments, not prose that narrates code | 138 | reference (rules for writers) | with its parent: How the documentation is written |
| 394 | · 4. Facts: whose, how measured, and where they live | 89 | reference (rules for writers) | with its parent: How the documentation is written |
| 483 | · 5. Sentences | 147 | reference (rules for writers) | with its parent: How the documentation is written |
| 630 | · 6. The pass itself | 81 | how-to (the pass) | WRITING.md §6 stays in place, a how-to in a reference page, since every rule cites it |
| 711 | · 7. Tables, not bullet lists | 27 | reference (rules for writers) | with its parent: How the documentation is written |
| 738 | · 8. Rewrites, as examples | 6 | explanation (worked examples) | WRITING.md §8 stays in place |
| 744 | · · The rewrite of 2026-09-23 | 28 | explanation (worked examples) | with its parent: 8. Rewrites, as examples |
| 772 | · · A paragraph the user rewrote, 2026-10-04 | 43 | explanation (worked examples) | with its parent: 8. Rewrites, as examples |
| 815 | · 9. Four kinds of page | 45 | reference (rules for writers) | with its parent: How the documentation is written |
| 860 | · 10. Reference entries and docstrings | 30 | reference (rules for writers) | with its parent: How the documentation is written |
| 890 | · 11. Errors, warnings and notices | 56 | reference (rules for writers) | with its parent: How the documentation is written |
| 946 | · 12. One word for one thing | 22 | reference (rules for writers) | with its parent: How the documentation is written |
| 968 | · 13. Checks that hold the rules | 19 | reference (rules for writers) | with its parent: How the documentation is written |
