# Documentation phrasing

A running list of specific phrasing problems in the shipped docs — the
README, docstrings, `docs/*.md` guides — collected as they are found, for
the eventual rewrite pass (`docs/PLAN.md` task 68, the README clarity pass,
generalizes to every doc once this list is long enough to show the pattern).

This is not a style guide; the standard is already stated: one idea per
sentence, a sweep belongs in a table, name the mechanism rather than
alluding to it (can a sentence be understood by someone not already told
what it is about?). What belongs here is instances — a specific quoted
phrase that fails the standard, where it is, and (once decided) what it
became.

## Format

One entry per problem, as the user reports it, in this shape:

```
### <file:line or section>

> <the phrase in the docs, quoted verbatim>

**Reported:** <the user's own words for what's wrong, verbatim and
unedited — not a paraphrase or an interpretation of the complaint>

**Status:** open | fixed (commit) | declined (why)
```

The `Reported` line is the user's original issue, kept exactly as given,
word for word — never adjusted, summarized, or reworded to fit how it was
eventually understood or fixed. If a fix or a reading of the complaint is
worth recording, it goes in a separate `**Note:**` line underneath, so the
original report is never edited to match it.

## Log

### README.md:65

> **Three ways to run a bank, same numbers from each.**

**Reported:** this phrase has multiple meanings to the casual reader. there
has been no discussion of signals banks before in this doc and signal banks
are uncommon enough that we need at least to state that "Three ways to run
a signal bank" and the words signal bank should link to the signalbank api
to make it totally clear.

**Status:** fixed (`README.md:65`, "Three ways to run a [model bank](...)").

**Note:** the library has no API named `SignalBank` / `signalbank` — the
class this sentence refers to is [`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank).
Flagged back to the user rather than added a link to a page that does not
exist; asked, and the user chose "model bank", linked to `ModelBank`'s docs
the same way the report asked for "signal bank" to link to its API.

### README.md:65-68

> A Python loop over chunks (`ModelBank`); a Polars query
> (`lf.online.fit_predict(specs)` is a `LazyFrame` you `collect`, `sink` or
> batch like any other); or a file-to-file job (`po.run(...)` from Python,
> or the `online` CLI from a TOML with no Python at all).

**Reported:** The sentence beginning with A Python loop over chunks.  This
is a polars library by name, the more polars centric phrasing should come
first followed by the loop phrasing. Leave out po.run and the rust
instructions from this document.

**Status:** open — analyzed and a resolution agreed below, but **not
applied**: the user chose to batch phrasing issues here for one rewrite
pass rather than fix them one at a time, so this waits with the rest.

**Note:** the heading directly above this sentence reads "Three ways to run
a model bank" and counts the Python loop, the Polars query and the
file-to-file job as the three; the fourth thing mentioned (the expression
form) is explicitly carved out with "there is also", not counted among the
three. Removing the file-to-file clause (`po.run`/the CLI) leaves two
ways enumerated under a heading that still says three — asked the user how
to reconcile that before touching the text.

**Resolution, asked and answered, to apply at the rewrite:**
1. There are still three: the Polars query, the Python loop, and the
   expression form (which the user confirmed is one of the three, not the
   carved-out fourth thing — "one of the ways is polars syntax that does
   not stream").
2. `po.run` and the `online` CLI move out of README.md entirely, into a new
   `docs/RUNNER.md`, along with "The runner." (Parallelism's architecture
   write-up on the shared three-stage pipeline) — one document for
   everything about running a job outside a live Python loop, per the
   user's choice between one document and two. Every other README mention
   of `po.run`/the CLI (the save-state bullet under "As a query", the
   memory table's file-in/file-out row, the closed-groups example, the
   Parallelism section's "The runner." and "Python." bullets, the Chunk
   size and Versioning sections) gets a short cross-reference to
   `docs/RUNNER.md` in place of the detail, not a bare deletion — checked
   against `tests/test_production_hardening.py`'s README-block-execution
   test and `tests/test_llms_txt.py`'s anchor checks, neither of which this
   entry's fix breaks as scoped.

### README.md:71

> "There is also an expression form for a frame in memory — it cannot
> stream, and it says so."

**Reported:** This sentence is too ambiguous and should either be fully
explained with an example. The statement "it cannot stream" relies on the
user to know that polars collects aloads non streaming plugins that do not
stream. Add a few words to explain the implication of not streaming - for
example "this syntax cannot stream in chunks and will read all the input
data to memory"

**Status:** open — not applied yet, batched for the same rewrite pass as
the entry above (this sentence folds into that paragraph's restructuring,
not a fix standing on its own).

**Note:** drafted and reverted on request, to be re-applied at the rewrite:
"That third form cannot stream in chunks: polars hands a stateful
expression its whole column at once, in either engine, so it reads the
entire input into memory regardless of how the query around it is
written — every call warns (`InMemoryExpressionWarning`) to say so."

### README.md:72

> "a ceiling on gaps, session boundaries, a policy for a clock that runs
> backwards"

**Reported:** relies on the user to understand what these things mean.
Even the word session is not a common concept outside of capital markets.
We should have a clear section in the readme on time. This phrase should
become something like  "handling for market-data-like sessions including
session boundaries, clock gaps and clock resets between sessions. "

**Status:** open — not applied, batched for the rewrite pass.

**Note:** two distinct asks here — (1) this specific phrase, drive-by list
of terms with no gloss, and (2) a structural one, a dedicated "Time" section
in the README (the concepts are currently spread across this "Time, built
in" paragraph, "A clock that is not time", "Groups, weights and warm-up",
and "How a bank sees a stream" generally). Both belong to the same rewrite
pass; the structural one is worth flagging back to the user for scope
before it is folded in, since it is more than a sentence-level fix.

### README.md:73

> "One state per group, row weights, warm-up thresholds."

**Reported:** This is in the section on time but is not related to time
except for warmup thresholds.

**Status:** open — not applied, batched for the rewrite pass.

**Note:** the same "Time, built in" paragraph this entry and the one above
both point at — a second, independent problem with it (misplaced content,
not unclear wording), which supports the structural fix proposed above:
group/weight/warm-up material wants its own place (or the section this
sentence already sits beside further down, "Groups, weights and warm-up"),
separate from what is actually about the clock.

### README.md:74-76

> "Or no clock at all: then row order is the clock — and with decay off,
> the bank is plain least squares over everything it has seen, in any row
> order."

**Reported:** This phrase is under a section on time and at the end, a
brief mention that you do not need time. Also the bank is not least
squares. We need to be clear per model what happens if no time is given
and if no decay is given. This section is really about incremental
updating (although this phrase may itself need work). We may use decay
with time and ordering, some models may converge to least squares given no
decay and then ordering is not important. We need to rewrite the section
on incremental updating to state the various ways it can work up front.

**Status:** open — not applied, batched for the rewrite pass; this one is
a correctness finding as well as phrasing, see note.

**Note:** the "not least squares" objection checks out against the
README's own more careful statement of the same fact, in "Against
scikit-learn": *"the primary regression here is a different algorithm
class — `ewridge`, `rls`, `lasso`, `huber` and `quantile` accumulate
sufficient statistics and solve ... with decay off `ewridge` is ordinary
least squares ... in any row order."* That is five model families out of
twenty, and only because they solve a normal equation; `sgd`, `pa`, `ftrl`,
`kalman`, `holt`, the clustering and regime models are sequential
(gradient steps, a filter's update, a distance to the nearest centre) and
stay row-order-dependent whether or not decay is on. "The bank is plain
least squares" at README.md:76 states as a property of every model what is
true of five of them. This is the same root problem as the two entries
above it: the "Time, built in" paragraph is trying to cover ground —
per-model convergence and order-dependence — that the reported structural
fix (a section on incremental updating, stated per model, up front) is
the right place for, not a closing clause of a paragraph about the clock.

**Design note, refining the structural fix above:** the incremental-updating
section should not carry a per-model table — that moves into each model's
own description instead ("we will incorporate the use of decay into the
model description itself later"). What belongs here is the general idea and
API: what a clock and decay are, the vocabulary (`halflife`, `max_dclock`,
`session`, `group`, row weight, `min_periods`), and which behaviors are
universal versus per-model (the least-squares convergence above is one
example of a per-model fact that does not belong here). One idea to add
that is not in the current paragraph at all: decay lets some models produce
a *local* result over a long ordered stream — a halflife as a bandwidth —
without refitting many windowed models, which is the same idea the README
already has under "A clock that is not time" (the LOESS-in-one-pass point)
and belongs stated as a general capability of this section, not left to
that one example.

**Design note, second refinement — where this content actually lives:**
not a new section. `## How a bank sees a stream` already exists further
down the README (currently: "Time and decay" with the parameter table,
"A clock that is not time", "Groups, weights and warm-up", "Nulls", "Three
ways to hold a row back", "Any row order", "Two guarantees", "Mistakes are
named" — "these parameters are shared by every model" is already its own
opening line). The "Time, built in" paragraph these entries are about
(README.md:71-76, in "## What you get") is a short teaser in the opening
overview, ahead of that fuller section; the fix folds its content, and the
misplaced "One state per group, row weights, warm-up thresholds" clause,
into the existing section rather than standing up a separate one. What (if
anything) stays behind as a one-line teaser in "## What you get" is an open
question for the rewrite, not decided here.

### README.md:37-112 (the whole "## What you get" section)

**Reported, verbatim, the whole instruction:** All the text in what you get
should be shortened and moved above the table of models. This will become
the introduction to the library. Any content that we remove from here
should have a place somewhere else. In this section we have the basic api
example, concepts of streaming, ordering, clocks, decay and convergence,
prediction and chunking guarantees. Testing should be very short in this
section, simply to convey the large scale of the testing. We should
mention that state can be saved to a file but the rest of the state
paragraph is too much for an intro. The section on Introspection and
diagnostics has the right amount of information for an intro but should
state that they are all O(state) in memory, with documented complexity in
time. The section on Parallel by group, deterministic by construction is
too complex, if correct we should
Simply mention that we are parallel within each chunk and a teaser on
parallel performance impact - performance improves by x on average using 5
threads or whatever we know.

**Status:** open — not applied, batched for the rewrite pass.

**Note, piece by piece, checking each removed piece has a place to land:**

- **The whole section moves above the model table**, becomes the library's
  introduction. The model table itself (`## What you get`'s current first
  half) stays below it, presumably still titled something, but that title
  and its exact boundary are not specified here — worth pinning down at
  the rewrite.
- **The basic API example** — not currently in this section at all; the
  README's example is "## Quick start" (line 137), right after this
  section today. Reads as: fold Quick Start's code into the new
  introduction, or keep it as its own following section — not specified,
  worth confirming before writing.
- **Streaming, ordering, clocks, decay, convergence** — matches the four
  entries directly above this one: this content (currently "Three ways...",
  "Time, built in", "The clock does not have to be a time") folds into
  "## How a bank sees a stream" (line 167), not into the new intro,
  *unless* this instruction means a short mention stays in the intro too
  with the detail there — the four entries above assumed the detail moves
  out entirely; this instruction says the intro should still *cover the
  concepts*, just shortened. These two directions want reconciling at the
  rewrite: how much of "clocks, decay, convergence" stays as a sentence in
  the intro versus moves wholesale.
- **Prediction and chunking guarantees** — "## Two guarantees" (line 287)
  is already exactly this, already short (two sentences); likely stays
  close to as-is, folded into the intro rather than left as a numbered
  guarantee list further down, or kept in both places at different
  lengths — again a call for the rewrite.
- **Testing** — full detail already lives in "## Testing" (line 2612): about
  650 Rust tests, ~2,200 pytest cases, oracles, `river` cross-checks,
  hypothesis, golden numbers, invariants, hardening. The intro's version
  (currently a full paragraph) shrinks to a scale statement only, per the
  report.
- **State can be saved to a file** — full detail already lives in
  "## Saving, loading and serving" (line 526): atomicity, cross-entry-point
  byte identity, host-independence. The intro keeps one clause, drops the
  rest.
- **Introspection and diagnostics** — user says the current intro paragraph
  is already the right length; add one thing: "O(state) in memory, with
  documented complexity in time." Checked whether that second half is true
  today — **it is not, cleanly**: no single table states each diagnostic's
  time cost the way `docs/OUTPUTS.md` does for output fields; time
  complexity is scattered through `docs/PERFORMANCE.md`'s prose (`O(1)` for
  decay, `O(k)` for a few named things, no per-diagnostic accounting).
  Either this becomes true before the rewrite ships the claim, or the
  claim is softened — flagging rather than deciding.
- **Parallel by group, deterministic by construction** — judged too complex
  for an intro; shrinks to "parallel within each chunk" plus one throughput
  number. Full detail already lives in "## Parallelism" (line 2142), which
  has real numbers to pull the teaser from rather than invent one: **8.0×**
  on a 14-core machine (k=20, 64 groups, 1→14 threads), eight single-group
  specs in **130 ms against 515 ms** one at a time. "5 threads" in the
  report is a placeholder ("or whatever we know") — the measured points are
  at 1, 2, 4, 8 and 14 threads (1.02M/1.91M/3.52M/6.44M/8.20M rows/s), not
  5, so the teaser number should be picked from what is actually measured,
  not interpolated.

### README.md:184-220 ("A clock that is not time")

> "`pred_y` is the fitted curve read at each row's own `x0`, and `coef` is
> the line through that row's neighbourhood. These are the batch numbers:
> against a kernel-weighted least squares recomputed from scratch at every
> row they agree to 1e-12 (`tests/test_ewridge.py`,
> `test_a_feature_as_the_clock_is_a_local_linear_regression`)."
>
> "Three things carry over unchanged. `features` need not include the
> clock column — with other features the same fit is a varying-coefficient
> model, whose coefficients move along the clock. `group` gives one local
> fit per key from the same pass. And the clock belongs to every model,
> not to `ewridge`: clocked on a feature, `ew_cov` reports moments and
> correlations local in it, and `marginal` does the same pair by pair."

**Reported:** Section "A clock that is not time" the section starts
talking about a clock and then drifts into the specifics of a model
implementation and testing.  We should keep sections focussed on their
headline goals. Since this section is about the clock and within a
heading about the stream, model specific discussion should be avoided
unless they illustrate points about the clock and the stream that are
very hard to make otherwise. The paragraph starting "Three things carry
over unchanged." Focuses on grouping and while it contains good
information about local results using decay that is sitting among a
discussion of state that is not focussed on the section.

**Status:** open — not applied, batched for the rewrite pass.

**Note, what drifts and where the drifted material might go:**
- The test citation (`tests/test_ewridge.py`,
  `test_a_feature_as_the_clock_is_a_local_linear_regression`) is the
  clearest case of the complaint — a specific pytest test named inside a
  conceptual section. It backs the accuracy claim ("agree to 1e-12"),
  which is a claim worth keeping; the test name itself is not needed for
  the reader here and reads as implementation detail bleeding through.
- "Three things carry over unchanged" mixes three different claims under
  one heading that don't share the section's focus: (1) `features` need
  not include the clock column — this one is about the clock, arguably
  belongs; (2) `group` gives one local fit per key — this is about
  grouping, and "### Groups, weights and warm-up" is the *very next
  section*, so it is not just off-topic but pre-empts content that already
  has a home one heading down; (3) the clock applies to `ew_cov` and
  `marginal` too, named specifically — a model-specific illustration of a
  clock-general point, which is closer to the carve-out the report allows
  ("unless they illustrate points ... very hard to make otherwise") than
  the grouping sentence is.
- Not flagged by the report but adjacent: the code block's
  `max_rows_between_solves=1` and `min_periods=10.0` are `ewridge`-specific
  knobs inside an example that is otherwise about the clock generally: an
  example needs a real spec to run, so some model-specific surface is
  probably unavoidable here — left for the rewrite to judge against the
  report's own carve-out, not decided as a violation here.
- The `sin(x)` bandwidth/lag numbers (0.08 against 0.39, 0.29 at bandwidth
  1.0) were not named in the report. They read as exactly the carve-out's
  exception — a concrete point about the clock (one-sidedness, the
  lag/noise trade-off) that is hard to make without a number — so likely
  intended to stay, but not confirmed; flagging rather than assuming.

**Clarification, narrowing the complaint:** an illustrative example that
uses real models is fine — "that's what the library does" — so the code
block and its `ewridge`-specific kwargs above are not the concern; that
uncertainty is resolved. The concern is specifically prose like "`group`
gives one local fit per key from the same pass. And the clock belongs to
every model, not to `ewridge`: clocked on a feature, `ew_cov` reports
moments and correlations local in it, and `marginal` does the same pair by
pair" — three model names carrying only a little actual information about
the clock. So the working distinction for the rewrite is: a **runnable
example** naming a model is welcome; a **sentence surveying several models
by name** to make one clock-point is the thing to cut or compress down to
the point itself.

### README.md:226-243 ("Groups, weights and warm-up")

> "| `targets`, `features` | column names, ≥1 target; ... |
> | `add_intercept` | default `True` |
> | `group` | one state per key |
> | `group_close` | ... |
> | `weight` | row weight column |
> | `min_periods` | in `n_eff` units; outputs are null until it is reached.
> A list gives one threshold per target. Warm-up gates output, not
> learning |
> | `coef_every` | snapshot the coefficients every N rows ... |
> | `label_delay` | hold each row back from *learning* ... |
>
> `n_eff` is the exponentially weighted observation count: the weight
> behind the state that produced *this row's* prediction, measured before
> the row's own update and decay. So it is `0` on a stream's first row,
> lags the row count by one, saturates at `1 / (1 − λ)`, and means the
> same thing in every model — which is what makes `min_periods` portable
> across a bank."

**Reported:** Under "Groups, weights and warm-up" there is a section about
model parameters that, while useful, is not part of a focussed
explanation. Also groups has nothing to do with the paragraph of text in
this section. The section is about warmup and contains good information
except that it should start with something like: min_periods allows a
model to output only when it has seen enough data to converge, avoiding
uninformed decisions. min_periods is in neff units and …

**Status:** open — not applied, batched for the rewrite pass.

**Note:** two findings, matching the report's two sentences —
1. The table's eight rows are not "groups, weights and warm-up": `targets`/
   `features` and `add_intercept` are neither; `coef_every` and
   `label_delay` are neither (label_delay is its own README subsection
   already, "Labels that arrive late" — this table is a second, thinner
   description of the same parameter). Only `group`, `group_close`,
   `weight` and `min_periods` actually match the heading.
2. The prose paragraph after the table is entirely about `n_eff` (warm-up);
   it says nothing about groups or weights, confirming the heading
   promises three things and delivers prose for one. The user's suggested
   opening — "`min_periods` allows a model to output only when it has seen
   enough data to converge, avoiding uninformed decisions. `min_periods`
   is in `n_eff` units and …" — leads with *why* before the units, where
   today's paragraph leads with `n_eff`'s definition and never states the
   plain-language reason for warm-up at all.

This implies the fix is a real split, not a rename: a focused warm-up
subsection (`min_periods`, opening with the reported sentence, then the
existing `n_eff` paragraph), with `group`/`group_close`/`weight` given
their own focused treatment (possibly folded into the surrounding stream
section per the earlier entries in this log), and `targets`/`features`/
`add_intercept`/`coef_every` relocated to wherever they actually belong —
not decided here, left for the rewrite.
