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

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

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

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

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

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

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

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

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

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

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

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

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

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

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

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

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

### README.md:269-285 ("Any row order")

> "A bank is a set of sufficient statistics, so row order reaches the fit
> only through decay. With decay off, an `ewridge` with `ridge=0` is
> ordinary least squares over every row it has seen, in whatever order
> they came: forwards, backwards or shuffled, its coefficients match
> `numpy.linalg.lstsq` to 2e-13. Memory use is proportional to the model's
> state, not to the amount of data that has passed through it — 6M rows ×
> 20 features from a parquet stream peak at 1.4 GB, against 3.97 GB for
> `lstsq` on the same rows, and the frame never has to fit. Set
> `solve_every=1000` to solve less often than every row (1.4 s instead of
> 11 s there, coefficients at most 1000 rows stale).
>
> A finite halflife with no clock discounts each row by how far back it
> sits, so the fit is a weighted least squares of *that* order. One trap:
> a huge finite halflife is not `inf`. Its solve cadence defaults to
> `halflife/50`, so `halflife=1e12` solves once, at `min_periods`, and
> never again. Say `inf`, or set `solve_every`."

**Reported:** Section "Any row order" should be renamed something like
"without a decay" explaining convergence to regression functions in
bounded memory while the clock section can explain "with a decay" for
local regression with a clock. Three sections should be together and the
without decay section should be half the number of words by cutting out
unneeded measurement specifics replaced by assurance of bounded memory.
specifics per model can go in a table of models.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:**
- The "three sections together" reading: "### Time and decay" (171,
  general concept), "### A clock that is not time" (184, the "with a
  decay" / local-regression case), and this section renamed to the
  "without a decay" case — currently these three do not sit together at
  all; "Groups, weights and warm-up", "Nulls" and "Three ways to hold a
  row back" (226–268) sit between the second and third. Grouping them
  means moving this section up, not just renaming it in place.
- Halving the length by cutting measurement specifics to a table: the two
  numeric contrasts (`1.4 GB` vs `3.97 GB` at 6M×20; `1.4 s` vs `11 s` at
  `solve_every=1000`) and the `2e-13` precision figure are the "measurement
  specifics" — these read as `ewridge`-specific (they name it directly),
  so they are exactly what the report says belongs in a model table
  instead of prose here. What should stay in prose, per the report: the
  *assurance* — memory is proportional to state, not data, whatever the
  amount of data — without ewridge's own numbers backing it.
- **Not addressed by the report, flagged rather than assumed:** the
  section's second paragraph (finite halflife, no clock — order *does*
  matter there, and the `halflife=1e12` solve-cadence trap) is a third
  case, not "without a decay" at all — it is decay *on*, clock *off*. If
  this section becomes strictly the "without decay" counterpart to "A
  clock that is not time", this paragraph needs its own place; it does not
  fit either half of the proposed with-decay/without-decay split as
  named.

### README.md:297-303 ("Mistakes are named")

> "A builder checks each keyword against its type hints —
> `halflife="10"` says `spec "m": halflife must be a number or a list of
> numbers, got str '10'`. A missing column says which spec wanted it, in
> what role, and what the frame has. A spec named like an input column is
> refused rather than silently replacing it."

**Reported:** The section Mistakes are named should be reduced to two
sentences at most and added to the quick start. Users will see this level
of checking anyway.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:** "## Quick start" (line 137) is a single code block plus one
short paragraph today, no error-handling content at all — this would be
the first mention of validation there. A two-sentence compression, offered
as one option and not a decision: "A builder checks every keyword against
its type hints and names a missing column by which spec wanted it, in
what role. Errors name the mistake and the frame, not a stack trace." Three
concrete cases collapse to two general claims (type-checked keywords,
named missing columns) and drop the third (a spec named like an input
column) entirely — worth confirming that is an acceptable loss, or picking
a different pair to keep, at the rewrite.

### README.md:325-330 ("In a loop: `ModelBank`")

> "A bank says what it holds: `repr(bank)` is
> `ModelBank(['ridge'], groups=412, rows_seen=3000000)`, `bank.specs` gives
> back the spec dicts, and `bank.groups()` is a frame of every `(spec,
> group)` with its row count and last clock value. Groups live until
> dropped, so a long-running bank forgets the quiet ones with:"

**Reported:** The section starting with A bank says what it holds should
just be sample code.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:** candidate, one option and not a decision — fold the prose into
the existing code block (or a small one right after it) as comments
instead of running text:

```python
repr(bank)      # ModelBank(['ridge'], groups=412, rows_seen=3000000)
bank.specs      # the spec dicts back
bank.groups()   # one row per (spec, group): row count, last clock value

# Groups live until dropped -- a long-running bank forgets the quiet ones:
stale = bank.groups().filter(pl.col("last_clock") < now - 30 * 86400)
bank.drop_groups(stale["group"])           # they start cold if they reappear
```

The one thing this drops that prose currently states outright: *why* a
long-running bank would want to drop groups at all (unbounded key space,
memory). Worth a one-line comment or keeping one clause of prose above the
block, rather than losing the reason along with the explanation.

**Clarification:** the code may carry comments — the objection is to dense
running prose as the vehicle, not to explanation existing at all. So the
"unbounded key space" reason belongs as a comment in the code (as sketched
above: "Groups live until dropped -- a long-running bank forgets the quiet
ones"), not dropped and not kept as a separate paragraph of text.

### README.md:336-391 ("As a query: `lf.online.fit_predict`")

> "The loop above as a `LazyFrame`. Executing it — `collect()`,
> `collect_batches()`, any `sink_*()` — streams the plan's rows through a
> fresh bank in `chunk_rows` chunks, so the query stays at *state + one
> chunk* however long the stream, and everything after the bank is
> ordinary polars:"

**Reported:** The text "The loop above as a LazyFrame" makes little sense,
no one is thinking of the query as a loop. This section should also be
code with comments. Maybe even code with output.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:** two asks — (1) the specific opening sentence, which also has a
standing problem independent of this report: an earlier entry in this log
(README.md:308, "As a query" reordered ahead of "In a loop") already
means "the loop above" would not even be *above* any more once that
reorder lands, so the sentence needs to change regardless of this report.
(2) the same section-wide ask as the previous two entries — the "Things
worth knowing about the plan" bullets (purity, `save_state` timing, filter
placement, filter-before-vs-after) are prose explaining what code does;
this is the third section in a row asked to become code-with-comments
rather than prose (after "A bank says what it holds" and "the loop above"
line here). Worth treating as one style decision for the whole "Running a
bank" section at the rewrite — code, comments, and where useful actual
printed output — rather than deciding each subsection separately.

**Clarification, the style decision:** not all-code — converting most of
the bullets to code-with-comments frees room to keep the couple that
matter most as text (the filter-before-vs-after advice is named
specifically as important enough to keep). Text that stays must (a) follow
good writing practice on its own terms — the standard already on file in
[[readme-voice-plain]] and applied throughout this log — and (b) not
substitute for the code-with-comments, i.e. not re-explain in prose what a
comment already shows; text is for advice code-with-comments cannot carry
by itself (a *reason* or a *warning*, not a restatement of *what*). Which
specific bullets stay as text is not decided here — "the filter advice"
is named as an example of the kind that qualifies, not necessarily the
final list.

### README.md:486

> "The numbers are the bank's — the expression *is* the bank, run over the
> column polars hands it."

**Reported:** This phrase is not normal English.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:** the awkwardness is likely the opening clause — "The numbers are
the bank's" is a possessive standing in for "the numbers come from the
bank" or "these are the bank's numbers," compressed to the point of
reading unnatural, then stacked with a second em-dash clause and an
italicized *is* in the same sentence. One candidate, not a decision:
"These are the bank's numbers: the expression *is* the bank, run over the
column polars hands it." The point worth keeping either way — the
expression *is* the model, not a wrapper around one — is not in question,
only the sentence carrying it.

**Broadened:** the next report widens this from the one sentence to the
whole section — see below, which supersedes the "candidate" above (the
"expression *is* the bank" point itself is judged not to belong in an
intro at all, so there is no longer a sentence to repair, only one to
cut).

### README.md:471-503 ("The expression form (in memory only)"), whole section

**Reported:** The fact that the expression is the bank is a detail that
doesn't matter for introduction to the library.  This whole section is
better communicated as simple code examples with a couple sentences
mentioning that this syntax triggers polars to read all the data upfront.
No need to go into the details of why in this doc, deep details can go
somewhere else.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:**
- Cut entirely from the intro-level treatment, per the report: "the
  expression *is* the bank" (the entry above), and the mechanism —
  "polars gives a stateful user expression its whole column at once, in
  either engine, so wrapping the expression in a lazy query does not make
  it stream." What stays: the code example, and "this syntax reads all
  the data upfront" as a consequence stated plainly, no *why*.
- This changes an earlier resolution already on file: the "Three ways"
  fix recorded near the top of this log (README.md:65-68/71) drafted
  replacement wording that *keeps* this same mechanism sentence almost
  verbatim ("polars hands a stateful expression its whole column at once,
  in either engine, so it reads the entire input into memory regardless
  of how the query around it is written"). That draft needs revisiting
  against this report at the rewrite — likely trimmed to the same
  consequence-only statement this entry asks for.
- "Somewhere else" for the deep details substantially exists already:
  `docs/PLAN.md` §6 is already linked at the end of this section ("has the
  design and the condition under which the warning would go away"), and
  `docs/PERFORMANCE.md` covers the same mechanism from the measurement
  side. So this is mainly a cut from the README, not a new document to
  write — worth confirming those two are judged sufficient rather than
  needing a more approachable, user-facing write-up of the mechanism
  somewhere.
- The 7.3 GB vs 1.35 GB peak-memory contrast is a specific number, not the
  mechanism explanation — not named in the report either way; per the
  pattern in the "Any row order" entry above (measurement specifics move
  to a table, the assurance stays in prose), this may be a candidate for
  the same treatment rather than automatic removal.

### README draft, round 1 (the whole draft; reported on the review page, not a committed line)

**Reported:** This shows some improvement, the sections are more organized
and focused on their topic. It is getting better but you still rely on a
some specific knowledge to make sense of it. You still refer to a bank,
not as a model bank and without highlighting the model bank concept or
defining what is a bank. How much knowledge is required to know you mean
model bank there? More knowledge than the casual browsing reader has. Also
you say things like "streams the plan's rows through a fresh bank " when
the casual reader does not know what is a plan, what makes a bank fresh or
not or even what is a model bank really. Are you able to take this
feedback and try again, using special care to try not to use words that
make sense to you but are unknown to the reader with less context.
Imagine that the readers context is knowledge of statistics, a little
polars, time ordering  but no deep understanding of polars or the
intervals of our project.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:** this is a rule the first round of `docs/WRITING.md` did not
have and the strongest one so far: a *stated reader*, and a term-of-art
audit against that reader. The words the round-1 draft used without
defining, found by reading it as that reader: *bank* (never "model bank"
after the first mention, never defined), *spec*, *state*, *stream*,
*chunk*, *plan*, *fresh* (bank), *sink*, *collect*, *struct column*,
*out-of-sample*, *n_eff*, *accumulator*, *sufficient statistics*,
*halflife* (assumed to mean exponential weighting), *clock*, *session*,
*warm-up*. Each is either the library's own word or deep-Polars
vocabulary; a reader with statistics, a little Polars and time-ordered
data has none of them. Round 2 defines the library's words once, up front,
and again where each concept lives, and replaces the Polars-internal
words with what they do.

### README.md:97 ("State is a file")

> "**State is a file.** Save a bank; load it to keep learning, or to serve
> predictions without learning."

**Reported:** You say State is a file but state is not a file, it can be
saved and loaded from a file. The average reader will think that state is
literally a file until they figure out otherwise.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:** present in the actual README.md, not only the review drafts —
this is a real fix, not something introduced while redrafting. The
introduction's own glossary (added in round 2) already gets this right
one entry down — *state: everything a bank has learned* — so "State is a
file" as a heading contradicts the definition the reader was just given
two lines above it. A fix in the shape the report asks for: state is what
a bank has learned, and it can be saved to a file and loaded back — the
heading names the capability, not an identity.

### README.md:588-590 ("`bank.specs` is a copy, and read-only")

> "`bank.specs` is a **copy, and read-only**. The bank's behaviour comes
> from the state built at construction, so a list on the Python side could
> only ever disagree with it — and used to: editing
> `bank.specs[0]["features"]` in place left `coef()` labelling
> coefficients from a spec the bank was not running."

**Reported:** Alamo [sic] you sometimes write sentences like the following
which tell a story you remember and have documented from development but
is not interesting to a new user only you when developing the code : a
list on the Python side could only ever disagree with it — and once did:
editing bank.specs[0]["features"] in place left coef() labelling
coefficients from a spec the bank was not running.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:** a pattern to watch for beyond this one sentence: a past-tense
bug narrative ("and once did", "and used to") told from the author's
memory of a fix, where the reader only needs the current rule and the
reason for it — *why* `bank.specs` is read-only and a copy, not the
history of a bug that motivated making it so. The reason itself
(coefficients would be labelled from a spec no longer running) is worth
keeping; the anecdote framing ("this used to be possible and broke
something") is what does not belong in front of a new reader. Worth
checking the rest of the README for the same shape — a fix framed as a
memory of development rather than as a stated rule — since this is
probably not the only instance.

### Whole document — prose that should be code with comments

**Reported:** Overall a lot of prose would be simpler as code examples
with comments. If a section of prose is just describing what parameters
do or what comes out of a structure, or how to use an api, turn that into
a code example and keep only the concepts that would produce comments
that are too long for the remaining prose.

**Status:** fixed (the README rewrite of 2026-09-09; the rule it produced is in `docs/WRITING.md`).

**Note:** generalizes `docs/WRITING.md` rule 3 ("Code with comments, not
prose that narrates code"), which the log so far applied only to the
"Running a bank" section, into a whole-document rule with a sharper test:
convert prose to a code example whenever it is *describing* — a
parameter's meaning, a structure's fields, how to call something — and
keep prose only for what a comment cannot hold at comment length (a
reason, a warning, a trade-off). By that test, candidates well beyond
"Running a bank": the model sections' parameter write-ups ("Constrained
coefficients" under `sgd`, "Reverting coefficients" under `kalman`,
`ew_class`'s "Choosing the shape", `micro`'s "Choosing `eps`"), the
output-record field tables already in table form but introduced with a
paragraph of description first, and "A state file describes itself" /
"Reading a state without this library", which already lean
code-with-comments but still carry descriptive prose alongside it. Not
resolved here which of these the rule actually reaches — that is for the
next synthesis pass, and some (the field tables) may already satisfy the
spirit of the rule in table form rather than needing conversion to code.

### README.md:35-41 ("The idea")

> "That order is what makes every prediction honest — no row's own outcome
> is in the number predicted for it — and it is what lets the whole thing
> run on far more rows than fit in memory:"

**Reported:** This phrase is not true. It implies that the ordering is
what makes the algorithm bounded in memory while in fact the ordering is
only needed if there is a decay specified: That order is what makes every
prediction honest — no row's own outcome is in the number predicted for
it — and it is what lets the whole thing run on far more rows than fit in
memory:

**Status:** fixed (commit following this entry).

**Note:** a correctness finding, and the same shape as "the bank is plain
least squares" earlier in this log — an effect attributed to the wrong
cause. Two things were fused by one "and": the predict-then-learn order
(which is what makes a prediction honest) and bounded memory (which comes
from the models keeping only their state, never the rows — true in any
order). The paragraph also said "in a single pass over your rows, in time
order" as if time order were always required; it is required only when a
decay is on, and the README's own "Without a decay" section says so. The
fix separates the three claims and states each with its own cause, and
adds to `docs/WRITING.md` rule 4: an effect is attributed to its own cause,
not to whatever the sentence happened to be about.

### README.md:43-45 ("The idea", last sentence)

> "without one, a model that solves or accumulates gives the same answer
> in any order"

**Reported:** This phrase lacks context, without one of what? "without
one, a model that solves or accumulates gives the same answer in any
order"

**Status:** fixed (commit following this entry): "without a decay".

**Note:** an instance of a rule already on file — `docs/WRITING.md` rule
5, a back-reference must point at something the reader has actually just
seen. "One" stood for "a decay" across a semicolon and a full clause, far
enough that it read as pointing at nothing. The noun is repeated; no new
rule.
