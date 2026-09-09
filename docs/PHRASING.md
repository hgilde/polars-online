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
