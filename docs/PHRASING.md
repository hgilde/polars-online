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

**Status:** open — checking a consequence before editing, see note.

**Note:** the heading directly above this sentence reads "Three ways to run
a model bank" and counts the Python loop, the Polars query and the
file-to-file job as the three; the fourth thing mentioned (the expression
form) is explicitly carved out with "there is also", not counted among the
three. Removing the file-to-file clause (`po.run`/the CLI) leaves two
ways enumerated under a heading that still says three — asked the user how
to reconcile that before touching the text.
