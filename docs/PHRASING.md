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

*(nothing logged yet — entries are appended here as they are reported)*
