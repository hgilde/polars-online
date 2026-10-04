# The README's iterations

This record tracks each pass over the README: the prompt that started it,
word for word, the ideas it applied, the counts before and after, and the
verdict on the result. Read it to see which prompts produced a better
document, and to choose the next ideas to apply. The rules the ideas became
are in [WRITING.md](WRITING.md), and the reports behind them in
[PHRASING.md](PHRASING.md).

| section | what it holds |
|---|---|
| [How an iteration is recorded](#how-an-iteration-is-recorded) | the fields every entry carries |
| [The iterations](#the-iterations) | every pass so far, its prompt and its verdict |
| [The ideas](#the-ideas) | every idea the reviews raised, with its status |

## How an iteration is recorded

**Every pass gets an entry before it starts, and its verdict when the user
gives one.** The prompt is kept verbatim, typing slips included, so the
outcome can be traced to what was asked. The counts come from the same
script each time (WRITING §6), so two passes compare.

| field | what it holds |
|---|---|
| prompt | the user's words, unedited |
| ideas applied | the ids from [The ideas](#the-ideas) this pass took on |
| commit | where the result landed |
| counts | prose words, sentences, the mean sentence, sentences of 35+ and 45+ words, tables, and headings with no prose opener, before and after |
| verdict | the user's words on the result, unedited, or *pending* |

## The iterations

Three rewrites, two reviews and one rewrite in progress, oldest first:

| pass | date | what was asked | commit | verdict |
|---|---|---|---|---|
| I1, task 89 | 2026-09-23 | rewrite to the writing style: plan every section first, tables over bullets | `b73ab46` | "This is an improvement" |
| I2, task 138 | 2026-09-29 | rewrite against the current API and features | `80df0ac` | none recorded |
| I3, task 154 | 2026-10-03 | rewrite every reader-facing document after tasks 139 to 153 | `f6c8b4b` | the opening paragraph "very clunky now"; the `refresh_time` example needed its input described |
| R1, review | 2026-10-04 | suggest phrasing and organization changes, section by section | none: suggestions | given as R2's prompt |
| R2, review | 2026-10-04 | the same, under three new rules: openers, kinds first, nothing tacked on | `c7580e0`, `21bedd7`: the rules | given as I4's prompt |
| I4, rewrite | 2026-10-04 | keep a record of the ideas, then rewrite the README by them | pending | pending |

### I1, task 89 (2026-09-23)

**Prompt:** Rewrite the readme following the repo writing style. First plan
every section, organize the sections to put similar concepts together with
a hierarchical organization so that the table of contents contains a
moderate number of larger sections containing subsections in a. Grouping
and order that will make the most sense. Then iterate through each section
ensuring clarity and succinctness, preferring code and comment examples
wheee possible and tight and clear text where the comments would grow
unwieldy or where formatting or math text is needed. Prefer tables to long
lists of bullet points and follow all the repo writing style guides.

**Verdict:** This is an improvement rewrite the writing and phrasing doc so
that future writing is more like this

**Counts:** prose words 13,496 to 11,590; the mean sentence 23.4 to 19.9
words; sentences of 35+ words 100 to 33, and of 45+ 44 to 2; tables 15 to
37 (PHRASING, the task 89 entry).

### I2, task 138 (2026-09-29)

**Prompt:** After the gate commit rewrite the readme after analyzing the
current state of the api and features. Be sure to follow phrasing and
writing guidelines

**Verdict:** none recorded.

**Counts before:** 13,282 prose words, 709 sentences averaging 18.7 words,
34 of 35+ words and 2 of 45+, 43 tables (`docs/PLAN.md` task 138).

### I3, task 154 (2026-10-03)

**Prompt:** Rewrite each user facing non requirements Md document ensuring
that it has all the necessary sections after the recent additions. Review
the organization of the sections to keep similarities together and ensure
that important information is presented first in each section and that all
phrasing and documentation rules are followed.

**Verdict, as corrections:** The intro paragraph is very clunky now.
Section named Four words should be called Terminology. Then: The example of
refresh_time would be easier if it described ticks or showed the schema.

**Counts:** prose words 15,805 to 18,614; sentences 843 to 1,024; the mean
sentence 18.7 to 18.2 words; 35+ 18 to 5, 45+ 6 to 0; tables 49 to 60
(`docs/PLAN.md` task 154, with both corrections applied).

### R1, review (2026-10-04)

**Prompt:** Review the readme, suggest phrasing and organization changes
that would improve readability including changes to the organization of
each individual section.

**Result:** six reviewers read the README in parts. They found eight
errors, the E ideas below, and raised the phrasing and order ideas marked
R1.

### R2, review (2026-10-04)

**Prompt:** We we plan the next documentation update we ant to be sure
that important information is presented first in a section, for example we
do not want sentences at the end of a section that tack on important
details. Instead we want to pull those details into the section. Each
section should introduce its main contents in a succinct opening one to
three sentences. This library provides a lot of functionality and it is
easy to blur a lot of dense functionality into a single paragraph. The
structure should be to first briefly explain the kinds of functionality
available and then develop subsections that go into detail. Ensure that
the phrasing and writing docs contain these best practices and then go
through the readme again and suggest changes as you just did (redo the
work)

**Result:** three rules in WRITING §2 and a count in §6. The same six
reviewers redid the README against them, raising the ideas marked R2.

### I4, rewrite (2026-10-04)

**Prompt:** Keep a doc of these ideas, we will be iterating on the readme
and we want to keep track of which promos get us to a better doc. Then
rewrite the readme according to those ideas and present me the html
rendering

**Ideas applied:** every idea below marked *I4*.

**Counts before:** 18,614 prose words, 1,024 sentences averaging 18.2
words, 5 of 35+ and 0 of 45+, 60 tables, and 16 headings with no prose
opener.

**Counts after:** 22,627 prose words, 1,298 sentences averaging 17.4
words, 7 of 35+ and 2 of 45+, 112 tables, and no heading without a prose
opener. Every sentence of 35+ words, before and after, is two sentences
the splitter merged, because the second starts with a lowercase name such
as `polars-online`, `docs/…` or `sklearn`. Read one by one, both versions
have none. The python blocks went from 65 to 70, every one of them run by
the tests, and the comment lines that continue a comment (C1) from 111 to
13. The README grew from 3,889 lines to 4,789.

**What the counts show:** the openers, the kinds-first tables and the
frames each example now introduces cost about 4,000 words and 52 tables.
Two patterns the rules produced are worth the verdict's attention. Several
openers name their own subsections in prose, as the Introduction's does,
which repeats the contents table. And a section's kinds now often come as
a table of subsections before the prose.

**Verdict:** pending.

## The ideas

Every idea the reviews raised, grouped by kind. A status is *I4* when the
fourth rewrite applies it, or *later* with the reason it waits.

### Structure

| id | idea | from | status |
|---|---|---|---|
| S1 | every section and subsection opens with one to three sentences naming what it holds; sixteen headings had none | R2 | I4 |
| S2 | name the kinds of functionality first, then give each a subsection or a bold-led paragraph; split every paragraph that introduces two parameters, modes or outputs | R2 | I4 |
| S3 | pull every important detail tacked on at the end of a section or paragraph into the paragraph whose topic it is, or into the opener | R2 | I4 |
| S4 | state what every subsection obeys in the parent, before the first subsection: the window rules, the family rules, "after the last row each group learned from" | R2 | I4 |
| S5 | the most important information first: the release refusal in Saving's opener, the selection switches in a subsection of their own | R1, R2 | I4 |
| S6 | within-section order: *Output field names* before *Coefficients*; *Row order* after the forgetting subsections; windowed means before refresh time; the regime family as `seqtest`, `corrchange`, `bocpd`, `hmm`; Performance as Throughput, Parallelism, Chunk size, Memory, Tuning memory, Window operators, scikit-learn; Versioning with the action first; Databases before Pathway | R1, R2 | I4 |
| S7 | a model entry's template: when to reach for it, symbols before the rule, one-line comments, checks last | R1, R2 | I4 |
| S8 | move *What this is not* into the Introduction | R1 | later: R2 kept it in place with an opener |
| S9 | fold *Pathway* and *Databases* into *Running a bank* | R1 | later: R2 kept *Scope and integrations* with an opener |
| S10 | move most of *Testing* to TESTING.md, since WRITING §1 keeps how it is tested out of the guide | R1 | later: R2 restructured it in place; an altitude question for its own pass |

### Terms and words

| id | idea | from | status |
|---|---|---|---|
| T1 | define *model bank* where it first appears, in *The idea* | R1, R2 | I4 |
| T2 | define a term before using it: *stamp*, *stretch*, *break*, *Gram*, *slot*, *grid*, IC, vech, ARI, TVTP | R1, R2 | I4 |
| T3 | one word, one meaning: Parallelism's unit of work is a *task*, not a *stream*; `hmm`'s regime is a *hidden state*; a list of parameter values is a *grid* only where it is introduced as one | R1, R2 | I4 |

### Code and examples

| id | idea | from | status |
|---|---|---|---|
| C1 | a comment longer than its line is a concept: move it into prose or a table | R1, R2 | I4 |
| C2 | introduce every frame an example reads: `df`, `lf`, `out`, `spec` and `bank` come from the test namespace, unseen | R1, R2 | I4 |
| C3 | no name that collides: a spec named `"w"` beside a weight column `w`, `out` as a frame and later a dict | R1, R2 | I4 |

### Errors

| id | where | the error | status |
|---|---|---|---|
| E1 | *Which interfaces carry a promise* | "The third narrows the exposure" means the fourth row, the Arrow interface | I4 |
| E2 | *Windows as columns* | a VWAP's two sums do not hold decayed time to cancel | I4 |
| E3 | `holt` | "the one model that takes no features": `seqtest` takes none either | I4 |
| E4 | `kalman` | `obs_var / w` is a variance, not a precision | I4 |
| E5 | `kmeans` | the move frees the emptier of the two nearest centres, not the closer | I4 |
| E6 | *One row per finished group* | a closed group writes a row per half-life and per Gram, not one | I4 |
| E7 | *Preparing a stream* | `refresh_time` takes only `clock` and `group`, not a spec's clock keywords | I4 |
| E8 | *Versioning* | the two copies of Polars share a process; data crosses through the Arrow interface | I4 |
| E9 | *Install* | the wheel sizes are 0.12.0's | later: needs a measurement of 0.13.0's wheels |
