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
script each time (WRITING §6), so two passes compare. Since 2026-10-04 that
script is in the repository, `scripts/doc_review.py`, beside
`scripts/doc_structure.py`, so a pass in a new session takes its counts the
way every earlier pass did.

| field | what it holds |
|---|---|
| prompt | the user's words, unedited |
| ideas applied | the ids from [The ideas](#the-ideas) this pass took on |
| commit | where the result landed |
| counts | prose words, sentences, the mean sentence, sentences of 35+ and 45+ words, tables, and headings with no prose opener, before and after |
| verdict | the user's words on the result, unedited, or *pending* |

## The iterations

Seven rewrites, the last three tests, and two reviews, oldest first:

| pass | date | what was asked | commit | verdict |
|---|---|---|---|---|
| I1, task 89 | 2026-09-23 | rewrite to the writing style: plan every section first, tables over bullets | `b73ab46` | "This is an improvement" |
| I2, task 138 | 2026-09-29 | rewrite against the current API and features | `80df0ac` | none recorded |
| I3, task 154 | 2026-10-03 | rewrite every reader-facing document after tasks 139 to 153 | `f6c8b4b` | the opening paragraph "very clunky now"; the `refresh_time` example needed its input described |
| R1, review | 2026-10-04 | suggest phrasing and organization changes, section by section | none: suggestions | given as R2's prompt |
| R2, review | 2026-10-04 | the same, under three new rules: openers, kinds first, nothing tacked on | `c7580e0`, `21bedd7`: the rules | given as I4's prompt |
| I4, rewrite | 2026-10-04 | keep a record of the ideas, then rewrite the README by them | `4bcd286` | the Introduction's opener is "like a small table of contents"; the first fit should come first, show its tables, and include a forward `rewm_mean` target; *What a spec names* joins two ideas in its opener, and should say plainly what a spec's name is used for; *The idea*'s windows paragraph shows no forward-looking target; an example rewrite of *A first fit*'s paragraph on the local fit; *The diagnostics are out-of-sample too*, whose *too* is four paragraphs from what it leans on; *The paragraphs below say …*, a contents of the paragraphs that follow; and a section opens with a table of its contents only when the table explains the subsections, otherwise with a short introduction |
| I5, rewrite | 2026-10-04 | write all the rules, then rewrite the README by them as a test | pending: uncommitted until the verdict | *each exception is named* reads as a Python exception; the first fit is "much better", and wants brief comments in its code; relative targets should be Polars expressions, built with `with_columns` and with forward windows |
| I6, rewrite | 2026-10-04 | rewrite the README with the latest updates | pending: uncommitted, rendered at claude.ai/artifact/QkEc6s4eAiY171WtgVzkqY | write procedures as procedures: name the tool, the method and the step; both branches of a condition; no definition by negation or aphorism; anchor every relative word |
| I7, rewrite | 2026-10-04 | start the next README, rewriting phrasing completely under the procedural rules | pending: uncommitted, rendered at claude.ai/artifact/P6o4QeGH2bBpVnwVT6yQRT | in progress: *The examples from here on read two frames* is out of context and very hard to understand; why did the reviewers miss it, and how will they catch it in future; the `po.target` paragraph should be a code example with comments; every example on the example data says so just above it, with a link |

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

**Verdict:** This section of the introduction is like a small table of
contents for the section and does not help the user much.  “The idea says
how, Terminology defines the words this README uses, and What you can rely
on lists what holds in every run. Install and A first fit then get it
running.” Instead be better to summarize the section for example, “here is
a first model fit and an introduction to the library” the first fit should
come first and it should explain the inputs and outputs of each line, what
do the tables look like. The example should also include a forward rewma
target.

Then: This phrase is not good “A spec names its model, the columns it reads
and how it treats time; its first argument is the spec's own name, which
its output column takes. “ the second section of this sentence not only
does not belong so closely joined to the first part but should be phrased
more like “the name of the spec (the first parameter) is used to name the
output column, for example …

Then: This subsection “Features and targets can be windows over the
stream.” Gives an example of a backward looking window but not a forward
looking target.

Then: I will give an example rewrite of this paragraph “Give the same spec
a clock, the timestamp column ts, and a finite half_life, and each row's
weight halves every ten minutes of ts. The fit is now local: it describes
the recent past, and it moves from row to row. Its state is weighted
toward the last few tens of minutes, so serving from the final state
predicts with the most recent fit alone. The thing to read is the path the
coefficients took, which coef_every=1 writes on every row, rather than the
state they ended on. That path has one row per input row: each stock's
exposure to each signal, as it stood before that row. It is a time series,
to plot, difference, or compare between two stocks over a day.” Give the
same spec a clock (timestamp column ts) and half_life=“10m” and each row's
weight halves every ten minutes of ts. The fit is now local: it describes
the recent past, and it moves from row to row, weighted toward the last few
tens of minutes. Because this is a local regression, remember that saving
the final state may be of limited value. But using coef_every=1 produces a
time series of the betas at every row, as they stood before learning that
step. This can be used to track the betas over time.

Then: the phrase “The diagnostics are out-of-sample too.” Is very far from
the place that too is referencing, an indication that this paragraph should
be combined with the referenced paragraph and simplified.

Then: For example the paragraph should start with something like “Every
prediction and diagnostic is out-of-sample”

Then: This phrase is not useful “The paragraphs below say what that pass
guarantees, when row order matters, and how a bank is run, saved and
checked.” And should say something like “here are a few points to remember
about fitting models with polars-online”

Then: A section should open with a table of its contents if that table
explains a lot about the subsections or adds more information. Otherwise it
should be a short introduction. Do not just list the subsections in a
sentence, they are already listed at the top.

**What the verdict raised:** sixteen ideas below, eleven rules in WRITING
and one reworded,
with a count for each new pattern in §6, and the rewritten paragraph beside
its original in §8:

| rule in WRITING | section | idea |
|---|---|---|
| The introduction leads with its example | §2 | S12 |
| Every section opens with a short introduction, or with a table of its contents that explains the subsections (R2's opener rule, reworded) | §2 | S11 |
| An opener never lists what follows in a sentence | §2 | S11, S14 |
| An example shows what goes in and what comes out | §3 | C4 |
| An example matches each part of its claim, and says which part it shows | §3 | C6 |
| A semicolon joins the halves of one idea, never two ideas | §5 | W1 |
| Say what a parameter is used for, then show it | §5 | W2 |
| An aside inside a list goes in parentheses | §5 | W3 |
| Give the value, not a description of it | §5 | W4 |
| Turn a mechanism into the advice it implies, and give the reason first | §5 | W5 |
| Lead with the setting and what it produces, and call the result by its short name | §5 | W6 |
| A *too* far from what it points at marks one topic split in two | §5 | W7 |

Five ideas belong to the README alone: C5, the forward `rewm_mean`
target in the first fit; C7, the user's paragraph in place of the old one;
S13, the two out-of-sample paragraphs merged; S14, the user's lead for
*The idea*'s points; and E10, the fact the user's paragraph inherited. A row's betas are the
fit *after* learning that row, as *Coefficients* says, where the first fit
has said *before* since `9f6a5f9`. The Introduction's opener came from S1 as
WRITING then worded it: "it names the section's main contents, and which
subsection holds each". Seventeen of I4's openers do the same, mapping
their subsections or their paragraphs, and S11 lists them.

### I5, rewrite (2026-10-04)

**Prompt:** Write all the rules and the rewrite the readme as a test and
show me the html

**Ideas applied:** every idea marked *I5* below, under the nine principles
WRITING's preamble now states, P1 to P9. The README stays uncommitted
until the verdict, as the prompt asks for a test.

**Counts before:** I4's counts after: 22,627 prose words, 1,298 sentences
averaging 17.4 words, 7 of 35+ and 2 of 45+ (each a merge), 112 tables,
and no heading without an opener. Counted for the first time: 17 openers
that list what follows, 29 prose sentences with a semicolon, and 38
back-references (17 *too* meaning *also*, 17 *also*, 3 *the same way*, 1
*as well*).

**Counts after:** 20,147 prose words, 11% fewer than I4's; 1,189 sentences
averaging 16.9 words, none of 35+; 116 tables; 69 python blocks, every one
run by the tests. No opener lists what follows (19 before, by the widened
count), no prose sentence has a semicolon (29), no clause is inverted (2),
and no sentence is about the document (13). Of the 38 back-references, 5
remain, each pointing within its own paragraph. The account against I4
loses no number or link: the measurements cut from the models now live in
the deep docs the README points to.

**How it was made:** seven writers, one per part, each held to the brief
in the scratchpad (the principles, the ideas, the account, the examples run
in the README's namespace), then one read of the whole. The first fit
builds a trading day of made-up rows with known betas, prints five tables
and is held to them by a new test, `tests/test_readme_first_fit.py`.

**Found and fixed on the way:** I4's `with_windows(..., like=edge)` call
raised, because `like=` needs the spec's features in the frame; it now runs
in a block. `fit`'s order warning exempts three diagnostics, not none.
`ftrl`'s `strict_binary` refuses any target but 0 or 1. The streaming R²
is not `po.eval`'s. `bank.groups()`'s printout had dropped two of its four
groups. `bocpd`'s robust mode finds a four-sigma shift within three rows.
`λ` now means one thing, a step's decay, wherever it stands alone. Two
names that changed meaning between examples, `ahead` and `labelled`, were
split. Left for a docstring pass: `spec.py`'s count of models that keep a
weight per target, its `session_gap` bound, `holt`'s "the one model that
takes no features", and `bocpd`'s claim about dating a change in
correlation alone.

**Verdict:** This phrase discusses an exception, it is not clear if this means a
python exception or an exception to the rule of every “model takes all of
them Most models take all of them, and each exception is named where its
parameter is introduced.”

Then: The first fit section is much better but add comments to the code
briefly describing what is happening but not high detail since the section
has good detail

Then: The examples use relative targets rather than the new polars expressions
that let you construct relative targets. The new expressions are more clear

Then: We will  need example of constructing targets using with columns and forward windows

**What the verdict raised:** the ideas W8, C8, C9 and C10 below, the error E11,
and three rules: "A word with a Python meaning keeps it" in WRITING §5,
and in §3 "A long example names its steps in brief comments, and leaves
the detail to the prose" and "An example builds a thing in Polars' own
terms when the library takes them". C8 was applied to the I5 working tree at once, since the note asked
for it there.

### I6, rewrite (2026-10-04)

**Prompt:** Rewrite the readme with the latest updates

**Ideas applied:** the ideas I5's verdict raised, each marked *I6*: W8
(*exception* and *raise* keep their Python meaning), C9 and C10 (relative
targets as Polars expressions, built both ways), with C8 and E11's doc
fix already in I5's tree. A pass over the places those ideas touch, not a
rewrite of every part.

**Counts before:** I5's counts after.

**Verdict:** When explaining how something is done, write procedural sentences, not
compressed definitions. Name the concrete tool, method, and point in the
pipeline (e.g. "made with Polars expressions and chained before the
model"), not abstractions like "a column made first." State conditions
explicitly and cover both branches: "X requires Y; otherwise, do Z with
[tool] at [step]." Don't define things by negation ("a target that needs no
window") or use aphoristic "An X that is Y is a Z" phrasing. Don't use
relative words like "first," "later," or "upstream" without saying
relative to what. Test each sentence: could a reader act on it without
having to infer a missing step or tool?

**What the verdict raised:** the principle P10 and the ideas W9 to W12
below, with four rules in WRITING §5 and two counts in §6.

**Counts after:** 20,407 prose words (260 more, for the second way to build
a target and its example); 1,205 sentences averaging 16.9 words, none of
35+; 70 python blocks, every one run by the tests. *Exception* outside an
error: 0 (7 before). *Raise* meaning *increase*: 0 (5). Openers that list
what follows, semicolons joining two ideas, inverted clauses and sentences
about the document: all still 0. The account against I5 loses no number or
link; the names it loses are `po.target`'s table of scales, which its
reference page, now linked, holds.

**What changed:** *Relative and look-ahead targets* builds a target both
ways, each a running example on `trades`: a window that looks ahead, less
the mid, as an expression in `targets`, and a column made with
`with_columns` and named as the target. `po.target` follows, as a
shorthand. The code taught two facts the section now states: an
expression that divides is read about zero by `hit_rate`, so a ratio is
written less 1 or as a log ratio (only `po.target`'s ratio is centred at
1), and a spec named like an input column is refused, since its output
column would replace it. *Per-row diagnostics* gives the same advice on
ratios. *Exception* and *raise* keep their Python sense everywhere.

### I7, rewrite (2026-10-04)

**Prompt:** Start the next readme and since the latest instructions have a deep
impact, ensure that you are willing to completely rewrite phrasing based on
the latest instructions

**Ideas applied:** P10 and W9 to W12, the procedural rules I6's verdict
raised, with every idea before them kept applied. A full pass: seven
writers, one per part, each licensed to rewrite any sentence, then one read
of the whole.

**Counts before:** I6's counts after, and by the new counts: 8 bold leads
of the form *An X … is a Z*, 1 noun defined by negation, and 17 relative
words to check for their reference.

**Verdict** (the notes so far):

> This statement is out of context, it’s very hard to understand this paragraph  what should be done to fix it? “The examples from here on read two frames”

> We want to know why the reviewers missed it and how to make them catch it in the future

> The paragraph starting with “To take one column against another of its row without “ looks like it should be a code example with comments instead

> Be sure that when a code section references the example data, there is a line of prose just above the code block stating that it uses example data with a link to that section.

**What the verdict raised:** C11 (build every frame an example reads in
code before its first use, once, in *Example data*), W13 (a position in
the document is not a reference), and two review steps that would have
caught it: V1, a test that reads the examples as a reader runs them, and
V2, a cold read. Applied to the I7 README at once: *Example data* now
builds `df`, `lf`, `trades`, `today` and `later`, the block that drops
quiet groups builds `now`, and the paragraph is gone.

**Why every pass missed it,** checked against the record of each:

| pass | why the paragraph survived it |
|---|---|
| R1, R2 | they found the gap it fills: C2, *`df`, `lf`, `out`, `spec` and `bank` come from the test namespace, unseen* |
| I4 | answered C2 with a list of `df`'s columns, put where `df` is first used, at the end of the opener of *How a bank sees a stream*. WRITING §3 then allowed an example's input "as its schema" |
| I5 to I7, the writers | each kept it. C2 read as done, and every fact in the paragraph was true. The account flags a lost name, which favours keeping a paragraph made of names. The rules they checked are about sentences: P10's test asks *could a reader act on it* of how-to sentences, and this one describes. In I7, A2 owned the section and rewrote the paragraphs around it |
| the counts | count 6, sentences about the document, looks for *this section*, *the table below* and *the paragraphs*; count 9, relative words, for *upstream*, *first* and *later*. Neither lists *from here on* |
| the README test | its namespace hands every block `df`, `trades`, `today`, `later` and `now` ready-made, so a block that reads a frame the README only describes still runs. Its docstring said a `NameError` would be "itself the finding", which held only for names missing from the namespace |
| the assembler's read | read for facts and rule breaks, part by part against the writers' reports, by a reader who knew what the paragraph was for: it had been in every version since I4 |

The common cause: every check ran from inside the project. The test
started from the project's namespace, and the writers and the assembler
from the project's knowledge. A reader starts from neither, and finds the
paragraph out of context in one read. V1 starts the test where the reader
starts. It fails on the I7 README with six names, `df`, `trades`, `lf`,
and three no one had reported, `now`, `today` and `later`. V2 starts a
reviewer there.

**V2, tried on the I7 README before the fix:** a fresh agent read the page
from the top, with no other file, no code and no hint of what to look for.
It made 32 findings, and ranked the `df` paragraph first of the three that
would most confuse a newcomer. It also found `trades`, `now`, `today` and
`later`, all of which V1 found. Its other findings fall in three groups:

| group | findings | done |
|---|---|---|
| the reported fault: an input the reader cannot build | `ticks.parquet` and `ticks/*.parquet` scanned and never written; a `ticks` described as "a hundred ticks of each series" where only eight were built; the low-memory example, which must start a fresh process, reading `spec` from another section; *Fit many models, save each*, which saved none and globbed every `*.state`, the window states included | fixed after the note: *Example data* writes the files, and each example builds what it reads. V1 cannot see a file, or a name rebuilt differently, so V2 is the check for these |
| a term used before the section that introduces it | about twenty: *How a bank sees a stream* names `sigma`, the metrics, `holt`, lags, `settled_frac`, `support_coef`, `ModelBank.fit` and `emit_drift` before their sections; the first fit's table shows `above_max_error_inflation`, and its step 3 never says what `gap_cap` does; `seqtest`'s "the clip", which the section never named (now *the `max(0, …)` in a stake*) | the clip named; the rest raised as W14, the user's call |
| a claim the page cannot place | "does not load in this build" leaves a reader unsure whether the README describes the version `pip` installs; the `sin(x)` figures of *A local fit along any feature* name no measurement | raised, the user's call |

**Counts after:** 20,375 prose words (32 fewer than I6, though a sentence
that names its tool and step is longer); 1,145 sentences averaging 17.8
words. One sentence of 35+ shows, and it is a splitter merge: the `*` in
`rolling_*_by` hides a bold lead's end, so an 11-word lead and a 25-word
sentence count as one. None of 45+. 70 python blocks, every one run; the
608 tests that read the README pass. Bold leads *An X … is a Z*: 0 (8 by
the count, and six more D found by eye that the count's *An* or *The*
start misses). Nouns defined by negation: 0 (1). Relative words: 4 left,
each with its reference (*ten minutes later*, *largest first*, *1.43.0 or
later*). Semicolons: 8, every one the two-branch form *X requires Y;
otherwise …*. *X, not Y*: 2 (8). Openers that list, inverted clauses and
sentences about the document: still 0. The account against I6 loses no
number or link. Of the four names it loses, two were widened into the
command a reader types (`polars-online~=0.13.0`, and one shell line that
sets both thread pools). `tests/test_second_opinion.py` is named in
docs/TESTING.md, which the *Models* opener now links. `po.target`'s ratio
centred at 1 is now stated once, in `po.target`'s paragraph.

**What changed:** every how-to became a procedure that names the tool, the
method and the step, such as *Compute the returns from prices with Polars'
`diff()` in the query before the bank, over each block with
`.over("block")`*, for I6's *Its rows are returns, so difference upstream*.
Conditions give both branches: *give a `warm_rows` large enough for those
rows to span more than one regime; otherwise the seeds split one regime in
two*. Bold leads open with what to do, as *To search over factor sets,
build one spec per set and pass the whole list to one
`lf.online.fit_predict` call*. They no longer open with a definition, such
as *A search over factor sets is a list of specs*. The examples in
*Relative and look-ahead targets* run as lazy queries, so *chained in the
query before the bank* is what their code does. Install says what to run in
each case, numpy and a platform with no wheel included.

**Facts the writers corrected,** each run in the README namespace or read
in the code (the reports give the file and line):

| writer | I6 said | the code does |
|---|---|---|
| A1 | the large-files query ended in `.collect()` | that holds the result in memory: it now ends in `.sink_parquet` |
| A1 | the fit over every row gave one number for the day | step 2 fits only the rows before 15:30 |
| A1 | (the user's paragraph) a clock, with no `gap_cap` | a clock requires `gap_cap`: step 3's bold lead names it, the paragraph is untouched |
| A2 | `seqtest` takes a weight of 0 or 1 | `seqtest` refuses a weight; the 0-or-1 rule is `rcov`'s, now a row of its own |
| A2 | four kinds of model "take neither" form of relative target | they refuse a target expression and `po.target`, and take a column made with `with_columns` |
| A2 | the error names the cast that fixes a text column | that cast fails on a category such as `venue`: compare a category, as `pl.col("venue") == "X"` |
| A2 | `test_weight_scale` checks each listed model | it holds every other model to the scale rule |
| B | a window state knows its input, and refuses another | only a state saved under a slice does: E13 |
| B | "whichever way it runs, its output can leave as Arrow" | only a `ModelBank` has `fit_predict_arrow` and `predict_arrow` |
| B | `trades` has a `mid` on its quotes | every row has a `mid`, trades included |
| B | an ordinary state carries an infinity | it can hold one; a finite `half_life` state's export holds `"nan"` |
| C | two decayed halves merge once `weight_sum` is decayed | `target_weights` needs the same factor: E12, fixed |
| C | a `"monotone"` key that is not an integer orders as text | a float or temporal key is refused, naming `cast`, and so is a null key |
| C | `coef_every=1` writes `coef` on every row | not on a skipped row, where it is null |
| D | `bank.solve_failures()` counts each retry | it counts each solve that needed a retry once |
| E | `ew_cov(stats=[])` writes nothing but `weight_sum` | it writes `settled_frac` and `withheld_reason` too |
| F | ADBC's trap gives both queries out of order, and nothing says so | the newer query is one batch short and in order; the older is out of order, and a spec with a `clock` refuses it |
| F | Polars reads the prefetch variables at each scan | it reads them when the query starts to run |

**The assembler's read** of the whole changed eleven places: the
embargo rule in *Relative and look-ahead targets*, stated in full in
*Windows as a model's inputs and target*, became a pointer there; the two
paragraphs on `hit_rate` and ratios became one place each, with *read
about 1* replaced by what the metric asks; *it over-covers* named its
subject; *read it otherwise than a weighted mean* became *differently
from*; *when the coefficients may lag* went back to I6's *when the fit
need not be current at every row*; `holt`'s seasonal lead put its reason
first; the `corrchange` table's *(defined below)* became the rows it means;
a code comment's *As a query's "ridge"* became *the "ridge" spec built in
As a query*; *before the bank* gained *in the query*; and two sentences
were split.

**A question for the user:** A1 cut *The idea*'s saving point as a
restatement of the first fit's step 2, and moved its link into the
paragraph on `served`. Keep *The idea* at five points, or restore it?

## The ideas

Every idea the reviews and the verdicts raised, grouped by kind. A status
names the rewrite that applies it, such as *I4* or *I5*, or is *later* with
the reason it waits.

### Principles

The nine principles of WRITING's preamble, each the generalization of
several verdicts. Every idea below serves one of them.

| id | principle | from | status |
|---|---|---|---|
| P1 | every sentence is about the library or the reader's task, never about the document | the contents-like openers, *The paragraphs below* | I5 |
| P2 | concrete first: show it working, then explain | first fit first, the ticks' schema, `half_life="10m"` | I5 |
| P3 | show what is distinctive about the library early | the forward target in the first fit, both out-of-sample claims up front | I5 |
| P4 | say what it means for the reader, with the reason | *Because this is a local regression, remember …* | I5 |
| P5 | plain spoken English, not compressed or literary phrasing | *is used to name the output column*, *clunky*, *Terminology* | I5 |
| P6 | one topic, one place, led by the point to remember | the far *too*, the semicolon join, nothing tacked on | I5 |
| P7 | fewer ideas per paragraph, not shorter sentences: I5 aims to come in shorter than I4 | the user's rewrite, 93 words for 124 | I5 |
| P8 | an example is complete: input, output and every part of its claim | the inputs and outputs of each line, the missing forward target | I5 |
| P9 | the opening is read hardest, and one instance stands for a pattern | every note in the first 250 lines, each with siblings | I5 |
| P10 | a reader can act on every how-to sentence without inferring a step or a tool | I6's verdict: *a column made first*, *a target that needs no window* | I7 |

### Structure

How the README is divided, ordered and opened.

| id | idea | from | status |
|---|---|---|---|
| S1 | every section and subsection opens with one to three sentences naming what it holds; sixteen headings had none | R2 | I4; its openers mapped their subsections, which S11 corrects |
| S2 | name the kinds of functionality first, then give each a subsection or a bold-led paragraph; split every paragraph that introduces two parameters, modes or outputs | R2 | I4 |
| S3 | pull every important detail tacked on at the end of a section or paragraph into the paragraph whose topic it is, or into the opener | R2 | I4 |
| S4 | state what every subsection obeys in the parent, before the first subsection: the window rules, the family rules, "after the last row each group learned from" | R2 | I4 |
| S5 | the most important information first: the release refusal in Saving's opener, the selection switches in a subsection of their own | R1, R2 | I4 |
| S6 | within-section order: *Output field names* before *Coefficients*; *Row order* after the forgetting subsections; windowed means before refresh time; the regime family as `seqtest`, `corrchange`, `bocpd`, `hmm`; Performance as Throughput, Parallelism, Chunk size, Memory, Tuning memory, Window operators, scikit-learn; Versioning with the action first; Databases before Pathway | R1, R2 | I4 |
| S7 | a model entry's template: when to reach for it, symbols before the rule, one-line comments, checks last | R1, R2 | I4 |
| S8 | move *What this is not* into the Introduction | R1 | later: R2 kept it in place with an opener |
| S9 | fold *Pathway* and *Databases* into *Running a bank* | R1 | later: R2 kept *Scope and integrations* with an opener |
| S10 | move most of *Testing* to TESTING.md, since WRITING §1 keeps how it is tested out of the guide | R1 | later: R2 restructured it in place; an altitude question for its own pass |
| S11 | a section opens with a table of its contents when that table explains a lot about the subsections or adds information, and otherwise with a short introduction that summarizes, as "here is a first model fit and an introduction to the library". It never lists what follows in a sentence: the contents table at the top already lists the subsections. Seventeen of I4's openers did: *Introduction*, *The idea*, *What a spec names*, *Time and decay*, *A hard window*, *Row order and the two guarantees*, *Labels that arrive late*, *Reading the fit*, *Performance*, *Throughput*, *Parallelism*, *Memory*, *Window operators*, *Scope and integrations*, *Versions, testing and development*, *Versioning and the Polars pin*, *Testing*. Nine open with a table of their subsections, each to judge by the test: *How a bank sees a stream*, *Preparing a stream*, *Windowed means*, *Running a bank*, *Reading the fit*, *Diagnostics, selection and evaluation*, *Models*, *Linear models*, and *Performance*, whose only column beside the subsection describes it | I4's verdict | I5 |
| S12 | the *Introduction* opens with the first fit, before *The idea*, *Terminology* and the rest | I4's verdict | I5 |
| S13 | *The idea*'s *The diagnostics are out-of-sample too* merges into *Every prediction is out-of-sample*, four paragraphs above, as one simpler paragraph led by the user's words, *Every prediction and diagnostic is out-of-sample*: both are read from what the models had learned before the row | I4's verdict | I5 |
| S14 | *The idea*'s lead-in to its bold-led points says what they are for, in the user's words: *here are a few points to remember about fitting models with polars-online*, in place of *The paragraphs below say what that pass guarantees, when row order matters, and how a bank is run, saved and checked* | I4's verdict | I5 |

### Terms and words

Which words the README defines, and where.

| id | idea | from | status |
|---|---|---|---|
| T1 | define *model bank* where it first appears, in *The idea* | R1, R2 | I4 |
| T2 | define a term before using it: *stamp*, *stretch*, *break*, *Gram*, *slot*, *grid*, IC, vech, ARI, TVTP | R1, R2 | I4 |
| T3 | one word, one meaning: Parallelism's unit of work is a *task*, not a *stream*; `hmm`'s regime is a *hidden state*; a list of parameter values is a *grid* only where it is introduced as one | R1, R2 | I4 |

### Wording

How a sentence is built and phrased.

| id | idea | from | status |
|---|---|---|---|
| W1 | a semicolon joins the halves of one idea, never two: *What a spec names*' opener joined what a spec describes to what its name is for; I4 has 29 prose sentences with a semicolon, each to be read | I4's verdict | I5 |
| W2 | say what a parameter is used for, plainly, then show it: "the name of the spec (the first parameter) is used to name the output column, for example …"; I4 has the inverted *which its … takes* twice, in *What a spec names* and in the null-target row of *Nulls, and three ways to hold a row back* | I4's verdict | I5 |
| W3 | an aside inside a list goes in parentheses: *a clock (timestamp column ts) and half_life="10m"*, not *a clock, the timestamp column ts, and a finite half_life*, which reads as three things | the user's rewrite | I5 |
| W4 | give the value the example uses, as code, rather than a description of it: `half_life="10m"`, not *a finite half_life* | the user's rewrite | I5 |
| W5 | turn a mechanism into the advice it implies, reason first: *Because this is a local regression, remember that saving the final state may be of limited value* | the user's rewrite | I5 |
| W6 | lead with the setting and what it produces, in a plain verb, and call the result by its short name: *using coef_every=1 produces a time series of the betas at every row* | the user's rewrite | I5 |
| W7 | a back-reference points at most at the paragraph before; farther back, the two passages merge. I4 has 38 prose sentences with one (17 *too* meaning *also*, 17 *also*, 3 *the same way*, 1 *as well*), each to be read | I4's verdict | I5 |
| W8 | a word with a Python meaning keeps it: *exception* and *raise* mean an error raised, so the everyday senses become *except*, *apart from*, *a model that does not take it*, *increase*. I5 has seven such *exception*s and five *raise*s meaning *increase* | I5's verdict | I6 |
| W9 | explain how something is done in procedural sentences that name the tool, the method and the point in the pipeline, as the user's *made with Polars expressions and chained before the model*, never *a column made first* | I6's verdict | I7 |
| W10 | state a condition with both branches: *X requires Y; otherwise, do Z with [tool] at [step]* | I6's verdict | I7 |
| W11 | no definition by negation (*a target that needs no window*) and no *An X that is Y is a Z*: I6 leads eight paragraphs with that form and defines one noun by negation | I6's verdict | I7 |
| W12 | a relative word (*first*, *later*, *earlier*, *upstream*, *downstream*) says what it is relative to: about ten in I6 do not, such as *difference upstream* and *made first* | I6's verdict | I7 |
| W13 | a position in the document is not a reference: *from here on*, *below* and *above* say where the writer stands, which a reader who jumped in does not share; name what they point at, or link to it | I7's verdict | I7, applied after the note |
| W14 | a term used before the section that introduces it links there, or says in a few words what it is: V2 found about twenty, most in *How a bank sees a stream*, which names diagnostics, models and warm-up readings before their sections | V2 on I7 | later: the user's call |

### Code and examples

What the examples show, and how.

| id | idea | from | status |
|---|---|---|---|
| C1 | a comment longer than its line is a concept: move it into prose or a table | R1, R2 | I4 |
| C2 | introduce every frame an example reads: `df`, `lf`, `out`, `spec` and `bank` come from the test namespace, unseen | R1, R2 | I4, by a description of `df`'s columns, which C11 replaces |
| C3 | no name that collides: a spec named `"w"` beside a weight column `w`, `out` as a frame and later a dict | R1, R2 | I4 |
| C4 | an example shows what goes in and what comes out: its input as a few rows or its schema, and the table each line returns, with what each line takes and gives; a test holds each printed table to the code. First the first fit; the `refresh_time` example was the first instance | I3's and I4's verdicts | I5 |
| C5 | the first fit includes a forward `po.rewm_mean` target: a window expression that looks ahead, learned under an `embargo` of at least its `window_size` | I4's verdict | I5 |
| C6 | *The idea*'s windows paragraph names a feature that looks back and a target that looks ahead, each as what it is, such as the last minute's time-weighted mean of the mid as a feature and the next minute's VWAP less the mid as the target; I4 gave both windows but never said which was the target | I4's verdict | I5 |
| C7 | *A first fit*'s paragraph on the local fit becomes the user's rewrite, in the user's words, with code formatting added and E10 corrected; its code writes `half_life="10m"` to match | the user's rewrite | I5 |
| C8 | a long example names its steps in brief one-line comments and leaves the detail to the prose: the first fit's four steps, numbered | I5's verdict | I5, applied after the note |
| C9 | a relative target is a Polars expression: a window that looks ahead less a column of the row, `(po.rewm_mean("price", ...) - pl.col("mid")).alias("fwd_move")`, with `/` for a ratio and `.log()` for a log ratio. A plain expression with no look-ahead is refused as a target, so such a target is a column made with Polars' `with_columns`. `po.target` comes second, if at all. In I5: *Relative and look-ahead targets* and the ratio in *Per-row diagnostics* | I5's verdict | I6 |
| C10 | show both ways to build a target, each in a running example: a window that looks ahead, less a column of the row, as an expression in `targets`, and a column made first with Polars' `with_columns` and named as the target | I5's verdict | I6 |
| C11 | build every frame an example reads in code before its first use: a frame several sections share is built once, in *Example data* after *Install*, and a section that runs on it links there. A prose list of a frame's columns is not a way to build it: *The examples from here on read two frames* was one, at the end of a section about parameters | I7's verdict | I7, applied after the note |
| C12 | a paragraph that walks through a call's forms becomes an example with comments: `po.target`'s three scales, each named with `name=`, the ratio's `hit_rate`, the null rule and the TOML table, in one spec on `trades`. WRITING §6 counts such paragraphs (V3) | I7's verdict | I7, applied after the note |
| C13 | the five paragraphs V3 judged code become examples with comments: the rate of a cumulative volume through `po.increment`, with an operator's output windowed in a second `with_windows` call; `refresh_time` in two runs, with an `assert` that the two grids are one run's; the Pathway operator as a class with `snapshot` and `restore`; `session_shrink` in one spec; and `chunk_rows` on a query, a frame and a frame chunked. Running them corrected one rule: a blend also needs a numeric `session_gap`, which the lead had left out | V3 | I7, applied on the user's word: 'All three' |
| C14 | the fifteen paragraphs the judging pass found become examples with comments, each run and its comments checked: the ratio target about zero (`fwd_return`, `fwd_log`); the command line's TOML, run through the built `online` to the same predictions as Python, bit for bit; a grid's `coef` blocks; the decayed-halves merge, with an `assert` against one run's Gram; `holt` grouped on the hour; the window target fed back as a column, whose predictions equal the native form's on all 2,891 scored rows; a DuckDB feed; every operator keyword in one call; the functions for a type checker; a dated state file per batch; `unnest` on a saved output; a model against `holt` with `emit_sigma` and a `seqtest`; `rcov`'s carried-forward, pre-averaged variant; reading a `seqtest` at the 5% level and as an e-value; and `corrchange`'s five fields, `since_flag` explained for the first time | V3's judging pass | I7, applied on the user's word: 'All' |
| C15 | every example on the example data says so in the line just above it, with a link: *This code uses … from [Example data](../README.md#example-data):*, naming what the block reads. 57 blocks, held by `test_an_example_on_the_example_data_says_so_just_above_it` | I7's verdict | I7, applied after the note |

### Review

How a pass checks itself, beyond the counts.

| id | idea | from | status |
|---|---|---|---|
| V1 | read the examples as a reader runs them: in order, from nothing. `test_every_name_a_readme_example_reads_was_built_by_an_earlier_one` parses each block and fails on a name no block before it builds, since the README test's namespace hands every block its frames ready-made | I7's verdict | I7: fails on the I7 README with six names, passes on the fix |
| V2 | a cold read: after the counts and the gate, a reader with none of the pass's context (no brief, no earlier version, no code) reads the rendered page from the top, and marks each paragraph whose purpose they cannot tell from what comes before it, and each example they could not run. WRITING §6, step 9 | I7's verdict | tried on the I7 README before the fix: 32 findings, the `df` paragraph ranked first; every pass from now on |
| V3 | find the paragraphs that should be code: list the prose that names three or more of the library's own names (parameters, calls, output fields, quoted values, read from the package's signatures and docs/OUTPUTS.md) that no code block under the same heading shows, each in its code form; then a reader judges each against WRITING §3's three forms, with the `po.target` before and after as the example. The detector cannot see a procedure told without the library's names, so the judging pass also reads the whole page | the user, after C12: "How can we find more paragraphs that should be code?" | built 2026-10-04, and in `scripts/doc_review.py` as count 10: ranks the `po.target` paragraph third of 25 on I7; on the fixed README, 24 listed, five judged code and converted (C13), three to look at, sixteen rules or lists. The judging pass ran on the whole page the same day: 15 findings, 7 strong (the ratio target written only in prose, the command line with no example, a grid's `coef` blocks, the decayed-halves merge in a table cell, `holt` grouped on the phase, a window target fed back as a column, the database feed) and 8 borderline, for the user's call; it also caught a pointer that said a section *builds* what it only describes, fixed |

### Errors

Facts the README got wrong, each checked against the code.

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
| E10 | *A first fit* | a row's betas are said to be the fit *before* that row; they are the fit after learning it, as *Coefficients* says and a three-row `rls` shows. In the README since `9f6a5f9`, and carried into the user's rewrite | I5 |
| E11 | `_spec.py`, `formula_target` (code, not the README) | a target expression with no operator looking ahead is refused with "add it as a column with po.stream.with_windows", which refuses the same expression; the advice should be Polars' `with_columns` | fixed in `fc51308` on the user's word: the refusal now names `with_columns` for a formula with no operator, and `with_windows` for one that only looks back |
| E12 | `gram.py`, `po.gram.merge`'s docstring (code, not the README) | to merge two halves of a decayed stream it decays only the earlier half's `weight_sum`; its `target_weights` need the same factor, `0.5 ** (dt / half_life)`, or the target moments come out 34% off (I7's writer for *Reading the fit*, measured; both scaled, the merge matches the whole stream to 1e-15) | fixed in `bc06b6c` on the user's word: the docstring gives both factors as steps, with a runnable example, and `tests/test_gram_module.py` holds every field of the merge to the whole stream |
| E13 | `po.stream.with_windows` resumed with `load_state=` (code) | after a state saved without a slice, an input whose first row repeats the last row read, at the same stamp, is accepted and that row comes out twice: `trades.head(2000)` saved, then `trades.slice(1999)` resumed, gives 3,001 rows against one run's 3,000. A state saved under a slice refuses the same input. Found by I7's writer for *Preparing a stream*, reproduced | raised for the user's decision: refuse it as the sliced state does, or document it |
| E14 | docstrings (code, not the README), found by I7's writers | ten claims the code contradicts or leaves short: `spec.py:91`, per-target weights; `spec.py:80`, `session_gap`'s bound; `holt`'s "the one model that takes no features" (`_spec.py:2478`, E3's twin); `bocpd`'s dating of a correlation break; `kalman`'s "a prediction far past the data is the intercept alone" (`_spec.py:1508`), true only where `gap_cap` spans several reversion half-lives; `OrderNotGuaranteedWarning`'s steps leave out `sort` (`_frame.py:216`); `po.target` leaves `sgd` with a logistic or Poisson loss out of the models that refuse it (`_spec.py:511`); `refresh_time`'s step back arrives as a `ComputeError`, not a `ValueError` (`stream.py:344`); `with_windows`' "Refused by name" holds only for a state saved under a slice (`stream.py:672`, E13); `ew_cov(stats=[])` writes `settled_frac` and `withheld_reason` beside `weight_sum` (`_spec.py:1967`) | later: a docstring pass, on the user's word |
| E15 | documents outside the README, found by I7's writers | `docs/PERFORMANCE.md:92` says the README calls a unit of work a "stream", where it has said "task" since I4; `docs/ARROW-SOURCES.md` §2 lacks that closing an ADBC SQLite cursor while `pl.scan_arrow_c_stream` has read only part of its stream segfaults the process (3 of 3 runs at 200,000 rows, adbc 1.12.0, polars 1.44.2; the example's 20,000 rows do not), and its "no error and no warning" holds only when no spec with a `clock` reads the older query; `tests/test_model_registry.py:192` quotes a README sentence that is gone, and nothing asserts it | later, on the user's word |
