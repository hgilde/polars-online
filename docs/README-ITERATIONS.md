# The README's iterations

This record tracks each pass over the README: the prompt that started it,
word for word, the ideas it applied, the counts before and after, and the
verdict on the result. Read it to see which prompts produced a better
document, and to choose the next ideas to apply. The rules the ideas became
are in [WRITING.md](WRITING.md), and the reports behind them in
[PHRASING.md](records/PHRASING.md).

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

Seven rewrites, the last three tests, two reviews, and I8, a rewrite begun
with its map, oldest first:

| pass | date | what was asked | commit | verdict |
|---|---|---|---|---|
| I1, task 89 | 2026-09-23 | rewrite to the writing style: plan every section first, tables over bullets | `b73ab46` | "This is an improvement" |
| I2, task 138 | 2026-09-29 | rewrite against the current API and features | `80df0ac` | none recorded |
| I3, task 154 | 2026-10-03 | rewrite every reader-facing document after tasks 139 to 153 | `f6c8b4b` | the opening paragraph "very clunky now"; the `refresh_time` example needed its input described |
| R1, review | 2026-10-04 | suggest phrasing and organization changes, section by section | none: suggestions | given as R2's prompt |
| R2, review | 2026-10-04 | the same, under three new rules: openers, kinds first, nothing tacked on | `c7580e0`, `21bedd7`: the rules | given as I4's prompt |
| I4, rewrite | 2026-10-04 | keep a record of the ideas, then rewrite the README by them | `4bcd286` | the Introduction's opener is "like a small table of contents"; the first fit should come first, show its tables, and include a forward `rewm_mean` target; *What a spec names* joins two ideas in its opener, and should say plainly what a spec's name is used for; *The idea*'s windows paragraph shows no forward-looking target; an example rewrite of *A first fit*'s paragraph on the local fit; *The diagnostics are out-of-sample too*, whose *too* is four paragraphs from what it leans on; *The paragraphs below say …*, a contents of the paragraphs that follow; and a section opens with a table of its contents only when the table explains the subsections, otherwise with a short introduction |
| I5, rewrite | 2026-10-04 | write all the rules, then rewrite the README by them as a test | `21b84a4` (task 157, 2026-10-05), inside I7, which built on it | *each exception is named* reads as a Python exception; the first fit is "much better", and wants brief comments in its code; relative targets should be Polars expressions, built with `with_columns` and with forward windows |
| I6, rewrite | 2026-10-04 | rewrite the README with the latest updates | `21b84a4` (task 157, 2026-10-05), inside I7, which built on it; rendered at claude.ai/artifact/QkEc6s4eAiY171WtgVzkqY | write procedures as procedures: name the tool, the method and the step; both branches of a condition; no definition by negation or aphorism; anchor every relative word |
| I7, rewrite | 2026-10-04 | start the next README, rewriting phrasing completely under the procedural rules | `21b84a4` (task 157, 2026-10-05), then the fixes of task 158 in `20c5e5f` and `a7a8f3c`; rendered at claude.ai/artifact/P6o4QeGH2bBpVnwVT6yQRT | in progress: *The examples from here on read two frames* is out of context and very hard to understand; why did the reviewers miss it, and how will they catch it in future; the `po.target` paragraph should be a code example with comments; every example on the example data says so just above it, with a link |
| I8, task 238 | 2026-10-10 | rewrite every reader-facing document, with recipes as the examples of capabilities and a README section for every feature page: first the map | the map merged to `main` (`3176294`), C1-C5 decided by the coordinator (the user: "Your call"); six writers merged at `9cce8f7`, then the assembler on `task238-asm` (`ccd4a3d` to `c88f55f`) | pending |

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
| a claim the page cannot place | "does not load in this build" leaves a reader unsure whether the README describes the version `pip` installs; the `sin(x)` figures of *A local fit along any feature* name no measurement | fixed 2026-10-05 (task 158), on the user's word: the opener says "a state saved by 0.13.0 or an earlier release does not load in the releases after it"; the `sin(x)` figures name their stream and `tests/test_ewridge.py`, which now holds the three of them to two places |

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

### I8, task 238 (2026-10-10)

**Prompt:** After the merge to main do a rewrite of the docs, make a plan
and reorganize sections if needed, go through every section of every doc
following the guidelines and ensuring that we prefer to have recipes as
examples of capabilities.

Then: We want to make sure that features with their own doc pages have a
section in the main readme so a casual user can find

Then, on the name of a page: Instead of data issues we need a better name

Then, on the order of tasks 238 and 239, as `docs/PLAN.md` quotes it: You
can flip the api and [docs] rewrite if that would be better

**Scope:** PLAN task 238. The README, every guide `docs/README.md` lists,
`docs/README.md` itself, CONTRIBUTING and `llms.txt`. The records stay as
written, and VALIDATION.md is generated. The API docstrings are task 239,
reviewed before this rewrite starts, while this map was drawn.

**Ideas applied:** S15 to S17, C17 to C19, V4 and V5, all from this map
(in [The ideas](#the-ideas)). It also takes on two ideas left for a pass
of their own: S10, R1's *Testing* to TESTING.md, and W14, from I7's cold
read, a term used before the section that introduces it.

**How the map was drawn:** on `899606d`, with nothing in the prose
changed. Each heading tree, with its lines and prose words per section, and
each python block's class came from two scratch scripts. A block was
classed by three things: the line above it (*This code uses … Example
data*), its body (whether it builds or prints its data), and what follows
its fence (a ```` ```text ```` block, or a table). The counts are
`scripts/doc_review.py measure` and `counts`, and the links to each guide
were counted with `grep`. A section's lines and prose words are the
scratch script's, which drops code, tables, headings and HTML as `measure`
does. The tests that run each document's blocks were
read in `tests/test_production_hardening.py` and `tests/doc_recipes.py`.

#### 1. The map

**Each example is one of four kinds.** A **recipe** is complete and
runnable: it generates its data, or reads the files *Example data* writes,
and its printed output is shown under it and held by a test. A **run**
block is run by the test suite, but nothing it prints is shown. An
**unrun** block is run by no test. **None** means prose and tables only.

The README, 5,465 lines, read by WRITING §0's reader, as the user guide:

| section and its subsections | lines | prose words | python blocks | examples | what belongs elsewhere |
|---|---:|---:|---:|---|---|
| **Introduction**: A first fit · The idea · Terminology · What you can rely on · Install · Example data | 313 | 917 | 2 | A first fit is a recipe (its five tables held by `tests/test_readme_first_fit.py`). Example data builds `df`, `lf`, `today`, `later` and `trades`, and its `df` is noise: "a model fitted on them finds no relation" | |
| **How a bank sees a stream**: What a spec names (Relative and look-ahead targets) · Time and decay (Clock types and units; Sessions, gaps and steps back) · Convergence without a decay · A hard window · A local fit along any feature · Row order and the two guarantees (How a bank detects rows out of order) · Groups · Weights · Warm-up · Labels that arrive late (The built-in delay; The delay as data) · Nulls, and three ways to hold a row back | 756 | 3,936 | 13 | run. Convergence without a decay and Nulls: none. The step-back error is a ```` ```text ```` typed by hand, which no code produces | Warm-up's table of the three gates is repeated in DIAGNOSTICS *Is it ready?* |
| **Preparing a stream**: Windowed means, looking back or ahead (Windows as columns; Windows as a model's inputs and target; Features in units of their spread; Saving and resuming a window run) · Series that tick at their own times | 465 | 2,197 | 9 | the windows' six blocks: run. `refresh_time`: a recipe (its grids held by `tests/test_refresh_time.py`) | the windows, 353 lines and 1,816 words, are a guide with no page (choice C2) |
| **Running a bank**: As a query · In a loop · Outside a live Python process · Output as Arrow | 208 | 762 | 6, a TOML, a shell line | run. The TOML runs through the built `online`, and nothing it prints is shown | RUNNER's section is *Outside a live Python process*, 37 lines, named by where it runs, not by what a reader wants |
| **Saving, loading and serving**: Save and load · Serving without learning · Reading a state without this library | 136 | 581 | 4 | run | its section links STATE-WORKFLOW once, in a parenthesis |
| **Reading the fit**: What a bank holds · Output field names · Coefficients · The running sums behind a fit · One row per finished group · Reading a correlation matrix | 440 | 1,487 | 11 | run | `check()`'s paragraph in *What a bank holds* belongs to data quality. *The running sums behind a fit*, 104 lines, repeats GRAM's table of keys, its caveats, its index of calls and its aging recipe |
| **Diagnostics, selection and evaluation** | 34 | 166 | 0 | none: a table of questions, each linked into DIAGNOSTICS | DIAGNOSTICS' section, with no recipe |
| **Models**: Linear models (9 entries) · Moments and correlation (5) · Clustering and classification (3) · Sequential tests and regimes (4) | 2,136 | 11,419 | 32, and 21 blocks of update rules | run, every entry. No block prints | 44% of the README's prose words (choice C1). `audit`'s use beside a model belongs to data quality |
| **Performance**: Throughput · Parallelism · Chunk size · Memory · Tuning memory with Polars' own settings · Window operators · Against scikit-learn | 477 | 2,103 | 4 | run | *Against scikit-learn*, 104 lines and six tables, repeats PERFORMANCE §19. *Window operators* summarizes §32 to §35 |
| **Scope and integrations**: What this is not · Databases: DuckDB and ADBC · Pathway · Training XGBoost and LightGBM: `po.gbdt` | 148 | 664 | 3 | run | `po.gbdt` has a section but no row in the contents table, and no guide (choice C3) |
| **Versions, testing and development**: Versioning and the Polars pin (This package's own versioning; What is pinned; How the pin moves; Which interfaces carry a promise) · Testing · Development · License | 327 | 1,499 | 0, a shell block | none | *How the pin moves* is for a maintainer, so RELEASE-READINESS. *Testing*'s five tables are TESTING's (S10). *Development*'s table of documents lists 7 of the 13 guides, and none of DIAGNOSTICS, DATA-QUALITY, GRAM, OUTPUTS, TESTING or RELEASE-READINESS |

Of the README's 84 python blocks, 55 read *Example data*, 5 build their own
data, none prints, and none is followed by a ```` ```text ```` block. Two
show their output as tables a test holds: the first fit and `refresh_time`.
Every other block is a run fragment.

The guides, and the pages beside them:

| document | its reader | what it is for | lines | prose words | its sections | examples |
|---|---|---|---:|---:|---|---|
| RUNNER | someone running a bank as a scheduled job, or with no Python | every key and flag of `online` | 364 | 2,031 | Running it · The configuration file · Memory, threads and chunk size · From Rust, and the Polars it needs | 6 shell and 3 TOML blocks, each run through the built `online` by `tests/test_production_hardening.py`. Nothing they print is shown: run |
| STATE-WORKFLOW | someone who saves, serves or resumes a bank | what each step guarantees, then the research that chose the design, a record kept in the same file | 593: the guide 185, the record 394 | 3,663 | The workflow in four steps · The research behind it, 2026-09-03 | 3 python blocks, unrun: two read `2025.parquet` and `rerun`, which no block builds |
| DIAGNOSTICS | someone asking whether a model works | one section per question: definition, threshold, cost, theory, recipe | 2,091 | 9,482 | How the diagnostics work · ten questions · A worked example · Data whose truth is known · Sources | 18 recipes, held by `tests/test_diagnostics_page.py` |
| DATA-QUALITY | someone `check()` sent, or about to fit | one section per finding: what it does to each model, the fix in Polars | 1,564 | 7,250 | How the checks work · eleven problems · Sources | 12 recipes, held by `tests/test_data_quality.py` |
| GRAM | someone fitting many features, or more rows than memory holds | fits, pools and selections from the Grams one pass leaves | 1,644 | 5,967 | How a Gram stands in for the rows · ten tasks · Sources | 14 recipes, held by `tests/test_gram_page.py` |
| OUTPUTS | someone reading a model's column | each model's fields | 371 | 458 | Reading this page · 22 models | generated by `scripts/outputs_doc.py`, compared by `tests/test_outputs_doc.py`. No code |
| REGIMES | someone about to trust a regime detector | what each finds on data whose truth is known | 678 | 4,339 | 0. The short version · nine experiments · 10. Running them | none: every number comes from `scripts/regime_experiments.py`, held by `tests/test_regimes_doc.py` on macOS |
| PERFORMANCE | someone who found a run slow or large | where the time and memory go; then every measurement, a record | 4,754: the guide part, *Reading this document*, 138 | 27,924 | Reading this document · five groups of numbered measurements | 2 python blocks, in §9 and §10, unrun |
| EXTENDING | a contributor adding a model | every layer a model touches, with the test that catches each omission | 666 | 3,391 | four layers · Docs · Adding an output or a parameter instead | shell lines (`git show`), unrun |
| TESTING | a contributor | what the suite proves, what it found, where it is thin; then the ledger of entries | 1,654 | 9,934 | five | none |
| RELEASE-READINESS | whoever cuts a release | the release, the Polars promise, the API policy; then two records | 1,703 | 10,442 | five, of which *CI cost while the repo was private* and *Going public, as recorded* are records | none run |
| WRITING | whoever writes a doc | the rules | 748 | 5,724 | 0 to 8 | none. No rule defines a recipe, and §2 says to record a map in `docs/PLAN.md`, where task 238 records it here |
| docs/README.md | anyone | which document to read for what | 83 | 151 | Guides · Records · At the top of the repository · Section numbers | |
| CONTRIBUTING | a contributor | the gate and the conventions | 148 | 977 | eight | one shell block, which repeats the README's *Development* |
| llms.txt | a coding agent | the map it reads before writing code against the library | 174 | 1,160 | Start here · API reference · Guides · Optional | one python block, which reads a `chunks` it never builds. `tests/test_llms_txt.py` holds its names and anchors |
| GBDT-ADAPTER, a record | someone asking how `po.gbdt` was chosen | five options measured, and §7, what task 242 built | 689 | 4,212 | 1 to 8 | one shell block |

#### 2. What belongs elsewhere

**Most moves take detail out of a README section that has grown into a
guide.** Where the guide exists, it holds the detail already, and the
README keeps a summary, a recipe and the link.

| the text | where it is | where it goes | why |
|---|---|---|---|
| a Gram's keys, its four caveats, the `po.gram` index of one-liners, the aging recipe | README, *The running sums behind a fit* (104 lines) | GRAM, which has each: its table of keys, *What a Gram cannot answer*, its table of calls, *Aging a block before it is pooled*. The README keeps a summary, a recipe and the link | a section grown into the guide it now links |
| `check()`'s paragraph, and `audit`'s *Run it beside the models* | README, *What a bank holds* and the `audit` entry | the new *Checking your data's quality* | the README has no section for DATA-QUALITY |
| the windows: nine operators, `closed`, `partial`, the formula rules, VWAP, spread units, resuming | README, *Windowed means, looking back or ahead* (353 lines) | a guide, `docs/WINDOWS.md`, with a summary, a recipe and the link in the README (choice C2) | a guide with no page |
| the 21 model entries: when to reach for each, its update rule, its parameters, its checks | README, *Models* (2,136 lines, 44% of its prose) | a guide, `docs/MODELS.md`, under the same four families and headings. The README keeps the model table, the *learns by* table and a recipe per family (choice C1) | WRITING §1 puts mechanisms in the deep docs, and a casual reader cannot scan 5,465 lines |
| the gates' table | README *Warm-up* and DIAGNOSTICS *Is it ready?* | the README, which owns the shared parameters. DIAGNOSTICS keeps its `withheld_reason` column and links the rest | one topic, one place (P6) |
| *Against scikit-learn* | README *Performance*, 104 lines, six tables | PERFORMANCE §19, with a summary and the link in the README. `llms.txt` links the anchor, so the heading stays | a record's numbers, repeated |
| *How the pin moves*, and the interfaces' detail | README *Versioning and the Polars pin* | RELEASE-READINESS, *Which Polars versions are promised*. The README keeps what to pin and what is promised | a maintainer's procedure in the user guide |
| *Testing*'s tables: references, invariants, hardening, contracts, where it runs | README *Testing* (97 lines) | TESTING, *What the suite proves*. The README keeps one table, each guarantee and how it is held | R1's S10, left for a pass of its own |
| the table of documents | README *Development* | `docs/README.md`, linked once. The README keeps the API reference, `llms.txt` and the map | the map exists twice, and the README's copy lacks six guides |
| the commands to build and test | README *Development*, CONTRIBUTING *Getting set up*, CLAUDE.md | CONTRIBUTING. The README keeps the build from a checkout | one block, three copies |
| *What the tests guarantee* | CONTRIBUTING | a link to the README's *Row order and the two guarantees* | a guarantee stated in two places |
| the dated research | STATE-WORKFLOW, after its guide | stays where it is: code and RELEASE-READINESS cite it by section and rule (`_frame.py` cites R2, and RELEASE-READINESS cites §4's R6) | a split would move cited sections for no reader's gain |
| the records inside guides | PERFORMANCE §1 onwards, TESTING's ledger, RELEASE-READINESS's last two sections | stay as written. Only each guide part is rewritten | `docs/README.md`: a record is not rewritten |
| `po.gbdt`'s guide | the README's 36 lines, the docstrings, GBDT-ADAPTER §7 | a guide, `docs/GBDT.md`, with recipes. GBDT-ADAPTER stays a record (choice C3) | a feature whose only page is a record |

#### 3. A README section for every feature page

**Each section has a heading a reader scans for, a short recipe, and the
link.** The heading names what the reader wants to do. The recipe plants
an answer and prints it, so its output shows the capability working.

| page | the README heading | where it sits | its recipe | the anchor it changes |
|---|---|---|---|---|
| DATA-QUALITY | *Checking your data's quality* (the user's) | new, under a new top-level *Checking the data and the model* | DATA-QUALITY's opening program, cut down: a sentinel, a constant and a missing feature planted, an `audit` beside a ridge, `check()`'s findings printed | none: new |
| DIAGNOSTICS | *Checking that a model works* | the same new section, after data quality, in place of *Diagnostics, selection and evaluation* | DIAGNOSTICS' opening program, cut to `emit_breaks`: a slope that flips at row 3,000, the first row after it where `break_wald` passes its threshold, and `r2` before and after | `#diagnostics-selection-and-evaluation`, linked twice, both in the README |
| GRAM | *Fitting many features from one pass* | *Reading the fit*, in place of *The running sums behind a fit* | ten weekly Grams of 50 features from one pass, pooled with `po.gram.merge`, solved, the planted coefficients and their t printed | `#the-running-sums-behind-a-fit`: 3 in the README, 3 in GRAM, 1 in DIAGNOSTICS |
| RUNNER | *Running a bank as a job, without Python* | *Running a bank*, in place of *Outside a live Python process* | the TOML and `online --config bank.toml`, with what `online` prints shown | `#outside-a-live-python-process`, 2, both in the README |
| STATE-WORKFLOW | *Saving, loading and serving* | where it is | fit the first half and save; resume on an overlapping input through `skip_learned`; print that the resumed predictions equal one run's, and that `predict` scores a row as `fit_predict` would | none |
| OUTPUTS | *Output field names* | where it is | a grid spec's fields read with `po.spec.output_index`, printed | none |
| REGIMES | *Detecting changes and regimes*, the fourth family's heading | *Models*, in place of *Sequential tests and regimes* | `po.sim.regimes` with known switches, `bocpd`'s run length and `hmm`'s filtered state printed beside the truth | `#sequential-tests-and-regimes`: 3 in the README, 1 in OUTPUTS' generator |
| PERFORMANCE | *Performance* | where it is | `fit` timed over a generated parquet at 5 and 50 features, rows a second printed: the words held, the numbers free, as GRAM's thread recipe is held | none |
| GBDT (C3) | *Training XGBoost and LightGBM on files larger than memory* | *Scope and integrations*, and a row in the contents table | a parquet with a planted target, `po.gbdt.xgboost_matrix` with `ref=` for the validation rows, 20 rounds, the best round and its error printed | the current heading's anchor, linked from nowhere |
| WINDOWS (C2) | *Making features and targets from windows* | *Preparing a stream*, in place of *Windowed means, looking back or ahead* | `trades`: a mean that looks back as a feature, a VWAP less the mid that looks ahead as a target, five rows printed | `#windowed-means-looking-back-or-ahead`, 3, all in the README |
| MODELS (C1) | *Choosing a model* | *Models*, which keeps the model table | one recipe per family opener, each from an entry's recipe in section 4 | the 21 model anchors move with the entries, linked from GRAM, DATA-QUALITY, DIAGNOSTICS, REGIMES and OUTPUTS' generator |

VALIDATION is exempt: it is generated, and says where a default came from.
The four guides for changing the library need only a link from *Versions,
testing and development*.

**V4 holds it.** A test, `test_every_guide_has_a_readme_section` in
`tests/test_doc_structure.py`, beside the test that holds `docs/README.md`
to the files, reads the rows of `docs/README.md` under *Using the library*.
For each guide but VALIDATION, it finds the README heading, at level 2 or 3,
whose own text, before its first subheading, links the guide first. That
heading must appear in the README's contents table. Its section must hold a
python block followed by a ```` ```text ```` block, which V5 runs and holds.
For each row under *Changing the library*, the README must link the guide.
A guide added to the table after this pass fails the test until its
section exists.

**V5 holds the recipes.** A README python block followed by a
```` ```text ```` block is run, and what it prints is compared with
`tests/doc_recipes.py`'s `differences`. It runs where the README test runs
each block, in a namespace built by running the README's own *Example
data* block, not the copy `_readme_namespace` keeps. RUNNER's shell
blocks gain the same form, run through the built `online` as they are now.

#### 4. Capabilities shown in prose or a fragment, and the recipe each becomes

Every README block is a run fragment but two, so each of these README
sections has a capability that no recipe shows. Count 10 lists 26 paragraphs that name three or
more of the library's names no code under the heading shows. Its top
three are the clock parameters (11 names, line 505), the temporal outputs'
units (7, line 548) and *The idea*'s windows (5, line 211).

| README section | the claim shown by prose or a fragment | the recipe |
|---|---|---|
| Relative and look-ahead targets | a target that looks ahead, and the same target as a column | both forms on one stream, their predictions printed side by side, equal |
| Time and decay, Clock types and units | every clock parameter acts on the clock; a temporal clock's outputs are in seconds | a temporal and a numeric clock over the same rows, `weight_sum` and `holt`'s trend printed in each unit |
| Sessions, gaps and steps back | `gap_cap` caps a step, `session_gap` acts at a boundary | a weekend gap with and without `gap_cap`, `weight_sum` after it printed |
| Convergence without a decay | the models that solve or accumulate give the batch answer in any order | rows shuffled, `half_life=float("inf")`, the coefficients printed beside `np.linalg.lstsq`'s |
| A hard window | `window_size` forgets a break that a half-life still holds | a slope that breaks, the coefficient after it under each |
| A local fit along any feature | the `sin(x)` errors, 0.08 against 0.39, and 0.29 at bandwidth 1.0, quoted from `tests/test_ewridge.py` | the same stream, the three errors printed |
| Row order, and how a bank detects rows out of order | `OrderNotGuaranteedWarning`, and the step-back error | a join without `maintain_order` and its warning; a step back and its message, printed where the hand-typed ```` ```text ```` stands (C19) |
| Groups, Weights | groups are independent; a row of weight 0 moves the clock and teaches nothing | a weight-0 row, `weight_sum` before and after, printed |
| Warm-up | the three gates, and `withheld_reason` | the rows each gate withholds, counted and printed |
| Labels that arrive late | each row is scored where it sits and learned from later | `embargo="5m"`, `scored_clock` against `learned_clock` printed |
| Nulls, and three ways to hold a row back | a table of what each way does to the score, the clock, the fit and `weight_sum` | the three ways on one stream, the table printed by the code |
| Windows (to WINDOWS.md under C2) | the operators' equivalence to Polars' `ewm_mean_by`; `closed`; VWAP; spread units; resuming | each a recipe: the operator against `ewm_mean_by`, printed; a resumed run equal to one run |
| As a query, In a loop | any chunking gives the same numbers, the library's central guarantee, shown nowhere by code | one chunk and a thousand, the largest difference printed: 0.0 |
| Output as Arrow | any reader of `__arrow_c_array__` takes it | pyarrow's table of a struct, its schema printed |
| Reading a state without this library | `"nan"` and `"inf"` as strings in the JSON | `to_json` of an infinite half-life's state, the strings printed |
| What a bank holds | `repr`, `groups()` and the rest, whose output is typed into comments | the same calls with their output printed |
| Coefficients | `coef_every=0` writes the fit's path | the path at five rows, printed |
| One row per finished group | a closed row equals `gram()` just before the group closed | the comparison printed |
| Reading a correlation matrix | `po.corr`'s repairs and shrinkage | a matrix that is not positive definite, repaired, its smallest eigenvalue before and after |
| Parallelism, Chunk size | 7.3 times the rows a second at 14 threads | rows a second at 1 and at all threads, printed, the numbers free |
| Databases, Pathway | a DuckDB feed; a checkpoint and restore | the rows fed, and the restored run equal to one run, printed |

The models, each entry's recipe (in MODELS.md under C1):

| model | the recipe |
|---|---|
| `ewridge` | a drifting beta, a grid of two ridges and two half-lives, each point's coefficient beside the truth |
| `rls` | the same stream, `rls` against `ewridge(ridge_scale="sum")` solved on every row, their largest difference |
| `lasso` | 20 features of which 3 matter, the penalty chosen and the features left nonzero |
| `kalman` | a coefficient that jumps, `kalman`'s and `ewridge`'s 50 rows after the jump |
| `huber`, `quantile` | 2% wild targets, the slope from `ewridge`, `huber` and the median |
| `sgd` | a count target under `loss="poisson"`, the fitted coefficients beside the truth |
| `pa` | the coefficients with and without `coef_max` binding |
| `ftrl` | a 0/1 target with 50 features of which 5 matter, the coefficients the L1 penalty zeroes and the hit rate |
| `holt` | a trending series with no features, `holt`'s `sigma` beside a regression's, and a `seqtest` between them |
| `ew_cov` | a correlation that moves from 0.2 to 0.8, `corr` before and after |
| `marginal` | 1,000 features of which 5 matter, the top five by t |
| `deco` | ten series whose equicorrelation moves, `rho` before and after |
| `rcov` | noisy ticks, the block estimate beside the noisy realised covariance and the truth |
| `audit` | (the data quality recipe) a frozen feed, `bank.audit()`'s row for it |
| `kmeans` | three clusters and a fourth born later, the label counts before and after |
| `micro` | two rings, the clusters found and the share flagged as noise |
| `ew_class` | three Gaussian classes, the share of rows classed right out of sample |
| `seqtest` | a column with a small positive mean, the first row its evidence passes the 5% level |
| `corrchange` | a correlation break at a known row, the row it is flagged |
| `bocpd` | a variance break, the run length's mode 20 rows after it |
| `hmm` | `po.sim.regimes` with two states, the share of rows in the true state |

The guides:

| document | shown by prose or a fragment | the recipe |
|---|---|---|
| RUNNER | every run: nothing `online` prints is shown | each shell block followed by what it prints (V5) |
| STATE-WORKFLOW | fit and keep the state; resume through `skip_learned` (unrun) | each step generating its data: the state saved, served and resumed, the resumed run equal to one run, printed |
| STATE-WORKFLOW | a window run's resume rules, as a table of tests | one resumed window run, its output equal to one run's |
| OUTPUTS | `po.spec.output_fields(spec)` answers at runtime | in *Reading this page*: one spec's fields printed, held by the generator's test |
| PERFORMANCE, its guide part | the headline: what moves each number | `ONLINE_TIMING=1` on a generated run, the section lines printed, the numbers free |
| REGIMES | none to add: it reports measurements, each held to its script | |
| llms.txt | its python block reads an unbuilt `chunks` | `chunks = lf.collect_batches()`, so the block runs |
| EXTENDING, TESTING, RELEASE-READINESS, WRITING, CONTRIBUTING | no capability: they are about changing the library | WRITING gains the recipe's definition (C17) and V4 and V5 in §6's step 8 |

#### 5. The reorganization

**The README after**, its top-level sections and what moves (`←` marks
text that comes from elsewhere):

| section | its subsections |
|---|---|
| Introduction | as now. Example data gains a planted truth (C18) |
| How a bank sees a stream | as now, each with a recipe |
| Preparing a stream | *Making features and targets from windows* (← the windows' 353 lines, to WINDOWS.md under C2) · *Series that tick at their own times* |
| Running a bank | As a query · In a loop · *Running a bank as a job, without Python* (← *Outside a live Python process*) · Output as Arrow |
| Saving, loading and serving | as now, a recipe in the opener |
| Reading the fit | What a bank holds (`check()`'s paragraph leaves) · Output field names · Coefficients · *Fitting many features from one pass* (← *The running sums behind a fit*, its detail to GRAM) · One row per finished group · Reading a correlation matrix |
| Checking the data and the model | *Checking your data's quality* (← `check()`'s paragraph and `audit`'s use) · *Checking that a model works* (← *Diagnostics, selection and evaluation*) |
| Models | the model table, *learns by*, accumulators. *Choosing a model*, a recipe per family. The 21 entries here, or in MODELS.md under C1. *Detecting changes and regimes* (← *Sequential tests and regimes*) |
| Performance | Throughput · Parallelism · Chunk size · Memory · Tuning memory · Window operators (a summary) · Against scikit-learn (a summary, ← §19) |
| Scope and integrations | What this is not · Databases · Pathway · *Training XGBoost and LightGBM on files larger than memory* (a row in the contents) |
| Versions, testing and development | Versioning (what to pin, what is promised) · Testing (one table) · Development (the map's link) · License |

New pages: `docs/WINDOWS.md` (C2), `docs/MODELS.md` (C1) and `docs/GBDT.md`
(C3), each with recipes held by a test of its own, as GRAM's are by
`tests/test_gram_page.py`.

**The order, each step gated before the next:**

| step | who | what | checked by |
|---|---|---|---|
| 0 | the user, then the coordinator | decide C1 to C5; merge task 239, so the guides are rewritten on checked docstrings | |
| 1 | one worker, no prose | V5 in `tests/test_production_hardening.py` and `tests/doc_recipes.py` (README blocks with output, shell blocks); C18's Example data, since every README recipe prints from it | the gate |
| 2 | one worker, moves only | the new pages made of moved text, verbatim; every inbound link and anchor; the tests that find model sections (`test_api_links.py`, `test_model_registry.py`, EXTENDING step 15, `scripts/outputs_doc.py`); `docs/README.md` and `llms.txt` rows | `doc_review.py account`, the old README against the new and the pages it fed: nothing lost; `doc_structure.py`; the gate |
| 3 | six writers, in parallel | every section to WRITING, each capability's example made a recipe (section 4) | each writer's blocks run as drafted; the account per file |
| 4 | one assembler | the contents table, Terminology, cross-section links, V4, the counts after, the render, and V2's cold read of the README and each new page | the gate, then the user's verdict |

**The writers.** No two edit one file. The README divides by top-level
section, and its first 25 lines (title, contents table) belong to the
assembler:

| writer | README sections (lines now) | other files |
|---|---|---|
| A | Introduction, How a bank sees a stream (26 to 1,094). Example data is frozen after step 1 | |
| B | Preparing a stream, Running a bank, Saving, loading and serving (1,095 to 1,903) | WINDOWS.md, RUNNER, STATE-WORKFLOW's guide part |
| C | Reading the fit, Checking the data and the model (1,904 to 2,377) | GRAM, DIAGNOSTICS, DATA-QUALITY (sentences and facts: DIAGNOSTICS has 45 sentences of 35+ words), `scripts/outputs_doc.py` for OUTPUTS |
| D1 | Models: its opener, the tables and Linear models (2,378 to 3,298) | MODELS.md's linear entries under C1 |
| D2 | Models: the other three families (3,299 to 4,513) | MODELS.md's other entries under C1, REGIMES |
| E | Performance, Scope and integrations, Versions, testing and development (4,514 to 5,465) | GBDT.md, PERFORMANCE's guide part, TESTING's and RELEASE-READINESS's guide parts, CONTRIBUTING |
| assembler | lines 1 to 25 | EXTENDING, WRITING, `docs/README.md`, `llms.txt` |

Under C1, D1 and D2 own disjoint halves of MODELS.md, and D1 owns its
opener. **Each writer edits only its own line range, in its own
worktree,** so the README's merges are disjoint hunks. A writer that
renames a heading lists its anchor, and the assembler fixes every link to
it after the merge. No writer reflows a line outside its range.

#### 6. Counts before

`scripts/doc_review.py measure`, on `899606d`:

| document | prose words | sentences | mean | 35+ | 45+ | cost words | *X, not Y* | tables | python blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| README | 25,802 | 1,443 | 17.9 | 19 | 3 | 10 | 4 | 121 | 84 |
| RUNNER | 2,031 | 108 | 18.8 | 5 | 2 | 0 | 0 | 4 | 0 |
| STATE-WORKFLOW | 3,663 | 216 | 17.0 | 4 | 1 | 0 | 0 | 14 | 3 |
| DIAGNOSTICS | 9,482 | 455 | 20.8 | 45 | 15 | 9 | 0 | 23 | 18 |
| DATA-QUALITY | 7,250 | 395 | 18.4 | 8 | 4 | 7 | 2 | 15 | 12 |
| GRAM | 5,967 | 335 | 17.8 | 0 | 0 | 0 | 0 | 9 | 14 |
| OUTPUTS | 458 | 28 | 16.4 | 0 | 0 | 0 | 0 | 28 | 0 |
| REGIMES | 4,339 | 238 | 18.2 | 6 | 1 | 2 | 1 | 21 | 0 |
| PERFORMANCE, whole | 27,924 | 1,574 | 17.7 | 39 | 5 | 105 | 45 | 150 | 2 |
| EXTENDING | 3,391 | 205 | 16.5 | 8 | 0 | 0 | 6 | 15 | 0 |
| TESTING | 9,934 | 649 | 15.3 | 8 | 5 | 5 | 0 | 47 | 0 |
| RELEASE-READINESS | 10,442 | 651 | 16.0 | 9 | 3 | 7 | 17 | 43 | 0 |
| WRITING | 5,724 | 324 | 17.7 | 13 | 1 | 12 | 11 | 15 | 0 |
| docs/README.md | 151 | 11 | 13.7 | 0 | 0 | 0 | 0 | 3 | 0 |
| CONTRIBUTING | 977 | 76 | 12.9 | 0 | 0 | 0 | 3 | 0 | 0 |
| llms.txt | 1,160 | 71 | 16.3 | 7 | 2 | 1 | 3 | 1 | 1 |

`scripts/doc_review.py counts`, the hits of each numbered count (WRITING
§6), each still to be read:

| document | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| README | 0 | 0 | 14 | 17 | 0 | 0 | 2 | 0 | 15 | 26 | 5 |
| RUNNER | 2 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 4 | 0 |
| STATE-WORKFLOW | 6 | 0 | 6 | 11 | 0 | 1 | 0 | 0 | 6 | 5 | 3 |
| DIAGNOSTICS | 0 | 0 | 8 | 7 | 0 | 0 | 3 | 0 | 3 | 3 | 0 |
| DATA-QUALITY | 0 | 0 | 1 | 6 | 0 | 0 | 0 | 0 | 1 | 2 | 0 |
| GRAM | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| OUTPUTS | 20 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| REGIMES | 1 | 0 | 9 | 2 | 0 | 2 | 1 | 0 | 1 | 1 | 1 |
| PERFORMANCE, whole | 10 | 0 | 66 | 25 | 0 | 14 | 9 | 0 | 21 | 10 | 6 |
| EXTENDING | 6 | 0 | 10 | 7 | 0 | 2 | 1 | 0 | 1 | 2 | 1 |
| TESTING | 5 | 1 | 16 | 22 | 0 | 2 | 1 | 0 | 8 | 5 | 7 |
| RELEASE-READINESS | 1 | 2 | 18 | 17 | 0 | 6 | 5 | 0 | 17 | 1 | 21 |
| WRITING | 0 | 0 | 9 | 7 | 3 | 7 | 2 | 1 | 19 | 1 | 8 |
| docs/README.md | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| CONTRIBUTING | 3 | 0 | 7 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| llms.txt | 4 | 0 | 5 | 3 | 0 | 0 | 1 | 0 | 0 | 2 | 1 |

OUTPUTS' 20 headings with no opener are its generated model sections, each
opening on its table. WRITING's hits in counts 5 to 11 are mostly the
patterns it quotes as examples.

The links from the README to each guide, by `grep` (PLAN 238 measured GRAM
1, DATA-QUALITY and STATE-WORKFLOW 2):

| guide | links | guide | links |
|---|---:|---|---:|
| OUTPUTS | 24 | RUNNER | 4 |
| PERFORMANCE | 22 | RELEASE-READINESS | 3 |
| DIAGNOSTICS | 17 | DATA-QUALITY | 2 |
| REGIMES | 11 | STATE-WORKFLOW | 2 |
| VALIDATION | 5 | TESTING | 2 |
| GRAM | 1 | EXTENDING, WRITING, GBDT-ADAPTER | 1 each |

#### The choices for the user

**Five choices are the user's, each with a recommendation.** The plan
assumes the recommended option of each, and its step 0 waits on them.

| | the choice | the options | recommended | why |
|---|---|---|---|---|
| C1 | where the 21 model entries live | (a) in the README, each made a recipe, which adds about 600 lines to its 2,136; (b) a guide, `docs/MODELS.md`, with the same families and headings, the README keeping the model table and a recipe per family; (c) a page per family | (b) | the entries are 44% of the README's prose. Their update rules are mechanisms, which WRITING §1 puts in the deep docs, and a casual reader scanning for a feature passes through 2,136 lines of them. One page keeps every model one link from the table |
| C2 | the windows | (a) in the README; (b) a guide, `docs/WINDOWS.md`, the README keeping a summary, a recipe and the link | (b) | 353 lines of semantics, a feature as large as GRAM's, with no page |
| C3 | `po.gbdt`'s guide | (a) a guide, `docs/GBDT.md`, with recipes; (b) a guide part opening GBDT-ADAPTER, as STATE-WORKFLOW opens its record; (c) the README's section and the docstrings alone | (a) | GBDT-ADAPTER is a record and stays as written. A guide inside it would move with the code, which a record never does, and STATE-WORKFLOW shows the cost of the mix: a guide whose sections cite a record's numbered parts |
| C4 | what the README's recipes read | (a) *Example data* with a planted truth, so a printed output shows the capability (C18); (b) each recipe makes its own stream, as the guides' do; (c) *Example data* as it is, noise, with output shown only where its shape matters | (a) | one frame built once keeps C11 and C15, the user's rules of 2026-10-05. Today's `df` is noise, so an output printed from it shows nothing working. The capability sections that need a shape of their own, such as a break, keep their own stream |
| C5 | the headings proposed in section 3 | *Checking that a model works*, *Fitting many features from one pass*, *Running a bank as a job, without Python*, *Detecting changes and regimes*, *Making features and targets from windows*, *Choosing a model*, *Training XGBoost and LightGBM on files larger than memory* | as proposed | each names what a reader wants. The user chose *Checking your data's quality* |

Left open from I7: A1 cut *The idea*'s point on saving, as a restatement
of the first fit's step 2. Whether to restore it is still the user's call.

#### 7. After the pass

**The rewrite:** six writers on `9cce8f7` (A to E, with D split in two),
each in its own README range and files, then one assembler on
`task238-asm` (`ccd4a3d` to `c88f55f`): the contents table and anchors; V4
(`test_every_guide_has_a_readme_section`); every guide held to the no-plan
rule (`test_every_guide_is_held`); WRITING's rules for a recipe, a README
section per feature page and citations; V5's integers held exactly; the
writers' pages made consistent; and a cold read by five readers with none
of the pass's context, then a render of the README and 18 guides through
GitHub's markdown API (`mode=markdown`): every table rendered, every
in-page link on an id GitHub wrote.

`scripts/doc_review.py measure`, on `c88f55f`, beside the counts before
(section 6) where the document existed:

| document | prose words | sentences | mean | 35+ | 45+ | cost words | *X, not Y* | tables | python blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| README | 15,614 (25,802) | 877 (1,443) | 17.8 (17.9) | 4 (19) | 2 (3) | 3 (10) | 1 (4) | 76 (121) | 67 (84) |
| MODELS, new | 13,268 | 736 | 18.0 | 3 | 0 | 1 | 3 | 34 | 52 |
| WINDOWS, new | 2,415 | 134 | 18.0 | 1 | 0 | 0 | 0 | 7 | 11 |
| GBDT, new | 1,090 | 55 | 19.8 | 0 | 0 | 0 | 0 | 2 | 5 |
| RUNNER | 2,248 (2,031) | 122 (108) | 18.4 (18.8) | 3 (5) | 0 (2) | 0 (0) | 0 (0) | 4 (4) | 0 (0) |
| STATE-WORKFLOW | 4,010 (3,663) | 238 (216) | 16.8 (17.0) | 3 (4) | 1 (1) | 0 (0) | 0 (0) | 14 (14) | 5 (3) |
| DIAGNOSTICS | 9,306 (9,482) | 504 (455) | 18.5 (20.8) | 1 (45) | 0 (15) | 6 (9) | 0 (0) | 31 (23) | 18 (18) |
| DATA-QUALITY | 7,246 (7,250) | 404 (395) | 17.9 (18.4) | 0 (8) | 0 (4) | 6 (7) | 2 (2) | 18 (15) | 12 (12) |
| GRAM | 6,177 (5,967) | 350 (335) | 17.6 (17.8) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 9 (9) | 14 (14) |
| OUTPUTS | 693 (458) | 53 (28) | 13.1 (16.4) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 28 (28) | 1 (0) |
| REGIMES | 4,566 (4,339) | 261 (238) | 17.5 (18.2) | 2 (6) | 0 (1) | 0 (2) | 2 (1) | 22 (21) | 0 (0) |
| PERFORMANCE, whole | 28,161 (27,924) | 1,585 (1,574) | 17.8 (17.7) | 39 (39) | 5 (5) | 110 (105) | 45 (45) | 150 (150) | 3 (2) |
| EXTENDING | 3,400 (3,391) | 205 (205) | 16.6 (16.5) | 9 (8) | 1 (0) | 0 (0) | 6 (6) | 15 (15) | 0 (0) |
| TESTING | 10,215 (9,934) | 677 (649) | 15.1 (15.3) | 6 (8) | 1 (5) | 4 (5) | 0 (0) | 51 (47) | 0 (0) |
| RELEASE-READINESS | 10,904 (10,442) | 685 (651) | 15.9 (16.0) | 3 (9) | 0 (3) | 7 (7) | 16 (17) | 42 (43) | 0 (0) |
| WRITING | 6,008 (5,724) | 346 (324) | 17.4 (17.7) | 13 (13) | 0 (1) | 13 (12) | 12 (11) | 15 (15) | 0 (0) |
| docs/README.md | 157 (151) | 11 (11) | 14.3 (13.7) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 3 (3) | 0 (0) |
| CONTRIBUTING | 1,096 (977) | 81 (76) | 13.5 (12.9) | 0 (0) | 0 (0) | 0 (0) | 2 (3) | 0 (0) | 0 (0) |
| llms.txt | 1,222 (1,160) | 74 (71) | 16.5 (16.3) | 7 (7) | 2 (2) | 1 (1) | 3 (3) | 1 (1) | 1 (1) |

The README lost 10,188 prose words: the model entries, the windows and
`po.gbdt`'s section went to MODELS, WINDOWS and GBDT (16,773 words there),
and the detail of the running sums, *Against scikit-learn*, the pin's
procedure and *Testing*'s tables went to the guides that held them. Its
python blocks fell from 84 to 67, and the recipes among them, a block
followed by what it prints, rose from 2 to 57. PERFORMANCE's whole-file
counts barely move: only its guide part was rewritten, and its records
stay as written.

`scripts/doc_review.py counts`, the hits of each numbered count after the
pass (before, section 6, in brackets), each still to be read:

| document | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| README | 0 (0) | 0 (0) | 3 (14) | 9 (17) | 0 (0) | 1 (0) | 0 (2) | 0 (0) | 11 (15) | 11 (26) | 1 (5) |
| MODELS, new | 0 | 0 | 3 | 11 | 0 | 0 | 0 | 0 | 3 | 3 | 0 |
| WINDOWS, new | 0 | 0 | 2 | 3 | 0 | 0 | 1 | 0 | 0 | 2 | 0 |
| GBDT, new | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| RUNNER | 0 (2) | 0 (0) | 1 (1) | 2 (1) | 0 (0) | 0 (1) | 0 (1) | 0 (0) | 0 (1) | 4 (4) | 0 (0) |
| STATE-WORKFLOW | 5 (6) | 0 (0) | 5 (6) | 11 (11) | 0 (0) | 0 (1) | 0 (0) | 0 (0) | 4 (6) | 5 (5) | 3 (3) |
| DIAGNOSTICS | 0 (0) | 0 (0) | 0 (8) | 6 (7) | 0 (0) | 0 (0) | 3 (3) | 0 (0) | 3 (3) | 2 (3) | 0 (0) |
| DATA-QUALITY | 0 (0) | 0 (0) | 1 (1) | 6 (6) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 1 (1) | 2 (2) | 0 (0) |
| GRAM | 0 (0) | 0 (0) | 0 (0) | 9 (8) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 3 (2) | 0 (0) |
| OUTPUTS | 0 (20) | 0 (0) | 0 (0) | 2 (2) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| REGIMES | 0 (1) | 0 (0) | 4 (9) | 5 (2) | 0 (0) | 0 (2) | 0 (1) | 0 (0) | 0 (1) | 1 (1) | 2 (1) |
| PERFORMANCE, whole | 10 (10) | 0 (0) | 67 (66) | 25 (25) | 0 (0) | 17 (14) | 9 (9) | 0 (0) | 25 (21) | 10 (10) | 6 (6) |
| EXTENDING | 6 (6) | 0 (0) | 10 (10) | 7 (7) | 0 (0) | 2 (2) | 1 (1) | 0 (0) | 1 (1) | 2 (2) | 1 (1) |
| TESTING | 2 (5) | 0 (1) | 15 (16) | 23 (22) | 0 (0) | 1 (2) | 1 (1) | 0 (0) | 5 (8) | 5 (5) | 3 (7) |
| RELEASE-READINESS | 0 (1) | 1 (2) | 9 (18) | 20 (17) | 0 (0) | 3 (6) | 4 (5) | 0 (0) | 15 (17) | 1 (1) | 20 (21) |
| WRITING | 0 (0) | 0 (0) | 8 (9) | 6 (7) | 3 (3) | 5 (7) | 2 (2) | 1 (1) | 19 (19) | 1 (1) | 9 (8) |
| docs/README.md | 2 (2) | 0 (0) | 0 (0) | 1 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 1 (1) |
| CONTRIBUTING | 0 (3) | 0 (0) | 4 (7) | 2 (2) | 0 (0) | 0 (0) | 1 (1) | 0 (0) | 0 (0) | 0 (0) | 0 (1) |
| llms.txt | 4 (4) | 0 (0) | 5 (5) | 3 (3) | 0 (0) | 0 (0) | 1 (1) | 0 (0) | 0 (0) | 2 (2) | 0 (1) |

The links from the README to each guide, by `grep` after the pass (before
in brackets): MODELS 45, DIAGNOSTICS 19 (17), PERFORMANCE 16 (22), GRAM 11
(1), WINDOWS 10, RELEASE-READINESS 5 (3), DATA-QUALITY 3 (2), RUNNER 3
(4), TESTING 3 (2), OUTPUTS 2 (24), STATE-WORKFLOW 2 (2), GBDT, REGIMES,
EXTENDING and WRITING 1 each (REGIMES 11), VALIDATION 0 (5). OUTPUTS and
REGIMES fell because the model entries that linked them moved to MODELS.
Every guide to using the library now has a README section that links it
first, sits in the contents table and shows the capability with a recipe,
which V4 holds.

**The assembler's verdict:** the README is half its length in prose and a
casual reader reaches every feature page from a section named for the
task. Its recipes, 57 where there were 2, print what each capability does
on data with a planted answer, and V5 holds every printed line, integers
exactly. The cold read found what the writers' own passes could not: a
fact stated in two writers' words with two numbers (bocpd's `p_change`,
the test counts, the `--predict` flags), terms used before the section
that introduces them, and counts that had drifted in the maintainers'
guides. Those with one answer are fixed; the ones that need a measurement
or a decision are left for the coordinator. The user's verdict is pending.

**Verdict:** pending.

## The ideas

Every idea the reviews and the verdicts raised, grouped by kind. A status
names the rewrite that applies it, such as *I4* or *I5*, or is *later* with
the reason it waits.

**Decided 2026-10-10 by the coordinator** (the user, on proceeding with the
plan without stopping for approval: "Your call"):

- **C1:** the 21 model entries move to a new guide, `docs/MODELS.md`; the
  README keeps the model table and one recipe per family.
- **C2:** the windows move to a new guide, `docs/WINDOWS.md`; the README
  keeps a summary, a recipe and the link.
- **C3:** `po.gbdt` gets a guide of its own, `docs/GBDT.md`, with recipes;
  `docs/GBDT-ADAPTER.md` stays the record of the research.
- **C4:** *Example data* plants a truth (C18), so a recipe's printed output
  shows something working; frames are still built once, each block naming
  the frames it reads (C11, C15).
- **C5:** the headings as proposed, with the user's *Checking your data's
  quality*.
- I7's open point on saving in *The idea*: restored if it states a
  guarantee the README no longer shows elsewhere, by writer A.
- WRITING §2's rule that maps go in PLAN.md is aligned with practice: a
  README pass's map goes in this file, beside its prompt (the assembler
  edits WRITING).
