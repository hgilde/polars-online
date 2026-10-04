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

Five rewrites, the last a test, and two reviews, oldest first:

| pass | date | what was asked | commit | verdict |
|---|---|---|---|---|
| I1, task 89 | 2026-09-23 | rewrite to the writing style: plan every section first, tables over bullets | `b73ab46` | "This is an improvement" |
| I2, task 138 | 2026-09-29 | rewrite against the current API and features | `80df0ac` | none recorded |
| I3, task 154 | 2026-10-03 | rewrite every reader-facing document after tasks 139 to 153 | `f6c8b4b` | the opening paragraph "very clunky now"; the `refresh_time` example needed its input described |
| R1, review | 2026-10-04 | suggest phrasing and organization changes, section by section | none: suggestions | given as R2's prompt |
| R2, review | 2026-10-04 | the same, under three new rules: openers, kinds first, nothing tacked on | `c7580e0`, `21bedd7`: the rules | given as I4's prompt |
| I4, rewrite | 2026-10-04 | keep a record of the ideas, then rewrite the README by them | `4bcd286` | the Introduction's opener is "like a small table of contents"; the first fit should come first, show its tables, and include a forward `rewm_mean` target; *What a spec names* joins two ideas in its opener, and should say plainly what a spec's name is used for; *The idea*'s windows paragraph shows no forward-looking target; an example rewrite of *A first fit*'s paragraph on the local fit; *The diagnostics are out-of-sample too*, whose *too* is four paragraphs from what it leans on; *The paragraphs below say …*, a contents of the paragraphs that follow; and a section opens with a table of its contents only when the table explains the subsections, otherwise with a short introduction |
| I5, rewrite | 2026-10-04 | write all the rules, then rewrite the README by them as a test | pending: uncommitted until the verdict | pending |

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

**Verdict:** pending.

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

### Code and examples

What the examples show, and how.

| id | idea | from | status |
|---|---|---|---|
| C1 | a comment longer than its line is a concept: move it into prose or a table | R1, R2 | I4 |
| C2 | introduce every frame an example reads: `df`, `lf`, `out`, `spec` and `bank` come from the test namespace, unseen | R1, R2 | I4 |
| C3 | no name that collides: a spec named `"w"` beside a weight column `w`, `out` as a frame and later a dict | R1, R2 | I4 |
| C4 | an example shows what goes in and what comes out: its input as a few rows or its schema, and the table each line returns, with what each line takes and gives; a test holds each printed table to the code. First the first fit; the `refresh_time` example was the first instance | I3's and I4's verdicts | I5 |
| C5 | the first fit includes a forward `po.rewm_mean` target: a window expression that looks ahead, learned under an `embargo` of at least its `window_size` | I4's verdict | I5 |
| C6 | *The idea*'s windows paragraph names a feature that looks back and a target that looks ahead, each as what it is, such as the last minute's time-weighted mean of the mid as a feature and the next minute's VWAP less the mid as the target; I4 gave both windows but never said which was the target | I4's verdict | I5 |
| C7 | *A first fit*'s paragraph on the local fit becomes the user's rewrite, in the user's words, with code formatting added and E10 corrected; its code writes `half_life="10m"` to match | the user's rewrite | I5 |

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
