# How the documentation is written

The rules the reader-facing docs are held to: the README first, then the
docstrings and the guides under `docs/`. Each rule is drawn from a report
in [PHRASING.md](PHRASING.md). Most reports name a problem, and one
confirms an approach that worked. Each rule names the entry it came from,
so it can be checked against the case. Where two rules pull against each
other, the lower-numbered one wins.

Every rule serves one test: **can this sentence be understood by a reader
who has not yet been told what it is about?** The writer already has the
frame, so the failure is invisible from the inside. It shows up in four
forms: a term used before it is named, a cost that does not say what is
spent, a mechanism alluded to rather than stated, and a document that
assumes the reader has the author's map of it.

Nine principles sit above the numbered rules. They generalize the user's
verdicts on the README's rewrites (README-ITERATIONS.md, I3 and I4), and
were written down at the user's word on 2026-10-04: "Write all the rules".
Each numbered rule is one way to keep one of them.

| | principle | in practice |
|---|---|---|
| P1 | **Every sentence is about the library or the reader's task, never about the document** | no sentence lists what follows; a lead-in says what the points are for, as *here are a few points to remember about fitting models with polars-online*; a table of a section's contents opens it only when the table explains the subsections (§2) |
| P2 | **Concrete first: show it working, then explain** | the introduction leads with its example; an example shows its input and its output; give the value, not a description of it (§2, §3, §5) |
| P3 | **Show what is distinctive about the library early** | the first example carries what sets the library apart: a target that looks ahead, predictions made before learning, a local fit (§2, §3) |
| P4 | **Say what it means for the reader, with the reason** | turn a mechanism into the advice it implies; why, then what (§5) |
| P5 | **Plain spoken English, not compressed or literary phrasing** | the sentence a colleague would say aloud: say what a parameter is used for, then show it; an aside in a list goes in parentheses; no inverted clause, placeholder subject or *X rather than Y*; headings a reader scans for, such as *Terminology* (§5) |
| P6 | **One topic, one place, led by the point to remember** | a far *too* merges two paragraphs; a semicolon never joins two ideas; nothing important is tacked on at the end (§2, §5) |
| P7 | **Fewer ideas per paragraph, not shorter sentences** | cut a list of uses to the one that matters; secondary detail goes to the document that owns it (§1, §5) |
| P8 | **An example is complete** | its input, built by code the reader has seen before it; its output; and every part of the claim it illustrates, each named as what it is (§3) |
| P9 | **The opening is read hardest, and one instance stands for a pattern** | the top of the README gets the most care; a reported fault is counted and fixed everywhere it occurs (§6) |
| P10 | **A reader can act on every how-to sentence without inferring a step or a tool** | a procedure in procedural sentences, naming the tool, the method and the point in the pipeline; a condition with both branches; no definition by negation or "An X that is Y is a Z"; a relative word with its reference (§5) |

| section | what it asks of a draft |
|---|---|
| [0. Write for one stated reader](#0-write-for-one-stated-reader) | a named reader, and every term defined for them |
| [1. Every document has an altitude](#1-every-document-has-an-altitude-and-stays-at-it) | consequences up top, mechanisms in the deep docs |
| [2. Plan the map, then give each section one job](#2-plan-the-map-then-give-each-section-one-job) | the structure, before any sentence: an opener, the kinds, then the detail |
| [3. Code with comments](#3-code-with-comments-not-prose-that-narrates-code) | description shown as runnable code |
| [4. Facts](#4-facts-whose-how-measured-and-where-they-live) | every claim true, sourced and checked |
| [5. Sentences](#5-sentences) | one idea each, near 20 words |
| [6. The pass itself](#6-the-pass-itself) | a sequence of steps, each with a check |
| [7. Tables, not bullet lists](#7-tables-not-bullet-lists) | the shapes a comparison takes |
| [8. Rewrites, as examples](#8-rewrites-as-examples) | what these rules produced, and a paragraph the user rewrote |

## 0. Write for one stated reader

The reader of the README **knows statistics, a little Polars, and what it
means for rows to be in time order, and nothing about the internals of
Polars or of this project** (PHRASING: "README draft, round 1"). Every
sentence is written to that reader, and checked by reading it as them.

What that reader does not have must be defined before it is used, or
replaced with what it does:

| this library's own words | Polars-internal words | assumed statistics, safe to use |
|---|---|---|
| bank, model bank, spec, state, stream, chunk, clock, decay, half-life, session, group, warm-up, `weight_sum`, accumulator, sufficient statistics, out-of-sample, fresh (bank) | plan, sink, collect, `collect_batches`, streaming engine, struct column, expression, scan, row group | regression, coefficient, residual, variance, correlation, kernel, weighted least squares, effective sample size |

**A term of art is defined at first use, and again where its concept
lives.** *The bank* is an English word with several meanings. It becomes a
term once the reader has been told it is a *model bank*, a set of models
fitted together over the same rows, held by the object `ModelBank`. The
introduction defines the library's words once, together. The section that
owns each concept restates its word, because readers arrive there from a
search as often as from the top.

**A Polars-internal word is replaced by what it does.** *Streams the plan's
rows through a fresh bank* asks the reader to know what a plan is, what
makes a bank fresh, and what streaming means here. *When the query runs,
its rows go through a bank that starts with nothing learned, one chunk at a
time* asks nothing. The replacements so far:

| the Polars word | say instead |
|---|---|
| a plan, a `LazyFrame` | a query, which runs only when you ask for its result |
| `sink_parquet` | runs the query and writes the result to a file without holding it all in memory |
| `collect_batches` | gives the result one chunk at a time |
| a struct column | one column whose value in each row is a record with named fields |
| a row group | a block of a parquet file |
| a fresh bank | a bank that starts with nothing learned |

**Audit the draft as the reader, not as the author.** Read each section
cold and list every word the reader above would stop at. Each one is
defined where it stands, defined earlier and linked, or replaced. The list
for the round-1 draft is in PHRASING.md: nineteen words, most of them the
library's own.

## 1. Every document has an altitude, and stays at it

Three altitudes, each with its own job:

| altitude | its job | what belongs there | what does not |
|---|---|---|---|
| **the introduction** (the top of the README) | make a reader who has never seen the library understand what it does and run one example | one paragraph per idea, the basic API example, the *consequence* of each design choice | mechanisms, measurements, per-model facts, deployment detail |
| **the user guide** (the rest of the README) | let a reader who has decided to use it do so correctly | the concepts and the shared API, code with comments, the reasons and warnings code cannot carry | how it is implemented, how it is tested, why it was designed that way |
| **the deep docs** (`docs/*.md`, docstrings, source comments) | answer the question the guide raised | mechanisms, measurements, design records, test citations | — |

**Depth has an address.** Detail removed from a higher altitude is moved,
never deleted. It goes to the document that owns it, and a one-line
cross-reference stays behind. *Any content we remove from here should have
a place somewhere else* (PHRASING: "What you get"). Before cutting, name
where the detail lands. If it lands nowhere yet, that is a document to
write, not a sentence to drop.

**State the consequence at the altitude that owns the consequence, and the
mechanism at the altitude that owns the mechanism.** In the introduction,
*this syntax reads all the data into memory up front* is the whole story.
*Polars hands a stateful expression its whole column at once, in either
engine* is the deep-doc sentence behind it. It does not belong up top,
even though it is true and interesting (PHRASING: "The expression form").
Neither altitude may *allude*: a cost is named by what is spent, and an
effect by what causes it, at whichever level that belongs.

## 2. Plan the map, then give each section one job

A reader meets a document's structure before its sentences, so the
structure is designed first, as a map, and only then written (PHRASING:
"the task 89 rewrite").

**Plan the whole map before rewriting a sentence.** List every section and
subsection, and say where each piece of the current text lands, before
touching the prose. Record the map in `docs/PLAN.md`, where decisions
live, with `←` marking what moved; task 89 is the model. A map catches
what a sentence-by-sentence pass cannot. The README's map caught these:

| what the map showed | where it went |
|---|---|
| *The models* and *Models*, two top-level sections nearly a thousand lines apart | one section, *Models*, with the model table at its head |
| `embargo`, a parameter every model shares, explained under *Preparing a stream* | *Labels that arrive late*, beside the other shared parameters |
| *Install* and *Against scikit-learn*, each a top-level section | subsections of *Introduction* and of *Performance* |

**A moderate number of large sections, each holding subsections.** About
ten top-level sections suit a document the size of the README, which had
seventeen. Group them by what the reader is doing: learning the idea,
running a bank, reading what it produced, choosing a model, tuning it, and
deciding whether to trust it. A catalogue goes one level down, under
families, rather than flat. The twenty-one models sit under four families, and
each family opens with one line on what its members share.

**The contents is a table.** One row per section, with its subsections
linked beside it, shows the hierarchy at a glance. The README's old
contents was one paragraph of sixteen links, all at the same level.

**The introduction leads with its example.** A reader who has watched a
bank run reads the idea and the terms as explanations of something seen.
So the README's *Introduction* opens with a first fit, its inputs and
outputs shown (§3). The idea, the terms and the guarantees follow it
(PHRASING: "the opener of Introduction"). §1 gives the introduction that
example as its job.

**Every section opens with a short introduction, or with a table of its
contents that explains the subsections.** A reader decides from the opener
whether the section answers their question. A table of the subsections
opens the section when it explains a lot about them or adds information
(PHRASING: "the opener of Introduction"). *How a bank sees a stream* opens
with the question each subsection answers and the parameters that answer
it. Otherwise the opener is one to three sentences that say what the
section offers: *here is a first model fit, and an introduction to the
library*. What every subsection
obeys is stated here too, before the first subsection rather than inside
one: the rules both of *Windowed means*' subsections follow sat inside the
first of them. A heading followed directly by a subheading, a table, a
list or a code block is a section with no opener. On 2026-10-04 sixteen of
the README's headings were. Four were top-level sections that went straight
to a subheading: *Introduction*, *Performance*, *Scope and integrations*,
and *Versions, testing and development* (PHRASING: "Openers, kinds and
tacked-on details").

**An opener never lists what follows in a sentence.** The subsections
are already listed at the top, in the contents table, and the bold leads
name the paragraphs. So a sentence that walks through them tells the reader
nothing new. One was *The idea says how, Terminology defines the words this
README uses, and What you can rely on lists what holds in every run*. The
user found it "like a small table of contents for the section" (PHRASING:
"the opener of Introduction"). Another was *The paragraphs below say what
that pass guarantees, when row order matters, and how a bank is run, saved
and checked*. For that one the user's wording was *here are a few points to
remember about fitting models with polars-online* (PHRASING: "The
paragraphs below"). Its signs are a subsection's title, in italics or as a
link, followed by *says*, *shows* or *lists*; *Below come*, *Below are* or
*The paragraphs below say*; and *follow the example*. The test is to strip
the names and topics of what follows from the opener: what remains must
still say something about the content. Seventeen of the README's openers
failed it on 2026-10-04.

**Name the kinds first, then give each its own subsection.** This library
does many things, and a paragraph that runs several of them together blurs
them all. When a section covers more than one kind of functionality, its
opener names the kinds, in a sentence or a short table. The kinds are what
the functionality does, such as fit, save and serve, and not the titles of
the subsections that develop them. Each kind is then
developed in a subsection of its own, or at least in a paragraph led by its
rule in bold. A paragraph that introduces two parameters, two modes or two
outputs is two paragraphs. The test is to list what each paragraph
introduces: more than one item means a split.

**Important information comes first, and nothing important is tacked on at
the end.** A section's last sentences are the ones a reader is least likely
to reach. So a rule, a caveat, a default, a limit or an exception never
closes a section as an afterthought. It moves into the paragraph whose
topic it is, or into the opener when the whole section depends on it. The
signs are a last paragraph that opens with *Also*, *Note*, *One more*, *The
exception is* or *Two limits remain*, and a lone sentence after the
example. *Output as Arrow* ended its longest paragraph with the rule that an
export can be read only once, and with the second call it documents,
`predict_arrow`.

**A heading's text is its anchor.** A heading may move to any level
without breaking a link to it. Change its wording only after finding every
link to it; `llms.txt` and `docs/RUNNER.md` both link into the README. The
model sections are the exception to moving freely.
`tests/test_api_links.py` and `tests/test_model_registry.py` find them by
their level, as `docs/EXTENDING.md` step 15 says.

**Every paragraph serves the heading.** A section about the clock does not
explain grouping, cite a test, or survey which models the clock applies to
(PHRASING: "A clock that is not time"). A section titled *Groups, weights
and warm-up* whose prose is entirely about warm-up is two sections that
have not been separated yet (PHRASING: "Groups, weights and warm-up").
When a paragraph does not serve its heading, the section that owns it
usually exists already. Move the paragraph there rather than widening the
heading.

**Siblings sit together.** Things a reader will compare are adjacent, in
the order the reader meets them. *With a decay* and *without a decay* are
one such pair, and the three ways to run a bank are another (PHRASING:
"Any row order").

**Lead with the library's own idiom.** This is a Polars library, so the
Polars form of anything comes first, the Python-loop form second, and the
exceptional form last (PHRASING: "A Python loop over chunks").

**The short version goes up front, the long version where it lives.** The
introduction carries one table row on testing, one sentence on saving
state and one number on parallelism. The sections that own those topics
carry the rest (PHRASING: "What you get", "Mistakes are named").

## 3. Code with comments, not prose that narrates code

**When the content is *describing*, show it as code.** If a passage says
what a parameter does, what comes out of a structure, or how to call
something, it becomes a code example with comments (PHRASING: "Whole
document — prose that should be code with comments"). One paragraph once
explained that `repr(bank)` shows its specs and groups, that
`bank.groups()` lists every group with its row count, and that
`bank.drop_groups(...)` forgets the quiet ones. It is now a code block
with comments (PHRASING: "A bank says what it holds", "As a query").
So is a paragraph that walks through a call's forms: *`po.target("price_5m",
relative_to="mid")` is the target `price_5m − mid` … In the CLI's TOML it
is a table* became three targets in one spec, a comment on each scale
(PHRASING: "`po.target`, in Relative and look-ahead targets").

| the prose was | it becomes |
|---|---|
| what each parameter does | one spec built with every parameter in play, each on its own line with its comment |
| what comes out of a structure | code that reads each field, with a comment saying what it holds |
| how to use an API | the call sequence, with a comment per step |

**An example shows what goes in and what comes out.** A reader cannot
follow a call on a frame they have never seen. So an example shows its
input, built in code the reader has seen, with a few of its rows printed in
a table where their shape matters, and the table each step returns. Its comments say what each line takes and what it
gives. The `refresh_time` example became readable once it built its eight
ticks in the open and printed the grid they make (PHRASING: "the
`refresh_time` example"). The first fit is held to the same rule (PHRASING:
"the opener of Introduction"). A printed table is a claim like any other,
so a test holds it to what the code returns, as
`tests/test_refresh_time.py` holds that grid.

**Every name an example reads was built by an example before it.** A
reader starts with nothing and runs the examples in order, so a frame
described in prose is a frame they cannot make. A frame that several
sections read is built once, in *Example data* after *Install*, and a
section that runs on it links there. I4 answered R1's *introduce every
frame an example reads* with a paragraph that listed `df`'s columns. It
sat at the end of a section about parameters for four rewrites, until the
user found it "out of context" and "very hard to understand" (PHRASING:
"The examples from here on read two frames"). No test could see it,
because the README test's namespace hands every block `df` ready-made. So
`test_every_name_a_readme_example_reads_was_built_by_an_earlier_one` parses
the README's blocks in order, and fails on a name that no block before it
builds.

**An example that reads the example data says so in the line just above
it, with a link.** The last sentence above such a block names what it
reads, as *This code uses `df` from [Example data](../README.md#example-data):*,
so a reader who opens the README at any section can tell where `df` comes
from, and build it (the user, 2026-10-05).
`test_an_example_on_the_example_data_says_so_just_above_it` holds every
block to it.

**An example's query reads a file, never a frame made lazy.** The library
is for streams too large to hold, and `df.lazy()` is a query over rows
already in memory. So an example saves its table to disk and reads it
back with `pl.scan_parquet`, as a reader's own stream is read. *Example
data* saves each frame it builds, and `lf` and `later` are scans of those
files. The user, 2026-10-06: "We don't want the example code in the readme
or anywhere else to be sprinkled with calls to .lazy(), show the example by
saving tables to disk and then reading them from a lazy frame".
`test_no_example_calls_lazy` holds the README, every document's examples
and the docstrings to it.

**An example matches each part of its claim, and says which part it
shows.** *Features and targets can be windows over the stream* promised
two things. Its examples, *the last minute's time-weighted mean or the next
minute's VWAP*, never said which was a feature and which a target. So the
user read it as a backward window with no forward target (PHRASING:
"Features and targets can be windows"). A claim about features and targets
names a feature that looks back and a target that looks ahead, each as
what it is.

**A long example names its steps in brief comments, and leaves the detail
to the prose.** The first fit runs four steps in one block: make up the
input, fit every row and serve, fit locally, and forecast a target that
looks ahead. The user found the section "much better" with its tables and
prose, and asked for comments "briefly describing what is happening but not
high detail since the section has good detail" (PHRASING: "the first fit's
code comments"). So each step opens with a one-line comment that says what
it does, and a line's own comment says what it gives.

**An example builds a thing in Polars' own terms when the library takes
them.** Where a spec takes a Polars expression, the example writes one, and
a helper that does the same job comes second, if at all. The user found
the expressions clearer than `po.target("price_5m", relative_to="mid")`
(PHRASING: "relative targets as expressions"). An expression such as
`(po.rewm_mean("price", ...) - pl.col("mid")).alias("fwd_move")` shows what
is computed and over which rows, in a language the reader already knows.

**Prose that stays must carry what a comment cannot hold at comment
length.** A reason earns its sentence: *filter after the bank, because a
filter before it makes Polars hold several blocks of each parquet file per
thread*. So does a warning, such as *a huge finite half-life is not `inf`*,
and so does a trade-off. A restatement of what the code already shows does
not (PHRASING: "As a query", clarification). The test for what remains:
would its comment be longer than the code it sits beside? Then it is a
concept, and it stays as text, above the code and not repeated inside it.

**Every code block runs.** A code example is a claim the reader trusts.
The test suite runs the README's python blocks and the docstrings'
examples, so a converted passage is also a test of what it describes. Run
a section's blocks while drafting it, not only at the end.
`_readme_namespace` in `tests/test_production_hardening.py` builds the
namespace each README block runs in, so a draft's blocks can run there
before they reach the README: `uv run python scripts/doc_review.py run
DRAFT.md` runs each the way the README test does. It holds what the README's examples have
built, so a block that runs there has not shown that a reader could run
it: the rule above checks that.

**An example may name a model. A survey may not.** A runnable example
needs a real spec, so it names one; that is what the library does. A
sentence that lists three models to make one general point is compressed
to the point itself, or cut. *Clocked on a feature, `ew_cov` reports
moments local in it, and `marginal` does the same pair by pair* was one
(PHRASING: "A clock that is not time", clarification).

## 4. Facts: whose, how measured, and where they live

**A fact about some models is not a fact about the bank.** *With decay
off, the bank is plain least squares* is true of the five models that
solve a normal equation, and false of the sixteen that do not (PHRASING:
"Or no clock at all"). *The order of the rows matters only when a model
forgets* was false of the models that step, filter or test, which depend
on the order either way (PHRASING: "the task 89 rewrite"). Shared sections
state what is shared: the concept and the API. Each model's own section
states what is that model's. When a claim is tempting to make about "the
bank", check it against the model table first.

**Check a fact against the code, not against the old prose.** A rewrite is
where stale facts get copied forward. The README listed the reasons a
prediction is withheld in an order other than the one `WITHHELD_REASONS`
declares, and that order is their precedence. Reading the constant, not
the README, is what caught it.

**A rewrite may drop words, never add facts.** A new form can ask for a
fact the old text never gave. In task 89 that form was the table: turning
prose into tables created cells, and two were filled without evidence
before both were caught. A column for scikit-learn said its chunking was
"not tested", a claim about another project that nobody had checked. A
column of agreements gave the lasso's check against its KKT conditions as
"exact", a precision nobody had measured. Leave such a cell empty, or fill
it with only what the source says. The lasso's cell now reads "the
conditions hold".

**Assurance in prose, measurements in tables.** A conceptual section
states the guarantee, such as *memory is proportional to the model's
state, not to the data that has passed through it*. The numbers behind it
(1.4 GB against 3.97 GB, 2e-13 against `lstsq`) go to the model's section,
a table, or `docs/PERFORMANCE.md` (PHRASING: "Any row order"). The
exception is a number that makes a point prose cannot. The lag a one-sided
kernel implies is easier to believe as *0.08 against 0.39* than as an
adjective.

**Numbers come from measurements, not from round figures.** A throughput
teaser is picked from a measured thread count, not interpolated to a tidy
one (PHRASING: "What you get", note on "5 threads").

**Do not claim documentation that does not exist.** *With documented
complexity in time* is a claim the reader will follow. If no document
states each diagnostic's cost, either write it or say what is actually
documented (PHRASING: "What you get", note on the diagnostics paragraph).

**State the rule and its reason, not the story of the bug.** *`bank.specs`
is a copy, and read-only … and used to: editing it in place left `coef()`
labelling coefficients from a spec the bank was not running* tells a
development memory. The reader needs the rule, a copy and read-only, and
the reason, that a stale copy would mislabel coefficients, in the present
tense (PHRASING: "`bank.specs` is a copy"). The history belongs in
`CHANGELOG.md` or `docs/PLAN.md`. Outside those two files, grep for
`used to`, `once did`, `the first version` and `before this was fixed`.

**An effect is attributed to its own cause.** *That order is what makes
every prediction honest … and it is what lets the whole thing run on far
more rows than fit in memory* fused two claims with one "and". The
predict-then-learn order does make a prediction honest. Bounded memory
comes from the models keeping only their state, in any order (PHRASING:
"The idea"). When a sentence carries two effects, check that each is
credited to the mechanism that actually produces it.

**A thing is not its file.** *State is a file* names a serialization as if
it were the concept. It sat two lines below a glossary that defines state
as what the bank has learned (PHRASING: "State is a file"). Say what the
thing is, then what can be done with it: *state can be saved to a file and
loaded back*. A heading never contradicts a definition the reader was just
given.

## 5. Sentences

**One idea per sentence.** A sentence carrying a rule, its reason and its
exception makes the reader hold all three to get any one.

**A semicolon joins the halves of one idea, never two ideas.** *A spec
names its model, the columns it reads and how it treats time; its first
argument is the spec's own name, which its output column takes* joined what
a spec describes to what its name is for. The user found that the second
half "does not belong so closely joined to the first part" (PHRASING: "the
opener of What a spec names"). Two ideas take two sentences, and an idea on
another topic goes to the paragraph that topic owns. The README of
2026-10-04 had 29 prose sentences with a semicolon, most of them two ideas.

**Aim near 20 words a sentence, and never 45.** Two claims joined by
*and* are usually two sentences. Task 89 took the
README's mean sentence from 23.4 words to 19.9. Its sentences of 45 words
or more went from 44 to 2, and both of those are two sentences the count
merges (§6).

**Lead a rule with its claim, in bold, then give the reason.** A reader
scanning for what to do finds it without reading the argument. In the
README, *Filter after the bank, not before it, unless the model must skip
those rows* is in bold, and the memory figures behind it follow.

**A sweep belongs in a table.** Six things behind semicolons is a table
that has not been drawn yet. §7 has the shapes.

**Name a thing in full on first use, and link it.** *A bank*, to a reader
who has not met `ModelBank`, is an English word with several meanings.
*A \[model bank\](...)* is the class (PHRASING: "Three ways to run a bank").
A term from a domain the reader may not share is glossed where it first
appears, or replaced by what it handles. *Session* is a capital-markets
word, so it became *market-data-like sessions, with their boundaries,
clock gaps and clock resets* (PHRASING: "a ceiling on gaps").

**Why, then what, then units.** The sentence on `min_weight` gives the
reason first: a model reports only once it has seen enough data to have
converged, so it never reports an uninformed number. Its units, `weight_sum`,
come last. The draft put the units first and the reason never (PHRASING:
"Groups, weights and warm-up").

**Plain English.** A compressed possessive (*the numbers are the bank's*)
reads as a puzzle before it reads as a sentence. So do a stacked pair of
em-dash clauses and an italicised copula doing a verb's work (PHRASING:
"The numbers are the bank's"). A pronoun or a back-reference, such as *the
loop above*, must point at something the reader has just seen, in the
document's current order (PHRASING: "As a query").

**Explain how something is done in procedural sentences, naming the tool,
the method and the point in the pipeline.** A compressed definition leaves
the reader to infer the step. *Or the target can be a column you make
first* names no tool and no point; *make the column with Polars'
`with_columns`, chained in the query before the bank* names both. The
user's model wording is *made with Polars expressions and chained before
the model* (PHRASING: "procedural sentences"). The test, for every how-to
sentence: could a reader act on it without inferring a missing step or
tool?

**State a condition with both branches.** The user's form is *X requires
Y; otherwise, do Z with [tool] at [step]*. A sentence that gives only the
case where something works leaves the reader to guess what to do in the
other. Its semicolon joins the two halves of one idea, as §5 allows.

**Do not define a thing by negation, or by "An X that is Y is a Z".** *A
target that needs no window is a column made first* does both: it names a
thing by what it lacks, and states the procedure as an identity. Say what
to do instead. *A transition is one row, so a weekend is one step* and *The
speed difference is a difference in semantics* are the same form; the
README of 2026-10-04 led eight paragraphs with it.

**A relative word says what it is relative to.** *First*, *later*,
*earlier*, *upstream* and *downstream* need their reference: first before
what, upstream of which step. *So difference upstream* becomes *so compute
the returns with Polars' `diff()` in the query before the bank*. An
ordering inside a list, as *the intercept first*, and a span with its
reference, as *five minutes later*, are fine. A position in the document
is not a reference: *from here on*, *below* and *above* say where the
writer stands, which a reader who jumped to the section does not share.
Name what they point at, such as *every example after the first fit*, or
link to it.

**A word with a Python meaning keeps it.** In a Python library's docs,
*exception* and *raise* mean an error raised. *Most models take all of
them, and each exception is named where its parameter is introduced* left
the user asking whether a model raised something (PHRASING: "each exception
is named"). For the everyday sense, say *except*, *apart from*, *a model
that does not take it*, *increase* or *a larger*. On 2026-10-04 the README
used *exception* in that sense seven times and *raise* for *increase* five.

**A *too* far from what it points at marks one topic split in two.**
*The diagnostics are out-of-sample too* sat four paragraphs below *Every
prediction is out-of-sample*, the claim its *too* leans on. The user read
that distance as a sign that the two paragraphs are one, to be combined and
simplified (PHRASING: "The diagnostics are out-of-sample too"). So a *too*,
an *also* or a *the same way* points at the sentence before it, or at most
the paragraph before. Farther than that, the two passages merge, under a
lead that states what both share. The user's lead for this one was
*Every prediction and diagnostic is out-of-sample*.

**Say what a parameter is used for, then show it.** Name the parameter by
what it is, say in a plain verb what it does, and give an example. The
user's own wording is the model: *the name of the spec (the first
parameter) is used to name the output column, for example*
`po.spec.ewridge("ridge", ...)` *adds a column named* `ridge`. The draft
said *its first argument is the spec's own name, which its output column
takes*. Its inverted clause, *which its output column takes*, makes the
reader rebuild the sentence before reading it (PHRASING: "the opener of
What a spec names").

The user rewrote one paragraph of *A first fit* to show the voice wanted
(PHRASING: "the local fit in A first fit"), and §8 sets the two versions
side by side. Four rules come from it.

**An aside inside a list goes in parentheses.** *Give the same spec a
clock, the timestamp column ts, and a finite half_life* reads as three
things to give. *A clock (timestamp column `ts`) and `half_life="10m"`*
reads as the two it is.

**Give the value, not a description of it.** `half_life="10m"`, the value
the example uses, tells the reader what to type. *A finite half_life* tells
them only what kind of thing to look for.

**Turn a mechanism into the advice it implies, and give the reason first.**
*Its state is weighted toward the last few tens of minutes, so serving from
the final state predicts with the most recent fit alone* left the reader to
work out what to do. *Because this is a local regression, remember that
saving the final state may be of limited value* says it, after its reason.

**Lead with the setting and what it produces, and call the result by its
short name.** *Using `coef_every=1` produces a time series of the betas at
every row* names the setting, a plain verb and the result. *The thing to
read is the path the coefficients took, which coef_every=1 writes on every
row, rather than the state they ended on* put a placeholder subject first,
the setting in a relative clause, and a contrast last.

## 6. The pass itself

A rewrite is a sequence of steps, and each has a check that does not rely
on rereading (PHRASING: "the task 89 rewrite"):

| step | what to do | how it is checked |
|---|---|---|
| 1. measure | count the prose before touching it | the counts below, with `scripts/doc_review.py measure` and `counts` |
| 2. map | plan every section, and where each piece of the current text lands (§2) | the map is in `docs/PLAN.md` before any prose changes, and gives each section's opener and the subsections its kinds become |
| 3. draft | rewrite one section at a time | each section's code blocks run as it is finished |
| 4. account | lose nothing | every number, backticked name and link target in the old text is in the new, or in the document it moved to: a diff, not a reading. `uv run python scripts/doc_review.py account OLD.md NEW.md [--also DOC.md ...]` lists what was lost |
| 5. structure | break nothing | every table row has its header's cell count; every in-page link lands on a heading; every anchor another file uses survives; a link to moved text points where it went. `uv run python scripts/doc_structure.py [FILE.md ...]` runs all four as GitHub renders the file, and reports a table GitHub shows as code or text |
| 6. measure again | compare | the same counts as step 1, side by side |
| 7. render | read it as the repository will show it | GitHub's API renders it: `POST /markdown` with `mode=markdown`. `uv run python scripts/doc_review.py render FILE.md OUT.html` does it through `gh`, as a standalone page |
| 8. gate | run the tests that read the README | `tests/test_production_hardening.py` runs every python block and checks that each name a block reads was built by a block before it, `tests/test_llms_txt.py` checks every anchor `llms.txt` uses, `tests/test_api_links.py` resolves every API link and walks every model section, and `tests/test_doc_structure.py` runs step 5 over every Markdown file git tracks |
| 9. cold read | read it in order as a newcomer | a reader with none of the pass's context (no brief, no earlier version, no code) reads the rendered page from the top. They mark each paragraph whose purpose they cannot tell from what comes before it, and each example they could not run from what the page has shown. A writer who checks one part against the rules misses both: the rules are about sentences, and a writer knows what each paragraph is for (README-ITERATIONS, I7's verdict) |

The counts, taken outside code blocks, tables and headings. `uv run python scripts/doc_review.py measure FILE.md` takes the first seven, and `counts FILE.md` lists every hit of the eleven numbered ones, in this order:

| count | what it finds |
|---|---|
| prose words | padding |
| the mean sentence, in words | density; near 20 reads easily |
| sentences of 35 words or more, and of 45 or more | the sentences to split first |
| *costs*, *pays*, *buys*, *for free*, *the price*, *the point* | a cost that does not say what is spent |
| the aphorism *X, not Y* | a contrast that alludes rather than states |
| tables, as rendered | comparisons drawn for the reader |
| code blocks | description shown as code |
| 1. headings whose next non-blank line is another heading, a table, a list or a code block | a section with no opener, unless the table is one of its contents that explains the subsections (§2) |
| 2. openers that name a subsection followed by *says*, *shows* or *lists*, or that say *Below come*, *Below are*, *The paragraphs below say* or *follow the example* | a contents in place of a summary (§2) |
| 3. prose sentences with a semicolon, outside tables and code | two ideas joined, unless the halves are one idea (§5) |
| 4. *too* meaning *also*, *also*, *as well*, *the same way* | a back-reference to read: farther back than the paragraph before, it marks a topic split in two (§5) |
| 5. a clause turned around, as *which its … takes* | an inverted clause, where the plain order reads more easily (§5) |
| 6. a sentence about the document: *the table below*, *this section*, *from here on* | a sentence about the page, where one about the library or the reader's task belongs (P1) |
| 7. bold leads of the form *An X … is a Z* | a definition where a procedure belongs (§5) |
| 8. nouns defined by what they lack (*that needs no*, *with no*) | a definition by negation, where the thing itself belongs (§5) |
| 9. *first*, *later*, *earlier*, *upstream*, *downstream* with no reference, and *from here on*, *below* or *above* pointing into the document | a step, or a place, the reader must find for themselves (§5) |
| 10. prose that names three or more of the library's own names (a parameter, a call, an output field or a quoted value) that no code block under the same heading shows, a parameter as `name=`, a call as `name(`, a value verbatim | a parameter, a structure or a call described where an example with comments belongs (§3). It ranked the `po.target` paragraph third of 25 on the I7 README; about a third of what it lists should be code, and the rest are rules and lists of names |
| 11. *exception* outside an error, *raise* meaning *increase*, table cells included | a word whose Python meaning the reader will hear first (§5) |

The cost words, as one pattern for `grep -E`, are
`costs|pays|buys|for free|the price|the point`. They and the aphorisms
count only outside a section whose heading is the frame for them, such as
*Performance*.

A flattering count is easy to produce, so measure honestly:

| trap | what it did | the fix |
|---|---|---|
| collapsing whitespace before splitting sentences | merged text across headings, tables and code: it reported 82 long sentences before and 50 after, where the fair counts were 44 and 2 | drop headings, tables and code first, then split each paragraph on its own |
| sentences the splitter cannot see | it joins two bullets with no blank line between them, a rule in bold with the sentence after it, and a sentence that opens with a lowercase name to the one before | read every long sentence the count reports; the README's last two were both merges |
| a check that finds nothing | a link check filtered out the renderer's `user-content-` prefix, and so checked zero links | print every count; zero is not a pass until it is shown to be one |
| counting a pattern's hits unread | `the point` matched "the point-biserial correlation" | read each hit before counting it |
| rendering with `mode=gfm` | used GitHub's comment rendering, which turns every line break into `<br>` | `mode=markdown`, which is how GitHub renders a README file |

**A docstring pass runs the same steps, with four differences** (task 149,
2026-10-03). A docstring is a deep doc, reached from the API reference or
`help()` by a reader who already has the README's words, so it carries the
mechanism, the equations, the units, the defaults and the refusals:

| step | in a docstring pass |
|---|---|
| 1 and 6, measure | the docstring's own text, with headings, code blocks and list-tables dropped and each paragraph split on its own. A definition-list header merges with its body in the count, so read every long sentence it reports |
| 4, account | page by page: the old docstring saved beside the draft, and a diff of its backticked names, numbers and link targets |
| 7, render | Sphinx with `-W`, which the gate runs: a docstring that is not valid reStructuredText fails there |
| 8, gate | the doc tests, and the tests that pin a docstring's text. `tests/test_temporal_clock.py` wants each clock parameter's entry to say "clock units" on one line, and `tests/test_weight_scale.py` pins whole phrases, line wraps included. Grep the tests for `__doc__` before a batch, and run them |

**Record every pass over the README in
[README-ITERATIONS.md](README-ITERATIONS.md).** Its prompt goes in
verbatim before the pass starts, with the ideas it takes on and the counts
of step 1. The counts of step 6 and the user's verdict follow, so the
record shows which prompts produced a better document.

**Keep the report verbatim.** When a reader names a problem, or confirms
an approach, PHRASING.md keeps their words unedited and the
interpretation separate. The rule drawn from it can then be checked
against what was actually said (PHRASING, "Format").

## 7. Tables, not bullet lists

A bullet list of things that differ along the same dimensions makes the
reader build the comparison in their head. A table builds it for them, so
use one whenever the items share dimensions (PHRASING: "the task 89
rewrite"):

| the content | the table | where the README does it |
|---|---|---|
| options compared | a row per option, a column per dimension | the three ways to hold a row back: scored, clock advances, fit moves, `weight_sum`, use it to |
| settings | the setting, its default, what it does, what it reads | the two warm-up gates |
| caveats | the caveat, and why | the three things to know before reading a Gram |
| failure modes | what you see, what it means, what to do | `micro`'s two wrong values of `eps` |
| a sweep of measurements | the setting, and the number at each | `bocpd`'s `prune_below`; the two thread configurations |
| a translation | the problem, and its fix | the query steps that can reorder rows |
| a catalogue | the item, how it works, what it is | the model table, grouped by family |
| a map | the section, and its subsections | the README's contents |

**The header names the dimensions.** Only a column of row labels may go
unnamed. A header left wholly empty is for a list of labels and values,
as in *What you can rely on*. A cell may stay empty; a cell filled only to
complete the grid may not (§4).

**A list stays a list when there is nothing to compare,** such as a
single point, or steps that are only an order. A reason, a warning or a
trade-off stays prose: it is an argument, and a table cell cuts an
argument into fragments.

## 8. Rewrites, as examples

Two rewrites show what the rules produce, as something concrete for a
future draft to be measured against: the task 89 rewrite of the whole
README, and one paragraph the user rewrote by hand.

### The rewrite of 2026-09-23

What the rules above did to the README's own text:

| before | after | rule |
|---|---|---|
| seventeen top-level sections, listed in a contents paragraph of sixteen links, with the twenty models flat | ten sections holding subsections, a contents table, and the models under four families | §2 |
| three bullets comparing weight `0`, a null target and `predict`, each a paragraph | a table with one row per way, and a column each for scored, clock, fit, `weight_sum` and use | §7 |
| a paragraph listing the readiness fields a model writes | a runnable block that fits a spec and reads each field, with a comment on each | §3 |
| the withheld reasons as `below_min_settled_frac`, `above_max_error_inflation`, `below_min_weight` | the order `WITHHELD_REASONS` declares, which is their precedence: `below_min_settled_frac`, `below_min_weight`, `above_max_error_inflation` | §4 |
| *The order of the rows matters only when a model forgets* | *Row order matters when a model forgets … Without one, the models that solve or accumulate give the same answer in any order* | §4 |
| *14 and 14 takes 2.6 s at a peak of 1.1 GB; 4 and 14 takes the same 2.6 s at 0.8 GB* | a table whose header says which number is which: Polars threads, bank threads, time, peak memory | §7 |
| *Two things the comments cannot carry*, then both rules in one paragraph | two paragraphs, each led by its rule in bold | §5 |

Measured the same way before and after, on paragraph boundaries (§6):

| measure | before | after |
|---|---:|---:|
| prose words | 13,496 | 11,590 |
| sentences of 45+ words | 44 | 2, both two sentences the count merges |
| sentences of 35+ words | 100 | 33 |
| mean sentence, in words | 23.4 | 19.9 |
| cost words | 16 | 0 |
| bullet items | 24 | 2 |
| tables, as rendered | 15 | 37 |
| top-level sections | 17 | 10 |
| python blocks, all running | 59 | 59 |

### A paragraph the user rewrote, 2026-10-04

The user rewrote *A first fit*'s paragraph on the local fit (PHRASING:
"the local fit in A first fit"). It went from six sentences and 124 words
to five and 93, at about the same length a sentence, so the gain is in how
each sentence is built.

Before:

> Give the same spec a clock, the timestamp column `ts`, and a finite
> `half_life`, and each row's weight halves every ten minutes of `ts`. The
> fit is now local: it describes the recent past, and it moves from row to
> row. Its state is weighted toward the last few tens of minutes, so
> serving from the final state predicts with the most recent fit alone. The
> thing to read is the path the coefficients took, which `coef_every=1`
> writes on every row, rather than the state they ended on. That path has
> one row per input row: each stock's exposure to each signal, as it stood
> before that row. It is a time series, to plot, difference, or compare
> between two stocks over a day.

After, in the user's words, with code formatting added and one fact
corrected (§4): a row's betas are the fit *after* learning that row.

> Give the same spec a clock (timestamp column `ts`) and `half_life="10m"`
> and each row's weight halves every ten minutes of `ts`. The fit is now
> local: it describes the recent past, and it moves from row to row,
> weighted toward the last few tens of minutes. Because this is a local
> regression, remember that saving the final state may be of limited
> value. But using `coef_every=1` produces a time series of the betas at
> every row, as they stood after learning that row. This can be used to
> track the betas over time.

| what changed | before | after | rule |
|---|---|---|---|
| an aside inside a list | *a clock, the timestamp column ts, and a finite half_life*: three things, apparently | *a clock (timestamp column ts) and half_life="10m"*: two | §5, an aside goes in parentheses |
| the value | *a finite half_life* | `half_life="10m"` | §5, give the value |
| what the final state is good for | *serving from the final state predicts with the most recent fit alone* | *Because this is a local regression, remember that saving the final state may be of limited value* | §5, the advice a mechanism implies |
| the setting and its result | *The thing to read is the path the coefficients took, which coef_every=1 writes on every row* | *using coef_every=1 produces a time series of the betas at every row* | §5, lead with the setting |
| where a fact sits | *weighted toward the last few tens of minutes* in the sentence about serving | the same words in the sentence about the fit they describe | §2, every paragraph serves its heading |
| the uses | *to plot, difference, or compare between two stocks over a day* | *to track the betas over time* | §5, one idea per sentence |
| how the sentences connect | six facts in a row | *Because …, remember … But …*: an argument | |
