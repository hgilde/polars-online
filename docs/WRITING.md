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

| section | what it asks of a draft |
|---|---|
| [0. Write for one stated reader](#0-write-for-one-stated-reader) | a named reader, and every term defined for them |
| [1. Every document has an altitude](#1-every-document-has-an-altitude-and-stays-at-it) | consequences up top, mechanisms in the deep docs |
| [2. Plan the map, then give each section one job](#2-plan-the-map-then-give-each-section-one-job) | the structure, before any sentence |
| [3. Code with comments](#3-code-with-comments-not-prose-that-narrates-code) | description shown as runnable code |
| [4. Facts](#4-facts-whose-how-measured-and-where-they-live) | every claim true, sourced and checked |
| [5. Sentences](#5-sentences) | one idea each, near 20 words |
| [6. The pass itself](#6-the-pass-itself) | a sequence of steps, each with a check |
| [7. Tables, not bullet lists](#7-tables-not-bullet-lists) | the shapes a comparison takes |
| [8. The rewrite of 2026-09-23, as examples](#8-the-rewrite-of-2026-09-23-as-examples) | what these rules produced |

## 0. Write for one stated reader

The reader of the README **knows statistics, a little Polars, and what it
means for rows to be in time order, and nothing about the internals of
Polars or of this project** (PHRASING: "README draft, round 1"). Every
sentence is written to that reader, and checked by reading it as them.

What that reader does not have must be defined before it is used, or
replaced with what it does:

| this library's own words | Polars-internal words | assumed statistics, safe to use |
|---|---|---|
| bank, model bank, spec, state, stream, chunk, clock, decay, halflife, session, group, warm-up, `n_eff`, accumulator, sufficient statistics, out-of-sample, fresh (bank) | plan, sink, collect, `collect_batches`, streaming engine, struct column, expression, scan, row group | regression, coefficient, residual, variance, correlation, kernel, weighted least squares, effective sample size |

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
| `label_delay`, a parameter every model shares, explained under *Preparing a stream* | *Labels that arrive late*, beside the other shared parameters |
| *Install* and *Against scikit-learn*, each a top-level section | subsections of *Introduction* and of *Performance* |

**A moderate number of large sections, each holding subsections.** About
ten top-level sections suit a document the size of the README, which had
seventeen. Group them by what the reader is doing: learning the idea,
running a bank, reading what it produced, choosing a model, tuning it, and
deciding whether to trust it. A catalogue goes one level down, under
families, rather than flat. The twenty models sit under four families, and
each family opens with one line on what its members share.

**The contents is a table.** One row per section, with its subsections
linked beside it, shows the hierarchy at a glance. The README's old
contents was one paragraph of sixteen links, all at the same level.

**Open a section with what its subsections share,** when the heading does
not already say it. *How a bank sees a stream* opens by saying that every
parameter in it belongs to every model, and where to find each one's units
and default.

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
with comments (PHRASING: "A bank says what it holds", "As a query"). Where
the output is what the reader needs to see, show the output too.

| the prose was | it becomes |
|---|---|
| what each parameter does | one spec built with every parameter in play, each on its own line with its comment |
| what comes out of a structure | code that reads each field, with a comment saying what it holds |
| how to use an API | the call sequence, with a comment per step |

**Prose that stays must carry what a comment cannot hold at comment
length.** A reason earns its sentence: *filter after the bank, because a
filter before it makes Polars hold several blocks of each parquet file per
thread*. So does a warning, such as *a huge finite halflife is not `inf`*,
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
before they reach the README.

**An example may name a model. A survey may not.** A runnable example
needs a real spec, so it names one; that is what the library does. A
sentence that lists three models to make one general point is compressed
to the point itself, or cut. *Clocked on a feature, `ew_cov` reports
moments local in it, and `marginal` does the same pair by pair* was one
(PHRASING: "A clock that is not time", clarification).

## 4. Facts: whose, how measured, and where they live

**A fact about some models is not a fact about the bank.** *With decay
off, the bank is plain least squares* is true of the five models that
solve a normal equation, and false of the fifteen that do not (PHRASING:
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

**Aim near 20 words a sentence, and never 45.** Two claims joined by
*and*, or by a semicolon, are usually two sentences. Task 89 took the
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
*A [model bank](...)* is the class (PHRASING: "Three ways to run a bank").
A term from a domain the reader may not share is glossed where it first
appears, or replaced by what it handles. *Session* is a capital-markets
word, so it became *market-data-like sessions, with their boundaries,
clock gaps and clock resets* (PHRASING: "a ceiling on gaps").

**Why, then what, then units.** The sentence on `min_periods` gives the
reason first: a model reports only once it has seen enough data to have
converged, so it never reports an uninformed number. Its units, `n_eff`,
come last. The draft put the units first and the reason never (PHRASING:
"Groups, weights and warm-up").

**Plain English.** A compressed possessive (*the numbers are the bank's*)
reads as a puzzle before it reads as a sentence. So do a stacked pair of
em-dash clauses and an italicised copula doing a verb's work (PHRASING:
"The numbers are the bank's"). A pronoun or a back-reference, such as *the
loop above*, must point at something the reader has just seen, in the
document's current order (PHRASING: "As a query").

## 6. The pass itself

A rewrite is a sequence of steps, and each has a check that does not rely
on rereading (PHRASING: "the task 89 rewrite"):

| step | what to do | how it is checked |
|---|---|---|
| 1. measure | count the prose before touching it | the counts below |
| 2. map | plan every section, and where each piece of the current text lands (§2) | the map is in `docs/PLAN.md` before any prose changes |
| 3. draft | rewrite one section at a time | each section's code blocks run as it is finished |
| 4. account | lose nothing | every number, backticked name and link target in the old text is in the new, or in the document it moved to: a diff, not a reading |
| 5. structure | break nothing | every table row has its header's cell count; every in-page link lands on a heading; every anchor another file uses survives; a link to moved text points where it went |
| 6. measure again | compare | the same counts as step 1, side by side |
| 7. render | read it as the repository will show it | GitHub's API renders it: `POST /markdown` with `mode=markdown` |
| 8. gate | run the tests that read the README | `tests/test_production_hardening.py` runs every python block, `tests/test_llms_txt.py` checks every anchor `llms.txt` uses, and `tests/test_api_links.py` resolves every API link and walks every model section |

The counts, taken outside code blocks, tables and headings:

| count | what it finds |
|---|---|
| prose words | padding |
| the mean sentence, in words | density; near 20 reads easily |
| sentences of 35 words or more, and of 45 or more | the sentences to split first |
| *costs*, *pays*, *buys*, *for free*, *the price*, *the point* | a cost that does not say what is spent |
| the aphorism *X, not Y* | a contrast that alludes rather than states |
| tables, as rendered | comparisons drawn for the reader |
| code blocks | description shown as code |

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
| options compared | a row per option, a column per dimension | the three ways to hold a row back: scored, clock advances, fit moves, `n_eff`, use it to |
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

## 8. The rewrite of 2026-09-23, as examples

What the rules above did to the README's own text, so a future draft has
something concrete to be measured against:

| before | after | rule |
|---|---|---|
| seventeen top-level sections, listed in a contents paragraph of sixteen links, with the twenty models flat | ten sections holding subsections, a contents table, and the models under four families | §2 |
| three bullets comparing weight `0`, a null target and `predict`, each a paragraph | a table with one row per way, and a column each for scored, clock, fit, `n_eff` and use | §7 |
| a paragraph listing the readiness fields a model writes | a runnable block that fits a spec and reads each field, with a comment on each | §3 |
| the withheld reasons as `below_min_settled_frac`, `above_max_error_inflation`, `below_min_periods` | the order `WITHHELD_REASONS` declares, which is their precedence: `below_min_settled_frac`, `below_min_periods`, `above_max_error_inflation` | §4 |
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
