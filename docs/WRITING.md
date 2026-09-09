# How the documentation is written

The rules the reader-facing docs are held to — the README first, then the
docstrings and the guides under `docs/`. Each rule is drawn from a specific
problem the user reported in [PHRASING.md](PHRASING.md); the entry that
produced it is named so the rule can be checked against the case, and the
list grows as that log does. Where two rules pull against each other, the
earlier-numbered one wins.

The one test every rule serves: **can this sentence be understood by a
reader who has not yet been told what it is about?** The writer already has
the frame, so the failure is invisible from the inside. It shows up as a
term used before it is named, a cost asserted without naming what is spent,
a mechanism alluded to rather than stated, or a section that assumes the
reader shares the author's map of the library.

## 1. Every document has an altitude, and stays at it

Three altitudes, each with its own job:

| altitude | its job | what belongs there | what does not |
|---|---|---|---|
| **the introduction** (the top of the README) | make a reader who has never seen the library understand what it does and run one example | one paragraph per idea, the basic API example, the *consequence* of each design choice | mechanisms, measurements, per-model facts, deployment detail |
| **the user guide** (the rest of the README) | let a reader who has decided to use it do so correctly | the concepts and the shared API, code with comments, the reasons and warnings code cannot carry | how it is implemented, how it is tested, why it was designed that way |
| **the deep docs** (`docs/*.md`, docstrings, source comments) | answer the question the guide raised | mechanisms, measurements, design records, test citations | — |

Two rules follow.

**Depth has an address.** Detail removed from a higher altitude is moved,
never deleted: it goes to the document that owns it, and a one-line
cross-reference stays behind. *Any content we remove from here should have
a place somewhere else* (PHRASING: "What you get"). Before cutting, name
where it lands; if nowhere yet, that is a document to write, not a sentence
to drop.

**State the consequence at the altitude that owns the consequence; state
the mechanism at the altitude that owns the mechanism.** In the
introduction, *this syntax reads all the data into memory up front* is the
whole story; *polars hands a stateful expression its whole column at once,
in either engine* is the deep-doc sentence behind it, and it does not
belong up top even when it is true and interesting (PHRASING: "The
expression form"). Neither altitude may *allude*: a cost is named by what
is spent, an effect by what causes it, at whichever level that belongs.

## 2. A section does one job, and its heading names it

**Every paragraph serves the heading.** A section about the clock does not
explain grouping, cite a test, or survey which models the clock applies to
(PHRASING: "A clock that is not time"). A section titled *Groups, weights
and warm-up* whose prose is entirely about warm-up is two sections that
have not been separated yet (PHRASING: "Groups, weights and warm-up").
When a paragraph does not serve its heading, the section that owns it
usually already exists — move the paragraph there rather than widening the
heading to cover it.

**Siblings sit together.** Things a reader will compare — *with a decay*
and *without a decay*; the three ways to run a bank — are adjacent, in the
order the reader will meet them, not separated by unrelated sections
(PHRASING: "Any row order").

**Lead with the library's own idiom.** This is a Polars library; the Polars
form of anything comes first, the Python-loop form second, the exceptional
form last (PHRASING: "A Python loop over chunks").

**The short version goes up front, the long version where it lives.** The
introduction carries one sentence on testing, one clause on saving state,
one number on parallelism; the sections that own those topics carry the
rest (PHRASING: "What you get", "Mistakes are named").

## 3. Code with comments, not prose that narrates code

**When the content is *what this call does*, show the call.** A paragraph
explaining that `repr(bank)` shows its specs and groups, that
`bank.groups()` is a frame of every group with its row count, that
`bank.drop_groups(...)` forgets the quiet ones — that is a code block with
comments, and it was written as prose (PHRASING: "A bank says what it
holds", "As a query"). Where the output is what the reader needs to see,
show the output too.

**Prose that stays must carry what a comment cannot.** A reason ("filter
after the bank, not before, because a filter before it holds several row
groups per thread in the reader") or a warning ("a huge finite halflife is
not `inf`") earns its sentence. A restatement of what the code already
shows does not (PHRASING: "As a query", clarification).

**An example may name a model. A survey may not.** A runnable example needs
a real spec, so it names one — that is what the library does. A sentence
that lists three models to make one general point (*clocked on a feature,
`ew_cov` reports moments local in it, and `marginal` does the same pair by
pair*) is compressed to the point itself, or cut (PHRASING: "A clock that
is not time", clarification).

## 4. Facts: whose, how measured, and where they live

**A fact about some models is not a fact about the bank.** *With decay off,
the bank is plain least squares* is true of the five models that solve a
normal equation and false of the fifteen that do not (PHRASING: "Or no
clock at all"). Shared sections state what is shared — the concept and the
API — and each model's own section states what is that model's. When a
claim is tempting to make about "the bank", check it against the model
table first.

**Assurance in prose, measurements in tables.** A conceptual section states
the guarantee — *memory is proportional to the model's state, not to the
data that has passed through it* — and the numbers that back it (1.4 GB
against 3.97 GB, 2e-13 against `lstsq`) go to the model's section, a
table, or `docs/PERFORMANCE.md` (PHRASING: "Any row order"). The exception
is a number that makes a point about the concept itself that prose cannot
make — the lag a one-sided kernel implies is easier to believe as *0.08
against 0.39* than as an adjective.

**Numbers come from measurements, not from round figures.** A throughput
teaser is picked from a measured thread count, not interpolated to a
tidy one (PHRASING: "What you get", note on "5 threads").

**Do not claim documentation that does not exist.** *With documented
complexity in time* is a claim the reader will follow; if no document
states each diagnostic's cost, either write it or say what is actually
documented (PHRASING: "What you get", note on the diagnostics paragraph).

## 5. Sentences

**One idea per sentence.** A sentence carrying a rule, its reason and its
exception makes the reader hold all three to get any one.

**A sweep belongs in a table.** Six things behind semicolons is a table
that has not been drawn yet.

**Name a thing in full on first use, and link it.** *A bank* to a reader
who has not met `ModelBank` is an English word with several meanings; *a
[model bank](...)* is the class (PHRASING: "Three ways to run a bank"). A
term from a domain the reader may not share — *session* is a
capital-markets word — is glossed where it first appears, or replaced by
what it handles: *market-data-like sessions, with their boundaries, clock
gaps and clock resets* (PHRASING: "a ceiling on gaps").

**Why, then what, then units.** *`min_periods` lets a model report only
once it has seen enough data to have converged, so it never reports an
uninformed number; it is measured in `n_eff` units, and …* — not the units
first and the reason never (PHRASING: "Groups, weights and warm-up").

**Plain English.** A compressed possessive (*the numbers are the bank's*),
a stacked pair of em-dash clauses, an italicised copula doing the work a
verb should — each reads as a puzzle before it reads as a sentence
(PHRASING: "The numbers are the bank's"). A pronoun or a back-reference
(*the loop above*) must point at something the reader has actually just
seen, in the order the document is now in (PHRASING: "As a query").

## 6. The pass itself

- **Measure it.** Count sentences of 45+ words before and after; count
  `costs|pays|buys|for free|the price|the point` and the aphorism shape
  *X, not Y* outside the sections whose heading is the frame for them.
- **Render before judging.** Read the page as the repo will show it, not
  the source.
- **Keep the report verbatim.** When a reader names a problem, the log
  keeps their words unedited and the interpretation separate, so the rule
  drawn from it can be checked against what was actually said
  (PHRASING, "Format").
- **Check the cross-references both ways** after moving anything: the
  anchor it left and the anchor it arrived at, and the tests that read the
  README (`tests/test_production_hardening.py` runs every python block;
  `tests/test_llms_txt.py` checks every anchor `llms.txt` uses;
  `tests/test_api_links.py` resolves every API link).
