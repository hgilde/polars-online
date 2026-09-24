# The state workflow: fit, save, serve, learn on

| | |
|---|---|
| **status** | decided and implemented on 2026-09-03 (PLAN task 20); the decisions are in §7 |
| **the syntax** | `lf.online.fit_predict(.., save_state=)`, proposed in §4 under the rules R1–R7 |
| **checked by** | §5's checks, which are `tests/test_frame.py`'s E35 tests |
| **measured on** | polars 1.34.0 (the floor), 1.38.1 and 1.44.1 (the pin), by `scripts/io_source_semantics.py` (§2) |
| **declined** | the memory side: a query that updates a `ModelBank` object, or takes one as `load_state`. The file is the state's one form in a query, and a bank object stays the tool of your own Python loop over chunks (§1, §3 B/E) |

**Read with two changes since.** Task 83 (2026-09-17) removed `po.run`,
the Python runner this document names as a third surface. The command line
`online` keeps the file-to-file runner, and in Python a query goes to
`ModelBank.fit(lf)` or `fit_predict_batches(lf)`, which chunk it
themselves. Every measurement and decision below stands. Where the text
says `po.run` as a surface that exists, read the CLI for the file-to-file
role and `ModelBank.fit(lf)` for the in-process one.

**And on py-polars 2.0.0rc1, R6 narrows** (measured 2026-09-10). When a
query fails after the bank, polars now stops the bank instead of running it
to the end of its input. It does so once the stream is long enough for the
failure to arrive before the bank reaches that end, and the state is then
not written. It was measured with chunks of 500 rows and a cast that fails
after the bank ([docs/RELEASE-READINESS.md](RELEASE-READINESS.md#polars-200rc1-measured-2026-09-10),
*Polars 2.0.0rc1, measured*):

| rows | chunks | 1.44.1 writes the state | 2.0.0rc1 writes the state |
|---:|---:|---|---|
| 4,000 | 8 | yes | yes |
| 40,000 | 80 | yes | no |

`tests/test_frame.py` accepts either outcome.

| section | subsections |
|---|---|
| [The workflow in four steps](#the-workflow-in-four-steps) | [each step, and what it guarantees](#each-step-and-what-it-guarantees) · [fit, and keep the state](#fit-and-keep-the-state) · [the one gap](#the-one-gap-a-failure-after-the-bank) · [the rules, and the evidence for each](#the-rules-and-the-evidence-for-each) |
| [The research behind it, 2026-09-03](#the-research-behind-it-2026-09-03) | [0. the ask](#0-the-ask-and-the-answer-in-one-paragraph) · [1. what existed before](#1-what-existed-before-the-decision-per-step-and-surface) · [2. how polars runs a source](#2-how-polars-executes-a-python-io-source--measured) · [3. the candidates](#3-the-candidates) · [4. the proposal](#4-the-proposal-a-made-exact) · [5. the prototype's checks](#5-the-prototypes-checks-real-bank-3-groups--1000-rows-chunks-of-250) · [6. usability, performance, stability](#6-usability-performance-stability--checked-not-assumed) · [7. decisions](#7-decisions-taken-2026-09-03) · [8. where it landed](#8-where-it-landed) · [9. reproduce](#9-reproduce) |

## The workflow in four steps

This section is the one to read to *use* the state workflow. The numbered
sections after it are the research that decided it.

A state file is the same file, byte for byte, whichever wrote it (§5 C2).
A query and an eager frame take it as `load_state` and `save_state`, and
the TOML under the same two keys. The command line spells them `--resume`
and `--save-state`, and a `ModelBank` has `load` and `save`.

### Each step, and what it guarantees

| step | in Python | on the command line | what it guarantees | rules |
|---|---|---|---|---|
| **1. Fit, and keep the state** | `lf.online.fit_predict([spec], save_state="ridge.state")` | `save_state` in the TOML, or `--save-state` | the state is written whole, once the bank has been fed its last row | R1, R2, R4, R6, R7 |
| **2. Inspect it** | The file is a `ModelBank`: `po.ModelBank.load(path)` gives the object back, with `coef()` (the betas), `last_row()`, `summary()` and `describe()` (what it was fed), `gram()` and the rest. `save_bytes()` / `load_bytes()` are the same state as bytes, for a store that is not a file | | none of them needs a row of data | |
| **3. Serve from it, learning nothing** | `lf.online.predict("ridge.state")` | `online --resume p --predict`, which refuses `--save-state` | every row is scored against the state as it stands, and the state never moves, so the same rows score the same way twice, on any thread count | R3, R5 |
| **4. Learn on from it** | `lf.online.fit_predict(load_state="ridge.state", save_state="ridge.state")` | `online --resume p --save-state p` | it resumes and replaces. The specs come from the file, and a `load_state` that is not a bank this build can read, or whose specs disagree with the ones passed, is a `ValueError` before any row is read | R2, R3 |

### Fit, and keep the state

The bank in a query learns as the rows stream through it, and the state
is written once the bank has been fed its last row:

```python
import polars as pl
import polars_online as po

spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                       clock="t", halflife=600.0, max_dclock=300.0)

(pl.scan_parquet("2025.parquet")                           # a query over the file; nothing is read yet
   .online.fit_predict([spec], save_state="ridge.state")   # the bank learns a chunk at a time, and saves at the last row
   .sink_parquet("2025_scored.parquet"))                   # runs the query, writing the scored rows without holding them all
```

**`save_state` is written whole, and only once the bank has been fed its
last row.** The write is atomic: a reader sees the old file or the new
one. A run the caller abandons, or one ended by an error inside the bank,
leaves the file as it was. An error in a later step of the query still
writes it, which is [the one gap](#the-one-gap-a-failure-after-the-bank).
The command line does the same from a `save_state` in the TOML, and saves
only after its output is committed.

### The one gap: a failure after the bank

**A query that fails after the bank can still write the state.** Polars
runs the bank to the end of its input before it reports an error from a
later step (F5), so the bank reaches its last row and writes. A full disk
under `sink_parquet`, or a bad cast, then leaves `save_state` written with
the whole stream's state, while the query's own output is missing. With
`load_state=p, save_state=p`, a rerun learns the same data twice.

**Where the state must land only with the output, use the command line.**
It saves the state only after its writer has committed the output.

**Otherwise, write a dated `save_state` per batch of data**, such as
`ridge-2026-01.state`. That keeps a rerun from consuming the state it
needs, and the files keep an audit trail.

On py-polars 2.0.0rc1 the gap narrows: a long enough stream is stopped
rather than run to its end, and the state is not written (the note at the
top).

### The rules, and the evidence for each

Each guarantee above is one of seven rules, stated in full in §4. The facts
about Polars they rest on were measured (§2, F1–F8), and the prototype's
checks held them (§5, C1–C8b). Why the alternatives were rejected is §3 and
§7.

| rule | what it guarantees | measured | checked |
|---|---|---|---|
| R1 | `save_state` is written when the bank has been fed its last row: the input's end, or the rows a `head(n)` asked for. Never on a run the caller abandoned, or one the bank ended with an error. The write is atomic | F7 | C5, C6 |
| R2 | every run of a query ends in the same state, so two runs write the same bytes. A run with nothing to write runs once where a query uses it twice | F1, F2 | C3 |
| R3 | `load_state` and `predict(path)` are read when the query is built, so a query run twice gives the same result | F2 | C8b |
| R4 | the bank is fed exactly the rows the query pulled | F4 | C1, C4, C4b |
| R5 | `predict` refuses `save_state`, as `online --predict` does | | `predict(save_state=)` refused (§8) |
| R6 | a failure after the bank still writes the state: [the one gap](#the-one-gap-a-failure-after-the-bank) | F5 | C7 |
| R7 | building a query writes nothing | F3 | |

## The research behind it, 2026-09-03

What follows is the research that decided the design, kept as a dated
record: the code, the tests and other documents cite its rules,
measurements and checks by their IDs. Where the code has moved on since, a
note headed *Since then* says so, and the sentences around it keep their
date. The record uses Polars' own words, and one of this library's:

| word | what it means here |
|---|---|
| plan | a `LazyFrame`: a query, which runs only when you ask for its result. The *plan surface* is `lf.online.*` |
| source | the node the bank is registered as, through `register_io_source`, which the engine pulls chunks from |
| engine | one of the two ways Polars runs a plan: in memory, or streaming |
| pushed | handed to the source by Polars: a `head(n)` as `n_rows`, a filter as `predicate` (F4) |
| drained | run to the natural end of its input (F5) |
| pure | every execution starts from the same state, so collecting twice gives the same frame. Polars' `is_pure` flag lets it run a pure source once where a query uses it twice (R2) |
| the loop | a `ModelBank` fed chunk by chunk in your own Python loop |

### 0. The ask, and the answer in one paragraph

The ask was four steps. **(1)** Fit a model online in bounded memory.
**(2)** Export the model state, optionally to disk. **(3)** Load the state
and predict online in bounded memory *without* updating it. **(4)** Load
the state and update it with new data. The requirement is that these read
naturally in the polars-native syntax, `lf.online.*`.

Every step already existed on the `ModelBank`, `po.run` and CLI surfaces,
and steps (1), (3) and the load half of (4) exist on the plan surface. The
one thing the plan surface cannot do is **let state out**:
`lf.online.fit_predict` was made *pure* on purpose (ENHANCEMENTS E33, PLAN
§11a), with a fresh bank per execution and nothing saved. The research
question was therefore narrow: *is there a way for a plan to carry state
out that survives the way polars actually executes a Python source?* The
measurements say yes, with exactly one way: **`save_state=`, the runner's
keyword, on the plan**, written atomically when the source has fed the bank
its last row. It is safe precisely *because* the plan is pure. Polars runs
a plan's source once per execution, and twice, concurrently, when a query
uses the plan twice. A pure plan's end state is the same every time, so the
write is idempotent. Two things must change with it for the semantics to be
exact. The source must feed the bank only the rows a `head(n)` asked for,
where on 2026-09-03 it fed the whole chunk and trimmed the output. And
`load_state` must be read when the plan is built, rather than when it runs.
One thing cannot be had: the source does not learn whether the *query*
succeeded. So a node after it failing still leaves the state written,
because polars drains the source first, in every engine, on the 1.x
versions measured; the note at the top has 2.0. Where
"state only if the output landed" is required, the CLI is that call and
stays so, as `po.run` was until task 83 removed it.

### 1. What existed before the decision, per step and surface

| step | `ModelBank` | TOML / CLI (and `po.run`, until task 83) | plan `lf.online.*` (and `df.online.*`, `po.fit_predict`) |
|---|---|---|---|
| (1) fit online, bounded | `bank.fit(lf)` / `fit_predict_batches(lf)`, which chunk the plan; or `fit_predict(chunk)` in a loop of your own | `online --config bank.toml` — polars reads in chunks, the bank fits, a writer thread writes; O(state + chunk) | `lf.online.fit_predict(specs).sink_parquet(..)` / `.collect_batches()` — O(chunk) |
| (2) export state | `bank.save(path)` (atomic), `bank.save_bytes()`, `pickle`; inspect with `groups()`, `coef()`, `last_row()`, `summary()`, `describe()`, `gram()` | `save_state=` / `save_state = "…"` / `--save-state`, written after the output is committed | **none** — the bank was dropped when the source ended (E33: "no `save_state`"); now `save_state=` (§4) |
| (3) load, predict, no update | `ModelBank.load(path).predict(df)`; `load_bytes` | `--resume p --predict` | `lf.online.predict(bank_or_path)` — pure, the bank does not move |
| (4) load, update | `ModelBank.load(p).fit_predict(df)` then `save` | `--resume p --save-state p` | `lf.online.fit_predict(load_state=p)` learns on from `p`, **could not save**; now `load_state=p, save_state=p` |

So the gap was one cell, twice: getting state *out* of a streamed plan. The
Rust side (`crates/online-polars/src/runner.rs`, the CLI) needs nothing. It
has both keywords, and it saves after the writer commits.

The state file itself is one msgpack blob for the whole bank: `BankFile`,
in `crates/online-polars/src/bank.rs`. It holds a magic string,
`format_version`, `schema_version`, the package version, the specs and
`rows_fed`, and per spec a sorted list of (group key, stream state) pairs.
A stream state is the models and the diagnostic accumulators. Since 0.2.0
it also holds the output row of the last learned row, which `last_row()`
reads, and the data summary, which `summary()` and `describe()` read. The
file is written through `atomic.rs`: a temporary sibling, fsync, then a
rename over the destination. The temporary was named `.{file}.tmp{pid}` at
the time of the research, and has been `.{file}.tmp{pid}-{seq}` since
decision 4 (§7). Serialization is deterministic, since the groups are
sorted, so two saves of the same state are byte-identical; §5 relies on
that to compare states.

*Since then* (checked on 2026-09-23 against `BankFile`): the file has
gained five optional fields, each skipped when it is empty. They are
`closed`, the rows of closed groups that nobody has drained yet (E54),
`high_water`, `pca_prev`, `pca_prev_by_group` and `key_integer`. A stream
state holds more too, such as `pending`, the rows a `label_delay` has
accepted and not yet learned from.

### 2. How polars executes a Python IO source — measured

`lf.online.fit_predict(specs)` is a `register_io_source` source, and
nothing in polars' documentation says four things about such a source. How
often does it run per query? Do two runs of it overlap? What does it see
when a later node fails? And what happens to a run the caller abandons?
Each of those decides whether a side effect inside the source is sound.
`scripts/io_source_semantics.py` measures them with a toy source that logs
its own life. The numbers are the same on 1.34.0, 1.38.1 and 1.44.1 unless
noted.

| | fact | measured |
|---|---|---|
| **F1** | **One run per execution, and no sharing.** A plan used twice in one query — `lf.join(lf)`, `pl.concat([lf, lf])`, `pl.collect_all([lf.sink_parquet(a, lazy=True), lf.select(..).sink_parquet(b, lazy=True)])` — runs its source twice. Common-subplan elimination does not apply to a Python source, on or off. | 2 runs in every case |
| **F2** | **Those two runs are concurrent**, on two threads, in both engines. | second start before the first end, every case |
| **F3** | **Building or inspecting a plan runs nothing**: `collect_schema()`, `explain()`, `explain(engine="streaming")`. | 0 runs |
| **F4** | **`head(n)` reaches the source as `n_rows`** (when it is pushed — not through a `sort`), and the source ends its own run by returning; a filter reaches it as `predicate`; the streaming engine hands `batch_size=100_000`, the in-memory engine `None`. | as `_frame.py` already assumes |
| **F5** | **A node after the source failing does not stop the source.** With a source that takes 0.5 s and a strict cast that fails on its second chunk, every engine (`collect()`, `collect(engine="streaming")`, `sink_parquet()`) drains the source to its natural end and raises *afterwards*. A failed `sink_parquet` leaves its file behind (0 bytes on 1.34.0, 323 bytes on 1.38.1/1.44.1). | raised after 0.58 s, source ended, in all three |
| **F6** | An exception inside the source surfaces as `ComputeError: caught exception during execution of a Python source, exception: ...`. | as `_frame.py` documents |
| **F7** | **An abandoned run is not ended.** `next()` on `collect_batches` then dropping the iterator, or `for .. break`, leaves the generator suspended after it ran 4–5 chunks ahead; on 1.38.1/1.44.1 it is closed (`GeneratorExit`, `finally`) only when the *plan object* is dropped, on 1.34.0 when the iterator is. | no `end`, `finally` later or much later |
| **F8** | `sink_batches` exists from 1.34.0 (a callback per batch). A callback returning `True` stops the run on 1.34.0/1.38.1 and drains it on 1.44.1. Not needed by anything below; recorded because it differs. | |

Two consequences drive everything in §3–§4.

**Any state a plan mutates is fed the stream twice, at once, whenever the
plan is used twice in a query** (F1+F2). That is E33's objection to a
`bank=` the plan would learn into, now with the failure mode measured: not
a wrong count but a data race.

**A side effect can only be tied to the source's own end** (F5, F7). The
source never learns whether the query succeeded, and `finally` runs at a
version-dependent moment that can be long after the user moved on. So a
write happens when the source has delivered its last row, or not at all.

### 3. The candidates

| | form | verdict |
|---|---|---|
| **A** | **`lf.online.fit_predict(specs, save_state=path)`** — the runner's keyword on the plan; the state is written when the source ends, atomically. | **Recommended.** Sound under F1/F2 because the plan is pure: both concurrent runs write the same bytes (§5 C3). Needs the two changes in §4 to be exact under `head(n)` and under a file that changes between build and run. One documented gap (F5): a downstream failure still writes. |
| B | `lf.online.fit_predict(bank)` — a caller's bank the plan learns into. | **Rejected**, as in E33, now on measurement: `lf.join(lf)` or `collect_all` feeds it the stream twice concurrently (F1+F2). A "used once" guard cannot distinguish the second run inside one query from a legitimate second `collect()`, and would fail the query from inside the source. |
| C | State as data: `lf.online.fit(specs)` → a plan of one row, `state: Binary`, that a user sinks or `ModelBank.load_bytes` reads; `load_state` accepting a frame. | Pure and polars-native, but **predictions and state come from two plans, so one pass of the input becomes two** (F1: `collect_all` does not share the source), and the bank runs twice for a user who wants both. Nothing single-pass is possible without a side effect. Keep as an idea if a "state is a frame" use case appears (a join of states? none known). |
| C′ | State on the output's last row: a `state: Binary` column, null except on the last row the source delivers. | Pure and single-pass, but the consumer must find that row after a sink (re-scan the output), a filter after the bank can drop it, and every row carries a null cell. Judged heavier than A for no gain in safety: it too is written before the query's outcome is known. |
| D | Keep the plan pure; state comes from `po.run` / `ModelBank` (status quo). | What the README said on 2026-09-03. It answers the four steps, but not in the polars-native syntax the user asked for; and `po.run(input=lf, output=path)` has no lazy output — it is the terminal op. Stays as the *transactional* call (§4 R6). |
| E | `save_state=callable`: `fit_predict(specs, save_state=lambda bank: ...)` — in-memory export from a plan, in the shape of `sink_batches`. | Possible later on top of A (same trigger, same F1/F2 caveat: called once per run, twice in a self-join, on polars' threads). Not needed for the four steps: in-process state is `ModelBank`'s job (`fit_predict_batches`, `save_bytes`). Not recommended now. |
| F | `lf.online.run(specs, output=path, save_state=p)` — the runner as a namespace method, for a uniform reading of the four steps. | A one-line alias of `po.run(input=lf, ..)`; adds surface, no capability. Not recommended; `po.run` takes a `LazyFrame` already. |

### 4. The proposal: A, made exact

The four steps in the polars-native syntax, as they would read:

```python
import polars as pl
import polars_online as po

spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                       clock="t", halflife=600.0, max_dclock=300.0)

# (1) + (2): fit online in O(chunk); the state is written when the stream ends
(pl.scan_parquet("2025.parquet")
   .online.fit_predict([spec], save_state="ridge.state")
   .sink_parquet("2025_scored.parquet"))

# (2) as bytes, or to inspect: the file is a ModelBank
bank = po.ModelBank.load("ridge.state")
blob = bank.save_bytes()

# (3): score new rows against the state, learning nothing (exists today)
(pl.scan_parquet("2026-01.parquet")
   .online.predict("ridge.state")
   .sink_parquet("2026-01_scored.parquet"))

# (4): learn on from the state; the new state replaces the old when the stream ends
(pl.scan_parquet("2026-01.parquet")
   .online.fit_predict(load_state="ridge.state", save_state="ridge.state")
   .sink_parquet("2026-01_scored.parquet"))
```

`df.online.fit_predict(specs, save_state=)` and `po.fit_predict(frame, specs, save_state=)`
are the eager and typed twins (`bank.fit_predict(df)` then `bank.save`).
The proposal also had `load_state` accept a `ModelBank`, copied at build
time (R3), so that step (4) had an in-process form,
`lf.online.fit_predict(load_state=bank, save_state=...)`. Decision 3 (§7)
declined it, and `load_state` takes a path. The vocabulary is then one pair
of words on every surface: `load_state` / `save_state` on the plan, in
`po.run` and in the TOML, spelled `--resume` / `--save-state` on the CLI.

The rules, each checked in §5:

**R1 — the write is tied to the source's end.** `save_state` is written
when the source has fed the bank its last row: the input's natural end, or
the `n_rows` a pushed `head(n)` asked for. It is never written in `finally`
(F7), never on an error the bank raised, and never on an abandoned run.
After a bank error the run ends without reaching the write, so the file, if
any, is untouched. The write is atomic, through `ModelBank.save`, so a
reader sees the old file or the new one.

**R2 — idempotent under re-execution.** Every run of a plan ends in the
same state, because every run builds its own bank from the same starting
bytes and feeds it the same rows in the same order. So two runs write the
same bytes (C3): `collect()` twice writes twice, and two concurrent runs
write the same file at the same time. Two in-process writers then need
distinct temporaries. `atomic.rs` named its temporary by pid only, so
either the Python side serialises the writes with a lock (the prototype),
or the temporary's name gains a thread id or a counter. One of the two is
required (F2). *Taken: the counter, in `atomic.rs` (§7, decision 4).*

Idempotence is not purity, and the difference is what decides `is_pure`
(`register_io_source`, py-polars 1.34+). Since 2026-09-10 the source
declares it **when the run has no writes**, meaning no `save_state` and no
`closed_groups` sidecar, and not otherwise. Polars then runs one plan once
where a single query uses it twice (a self-join, `pl.concat([plan, plan])`),
instead of twice, concurrently, and the duplicate work goes away. A run
that writes keeps the pair of runs, and the counter with it. Task 77 has
the reasoning and the measurements.

**R3 — `load_state` is read when the plan is built.** The plan carries the
state it was built from, as bytes that each run deserialises, the way
`df.lazy()` carries the frame. It does not re-read the file at run time,
the way `scan_parquet` re-reads a file. Without this,
`load_state=p, save_state=p` used twice in one query races the second
run's load against the first run's write (F2). And a plan collected twice
would not be the same frame if the file changed in between (C8b). The probe
that computes the output schema already opens the file at build time, so
reading the state then adds no extra read. The trade-off is that a plan
built against `p` does not see a later `p`; build it again. `predict(path)`
follows the same rule, for consistency; before this rule its docstring said
"loaded each time the plan runs". `predict(bank_object)` stays by
reference, as documented, since `predict` never moves the bank.

**R4 — the bank is fed exactly the rows the query pulled.** Before this
rule, a pushed `head(n)` fed the bank the whole first chunk and trimmed the
*output* to `n`. With state observable, that would make
"the state after `head(5)`" mean "after 100,000 rows". The source therefore
truncates the *input* chunk to the rows still wanted, before the bank sees
it. The
delivered rows are bit-identical, since they are out-of-sample by
construction: row *i* depends on rows < *i*. The one difference is `coef`,
a reporting cadence emitted on each chunk's last row, which now appears on
the last delivered row (C1). A `head(n)` does less work, and the README's
existing sentence "`head(n)` learns from the first `n` rows and no more"
becomes true of the bank as well as of the output. `sort().head(n)` is not
pushed (F4), so the bank sees every row and the state is the whole
stream's (C4b). That is right, since the query did read every row.

**R5 — `predict` refuses `save_state`**, as `online --predict` does. It
learns nothing, so there is nothing to save.

**R6 — the one gap, stated.** Because polars drains the source before
surfacing a later node's error (F5), a query that fails *after* the bank
leaves `save_state` written, while the query's own output is missing. A
full disk under `sink_parquet` is such a failure, and so is a bad cast. The
state written is the complete, valid state of the whole stream. With
`load_state=p, save_state=p`, a rerun then learns the data twice. The CLI
saves only after the writer has committed the output, and is the call for
"state only if the output landed". The plan's docstring says so, and
recommends a dated `save_state` per batch of data (`ridge-2026-01.state`)
for the in-place pattern, which also keeps an audit trail. This is polars'
own precedent: `sink_parquet` too leaves its file when the query fails
(F5).

*Since then:* the README states the gap under *As a query:
`lf.online.fit_predict`*, and the dated-file advice is in the plan's
docstring alone. On py-polars 2.0.0rc1 the gap narrows, as the note at the
top records.

**R7 — building a plan writes nothing** (F3). The schema probe runs the
bank on zero rows and drops it.

### 5. The prototype's checks (real bank, 3 groups × 1,000 rows, chunks of 250)

The checks ran a stand-alone `register_io_source` source with R1–R4 built
in, against `po.ModelBank` and `po.run`, in a scratchpad script,
`proto_save_state.py`. They became `tests/test_frame.py`'s E35 tests when
the rules were implemented (§8).

| | claim | result |
|---|---|---|
| C1 | `head(613)` with the input truncated (R4) delivers the same rows as today's output-side trim, except `coef` on the last delivered row | identical bar `coef`; `coef` present on row 613 with R4, absent today |
| C2 | the state a plan writes == `po.run(save_state=)`'s == a `ModelBank` fed the same chunks | byte-identical, all three |
| C3 | self-join, `pl.concat([plan, plan])`, `collect_all([sink, sink])`: two writes, two threads, file == reference | 2 writes on 2 threads in each case, file byte-identical to C2 |
| C4 | `head(613)` writes the state of a bank fed `df.head(613)` | byte-identical (`rows_seen=613`, 1 group) |
| C4b | `sort().head(5)` writes the whole stream's state | byte-identical to C2 (`rows_seen=3000`) |
| C5 | `for .. break` over `collect_batches` writes nothing | no file |
| C6 | a bank error mid-stream (a null clock at row 700) writes nothing | `ComputeError`, no file |
| C7 | a failing node after the bank still writes (R6) | file present, byte-identical to C2 |
| C8 | `save_state=p` over rows 0–599, then `load_state=p, save_state=p` over 600– : the resumed rows equal one continuous stream's (bar `coef`'s cadence at the chunk boundary, as the README states for any resume) and the final state is byte-identical to the continuous run's | both hold |
| C8b | the plan built against `p` re-run after `p` was overwritten gives the same frame (R3) | same frame |

### 6. Usability, performance, stability — checked, not assumed

**Usability.** One keyword, already known from the runner, sits in the
place a user looks for it, and the four steps read in one syntax (§4). The
eager and typed twins take the same keyword. Nothing existing changes
meaning except two things, both visible only in cases that were
unobservable before: `predict(path)`'s read time (R3), and `coef` on a
`head(n)`'s last row (R4).

**Performance.** `save_state` adds one serialization at the end of a run,
which takes milliseconds: the state is O(specs × groups × p²). R3 holds the
state bytes in the plan for its lifetime, the same order as one bank. R4
does strictly less work for a `head(n)`. Nothing on the per-row path moves,
so the 12M-row numbers in `docs/PERFORMANCE.md` §11 are unaffected.

**Stability.** The design depends on none of the version-specific facts. It
is correct whether the source runs once or twice (R2), and whether polars
drains the source or stops it on error
(R6 narrows if polars ever stops it). It is correct whenever `finally`
runs, too (R1 never uses it). The write is atomic and idempotent. The only
new failure mode is R6, which is documented and has a supported
alternative. The floor stays 1.34.0. *Since then:*
py-polars 2.0.0rc1 does stop a long enough stream, as the note at the top
records.

**Rules kept.** Hard rules 2 and 3 are untouched (C1, C8), and so are
`n_eff` and zero weight. There is no new Rust surface unless the
temporary's name changes (R2). That would be a
`crates/online-polars/src/atomic.rs` change, with no linkage and no
`SCHEMA_VERSION` bump, since the file format does not change.

### 7. Decisions (taken 2026-09-03)

1. **Add `save_state=` to the plan (A)**: yes, with R1–R7. The user's
   framing: the file saves are good and easier for a user to understand,
   and the memory side is not worth its problems.
2. **R3, `load_state` read at build time**: yes, including for
   `predict(path)`. The alternative, re-reading per run like
   `scan_parquet`, keeps a race in `load_state=p, save_state=p` when a plan
   is used twice in one query. It was taken with decision 1, as the rule
   set proposed. It is the one change in observable behaviour for existing
   code: a plan built before the file changed keeps the frame it had
   (`tests/test_frame.py`, `test_load_and_save_the_same_path_resumes_in_place`).
3. **`load_state` accepting a `ModelBank`**: no. The memory side is out of
   scope: a bank object is the loop's, and the file is the plan's. Nothing
   on the plan surface mutates a `ModelBank`.
4. **R2's mechanism**: fixed at the root, where the collision was, rather
   than with a lock. The concern was parallel writers colliding on one
   file, and the collision was in `crates/online-polars/src/atomic.rs`. Its
   temporary was named by pid alone, so two threads saving one path in one
   process created and wrote *the same temporary*, and the rename published
   whichever mixture resulted. The temporary is now `.{name}.tmp{pid}-{seq}`,
   with a process-wide counter, so no two writers share one, whether
   threads or processes. The destination is the old file or one writer's
   whole file. This also closes the same hole for `ModelBank.save` called
   from two threads, which existed before the plan could write anything,
   and for the runner's output file (`AtomicFile::create` in `runner.rs`).
   It is held by `two_writers_of_one_destination_do_not_share_a_temporary`:
   50 rounds × 2 threads × 200 kB, every read is one writer's bytes, and it
   fails under the pid-only name. There is no Python-side lock, since the
   plan's writes are idempotent (R2) and now collision-free.
5. **Candidates C / E / F**: none. C is the one to revisit if state as a
   frame ever has a consumer.

### 8. Where it landed

| file | what it holds for this workflow |
|---|---|
| `python/polars_online/_frame.py` | `_source`: R1, R4 and the write. `_bank` / `_read_state`: R3, for `load_state` and `predict(path)`. `_save_path`: the directory, checked at build. `fit_predict` on the lazy, eager and typed forms. `predict`: R5, by having no such keyword |
| `crates/online-polars/src/atomic.rs` | decision 4 |
| `tests/test_frame.py` | C1–C8b as the E35 tests, `predict(save_state=)` refused, the R3 snapshot, the same-path double use |
| `README.md` | "As a plan": the keyword, purity as what makes the write safe, the R6 note and the dated-file recommendation. *Since then* the section is *As a query: `lf.online.fit_predict`*, and the dated-file advice is in the docstring alone |
| `docs/PLAN.md` §11a, `docs/ENHANCEMENTS.md` (E35), `CHANGELOG.md`, `tests/api_surface.txt` | the decision, the enhancement entry, the release note and the public API's snapshot |

### 9. Reproduce

```sh
uv run python scripts/io_source_semantics.py        # §2, the installed polars
uv venv /tmp/v && uv pip install --python /tmp/v/bin/python 'polars==1.34.0' \
  && /tmp/v/bin/python scripts/io_source_semantics.py   # the floor
```
