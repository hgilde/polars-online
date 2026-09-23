# Arrow producers as sources, and what DuckDB already does — a plan

**Status: research and a proposal, 2026-09-17. Nothing built.** The goal is to
let anything that produces Arrow chunks feed a bank, DuckDB first. §1–§3 are
what the ecosystem actually offers, measured or quoted rather than recalled.
§4 is the one decision that is not mine to take. §5 compares DuckDB's own
statistics and learning extensions with this library, which is the question of
whether this expansion is worth building at all. **§7, added 2026-09-21,**
surveys the wider field against two criteria — chunks over more data than
fits in memory, and chunks in clock order — and is reasoned from each
system's documented contract rather than measured.

---

## 1. The finding that reframes it: one tier already works

A capsule producer can reach the bank **today**, with no new code and no new
dependency, by letting polars do the import:

```python
import duckdb, polars as pl, polars_online as po

rel = duckdb.sql("SELECT t, x0, y FROM read_parquet('ticks/*.parquet') ORDER BY t")
bank = po.ModelBank([spec])
# `scan_arrow_c_stream` rides the capsule interface, so it needs no pyarrow --
# `rel.pl()`, `rel.to_arrow_reader()` and `rel.fetch_record_batch()` all do
# (checked on duckdb 1.5.5: each raises ModuleNotFoundError without it). It is
# also the streaming path: the bank chunks the plan itself.
for out in bank.fit_predict_batches(pl.scan_arrow_c_stream(rel), chunk_rows=100_000):
    ...
```

Two corrections to what this block used to say, both measured on duckdb 1.5.5
rather than recalled. `rel.record_batch(...)` is **not** a method — a relation
resolves an unknown attribute as a column name, so it raises
`AttributeError: This relation does not contain a column by the name of
'record_batch'`; the surviving spellings are `fetch_record_batch` and
`to_arrow_reader`, and both need pyarrow. And `pl.from_arrow(rel)` works but
warns: polars 2.0 will return a `Series` rather than a `DataFrame` from an
`ArrowStreamExportable`, so it is the wrong spelling to put in a document that
outlives 1.x.

This works because py-polars consumes the PyCapsule interface: `pl.Series(obj)`
dispatches to `PySeries.from_arrow_c_array` and `polars._utils.pycapsule`
handles `__arrow_c_stream__`. Our own `ArrowStruct` is a *foreign* capsule
producer to polars, and `tests/test_arrow_capsule.py` already proves polars
imports from it — so the mechanism is evidenced here, not assumed.

**What a native import would buy over this.** Only two things, and they should
be stated plainly because they are the whole justification for §4's cost:

1. **Polars leaves the input path.** Task 86 took it off the output; the input
   still crosses on `PyDataFrame`, i.e. pyo3-polars' private `_export`/`_import`.
   A native import finishes that job and removes the reason this package
   carries a polars floor at all.
2. **One conversion instead of two.** Today a DuckDB chunk becomes a polars
   frame and then an `ArrowChunk`. Both hops are buffer-sharing, so the cost is
   metadata and validity handling, not data — and §4 of
   `docs/REVIEW-2026-09-17.md` measured a comparable extra hop at *nothing*
   (eight chunks ran at 0.93× one chunk). **Do not sell this on speed.**

So the honest framing is: tier 0 is a documentation and testing job; tier 1 is
an architecture job whose payoff is the boundary, not the clock.

---

## 2. What to expand to

Producers implementing the Arrow PyCapsule interface, which is the only
integration surface worth targeting — it is a specification, not a library, so
supporting it supports everything below at once.

| producer | how it exposes Arrow | tier 0 today | notes |
|---|---|---|---|
| **DuckDB** | `DuckDBPyRelation.__arrow_c_stream__`; also consumes capsules | yes | the driving case; see §3 for a real wrinkle |
| **PyArrow** | `Table`, `RecordBatchReader` | yes | the reference implementation |
| **Pandas** ≥2.2 | export via the interface | yes | `ArrowDtype` frames are zero-copy; object columns are not |
| **Polars** | native | yes | the existing path |
| **ibis** | capsule export | yes | a front end to many engines; supporting it is indirect breadth |
| **Daft** | capsule export | yes | distributed; a bank would sit per-partition |
| **Lance / LanceDB** | capsule export | yes | columnar store for ML data |
| **fastexcel** | capsule export | yes | small, but proves the "anything" claim |
| **ADBC** (Postgres, Snowflake, BigQuery, Flight SQL) | `ArrowArrayStreamHandle` implements the interface; `adbc_ingest` accepts `__arrow_c_array__`/`__arrow_c_stream__` | yes | **the highest-value entry after DuckDB**: it is one driver manager in front of most warehouses |
| **Arrow Flight SQL** | via the ADBC Flight SQL driver | yes | the network case; a bank fed from a Flight stream is the "fit where the data is" story |

Two deliberate non-targets. **cuDF** would need device capsules
(`arrow_device_array`), a different contract and a GPU story this library does
not have. **Spark** exports through Arrow but its natural unit is a partition,
which is `applyInArrow`'s problem rather than a source's.

**Ordering.** DuckDB, then ADBC, then PyArrow/Pandas as a conformance pair. The
first two are where the data actually lives; the second two are the cheapest
way to keep the contract honest, because PyArrow is the reference and Pandas is
the one most likely to hand us something surprising.

### ADBC, measured

Measured 2026-09-23 with `adbc_driver_manager` 1.12.0 and the SQLite driver,
**no pyarrow installed**, against the released `polars-online` 0.9.1's real
output rather than a stand-in:

| direction | call | without pyarrow |
|---|---|---|
| write | `cur.adbc_ingest(name, pl.DataFrame)` | works |
| write | `cur.adbc_ingest(name, ArrowStruct)` | refused on SQLite: list columns |
| write | the same, `coef` and `support_coef` dropped | works |
| read | `pl.DataFrame(cur.fetch_arrow())` | works |
| read | `cur.fetch_polars()` | works, but warns; see below |
| read | `fetchall()`, `fetch_arrow_table()`, `fetch_record_batch()` | "This API requires PyArrow to be installed" |

So the capsule protocol itself needs no pyarrow on either side of ADBC. Even
DB-API `fetchall` needs pyarrow in this driver manager, so reads go through
`fetch_arrow()`.

**List columns are the driver's call, not the protocol's.** SQLite's driver
refuses our output with `NOT_IMPLEMENTED: Column 5 has unsupported type
large_list`. Column 5 is `coef`, a `List(Float64)`, and `support_coef` is the
same. With both dropped, ingest works. SQLite has no list type; Postgres has
array types and may accept them, untested. So the portable route out through
ADBC is a flat output, which those two fields are not.

**Avoid `fetch_polars()`.** It works today, but it is
`polars.from_arrow(self.fetch_arrow())`, and polars already warns that
`from_arrow` on a stream-exportable will return a `Series` rather than a
`DataFrame` in 2.0. `pl.DataFrame(cur.fetch_arrow())` is the spelling that
survives.

**Corrected, and recorded because I said it first.** An earlier probe
suggested ADBC ingest takes our output directly, where DuckDB refuses it. That
probe used a stand-in with two flat float fields and ran with pyarrow
installed. Lacking the list columns, it could not see the refusal; the real
output can. On output, then, DuckDB needs a hop through a struct `Series` and
ADBC needs flat columns, and neither is strictly ahead. On input, both work
without pyarrow.

---

## 3. What DuckDB specifically requires, and one wrinkle worth knowing

DuckDB consumes capsules natively: its
`PythonTableArrowArrayStreamFactory` accepts "PyCapsule wrapping
`ArrowArrayStream`" and objects implementing the capsule interface, extracting
the stream pointer directly and checking it has not been consumed. So
**round-tripping a bank's output back into DuckDB is a supported path**, not a
hope.

The README claimed output could go "straight to pyarrow or duckdb" and **no
test exercised it**. Now measured (2026-09-22), and **the duckdb half was
false**: on duckdb 1.5.5 an `ArrowStruct` is refused by `from_arrow`
("not an accepted Arrow Object"), by `register`, and by a replacement scan
("not suitable for replacement scans"). DuckDB consumes the **stream**
interface, `__arrow_c_stream__`, which `pl.Series` has and our struct does not,
so the struct is rejected on type before its data is looked at.

`duckdb.from_arrow(pl.Series(s))` works, and it is worth knowing *why*, because
the dunder alone is not sufficient. DuckDB wants something **table-shaped**. A
spec's output is a *struct* Series, so it presents as one column per field and
arrives as a 7-column relation. A *flat* `Float64` Series carries the same
dunder and is still refused, with "Provided table/dataframe must have at least
one column" (measured 2026-09-22). So the route to DuckDB is through a struct
`Series`, and the README now says that instead. The pyarrow half stays **unverified here**, because pyarrow is
deliberately not a dependency of this project and must not become one to test
a sentence. `tests/test_arrow_capsule.py` pins all of this.

**A wrinkle worth knowing — and it no longer cuts our way.** DuckDB issue
[#17084](https://github.com/duckdb/duckdb/issues/17084) reported that
`__arrow_c_stream__` on a DuckDB relation works **once**, a second call raising
`InvalidInputException: There is no query result`, and an earlier draft of this
section cited that as precedent for our own single-use export. **Measured on
duckdb 1.5.5, it does not reproduce: both calls succeed.** The issue has been
fixed since it was filed, so the precedent is gone, and citing it was reading a
bug report as current behaviour. `tests/test_consumed_source.py` now pins the
live behaviour, so if it changes again this section hears about it.

What remains true is the more useful fact, and it sits one level down: the
*stream* a consumer captures is single-use even though the relation is not.
`pl.scan_arrow_c_stream(rel)` collected twice gives its rows and then **zero**,
silently, with no error from polars — measured, and now reported by
`ConsumedSourceWarning`.

### The dependency-free surface, and a trap when pyarrow is absent

Measured on duckdb 1.5.5 with **no pyarrow installed**, each call on its own
fresh relation:

| call | without pyarrow |
|---|---|
| `rel.__arrow_c_stream__()` | works |
| `pl.scan_arrow_c_stream(rel)` | works |
| `pl.DataFrame(rel)` | works |
| `rel.fetchall()`, `rel.df()` | works |
| `rel.pl()`, `rel.arrow()` | `ModuleNotFoundError` |
| `rel.to_arrow_table()`, `rel.to_arrow_reader()` | `ModuleNotFoundError` |
| `rel.fetch_record_batch()` | `ModuleNotFoundError` |

So **the capsule path is the only Arrow route out of DuckDB that carries no
dependency**, which is why §1's recipe uses it and why this project can speak to
DuckDB while refusing to depend on pyarrow. Note also that `fetch_arrow_table`,
`fetch_arrow_reader` and `fetch_record_batch` are all **deprecated** in favour of
`to_arrow_table` / `to_arrow_reader` — and every one of them routes through
pyarrow anyway, so the rename changes nothing for us.

**The trap: a call that fails for want of pyarrow leaves that relation broken.**
A successful Arrow read does *not* spend a relation — `fetchall()` works twice,
`__arrow_c_stream__()` works twice, and `fetchall()` still works after
`__arrow_c_stream__`, after `pl.scan_arrow_c_stream`, and after `pl.DataFrame`.
But once `rel.to_arrow_table()` or `rel.pl()` has failed on the missing pyarrow,
**that same relation is finished**:

- `fetchall()` raises `NotImplementedException: Can't 'FetchRaw' from ArrowQueryResult`
- `df()` raises `InternalException: Failed to cast query result`
- `__arrow_c_stream__()`, which would have worked a moment earlier, now raises
  the pyarrow `ModuleNotFoundError` too

The damage is **per relation, not per connection**: a new relation off the same
connection is fine. Every one of those follow-on errors names the wrong cause,
so this is easy to misread as a polars or capsule fault. **Build a new relation
after any such failure rather than reusing one.**

Recorded because I got it wrong first: an earlier probe reused a single relation
across every call and I concluded from it that "a relation's Arrow calls consume
its result". The isolating experiment refuted that — successful calls are
harmless, failed ones are destructive. Not pinned by a test, deliberately: the
behaviour only appears when pyarrow is *absent*, so a test for it would assert a
fact about the environment rather than about this package.

So on the export side we stand with the specification and against the installed
base, without DuckDB beside us any more:

- The spec is unambiguous: capsules "can only be consumed once", and exporting
  hands the buffers away. `ArrowStruct.__arrow_c_array__` **deliberately**
  raises on a second call.
- PyArrow, Polars, Pandas — and now DuckDB — all permit repeated calls, so a
  consumer written against their leniency will break on ours.

**Decision to take, not defer:** keep the single-use contract (it is what the
spec says, and what prevents a double free), but document it as *ours* rather
than as conformance with a neighbour, because the permissive behaviour is now
what a user will have met everywhere else.

---

## 4. The import blocker that was not one

**Corrected 2026-09-22, and the decision this section used to ask for is
withdrawn.** An earlier draft said a native capsule *import* needed a consumer
obligation we could not discharge, and put three costly ways out to you. The
premise was wrong. There is nothing to decide, nothing to add, and nothing to
ask upstream.

The obligation itself is real. Quoting the specification:

> "If the capsule has been passed to a consumer, the consumer should have moved
> the data and marked the release callback as null" — so there is not "a risk of
> releasing data the consumer is using".

Nulling `release` is **mandatory**, and the producer's destructor calls release
only if it is not already null. What was wrong was the claim that we cannot do
it. The old reasoning ran: polars-arrow keeps `ArrowArray`'s fields
`pub(super)`, so from outside that module we can move the struct out but cannot
mark the original released. **Marking it released does not need field access.**
`ArrowArray::empty()` is `pub`, is documented "creates an empty `ArrowArray`,
which can be used to import data into", and builds the struct with
`release: None` and every pointer null. So one public call both moves the
producer's struct out and leaves a released struct behind:

```rust
let owned = std::ptr::replace(ptr, ArrowArray::empty());
let array = import_array_from_c(owned, dtype)?;
```

`ArrowSchema::empty()` is public too, and `import_array_from_c` and
`import_field_from_c` are public and take exactly those forms.

**polars itself does this**, which is the reference to copy rather than invent:
`crates/polars-python/src/series/import.rs`, `import_array_pycapsules`, is the
two lines above with `capsule.pointer_checked(Some(c"arrow_array"))` supplying
the pointer. Note it does *not* replace the schema: `import_field_from_c` takes
it by reference and copies into a `Field`, so the schema capsule's own
destructor releases it. The stream-shaped alternative is `open_stream_capsule`,
which is the same idiom with `ArrowArrayStream::empty()` and
`ArrowArrayStreamReader::try_new`.

**Verified by a compiling test, not by reading.** `crates/online-polars/tests/
arrow_capsule_import.rs` exports an array to C, performs the replace-and-import
round trip from *outside* polars-arrow, asserts the values survive and that the
source's `release` is now null, and drops the emptied original with no double
free. It runs against the pinned `polars-arrow =0.55.2` that the wheel actually
ships, so this is a property of our build and not only of polars' main branch.

What this dissolves, so none of it is reopened by accident:

- **A second Arrow implementation is not needed.** Adding `arrow-rs` +
  `pyo3-arrow` would have put a complete second Arrow library in the wheel to
  reach one call that turns out to be public. The CLAUDE.md rule 12 question
  this section used to raise is **withdrawn**, not pending.
- **No raw-pointer layout hack.** The old fallback zeroed `release` through a
  raw pointer at a guessed offset, in `unsafe`, where the failure mode is a
  double free. Unnecessary.
- **No upstream ask.** There is no second polars patch to file. A duplicate
  search, control-validated first, found nothing; and pola-rs/polars#28190 is
  `accepted` but `P-low`, with a maintainer noting this FFI surface is "not
  meant to be publicly used" — so an ergonomic `mark_released` request would
  likely not have landed anyway.

**The residual risk, which is real and small.** These items are public but carry
no stability promise, exactly like the three interfaces in CLAUDE.md rule 13, so
this is a fourth entry on that list rather than a departure from it. It adds no
dependency: `crates/online-polars` already depends on polars-arrow directly and
re-exports these `ffi` items. The test above is the canary — if a future
polars-arrow drops `empty()`, it fails loudly at build time rather than silently
at a double free.

---

## 5. DuckDB's statistics and learning extensions, against this library

This is the part that decides whether the expansion is worth building, because
if DuckDB already does the modelling, the integration is pointless.

### What DuckDB has

**Built in.** The SQL-standard regression aggregates: `regr_slope`,
`regr_intercept`, `regr_r2`, `regr_count`, `regr_avgx`, `regr_avgy`, `regr_sxx`,
`regr_sxy`, `regr_syy`, plus `corr` and `covar_pop`. `regr_slope` *is*
`covar_pop(x,y)/var_pop(x)`.

**Community extensions** (200+ in total; the modelling ones):

| extension | what it provides |
|---|---|
| `anofox_statistics` | OLS, Ridge, Elastic Net, LARS/LassoLars, WLS, robust (Huber, RANSAC, Theil-Sen), GLMs (Poisson, NegBin, Binomial, Tweedie, Gamma, Logistic), mixed-effects, AFT survival, isotonic, quantile, PLS; t-tests, ANOVA, Kruskal-Wallis, Mann-Whitney, Wilcoxon, chi-squared, Fisher, correlation and normality tests; AIC/BIC, VIF, residual diagnostics |
| `anofox_forecast` | ARIMA, ETS, Theta, TBATS, MFLES, MSTL, Croston, GARCH |
| `anofox_tabfm` | zero-shot tabular foundation models |
| `mlpack` | five supervised/unsupervised methods |
| `ai`, `flock` | LLM/RAG functions |
| `finance`, `behavioral` | quant finance; sessionisation, funnels, retention |

This is a **serious** statistics surface — broader than this library's model
list in several directions (hypothesis tests, GLM families, survival,
forecasting). Any comparison that pretends otherwise is not worth writing.

### Where the two genuinely differ

The difference is not "which has more models". It is **what a fit is**.

| | DuckDB + `anofox_statistics` | polars-online |
|---|---|---|
| **fit shape** | aggregate over a group; recomputed per query | one state per (spec, group), updated per row |
| **state between queries** | none — "models are computed per query; no persistent state" | the state *is* the product; `save`/`load`, resume mid-stream |
| **memory** | O(rows in the group) for the scan | O(state) in the bank; the *pipeline* is only as bounded as its source — see below |
| **decay** | none | every row's weight halves every `halflife` **clock** units, on a column you name |
| **out-of-sample** | `*_fit_predict_agg` splits train/test by a column | by construction: every row predicted *before* its own target is learned |
| **new data** | rescan and refit | feed the chunk; the state moves forward |
| **drift** | refit on a window you choose | `emit_drift`, `drift_action="reset"` |
| **row order** | irrelevant to an aggregate | *is* the model |

**On that memory row, measured rather than assumed.** The bank's own state is
O(state) and that is not in question. What is *not* earned is the end-to-end
claim: draining a DuckDB relation **with no bank attached at all** cost 106 MB
at 1M rows and 271 MB at 4M, so the growth is the producer's side, not ours.
Nor is it the chunk size — a sweep held 410 MB at both 10k and 50k rows per
chunk and 588 MB at 1M, which is not O(chunk) at the low end, and page-release
environment variables moved none of it. So "a bank over DuckDB runs in bounded
memory" is a claim this document must not make until the source side is
measured properly; what it may say is that the bank contributes O(state) to
whatever the source costs.

DuckDB is batch-first by design; its incremental support is
[described as experimental rather than production-grade streaming](https://medium.com/@bhagyarana80/streaming-analytics-with-duckdb-incremental-updates-redefining-olap-50a4238ec3cf).
An aggregate is the right tool when the answer is "fit this group"; it is the
wrong tool when the answer is "what did the model believe at each row, given
only what it had seen".

### The honest conclusion

**They are complements, and the integration is the point of contact.** The
natural division:

- **DuckDB does the data work** it is best at: scanning, joining, filtering,
  windowing — and, for a *static* fit over a finished group, `anofox_statistics`
  is likely better than reaching for this library at all.
- **polars-online does the sequential fit**: a rolling or decayed model over an
  ordered stream, with the state as a durable artifact and out-of-sample
  discipline enforced rather than arranged.

Which is exactly why the integration is worth building and why it should be
*narrow*: not "a DuckDB ML extension", but "a DuckDB relation is a source a
bank can read", so the two sit in one pipeline.

**One caution that belongs in the docs before any of this ships.** A DuckDB
relation's row order is whatever the query produced, and an online model learns
in row order. This is exactly the hazard `OrderNotGuaranteedWarning` was added
for on the polars side, and the equivalent SQL advice is: put an explicit
`ORDER BY` on the relation feeding a bank. The warning cannot see a DuckDB plan,
so here documentation is the only guard — which is an argument for tier 0 being
a *documentation* job first and a code job second.

**A second caution, and this one is now guarded in code.** A relation is
consumed once (§3), and so is the `LazyFrame` built over it: collected a
second time it yields **nothing**, silently, with no error from polars. A bank
fed the same plan twice therefore learns from every row and then from none,
and the second run leaves a state that looks finished and is empty — measured
at 1,000 rows, then 0. `ModelBank.fit`, `ModelBank.fit_predict_batches` and
`lf.online.fit_predict` now raise `ConsumedSourceWarning` when a plan whose
source is a Python scan delivers no rows at all. The pair is the
discriminator, not either half: re-collecting a `scan_parquet` is legitimate,
and this package's own plan form carries the same `PYTHON SCAN` marker while
being reusable, so neither trips it. The fix in a DuckDB pipeline is to build
the relation *inside* the loop, not outside it.

---

## 6. Proposed work, in order

1. **Tier 0, no decision needed.** A `docs/ARROW-SOURCES.md` user-facing section
   plus tests: DuckDB → bank (with `ORDER BY`), ADBC → bank, PyArrow and Pandas
   as conformance. Test the README's untested "straight to duckdb" claim, in
   both directions. Add `duckdb`, `pyarrow`, `adbc-driver-manager` to the `dev`
   dependency group only — never to the package, which keeps its single
   `polars` dependency.
   **Started 2026-09-17:** `duckdb` is in the `dev` group, and
   `tests/test_consumed_source.py` covers DuckDB → bank over an ordered
   relation, the pyarrow-free path, and what `#17084` actually does on the
   installed version. Still to do: `pyarrow` and `adbc-driver-manager`, the
   README's untested "straight to duckdb" claim, and the user-facing section.
2. ~~**Document the single-use contract** beside the DuckDB precedent (§3).~~
   **Done — and the premise changed.** There is no DuckDB precedent any more
   (§3, measured on 1.5.5), so the contract is documented as *ours*. The
   related hazard one level down — a captured *stream* being spent while the
   relation is not — is documented and now guarded by `ConsumedSourceWarning`.
3. **Order guidance** for SQL sources (§5's caution).
4. **Decide §4.** My recommendation: pursue C upstream, hold B, decline A.
5. **Only then**, if §4 resolves, the native import: `ModelBank.fit_predict_capsule(obj)`
   taking anything with `__arrow_c_stream__`, and a `Bank::fit_predict_stream`
   under it.

Tier 0 delivers the user-visible capability. Everything after it is about
removing the last private call from the boundary, which is worth doing and is
not worth pretending is urgent.

---

## 7. The wider field: which producers stream, and which can promise order

**Added 2026-09-21, and reasoned rather than measured.** §1–§3 earn their
claims against a version (`duckdb 1.5.5`); this section does not. It is read
off each system's own documented contract, and every row that becomes work
needs that same treatment before it is believed.

Two criteria, both the user's: a producer must hand over **Arrow chunks over
more data than fits in memory**, and — for any pipeline that uses a clock —
hand them over **in clock order**.

**Order gates features, not the integration** (the user's correction,
2026-09-21; an earlier draft of this section had it stopping the run
outright). An unordered producer is usable. What it costs is every feature
denominated in a clock. Three tiers, in the order to consider them:

1. **No `clock` column at all — nothing can refuse.** The row count is the
   clock: `ClockState::advance` returns `Some(1.0)` for every row after the
   first, so Δ is never negative and the disorder branch is unreachable. All
   three refusals are clock-gated — `on_clock_reset needs clock`,
   `min_backwards_jump needs clock`, and `max_dclock is required when clock is
   given`. What is given up is what a clock buys: the gap ceiling, sessions and
   `session_gap`, `window`, `label_delay`, and a `halflife` that means elapsed
   time rather than rows. Decay still works, per row.

2. **`fit()` over an order-free spec — the state really is
   order-independent.** `ew_ridge`, `rls`, `huber` or `lasso` with no decay
   (`lam = 1.0`, no `halflife`) and every path-changing key at its neutral
   value: no window, no session, no `label_delay`, no Gram blocking,
   `drift_action = "flag"`, the diagnostics off (`_ORDER_FREE_ONLY_WHEN` in
   `python/polars_online/_frame.py` is the full table). Then the sums commute
   and the fit reaches the same state whatever order the rows arrived in —
   measured 3.3e-16 over 200 rows. That table is conservative by construction
   ("unknown means no"), so several entries are denied as *unproven* rather
   than refuted — `weight` among them — while the refuted ones carry their
   cost: `window` 8.3e-03, `gram_block_rows` 6.3e-04, `label_delay` 4.3e-04,
   and `drift_action = "reset"` **8.9e-01**, the largest, and the one that read
   as harmless until a fixture made drift actually fire.

3. **Everything else — and every prediction.** Order-freeness is about
   `fit()`, whose product is the state. It is never about a prediction: `pred`
   is out-of-sample by construction, so row *i* is predicted from the rows
   before it and reordering moves every one of them — **1.33** on the same
   no-decay `ewridge` whose coefficients agreed to 3.3e-16. And *with* a clock
   column, out-of-order rows are refused rather than absorbed (0.9.0), so there
   the question does become whether the run completes.

So the filter to apply to a producer below is not "can it promise order?" but
"which tier is this pipeline in?". A producer that cannot promise one is still
a source for tiers 1 and 2; only tier 3 needs the promise. The polars side has
the same hazard and only a warning (`OrderNotGuaranteedWarning`), because a
`LazyFrame` cannot be asked for a guarantee; a SQL source *can* be, with
`ORDER BY`.

### Query engines that can declare an ordering

| producer | bounded memory | clock order | how it would connect |
|---|---|---|---|
| **DataFusion** | yes — streams `RecordBatch`es, spills large sorts | **yes, as a plan property**: output ordering is tracked through operators, and partitioned streams merge order-preserving | Rust, in process (the CLI is already a Rust binary), or its Python bindings as a capsule producer |
| **DuckDB** | yes (§1) | yes, with an explicit `ORDER BY` (§5's caution) | already tier 0 |
| **ClickHouse** | yes | yes — a table is stored physically in its `ORDER BY` key, so a time-keyed table scans in time order | Arrow output, over ADBC or HTTP |

DataFusion is the one worth singling out: it is the only engine here whose
ordering is a first-class property of the plan rather than something the query
author must assert and the reader must trust.

### Time-series stores, where clock order *is* the storage model

| producer | bounded memory | clock order | how it would connect |
|---|---|---|---|
| **InfluxDB 3** | yes | yes — time is the organising dimension, and it is itself DataFusion + Arrow | Flight SQL, so: ADBC |
| **QuestDB** | yes | yes — a designated timestamp, rows stored in it | ADBC / Postgres wire |
| **TimescaleDB** | yes | yes with `ORDER BY` on the time column; hypertable chunking makes that scan cheap | ADBC Postgres driver |
| **Lance / LanceDB** | yes | an ordered scan over a table (§2) | capsule export |

This group is why **ADBC's rank in §2 understates it**: one driver manager is
the door to three of those four. If exactly one thing after DuckDB gets built,
the evidence points at ADBC rather than at PyArrow.

### Lakes: order is a property of the layout, not of the format

Parquet, Iceberg and Delta expose Arrow readers and stream happily, but none
of them *creates* an order. Parquet records `sorting_columns`; Iceberg
declares a sort order in table metadata — both describe what the writer did.
If the data was written sorted by the clock a scan can preserve it, and if it
was not, no reader can conjure it — and a clock-using spec will then refuse the
result, while a clock-free one will accept it and mean something different by
it. For a lake the integration note is therefore a **precondition on the
writer**, not a capability of the source.

### Stream processors with per-key state

A different shape: not "a source the bank reads" but "a host the bank runs
inside", one model per key.

| host | fit | the ordering it actually guarantees |
|---|---|---|
| **Arroyo** | closest of the four — Rust, Arrow-based, event time with watermarks, unbounded by design | event time, per key |
| **Bytewax** | Python dataflow on a Rust core; keyed stateful operators with snapshot recovery | per key within a partition |
| **Quix Streams** | Kafka-native, per-key state | per *partition* — the real contract, and it maps onto one bank per group |
| **Spark `transformWithState` / `applyInArrow`** | per-group state exists | **not ordered within a group without an explicit sort**, and its unit is a partition, which is §2's standing objection to Spark |

`arrow-udf` is a mechanism rather than a target: it is how a Rust function over
Arrow gets embedded in an engine that wants one.

### Rejected on semantics, not on plumbing

**Materialize and Feldera (DBSP).** Both maintain incremental views, and both
deliver change as *retraction*: an update arrives as a negative multiplicity
cancelling an earlier row. An online model cannot un-learn a row. Its only
forgetting is decay, which is time-directed and applies to the whole state at
once rather than being addressed to one row — and `support_coef`, `n_eff` and
every co-moment would all have to be walked back for a retraction to mean
anything. So this is not an awkward fit to be worked around: it is a different
computational contract, and the mismatch is in the model, not the transport.
**cuDF** stays out for §2's reason — device capsules are a different contract
again.

### What this changes about §6

Three things, none of which reorders the list:

1. **ADBC's case is stronger than §2 states** — it fronts not only the
   warehouses but QuestDB, TimescaleDB and, through Flight SQL, InfluxDB 3:
   the systems whose storage model already *is* clock order.
2. **The order guidance (§6.3) should be split in two.** For a spec with a
   clock it is a hard precondition — since 0.9.0 the bank refuses out-of-order
   rows rather than absorbing them, so "sorted by the clock within each group"
   is what makes the source work at all. For a spec without one it is a
   statement about meaning rather than about completion: the run succeeds
   either way, and what arrival order decides is what "recent" weighs and what
   every out-of-sample prediction was scored against. Both belong beside the
   `ORDER BY` note §5 already carries.
3. **DataFusion deserves a spike, and it is not blocked on §4.** It would let
   the Rust CLI read any DataFusion source in a declared order with no Python
   in the path. It does bring `arrow-rs` — but into `online-cli`, a *separate
   binary* that already links Rust `polars` and never touches py-polars, not
   into the wheel. §4 and rule 12 are about two Arrow implementations inside
   the published wheel; a CLI-only dependency does not incur that, which makes
   this the cheapest way to find out what living with arrow-rs is actually
   like before answering §4 at all.
