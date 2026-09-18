# Arrow producers as sources, and what DuckDB already does — a plan

**Status: research and a proposal, 2026-09-17. Nothing built.** The goal is to
let anything that produces Arrow chunks feed a bank, DuckDB first. §1–§3 are
what the ecosystem actually offers, measured or quoted rather than recalled.
§4 is the one decision that is not mine to take. §5 compares DuckDB's own
statistics and learning extensions with this library, which is the question of
whether this expansion is worth building at all.

---

## 1. The finding that reframes it: one tier already works

A capsule producer can reach the bank **today**, with no new code and no new
dependency, by letting polars do the import:

```python
import duckdb, polars as pl, polars_online as po

rel = duckdb.sql("SELECT t, x0, y FROM read_parquet('ticks/*.parquet') ORDER BY t")
bank = po.ModelBank([spec])
for chunk in pl.from_arrow(rel.record_batch(100_000)):   # or pl.DataFrame(rel)
    out = bank.fit_predict(chunk)
```

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

---

## 3. What DuckDB specifically requires, and one wrinkle worth knowing

DuckDB consumes capsules natively: its
`PythonTableArrowArrayStreamFactory` accepts "PyCapsule wrapping
`ArrowArrayStream`" and objects implementing the capsule interface, extracting
the stream pointer directly and checking it has not been consumed. So
**round-tripping a bank's output back into DuckDB is a supported path**, not a
hope.

But the README currently claims output can go "straight to pyarrow or duckdb"
and **no test exercises it** (`grep duckdb tests/` is empty). That claim should
be tested or softened; it is the same class of untested assertion the
2026-09-17 review was written to catch.

**The wrinkle, and it cuts our way.** DuckDB issue
[#17084](https://github.com/duckdb/duckdb/issues/17084) reports that
`__arrow_c_stream__` on a DuckDB relation works **once**; a second call raises
`InvalidInputException: There is no query result`. The reporter observes that
PyArrow, Polars and Pandas all permit repeated calls.

This matters for us because `ArrowStruct.__arrow_c_array__` **deliberately**
exports once and raises on a second call — the spec says capsules "can only be
consumed once", and exporting hands the buffers away. So:

- Against the spec, we are right, and so is DuckDB.
- Against the *installed base*, PyArrow/Polars/Pandas are more permissive, and
  a consumer written against their leniency will break on ours.

**Decision to take, not defer:** keep the single-use contract (it is what the
spec says and what prevents a double free), and document it beside the DuckDB
precedent so it reads as conformance rather than as our quirk.

---

## 4. The one decision that is not mine: a second Arrow implementation

A **native** capsule import needs a consumer obligation we cannot currently
discharge. Quoting the specification:

> "If the capsule has been passed to a consumer, the consumer should have moved
> the data and marked the release callback as null" — so there is not "a risk of
> releasing data the consumer is using".

Nulling `release` is **mandatory**, and the producer's destructor calls release
only if it is not already null. polars-arrow keeps `ArrowArray`'s fields
`pub(super)`, so from outside that module we can move the struct out but cannot
mark the original released. That is the blocker `docs/PLAN.md` task 86 recorded,
and the spec confirms it is real rather than cautious.

Three ways out, with their costs:

**A. Add `arrow-rs` + `pyo3-arrow`.** `pyo3-arrow` does exactly this import and
discharges the obligation correctly. **But `arrow-rs` is not in the tree** —
verified, `cargo tree` finds no `arrow`/`arrow-array`/`arrow-buffer` among 335
crates. Adding it means a **second, complete Arrow implementation** statically
linked beside polars-arrow, and every chunk converting between the two type
systems. **CLAUDE.md rule 12 says I must raise this rather than decide it, and
I am raising it.** My own recommendation is against: the wheel would carry two
Arrow libraries to remove one private call, and §1 shows the speed argument is
not there.

**B. Write the release-nulling by hand against polars-arrow.** `ArrowArray` is
`#[repr(C)]` with the C Data Interface layout, so `release` is at a known
offset and can be zeroed through a raw pointer. Small, no new dependency, and
genuinely fragile: it depends on a layout polars-arrow does not promise, in
`unsafe` code, where the failure mode is a double free rather than a test
failure. Viable only with a test that constructs a capsule, imports it, and
asserts the source is marked released.

**C. Ask polars-arrow to expose it.** The clean fix is upstream: a safe
`ArrowArray::mark_released()` or an import that takes a capsule pointer. This is
a small, well-motivated patch. Note the standing constraints — polars work
lives in `polars-patch/`, nothing is filed without your go, and an issue must be
accepted before an AI-authored PR.

**Proposed sequencing, pending your call on A:** build tier 0 now (it needs no
decision), pursue C in parallel because it is the only option that leaves the
codebase simpler, and hold B as the fallback if C stalls. A stays on the table
but should be a deliberate "yes, two Arrow libraries is the right price", not a
default.

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
| **memory** | O(rows in the group) for the scan | O(state), independent of stream length |
| **decay** | none | every row's weight halves every `halflife` **clock** units, on a column you name |
| **out-of-sample** | `*_fit_predict_agg` splits train/test by a column | by construction: every row predicted *before* its own target is learned |
| **new data** | rescan and refit | feed the chunk; the state moves forward |
| **drift** | refit on a window you choose | `emit_drift`, `drift_action="reset"` |
| **row order** | irrelevant to an aggregate | *is* the model |

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

---

## 6. Proposed work, in order

1. **Tier 0, no decision needed.** A `docs/ARROW-SOURCES.md` user-facing section
   plus tests: DuckDB → bank (with `ORDER BY`), ADBC → bank, PyArrow and Pandas
   as conformance. Test the README's untested "straight to duckdb" claim, in
   both directions. Add `duckdb`, `pyarrow`, `adbc-driver-manager` to the `dev`
   dependency group only — never to the package, which keeps its single
   `polars` dependency.
2. **Document the single-use contract** beside the DuckDB precedent (§3).
3. **Order guidance** for SQL sources (§5's caution).
4. **Decide §4.** My recommendation: pursue C upstream, hold B, decline A.
5. **Only then**, if §4 resolves, the native import: `ModelBank.fit_predict_capsule(obj)`
   taking anything with `__arrow_c_stream__`, and a `Bank::fit_predict_stream`
   under it.

Tier 0 delivers the user-visible capability. Everything after it is about
removing the last private call from the boundary, which is worth doing and is
not worth pretending is urgent.
