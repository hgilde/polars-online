# Improvements review: testing, performance, usability, extensibility

A pre-release pass over the code with one question per axis: what would a
user hit in the first week, what is slower than it needs to be, what is
harder to extend than it needs to be, and what is untested. Every finding
below was reproduced against the code before it was written down — a probe
script, a measurement, or a failing call — and the decision is recorded next
to it. Items are numbered by axis: **C** correctness, **P** performance,
**U** usability, **X** extensibility, **T** testing. Status per item:
*done*, *proposed*, or *rejected* (with the reason).

Machine for every number: Apple M-series, 10 performance + 4 efficiency
cores, release build, the same setup as docs/PERFORMANCE.md.

## 1. Correctness

### C1 — `emit_drift` / `emit_selected` are unusable through the expression plugin — *done*

`pl.col("y").online.ewridge(..., emit_drift=True)` fails with

```
ComputeError: dtypes don't match, got struct {... drift_y__r0.000001: bool, ...},
expected struct {... drift_y__r0.000001: f64, ...}
```

and `emit_selected=True` with a ridge grid the same way (`selected_y: str`).
The bank is fine. The plugin declares its output dtype from the field *name*
(`coef*` → `List(Float64)`, everything else `Float64`) while the bank
materializes `drift_*` as `Boolean` and `selected_*` as `String`; polars
checks the declaration against the realized column and refuses. Nothing in
`tests/test_expr.py` exercised either flag.

Fix: the dtype is part of the output descriptor. `FieldMeta` (the S1 index
that already carries kind/target/halflife/...) gains a `dtype`, the plugin
reads it, and `assemble` is checked against the same table — one source of
truth instead of a name heuristic. `tests/test_expr.py` runs every emit flag
through `.over()` and compares with the bank (T3).

### C2 — one absurd row can kill a model for good — *done*

Probe: 600 well-behaved rows, halflife 50, and one value at row 200 set to a
magnitude `M`. Predictions were then read 400 rows later.

| model | `x = 1e200` | `y = 1e200` | `x = 1e100` | `y = 1e100` | `x = inf` |
|---|---|---|---|---|---|
| ewridge, rls, lasso | recovers | pred ≈ 5e196 | feature dead (err 1.7) | pred ≈ 5e96 | fine |
| kalman | **null forever** | **null forever** | feature dead | pred ≈ 3e97 | fine |
| ftrl | **null forever** | fine | fine | fine | fine |
| sgd | wrong coef (clipped step) | fine | same | fine | fine |
| huber | feature dead | fine | feature dead | fine | fine |
| pa, quantile | fine | fine | fine | fine | fine |

Two different things are going on. `inf` and NaN are already treated as
missing (that column is all "fine"). A *finite* absurd value is legal input,
and an exponentially weighted least-squares model responds to it exactly as
its equations say: the accumulators absorb `M`, and it takes `log2(M)`
halflives to wash out — 330 halflives for `1e100`. That is the model, not a
defect; `huber`, `pa` and `quantile` are immune because their updates are
bounded, and that is what they are for.

The defect is the first two rows of the table: once an accumulator holds
`inf` (`x² = 1e400`), it never decays (`inf·λ = inf`), so the model is dead
for the rest of the stream. Kalman gets there through `sig2`: the residual
of the absurd row squares to `inf`, `Q ∝ sig2` puts `inf` on the diagonal of
`P`, and the gain is `inf/inf = NaN`. FTRL gets there through `n_i += g²`.

Fix, two layers:

1. **Bounded input at the boundary.** `online-polars` treats any feature,
   target or weight with `|v| > 1e100` as missing, exactly like NaN. Products
   of two such values (`x²`, `x·y`, `w·x²`) then stay below `1e300`, so no
   accumulator can overflow from the input side, and the state stays finite
   for every row the stream accepts. `1e100` is far outside any measured
   quantity and inside the sentinel range that actually shows up in data
   (`f64::MAX`, `1e300`, `1e308`).
2. **Guards where a bounded input still overflows a derived quantity.**
   Kalman's innovation variance and residual-variance updates and FTRL's
   gradient accumulator skip the row when the increment is non-finite
   (a standardized `z` can still overflow when a feature's scale is tiny).
   Rule 9 in CLAUDE.md already says "an unguarded division poisons the state
   with a NaN that never washes out"; this extends it to overflow.

`crates/online-core/tests/model_contract.rs` gets the property: for every
model, after a stream that includes the largest accepted magnitudes and
tiny scales, the state is finite and the model predicts a fresh well-behaved
tail to within tolerance (T4).

### C3 — `fit_predict` is not atomic under `on_clock_reset="error"` — *proposed*

Two groups; group `b` goes backwards at row 80. The error is raised, but by
then group `a` has been fully updated and `b`'s clock has advanced to row 79.
Re-feeding the corrected chunk fails with `goes backwards by 49 at row 0`,
and nothing short of a `load` from an earlier save recovers the bank. The
error mode is documented as "a data error here, and `on_clock_reset='error'`
will say so", but saying so and then leaving the bank half-updated turns a
recoverable data error into a restart.

Fix: validate the clock schedule of the whole chunk on a *clone* of each
stream's `ClockState` before any model is touched; only then run pass 2.
The check is O(rows) and only runs under `"error"`.

### C4 — silent nonsense from a few parameters — *proposed*

Each of these builds a spec, runs, and produces garbage without a word:
`ridge=-1` / `ridge=nan` (solve failures, junk coefficients), `max_dclock=-5`
(`n_eff = 2e30`), `session_gap=-1`, `solve_every=-1`, `solve_every=nan`, and
`halflife=[10, 10]` is caught only at bank construction, not by
`validate_spec`. Each gets a validation line with the spec's name in it.

## 2. Performance

### P1 — `.over(group)` ran the groups one at a time — *done*

docs/PERFORMANCE.md P5 measured the expression plugin at a fraction of the
bank's throughput on grouped data and concluded the gap was polars' own
gather/scatter, out of reach from a plugin. That was wrong. `POLARS_MAX_THREADS=1`
gave the same throughput as 14 threads, and polars-expr's source shows why:
`apply_multiple_group_aware` (a group-aware function with *several* inputs)
walks the groups in a plain `for` loop, while `apply_single_group_aware`
(one input) runs them through rayon.

The plugin now receives every input column packed into **one struct**
(`pl.struct([target, *features, clock, session, weight])`, fields named
positionally so a column in two roles cannot collide), and unpacks it on
the Rust side. Measured, 2M rows:

| case | before | after |
|---|---|---|
| k=5, 1000 groups, `.over` | 4.23M rows/s | **21.4M rows/s** |
| k=20, 100 groups × 4000 rows | 0.94M | **6.4M** |
| k=20, 1000 groups × 400 rows | 0.90M | **6.8M** |
| k=20, 5000 groups × 80 rows | 0.74M | **5.0M** |
| k=5, no group | 4.48M | 4.46M |

At parity with the bank on the same data (5.1–6.4M rows/s), which is what
the P5 target asked for. Every expression == bank test is unchanged.

### P2 — the per-row `Step.pred` allocation — *rejected*

docs/PERFORMANCE.md P1 claims a `pred_buf` catches the `Vec` each `Step`
returns; no such buffer exists — every model allocates `pred` per row. Measured
in a pure-core loop (`EwRidge`, k=5): a `vec![NAN; 1]` alloc+free is 14 ns
against a 95 ns non-solve step (15%), or ~4% at the default solve cadence
(halflife/50, where a solve row is 1.1 µs). Removing it means changing the
`OnlineModel` trait for every model to write into a caller buffer. Not worth
it at 4%; the stale claim is corrected in PERFORMANCE.md and the code comment.

### P3 — group-key extraction casts every key to a string — *rejected*

`group_indices` hashes group values after a `String` cast. Measured on 2M
rows / 1000 groups: int keys 167 ms, string keys 146 ms, categorical 176 ms,
end to end. The cast is not where the time goes; nothing to do.

## 3. Usability

### U1 — features as expressions — *done*

The expression API took feature *names* only, so an AR term meant
materializing `pl.col("y").shift(1).over("g")` as a column first. Features
are now `str | pl.Expr`; an expression's output name is its feature name (so
it must be determinable and unique — `.alias` settles it), and under
`.over(group)` it is evaluated per group, so a lag stays inside its group.
Falls out of P1: the packed struct carries expressions as easily as columns.

### U2 — error messages that do not name the problem — *proposed*

Collected from the probes:

| what happened | message |
|---|---|
| feature column missing | `not found: "nope" not found` |
| `targets="y"` (string, not list) | `invalid spec: invalid type: string "y", expected a sequence at line 1 column 42` |
| `halflife="10"` | `invalid spec: data did not match any variant of untagged enum FloatOrList at line 1 column 334` |
| `quantile=[0.5]` (list, expects float) | `invalid spec: invalid type: sequence, expected f64 at line 1 column 175` |
| `fit_predict(lazy_frame)` | `AttributeError: 'LazyFrame' object has no attribute 'get_columns'` |
| String target column | no error; every prediction null |
| spec named like an input column | no error; the input column is silently replaced |
| two threads calling `fit_predict` | `RuntimeError: Already borrowed` |

Fix: column lookups name the spec, the role and the frame's columns;
non-numeric feature/target/weight/clock columns are rejected with the dtype;
the Python builders check list-vs-scalar shapes before serializing so the
message names the parameter rather than a JSON offset; `fit_predict` says
"collect the LazyFrame first"; a spec name colliding with an input column is
rejected at bank construction; concurrent `fit_predict` says the bank is
already running on another thread.

### U3 — `ModelBank` ergonomics — *proposed*

No `__repr__`; `bank.specs` is `[]` after `load`; no way to list the groups a
bank holds or to drop stale ones, so memory grows with every group ever
seen. Add `__repr__` (spec names, groups, rows seen), keep `specs` after
load, and `groups()` / `drop_groups(keys)`.

### U4 — `**kwargs: Any` on every namespace method — *proposed*

Ten expression-namespace methods take `**kwargs: Any`, so an IDE shows
nothing and a typo is a runtime `TypeError` from a private function
(`_common() got an unexpected keyword argument`). PEP 692 `Unpack[TypedDict]`
gives completion and type checking without changing the call syntax; a test
pins each TypedDict to the builder's actual signature so they cannot drift.

## 4. Extensibility

### X1 — the output dtype lives in one place — *done with C1*

Adding an output kind used to mean a name-prefix rule in the plugin *and* a
branch in `assemble`. With `dtype` on `FieldMeta` both read the same
descriptor.

### X2 — adding a model touches eleven places — *proposed*

Core file, `ModelState`, `AnyModel` plus its `dispatch!`/restore/
coefficients/solve_failures arms, `ModelKind` plus validation, `build_one`,
`combos`, the Python builder, the namespace method, the README table, and
tests. None of it is avoidable — each is a real decision — but nothing lists
them. `docs/EXTENDING.md` is that list, in order, with the test that catches
each omission.

## 5. Testing

### T1 — ten `cargo run` calls cost half the suite — *proposed*

`cargo run -q -p online-cli` takes 2.8 s even when nothing needs building
(cargo's freshness check over the polars dependency graph); the binary itself
starts in 1 ms. Ten CLI tests ≈ 28 s of a 58 s suite. A session-scoped
fixture builds once and runs the executable directly.

### T2 — zero Rust doc tests — *proposed*

`cargo test --doc` compiles nothing. The public core API (`EwRidge`,
`OnlineModel::step`, `Step`) gets examples that are the first thing a Rust
user reads and that break when the API changes.

### T3 — emit-flag matrix through the expression path — *done with C1*

### T4 — bounded-input contract for every model — *done with C2*

## 6. Rejected or deferred

- **P2, P3** above — measured, not worth it.
- **Reentrant `ModelBank`.** A lock that serializes concurrent `fit_predict`
  calls would silently reorder chunks; raising is right, only the message
  changes (U2).
- **Adaptive magnitude bound** (per-column scale instead of a constant). The
  constant catches sentinels and overflow, which is the failure that exists;
  everything below it is the model's own arithmetic and the user's choice of
  a robust loss.
