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
halflives for a first moment and `2·log2(M)` for a Gram matrix (which holds
`M²`) to wash out — 664 halflives for `x = 1e100`. A weight of `W` freezes
the model for `log2(W)` halflives: the row's own information dominates until
its weight has decayed back to the order of the rest. That is the model, not
a defect.

The defects were elsewhere:

- Once an accumulator holds `inf` (`x² = 1e400`), it never decays
  (`inf·λ = inf`), so the model is dead for the rest of the stream. Kalman
  got there through `sig2`: the residual of the absurd row squares to `inf`,
  `Q ∝ sig2` puts `inf` on the diagonal of `P`, and the gain is
  `inf/inf = NaN`. FTRL got there through `n_i += g²`.
- `rls` and `ew_cov`'s tracked inverse died deterministically on a finite
  row and never recovered — a different mechanism, written up as **C5**.
- `pa` scaled its step by the row weight, so `w = 1e100` was a step of
  `1e100·(loss/‖z‖²)` and the coefficients left the representable range.
  PA-I's step is already the *minimum* change that fits the row; a weight
  only makes sense as a cap on it. The weight is now applied as
  `min(w, 1)`, documented in the spec.
- The robust IRLS weights (`huber`'s `cut/|r|`, `quantile`'s `s/|r|`) are
  `0/0` when the residual and the scale estimate are both 0, which a run of
  exactly-fitted rows produces; guarded.

Fix, two layers:

1. **Bounded input at the boundary.** `online-polars` treats any feature,
   target or weight with `|v| > online_core::INPUT_BOUND` (= `1e100`) as
   missing, exactly like a null: a feature or weight skips the row, a
   target makes it predict-only. Products of two such values (`x²`, `x·y`,
   `w·x²`) then stay below `1e300`, so no accumulator can overflow from the
   input side, and the state stays finite for every row the stream accepts.
   `1e100` is far outside any measured quantity and inside the sentinel
   range that actually shows up in data (`f64::MAX`, `1e300`, `1e308`). The
   constant lives in `online-core` next to the `OnlineModel` trait because
   it is part of the model contract: every model must survive any row inside
   it.
2. **Guards where a bounded input still overflows a derived quantity.**
   Kalman's innovation variance and residual-variance updates and FTRL's
   gradient accumulator skip the row when the increment is non-finite
   (a standardized `z` can still overflow when a feature's scale is tiny).
   Rule 9 in CLAUDE.md already says "an unguarded division poisons the state
   with a NaN that never washes out"; this extends it to overflow.

`crates/online-core/tests/model_contract.rs` holds the property (T4): two
copies of every model, one fed a script that puts the bound in every
position and sign (`x`, `y`, `w`, all at once, then a run at scale `1e-100`
and the bound again), the other only the well-behaved rows; over the last
thousand of a 30,000-row tail every prediction of the first is finite and
agrees with the twin's to `1e-9` (`Recovery::Twin`). `pa` and `quantile`
are held to a different criterion, because they do not converge to one
answer on clean data — PA-I stops updating inside its epsilon tube and the
quantile IRLS weight `s/|r|` never settles closer than its residual floor —
so two histories legitimately end at different points of the band: for them
both copies must be within the band of the target (`Recovery::Tube`). The
measured worst cases after the fixes: every least-squares model agrees with
its twin to `1e-14`, `pa` is within `0.087` of a `0.1` tube, `quantile`
within `1e-4` of a `1e-3` one.

### C5 — covariance-form RLS dies of cancellation; so did `ew_cov`'s inverse — *done*

Found by T4: `rls` failed the bounded-input test with coefficients stuck at
`[0, 0, -4.3e68]` from row 316 to row 30,000, long after the extreme rows
were gone. Two separate mechanisms, both in the textbook recursion
`P ← (P − g zᵀP)/λ`:

1. **Asymmetry amplification.** `g_i (Pz)_j` and `g_j (Pz)_i` differ by an
   ulp; the rank-1 downdate never touches the antisymmetric part, and the
   `1/λ` multiplies it every row. Measured in numpy on well-behaved data:
   the antisymmetric part of `P` grows as `λ^-n`, and is order 1 relative
   to `P` after ~60 halflives. The symmetric downdate (write `P_ij = P_ji`
   from one product) fixes this one.
2. **Cancellation death.** A row whose information exceeds the prior's by
   `1/ulp` in some direction (a feature `1e8×` its usual scale, or a weight
   of `1e100`) computes `P_ii − u_i²/d` as exactly `0` or `±ulp`. An exact
   zero never regrows: the only growth `P` has is multiplicative (`P/λ`),
   and `0/λ = 0`, so the gain in that direction is `0` and the coefficient
   is frozen for good. The probe printed `P` becoming all zeros by row 316.
   In numpy a *single* outlier usually heals — a sub-ulp negative eigenvalue
   gets kicked positive by later rounding — which is why the failure is
   sequence-specific and why it had not been seen before.

The information form has neither problem: an outlier's information decays
multiplicatively and new information is added, so nothing ever cancels.
`ewridge`, which keeps the Gram matrix, passed T4 on the first run. `rls`
is now the **square-root information (QR) form**: the state is the Cholesky
factor `R` of `A = λA + w zzᵀ` and `u = R⁻ᵀb`; a row is folded in by `k`
Givens rotations on the stacked `[R u; √w zᵀ √w y]` and `β` read off by one
back-substitution. Same O(k²) per row (~5k² flops, as before), backward
stable, holds only the square root of the scale (`1e100` inputs with a
`1e100` weight give entries of `1e150`, no overflow), and is still exactly
`ewridge(ridge_decay=True)` solved every row — the agreement test is
unchanged and T4 now passes at `1e-14`. Over the 14,284-row validation
stream (docs/VALIDATION.md) the old form had drifted `1.8e-4` relative from
the exact solve; the new one agrees to `5e-13`.

`ew_cov`'s Sherman-Morrison precision matrix had the same two problems
(`inv_ii → 0` exactly, then `partial_corr = 0/0`). Tracking it is not worth
the fragility: the precision matrix is now solved from the co-moments
(`solve_spd` against the identity, O(k³)) on each row that reads
`partial_corr`, and never otherwise. The prior's semantics
(`M = C + s·prior·I`, `s` decaying with the co-moments) are unchanged, and
the golden outputs match to `1e-12`.

State layout: `SCHEMA_VERSION` is 2. Schema-1 `rls` states (`P`, `β`) are
converted on load (`R` from the reverse-Cholesky of `P`, `u = Rβ`); schema-1
`ew_cov` states load with the stored inverse ignored. `MIN_SCHEMA_VERSION`
records the oldest layout a build still loads, and both `check_schema` and
the bank accept the range.

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

The test is described under C2. It found C5, which is the argument for
having it: a property test over the whole input range, run on every model,
against a twin that saw only clean data.

## 6. Rejected or deferred

- **P2, P3** above — measured, not worth it.
- **Reentrant `ModelBank`.** A lock that serializes concurrent `fit_predict`
  calls would silently reorder chunks; raising is right, only the message
  changes (U2).
- **Adaptive magnitude bound** (per-column scale instead of a constant). The
  constant catches sentinels and overflow, which is the failure that exists;
  everything below it is the model's own arithmetic and the user's choice of
  a robust loss.
