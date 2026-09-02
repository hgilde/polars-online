# Improvements review: testing, performance, usability, extensibility

A pre-release pass over the code with one question per axis: what would a
user hit in the first week, what is slower than it needs to be, what is
harder to extend than it needs to be, and what is untested. Every finding
below was reproduced against the code before it was written down — a probe
script, a measurement, or a failing call — and the decision is recorded next
to it. Items are numbered by axis: **C** correctness, **P** performance,
**U** usability, **X** extensibility, **T** testing. Status per item:
*done*, *proposed*, or *rejected* (with the reason). C6 onward are a second
pass, made after the first one was closed.

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

### C3 — `fit_predict` is not atomic under `on_clock_reset="error"` — *done*

Two groups; group `b` goes backwards at row 80. The error is raised, but by
then group `a` has been fully updated and `b`'s clock has advanced to row 79.
Re-feeding the corrected chunk fails with `goes backwards by 49 at row 0`,
and nothing short of a `load` from an earlier save recovers the bank. The
error mode is documented as "a data error here, and `on_clock_reset='error'`
will say so", but saying so and then leaving the bank half-updated turns a
recoverable data error into a restart.

Done: `Stream::check_clock` walks the chunk's clock and session columns on a
*clone* of the stream's `ClockState` and reports the first backwards row
exactly as `process_chunk` would; the bank runs it over every (spec × group)
of the chunk before it runs `process_chunk` on any, so a refusal leaves the
bank byte-identical (`a_refused_chunk_updates_nothing` compares
`save_bytes()` before and after, then feeds the corrected chunk and matches a
bank that never saw the bad one). `process_chunk`'s own pass 1 now runs on a
copy too, so a stream is atomic on its own as well. The check is O(rows) over
two columns and is skipped outright unless the policy is `"error"` and there
is a clock column — a row-count clock cannot go backwards. The message says
"the bank was not updated".

### C4 — silent nonsense from a few parameters — *done*

Each of these built a spec, ran, and produced garbage without a word:
`ridge=-1` (pred −1.2 on a target of 1), `ridge=nan` / `ridge=inf` (every
coefficient 0), `max_dclock=-5` (`n_eff = 5.7e7` on 50 rows — every delta
clipped to −5, so the "decay" grows), `session_gap=-1` (clamped to 0),
`solve_every=-1` (solved every row), `halflife=nan` (`n_eff` NaN for good),
and `halflife=[10, 10]` was caught only at bank construction, by the
field-name tripwire, not by the builder.

Done, all at `Spec::validate` so the builder, `validate_spec`, the bank and
the CLI agree: `ridge` finite and ≥ 0 (zero is plain least squares; `rls`
keeps > 0), grid values (`ridge`, `halflife`) unique, `max_dclock` ≥ 0 (zero
is the documented "no decay", `inf` now expressible from Python — the field
is a `Num`), `session_gap` ≥ 0, `solve_every` finite and ≥ 0, `lasso_path`
finite, ≥ 0 and *strictly* decreasing, `select_halflife`/`long_halflife`/
`cd_tol` > 0, Kalman `q` ≥ 0 and `obs_var`/`p0` finite; every
`x <= 0` test became `!(x > 0)` so NaN fails it too. The Python encoder
refuses any NaN by parameter name (`spec "m": lam must not be NaN`) instead
of letting serde report a JSON column offset. `tests/test_spec_validation.py`
is the table: every refused value, its message, and the legal neighbour that
still runs.

### C6 — an interrupted save destroys the last good state — *done*

Second pass, after everything above was closed. `Bank::save` was
`std::fs::write`, which truncates the destination and then writes into it, so
an interrupted save loses the file it was updating. Reproduced with
`RLIMIT_FSIZE` set to a third of the state size (a full disk or a quota,
deterministically): the save raises `File too large`, the state file goes from
221,612 bytes to 73,870, and loading it fails with "failed to fill whole
buffer". Both errors are clean — the C2 hardening holds — but the state is
gone, and for the `--resume` loop the CLI documents, that is the stream
starting over. A crashed process is the *reason* the file exists.

Fixed in `crates/online-polars/src/atomic.rs`: write a temporary sibling,
`sync_all`, rename over the destination. Rust's `fs::rename` is documented to
replace an existing destination on both target platforms (`rename` on Unix,
`MoveFileExW` on Windows), and a rename either happens or does not, so a
reader sees the whole old file or the whole new one. The temporary is a
sibling because a rename cannot cross a filesystem, and carries the pid so two
processes saving to one path do not share it; `Drop` removes it on any error
path. A symlink destination is resolved first, because `fs::write` followed it
and a rename would replace it — atomicity is the only thing that changed.

The sync is not free, and on macOS it is the expensive one: std's `sync_all`
is `fcntl(F_FULLFSYNC)` — so is `sync_data`, so there is no cheaper honest
option — which flushes the drive's own cache. Measured on 396 KiB: write 0.10
ms, `+fsync` 0.13 ms, `+F_FULLFSYNC` 4.02 ms, so a 500-group save goes from
0.5 ms to 5.0 ms. Kept anyway, and documented at the call site: a state file
that is not there after a power loss is not a state file, and a caller saving
after every small chunk can save less often.

The CLI's output parquet got the same treatment, since it is the same
mistake: it was written straight to `--output`, so a run that died on chunk
eight replaced yesterday's output with seven chunks and no footer. Now the
run publishes with a rename after the footer is written, and a failed run
leaves the previous output byte-identical (`test_a_failed_run_leaves_the
_previous_output_intact`).

Tests: two unit tests in `atomic.rs` (a failed write leaves the destination
alone and litters nothing; a symlink is written through, not replaced), the
runner test above, and `test_an_interrupted_save_keeps_the_last_good_state`,
which is the RLIMIT_FSIZE reproduction in a subprocess. Each was mutated back
to `fs::write` / `File::create` once, and each failed.

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

### P4 — the published throughput table was two changes stale — *done*

Not a code finding: a documentation one, found by re-running the command the
README cites. Every `ewridge` case was 14–62% *faster* than the table claimed
(the allocator fix, PERFORMANCE §6, landed after the table was written), and
`rls` was **48% slower** — 3.13M rows/s published, 1.63M measured.

`rls` is C5's bill, and it is now attributed rather than assumed:
`crates/online-core/examples/rls_bench.rs` times `Rls::step` alone and
compiles unchanged against `50c1a38^`, the commit before the square-root
rewrite, so the A/B isolates the model arithmetic — 4.39M vs 1.70M rows/s at
k=20, 0.39×, and 0.52×/0.49× at k=5/k=50. Both forms are O(k²); the QR form
does `k` Givens rotations with a square root each, and in exchange `rls` no
longer dies of cancellation on one extreme row. The right trade, wrongly
advertised.

README table and prose refreshed, PERFORMANCE §8 records the comparison, and
`scripts/scaling_bench.py` now runs up to every core the machine has — it
stopped at 8, which on a 14-core box hides the row the script exists to show
(1→14 threads is 6.6×).

Then the bill was itemized (PERFORMANCE §8, 2026-09-02): half of the QR
form's row was the back-substitution, not the rotations, because its sum ran
in the one order that serialized every row on the coefficient just solved.
Summing from the far end — one `.rev()`, a rounding-level change inside the
golden tolerance — took `rls` to 2.12M rows/s at k=20 and 800k at k=50 on
the model arithmetic (1.93M through the bank), so the QR form now costs
1.3–2.1× the covariance form rather than 2–2.6×.

## 3. Usability

### U1 — features as expressions — *done*

The expression API took feature *names* only, so an AR term meant
materializing `pl.col("y").shift(1).over("g")` as a column first. Features
are now `str | pl.Expr`; an expression's output name is its feature name (so
it must be determinable and unique — `.alias` settles it), and under
`.over(group)` it is evaluated per group, so a lag stays inside its group.
Falls out of P1: the packed struct carries expressions as easily as columns.

### U2 — error messages that do not name the problem — *done*

Collected from the probes, before and after:

| what happened | before | after |
|---|---|---|
| feature column missing | `not found: "nope" not found` | `spec "m": feature column "nope" not found; the frame has columns ["t", "x0", "y", "s"]` |
| `targets="y"` (string, not list) | `invalid spec: invalid type: string "y", expected a sequence at line 1 column 42` | `spec "m": targets must be a list of strs, got str 'y'` |
| `halflife="10"` | `... did not match any variant of untagged enum FloatOrList at line 1 column 334` | `spec "m": halflife must be a number or a list of numbers, got str '10'` |
| `quantile=[0.5]` (list, expects float) | `invalid spec: invalid type: sequence, expected f64 at line 1 column 175` | `spec "m": quantile must be a number, got list [0.5]` |
| `solve_every=inf` | `invalid type: string "inf", expected f64 at line 1 column 300` | `spec "m": solve_every must be finite, got float inf` |
| `coef_every=-1` | `invalid value: integer -1, expected u32 at line 1 column 250` | `spec "m": coef_every must be >= 0, got -1` |
| `fit_predict(lazy_frame)` | `AttributeError: 'LazyFrame' object has no attribute 'get_columns'` | `TypeError: ... collect it first (lf.collect()), or feed it in chunks with fit_predict_batches(lf.collect_batches())` |
| String target column | no error; every prediction null | `spec "m": target column "s" has dtype str; it must be numeric (cast it, e.g. pl.col("s").cast(pl.Float64))` |
| List column as group key | `cannot cast List type (inner: 'Float64', to: 'String')` | `spec "m": group column "l" has dtype list[f64], which cannot be used as a key: ...` |
| spec named like an input column | no error; the input column silently replaced | `spec "y" has the same name as an input column; the output struct would replace it. Rename the spec.` |
| two threads calling `fit_predict` | `RuntimeError: Already borrowed` | `RuntimeError: ModelBank.fit_predict: the bank is in use on another thread; a bank is one ordered stream ...` |
| `np.float64` / `np.int64` parameters | `Object of type float64 is not JSON serializable` | accepted |

What was done, and where each message comes from:

- **Builders check their own annotations.** `_checked` (in
  `python/polars_online/_spec.py`) wraps every `po.spec.*` builder and
  compares each keyword with the function's type hints (and `_common`'s for
  the shared parameters): shape, sign for the count parameters, and
  finiteness for everything the Rust side parses as a plain `f64`. The hints
  are the contract, so a new parameter is checked by writing its annotation.
  The table of parameters that *do* admit `inf` (`_INF_OK`: no decay, no
  ceiling, a pinned coefficient, no clip) is checked against the Rust side by
  `tests/test_error_messages.py::test_the_inf_table_matches_the_rust_side`,
  which feeds `"inf"` straight to the parser for every float parameter of
  every builder.
- **The Rust parser names the path.** `serde_path_to_error` gives
  `invalid spec: [0].halflife[1]: ...` for a hand-built dict. It cannot see
  inside the model's `#[serde(tag = "type")]` union (serde buffers the
  content), which is exactly why the shape checks live in Python.
- **The untagged enums got visitors.** `FloatOrList`, `SessionGapSpec` and
  `Num` now say what they expect (`expected a number or a list of numbers
  ("inf" allowed)`) instead of `data did not match any variant of untagged
  enum FloatOrList` -- in JSON, in the CLI's TOML, and in the state file, all
  three of which are self-describing. `session_gap = "inf"` (never) is now
  accepted, since the visitor reads the same words as `Num`.
- **Column lookups carry the spec and the role** (`column`, `f64_column`,
  `key_column` in `crates/online-polars/src/bank.rs`), and `f64_column`
  accepts only numeric, Boolean and Null dtypes: anything else was cast
  non-strictly, and a String column of anything became all-null predictions
  with no error.
- **`Bank::fit_predict` refuses a spec named like an input column**, since
  both the Python bank and the CLI runner attach outputs with `with_column`.
- **`PyModelBank` borrows explicitly** (`try_borrow_mut` / `try_borrow`) so
  a second thread gets a sentence rather than pyo3's "Already borrowed";
  `fit_predict` releases the GIL, so this is reachable.
- `ModelBank.fit_predict` checks for a `DataFrame` first.

### U3 — `ModelBank` ergonomics — *done*

A bank was opaque: no `__repr__`; `bank.specs` was `[]` after `load`; no way
to list the groups a bank held or to drop stale ones, so a long-running
bank's memory grew with every group ever seen.

| | before | after |
|---|---|---|
| `repr(bank)` | `<polars_online._bank.ModelBank object at 0x...>` | `ModelBank(['m', 'r'], groups=3, rows_seen=60)` |
| `ModelBank.load(path).specs` | `[]` | the builders' dicts, `==` the originals |
| which groups are held | — | `bank.groups()` → frame of `spec, group, rows_processed, last_clock` |
| forgetting a group | — | `bank.drop_groups(keys, spec=None)` → number of streams dropped |

- **`specs` survive the file** through `specs_json()` on the native bank and
  `_from_json` in Python, which turns `"inf"` back into `float("inf")` --
  but only under parameter names the builders type as `float`
  (`_NUMERIC_KEYS`, derived from their annotations), so a column literally
  named `"inf"` stays a name. Two things had to change for `loaded.specs ==
  bank.specs` to hold: `SessionGapSpec::Gap` became a `Num` (serde_json
  writes a bare infinite `f64` as `null`), and `ewridge` stores
  `feature_sets` as lists rather than tuples.
- **`groups()`** reports each stream's processed-row count and its last
  clock value (`ClockState::last_clock`), sorted by key, with `""` for an
  ungrouped spec exactly as `solve_failures()` does. The stale-group idiom
  is `bank.drop_groups(bank.groups().filter(pl.col("last_clock") < cutoff)["group"])`.
- **`drop_groups`** removes the streams in every spec or one, and a dropped
  group restarts as a never-seen one would; the tests check the other groups
  continue bit-for-bit.
- **`rows_seen()`** is a bank-level `rows_fed` counter rather than a sum over
  streams, because a stream's count leaves with its group and excludes the
  rows the null policy skipped. It is an optional field of the map-encoded
  state file (`#[serde(default)]`, no format bump); a file from before it
  existed falls back to the streams' sum, and `tests/state_v1.rs` checks the
  v1 fixture does.
- `groups`, `drop_groups` and `rows_seen` are new rows in
  `tests/api_surface.txt`; `tests/test_bank_ergonomics.py` pins the rest.

### U4 — `**kwargs: Any` on every namespace method — *done*

Ten expression-namespace methods took `**kwargs: Any`, and every builder
took its shared parameters as `**common: Any` -- so for `halflife`, `clock`,
`session_gap` and the other twenty-odd parameters most calls are made of, an
editor showed nothing and a typo was a runtime error. Now:

- **`python/polars_online/_kwargs.py`** holds one PEP 692 `TypedDict` per
  model plus `CommonKwargs` / `ExprKwargs` (the shared parameters with and
  without `group`). The builders take `**common: Unpack[CommonKwargs]`; the
  namespace methods take `**kwargs: Unpack[EwridgeKwargs]` and so on, with
  `Required[...]` on `lasso_path`, `coef_halflife` and `quantile`. mypy
  now reports `Unexpected keyword argument "halflif" ... did you mean
  "halflife"?`, a missing `lasso_path`, a `str` where a float goes, and
  `group=` on the expression form. A TypedDict is a copy of a signature and
  copies drift, so `tests/test_kwargs_typing.py` pins every one to its
  builder: same keys, same annotations, same required set.
- **`po.online(expr)`**. A registered namespace is attached at runtime, so
  `pl.col("y").online` is `"Expr" has no attribute "online"` to every type
  checker -- a polars limitation no annotation here can fix. `po.online(
  pl.col("y"))` returns the same namespace object, visibly typed; the two
  spellings build identical expressions and a test says so.
- **`group=` on the expression form is refused** with `use .over(...)`.
  The Rust side set `spec.group = None`, so it had been silently ignored.
- **The pyo3 stub was stale** (`_polars_online.pyi` had no `gram` and no
  `spec_output_index`), which mypy found the moment it ran; it is now
  complete and a test compares it with what the built module exports.
- **mypy is in the gate and the lint job** (`uv run mypy`, package only,
  against the stub, so it needs no build; seconds, pure Python). It found
  the stub, a `_json` that accepted a list but said `dict`, and an `int`
  comparison typed as `object`.

### U8 — nothing said how to score without learning — *done*

The deployment question — load yesterday's fit, score today's rows, learn
nothing — has an answer (`weight = 0`) that appears nowhere in the docs, and a
plausible wrong answer (a null target) that quietly degrades the model.
Measured, 100 rows scored against a fit at halflife 20:

- **weight 0**: coefficients frozen *bit for bit*. Mean-form accumulators
  decayed with nothing added are themselves — `S' = (λ·W·S + 0)/(λ·W) = S`.
- **null target**: coefficients 1.00 → 1.21 and 2.00 → 2.39. The feature
  moments keep updating while the target's cross-moment does not, so the two
  halves of `S·β = r` end up estimated over different windows and β wanders
  with feature-moment noise. Right for a label that has not arrived yet;
  wrong as a scoring mode.

And one trap in the recommended path: a zero-weight row still advances the
clock, so `n_eff` decays while scoring — 29.4 → 0.95 over 100 rows — and once
it passes `min_periods` the outputs go null although the fit behind them never
changed (34 of the 100 rows, in the measurement). `min_periods` is baked into
the saved state, so a deployment cannot lower it for a scoring pass; it has to
be chosen with the scoring tail in mind.

README documents all three. A state-free `predict(df)` was recorded as
`docs/ENHANCEMENTS.md` E31 rather than built here, because `holt` and
`kalman` legitimately extrapolate with elapsed time, so "as of when" is a
parameter of such a call and not an omission; it has since been built with
that parameter answered (the clock distance from the last learned row) and
is the recommended way to serve — see E31. `TestScoringWithoutLearning`
still pins the frozen coefficients, the drift, and the decay of the
in-stream route.

### U7 — `coef` said "nothing yet" in two different spellings — *done*

Checking the docstring examples the README does not carry (T5 covers those)
turned up one that raises: `coef_index`'s own example,
`out["m"].struct.field("coef@h100").list.get(pos)`, fails with an
index-out-of-bounds `ComputeError` on real data.

The cause is a spelling inconsistency. Rows between `coef_every` snapshots are
`null`, which is how every other output says "nothing here" — but rows before
the model's first solve were an **empty list**, and `list.get` on an empty
list is out of bounds rather than null. So the documented way to read one
coefficient worked only on a spec whose first row happens to be a solve.

`null` in both cases now. The gradient models (`rls`, `kalman`, `ftrl`, `sgd`,
`pa`, `holt`) carry coefficients from the first row and were never affected;
the four that solve on a schedule were. Pinned for every model by
`test_coef_is_null_or_complete_never_empty`, which delays the first solve on
purpose — the first version of that test passed against the bug, because the
sweep's own specs solve on row one.

### U6 — the README's own `holt` example did not run — *done*

Found by running the README's python blocks instead of only compiling them
(T5). `po.spec.holt("baseline", targets=["y"], clock="t", level_halflife=200.0,
trend_halflife=2000.0)` — the example under the model's own section — raised
`spec "baseline": one of halflife/lam is required`.

The example was right and the rule was wrong. For `holt` the level halflife
*is* the spec's halflife: `build_one` defaults `level_halflife` from it, so
they are one knob spelled two ways, and a spec that gives `level_halflife` has
said what the decay is. `decays()` now accepts that spelling for `holt` alone
and nothing else — the two spellings produce identical output, ungridded field
names, and `trend_halflife` by itself is still refused, as is a non-positive
`level_halflife`.

### U5 — a `Decimal` column anywhere in the frame aborted the process — *done*

`ModelBank.fit_predict` takes the whole frame, and the whole frame crosses
into Rust, so every column's dtype has to be one this build can receive. Two
were not, and they did not fail cleanly: a `Decimal` or `Int128` column
panicked inside polars' own conversion with `activate 'dtype-decimal, '
feature` — a message naming a cargo feature, raised before any validation of
ours could name the column, and on a column the spec never asked for. Prices
in parquet are commonly `Decimal`, so this is a first-minute failure for
exactly the data this library is for. `Int8`/`UInt8`/`Array` were not much
better: `ComputeError: cannot create series from Int8`, which at least is an
error, but names neither the column nor the fix.

Probed every dtype polars has, as an unused column and as a feature. The
build was missing `dtype-i8`, `dtype-u8`, `dtype-i16`, `dtype-u16`,
`dtype-i128`, `dtype-decimal` and `dtype-array`; with them on, every dtype
either works or is refused by our own message, and the narrow numeric ones
(`UInt8`, `Decimal`, …) are legal features that give bit-identical answers to
the `Float64` columns they cast from.

The cost is code size, and no new dependency: the graph is 453 crates either
way, the extension goes 53.6 MB → 59.3 MB, and gzipped — the wheel's own
measure — 17.6 MB → 18.9 MB, +7.4%. That is the price of not aborting on a
column the user did not name.

`tests/test_error_messages.py` holds the table: every dtype, unused and as a
feature, plus a cast-equivalence check; `tests/test_runner.py` reads a
`Decimal` parquet through the CLI's own path.

## 4. Extensibility

### X1 — the output dtype lives in one place — *done with C1*

Adding an output kind used to mean a name-prefix rule in the plugin *and* a
branch in `assemble`. With `dtype` on `FieldMeta` both read the same
descriptor.

### X2 — adding a model touches eighteen places — *done*

Counted against the `holt` commit (`aa96ad3`), it is eighteen steps, not
eleven: the core file and its `pub use`, `ModelState` and the contract
probe, `ModelKind` with `kind_name`/`validate`, the six `AnyModel` arms,
`output_index` for an unusual layout, the builder with `_INF_OK`, the
`TypedDict`, the namespace method, the API snapshot, four per-model sweep
lists, the golden pipeline, the README heading, the changelog. None of it is
avoidable — each is a real decision — but nothing listed them, and the
compiler only covers the Rust half: every Python-side list was a plain list,
and a model left out of one was simply never swept.

`docs/EXTENDING.md` is the list, in order, with the check that fails on each
omission. The check is a registry: `ModelKind::KINDS` (held to the enum by a
unit test that reads serde's own "expected one of" error, the one place the
variant list exists outside the enum), exposed as `_polars_online
.model_kinds()`, and `tests/test_model_registry.py` holds the builders, the
namespace, the four sweeps, the golden bank, the API snapshot's output-field
blocks and the README headings to it. `model_contract.rs` does the same for
`ModelState` with `PROBED`. Every new check was mutated once — an entry
removed from each list — and each failed on its own list and nothing else.

Writing the checks found two gaps in the existing suite: the cross-platform
golden pipeline pinned nine models and not `ftrl`, and the API snapshot pinned
output fields for every model but `lasso`. Both are pinned now.

Two steps were left with **no check** on purpose and have since acquired one
(2026-09-02). Step 5, `golden.rs`, was "optional" — and four of the eleven
kinds (`sgd`, `pa`, `holt`, `ew_cov`) had no signature there, which is
exactly the set whose only oracles are the longhand recursions in their own
modules, the set the mutation blind spot (`docs/TESTING.md`) bites hardest.
Each has one now (`sgd` two, the plain and the busy path), each was mutated
once to confirm it moves, and
`test_model_registry::test_the_core_golden_file_pins_every_model` holds the
file to `KINDS`. Step 13, the per-model Python test file, is the other; see
below.

## 5. Testing

### T1 — eight `cargo run` calls cost half the suite — *done*

`cargo run -q -p online-cli` took 2.9 s per call with nothing to build, and
the eight CLI tests took 33 s of a 64 s suite. The proposal blamed cargo's
freshness check; measuring it showed that is 0.15 s. The rest is the launch:
on macOS cargo re-clones the binary into `target/debug` on every fresh
build (a new inode each time, `stat` shows), and the first exec of a new
file of a 418 MB debug executable spends ~2.7 s before `main` -- the
kernel validating the linker's ad-hoc code signature, 101k page hashes;
it scales with size (1.5 s for the stripped 208 MB) and the second exec of
the same file takes 10 ms. So `cargo run` paid it on every call.

`tests/conftest.py` now has a session-scoped `online_cli` fixture: one
`cargo build -q -p online-cli`, the executable's path taken from cargo's
`--message-format=json` artifact message (so `CARGO_TARGET_DIR` and the
`.exe` suffix are cargo's problem), and the two CLI test classes run it
directly. The eleven CLI tests went from 33.1 s to 3.6 s, of which 2.7 s is
the one launch the session still pays; the whole suite from 64 s to 31 s.

### T2 — zero Rust doc tests — *done*

`cargo test --doc` compiled nothing. Four examples now run under it, each
the first thing a Rust reader of that item sees and each asserting the
property the prose claims:

- **`online_core`** (crate root): `EwRidge` on `y = 1 + 2x` -- `pred` is
  NaN exactly while `Step::n_eff < min_periods`, converges to the line, and
  `restore(&model.state())` produces a model whose next `Step` equals the
  original's.
- **`OnlineModel`**: `Holt` with a missing target predicts without learning,
  and a zero-weight row with a target of `1e9` leaves the coefficients
  untouched (hard rule 9).
- **`ClockState::advance`**: the first row's delta is 0, a skipped row's
  delta is carried into the next accepted row, and a gap is capped at
  `max_dclock`.
- **`online_polars`** (crate root): a `Spec` from JSON, `Bank::fit_predict`
  with `pred_y` null until `min_periods` and correct after, and a bank
  loaded from `save_bytes` giving output identical to the one it was saved
  from.

`cargo test --workspace` already ran doc tests, so the gate and CI pick
them up without a change. The first draft of the bank example asserted
`pred_y` to `1e-6` and failed at `2e-6`: the ridge penalizes mean-form
moments (`S` is an EW mean, not a sum, so the penalty does not fade with
the sample), and on O(1) features `ridge = 1e-6` is a `2e-6` bias -- the
kind of thing an example that runs teaches and one that does not cannot.

### T3 — emit-flag matrix through the expression path — *done with C1*

### T4 — bounded-input contract for every model — *done with C2*

The test is described under C2. It found C5, which is the argument for
having it: a property test over the whole input range, run on every model,
against a twin that saw only clean data.

### T5 — the README was compiled, not run — *done*

`test_python_blocks_compile` checked that every python block in the README
parses. Parsing is not much of a guarantee: a block can parse and still call a
keyword that does not exist, name a field that was renamed, or — as it turned
out — be refused by the library's own validation. Running the nine blocks
found two: `po.spec.holt(..., level_halflife=200.0)` was refused by a rule
that should not have applied (U6, fixed), and the same example was missing
`max_dclock`, which *is* required with a clock (example fixed).

Each block now runs in its own copy of a namespace holding what the prose has
already introduced by that point — a frame with every column the README names,
the grid its field-name examples filter on, a fed bank, a scored output frame,
and the parquet/TOML files the runner examples read — in a `tmp_path` working
directory. Blocks do not see each other's leftovers, so the order in the file
is not load-bearing. A block that needs a name the prelude does not have fails
with `NameError`, which is the finding rather than a nuisance: the README
would be using something it never showed the reader.

Mutation-checked by renaming a keyword in a README example (`by=` to
`group_by=`): the block for that line fails, and only that one.

## 6. Rejected or deferred

- **P2, P3** above — measured, not worth it.
- **Reentrant `ModelBank`.** A lock that serializes concurrent `fit_predict`
  calls would silently reorder chunks; raising is right, only the message
  changes (U2).
- **Adaptive magnitude bound** (per-column scale instead of a constant). The
  constant catches sentinels and overflow, which is the failure that exists;
  everything below it is the model's own arithmetic and the user's choice of
  a robust loss.
