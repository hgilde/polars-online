# Review of tasks 45–56 (E54–E64): items to fix

A code review of the closed-group queue, `deco`, `clear_lags`, `ew_cov`
lags, `po.prep.refresh_time`, `rcov`, `po.corr`, `po.sim.regimes`, `hmm`,
`corrchange` and `bocpd`, as merged at `ac007e3` (2026-09-06). Each item
names the file and line, says what goes wrong, gives the input that shows
it, the direction of the fix, and the test that is missing. Items marked
**verified** were reproduced against the built package; the numbers quoted
are from those runs. The rest were found by reading and should be reproduced
first — the test that shows the bug is the first half of the fix.

Nothing here is a change of design. Where an item says *decide*, the code
does something defensible that the documentation does not say; either is
fine, but one of the two has to move.

Order: by severity, then by module. Tick items here as they land, with the
commit; keep the numbering.

**All 29 are fixed** (`docs/PLAN.md` task 57, 2026-09-06), each with the test
named under it; a ✅ on the heading says so. Where an item asked for a
decision rather than a repair, the decision is in `docs/PLAN.md` §11a under
*Fixing the review of 45–56*, and the item's own text below is left as it was
written -- it is the record of what was wrong, not of what the code does now.

## High

### L1 — `ew_cov` lag ring never refills after a save/load of a partially filled ring — **verified** ✅

`crates/online-core/src/ewlagcov.rs:161`

```rust
if self.ring.len() == self.ring.capacity().max(1) {   // capacity, not max(lags)
    self.ring.pop_front();
}
```

`new` allocates the ring with `VecDeque::with_capacity(max_lag)`, and the
push guard reads that capacity back as the ring depth. `VecDeque::clone` and
`serde` deserialization allocate exactly `len`, so a ring cloned or restored
while it held fewer than `max_lag` rows keeps a capacity of `len` for ever:
the guard pops on every push and the ring never grows past that length. The
lags deeper than the ring only decay from then on.

`state()` clones, so this is every `save_bytes`/`load_bytes` of a bank whose
lag ring was not full at the save: a stream shorter than `max_lag` rows, or a
save right after a session change or a capped gap (both call `clear_lags`).

Reproduced: `ew_cov(lags=[1, 2, 3], halflife=1e9)`, save after 1 or 2 rows,
load, continue: `lag_comoments[2]` is `0.0` in the resumed run and `0.0407`
in the continuous one. After 3 or 10 rows the two agree. In Rust:
`original depth=3 clone depth=1`.

Fix: store `max_lag` (`*lags.last()`) and compare against it. Never read a
container's capacity as a size.

Tests: (1) `test_ew_cov.py`: save after fewer than `max(lags)` rows, load,
continue, compare every `lag_comoments` slot with the continuous run; also
save right after a capped gap. (2) `model_contract.rs`: the ew_cov probe
uses `lags: Vec::new()`; give it lags so the contract's own save/restore
sees the ring.

### R2 — `rcov::clear_lags` emits the first post-break return `m+1` times and drops the tail — **verified** ✅

`crates/online-core/src/rcov.rs:842`

`clear_lags` clears `fin`, `pre_ring` and `tail`, but not `n`, `emitted` or
`head`. The kernel path's `push` emits from `tail.front()` once `n > 2m`,
keeping the tail at `m + 1`; after the clear the tail is empty but `n` still
says the steady state, so the next return is pushed and emitted, and then
re-emitted as `tail.front()` until the ring is `m + 1` deep again. The `m`
unemitted trailing returns before the break are lost outright.

Reproduced with `H = 0` (so the kernel estimate must equal the plain one)
over returns `1..20` with a clear at row 10:

| `m` (jitter) | kernel after clear | plain | difference |
|---|---|---|---|
| 1 | 2891 | 2870 | −100 + 121 |
| 2 | 3012.25 | 2951.25 | |
| 3 | 3304.56 | 3186.56 | |

Reported `n` is unchanged, so nothing downstream can tell.

Reachable through `clock` + `max_dclock` on an rcov spec (any intra-block
gap over the cap), or a session change without a reset — see R10.

Fix: on a clear, drop the tail *and* the trailing `m` slots' worth of `n`
(so the next return is treated as a block start: emitted only when `n == m`
again), or flush the tail into the sums before clearing — pick the one that
matches what the doc comment says a break means. Either way `n`/`emitted`
must agree with what has actually been emitted.

Test: `kernel(H=0, jitter=m) == plain` across a `clear_lags`, for `m` in
1..=3; and the existing `clear_lags_drops_the_rings_and_keeps_the_sums`
should go on to push rows and compare, not stop at "rings empty".

### H1 — `hmm.min_periods` gates the update, not only the report — **verified** ✅

`crates/online-core/src/hmm.rs:391`

`read` returns `(nan, None)` while `n_eff < min_periods`; `step` treats the
`None` as a solve failure and learns nothing. So `min_periods` withholds
rows from the filter, and under decay `n_eff` may never reach it, which is
every row null for ever.

Reproduced, `hmm(k=2, halflife=1e9, warm_rows=20)`, rows 60 onward:
`min_periods=0` and `min_periods=50` give different `p_0` (0.391 vs 0.274 on
row 60) — the second run never learned rows 21–50. The same experiment on
`bocpd`, `deco` and `corrchange` gives identical rows after warm-up, as it
should. With `halflife=10, min_periods=50` every one of 400 rows is null.

Fix: gate the *report* only — compute the update regardless of
`min_periods` and blank `pred` on the way out; `solve_failures` must not
count a warm-up row.

Test: `test_hmm.py` has no `min_periods` test at all. Add: outputs after
warm-up equal the `min_periods=0` run bit for bit; `min_periods` larger
than the decayed plateau still produces output once the row count passes
it under `halflife=None`.

### B1 — `bocpd` silently drops rows when the hazard column is ≤ 1 (or the emission fails) — **verified** ✅

`crates/online-core/src/bocpd.rs:487–493, 627–648`

`read` returns `(nan, None)` when `1/hazard` is not in `(0, 1)` and when
`log_predictive` fails (the `SpdFactor`). `step` then does
`self.n_eff += weight` and returns without touching the posterior: the row
counts toward `n_eff`, reports null, and leaves no trace. There is no
`solve_failures` counter on this model.

Reproduced: `hazard_col` = 0.5 from row 100 → 100/100 nulls, `n_eff` = 199
at the end.

Fix: a hazard-column value ≤ 1 (or non-finite) on a row is a bad input, not
a null output — refuse it with the row named, the way `weight` is refused;
count an emission failure in a `solve_failures` the diagnostics report, as
`hmm` does. Decide whether `n_eff` should advance on a row that taught
nothing (the other models' answer: no).

Test: `test_bocpd.py` reads `hazard_col` only with valid values through
`fit_predict`. Add ≤ 1, 0, negative, NaN and null values; the null case is
the one that should silently fall back to `hazard` (it does — cover it).

## Medium

### B3 — `bocpd.predict` ignores `hazard_col` — **verified** ✅

`crates/online-core/src/bocpd.rs:671`

`predict` calls `read(x, self.cfg.hazard)`; `step` reads the row's hazard.
Under `hazard_col` the two differ on every row where the column is not
`cfg.hazard`. Reproduced: `hazard_col` alternating 20/500, row 150:
`p_change` 0.0503 from `fit_predict` and 0.0203 from `predict`.

The Rust `predict_is_the_step_without_the_step` for bocpd runs with
`hazard_from_row: false`, so it cannot see this. See C1 for the root cause.

### H2 — `hmm.predict` uses `exog = 0` under `tvtp` — **verified** ✅

`crates/online-core/src/hmm.rs:543`

`predict` passes `Some(0.0)` as the exogenous value. Reproduced with
`exog_tvtp="z"`: `p_0`/`p_1` agree (they are the prior state), but `p1_*`,
`state` and `loglik` differ — row 130: `state` 0 from `fit_predict`, 1 from
`predict`; `loglik` −4.18 vs −2.35.

Also undocumented and untested: a null or non-finite `z` is read as `0`
(`transition_at`, line 276), and the row is still learned. Reproduced: a
null at row 150 changes row 151. Decide (skip the row like a null feature,
or keep `z = 0` and say so) and test either.

### C1 — `OnlineModel::predict` cannot see the targets slot ✅

`crates/online-core/src/model.rs` (`predict(x, d_clock)`),
`crates/online-polars/src/bank.rs:2416` (`predict`)

B3 and H2 are one bug: `hazard_col` and `exog_tvtp` ride in the targets
slot, which `predict` does not receive. Either give `predict` the `y` slice
(the models that ignore it keep ignoring it), or have `Bank::predict` and
`predict_chunk` refuse a spec that reads a per-row column through that slot.
The first is the honest fix; the second is one `polars_bail!`.

Test: `predict_is_the_step_without_the_step` in both Rust files with a
*varying* hazard / exogenous column, and one pytest per model comparing
`ModelBank.predict` with `fit_predict` on a column that differs from the
config default.

### H3 — `hmm` and `corrchange(scalar)` do not decay `n_eff` on a zero-weight row — **verified** ✅

`crates/online-core/src/hmm.rs:485` (early return before the `n_eff`
update), `crates/online-core/src/corrchange.rs:642–652` (same shape)

Rule 8: `n_eff` is the weight before the row's update and before its own
decay, one recursion for every model. A zero-weight row still decays it
(`w_sum = lam·w_sum + 0`). Reproduced, `halflife=10`, weights 0 on rows
30–39, `n_eff` on rows 31/35/40:

| model | row 31 | row 35 | row 40 |
|---|---|---|---|
| `ewridge` (reference) | 12.19 | 9.24 | 6.53 |
| `deco` | 12.19 | 9.24 | 6.53 |
| `hmm` | 13.07 | 13.07 | 13.07 |
| `corrchange(scalar)` | 13.07 | 13.07 | 13.07 |

So `min_periods` means a different number of rows for these two after any
zero-weight stretch. (`hmm` does decay its states and counts on the row —
only `n_eff` is frozen.)

Fix: apply `n_eff = lam·n_eff` before the early return.

Test (C2): the `n_eff` recursion probe in `model_contract.rs` feeds only
weight-1 rows; feed it zero-weight rows under decay and compare every model
against the reference recursion. The parity probe already feeds them but
checks only `predict == step`.

### L2 — `label_delay` replays across a capped gap after the ring was cleared ✅

`crates/online-polars/src/stream.rs:2019` (`apply_label_delay`),
`stream.rs:2552` (`clear_lags` in `run_instance`)

On `session_changed` every pending row is released first; on `capped` only
those with `remaining <= 0`. The capped row's own `clear_lags` fires when it
is *scored*, and the still-pending pre-gap rows are replayed afterwards with
`learn: true`, followed by the post-gap rows: the lag ring (`ew_cov` lags,
`corrchange` span, `rcov` tail) pairs rows from both sides of the break it
was just cleared for.

Fix: release all pending rows on `capped` as on `session_changed`, or carry
the clear as a pending event so it replays in order.

Test: `ew_cov(lags=[1], label_delay=k)` with a capped gap, against the same
stream without a delay; the lag comoments must agree once the replays are
in.

### RT1 — `refresh_time` returns the `by` column as `str` whatever it was — **verified** ✅

`crates/online-polars/src/refresh.rs:213–253, 323`

`feed` casts `by` to `String` and `frame` builds the output column from the
key strings. An `i64` group comes back as `str`. `RefreshTime::schema()`
declares the input dtype, but `prep.py` takes the schema from
`build().feed(empty)`, so the two never disagree in practice — and both are
wrong for the user who joins the result back to the input.

Fix: cast the column back to the input dtype in `frame` (a failed cast
cannot happen: the strings came from that dtype), or document "the `by`
column is returned as `str`". Test: an integer and a categorical `by`
round-trip their dtype.

### RT4 — ties at a grid point count toward the next interval — **verified** ✅

`crates/online-polars/src/refresh.rs:213` (`feed`)

A tick at the *same* timestamp as the one that just closed a grid point, but
later in row order, is counted in the next interval. Refresh time (BNHLS
Definition 1) takes τ_{j+1} as the first time at which *every* series has
ticked strictly after τ_j. Reproduced: rows `(a,1) (b,1) (b,1: v=2.5) (a,2)
(a,3) (b,3)` give a grid point at `t = 2` with `b_value = 2.5`, which is a
`t = 1` tick.

Decide: buffer equal times (close the point only when a strictly later
time arrives), or document that ties are broken by row order and that
`b`'s second tick at `t = 1` belongs to the next point. Test either.

### G1 — the high-water guard is one-directional ✅

`crates/online-polars/src/bank.rs:611–618`

The guard refuses an integer column against a saved non-integer mark. The
other way — an integer mark saved, the column now `String` — is not caught:
bytewise `"9" > "10"` lets a lower group past the mark. `key_integer` is not
in `BankFile` (line 1530), so a symmetric guard needs it persisted (a format
bump with the loader kept) or a heuristic on the saved mark.

Test: `test_closed_groups.py` covers the integer-after-text direction;
add text-after-integer.

### G2 — Categorical keys compare bytewise, polars sorts them physically ✅

`crates/online-polars/src/bank.rs:922` (`key_cmp`), `941`
(`group_key_is_integer`)

A frame sorted by a Categorical column (physical order by default) can be
refused as non-monotone because the keys are compared as strings. Document
it in `group_close`'s docstring ("sort a Categorical key lexically, or cast
to `String`") and test the refusal message names the sort.

### R10 — `rcov` accepts `clock` / `max_dclock` / `session` although nothing in it decays ✅

`crates/online-polars/src/spec.rs:2126–2143`

The arm refuses `halflife`/`lam` and a missing `group`/`group_close`, but
not the clock options. They do nothing except make R2 reachable. Refuse them
with the same wording as `halflife`, or document what a capped gap means for
a block (once R2 says so).

## Low

### B2 — `bocpd` prior validation gaps ✅

`crates/online-core/src/bocpd.rs:214–231`

A scalar `prior_scale ≤ 0` passes (only finiteness is checked) and gives NaN
for ever; a `d×d` `prior_scale` is not checked SPD; a non-finite
`prior_mean` is accepted. `test_bocpd.py` refuses a 2-vector `prior_scale`
and nothing else about it.

### B4 — `logjoint` is never renormalised ✅

`crates/online-core/src/bocpd.rs:546–555, 581`

The run-length log-joint is carried unnormalised: `read` builds `new` from
`logjoint + log π` and `prune` reads `z` without subtracting it. Every
output is a difference against `z`, so this is exact until `|logjoint|`
is large enough for the subtraction to lose digits — at roughly 1.4 nats a
row that is past 1e9 rows before it costs 1e-7, so it is tidiness, not a
bug. Subtracting `z` in `prune` makes it exact for ever at no cost. Test:
`logjoint.max()` stays bounded over a long stream.

### H4 — `hmm` given moments ✅

`crates/online-core/src/hmm.rs` (`covs` / `means` handling)

Given `covs` are length-checked only; a non-PSD one makes every row a solve
failure (all NaN, no message). Given moments carry `w_sum = 1.0`, which the
docs do not say and which washes out under `learn=true` — so the "known
regimes" configuration drifts from the given ones at a rate nobody chose.
Refuse non-PSD; document the weight or make it a parameter. `test_hmm.py`
uses `covs` only as valid diagonals.

### CC1 — `corrchange.crit` is never validated ✅

`crates/online-core/src/corrchange.rs:139–200`

`crit = NaN` never flags, `crit ≤ 0` flags every row, `inf` never. One line
in `validate`; one case in the bad-config test.

### CC2 — a zero-weight row's `corrchange` output includes the unlearned row ✅

`crates/online-core/src/corrchange.rs:578–581, 639–646`

`read` has a `weight <= 0.0` guard that is dead: `step` reports through
`self.predict`, which calls `read(x, 1.0)`. So a zero-weight row reports the
span-closing statistic computed *with* itself, at a `since` count one past
the truth, and a flag on it does not reset `since_flag` (the row is never
pushed). The contract's parity probe currently *requires* this (predict ==
step on zero-weight rows). Decide what a zero-weight row reports here —
null is the honest answer, given the row is not in the span — and change
the parity probe to allow it, or keep the current output and say so in the
docstring. Either way `since_flag` must not be reset by a phantom flag.

Test: the zero-weight test checks depth and `n_eff` only; check the
output row.

### CC3 — bad-config cases missing ✅

`crates/online-core/src/corrchange.rs` tests

`perm_block = 0` and `> window`, `permute_every = 0`, `bandwidth = Some(0)`
are refused by `validate` and not covered.

### R9 — `rcov` validation gaps ✅

`crates/online-core/src/rcov.rs:352` (`validate`), `313` (`window_for`),
`338` (`ring_for`)

An explicit `window: Some(0 | 1)` bypasses the `.max(2)` of the derived
window, so `kn < 2` and the pre-averaging branch (`push`, line 590) is
skipped on every row: no block ever accumulates. `h_max: Some(0)` beside a
`bandwidth` silently clips `H` to 0 in `estimate` (the kernel estimate is
the plain one). `n_max: Some(0)` is accepted and means a ring of depth 1
and a window of 2, which nobody asked for. Refuse the three in `validate`;
three cases in the bad-config test.

### D1 — `deco` block edge cases ✅

`crates/online-core/src/deco.rs` (`deco_cfg`),
`crates/online-polars/src/spec.rs` (Deco arm)

`blocks` is an array of pairs in TOML/JSON, so duplicate names are
expressible; they are refused, but by the output-name collision check with
a message about "a target or grid label" that does not mention blocks.
`blocks = []` is accepted and silently means unblocked. Refuse duplicates in
`deco_cfg` with a block message; decide on `[]` (refuse, or document as
"unblocked"). Python builds `blocks` from a dict, so `test_deco.py` cannot
reach either — test from a JSON spec.

### G3 — the IO plugin keeps every closed frame in memory until the end ✅

`python/polars_online/_frame.py` (`_source`, `_write_closed`)

`closed_path` frames are appended to a Python list per chunk and written
once at the end. The queue in the bank is bounded by the drain; the sidecar
is not — a long run with many closes holds them all. Write incrementally
(sink per chunk, or a `pl.LazyFrame` concat over the parts), or document
the bound.

### G4 — PCA sign continuity is across groups per `(spec, instance)` ✅

`crates/online-polars/src/bank.rs:2510–2520`

By design (the previous closed row of the same instance, in queue order),
but not said anywhere a user of `closed_groups` would look: a group's first
component is signed for continuity with the *previous group's*. Document in
the `closed_groups` docstring and `docs/PLAN.md` §11a.

### K1 — `po.corr.nearest` on non-finite input; `max_iter = 0` — **verified** ✅

`python/polars_online/corr.py:116`

No finiteness check: `nearest([[1, nan], [nan, 1]])` returns the NaN matrix
with distance `nan` and `iters = 1`, no error. `max_iter = 0` returns the
input unchanged — diagonal `1.2` and all — with distance `0.0` and
`iters = 0`, which reads as "already a correlation matrix". Refuse
non-finite input; refuse `max_iter = 0`, or report the true distance.

### K2 — `po.corr.shrink` mixes `r` and `x` ✅

`python/polars_online/corr.py:206`

The target `f` is built from `r` but `sam`, `pi` and `theta` from `x`, so
`gamma = Σ(f − sam)²` compares two different matrices when `r ≠ cov(x)` —
which is the documented use, a bank's exponentially weighted `r` beside the
plain rows it came from. Either build `f` from `sam` too (so `alpha` is
Ledoit–Wolf's for the sample, applied to `r`), or say in the docstring that
`x` must be the rows whose plain covariance is `r`. Also `T < 2` is not
refused (**verified**): `T = 1` silently gives `alpha = 0`; `T = 0` fails
with a bare `ZeroDivisionError`.

### K3 — boundary tests missing in `po.corr` ✅

`epps_invert` (needs every lag `1..L−1`; `L = 1`), `fisher_se` (`n ≤ 3` →
NaN), `signal_share` and `mp_edge` (`Q < 1`) behave sensibly but nothing
pins the boundary values.

### S1 — `sim.regimes` validates rows with `allclose`, numpy draws with a tighter tolerance — **verified** ✅

`python/polars_online/sim.py:98`

A row summing to `1 + 1e-6` passes `np.allclose` and then fails inside
`rng.choice` with numpy's "Probabilities do not sum to 1". Normalise each
row before drawing (after the check), or use numpy's tolerance in the check.
Test: the row above.

### P1 — a run of literal spaces in an error message ✅

`crates/online-polars/src/spec.rs:1540`

`"... 0 is no delay,                      which is the default"` — a missing
`\` continuation. Cosmetic.

## Checked and found right

So the fixer does not chase them again:

- `min_periods` is report-only for `bocpd`, `deco` and `corrchange` (rows
  after warm-up are bit-identical to the `min_periods = 0` run).
- `bocpd.predict == step` when `hazard_col` is not set; `hmm.predict == step`
  for `p_*` under `tvtp` (the prior state does not read `z`).
- `deco` overlapping blocks are refused (`column 1 is in more than one
  block`); duplicate block names are refused (see D1 for the message).
- The closed-row queue is sorted at enqueue with the spec's `key_integer`
  set from the chunk first, so a loaded bank drains in the right order.
- `refresh_time` refuses a null series name, an unknown name, a null or
  non-finite time and a backwards time, naming the row; a null value is
  skipped without counting a tick.
- `sim.regimes` with `durations` terminates on an absorbing state.
- The `coef` column being null on a mid-chunk row and populated by a
  one-row `predict` is the chunk's-last-row rule, not a discrepancy.
