# `marginal` at width, round two: what a row costs at 10⁴ features × 50 targets, and five asks

*2026-09-25. Five asks against 0.10.0, from the same caller as
`MARGINAL-LAGS-AND-BINS.md` (its first pass, the screen of every factor
against every target and its null targets, one `marginal` per warm-up
class per block): `ENHANCEMENTS.md` §14, E70–E74. None
changes what `marginal` reports or the contract it reports it under (no
per-row output but `weight_sum`, pairs read back as a frame, closed-group
rows, bit-level chunk invariance). Three remove work the model repeats once per
target that depends on the feature alone; one lets a wide spec use the
pool; one is about the loop order, which a measurement showed carries a
fixed cost per feature per row larger than the pair's own step. Each
section gives the motivation, the API, the state and cost, the invariance
argument, and the tests that would pin it. The measurements are the
caller's, made against the wheel this checkout built (0.10.0, M4 Pro, 14
cores), with the scripts named so they can be rerun here.*

*Examined here on 2026-09-25, against the code and a benchmark of the
model alone at `p = 10,000` (`docs/PLAN.md` tasks 122–127). E70 and E71
hold as written. Two claims do not, and each is noted where it is made:
E72's `cov` is not exact where a target is absent on some learned rows,
and the fixed cost E74 is built on is not in the model.*

*Built the same day: E71 as task 122 and E70 as task 123, each
bit-identical where the request said it would be. `PERFORMANCE.md` §22 and
§23 have the measurements. E71's one-target row moved too, by a fifth,
where the request expected no change; E70's single cross lag took 30% off
the lags' cost, where the moment count suggested 56%.*

*E73 was built the same day as task 126, over a batch of held rows as its
Mechanics below describe, after a fork-join per row measured at best 0.6×
at one target. `marginal(shards=)` takes a count or `"auto"`, bit for bit
at any count. At 10,000 features, nine targets, six lags and sixteen bins
the model alone ran 5.4× on 14 threads and the bank 4.9×; the moments of
one target, 1.8× and 1.2×, where each held row's copy and the bank's own
row work do not split (`PERFORMANCE.md` §25).*

*E72 was built on 2026-09-29 as task 125.
`marginal(feature_moments="shared")` keeps one mean and one variance per
feature over every learned row, the lagged autocovariance `cxx` per
feature too, and each pair only its covariance. Where a target is absent on some learned rows that is the
other estimator its section below examines, and the docstring says so. A
window is refused by name. On the model alone it ran 2.7× at ten targets
and 3.2× at thirty, level at one, and 4.6× with lags at ten
(`PERFORMANCE.md` §27).*

---

## The caller's shape, and what it reads

**The first pass** runs one `marginal` per warm-up class (five to ten classes) over
`p = 10,000` features against `T` target columns, per block of a
1 Hz table: `lam = 1`, a weight column (0 on warm-up and purge rows),
`clock` in seconds with `gap_cap` a day, no session, no window,
`lags = [1, 2, 5, 10, 20, 50]` (`[..., 100, 200]` under its `full`
profile), `serial_rule = "geometric"`, `bins = 16`, `chunk_rows = 100,000`.
`T` is 9 under its `lean` profile — three targets, each with two null
copies (the target at the same time of day one and five dates earlier) —
and 50 under `full` — ten targets, three nulls and a clipped copy each. A
block is about 5 × 10⁵ rows, and a window holds 27 of them: `1.35 × 10⁷` rows × `10⁴` features = `1.35 × 10¹¹` feature-rows,
times `T`.

The pairs are read back once per block with `bank.marginal()` today, and
with `group = "block", group_close = "monotone"` and the closed-group row
in the design the caller is moving to (prototyped against 0.10.0: the
closed rows reproduce the per-block frame bit for bit, every column,
`bin_*` and `split_*` included). Of the pair's columns the caller's
decision reads exactly these:

| column | read for |
|---|---|
| `corr` | the Fama–MacBeth t over blocks on Fisher's z, and every descriptive |
| `n_serial` (`n_kish` where null) | the honest count behind that t (Bartlett) |
| `split_gain` | the nonlinear route, paired against the null targets' gains |
| `lagcorr_xy[0]`, `lagcorr_yx[0]` | the first lag only: the shift test (is a feature sampled late). *2026-09-27 (E75): the caller now reads `lagcorr_xy` at lag 1 and `lagcorr_yx` at lags 1 and 2, `cross_lags=[1, 2]`, about 3 % more state a pair* |
| `weight_sum` (`n_eff` until task 144), `n_kish` | counts |

Not read anywhere: the cross terms past the first lag, the `lagcorr_xx` /
`lagcorr_yy` lists themselves (only through `n_serial`), `phi_x`, `phi_y`,
`t`, `t_serial`, `beta`, `mean_*`, `var_*`, `split_at`, `split_gain_t`,
and the four `bin_*` lists. Everything below keeps every one of them
available; the asks are about what is *computed per row*, not what is
reported.

## Where the row goes

Measured with `probe_marginal_cost.py` (one `ModelBank`, one spec, 10,000
rows × 10,000 features × 9 targets, `lam = 1`, a unit weight column,
`chunk_rows = 100,000`, a child process per variant so the pool is sized
once), wall time in nanoseconds **per row × feature × target column**:

| variant | ns per pair-row | share |
|---|---:|---:|
| the moments alone (no lags, no bins) | 3.97 | 21% |
| + 16 bins | 10.42 | bins: 6.4 ns, 33% |
| + 6 lags, no bins | 13.30 | lags: 9.3 ns |
| 6 lags and 16 bins, one thread: **as the first pass runs a class** | **19.30** | 100% |
| lag 1 only, 16 bins | 11.92 | five lags of six: 7.4 ns |
| 6 lags and 16 bins, `chunk_rows = 2,000` | 21.51 | the per-chunk cost, 11% |
| the same features split over 7 specs, 7 threads | 3.84 | 5.0× |
| split over 14 specs, 14 threads | 2.98 | 6.5× |

So at width the pair's linear moments are a fifth of the row; the lags
are nearly half and the bins a third. How that scales with the number of
target columns — `probe_marginal_t.py`, `T` = 1, 3, 9, 30 (5,000 rows ×
10,000 features, one thread), nanoseconds per row per feature, the
per-target-column figure in brackets:

| variant | T = 1 | T = 3 | T = 9 | T = 30 | per feature + per pair |
|---|---:|---:|---:|---:|---:|
| 6 lags, 16 bins (the first pass's shape) | 36.2 (36.2) | 73.5 (24.5) | 189.0 (21.0) | 591.2 (19.7) | **17.1 + 19.1·T** |
| lag 1 only, 16 bins | 28.2 (28.2) | 50.0 (16.7) | 114.5 (12.7) | 341.1 (11.4) | 17.4 + 10.8·T |
| no lags, 16 bins | 25.7 (25.7) | 44.7 (14.9) | 100.0 (11.1) | 293.9 (9.8) | 16.5 + 9.2·T |
| 6 lags, no bins | 27.1 (27.1) | 52.2 (17.4) | 128.5 (14.3) | 400.1 (13.3) | 14.3 + 12.9·T |
| moments only | 17.4 (17.4) | 23.9 (8.0) | 43.4 (4.8) | 109.3 (3.6) | 14.3 + 3.2·T |

The last column is the line through `T = 1` and `T = 30` (the `T = 9`
figures land within 1% of it). Two things it says:

- **Per pair-row, the model does `T` times the same work**: 3.2 ns for
  the moments, 6.1 for the bins, 9.9 for six lags (1.5 for one), 19.1 in
  all. Reading `learn`, `learn_lags` and `MarginalBins::update_target`
  says why: everything is indexed `[t][j]`, and three things in that row
  depend on the feature alone.
- **And there is a per-feature term of 14–17 ns per row that has nothing
  to do with `T`.** For the moments alone it is 14.3 ns against a 3.2 ns
  pair step — at `T = 1` the fixed part is 80% of the row, at `T = 9` a
  quarter. It is not the pair loop, since the pair loop is the `T`
  coefficient; what is left per feature per row is getting the value to
  the model — one value from each of 10,000 columns per row, a gather
  across 10,000 arrays — and the per-row `x` bookkeeping around it. E74
  below is the ask that follows from it.

Under `full` the caller runs `T = 50`: fifty searches per feature per
row, fifty identical feature means, and 150 lag updates per pair where
`n_serial` uses six. The three redundancies:

1. **The bin index.** `MarginalBins.edges` is one edge list *per feature*
   (`edges: Vec<Vec<f64>>`, ragged), but `update_target(t, x, y, w)`
   runs `bin_of(&self.edges[j], x[j])` — a `partition_point` over 15
   edges — inside the per-target call, so a row of `T` present targets
   searches each feature's edges `T` times for the same answer.
2. **The feature's lagged autocovariance.** `cxx[ℓ][t][j]` is the
   feature's own autocovariance, kept once per target because it is
   centred on that target's `mx[t][j]`, and the cross terms `cxy` and
   `cyx` — two of the three lag updates per pair — are never read past
   the first lag by this caller, and enter `n_serial` at no lag:
   Bartlett's factor is `1 + 2 Σ ρ_x(ℓ) ρ_y(ℓ)`, autocorrelations only.
3. **The feature's mean, variance and run.** `mx[t][j]`, `mx_lo[t][j]`,
   `sxx[t][j]` and `x_runs[t·p + j]` are per target because each target's
   weight `W_t` is its own; on a row where every target is present they
   take identical steps from identical values, `T` copies of one number.

---

## E70 — cross-lag terms on request: `marginal(lags=..., cross_lags=...)`

### Why

`lags` keeps three lagged moments per pair per lag: the feature's
autocovariance, and the cross-covariance in both orientations. Of the
three, `n_serial` — the reason the lags exist — uses one. The two cross
terms are the lead/lag by-product (`lagcorr_xy`, `lagcorr_yx`), worth
having at the first lag or two and rarely beyond: a feature that leads its
target by fifty rows is a different question from the one `marginal`
screens. At `L = 6` that is `18·p·T` multiply-adds per row of which six
serve the count; measured, the five lags past the first cost 7.4 of the
row's 19.3 ns.

### API

```python
po.spec.marginal("m", features=[...], targets=[...],
                 lags=[1, 2, 5, 10, 20, 50],   # the autocorrelations, as today
                 cross_lags=[1],               # the cross terms, at these lags only
                 serial_rule="geometric", **common)
```

- `cross_lags`: a strictly increasing list of lags ≥ 1, each also in
  `lags` (the ring is sized by `max(lags)`, and a cross term at a lag the
  ring does not hold has nothing to read). `None`, the default, keeps
  today's behaviour — every lag in `lags` — so no caller changes; `[]`
  keeps no cross term and the two columns are absent, as they are for a
  spec without `lags`.

### Outputs

`lagcorr_xy` and `lagcorr_yx` become lists over `cross_lags`, in that
order, where they were lists over `lags`; with the default they are what
they are today. `lagcorr_xx`, `lagcorr_yy`, `n_serial`, `t_serial`,
`phi_x` and `phi_y` are untouched by any setting of `cross_lags`.

### State and cost

`cxy` and `cyx` go from `L·p·T` doubles each to `|cross_lags|·p·T`; the
per-row lag work from `3L·p·T` to `(L + 2|cross_lags|)·p·T`, at the
caller's settings `8·p·T` for `18·p·T`. The ring is unchanged. With E72
below, `cxx` moves to the feature (`L·p`) and the lag work to
`L·p + L·T + 2|cross_lags|·p·T`: six per feature and two per pair, where
today it is eighteen per pair.

### Invariance

The recursion per kept moment is the same operation sequence as today, so
the moments that remain are bit-identical to today's at the same lags
(`lagged_pair_moments_are_ew_covs_to_the_bit` stays the oracle), and chunk
invariance is untouched. The state gains `cross_lags` as an optional part
with a `serde` default (a state saved before it reads back as
`cross_lags = lags`, which is what it was); the compact-msgpack rule from
E66 applies (at most one skipped field, last). No schema bump.

### Tests

- `n_serial`, `phi_x`, `phi_y`, `lagcorr_xx`, `lagcorr_yy` bit-identical
  across `cross_lags = None`, `[1]`, `[]` on a random multi-target stream.
- `lagcorr_xy` / `lagcorr_yx` under `cross_lags = [1, 5]` equal the
  entries at those lags under the default.
- `cross_lags` naming a lag not in `lags`, or not increasing, refused by
  name at construction.
- A state saved without the field loads with the default.

---

## E71 — bin each feature once per row

### Why

The edges are per feature and the search is per pair. On a row with `T`
present targets, `update_target` is called `T` times and each call
searches every feature's edges again; the answer is the same each time.
Measured, 16 bins add 6.4 ns per pair-row to the moments' 4.0, and the
Welford step on the cell that follows the search is three multiply-adds
and a compensated add — on the order of the moments' own step — so most
of that 6.4 is the search, and a row of `T` targets pays it `T` times.
(The split is inferred from the shape of the code, not measured
separately; the benchmark below would measure it.)

### API

None. A `marginal` with `bins` computes what it computes today.

### Mechanics

`learn` bins the row once — a `Vec<Option<u32>>` (or a sentinel) of `p`
indices, held on the model and reused, filled before the per-target loop
when at least one target is present on the row — and `update_target`
takes the indices instead of the values. Nothing else moves: the decay
trick, the fold, the warm-up hold (which replays held rows through the
same path) and the ragged edges are as they are. On a row where no target
is present the indices are not formed.

### State and cost

No state. Per row, `p` searches instead of `p·T`; the Welford step stays
per pair. At `T = 9`, if the search is two thirds of the bins' 6.4 ns,
about 4 ns of the 19.3 go; at `T = 50` the same absolute saving per pair,
which is the same share.

### Invariance

The bin index is a pure function of the feature's value and its fixed
edges, so every cell receives exactly the rows it receives today, in the
same order, with the same weights: `bin_n`, `bin_mean_y`, `bin_var_y`,
`split_gain`, `split_at` and `split_gain_t` are bit-identical, and chunk
invariance is untouched (the indices are per row, formed and dropped
inside it). No state change, no schema bump.

### Tests

- On a random stream with five targets present on every row, and on one
  where each target is absent on a random third of the rows, every `bin_*`
  and `split_*` column equals today's to the bit (the current build as the
  oracle, the way `marg_bench` was held against `7c5327f` in
  PERFORMANCE.md §17).
- `marg_bench` with `bins = 16` at `T = 1` and `T = 9`: the `T = 9` row
  should move and the `T = 1` row should not.

---

## E72 — feature moments shared across targets: `marginal(feature_moments="shared")`

### Why

The pair's own step is small — two multiply-adds on `sxx` and `sxy` — but
what surrounds it is per pair too: the compensated deviation against
`mx[t][j]`, the compensated `add` that advances it, the run tracker's
compare-and-branch, and under `lags` the feature's own lagged
autocovariance. All of it is the feature's, repeated per target because
each target's rows are its own. When every target is present on every
learned row (the wide-screen case: the caller's targets are absent only
where a null target's source date does not exist, under a tenth of the
rows, and on whole sessions), the `T` copies of `mx[t][j]` take identical
steps — same `a`, same `b`, same `x_j` — and hold identical bits. Sharing
them leaves, per pair, the one multiply-add that is genuinely the pair's:
`sxy`. Measured, the moments alone cost 4.0 ns per pair-row; the pair's
share of that is the `sxy` line.

### API

```python
po.spec.marginal("m", features=[...], targets=[...],
                 feature_moments="shared",     # "per_target" (today, the default) | "shared"
                 **common)
```

Under `"shared"` the feature's mean, variance, run, lagged autocovariance
and (with E71) bin index are kept once per feature over every learned row
of the group, weighted and decayed by the group's own `W` (the model's
`w_sum` path, which `marginal` already advances per row); the target's
mean, variance and lagged autocovariance stay per target over the rows the
target is present on; the pair keeps `sxy` alone.

### What it reports, and the one place it differs

`cov` is exactly today's number under either setting. The pair's
co-moment is centred on the target's own mean over the target's rows,
`Σ w (x − c)(y − m_y) = Σ w (x − m_x^{(y)})(y − m_y)` for any constant
`c`, because `Σ w (y − m_y) = 0` over those rows — the feature's centring
constant does not enter it. What can differ is `var_x`: over every learned
row rather than over the target's rows. So `corr`, `beta`, `t` and
`n_serial` (through `ρ_x(ℓ)`, likewise over every row) differ from
`"per_target"` exactly where a target is absent on some learned rows, and
agree bit for bit where it is not. That is a valid estimator of the same
quantity — the feature's variance and autocorrelation do not depend on
which rows the target happened to be present on unless the absence is
informative — and its documentation should say so in one line: *under
`"shared"` a feature's moments are over every learned row; where a target
is absent on some rows its pairs use the feature as the other targets see
it.* A caller whose targets are absent on rows chosen by the feature keeps
`"per_target"`.

> **Examined 2026-09-25: `cov` is not exact where a target is absent.**
> The batch identity above holds; the recursion that keeps `sxy` does
> not follow it. `sxy` is stepped from the deviation against the
> feature's pre-row mean over the target's rows, and stepping it from the
> shared mean instead gives another `cov` wherever the target is absent
> on some learned rows: 1.8e-3 of it, relative, with the target absent on
> a random tenth of the rows at `lam = 0.99`, in a simulation of both
> recursions. It is bit-identical only where every target is present on
> every learned row. So under `"shared"`, `cov`, `var_x`, `corr`, `beta`,
> `t` and `n_serial` all differ there (`docs/PLAN.md` task 125).

### State and cost

`5·p·T` doubles become `3·p + p·T + 3·T` (`mx`, `mx_lo`, `sxx` per
feature; `sxy` per pair; `my`, `my_lo`, `syy` per target); the runs `p`
instead of `p·T`; under `lags`, `cxx` `L·p` instead of `L·p·T`. Per row:
the feature loop once, the pair loop with one multiply-add. The moments'
4.0 ns per pair-row become an estimated 1 ns at `T = 9`; with E70 and
E71 beside it, the row is roughly `10 + 5.5·T` ns per feature against
`19.3·T` today — 2.9× at `T = 9`, 3.4× at `T = 50`. Estimates, from the
op counts; the table in "Where the row goes" is what to hold them against.

### Invariance

Chunk invariance holds as today: every accumulator is still advanced
one learned row at a time by the same mean-form recursion. `"shared"`
against `"per_target"` on a stream where every target is present on every
learned row: bit for bit, since each shared accumulator performs exactly
the operation sequence each of the `T` copies performed. A new spec
parameter with a default that is today's behaviour: a saved state carries
the mode; a state saved before it loads as `"per_target"`. Whether the
`"shared"` layout is a new state variant or the same struct with `T = 1`
feature blocks is the implementer's call; either way the bank file's
format version need not move if the field is optional.

### Tests

- Every target on every row: `marginal()` under `"shared"` equals
  `"per_target"` bit for bit, lags and bins included.
- A target absent on a random tenth of the rows: `cov` bit-identical,
  `var_x` and `corr` within the documented difference (and equal to a
  from-scratch weighted computation over every learned row).
- Chunk invariance under `"shared"` at 1 and 1,000 chunks.
- `embargo`'s doubled stream (`test_label_delay_is_the_doubled_stream_here_too`,
  named for `embargo`'s name before task 144) under `"shared"`.
- `marg_bench` at `T = 9` and `T = 50` under both settings.

---

## E73 — a wide `marginal` across the pool

### Why

The pool's unit of work is a stream: one spec on one group (README,
*Parallelism*). The first pass has one group per block and one spec per class, so
a 10,000-feature spec is one thread's work per chunk while thirteen cores
wait, and eight worker processes were measured at 1.1× one — each worker
already had its few classes on a few threads, and eight of them
oversubscribed the machine. Splitting a class into 14 equal specs by hand
gives 6.5× on 14 threads (the table above), and it is what the caller is
about to do. It leaks: fourteen names for one screen, fourteen frames to
concatenate from `marginal()`, fourteen closed rows per block, fourteen
entries in the state. A `marginal`'s pairs are independent of each other,
so the split is the library's to make, invisibly.

### API

```python
po.spec.marginal("m", features=[...], targets=[...], shards="auto", **common)
```

- `shards`: `1` (today), an integer, or `"auto"` — the pool's size, or
  fewer when the spec is narrow (a shard under a few hundred features is
  not worth a task). One spec in every surface: one `marginal()` frame,
  one closed row, one state entry, one name.

### Mechanics

The model's feature range is split into contiguous shards; each shard's
per-row work — the pair loop, the lag update, the bins — is one task on
the pool, over the chunk's rows in order, the same recursion on its own
slice of the `[t][j]` arrays (`chunks_mut` over `p`, so no shard writes
another's cells). The per-target scalars (`W_t`, `Q_t`, `my`, `syy`,
`cyy`, the target's run) are advanced once, by whichever shard is first or
by the driver before the shards run, and read by all. The ring is shared
read-only within a row.

Only a model whose per-feature work is independent qualifies; `ew_cov`'s
`k × k` update does not, and an `ewridge`'s solve does not. `marginal` is
the one wide model, and the one this asks for.

### State and cost

No state, no per-row cost beyond the task dispatch per chunk, which
`process` already pays per stream. Memory unchanged.

### Invariance

Shard-invariant bit for bit — each pair's arithmetic touches only its own
cells and the row's shared scalars — so `shards = 1` and `shards = 14`
give identical `marginal()` frames and identical states, and a state
saved at one pool size loads at another (`shards` is a run setting, not a
state). Chunk invariance is untouched.

### Tests

- `marginal()` and the closed row identical at `shards = 1`, `3`, `14`,
  lags and bins on, on a multi-target stream with absent targets.
- A bank saved under `shards = 14` loads and continues under `shards = 1`
  to the same state.
- The thread-scaling table in PERFORMANCE.md §13 gains a `marginal`
  row at `p = 10,000`, one spec, one group.

---

## E74 — the row's fixed cost per feature: run a chunk pair-major

### Why

The `T`-scaling table has an intercept: 14–17 ns per feature per row
whatever `T` is, four to five times the moments' own pair step. At
`T = 1` — one target, the shape `marginal`'s docstring leads with — it is
80% of the row. The model is fed a row at a time (`step(x, y, ..)` with
`x: &[f64]` of `p` values), so per row the bank gathers one value from
each of `p` columns, and the model walks `p` cells of every `[t][j]`
array, loading and storing each pair's four or five doubles once per row:
at `p·T = 90,000` pairs that state is 3.6 MB, in and out of L2 on every
row, for eight flops each. Nothing in the pair's arithmetic needs the row
to be the outer loop: pair `(t, j)` depends on its own previous state and
on the row's per-target scalars (`a_t`, `b_t`, `dy_t`, the presence of
`y_t`), never on another pair.

> **Examined 2026-09-25: the fixed cost is not in the model.** The bank
> already casts and transposes each chunk once, in one tiled pass, and
> hands the model one contiguous row (`feature_rows` in
> `crates/online-polars/src/bank.rs`, `docs/PERFORMANCE.md` §20): there
> is no per-row gather. Measured on the model alone at `p = 10,000`,
> with no bank, frame or Python around it, the moments cost
> `0.0 + 3.3·T` ns per feature per row and the first pass's shape
> `3.5 + 19.8·T`. The slopes match the caller's `3.2·T` and `19.1·T`,
> but the 14–17 ns intercept is in the stream around the model, not in
> its loop order (`docs/PLAN.md` task 124). And a lag here is the ℓ-th
> previous *learned* row (weight above 0, every feature finite), not
> `x_j[r − ℓ]`: the caller's weight is 0 on its warm-up and purge rows
> (task 127).

### Mechanics

Within a chunk (or a run of a few thousand rows, the cache-sizing device
`process` already uses), form the per-target scalars once per row —
`W'_t`, `a_t`, `b_t`, `dy_t` against the target's mean as it advances,
which is `T` vectors of the run's length — then loop **feature outer, target
inner, row innermost**: each pair's state sits in registers for the run,
`x_j` is read contiguously from the column's own buffer (no gather, no
transpose — the Arrow column *is* the row-innermost layout), the bin
index of `x_j[r]` is formed once per `(j, r)` for every target (E71 for
free), and a lag is `x_j[r − ℓ]` inside the run with a per-feature ring of
`max(lags)` values for the run's head (today's ring, kept per feature
instead of per row). `x_runs` per `(t, j)` and `y_runs` per target run in
the same order they run today. The per-row output, `weight_sum`, is the
model's `w_sum` path and does not depend on the pairs.

### Invariance

Each pair's operation sequence over the rows is exactly today's — the
same `a`, `b`, `dx`, `dy` in the same order, since none of them depends
on any other pair — so the state and every reported column are
bit-identical, and chunk invariance is untouched by construction: a run
boundary is a row boundary, and the per-feature ring makes the lag at a
run's head what it is mid-run. No API, no state change.

### Cost

The per-feature intercept should fall to the cost of touching `x_j` once
per row (a fraction of a nanosecond), and the pair step to the flops
themselves. Whether that is 3× or 5× on the row at `T = 9` is for the
benchmark to say; the intercept alone is 9% of the row at `T = 9`, 33% at
`T = 3`, and 80% at `T = 1`. It composes with the four above: E72's
shared feature moments are the feature-outer loop's natural home, and
E73's shards are contiguous feature ranges of the same loop.

### Tests

- Bit-identical `marginal()` and closed rows against the row-major build
  on the multi-target, absent-target, lags-and-bins stream, at 1 and
  1,000 chunks.
- `marg_bench` at `T = 1` and `T = 9`: both rows should move, `T = 1` the
  most.

---

## What it is worth

Per feature per row, one thread, at the caller's settings (6 lags, 16
bins), from the fitted line `17.1 + 19.1·T` and op counts for the rest —
estimates past the first line, stated so the benchmark can hold them to
account:

| build | per feature per row | T = 9 | T = 50 | against today |
|---|---|---:|---:|---:|
| today, measured | 17.1 + 19.1·T | 189 ns | 972 ns | — |
| E70 (`cross_lags=[1]`: 8 lag updates per pair for 18) | 17.1 + 13.7·T | 140 | 702 | 1.35× |
| + E71 (bin once: the search per feature, Welford per pair) | 19 + 9.6·T | 105 | 499 | 1.8× / 1.9× |
| + E72 (shared feature moments: `sxy`, a cell, two cross lags per pair) | 26 + 3.7·T | 59 | 211 | 3.2× / 4.6× |
| + E74 (pair-major runs: the intercept and the state traffic gone) | ~2 + ~2·T | ~20 | ~100 | ~9× |
| + E73, 14 threads at the measured 6.5× | | ~3 | ~16 | ~60× |

Over a window (`1.35 × 10¹¹` feature-rows), today's first pass is
7.1 h of one thread at `T = 9` and 36 h at `T = 50`; with E70–E72, 2.2 h
and 7.9 h; with E74, under an hour and about four; the threads divide
that. The caller will take E73's share by hand in the meantime, so the
rows that matter to it are the first five.

## Order

E71 first: no API, no state, bit for bit, and a benchmark line to prove
it. E70 second: a parameter with today's default, and a state field with a
default. E72 third: a new mode, the largest saving of the three and the
only one that changes an estimator, so it wants its documentation line
and its tests before its code. E74 fourth, or first if the loop order is
going to change anyway — E71 and E72 fall out of it, and building them
row-major first means building them twice. E73 last, or never if E74
makes one thread enough: the caller can shard by hand.

## The caller's side

`probe_marginal_cost.py` and `probe_marginal_t.py`, the two scripts
behind the tables, are the caller's and live with it, not here; its own
tests and the closed-group prototype's bit-for-bit check are what it runs
against a build carrying any of the five.
