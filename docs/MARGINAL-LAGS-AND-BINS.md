# Two extensions to `marginal`: lagged moments for an honest sample size, and binned target moments for a single-split gain

*2026-09-07. Two asks against 0.2.0, in the library's own terms:
`ENHANCEMENTS.md` §12, E66 and E67. Both extend `marginal` (E44) — the
model that keeps every (feature, target) pair's moments at `O(p·T)` per
row — with one more accumulator family each, keep its contract (no per-row
output but `n_eff`, pairs read back as a frame, closed-group rows, bit-level
chunk invariance), and reuse mechanics that already exist in the crate
(`ew_cov`'s lag ring, `ewlagcov.rs`; the Welford mean-form update every
model uses). Each section gives the motivation, the API, the outputs in the
field grammar, the state and cost, the invariance argument, and the tests
that pin it. Both shipped as PLAN tasks 65 and 66 and were reviewed and
revised under task 67; the sections below are what shipped, with the
sketch's departures noted where the review changed them.*

---

## E66 — `marginal(lags=[...])`: lagged pair moments and a serial-dependence-honest `n`

*Shipped 2026-09-07 as PLAN task 65 and revised under task 67. This section
is what shipped; the sketch it replaced had `serial_n`, per-lag columns and
a schema bump, and none of those survived contact with the repo's
conventions.*

### Why

`marginal` reports `t = corr · sqrt((n_kish − 2) / (1 − corr²))`. `n_kish`
is the right count for *unequal weights*; it says nothing about *serial
dependence*. On a stream where both the feature and the target are
smooth — an exponentially weighted feature against an exponentially
weighted forward target, sampled every second — consecutive rows are
nearly the same observation, and the variance of a sample correlation is
not `1/n` but, to first order (Bartlett 1935; the standard result for the
correlation of two stationary series),

```
Var(r) ≈ (1/n) · Σ_{ℓ=−∞}^{∞} ρ_x(ℓ) · ρ_y(ℓ)
       = (1/n) · [ 1 + 2 Σ_{ℓ≥1} ρ_x(ℓ) ρ_y(ℓ) ]
```

with `ρ_x(ℓ)`, `ρ_y(ℓ)` the two series' autocorrelations. The count that
makes `Var(r) = 1/n_serial` true is therefore

```
n_serial = n_kish / [ 1 + 2 Σ_{ℓ≥1} ρ_x(ℓ) ρ_y(ℓ) ]
```

For two exponentially weighted series with per-row decays `φ_x`, `φ_y`
(`ρ(ℓ) = φ^ℓ`) the sum is geometric: `1 + 2 φ_xφ_y / (1 − φ_xφ_y)`. At
halflives of 60 and 300 rows that factor is about 200 — a `t` of 20 on
`n_kish` is a `t` of 1.4. Nothing in the pair's contemporaneous moments
can see this; the lagged moments can, and the library already accumulates
exactly those for `ew_cov` (`lags=`, `ewlagcov.rs`): the same Welford
mean-form update one step further out, with a ring of the last `max(lags)`
learned rows per group.

### API

```python
po.spec.marginal("m", features=[...], targets=[...],
                 lags=[1, 2, 5, 10, 20, 50],   # learned rows within the group, strictly increasing, ≥ 1
                 serial_rule="geometric",     # None | "truncated" | "geometric"  (default None)
                 **common)
```

- `lags`: as on `ew_cov` — counted in learned rows within the group, not
  clock units; the ring is emptied on a session change and on a clock gap
  beyond `max_dclock`. A zero-weight row holds the moments (its mix is
  `a = 1, b = 0`) and does not enter the ring: it taught nothing, so it is
  not something a later row can be `ℓ` rows after. A row where a *target*
  is absent **is** pushed, since the ring is shared across targets, and
  that target's moments hold — see below for why they hold rather than
  decay.
- `serial_rule`: how `n_serial` is formed from the lags that were kept.
  `"truncated"` sums the kept lags as they are, and so misses whatever
  tail lies past `max(lags)` — for positively autocorrelated series a lower
  bound on the correction, an upper bound on `n_serial`. `"geometric"` fits
  `φ_x` and `φ_y` by least squares on `log ρ(ℓ)` over the kept lags where
  `ρ(ℓ) > 0` and extrapolates the tail in closed form — the right choice
  when both series are exponentially weighted, and the reason `lags` need
  not be dense. It says nothing (`n_serial` null) when fewer than two kept
  lags are positive for a series, or when `φ_xφ_y ≥ 1`, where the tail does
  not sum. `None` keeps the lag moments and derives nothing.

`label_delay` is **not** refused. The ring sits inside the model, downstream
of the delay buffer, and is fed rows in the order they are *learned* — the
order the delay releases them in — so the lag it counts is the lag `corr`
would count. `tests/test_marginal_bins.py::test_label_delay_is_the_doubled_stream_here_too`
holds a spec with lags and bins to `prep.embargo`'s doubled stream to the
bit.

**Names, against the sketch.** `serial_n` became `serial_rule`: it picks a
method, and `n_serial` is the output. `acf_x_l<ℓ>` became `lagcorr_xx`:
`ew_cov` already calls this statistic `lagcorr`, and the two orientations of
the cross term are `lagcorr_xy` and `lagcorr_yx` as they are there.

### Outputs

Nothing per row. In `ModelBank.marginal()`, per pair, beside today's
columns:

| column | meaning |
|---|---|
| `lagcorr_xx` | list, one entry per lag: `ρ_x(ℓ) = C_xx(ℓ) / (sd_x · sd_x)`, the feature's own autocorrelation |
| `lagcorr_yy` | `ρ_y(ℓ)`, the target's (accumulated once per target, reported on each of its pairs) |
| `lagcorr_xy` | `C(x_t, y_{t−ℓ}) / (sd_x · sd_y)` — the feature *now* against the target ℓ rows *ago*: how much of the contemporaneous correlation is the feature reacting to the target's past (a nowcast) rather than leading it |
| `lagcorr_yx` | `C(y_t, x_{t−ℓ}) / (sd_x · sd_y)` — the target now against the feature ℓ rows ago: the lead. Both orientations, because a lagged matrix is not symmetric |
| `n_serial` | `n_kish / [1 + 2 Σ_ℓ ρ_x(ℓ) ρ_y(ℓ)]` per `serial_rule`; null without one, and null where `"geometric"` cannot fit |
| `t_serial` | `corr · sqrt((n_serial − 2) / (1 − corr²))`, `±inf` where `t` is |
| `phi_x`, `phi_y` | the fitted decays under `"geometric"`; null otherwise |

Lists rather than `_l<ℓ>` columns, because `marginal()` is a frame with one
row per pair and its width must not depend on the spec (`ew_cov` emits
per-row struct fields, where a field per lag is the grammar). The
normalisation is `ew_cov`'s exactly — the lagged covariance over the two
contemporaneous standard deviations, in every orientation, and **not
clamped** to `[−1, 1]`: a lagged correlation is not bounded by one in finite
samples, clamping would hide that, and the serial correction guards itself.
So the eight numbers a pair and `ew_cov(lags=, stats=["lagcorr"])` report
for the same two columns agree to the bit, and a test holds them to it
through weights, a session change and a capped gap.

`t` stays as it is. The lead/lag pair (`lagcorr_yx` vs `lagcorr_xy`) is the
by-product worth having: for a forward-looking target, a feature whose
`lagcorr_xy[0]` exceeds its `corr` is one that *follows* the target — a
late-sampled column — and the caller can see it without a second pass.

### The recursion, and why a missing target holds

Per learned row, with `W` the target's weight before the row and `m` the
means before it — the operands the pair update uses, in the same
expressions, so lag 0 would agree with the contemporaneous co-moments to
the bit:

```
a = λ·W/W'      b = w/W'
C_ℓ' = a·C_ℓ + a·b·(v_t − m)(v_{t−ℓ} − m)     when row t−ℓ is in the ring
C_ℓ' = a·C_ℓ                                  when it is not yet
```

These are normalised moments, `E_w[·]`, and decay reaches them only through
`a` on the rows that learn: `W` carries the ageing, and `E_w` of a history
that has merely aged is the same number. The first version of `marglag.rs`
*also* multiplied them by `λ` on a row where the target was absent — decay
applied twice — and the review of task 65 found the consequence: at a
halflife of twenty, a hundred rows without the target took a `lagcorr_xx`
of 0.75 to 0.023 (`0.75 · 2⁻⁵` exactly) while `corr` and `var_x` stood
still, `n_serial` doubled, and `t_serial` inflated. At steady state every
lagged correlation was biased toward zero by about the fraction of rows the
target was missing. Invisible to the tests, which ran entirely at
`halflife=inf`. Now a missing target holds, as the pair moments hold, and
`tests/test_marginal_lags.py::test_lag_moments_hold_across_null_targets`
keeps it so at a halflife of twenty.

### State and cost

Per target, `L` lagged target autocovariances (shared). Per pair, `L`
feature autocovariances and `2L` cross-covariances (both orientations).
So `3L·p·T + L·T` doubles beside today's `5·p·T` — at `p = 10,000`,
`T = 3`, `L = 6` that is 540k doubles, under 5 MB. The ring is
`max(lags)` rows of the `p + T` values per group: `50 × 10,003` doubles,
4 MB per group; a full ring reuses the row it drops, so the steady state
allocates nothing per row. Update cost `O(p·T·L)` per row (each lag is one
multiply-add per pair), about `L` times the current `marginal`, and still
linear in `p`.

Serialisation: the ring and the lag moments go into the state beside the
pair moments, as an optional part with a `serde` default, so a state saved
without lags loads with none. **No schema bump.** The one rule this
imposes is the compact msgpack form's: a struct is an array, so at most one
field may `skip_serializing_if` and it must be last — E66's first draft
put a skip in front of the window's and a marginal with a window but no
lags stopped round-tripping compactly. `crates/online-core/tests/state_encoding.rs`
now round-trips `marginal` through both encodings with every optional part
present and absent.

### Invariance

The ring is indexed by learned rows within the group, which is a function
of row order alone, so feeding one chunk or a thousand gives the same
moments to the bit — held with every event that moves the ring or the
moments in one stream: decay, unequal weights with zeros, null targets, a
skipped row, a session change and a capped gap, in chunks of 1, 37 and 400
rows. A closed-group row carries the lag columns as lists of lists, one
inner list per pair, under the same NaN-to-null rule as `marginal()`; a
bank whose closing marginals did not ask has no such columns, and a row
whose spec did not ask has null in them.

### Tests

`crates/online-core/src/marginal.rs`:
`lagged_pair_moments_are_ew_covs_to_the_bit` (the recursion against
`EwLagCov`), `t_serial_is_standard_where_t_is_over_dispersed` (200
replications of two independent AR(1) series at φ 0.9/0.8: `sd(t)` near
the Bartlett inflation, `sd(t_serial)` near one, the fitted decays within
0.05 of the truth), `lag_moments_hold_where_the_target_is_missing`,
`state_round_trips_and_continues_identically`, `validation_rejects_each_bad_field`;
`crates/online-core/tests/state_encoding.rs`.

`tests/test_marginal_lags.py`: `t` claims significance that `t_serial`
does not (on that stream `t = 2.39`, `t_serial = 1.03`); the geometric fit
recovers the decays; the autocorrelations decay like the process; the
cross-correlations say which series leads; chunk invariance, plain and
through every event; the lagged pair is `ew_cov(lags=)` to the bit; the
ring clears on a capped gap and a session change and not on a gap under the
ceiling; the lag moments hold across null targets; a saved bank resumes
with its ring; without `lags` the frame is what it was; every bad lag spec
is refused by name. `tests/test_closed_groups.py`: the closed row carries
the lists, value for value, and the columns follow the specs that asked.

---

## E67 — `marginal(bins=...)`: binned target moments per feature, and the best single-split gain

### Why

Every statistic `marginal` reports is linear. A feature whose relation to
the target is a threshold, a V, or a saturation has a small `corr` and a
large *split gain*: the reduction in the target's variance from cutting the
feature at its best threshold — the number a regression stump reports, and
the first number a gradient-boosted tree looks at. Trees need rows; a
stump needs only, per feature, the target's weight, mean and centred second
moment inside each of a fixed set of bins of the feature's value. That is a
histogram of target moments, `O(bins)` state per pair and `O(1)` update per
pair per row (one bin lookup, one Welford step), so the whole wide input
gets a nonlinear relevance number in the same pass that gives it `corr` —
and the histogram itself is the feature's one-dimensional response curve,
readable as a frame.

### API

```python
po.spec.marginal("m", features=[...], targets=[...],
                 bins=16,                  # int: how many bins, edges learned
                 bin_rule="quantile",      # "quantile" (equal weight) | "fixed" (equal widths)
                 bin_warm_rows=1_000,      # learned rows held before the edges are fixed
                 bin_edges={"x1": [...]},  # or: the edges outright, exact and reproducible
                 **common)
```

**Four names changed from the sketch, and one mechanism.** The sketch had
`bins` doing double duty as both a count and a dict of edges, and
`bin_edges` naming the *rule*; a parameter called `bin_edges` should hold
edges. So: `bins` is the count, `bin_rule` is the rule, `bin_edges` is the
edges (a list per feature in `features` order, or a dict keyed by name), and
`bin_warmup` is `bin_warm_rows` — the repo says `rows` when it counts rows,
as `max_rows_between_solves` does. `gain_split` and `gain_split_t` became
`split_gain` and `split_gain_t`, since the thing is a split and the gain is
its property. `bin_edges` is one way of fixing the edges and the other three
knobs describe the other, so they are **refused beside it**, by name, rather
than silently ignored.

`bin_warm_rows` defaults to **1,000**, not 10,000. The sketch's own
arithmetic is the reason: 10,000 rows at 10,000 features is 800 MB of held
values, and a thousand rows already put sixty in each of sixteen quantile
bins. A hold that would exceed **256 MiB** is refused when the model is
built, with the number in the message, rather than discovered as an OOM
halfway through a stream — and so is a histogram that would, whether its
edges are learned or given: a long explicit edge list opens the same
allocation `bins` would, and the first draft budgeted only the learned
kind.

**P² estimators are gone.** The warm-up rows have to be held anyway — that is
what makes the replay exact — so their quantiles can be read off a sort of
the held values directly. P² would add an approximation on top of data that
is already in hand, for no saving. `"fixed"` likewise uses the smallest and
largest value held, not P² tail estimates.

**The quantile rule is weighted, and keeps a point mass in its own bin.**
Weighted by the row weights — what a row counts for — and not by their
decay, which says when a row arrived and nothing about what the feature
looks like. Plain quantiles fail on the commonest wide-input feature there
is: an indicator that is zero on 95% of rows has every quantile sitting on
zero, which collapses to no edge at all, and the 5% that carry the
information vanish into the same bin as the zeros. The edges are placed one
at a time, each closing a bin of the weight still unassigned divided among
the bins still to come; an edge that would close an empty bin — because the
value that crosses the share is the first value still unassigned — is moved
up to the first value above it, so the mass fills one bin and the bins that
remain share what is left. The rare-indicator test (one in twenty rows,
`y = 5x` plus noise) now gets an edge at 1, two bins and a gain above 0.9;
before, every quantile sat on zero and it got one bin and no split.

`bins` and `window` are **refused together**. A window works by subtracting
an old snapshot of the accumulators, and a snapshot of the histogram is
`bins` times the size of one — too much to keep per snapshot. `label_delay`
is *not* refused: the hold sits inside the model, downstream of the delay
buffer, and sees exactly the rows `learn` sees, so the pairing it bins is the
pairing `corr` uses.

### Outputs

Nothing per row. In `marginal()` and the closed-group row, per pair:

| column | meaning |
|---|---|
| `split_gain` | the fraction of the target's variance removed by the best single cut, `max_c (w_L·w_R/W²)·(ȳ_L − ȳ_R)²/var_y` over the `bins − 1` cuts. This is the best stump's `R²`, directly comparable with `corr²` for the same pair, so `split_gain − corr²` is the nonlinear surplus |
| `split_at` | the edge that achieves it, in the feature's units |
| `split_gain_t` | `√((n − 2)·g/(1 − g))`: the `t` a `corr` would need to match that gain, against `n_serial` where `serial_rule` gives one and `n_kish` otherwise; `+inf` at a gain of one, as `t` is at `corr = ±1`. **Optimistic**, because the cut was chosen by maximising over `bins − 1` candidates and the statistic does not know that — a ranking, not a p-value |
| `bin_edges` | the fixed edges (list) |
| `bin_n`, `bin_mean_y`, `bin_var_y` | the histogram: the response curve `E[y | x ∈ bin]`, its weight and its dispersion. A bin no row has landed in is *present* with `bin_n = 0` and null moments — a `"fixed"` bin over a gap in the feature's support is empty, not absent, and the lists stay aligned with `bin_edges` |

The columns appear whenever the spec asked for bins, holding empty lists and
nulls until the edges are fixed. The lists are **ragged**: a feature keeps
only the bins it can support, so a binary feature has two whatever `bins`
says and a constant one has a single bin and no split. Refusing those would
be refusing real data, and an edge at or below the smallest value seen is
dropped because it can only ever open an empty bin.

In the closed-group row the same columns ride as `pair_bin_edges`,
`pair_bin_n`, … — lists of lists, one inner list per pair — and
`pair_split_gain`, `pair_split_at`, `pair_split_gain_t` as lists, all under
`marginal()`'s null rule (NaN is null, `±inf` stays), so `pair_split_gain_t`
on a perfect split is `+inf` in both places. A bank none of whose closing
marginals asked has no such columns; a row whose spec did not ask has null
in them.

### State and cost

`3·bins·p·T` doubles: at `p = 10,000`, `T = 3`, `bins = 16` that is 1.4M
doubles, 11 MB per group. Per row: one binary search over the edges
(`log bins`) and one Welford step per pair. The best split is computed on
read (`O(bins)` per pair), never per row.

**Per-bin Welford, not raw sums.** The sketch kept `(Σw, Σwy, Σwy²)`; the
review measured what that loses. A target at an offset of `1e6` with unit
variance keeps about `1e12` in `Σwy²/Σw` and `mean²`, whose difference is
the variance — nine of sixteen digits gone before any decay. A target at
`1e8` reports a variance of zero or slightly negative. Each bin now carries
`(w, mean, M2)` and updates the way every other accumulator in the crate
does: `mean += δ·u/w'`, `M2 += u·δ·(y − mean')`. Same state size, one more
multiply per row, and the far-offset tests (a target of `1e6` in Rust, of
`1e7` with noise of `1e-3` in Python) recover a variance the sums could not
see.

Decay is `O(1)` per row rather than `O(bins)` per pair per row, which is what
makes the whole thing affordable. With `s_t` the product of every decay
factor so far, `λ^(t − t_i) = s_t/s_{t_i}`, so the weights are kept
*undecayed* against one scale per group: a row adds `w/s`, and a read
multiplies by `s`. Every ratio a caller wants — a bin mean, a variance, a
gain — is scale-free and does not even need that. When `s` would fall below
`1e-150` the weights and `M2` are multiplied through and `s` reset to 1, at a
row that is a function of the clock alone, so chunk invariance holds across
it. The trick would serve any per-row-decayed histogram; it is not specific
to this model. Three details the review added, each with a test that fails
without it:

- **A factor of zero.** A clock gap past `max_dclock` is capped there, and
  at `halflife=10`, `max_dclock=1e5` that cap is a `λ` of `2^-10000 = 0`
  exactly — reachable from any stream with one long pause. `s·0 = 0` would
  have every later row add `w/0 = ∞`. The fold handles it: a factor below the threshold is
  multiplied through, and a factor of zero wipes the histogram, which is what
  the pair moments do on the same row.
- **An underflowing fold.** The fold multiplies by `s` and `λ` separately,
  never by their product, since the product is what just proved too small to
  keep.
- **An empty histogram takes no decay.** There is nothing to age, and a scale
  picked up while empty would divide every later row's weight and multiply
  every later read, each by a rounding — `label_delay`'s doubled stream
  begins with a zero-weight prefix, and the pair moments carry no trace of
  one (their mix on such a row is `a = 1, b = 0`), so neither may the bins.
  This is what holds the doubled-stream test to the bit in the bin block.

The gain is formed as a product of ratios, `(d_L/w_L)·(d_L/w_R)/var`,
because just before a fold the undecayed weights run to `1e150` times the
real ones and three of them multiplied together overflow.

### Invariance

Explicit edges: trivially. Learned edges: the freeze happens at the
`bin_warm_rows`-th *learned* row in the group, the edges are a function of
the held values in order, and every held row is then replayed with its own
decay — so the histogram equals one built with the edges known in advance,
to the bit. A zero-weight row is held as decay only, folded into the next
held row, which keeps a run of them from growing the hold at the price of
floating-point association (not of chunk invariance, which is exact either
way). A row where a target is missing ages the histogram and adds to no
bin, as it ages that target's weight and holds its moments — the same rule
in both blocks, and `tests/test_marginal_bins.py::test_weights_and_null_targets_count_as_they_do_in_the_pair`
holds each bin's weight to the pair's under it.

### Tests

`crates/online-core/src/margbins.rs` (15): the scale-factor histogram equals
a brute-force one that decays every bin every row, bin for bin, mean for
mean, variance for variance; a run long enough to force a renormalization
changes nothing readable; a total gap empties it; the scale cannot underflow
to zero; decay leaves no trace on an empty histogram; a far-offset target
keeps its variance; `split_gain` equals a best-split search computed the
long way; the edge values and junk a row can carry; the shapes of edges it
refuses; explicit edges are budgeted too; no split without variance; and the
quantile rule — without ties, with a point mass, weighted, and on degenerate
samples.

`crates/online-core/src/marginal.rs` (8 of its 21): learned edges give
exactly the histogram those edges given up front give, held rows are
replayed and not spent; a zero-weight row in the warm-up teaches nothing; a
V is invisible to `corr` and obvious to a split; refusals before and beyond
what it can do; degenerate features keep the bins they can support; a
zero-weight row across a total gap ages the target's weight; a total gap
empties the histogram with the moments; a perfect split has an infinite
statistic.

`tests/test_marginal_bins.py` (26 functions): a threshold is found and
located to within half a bin; a V has `corr ≈ 0` and a gain fifty times
its `corr²`; a straight line is found by both, with a monotone response
curve; no relation gives a small gain and a calibrated statistic; the
response curve is the target's moments in each bin, against a polars `cut`
+ `group_by`; chunk invariance over 97 chunks, including where the edges
landed; explicit edges skip the warm-up and are reported back; held rows are
replayed, not dropped; nothing is reported until the edges exist; ragged
shapes; the fixed rule gives equal widths; `split_gain_t` uses `n_serial`
when it has one; every refusal by name, `bin_edges` beside the learned
kind's knobs included; the decayed histogram is the ew moments per bin; a
tiny halflife folds the scale many times and changes nothing; weights and
null targets count as they do in the pair; each target gets its own
histogram; a clock gap past `max_dclock` empties the histogram with the
moments; a bank saved during the warm-up resumes with its held rows;
`label_delay` is the doubled stream here too; a rare indicator keeps its own
bin and is split; quantile edges are weighted; a far-offset target keeps its
variance; a fixed bin no row lands in is empty, not absent; the columns are
there before any row and for a group never seen.

---

## Order

E66 first: it changes what `t` means for every wide, smooth input and
reuses code that exists. E67 second: new state, a new read path, and the
warm-up rule to get right. Both are additive to `marginal`'s contract;
neither touches another model, and neither bumps the state schema. The
closed-group row grows the new columns in both cases, as `pair_*` lists of
lists; `po.gram.from_row` reads the Gram block and is untouched.
