# Two extensions to `marginal`: lagged moments for an honest sample size, and binned target moments for a single-split gain

*2026-09-07. Two asks against 0.2.0, in the library's own terms:
`ENHANCEMENTS.md` §12, E66 and E67. Both extend `marginal` (E44) — the
model that keeps every (feature, target) pair's moments at `O(p·T)` per
row — with one more accumulator family each, keep its contract (no per-row
output but `n_eff`, pairs read back as a frame, closed-group rows, bit-level
chunk invariance), and reuse mechanics that already exist elsewhere in the
crate (`ew_cov`'s lag ring, `ewlagcov.rs`; the P² quantile machinery behind
`resid_quantiles`). Each section gives the motivation, the API, the
outputs in the field grammar, the state and cost, the invariance argument,
and the tests that pin it.*

---

## E66 — `marginal(lags=[...])`: lagged pair moments and a serial-dependence-honest `n`

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
                 lags=[1, 2, 5, 10, 20, 50],        # learned rows within the group, strictly increasing, ≥ 1
                 serial_n="geometric",              # None | "truncated" | "geometric"  (default None)
                 **common)
```

- `lags`: as on `ew_cov` — counted in learned rows within the group, not
  clock units; the ring is emptied on a session change and on a clock gap
  beyond `max_dclock`; a zero-weight row ages the moments without entering
  the ring. Refused with `label_delay` for the same reason `group_close`
  is (the ring would hold rows the delay has not released).
- `serial_n`: how `n_serial` is formed from the lags that were kept.
  `"truncated"` sums the kept lags as they are (a lower bound on the
  correction, an upper bound on `n_serial`). `"geometric"` fits `φ_x` and
  `φ_y` by least squares on `log ρ(ℓ)` over the kept lags where `ρ(ℓ) > 0`
  and extrapolates the tail in closed form — the right choice when both
  series are exponentially weighted, and the reason `lags` need not be
  dense. `None` keeps the lag moments and derives nothing.

### Outputs

Nothing per row. In `ModelBank.marginal()` and the closed-group row, per
pair, beside today's columns:

| column | meaning |
|---|---|
| `acf_x_l<ℓ>` | `ρ_x(ℓ)`: the feature's own lag-ℓ autocorrelation, `C_x(ℓ) / var_x` |
| `acf_y_l<ℓ>` | `ρ_y(ℓ)`: the target's (shared across the target's pairs, stored once per target) |
| `lagcorr_xy_l<ℓ>` | `corr(x_t, y_{t−ℓ})` — the feature *now* against the target ℓ rows *ago*: how much of the contemporaneous correlation is the feature reacting to the target's past (a nowcast) rather than leading it |
| `lagcorr_yx_l<ℓ>` | `corr(y_t, x_{t−ℓ})` — the target now against the feature ℓ rows ago: the lead. Both orientations, as `ew_cov`'s `lagcorr` gives them, because a lagged matrix is not symmetric |
| `n_serial` | `n_kish / [1 + 2 Σ_ℓ ρ_x(ℓ) ρ_y(ℓ)]` per `serial_n`; null when `serial_n=None` |
| `t_serial` | `corr · sqrt((n_serial − 2) / (1 − corr²))` |
| `phi_x`, `phi_y` | the fitted decays under `"geometric"`; null otherwise |

`t` stays as it is. The lead/lag pair (`lagcorr_yx` vs `lagcorr_xy`) is the
by-product worth having: for a forward-looking target, a feature whose
`lagcorr_xy_l1` exceeds its `corr` is one that *follows* the target — a
late-sampled column — and the caller can see it without a second pass.

### State and cost

Per target, `L` lagged target autocovariances (shared). Per pair, `L`
feature autocovariances and `2L` cross-covariances (both orientations).
So `(3L + 1)·p·T + L·T` doubles beside today's `5·p·T` — at `p = 10,000`,
`T = 3`, `L = 6` that is 570k doubles, under 5 MB. The ring is
`max(lags)` rows of the `p + T` columns per group: `50 × 10,003` doubles,
4 MB per group. Update cost `O(p·T·L)` per row (each lag is one
multiply-add per pair), about `L` times the current `marginal`, and still
linear in `p`.

Serialisation: the ring and the lag moments go into the state with the
pair moments (`ew_cov` already does this for its lags; same loader rule —
a pre-E66 state loads with no lags). Schema bump.

### Invariance

The ring is indexed by learned rows within the group, which is a function
of row order alone, so feeding one chunk or a thousand gives the same
moments to the bit — `ew_cov(lags=)`'s existing test, run again for
`marginal`. A closed-group row carries the lag moments as it carries the
pair moments; `po.gram.from_row` gains the new lists.

### Tests

1. `acf_x_l<ℓ>`, `acf_y_l<ℓ>` and `lagcorr_*_l<ℓ>` equal `ew_cov(lags=[ℓ],
   stats=["lagcorr"])` over the two columns of the pair, to the bit.
2. Chunk invariance: 1 chunk vs 1,000 chunks, bit for bit, with weights,
   nulls, zero-weight rows, a session change and a `max_dclock` gap.
3. Calibration: simulate `x_t`, `y_t` as independent AR(1) processes with
   known `φ_x`, `φ_y` (so the true correlation is 0), many replications;
   the empirical variance of `corr` across replications must match
   `1/n_serial` under `"geometric"` to within Monte-Carlo error, and `t_serial`
   must be N(0,1) where `t` is not (its SD is `sqrt(1 + 2φ_xφ_y/(1 − φ_xφ_y))`).
4. `"geometric"` recovers `φ_x`, `φ_y` to 1% on the AR(1) stream; refuses
   (null `n_serial`) when fewer than two kept lags have `ρ > 0`.
5. `lagcorr_xy_l1 > corr` on a stream where `x_t = y_{t−1} + noise`
   (the follower), and `lagcorr_yx_l1 > corr` where `y_t = x_{t−1} + noise`
   (the leader).
6. Save/load round-trip with the ring; a legacy state loads with
   `lags=[]`.

---

## E67 — `marginal(bins=...)`: binned target moments per feature, and the best single-split gain

### Why

Every statistic `marginal` reports is linear. A feature whose relation to
the target is a threshold, a V, or a saturation has a small `corr` and a
large *split gain*: the reduction in the target's variance from cutting the
feature at its best threshold — the number a regression stump reports, and
the first number a gradient-boosted tree looks at. Trees need rows; a
stump needs only, per feature, the target's `(Σw, Σwy, Σwy²)` inside each
of a fixed set of bins of the feature's value. That is a histogram of
target moments, `O(bins)` state per pair and `O(1)` update per pair per
row (one bin lookup, three adds), so the whole wide input gets a nonlinear
relevance number in the same pass that gives it `corr` — and the histogram
itself is the feature's one-dimensional response curve, readable as a
frame.

### API

```python
po.spec.marginal("m", features=[...], targets=[...],
                 bins=16,                  # int: how many bins, edges learned
                 bin_rule="quantile",      # "quantile" (equal counts) | "fixed" (equal widths)
                 bin_warm_rows=1_000,      # learned rows held before the edges are fixed
                 bin_edges={"x1": [...]},  # or: the edges outright, exact and reproducible
                 **common)
```

**Four names changed from the sketch above, and one mechanism.** The sketch
had `bins` doing double duty as both a count and a dict of edges, and
`bin_edges` naming the *rule*; a parameter called `bin_edges` should hold
edges. So: `bins` is the count, `bin_rule` is the rule, `bin_edges` is the
edges (a list per feature in `features` order, or a dict keyed by name), and
`bin_warmup` is `bin_warm_rows` — the repo says `rows` when it counts rows,
as `max_rows_between_solves` does. `gain_split` and `gain_split_t` became
`split_gain` and `split_gain_t`, since the thing is a split and the gain is
its property.

`bin_warm_rows` defaults to **1,000**, not 10,000. The sketch's own
arithmetic is the reason: 10,000 rows at 10,000 features is 800 MB of held
values, and a thousand rows already put sixty in each of sixteen quantile
bins. A hold that would exceed **256 MiB** is refused when the model is
built, with the number in the message, rather than discovered as an OOM
halfway through a stream.

**P² estimators are gone.** The warm-up rows have to be held anyway — that is
what makes the replay exact — so their quantiles can be read off a sort of
the held values directly. P² would add an approximation on top of data that
is already in hand, for no saving. `"fixed"` likewise uses the smallest and
largest value held, not P² tail estimates.

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
| `split_gain_t` | `√((n − 2)·g/(1 − g))`: the `t` a `corr` would need to match that gain, against `n_serial` where `serial_rule` gives one and `n_kish` otherwise. **Optimistic**, because the cut was chosen by maximising over `bins − 1` candidates and the statistic does not know that — a ranking, not a p-value |
| `bin_edges` | the fixed edges (list) |
| `bin_n`, `bin_mean_y`, `bin_var_y` | the histogram: the response curve `E[y | x ∈ bin]`, its weight and its dispersion |

The columns appear whenever the spec asked for bins, holding empty lists and
nulls until the edges are fixed. The lists are **ragged**: a feature keeps
only the bins it can support, so a binary feature has two whatever `bins`
says and a constant one has a single bin and no split. Refusing those would
be refusing real data, and an edge at or below the smallest value seen is
dropped because it can only ever open an empty bin.

### State and cost

`3·bins·p·T` doubles: at `p = 10,000`, `T = 3`, `bins = 16` that is 1.4M
doubles, 11 MB per group. Per row: one binary search over the edges
(`log bins`) and three adds per pair. The best split is computed on read
(`O(bins)` per pair), never per row.

Decay is `O(1)` per row rather than `O(bins)` per pair per row, which is what
makes the whole thing affordable. With `s_t` the product of every decay
factor so far, `λ^(t − t_i) = s_t/s_{t_i}`, so the sums are kept *undecayed*
against one scale per group: a row adds `w/s`, and a read multiplies by `s`.
Every ratio a caller wants — a bin mean, a variance, a gain — is scale-free
and does not even need that. When `s` falls below `1e-150` the sums are
multiplied through and `s` reset to 1, at a row that is a function of the
clock alone, so chunk invariance holds across it. The trick would serve any
per-row-decayed histogram; it is not specific to this model.

### Invariance

Explicit edges: trivially. Learned edges: the freeze happens at the
`bin_warm_rows`-th *learned* row in the group, the edges are a function of
the held values in order, and every held row is then replayed with its own
decay — so the histogram equals one built with the edges known in advance,
to the bit. A zero-weight row is held as decay only, folded into the next
held row, which keeps a run of them from growing the hold at the price of
floating-point association (not of chunk invariance, which is exact either
way).

### Tests

`crates/online-core/src/margbins.rs` (6), `crates/online-core/src/marginal.rs`
(5) and `tests/test_marginal_bins.py` (20):

1. The scale-factor histogram equals a brute-force one that decays every bin
   every row, bin for bin, mean for mean, variance for variance; and a run
   long enough to force a renormalization changes nothing readable.
2. `split_gain` equals a best-split search computed the long way over the
   same edges, both in Rust and against a polars `cut` + `group_by` in
   Python.
3. Learned edges give exactly the histogram those edges given up front give —
   the held rows are replayed, not spent.
4. Chunk invariance over 97 chunks, including where the edges landed.
5. Signal: a V has `corr ≈ 0` and a gain fifty times its `corr²`; a threshold
   is located to within half a bin; a straight line is found by both, with a
   monotone response curve.
6. Ragged shapes: binary, constant, and the refusals (`bins` with `window`,
   `bin_warm_rows` below `bins`, an unknown `bin_rule`, edges that are not
   increasing, a dict missing a feature, a hold over budget).

---

## Order

E66 first: it changes what `t` means for every wide, smooth input and
reuses code that exists. E67 second: new state, a new read path, and the
warm-up rule to get right. Both are additive to `marginal`'s contract;
neither touches another model. `po.gram.from_row` and the closed-group row
grow the new lists in both cases.
