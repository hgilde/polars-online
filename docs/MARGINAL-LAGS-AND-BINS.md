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
                 bins=32,                                  # int, or {feature: [edge, ...]}
                 bin_edges="quantile",                     # "quantile" | "fixed"   (how an int is turned into edges)
                 bin_warmup=10_000,                        # learned rows over which quantile edges are set, then frozen
                 **common)
```

- `bins` as a dict gives explicit, fixed edges per feature (interior
  edges; the two outer bins are open). This is the exact form, and the
  one to use when a previous pass (`describe()`, or an earlier
  `marginal(bins=...)` read back) has provided the quantiles.
- `bins` as an int with `bin_edges="quantile"` sets each feature's edges
  to its P² quantile estimates after `bin_warmup` learned rows in the
  group, then **freezes** them. The rows of the warm-up are binned at that
  moment from the ring of warm-up values (they are held, not dropped), so
  no row is lost and the result is a function of row order alone, hence
  chunk-invariant. Before the freeze the histogram columns are null.
  `"fixed"` with an int uses equal-width bins between the P² 0.5% and
  99.5% points at the freeze, same rule.
- Decay: with `lam = 1` (the block use, one group per block) the bins are
  plain sums. With decay, each bin's three sums must scale by `λ^Δ` on
  every row, which is `O(bins)` per pair per row — too much at `p =
  10,000`. The implementation keeps the sums *undecayed* and one scale
  factor `s = λ^{t}` per group, adding `w/s`, `w·y/s`, `w·y²/s` to the
  bin; a read multiplies by `s`. `s` underflows after `~1,000` halflives;
  renormalise (multiply every sum by `s`, set `s = 1`) whenever `s <
  1e-150`, which is a deterministic function of the clock and so keeps
  chunk invariance. (The same trick would serve any per-row-decayed
  histogram; it is not specific to this model.)

### Outputs

Nothing per row. In `marginal()` and the closed-group row, per pair:

| column | meaning |
|---|---|
| `gain_split` | best single-split variance reduction as a fraction of `var_y`: `max_c [ (Σ_L w)(Σ_R w)/(Σw) · (ȳ_L − ȳ_R)² ] / (Σw · var_y)` over the `bins − 1` cut points `c`, with `Σ_L`, `Σ_R` the sums to the left and right of `c`. It is the `R²` of the best stump — comparable to `corr²` for the same pair, so `gain_split − corr²` is the nonlinear surplus |
| `split_at` | the edge that achieves it |
| `gain_split_t` | `gain_split` scaled to a rough test statistic, `(n_kish − 2) · gain_split / (1 − gain_split)` (the F-statistic of the stump at the chosen split; optimistic, since the split was chosen — the honest use is against the same statistic on null targets fed as extra target columns, which the caller can do today) |
| `bin_edges` | the frozen edges (list) |
| `bin_n`, `bin_mean_y`, `bin_var_y` | the histogram itself (lists of length `bins`): the response curve `E[y | x ∈ bin]` and its dispersion |

### State and cost

`3 · bins · p · T` doubles: at `p = 10,000`, `T = 3`, `bins = 32` that is
2.9M doubles, 23 MB per group; with `group_close` only the open block is
live. Per row: one binary search over the edges (`log bins`) and three
adds per pair — a few nanoseconds, comparable to the pair update itself.
The best split is computed on read (`O(bins)` per pair), never per row.

### Invariance

Fixed edges: trivially. Quantile edges: the freeze happens at learned row
`bin_warmup` in the group, the P² estimates are a deterministic function of
the rows in order, and the warm-up rows are binned from the held values at
the freeze — so the histogram after the freeze equals one built with the
edges known in advance. The held warm-up values are `bin_warmup × (p + T)`
doubles per group: at 10,000 rows and 10,003 columns, 800 MB — which is
why `bin_warmup` should be small (a few thousand rows suffice for 32
quantiles) and why the exact `bins={...}` form is preferred at width.
Refused with `label_delay`.

### Tests

1. Fixed edges: `bin_n`, `bin_mean_y`, `bin_var_y` equal a numpy
   `np.histogram`-style computation with the same edges and weights, to the
   bit; `gain_split` equals a brute-force best-split search over the same
   edges (`sklearn.tree.DecisionTreeRegressor(max_depth=1)` on the binned
   values gives the same cut).
2. Chunk invariance for both edge modes, with weights, nulls, zero-weight
   rows, sessions and gaps.
3. Decay: `lam < 1` with the scale-factor trick equals a from-scratch
   per-row-decayed histogram to rounding, across a renormalisation
   (`s < 1e-150` forced by a long stream).
4. Signal: on `y = |x| + noise`, `corr ≈ 0` and `gain_split` well above
   zero with `split_at ≈ 0`; on `y = x + noise`, `gain_split ≈ corr² ·
   (a known stump-efficiency constant, ~0.6 for Gaussian x)`.
5. Null: on `y` independent of `x`, `gain_split_t`'s distribution over
   many pairs matches its distribution on a shifted-target column fed as an
   extra target (the calibration the caller will use).
6. Save/load round-trip with the histogram, the edges and the scale factor;
   a legacy state loads with `bins=None`.

---

## Order

E66 first: it changes what `t` means for every wide, smooth input and
reuses code that exists. E67 second: new state, a new read path, and the
warm-up rule to get right. Both are additive to `marginal`'s contract;
neither touches another model. `po.gram.from_row` and the closed-group row
grow the new lists in both cases.
