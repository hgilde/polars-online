# Regimes: what the detectors find, measured

A regime detector makes a claim about a stream: that its correlation has
changed, how long the current regime has lasted, or which state a row
belongs to. A claim like that can only be checked on a stream whose answer
is written down. [`po.sim.regimes`](../README.md#data-whose-truth-is-known)
generates one from a seed and returns the truth beside the rows. This page
reports what the detectors find on such streams, and on Monte-Carlo draws
whose null distribution is published: where each is right, and what it
misses.

Tasks 45–56 (E54–E64, 2026-09-06) added five models whose output is a
claim about a stream: `deco`, `rcov`, `hmm`, `corrchange` and `bocpd`. This
page and its script were written as that batch closed, and measure all
five, with two remedies for [series that tick at their own
times](../README.md#series-that-tick-at-their-own-times): refresh-time
sampling and the lagged co-moments of `ew_cov(lags=...)`. `deco` and
`rcov` joined on 2026-09-27 (task 112), with `hmm` through a switch of
regime and the size study at ten times the draws, and `corrchange`'s
sequential detector on 2026-09-28 (task 114).

| section | the question | measured on |
|---|---|---|
| [0. The short version](#0-the-short-version) | what the experiments found, one paragraph each | |
| [1. Does `hmm` recover the stream that made it?](#1-does-hmm-recover-the-stream-that-made-it) | how many rows the filter puts in their true state, by how it was started, and how fast it follows a switch | `hmm` |
| [2. Is the monitor the size its paper says?](#2-is-the-monitor-the-size-its-paper-says) | how often the test rejects when nothing changed | `corrchange`, `kind="monitor"` |
| [3. And is it the power its paper says?](#3-and-is-it-the-power-its-paper-says) | how often it rejects a real break | `corrchange`, `kind="monitor"` |
| [4. Where the heavy tails go: `D-hat`](#4-where-the-heavy-tails-go-d-hat) | how well the statistic's denominator is estimated | `corrchange`, `kind="monitor"` |
| [5. Two changepoint detectors on the same break](#5-two-changepoint-detectors-on-the-same-break) | false alarms on a quiet stream, and the delay to a real break | `corrchange(kind="window")`, `bocpd` |
| [6. The Epps effect, and the two ways out](#6-the-epps-effect-and-the-two-ways-out) | what asynchrony does to a correlation, and what recovers it | `po.stream.refresh_time`, `ew_cov(lags=...)` |
| [7. `deco`: an equicorrelation that moves](#7-deco-an-equicorrelation-that-moves) | where `rho` settles, and how fast it follows a switch | `deco` |
| [8. `rcov`: three estimators against each block's truth](#8-rcov-three-estimators-against-each-blocks-truth) | each estimator's error under noise and asynchrony | `rcov` |
| [9. The sequential detector against its paper](#9-the-sequential-detector-against-its-paper) | how often a monitoring period ends in a flag when nothing changed | `corrchange`, `kind="sequential"` |
| [10. Running them](#10-running-them) | how to reproduce every number, and how long it takes | |

Every number comes from `scripts/regime_experiments.py`, run on 2026-09-27;
sections 2 to 4 and 9 on 2026-09-28, after task 114 put the monitor's
kernel on its paper's; section 8 on 2026-10-05, after task 158 read the
pre-averaging bias's θ from the window run and task 159 formed CKP's first
pre-averaged term, which moved two of its cells at the third decimal. Sections 1 to 4 end with a dated note that keeps
the figures they first reported, on 2026-09-06, and 2 to 4 one with the
figures before task 114.

## 0. The short version

**Seeded from the rows, `hmm` splits zero-mean rows by direction.** Its
default seeding is k-means over the rows, and two zero-mean states differ
in nothing k-means can see. On streams that hold a single regime, the
filter puts 57 % of the rows in the true state, counting the 500 it reports
nothing on before seeding. Given the covariances to start from, it puts
98 % there. On streams that switch, it holds each new state within a
median of 12 rows, where seeded from the rows it follows no switch at all
([§1](#1-does-hmm-recover-the-stream-that-made-it)).

**A prior that is too wide is the failure to watch for, and it is the same
failure in two models.** At 1.0, on returns whose variance is about 1.0,
`hmm`'s `precision_prior` halves every correlation. The share of rows in
the true state then falls from 98 % to 58 %. Set too large, `bocpd`'s
`prior_scale` makes every row unsurprising, so a real break is never found
([its README section](../README.md#bocpd--how-long-has-this-regime-lasted)).
Set both from the data's scale
([§1](#1-does-hmm-recover-the-stream-that-made-it)).

**`corrchange`'s monitor is close to its paper's tables.** Wied, Krämer and
Dehling (WKD) tabulate its size and power for "i.i.d. bivariate `t_5`
innovations", a phrase two distributions fit. At 20,000 draws a cell, the
size under the shared-scale reading is within 0.004 of their Table 1, and
under the independent reading 0.008 above it where `rho` is 0, so the
size points to the shared scale. The power is within 0.035 of their Table
2 under either. On Gaussian pairs the size runs from 0.041 to 0.047
against the nominal 0.05, the level `tests/test_corrchange.py` holds it
to
([§2](#2-is-the-monitor-the-size-its-paper-says),
[§3](#3-and-is-it-the-power-its-paper-says)).

**`D-hat`, the statistic's denominator, is exact on Gaussian pairs and
biased low on a tail-dependent `t_5`.** On Gaussian pairs it averages 0.750
against an asymptotic 0.750 at `T = 2000`. On the `t_5` it reads 18 % low
at `T = 500` and 9 % low at `T = 2000`, and its scatter grows with `T`
instead of shrinking. The monitor's size and power stay close to WKD's
even so ([§4](#4-where-the-heavy-tails-go-d-hat)).

**`corrchange`'s sequential detector is the size its paper says.** Wied
and Galeano tabulate the rate at which a monitoring period ends in a flag
when nothing changes, on GARCH pairs. Over about 1000 cycles a cell, on
their design, the detector here is within 0.023 of their Table 2 at
`boundary_gamma` of 0 and 0.25, and within 0.030 at 0.45. That is under
two standard errors of the difference in every cell. At 0.45 both read 0.12
to 0.18 against a nominal 0.05, as the paper warns
([§9](#9-the-sequential-detector-against-its-paper)).

**The two changepoint detectors are at opposite ends of one trade-off.** On
the same streams, `corrchange(kind="window")` finds a correlation break in a
median of 29 rows and flags 51 rows per 1000 quiet ones. `bocpd` takes 98
rows and raises 0.68 alarms per 1000. Neither is better: they answer
different questions ([§5](#5-two-changepoint-detectors-on-the-same-break)).

**Asynchrony leaves almost nothing of a correlation at the finest interval,
and both remedies recover most of it.** A true correlation of 0.8 reads as
0.095 from one-row returns. Refresh-time sampling recovers 0.54 from 19157
returns, 16 % of the 119991 at the finest interval. The lag inversion of
§6, on `ew_cov(lags=...)`, recovers 0.77 while reading every return, and levels
off there over lags from 192 to 512 rows
([§6](#6-the-epps-effect-and-the-two-ways-out)).

**`deco`'s level settles near two-thirds of the true equicorrelation, and
moves at its half-life.** The row's estimate is biased low, as its paper
warns, and the level is an EW mean of it
([§7](#7-deco-an-equicorrelation-that-moves)).

**Each of `rcov`'s estimators wins somewhere.** `plain` is the most
accurate on clean returns, the kernel under noise, and pre-averaging when
the series are observed on different rows
([§8](#8-rcov-three-estimators-against-each-blocks-truth)).

## 1. Does `hmm` recover the stream that made it?

`hmm` gives each row a probability for each of `k` hidden states. This
section asks how many rows the filter puts in the state that generated
them, for four ways of starting it.

**No stream in this table switched regime; [the switch
experiment](#through-a-switch) makes them switch.**
`po.sim.regimes` draws one state per block, and with ten blocks at a 1 %
chance per block of leaving state 0, every block of all eight seeds stayed
there. The table below therefore measures the share of rows the filter
puts in that one true state.

| condition | value |
|---|---|
| streams | 8 seeds of `po.sim.regimes`: two series, 10 blocks of 400 rows, 4000 rows each, differenced into 3999 returns |
| states drawn | correlation 0.1 and 0.8, under the sticky transition matrix `[[0.99, 0.01], [0.02, 0.98]]` |
| states realised | state 0, at correlation 0.1, in every block of every seed |
| the filter | `k = 2`, `half_life = 4000`, and `precision_prior = 0.01` except in the last row |
| seeded from the rows | `warm_rows = 500` and `transition_prior = 200` |
| started from the truth | the true transition matrix, zero means, and covariances built from the true correlations and each seed's sample variances |
| the score | the share of rows in state 0, read as `filtered_1` at most 0.5, with a row that has no `filtered_1` counted there. The script takes the better of the two ways to match states to the truth, which here is this share |

| how it was started | mean share in the true state | worst seed |
|---|---|---|
| seeded from the rows (`warm_rows`), learning | 0.568 | 0.537 |
| true covariances, filter only | 0.979 | 0.966 |
| true covariances, and go on learning | 0.948 | 0.897 |
| true covariances, filter only, `precision_prior` = 1.0 | 0.580 | 0.521 |

**Seeded from the rows, the filter splits zero-mean rows by direction.**
The default seeding is k-means over the rows, and these returns have a
mean of zero, so k-means finds two directions. The learned state means
reach 0.64 in absolute value, where the truth is zero; that is the largest
absolute mean per seed, averaged over the eight. The score counts the 500
rows before seeding, which report nothing, as the true state. Over the 3499
rows after seeding the share is about half: (0.568 × 3999 − 500) / 3499 ≈
0.51. So a regime that lives in the covariance needs the covariances to
start from, or a feature in which the regime is a shift in location.

**A ridge as large as the data's variance halves every correlation.** In
the last row the filter does not learn, so `precision_prior` adds a ridge
of 1.0 to each given covariance for the whole stream. The returns' variance
is about 1.0, so each state's correlation is halved, and the correlation is
the whole of the signal here. The share in the true state falls from 0.979
to 0.580. That is `bocpd`'s `prior_scale` lesson ([its README
section](../README.md#bocpd--how-long-has-this-regime-lasted)), in a
second model.

The learned transition matrix is not read back through the public API, so
what is scored is the state path, which is what a caller reads. The
matrix's decayed counts are in the state that `ModelBank.to_json()`
exports, but no method returns the matrix.

> **Measured 2026-09-06:** the seeded row read 0.559, or 56 % (worst seed
> 0.507), with the learned state means at 0.65. The other three rows read
> as they do now.

### Through a switch

**Started from the true covariances, the filter follows each switch within
a few dozen rows.** `durations=[3, 3]` makes each state last exactly three
blocks, so each stream alternates between correlation 0.1 and 0.8 and
switches three times. A switch counts as followed once the filter holds the
new state for 20 rows running. That excludes a filter that flips at random,
which agrees with either state within a row or two.

| condition | value |
|---|---|
| streams | 8 seeds of `po.sim.regimes` with `durations=[3, 3]`: two series, 12 blocks of 400 rows, 4799 returns each, switching at rows 1200, 2400 and 3600, 24 switches in all |
| the filter | as above, but the true start's transition is the per-row hazard `1/1200` |
| the score | the share of rows in the true state, a row with no `filtered_1` counted as a miss; and the rows after each switch until the filter holds the new state for 20 rows |

| how it was started | mean share in the true state | worst seed | median rows to hold the new state | longest | switches never held |
|---|---|---|---|---|---|
| seeded from the rows (`warm_rows`), learning | 0.457 | 0.450 | - | - | 24 of 24 |
| true covariances, filter only | 0.992 | 0.989 | 12 | 32 | 0 of 24 |
| true covariances, and go on learning | 0.972 | 0.898 | 10 | 294 | 0 of 24 |

**Seeded from the rows, it follows no switch.** It never holds either
state for 20 rows running, because it splits the rows by direction, as in
the table above. Its share is below a half because here the 500 rows before
seeding count as misses.

**Learning is slower on one switch in 24.** With learning on, the median
switch is held within 10 rows, but one took 294, and that seed's share is
0.898. Without learning, no switch took more than 32 rows, and no seed's
share fell below 0.989.

## 2. Is the monitor the size its paper says?

`kind="monitor"` is Wied, Krämer and Dehling's closed-sample constancy
test: were the correlations constant over a span of rows? A published test
comes with a published null, and its size is the rate at which it rejects
when the correlation never changes. WKD's Table 1 gives that rate for
"i.i.d. bivariate `t_5` innovations". The phrase does not say whether the
two coordinates share a scale, and the two readings differ in their tails,
so both are measured, with Gaussian pairs beside them:

| draw | how it is made | its tails |
|---|---|---|
| t5 shared scale | a Gaussian pair divided by one chi-square scale per row, shared by both coordinates, which leaves the correlation at `rho` | tail-dependent: the extremes arrive in both coordinates at once |
| t5 independent | the same pair, each coordinate divided by its own scale | the same marginals, with no tail dependence |
| Gaussian | the pair itself | |

| T    | rho  | WKD Table 1 | t5 shared scale | t5 independent | Gaussian |
|------|------|-------------|-----------------|----------------|----------|
| 500  | -0.5 | 0.040       | 0.041           | 0.043          | 0.045    |
| 500  | 0.0  | 0.035       | 0.031           | 0.043          | 0.041    |
| 500  | 0.5  | 0.041       | 0.040           | 0.040          | 0.046    |
| 1000 | -0.5 | 0.038       | 0.035           | 0.041          | 0.046    |
| 1000 | 0.0  | 0.034       | 0.034           | 0.042          | 0.042    |
| 1000 | 0.5  | 0.039       | 0.039           | 0.040          | 0.047    |

The table holds 20,000 replications per cell, at a nominal level of 0.05,
where a rate from 20,000 draws has a standard error of 0.0015. Each draw is
one span of `T` rows, tested once with `alpha_adjust="none"`.

**Under the shared-scale reading, the size is within 0.004 of WKD's
Table 1 in every cell.** The largest gap, at `T = 500` and `rho = 0`, is
2.7 of this study's standard errors, and WKD's own figures carry a Monte
Carlo error this study cannot see. Neither reading is liberal: no `t_5`
cell is above 0.043.

**The independent reading does not match at `rho = 0`.** There it rejects
0.043 and 0.042, 0.008 above WKD's figures at both lengths, 5.3 standard
errors. Elsewhere it is within 0.003. So WKD's "bivariate `t_5`"
is the shared-scale draw, a multivariate `t`, as far as the size can say.

**On Gaussian pairs the size is 0.041 to 0.047**, a little under the
nominal 0.05. `tests/test_corrchange.py` holds the test to that level on
Gaussian pairs at `T = 500`, for `rho` of 0 and 0.5, and again at 0.5 with
one column multiplied by 100. No test holds the size to WKD's own figure.

> **Measured 2026-09-27, before task 114 put the kernel on WKD's `1 −
> l/γ` (it was Newey–West's `1 − l/(γ+1)`), at 20,000 replications a
> cell.** In the table's row order, the shared-scale `t_5` read 0.040,
> 0.031, 0.040, 0.035, 0.033 and 0.038; the independent `t_5` 0.042, 0.042,
> 0.040, 0.041, 0.042 and 0.039; Gaussian pairs 0.045, 0.041, 0.046, 0.046,
> 0.042 and 0.046. No cell moved by more than 0.001.
>
> **Measured 2026-09-23, at 2000 replications a cell.** In the table's row
> order, the shared-scale `t_5` read 0.045, 0.024, 0.043, 0.033, 0.035 and
> 0.044; the independent `t_5` 0.046, 0.044, 0.041, 0.036, 0.041 and 0.041;
> Gaussian pairs 0.046, 0.045, 0.052, 0.043, 0.034 and 0.048. The shared
> scale's 0.024 at `T = 500` and `rho = 0`, 2.2 standard errors from WKD's
> 0.035, reads 0.031 at ten times the draws.

> **Measured 2026-09-06, before the S3 fix of 2026-09-19, which corrected
> `D-hat`'s gradient** ([docs/REVIEW-2026-09-18.md](REVIEW-2026-09-18.md)).
> In the table's row order, the shared-scale `t_5` read 0.075, 0.025,
> 0.076, 0.071, 0.040 and 0.081. The independent `t_5` read 0.028, 0.046,
> 0.025, 0.018, 0.040 and 0.021. Gaussian pairs read 0.048, 0.045, 0.054,
> 0.042, 0.034 and 0.051.

## 3. And is it the power its paper says?

Power is the rate at which the test rejects when the correlation does
change. WKD's Table 2 gives it for a break from 0.5 to 0.7 at the middle of
the span. The same three draws are measured, each against two critical
values:

| column | the draw is rejected when its statistic exceeds |
|---|---|
| power | the asymptotic critical value, 1.3581, which the model uses |
| size-adjusted | the empirical 95 % quantile of 1000 draws of the same distribution and length, with the correlation held at 0.5 |

| T    | innovations | power | size-adjusted | empirical 95% | WKD Table 2 |
|------|-------------|-------|---------------|---------------|-------------|
| 500  | t5          | 0.552 | 0.582         | 1.324         | 0.587       |
| 500  | t5_indep    | 0.588 | 0.622         | 1.317         |             |
| 500  | normal      | 0.833 | 0.846         | 1.337         |             |
| 1000 | t5          | 0.830 | 0.848         | 1.321         | 0.830       |
| 1000 | t5_indep    | 0.831 | 0.856         | 1.317         |             |
| 1000 | normal      | 0.995 | 0.995         | 1.352         |             |

The table holds 1000 replications per row, at a nominal level of 0.05.

**Under both `t_5` readings the power is within 0.035 of WKD's Table 2.**
At `T = 500`, where WKD give 0.587, the shared-scale draw reaches 0.552
against the asymptotic value and 0.582 size-adjusted, and the independent
draw 0.588 and 0.622. At `T = 1000` the two draws reach 0.830 and 0.831
against WKD's 0.830, and 0.848 and 0.856 size-adjusted.

**On Gaussian pairs the power is well above WKD's figure:** 0.833 at
`T = 500`, or 0.846 size-adjusted, and 0.995 at `T = 1000`.
`tests/test_corrchange.py` holds the Gaussian power at `T = 500` above a
floor set from WKD's 0.587, less a tolerance.

**Read the size-adjusted column across distributions.** It takes its
critical value from null draws of the same distribution and length, so a
size that differs between distributions does not tilt it. Every empirical
quantile, from 1.317 to 1.352, is below the asymptotic 1.3581. So the
size-adjusted power is the higher of the two in every row but the last,
where both are 0.995.

> **Measured 2026-09-27, before task 114's kernel.** In the table's row
> order, as power, size-adjusted and empirical 95 %: at `T = 500`, t5 read
> 0.553, 0.582 and 1.321, t5_indep 0.585, 0.613 and 1.327, and normal 0.830,
> 0.851 and 1.327. At `T = 1000`, t5 read 0.832, 0.844 and 1.324, t5_indep
> 0.832, 0.857 and 1.307, and normal 0.994, 0.994 and 1.355.
>
> **Measured 2026-09-06, before the S3 fix of 2026-09-19, which corrected
> `D-hat`'s gradient.** In the table's row order, as power, size-adjusted
> and empirical 95 %: at `T = 500`, t5 read 0.525, 0.428 and 1.498,
> t5_indep 0.400, 0.472 and 1.256, and normal 0.824, 0.837 and 1.335. At
> `T = 1000`, t5 read 0.779, 0.733 and 1.452, t5_indep 0.648, 0.750 and
> 1.195, and normal 0.994, 0.994 and 1.358.

## 4. Where the heavy tails go: `D-hat`

The monitor's statistic, `Q`, is
`max_j (j/sqrt(T)) |rho_j - rho_T| / D-hat`, where `rho_j` is the
correlation of the span's first `j` rows. `D-hat`, the denominator, is
the delta-method long-run standard deviation of `sqrt(T)*rho-hat`. For an
elliptical distribution, the value it estimates is known in closed form:
`(1 - rho^2) sqrt(1 + kappa)`, with `kappa = 2/(nu - 4)`. At `rho` = 0.5
that is 0.750 for a Gaussian pair, and 1.299 for a `t_5`, whose `kappa`
is 2.

The experiment recovers `D-hat` from each draw as the numerator, computed
longhand, divided by the reported `Q`, so it is the estimate the model
actually used.

| innovations | T    | mean  | median | sd    | asymptotic |
|-------------|------|-------|--------|-------|------------|
| normal      | 500  | 0.741 | 0.743  | 0.058 | 0.750      |
| normal      | 2000 | 0.750 | 0.750  | 0.031 | 0.750      |
| t5          | 500  | 1.071 | 1.016  | 0.255 | 1.299      |
| t5          | 2000 | 1.185 | 1.099  | 0.353 | 1.299      |
| t5_indep    | 500  | 0.812 | 0.795  | 0.111 | -          |
| t5_indep    | 2000 | 0.828 | 0.816  | 0.072 | -          |

The table holds 200 replications per row, at `rho` = 0.5. The independent
`t_5` has no closed form here, so its last column reads `-`.

**On Gaussian pairs the estimator is right and tight.** It averages 0.741
at `T = 500` and 0.750 at `T = 2000`, against 0.750. Its standard deviation
falls from 0.058 to about 0.03 (0.031) as `T` grows fourfold. That is what
the delta method promises on data with light tails: right, consistent and
quick.

**On the shared-scale `t_5` it is biased low, and its scatter does not
shrink.** Its mean is 1.071 at `T = 500` and 1.185 at `T = 2000`, against
1.299: 18 % and 9 % low. Its median is lower still, 22 % and 15 % low, and
its standard deviation grows from 0.255 to 0.353. The estimator's inputs
are fourth moments, and a `t_5` has a fourth moment only just, at
`nu > 4`. Those inputs' own variance is infinite, so there is no rate at
which the estimate settles. The independent `t_5` is steadier: its
standard deviation falls from 0.111 to 0.072.

The size in §2 and the power in §3 are close to WKD's tables even so.

> **Measured 2026-09-27, before task 114's kernel.** In the table's row
> order, as mean, median and sd, normal read 0.739, 0.742 and 0.060, then
> 0.749, 0.751 and 0.033. The shared-scale t5 read 1.071, 1.020 and 0.257, then
> 1.184, 1.097 and 0.353; the independent t5 0.811, 0.796 and 0.113, then
> 0.828, 0.815 and 0.073.
>
> **Measured 2026-09-06, before the S3 fix of 2026-09-19, which corrected
> `D-hat`'s gradient.** In the table's row order, as mean, median and sd:
> normal read 0.739, 0.739 and 0.063, then 0.750, 0.750 and 0.034. The
> shared-scale t5 read 1.050, 0.979 and 0.446, then 1.143, 1.032 and 0.464.
> The independent t5 read 0.932, 0.877 and 0.216, then 0.953, 0.915 and
> 0.156. The shared-scale `t_5` then read about 15 % low, with an sd 40 %
> of its level, and a denominator 15 % too small inflates `Q` by about
> 18 %.

## 5. Two changepoint detectors on the same break

Both detectors read the same 20 pairs of streams from `po.sim.regimes`,
each stream three series of 2000 rows. In each pair, one stream keeps a
correlation of 0.2 throughout, and the other breaks from 0.2 to 0.7 at row
1000. A false alarm is an alarm on the quiet stream, and the delay is the
number of rows from row 1000 to the first alarm at or after it.

| detector | settings | what counts as its alarm |
|---|---|---|
| `corrchange(kind="window")` | `kind="window"`, `span_rows=100`, `n_perm=100`, `permute_every=100`, `alpha=0.05`, `seed=0` | `flag`: the change between two adjacent 100-row windows is above a permutation quantile |
| `bocpd` | `emission="gaussian"`, `hazard=500`, `prior_nu=5`, `prior_scale=[0.1]`, `prune_below=1e-8`, `max_run=1200` | its own alarm, `p_change`, cannot see this break, so here: the row that `t - run_mode` names as the current run's start jumps forward by more than 50 |

**`bocpd` is measured with `emission="gaussian"`, not its default
`"diag"`.** The default models each feature on its own, and only the
Gaussian emission can see a break in the correlation alone ([its README
section](../README.md#bocpd--how-long-has-this-regime-lasted)).

| detector           | false alarms / 1000 quiet rows | found the break | median delay | worst delay |
|--------------------|--------------------------------|-----------------|--------------|-------------|
| corrchange, `kind="window"` | 51.33                          | 20/20           | 29           | 53          |
| bocpd              | 0.68                           | 20/20           | 98           | 351         |

**The two detectors answer different questions, which is why both are
here.** `corrchange(kind="window")` is a test of one thing: whether the
correlation matrix has moved between two adjacent windows. It is looking
straight at this break, so it flags it fast, in a median of 29 rows and 53
at worst. It also flags 51.33 rows per 1000 quiet ones. Two windows that
slide by one row are almost the same windows, so its flags come in runs.
That rate is nearer a fraction of rows above the quantile than a count of
events.

`bocpd` is a full joint model of the rows, and a correlation change is the
one break its `p_change` cannot see: no single row is surprising, only the
sequence is. Its run-length posterior has to accumulate that evidence,
which takes a median of 98 rows and 351 at worst. In return it dates the
regime, saying which row the current one began on. And it almost never
says so when nothing has happened: 0.68 alarms per 1000 quiet rows.

## 6. The Epps effect, and the two ways out

The Epps effect is the fall of a measured correlation toward zero as its
returns are taken over shorter intervals, when the series tick at their
own times. The stream here has two series with a true correlation of 0.8,
over 120,000 rows, each observed at an expected rate of 0.25 per row
(`async_rates`). Previous-tick sampling carries each series' last value
forward and takes returns every so many rows. At the finest interval that
is the naive thing to do, and it reads 0.095:

| sampling interval | returns | correlation |
|-------------------|---------|-------------|
| 1                 | 119991  | 0.095       |
| 2                 | 59995   | 0.172       |
| 5                 | 23998   | 0.346       |
| 20                | 5999    | 0.634       |
| 100               | 1199    | 0.752       |
| 500               | 239     | 0.745       |

The truth is 0.8. Two remedies recover most of it, and they differ in what
they read:

| remedy | what it reads | correlation |
|---|---|---|
| refresh-time sampling, `po.stream.refresh_time` | a grid point at the first row by which both series have ticked since the last point: 19157 returns, 16 % of the 119991 at the finest interval. On average a grid point kept 81 % of the ticks since the one before | 0.543 |
| lag inversion, `ew_cov(lags=...)` read through `po.corr.epps_invert` | every one of the 119991 fine returns, with no grid: the correlation at a scale of `L` rows, from the lagged co-moments at one row | 0.772 at `L` = 192 |

The lag inversion is read at 15 values of `L`, on this stream and on four
more from the next seeds, so its noise is printed beside it:

| L    | seed 7 | mean of 5 seeds | sd across seeds |
|------|--------|-----------------|-----------------|
| 1    | 0.095  | 0.101           | 0.004           |
| 2    | 0.170  | 0.178           | 0.005           |
| 4    | 0.294  | 0.301           | 0.005           |
| 8    | 0.456  | 0.457           | 0.003           |
| 16   | 0.608  | 0.602           | 0.004           |
| 32   | 0.699  | 0.697           | 0.006           |
| 64   | 0.742  | 0.744           | 0.005           |
| 96   | 0.757  | 0.758           | 0.003           |
| 128  | 0.765  | 0.767           | 0.001           |
| 192  | 0.772  | 0.774           | 0.004           |
| 256  | 0.770  | 0.775           | 0.008           |
| 384  | 0.766  | 0.775           | 0.015           |
| 512  | 0.765  | 0.774           | 0.019           |
| 768  | 0.760  | 0.771           | 0.026           |
| 1024 | 0.756  | 0.769           | 0.029           |

**`L` is the same trade-off the sampling interval is, and the sweep now
passes its best point.** Too small, and the cross-covariance at longer
lags is left out; too large, and the sum takes in noise. The mean over five
seeds levels off at 0.775 from `L` = 256 to 512 and falls after, while its
spread across seeds grows from 0.001 at 128 to 0.029 at 1024. Seed 7 peaks
earlier, at 192. Anywhere from 128 to 512 reads within 0.01 of the best,
and the best is still 0.025 short of the truth.

The lag inversion is here because its lagged co-moments are a state a bank
already keeps, as `ew_cov(lags=...)`. What the lags do to `ew_cov`'s
throughput is measured in [docs/PERFORMANCE.md](PERFORMANCE.md) §15.

## 7. `deco`: an equicorrelation that moves

`deco` reduces a correlation matrix to one number, the average of its
off-diagonal, and follows it with an EW mean of each row's own estimate,
`u`. The stream here has five series whose equicorrelation alternates
between 0.2 and 0.6.

| condition | value |
|---|---|
| streams | 8 seeds of `po.sim.regimes`: m = 5, `states=[0.2, 0.6]`, `durations=[3, 3]`, 12 blocks of 400 rows, 4799 returns each, 24 switches in all |
| the model | `po.spec.deco` with the default `dynamics="ew"`, at half-lives of 25, 100 and 400 rows |
| settled | `rho` averaged over the last 200 rows of each stretch of one state |
| rows to the midpoint | the rows after a switch until `rho` passes the midpoint of the two settled levels |

| half-life | `rho` settled, true 0.2 | `rho` settled, true 0.6 | median rows to the midpoint | range | switches never crossed |
|---|---|---|---|---|---|
| 25 | 0.129 | 0.377 | 26 | 8–58 | 0 of 24 |
| 100 | 0.130 | 0.381 | 100 | 66–172 | 0 of 24 |
| 400 | 0.146 | 0.348 | 330 | 246–520 | 0 of 24 |

**`rho` settles near two-thirds of the truth, because `u` does.** The
row's estimate `u` averages 0.130 where the truth is 0.2, and 0.379 where
it is 0.6. The spec's docstring warns of this, after Engle and Kelly: `u`
is a ratio, and its mean is not the ratio of the means. `rho` is an EW mean
of `u`, so it carries the same bias. Read `rho` as a signal that moves with
the correlation, not as the correlation.

**It moves at its half-life.** An EW mean reaches the midpoint of a step in
one half-life, and `rho` does: a median of 26 rows at a half-life of 25, and
100 at a half-life of 100. At 400 the median is 330. A state lasts 1200
rows, three of those half-lives, so `rho` has not settled when the next
switch comes. Its two levels, 0.146 and 0.348, are closer together than at
the shorter half-lives, and it starts nearer their midpoint.

## 8. `rcov`: three estimators against each block's truth

`rcov` estimates each block's covariance from its returns, at the block's
close. Its three estimators differ in what they survive: noise on the
observed prices, and series that are not observed on every row. The stream
here has 20 blocks of 2000 rows, each block's correlation drawn from 0.3
and 0.8, under four conditions:

| condition | what changes |
|---|---|
| clean | nothing: the returns are the latent ones |
| noise | noise with a standard deviation of 0.5 on each observed level (`noise=0.5`), which makes each return an MA(1) with a negative first autocorrelation, −0.17 here |
| noise, half observed, previous tick | the noise, and each series observed on half the rows (`async_rates=[0.5, 0.5]`), its last value carried forward within the block |
| noise, half observed, refresh time | the same rows through `po.stream.refresh_time`, block by block |

Each cell is the bias of the block's correlation, its standard error, and
the root-mean-square error, over 190 blocks: 10 seeds of 20 blocks, less
the last block of each stream, which never closes. `block_rows`, which
sizes the kernel's ring and pre-averaging's window, is each condition's
own median rows a block: about 2000, and 602 under refresh time, which
keeps only the rows where both series have moved.

| condition | `plain` | `kernel` | `preavg` | `preavg`, `psd=False` |
|---|---|---|---|---|
| clean | −0.001 ± 0.001 / 0.016 | −0.004 ± 0.003 / 0.042 | −0.013 ± 0.008 / 0.113 | −0.006 ± 0.005 / 0.074 |
| noise | −0.177 ± 0.006 / 0.197 | −0.011 ± 0.003 / 0.045 | −0.013 ± 0.008 / 0.113 | −0.006 ± 0.005 / 0.074 |
| noise, half observed, previous tick | −0.420 ± 0.014 / 0.465 | −0.094 ± 0.004 / 0.110 | −0.017 ± 0.008 / 0.114 | −0.021 ± 0.005 / 0.078 |
| noise, half observed, refresh time | −0.219 ± 0.008 / 0.246 | −0.017 ± 0.005 / 0.075 | −0.020 ± 0.011 / 0.150 | −0.010 ± 0.008 / 0.106 |

**On clean returns, `plain` is the estimator to use.** Its error is 0.016,
against 0.042 for the kernel and 0.074 at best for pre-averaging. The
other two trade variance for robustness that clean data does not need.

**Noise costs `plain` a fifth of the correlation and the kernel almost
nothing.** Noise adds variance to each series and nothing to their
covariance, so `plain` reads 0.177 low on average. The kernel's error moves
from 0.042 to 0.045. Pre-averaging does not move at all at this precision:
each block moves by about 0.003, and across 190 blocks those moves cancel
to the third decimal.

**Asynchrony is the harder case.** Carried forward, a series that was not
observed returns zero, and `plain` loses 0.420. The kernel, which sums its
lags, keeps most of it, at a bias of −0.094. Pre-averaging is the least
biased, at −0.017 and −0.021; its window, tens of rows long, spans the
rows a series missed. Refresh time takes the kernel's bias to −0.017, on
602 rows a block, and pre-averaging's error rises there, to 0.150 and
0.106, with fewer rows to average over.

**`psd=False` is the better pre-averaging form here.** It has the lower
error in every condition, 0.074 against 0.113 on clean data. `psd=True` is
the default because its estimate is always positive semi-definite: it
clips a negative eigenvalue where one appears, and the balanced form
promises neither. For a correlation read on its own, the balanced form was
the more accurate on these streams.

## 9. The sequential detector against its paper

`kind="sequential"` is Wied and Galeano's (2013) monitoring procedure:
`span_rows` rows of history, then each row of a monitoring period of
`monitor_rows` rows tested against it as it arrives, until a flag or the
period's end. Their `T` is `monitor_rows / span_rows`, and
`boundary_gamma`, their `γ`, lowers the boundary early in the period. Its
size is the share of monitoring periods that end in a flag when the
correlation never changes. Their Table 2 gives it for two independent
GARCH(1,1) series mixed to a correlation of 0.5, 1000 series a cell. Here
each cell is one stream of about 1000 back-to-back cycles on the same
design, and on Gaussian pairs beside it:

| gamma | T   | m   | W&G Table 2 | GARCH | Gaussian |
|-------|-----|-----|-------------|-------|----------|
| 0.0   | 0.5 | 250 | 0.059       | 0.060 | 0.064    |
| 0.0   | 0.5 | 500 | 0.058       | 0.049 | 0.059    |
| 0.0   | 1.0 | 250 | 0.077       | 0.086 | 0.059    |
| 0.0   | 1.0 | 500 | 0.069       | 0.052 | 0.060    |
| 0.0   | 2.0 | 250 | 0.066       | 0.062 | 0.072    |
| 0.0   | 2.0 | 500 | 0.054       | 0.066 | 0.053    |
| 0.0   | 4.0 | 250 | 0.063       | 0.074 | 0.076    |
| 0.0   | 4.0 | 500 | 0.071       | 0.059 | 0.049    |
| 0.25  | 0.5 | 250 | 0.075       | 0.088 | 0.067    |
| 0.25  | 0.5 | 500 | 0.079       | 0.064 | 0.062    |
| 0.25  | 1.0 | 250 | 0.075       | 0.076 | 0.086    |
| 0.25  | 1.0 | 500 | 0.064       | 0.043 | 0.070    |
| 0.25  | 2.0 | 250 | 0.087       | 0.071 | 0.075    |
| 0.25  | 2.0 | 500 | 0.063       | 0.072 | 0.060    |
| 0.25  | 4.0 | 250 | 0.073       | 0.087 | 0.064    |
| 0.25  | 4.0 | 500 | 0.077       | 0.054 | 0.061    |
| 0.45  | 0.5 | 250 | 0.169       | 0.176 | 0.167    |
| 0.45  | 0.5 | 500 | 0.125       | 0.155 | 0.123    |
| 0.45  | 1.0 | 250 | 0.174       | 0.172 | 0.157    |
| 0.45  | 1.0 | 500 | 0.136       | 0.139 | 0.135    |
| 0.45  | 2.0 | 250 | 0.164       | 0.158 | 0.149    |
| 0.45  | 2.0 | 500 | 0.138       | 0.116 | 0.123    |
| 0.45  | 4.0 | 250 | 0.161       | 0.133 | 0.150    |
| 0.45  | 4.0 | 500 | 0.128       | 0.134 | 0.131    |

The nominal level is 0.05 in every row. The difference between two rates
from 1000 draws has a standard error of about 0.011 at 0.07 and 0.016 at
0.15.

**On their design the detector is within two standard errors of their
table in every cell.** The largest gaps are 0.023 at `boundary_gamma` of 0
and 0.25 (at `T = 4`, `m = 500`) and 0.030 at 0.45 (at `T = 0.5`, `m =
500`), in both directions. The critical values differ a little too. The
paper simulates them on a grid, which reads the supremum low, and this
library solves the law behind them
(`crates/online-core/src/boundary.rs`). So its values are higher in 11 of
the 12 cells of their Table 1, and within 0.03 of it in all.

**Neither is the nominal 5 %, and the paper says so.** At 0 and 0.25 the
size runs from 0.043 to 0.088, a little above 0.05 at `m = 250` and
closer at 500, as their asymptotics expect. At 0.45 it runs from 0.116 to
0.176: the boundary is low at the start, and a noisy early correlation
crosses it. That is the price of catching an early change sooner, which
`tests/test_corrchange.py` holds it to on a change ten rows in.

**Gaussian pairs read the same as GARCH ones:** 0.049 to 0.086 at 0 and
0.25, and 0.123 to 0.167 at 0.45. The paper's assumptions admit GARCH, and
this design does not tell the two apart.

`tests/test_corrchange.py` holds the Gaussian size at `m = 250`, `T = 1`
between 0.025 and 0.075, and the critical values to the paper's Table 1.

## 10. Running them

```sh
uv run python scripts/regime_experiments.py all              # every experiment, in the order of the table below
uv run python scripts/regime_experiments.py size power dhat  # any of them by name, in the order given
```

| experiment | section | wall time |
|---|---|---|
| `recovery` | [§1](#1-does-hmm-recover-the-stream-that-made-it) | 0.2 s |
| `switch` | [§1](#through-a-switch) | 0.2 s |
| `size` | [§2](#2-is-the-monitor-the-size-its-paper-says) | 263.9 s |
| `power` | [§3](#3-and-is-it-the-power-its-paper-says) | 8.5 s |
| `dhat` | [§4](#4-where-the-heavy-tails-go-d-hat) | 2.2 s |
| `delay` | [§5](#5-two-changepoint-detectors-on-the-same-break) | 1.3 s |
| `epps` | [§6](#6-the-epps-effect-and-the-two-ways-out) | 3.9 s |
| `deco` | [§7](#7-deco-an-equicorrelation-that-moves) | 0.1 s |
| `rcov` | [§8](#8-rcov-three-estimators-against-each-blocks-truth) | 3.6 s |
| `sequential` | [§9](#9-the-sequential-detector-against-its-paper) | 23.9 s |

`all` runs them in that order in about 5 minutes of wall time, most of it
the size study's 20,000 draws a cell. That was measured on 2026-09-27, and
on 2026-09-28 for `size`, `power`, `dhat` and `sequential`, on an Apple M4
Pro (14 cores, 48 GB, macOS 15.7.3) with Python 3.12.13, Polars 1.44.2 and
a release build. Earlier runs took 40.3 s on 2026-09-23, with 2000 draws a
cell, and about 36 seconds on 2026-09-06, when this page was first
written. Every figure the 2026-09-23 run gave that this one re-ran is
unchanged.

**Every stream is generated from a seed, so two runs of one build give the
same numbers.** The seeds are derived with `zlib.crc32`, not `hash()`,
because Python randomises the hash of a string in each process. Nothing
downloads and nothing is cached. A different build can move the numbers,
and the dated notes in sections 1 to 4 record where one did.

The script is committed, and the gate does not run it: the size study
alone is 20,000 replications a cell across three distributions, 264 s of
the run.
