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
page and its script were written as that batch closed. It measures three
of the five, `hmm`, `corrchange` and `bocpd`, and two remedies for [series
that tick at their own
times](../README.md#series-that-tick-at-their-own-times): refresh-time
sampling and the lagged co-moments of `ew_cov(lags=...)`. `deco` and
`rcov` have no experiment here.

| section | the question | measured on |
|---|---|---|
| [0. The short version](#0-the-short-version) | what the experiments found, one paragraph each | |
| [1. Does `hmm` recover the stream that made it?](#1-does-hmm-recover-the-stream-that-made-it) | how many rows the filter puts in their true state, by how it was started | `hmm` |
| [2. Is the monitor the size its paper says?](#2-is-the-monitor-the-size-its-paper-says) | how often the test rejects when nothing changed | `corrchange`, `kind="monitor"` |
| [3. And is it the power its paper says?](#3-and-is-it-the-power-its-paper-says) | how often it rejects a real break | `corrchange`, `kind="monitor"` |
| [4. Where the heavy tails go: `D-hat`](#4-where-the-heavy-tails-go-d-hat) | how well the statistic's denominator is estimated | `corrchange`, `kind="monitor"` |
| [5. Two changepoint detectors on the same break](#5-two-changepoint-detectors-on-the-same-break) | false alarms on a quiet stream, and the delay to a real break | `corrchange(window)`, `bocpd` |
| [6. The Epps effect, and the two ways out](#6-the-epps-effect-and-the-two-ways-out) | what asynchrony does to a correlation, and what recovers it | `po.prep.refresh_time`, `ew_cov(lags=...)` |
| [7. Running them](#7-running-them) | how to reproduce every number, and how long it takes | |

Every number comes from `scripts/regime_experiments.py`, run on 2026-09-23.
Sections 1 to 4 end with a dated note that keeps the figures they first
reported, on 2026-09-06.

## 0. The short version

**Seeded from the rows, `hmm` splits zero-mean rows by direction.** Its
default seeding is k-means over the rows, and two zero-mean states differ
in nothing k-means can see. On streams that hold a single regime, the
filter puts 57 % of the rows in the true state, counting the 500 it reports
nothing on before seeding. Given the covariances to start from, it puts
98 % there ([§1](#1-does-hmm-recover-the-stream-that-made-it)).

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
innovations", a phrase two distributions fit. Under either, the size is
within 0.011 of their Table 1 and the power within 0.034 of their Table 2.
On Gaussian pairs the size runs from 0.034 to 0.052 against the nominal
0.05, the level `tests/test_corrchange.py` holds it to
([§2](#2-is-the-monitor-the-size-its-paper-says),
[§3](#3-and-is-it-the-power-its-paper-says)).

**`D-hat`, the statistic's denominator, is exact on Gaussian pairs and
biased low on a tail-dependent `t_5`.** On Gaussian pairs it averages 0.749
against an asymptotic 0.750 at `T = 2000`. On the `t_5` it reads 18 % low
at `T = 500` and 9 % low at `T = 2000`, and its scatter grows with `T`
instead of shrinking. The monitor's size and power stay close to WKD's
even so ([§4](#4-where-the-heavy-tails-go-d-hat)).

**The two changepoint detectors are at opposite ends of one trade-off.** On
the same streams, `corrchange(window)` finds a correlation break in a
median of 29 rows and flags 51 rows per 1000 quiet ones. `bocpd` takes 98
rows and raises 0.68 alarms per 1000. Neither is better: they answer
different questions ([§5](#5-two-changepoint-detectors-on-the-same-break)).

**Asynchrony leaves almost nothing of a correlation at the finest interval,
and both remedies recover most of it.** A true correlation of 0.8 reads as
0.095 from one-row returns. Refresh-time sampling recovers 0.54 from 19157
returns, 16 % of the 119991 at the finest interval. The lag inversion on
`ew_cov(lags=...)` recovers 0.76 while reading every return
([§6](#6-the-epps-effect-and-the-two-ways-out)).

## 1. Does `hmm` recover the stream that made it?

`hmm` gives each row a probability for each of `k` hidden states. This
section asks how many rows the filter puts in the state that generated
them, for four ways of starting it.

**No stream here switched regime, so following a switch is not measured.**
`po.sim.regimes` draws one state per block, and with ten blocks at a 1 %
chance per block of leaving state 0, every block of all eight seeds stayed
there. The table below therefore measures the share of rows the filter
puts in that one true state.

| condition | value |
|---|---|
| streams | 8 seeds of `po.sim.regimes`: two series, 10 blocks of 400 rows, 4000 rows each, differenced into 3999 returns |
| states drawn | correlation 0.1 and 0.8, under the sticky transition matrix `[[0.99, 0.01], [0.02, 0.98]]` |
| states realised | state 0, at correlation 0.1, in every block of every seed |
| the filter | `k = 2`, `halflife = 4000`, and `precision_prior = 0.01` except in the last row |
| seeded from the rows | `warm_rows = 500` and `transition_prior = 200` |
| started from the truth | the true transition matrix, zero means, and covariances built from the true correlations and each seed's sample variances |
| the score | the share of rows in state 0, read as `p_1` at most 0.5, with a row that has no `p_1` counted there. The script takes the better of the two ways to match states to the truth, which here is this share |

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
| 500  | -0.5 | 0.040       | 0.045           | 0.046          | 0.046    |
| 500  | 0.0  | 0.035       | 0.024           | 0.044          | 0.045    |
| 500  | 0.5  | 0.041       | 0.043           | 0.041          | 0.052    |
| 1000 | -0.5 | 0.038       | 0.033           | 0.036          | 0.043    |
| 1000 | 0.0  | 0.034       | 0.035           | 0.041          | 0.034    |
| 1000 | 0.5  | 0.039       | 0.044           | 0.041          | 0.048    |

The table holds 2000 replications per cell, at a nominal level of 0.05,
where a rate from 2000 draws has a standard error of 0.0049. Each draw is
one span of `T` rows, tested once with `alpha_adjust="none"`.

**Both readings of WKD's phrase are within 0.011 of their Table 1.** The
largest gap is at `T = 500` and `rho = 0`, where the shared-scale draw
rejects 0.024 of the time against WKD's 0.035. That gap is 2.2 of this
study's standard errors, and every other gap is 1.8 or less.

**The two readings give almost the same size.** At `|rho| = 0.5` they are
within 0.003 of each other. At `rho = 0` they differ by 0.006 at
`T = 1000` and by 0.020 at `T = 500`. Neither is liberal: no `t_5` cell is
above 0.046. At this precision, the ambiguity in WKD's phrase does not
decide the comparison.

**On Gaussian pairs the size is near the nominal 0.05**, from 0.034 to
0.052. `tests/test_corrchange.py` holds the test to that level on Gaussian
pairs at `T = 500`, for `rho` of 0 and 0.5, and again at 0.5 with one
column multiplied by 100. No test holds the size to WKD's own figure.

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
| 500  | t5          | 0.553 | 0.582         | 1.321         | 0.587       |
| 500  | t5_indep    | 0.585 | 0.613         | 1.327         |             |
| 500  | normal      | 0.830 | 0.851         | 1.327         |             |
| 1000 | t5          | 0.832 | 0.844         | 1.324         | 0.830       |
| 1000 | t5_indep    | 0.832 | 0.857         | 1.307         |             |
| 1000 | normal      | 0.994 | 0.994         | 1.355         |             |

The table holds 1000 replications per row, at a nominal level of 0.05.

**Under both `t_5` readings the power is within 0.034 of WKD's Table 2.**
At `T = 500`, where WKD give 0.587, the shared-scale draw reaches 0.553
against the asymptotic value and 0.582 size-adjusted, and the independent
draw 0.585 and 0.613. At `T = 1000` both draws reach 0.832 against WKD's
0.830, and 0.844 and 0.857 size-adjusted.

**On Gaussian pairs the power is well above WKD's figure:** 0.830 at
`T = 500`, or 0.851 size-adjusted, and 0.994 at `T = 1000`.
`tests/test_corrchange.py` holds the Gaussian power at `T = 500` above a
floor set from WKD's 0.587, less a tolerance.

**Read the size-adjusted column across distributions.** It takes its
critical value from null draws of the same distribution and length, so a
size that differs between distributions does not tilt it. Every empirical
quantile, from 1.307 to 1.355, is below the asymptotic 1.3581. So the
size-adjusted power is the higher of the two in every row but the last,
where both are 0.994.

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
| normal      | 500  | 0.739 | 0.742  | 0.060 | 0.750      |
| normal      | 2000 | 0.749 | 0.751  | 0.033 | 0.750      |
| t5          | 500  | 1.071 | 1.020  | 0.257 | 1.299      |
| t5          | 2000 | 1.184 | 1.097  | 0.353 | 1.299      |
| t5_indep    | 500  | 0.811 | 0.796  | 0.113 | -          |
| t5_indep    | 2000 | 0.828 | 0.815  | 0.073 | -          |

The table holds 200 replications per row, at `rho` = 0.5. The independent
`t_5` has no closed form here, so its last column reads `-`.

**On Gaussian pairs the estimator is right and tight.** It averages 0.739
at `T = 500` and 0.749 at `T = 2000`, against 0.750. Its standard deviation
falls from 0.060 to about 0.03 (0.033) as `T` grows fourfold. That is what
the delta method promises on data with light tails: right, consistent and
quick.

**On the shared-scale `t_5` it is biased low, and its scatter does not
shrink.** Its mean is 1.071 at `T = 500` and 1.184 at `T = 2000`, against
1.299: 18 % and 9 % low. Its median is lower still, 21 % and 16 % low, and
its standard deviation grows from 0.257 to 0.353. The estimator's inputs
are fourth moments, and a `t_5` has a fourth moment only just, at
`nu > 4`. Those inputs' own variance is infinite, so there is no rate at
which the estimate settles. The independent `t_5` is steadier: its
standard deviation falls from 0.113 to 0.073.

The size in §2 and the power in §3 are close to WKD's tables even so.

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
| `corrchange(window)` | `kind="window"`, `span_rows=100`, `n_perm=100`, `permute_every=100`, `alpha=0.05`, `seed=0` | `flag`: the change between two adjacent 100-row windows is above a permutation quantile |
| `bocpd` | `emission="gaussian"`, `hazard=500`, `prior_nu=5`, `prior_scale=[0.1]`, `prune_below=1e-8`, `max_run=1200` | its own alarm, `p_change`, cannot see this break, so here: the row that `t - run_mode` names as the current run's start jumps forward by more than 50 |

**`bocpd` is measured with `emission="gaussian"`, not its default
`"diag"`.** The default models each feature on its own, and only the
Gaussian emission can see a break in the correlation alone ([its README
section](../README.md#bocpd--how-long-has-this-regime-lasted)).

| detector           | false alarms / 1000 quiet rows | found the break | median delay | worst delay |
|--------------------|--------------------------------|-----------------|--------------|-------------|
| corrchange(window) | 51.33                          | 20/20           | 29           | 53          |
| bocpd              | 0.68                           | 20/20           | 98           | 351         |

**The two detectors answer different questions, which is why both are
here.** `corrchange(window)` is a test of one thing: whether the
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
| refresh-time sampling, `po.prep.refresh_time` | a grid point at the first row by which both series have ticked since the last point: 19157 returns, 16 % of the 119991 at the finest interval. On average a grid point kept 81 % of the ticks since the one before | 0.543 |
| lag inversion, `ew_cov(lags=...)` read through `po.corr.epps_invert` | every one of the 119991 fine returns, with no grid: the correlation at a scale of `L` rows, from the lagged co-moments at one row | 0.757 at `L` = 96 |

| L  | inverted correlation |
|----|----------------------|
| 1  | 0.095                |
| 2  | 0.170                |
| 4  | 0.294                |
| 8  | 0.456                |
| 16 | 0.608                |
| 32 | 0.699                |
| 64 | 0.742                |
| 96 | 0.757                |

**`L` is the same trade-off the sampling interval is.** Too small, and the
cross-covariance at longer lags is left out; too large, and the sum is
noise. In this sweep the correlation still rises at `L` = 96, the largest
value run, so the noisy end is not reached here.

The lag inversion is here because its lagged co-moments are a state a bank
already keeps, as `ew_cov(lags=...)`. What the lags do to `ew_cov`'s
throughput is measured in [docs/PERFORMANCE.md](PERFORMANCE.md) §15.

## 7. Running them

```sh
uv run python scripts/regime_experiments.py all              # every experiment, in the order of the table below
uv run python scripts/regime_experiments.py size power dhat  # any of them by name, in the order given
```

| experiment | section | wall time |
|---|---|---|
| `recovery` | [§1](#1-does-hmm-recover-the-stream-that-made-it) | 0.2 s |
| `size` | [§2](#2-is-the-monitor-the-size-its-paper-says) | 26.7 s |
| `power` | [§3](#3-and-is-it-the-power-its-paper-says) | 9.2 s |
| `dhat` | [§4](#4-where-the-heavy-tails-go-d-hat) | 2.5 s |
| `delay` | [§5](#5-two-changepoint-detectors-on-the-same-break) | 1.4 s |
| `epps` | [§6](#6-the-epps-effect-and-the-two-ways-out) | 0.3 s |

`all` runs them in that order, in 40.3 s of wall time. That was on
2026-09-23, on an Apple M4 Pro (14 cores, 48 GB, macOS 15.7.3) with Python
3.12.13, Polars 1.44.2 and a release build. When this page was first
written, on 2026-09-06, the run took about 36 seconds.

**Every stream is generated from a seed, so two runs of one build give the
same numbers.** The seeds are derived with `zlib.crc32`, not `hash()`,
because Python randomises the hash of a string in each process. Nothing
downloads and nothing is cached. A different build can move the numbers,
and the dated notes in sections 1 to 4 record where one did.

The script is committed, and the gate does not run it: the size study
alone is 2000 replications across three distributions, 26.7 s of the run.
