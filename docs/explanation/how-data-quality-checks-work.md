<!-- kind: explanation -->
# How the data quality checks work

This page explains, for anyone who runs `bank.check()` on a [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads),
how the check finds problems in a stream from what the bank keeps anyway,
at no cost to the run, and why each problem it finds harms a fit.

A model fitted on a stream fits whatever it is given. It cannot tell you
that a column is `-999` where it was not measured, or that a feed stopped
and its last value was carried forward. Nor can it tell you that a feature
is a price level where a change was meant. Each of these is learned as if
it were a measurement, and each moves the fit in a way its predictions
alone may not show. [DATA-QUALITY.md](../DATA-QUALITY.md) lists every
finding `check()` reports, with its threshold and its remedy, and [Check a
stream's data quality](../how-to/check-data-quality.md) runs the check.

## What the check reads

**`check()` adds no work to the run, because it reads only what a bank
keeps anyway.** That is the bank's per-spec counts and settings and each
group's last row (`summary()`, `describe()`, `solve_failures()` and
`last_row()`), the [Gram](../GLOSSARY.md#fitting-a-model) (the matrix of
running sums) of an `ewridge`, `lasso` or `ew_cov`, and the pairwise
moments a `marginal` keeps. So a bank loaded from a [state file](../GLOSSARY.md#saving-and-serving)
gives the same findings as the bank that saved it, and the chunking of the
stream moves none of them. Call it after a first pass and before trusting
a fit. A group closed under `group_close` has left the bank with its sums,
so `check()` sees only the groups still open, and an audit refuses
`group_close`.

**An audit counts what a model does not keep.** A model skips a row whose
feature is missing, so it never sees the row. An audit, the spec
`po.spec.audit`, learns nothing: it reads every row of its columns and
counts nulls, NaNs and infinities apart. It also counts repeated values,
runs of one value, persistence like a random walk's, heavy tails,
duplicated columns and the clock's stamps. Its memory does not grow with
the rows: about 14 KiB a column. Run it alone before choosing a model, or
beside the models in the same pass. Without one in the bank, `check()`
says nothing about sentinels, frozen feeds, few values, random walks,
tails, duplicates or the clock's spacing.

**Each threshold is set where a planted problem is found and clean data
stays quiet.** Every threshold was measured both ways, on seeded streams
of clean shapes and on the same streams with each problem planted, and
[How the checks work](../DATA-QUALITY.md#how-the-checks-work) gives the
shapes and the result.

## Why each finding harms a fit

Each finding that `check()` reports is a way the rows mislead a fit that
reads them as measurements. The entries in [DATA-QUALITY.md](../DATA-QUALITY.md) give
what each does to each model and its remedy.

### Nulls: when skipping a row is safe

**Leaving the gap alone is right whenever the gap does not depend on the
target.** Statisticians sort gaps by what they depend on (Rubin 1976).
Gaps *missing completely at random* depend on nothing. Gaps *missing at
random* depend only on values that are observed. Gaps *missing not at
random* depend on the value that is missing. Skipping every row with a gap,
which is called *listwise deletion*, is safe for a regression under a condition
that cuts across those three names. Its slopes are unbiased whenever the chance that
a row has a gap does not depend on the target, given the features. That
holds even when the gap depends on the missing feature's own value (Little
1992; Allison 2001). Its slopes are only noisier, since the fit sees fewer
rows. When the gap does depend on the target, the slopes are biased, and
no fill made from the features repairs it. An imputation that conditions
on the target does, and the library has none. On simulated regressions
with 30% of the rows missing and features correlated at 0.5, listwise
deletion was biased by 4.5 standard errors of the full data's slope when
the gap depended on `y`. It stayed within 0.2 when the gap depended on
another feature or on the feature itself.

**Do not fill a gap with a running mean.** A filled value carries no
variation, so on the filled rows the column is flat. Its slope shrinks
toward zero, and a feature correlated with it absorbs the rest of its
effect, which is the omitted-variable bias of the column the fill erased.
On the same simulated regressions, a running-mean fill biased the slopes
by 0.4 to 6 standard errors, where listwise deletion was unbiased. A
forward fill does the same, and adds the error of a stale value. Fill a
value forward only when it truly holds until it changes, such as the last
quote of an instrument. In that case the forward-filled value is the
observation itself.

**A Gram over each pair's own rows would lose fewer rows, and it was
measured and not built, because it is biased where listwise deletion is
not.** It would keep, for each pair of columns, the sums over the rows
where both are present, so a gap in one column would cost only that
column's pairs. Measured against listwise deletion on the same simulated
regressions, it was biased by 0.6 to 19 standard errors whenever the gap
depended on another feature, on the target or on the feature itself. At
30% missing and features correlated at 0.9, that was −4.6 when the gap
depended on another feature, −5.8 when on the feature's own value, and
−14.2 when on `y`. Listwise deletion was unbiased in the first two of
those cases. The pairwise matrix is also not a covariance matrix of any
one set of rows, so it need not be positive semi-definite. It was
indefinite in 62% to 100% of wide, correlated cases (20 to 200 columns,
correlation 0.9, 30% missing in every column). Every guard that kept it
usable acted as a ridge and removed no bias. The [noise
gate](warm-up-and-readiness.md#the-noise-gate-estimation-error-against-the-noise)'s
statistic, `error_inflation` (how much estimating the fit widens a
prediction's error), could not see it: it read 1.03 to 1.05 against 13.5
realized. The pairwise matrix
took 3.6 to 6.1 times as long a row, and 1.5 to 3.5 times the state. Its
gain was real only when gaps were completely at random in a model so wide
that listwise deletion kept no rows at all. An imputation by online
expectation-maximization, the one method that measured unbiased when gaps
depended on observed values, is not built.

### A sentinel erases a slope

**One value on 5% of the rows can erase a slope.** A slope is the
covariance of the feature with the target over the feature's variance.
`-999` on 5% of the rows adds about `0.05 × 999² ≈ 50,000` to the
variance of a column whose own variance is 1. It adds almost nothing to
the covariance, because the target is unrelated to which rows hold the
sentinel. So the slope falls by a factor of about 50,000.

### A frozen feature winds up `rls`

**`rls` loses the slope of a feature that holds one value.** `rls` keeps
`A`, the decayed sum of each row's `z zᵀ` (`z` is the intercept's 1 and
the features), and its prior `delta` decays with it. While a feature holds
the value `c`, each new row adds information only along `(1, c)`, the
direction of `intercept + c·slope`. The information that separates the
slope from the intercept decays every row, and
nothing renews it. Once it falls below a rounding step, rounding sets the
slope, and the slope wanders. Control engineers call this *estimator
windup* under exponential forgetting (Åström and Wittenmark 1995).
Predictions stay right while the feature is held, because only
`intercept + c·slope` is used. They fail the moment the feature moves
again. A forgetting rule that keeps a floor under `A` (Kulhavý and Zarrop
1993) was measured for `rls`, and is not built. In `rls`'s uncentered
coordinates the floor acted as a ridge whose pull grows with the square of
the level. The [`rls` section](../MODELS.md#rls--recursive-least-squares)
of MODELS.md gives the measurements at other levels.

### A level far above its spread

**A level loses precision in a model that does not center.** A double
keeps 53 bits, about 16 decimal digits. A value at level `L` with spread
`s` spends `log10(L/s)` of them on the level, so it keeps about
`16 − log10(L/s)` digits of its variation (Goldberg 1991). A sum-form
variance, `Σx²/n − x̄²`, subtracts two numbers that agree in their first
`2·log10(L/s)` digits, and loses those digits (Chan, Golub and LeVeque
1983). The models that center keep their running means compensated, as a
pair of doubles, and accumulate co-moments about them, so they lose
nothing at a level. Subtracting any origin near the level does the same
for a model that does not center: precision depends on `|x − origin| / s`,
whichever origin is chosen.

**A level breaks `rls`'s prior.** In uncentered coordinates, a feature at
level `L` and the intercept's column of ones point almost the same way.
The direction that separates the slope from the intercept carries
information of only about `n·s²/L²` after `n` rows. `rls` starts from a
prior `delta·I` in those same coordinates, `delta = 1` by default. Without
a decay the prior never fades, and once `L/s` passes about `sqrt(n/delta)`
it outweighs the data in that direction and pins the slope near zero.
Measured at 20,000 rows and no decay, a level of 100 spreads kept an R² of
0.42 against 0.51, and 1,000 spreads kept 0.015. With `delta=1e-6`, both
read 0.51. Under a decay the prior fades with the sums, so the same levels
left `rls`'s R² where `ewridge`'s was at half-lives of 20 and 200. A short
stream at a long half-life is still in the prior's shadow. Over 2,000 rows
at a half-life of 200, a thousandth of `delta` is left beside a weight of
289, and a level of 10,000 spreads took 0.16 off the R² over the last
tenth. That is why the check's limit for `rls` reads the prior left and
the weight held, not the level alone ([A level far above its
spread](../DATA-QUALITY.md#a-level-far-above-its-spread-and-scales-apart)).

**A step size needs a scale.** `sgd`, `pa` and `ftrl` move the
coefficients a step along the gradient on each row. Least mean squares
converges only when the step is below `2/λmax`, where `λmax` is the
largest eigenvalue of `E[z zᵀ]`, and its speed is set by the eigenvalue
ratio `λmax/λmin` (Haykin 2002). A feature at level 5,000 makes `λmax`
about 25 million, so a step tuned for unit features diverges. Features
whose standard deviations are 1,000 apart make that ratio about a million,
so the narrow feature's coefficient barely moves. `kalman` and `rls` are
second order and need no step, but a ridge in the features' squared units
penalizes a narrow feature's coefficient more for the same effect.

### Collinear features leave the coefficients to chance

**A coefficient's variance grows with how well the other features predict
its own.** A coefficient's variance is the residual variance over `n`
times the feature's variance, multiplied by its *variance inflation
factor*, `1/(1 − R²ⱼ)`. Here `R²ⱼ` is the R² of that feature regressed on
the others. At a correlation of 0.999 with another feature, the factor is
about 500, so the coefficient's standard error is 22 times what it would
be alone. Belsley's condition indexes find a dependency among three or
more columns that no single pair shows (Belsley, Kuh and Welsch 1980).

### Random walks make a relation out of drift

**The t-statistic of a regression of one random walk on another grows
with the rows instead of settling.** The t-statistic assumes residuals
that are independent from row to row. Regressing one random walk on
another leaves residuals that are a random walk themselves, so the
effective number of independent rows is a handful, not `n`. The statistic
grows with `sqrt(n)` instead of settling, so a longer stream makes the
false relation look more certain (Granger and Newbold 1974; Phillips
1986). Differencing both series turns each into its independent changes,
and the ordinary statistics hold again. The check reads a column's
persistence with the Dickey-Fuller statistic (Dickey and Fuller 1979),
which [Random walks](../DATA-QUALITY.md#random-walks) defines. Measured over 200
pairs of independent random walks of 5,000 rows, 98.5% had a slope with
`|t| > 1.96`, the median `|t|` was 33, and the median R² was 0.18. The
same pairs differenced had `|t| > 1.96` in 2.5%.

### Heavy tails move a squared loss

**The influence of one row on a least-squares fit grows without bound
with its residual:** a residual twice as large moves the fit twice as far
(Huber 1964). Huber's loss is squared near zero and linear past
`huber_delta` residual scales. That scale is `huber_scale`, by default the
exponentially weighted median absolute deviation (MAD). So a residual's
influence is capped, and quantile
regression caps it at a constant ([The robust scale and the refit of the
first rows](robust-scale.md)). `sigma` is an exponentially weighted root
mean square, so one residual of `k` times the usual size inflates it for
several half-lives, the more the larger `k` is. When the residuals have
infinite variance, as Student's t with 2 degrees of freedom does, `sigma`
never reaches a steady state. An interval of a multiple of `sigma` either
side of the prediction then over-covers in calm stretches, since `sigma`
is inflated by the last large row. `drift_<t>`, a Page-Hinkley test on
`|resid| / sigma` that `emit_drift` turns on, fires on stable data whose tails are heavier than its threshold allows. On ten streams of
200,000 stable rows with Student's t noise of 3 degrees of freedom,
`drift_threshold=20` flagged 7 to 14 rows in each, 11 on average. [Are its
tails heavy?](../DIAGNOSTICS.md#are-its-tails-heavy) has the thresholds
that flag fewer. A conformal interval tracks a quantile of `|resid|`
directly, so its long-run coverage is the one asked for whatever the tails
(Vovk, Gammerman and Shafer 2005).

### A target that looks ahead leaks without its embargo

**A model that learns a look-ahead target at its own row learns the
targets it has yet to predict.** A target that looks `k` rows ahead,
such as the mean return of the next ten rows, is known only `k` rows after
its own row. A model that learns it at its own row learns returns from the
rows it has yet to predict. Its coefficients then carry part of the very
targets they are scored on, and a feature built from recent returns will
seem to predict them. An `embargo` of `k` learns each row only once its
target would be known (López de Prado 2018, ch. 7). A target written as
a [window expression](../how-to/make-a-target-that-looks-ahead.md) that
looks ahead carries its own delay, and `fit_predict` refuses an embargo
shorter than its window. A target made as a column does not, so the
embargo is the caller's to set ([A target that looks
ahead](out-of-sample.md#a-target-that-looks-ahead)).

### Too little data leaves the fit to the ridge

**A group with fewer rows than coefficients, or a half-life too short to
gather `min_weight`, cannot be fitted from its data.** With `k`
coefficients and fewer than `k` rows, the normal equations have no unique
solution, and the ridge alone sets the fit. Under a decay, a stream whose
rows come one clock unit apart can gather at most `1/(1 − 2^(−1/h))` of
weight at half-life `h`. That is 3.4 at `h = 2`, so a `min_weight` of 10
is never met. After a time `T` on the clock, a stream has gathered
`1 − 2^(−T/h)` of that ceiling, which is `settled_frac` ([Warm-up and
readiness](warm-up-and-readiness.md)). The [Warm-up
gates](../MODELS.md#warm-up-gates) lists the gates.

## Sources

| source | for |
|---|---|
| Allison, P. D. (2001). *Missing Data*. Sage. | listwise deletion is unbiased for regression slopes when the gap does not depend on the target |
| Åström, K. J. and Wittenmark, B. (1995). *Adaptive Control*, 2nd ed. Addison-Wesley. | estimator windup under exponential forgetting |
| Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). *Regression Diagnostics*. Wiley. | condition indexes |
| Chan, T. F., Golub, G. H. and LeVeque, R. J. (1983). Algorithms for computing the sample variance: analysis and recommendations. *The American Statistician* 37(3), 242–247. | the precision a sum-form variance loses at a level |
| Dickey, D. A. and Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association* 74(366), 427–431. | the unit-root statistic `random_walk` reads |
| Goldberg, D. (1991). What every computer scientist should know about floating-point arithmetic. *ACM Computing Surveys* 23(1), 5–48. | the digits a double keeps |
| Granger, C. W. J. and Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics* 2(2), 111–120. | a regression of one random walk on another |
| Haykin, S. (2002). *Adaptive Filter Theory*, 4th ed. Prentice Hall. | the step-size bound and eigenvalue spread of least mean squares |
| Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics* 35(1), 73–101. | the influence of a large residual, and Huber's loss |
| Kulhavý, R. and Zarrop, M. B. (1993). On a general concept of forgetting. *International Journal of Control* 58(4), 905–924. | stabilized forgetting |
| Little, R. J. A. (1992). Regression with missing X's: a review. *Journal of the American Statistical Association* 87(420), 1227–1237. | what complete-case analysis does to a regression |
| López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 7. | purging and the embargo |
| Phillips, P. C. B. (1986). Understanding spurious regressions in econometrics. *Journal of Econometrics* 33(3), 311–340. | why the t-statistic grows with the rows |
| Rubin, D. B. (1976). Inference and missing data. *Biometrika* 63(3), 581–592. | missing completely at random, at random, and not at random |
| Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. | conformal prediction |
