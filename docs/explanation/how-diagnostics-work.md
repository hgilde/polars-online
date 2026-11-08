<!-- kind: explanation -->
# How the diagnostics work

This page explains, for anyone who reads the diagnostics a spec writes on
every row, how each keeps a memory of its own, when it starts to read,
how to read its statistic, and how a staged fit is checked.

A model fitted on a stream always returns a number. Whether that number
can be trusted is a set of separate questions. Is the prediction scaled
right? Has the relationship broken? Can a coefficient's t be believed? Is
the fit missing a lag or a curvature? Another model may predict better,
and the model may not be ready yet. Each question has its own
[diagnostic](../GLOSSARY.md#fitting-a-model): a
[switch](../GLOSSARY.md#fitting-a-model) in the
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) that adds fields
to the model's output on every row, or a call in `po.eval` that scores an
output after the run. [DIAGNOSTICS.md](../DIAGNOSTICS.md) lists each
question, its diagnostic and its fields.

Four ideas hold for every diagnostic a spec writes. Each reads a residual
the model has not learned ([Why every prediction is
out-of-sample](out-of-sample.md)). Each keeps a memory of its own. Its
statistic is read on every row, which changes what a threshold means. And
each has a cost a row and a set of models that take it ([What each costs,
and which models take
it](../DIAGNOSTICS.md#what-each-costs-and-which-models-take-it)).

## Run once or windowed

**A run-once test assumes a relationship that does not change and a fit
that does not forget.** Under those assumptions its statistic has a known
distribution, and a threshold has a known false-alarm rate. Once a fit
forgets, the relationship it estimates is allowed to move, and the
question becomes local: is the fit wrong *now*? A windowed diagnostic
answers that. It computes the same statistic from moments that decay as
the fit's do, and reads it at the Kish size of its weights in place of the
row count. The Kish size, `n_kish = (Σw)² / Σw²`, is the number of equally
weighted rows that a set of decayed weights is worth (Kish 1965). For
exponential weights at a half-life of `h` rows,
`n_kish = (1 + lam) / (1 − lam)` with `lam = 0.5 ** (1 / h)`, about `2.885 h`. A
half-life of 200 rows is worth
about 577 independent rows. Windowed, the specification tests and Jarque
and Bera's test exceeded their 5% critical values on 2% to 7% of the rows of clean
streams, as they did run once.

**A diagnostic's memory matches the fit's by default, because a statistic
read over a span the fit no longer weighs judges a fit that is not the one
predicting.** Each diagnostic's keyword, such as `calibration_half_life`,
sets its memory, `inf` runs it once over the whole stream, and [One memory per
diagnostic](../DIAGNOSTICS.md#one-memory-per-diagnostic-run-once-or-windowed)
gives each default. Under a [hard window](../GLOSSARY.md#fitting-a-model)
the default is the half-life whose exponential weights have the window's
Kish size. Beside `window_size=400` with no decay, `se_coef_hc0` read the
spread of the slope fitted under that hard window over 200 streams, 0.0495 against 0.0498, where at
the run-once memory it read a third of it. On 30 clean streams beside
`window_size=100`, `ramsey_reset` exceeded its 5% value on 4.8% of rows,
where run once it exceeded on 51%.

**A fit that forgets absorbs a miscalibration at its own pace, so a
diagnostic at the fit's own memory sees only part of it.** A calibration
slope (the slope of the target regressed on the prediction) of 0.7 says the predictions are 43% too large (`1 / 0.7` is 1.43).
The fit drifts toward the truth as it learns. Over one half-life it has
moved halfway, so the residuals a diagnostic at the same memory reads have
had half the error taken out of them. That diagnostic is conservative.
On a fit whose calibration slope was 0.7, the calibration test exceeded its
5% value on 5% of rows at the model's memory. It exceeded on 42% at four
times that memory, and on 89% run once. On calibrated fits it exceeded on
0.3% to 0.6% of rows at the model's memory and 0% to 0.3% at four times.
That is why the calibration's memory defaults to four times the model's,
and every other diagnostic's to the model's own.

**A shorter memory finds a change sooner and misses more small ones.** A
windowed statistic responds to a change within about one memory, so its
delay scales with its half-life. A shorter memory also has a smaller Kish
size, which makes the statistic of the same effect smaller, so a small
effect stays below the threshold. A diagnostic's memory is the place to
trade delay for power. On 200 streams of 3,000 rows whose intercept moved
by half a noise standard deviation at row 1,500, the CUSUM at a memory of 200 rows found
every break within 100 rows. Run once over the whole stream, it found each
244 rows after the break on the median.

## When a diagnostic starts to read

A diagnostic folds a row into its statistic only once the model is
[ready](../GLOSSARY.md#warm-up-and-readiness), and, beside a fit that
forgets, only once the stream has come 95% of the way to its steady
state on the longest finite memory the model keeps. [One memory per
diagnostic](../DIAGNOSTICS.md#one-memory-per-diagnostic-run-once-or-windowed)
gives the rule and the models that keep a second memory.

**A diagnostic waits for the longest memory because a residual from
before the stream nears its steady state on that memory carries its
warm-up bias, which reads as a break.** On 20 clean streams at a
half-life of 200, with its diagnostics started once its level alone was
ready, `holt`'s CUSUM had a spread of 2.7 and exceeded 3 on 28% of the 2,000
rows after its diagnostics started. Started once its trend was ready, whose
half-life is four times its level's by default, it had a spread of 0.74
and exceeded on none.

**The readiness warnings read the model's own half-life, so they can judge
a stream long before its diagnostics start.** `settled_frac`,
`min_settled_frac`, `summary()` and the readiness warnings that a
[gate](../GLOSSARY.md#warm-up-and-readiness) is not met read the decay
`half_life` sets. A readiness warning projects the weight a stream tops
out at, `1 / (1 − lam^d)` for a clock step `d` between rows, or the
[noise gate](warm-up-and-readiness.md#the-noise-gate-estimation-error-against-the-noise)'s
ratio, `error_inflation`, and both approach their steady
state on that clock. On `holt` at a
half-life of 50, with its trend at the default 200, the stream comes 95%
of the way to its steady state on the level at row 218. A `min_weight` above the weight's ceiling
of 72.6 is named at row 268, a half-life later. The first `cusum` comes at
row 878, past the trend's 95% at row 866.

**A fit far from its truth carries its warm-up error for many
half-lives.** `rls` and `kalman` learn the intercept from a prior at 0, so
on a target at 5,000 their first predictions are thousands of noise standard
deviations off. Beside `rls` at a half-life of 50, residuals folded from the first
prediction held Jarque and Bera's statistic past its 5% value on 81% of
rows 1,000 to 2,000, 20 to 40 half-lives in. They held the calibration
test past its own on 100%. Folded from readiness they exceeded on 4.9% and
24%. What remains is the prior's bias itself, which fades only as the fit
forgets it. At a half-life of 200, `rls`'s mean residual was still +2.1
noise standard deviations over rows 400 to 1,000, and +0.16 over rows 1,000 to 2,000. The
calibration and the CUSUM flag it there, as they should. `kalman` sizes
its prior from the first rows. From row 1,000 on it read the same at a
level as on a target at 0, each test past its 5% value on 0.1% to 8.9% of
rows.

**A fit that forgets nothing never reaches a steady state, so a run-once diagnostic has only the gates
to hold back the warm-up: give the model a `min_weight` of a few rows per
coefficient.** A fit with three rows for three coefficients predicts
wildly, and run once those first residuals weigh in the statistic forever.
On 40 clean streams of 6,000 rows, a run-once `ramsey_reset` beside an
`ewridge` with `half_life=float("inf")` at its default `min_weight` of 0 exceeded its 5% value on
26% of rows past row 2,000. At `min_weight=10` it exceeded on 2.6%. The
calibration test exceeded on 19% of rows at 0 and 9.4% at 10. Beside a fit
at a half-life of 200, a run-once `ramsey_reset` exceeded on 25.8% of rows
1,000 to 3,000 when it folded from the first prediction, and on 4.8% from
readiness.

## Reading a statistic that is read on every row

**Over many streams, a windowed statistic sits past its 5% value on about
5% of rows, but on any one stream that share can be anything from 0% to
20%.** Each row's statistic shares almost all its data with the row
before, so the flags come in long runs. Over 40 clean streams, Ljung and
Box's statistic at a memory of 1,000 rows sat past its 5% value on 4.2% of
rows. On single streams the share ran from 0% to 20%. Read a statistic's
median over a stretch, or how long it stays past a threshold, before
reading one row.

## Checking a fit stage by stage

A fit built in stages, such as a fair value whose residual feeds a second
regression, is checked stage by stage. Each stage asks for the
diagnostics its own construction puts at risk, and [Check each stage with
the diagnostics it asks
for](../how-to/check-a-two-stage-fit.md#check-each-stage-with-the-diagnostics-it-asks-for)
runs them on a two-stage fit whose answer is planted.

**The first stage's residual can be a feature because it is
out-of-sample.** The bank writes each row's residual from a prediction
made before it learned the row, so the residual never holds its own row's
target. Read as a feature, it is known at its row, as a feature must be.

**A residual used as a feature carries the first stage's own error, which
attenuates the second stage's coefficient toward zero.** That residual is
the quantity the second stage means to read plus the first stage's
estimation error, and noise in a feature shrinks a least-squares slope.
In the two-stage example of that how-to, a planted coefficient of −0.490 reads −0.450.

**The calibration says whether a second-stage coefficient needs
rescaling.** It regresses the target on the prediction, as Mincer and
Zarnowitz (1969) do: a slope of 1 says the
predictions are scaled right, so the coefficient that made them needs no
rescaling. Run once, it keeps every row from readiness on, so a second
stage with several coefficients keeps the early rows of a fit learned from
few rows, and can read a slope far from 1 long after the fit itself is
scaled right. A finite `calibration_half_life` reads it past those rows.

**A target that looks ahead overlaps the targets of the rows near it, so
its t needs Newey and West's standard error.** Two rows less than a
window apart share the moves their targets look ahead on, so their
residuals are correlated. Against a feature that moves slowly, a t that
treats them as independent is up to about `sqrt(h)` times too large, for
a flat window of `h` rows, and a coefficient that carries nothing then
looks significant. A target that weights its window exponentially, as in
that how-to's example, overlaps less, and there the ratio is smaller.
`se_coef_hac` (Newey and West 1987) allows for the overlap over the lags
`horizon_rows` sets, and the calibration's Wald statistic takes the same
variance ([Check a coefficient's standard
error](../how-to/check-standard-errors.md)).

## Sources

| source | for |
|---|---|
| Kish, L. (1965). *Survey Sampling*. Wiley. | the effective sample size of unequal weights |
| Mincer, J. A. and Zarnowitz, V. (1969). The evaluation of economic forecasts. In *Economic Forecasts and Expectations*, ed. J. A. Mincer, 3–46. NBER. | the calibration regression |
| Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708. | `se_coef_hac` |
