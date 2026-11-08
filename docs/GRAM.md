<!-- kind: explanation -->
# Wide fits from Grams

This page explains, for anyone who fits least squares from the Grams one
pass over a stream leaves, why a Gram stands in for the rows and what
each fit read from one assumes.

A least-squares fit reads its rows only through a few sums: the
weight of the rows, the columns' means and co-moments, and each target's
mean, variance and cross-moments with the columns. Together they are the
fit's *Gram*. `ewridge`, `lasso` and `ew_cov` keep one in their state, and
`bank.gram()` returns it as numpy arrays, in a dict typed
`po.gram.Gram` for a type checker. A Gram's size depends on its
number of columns alone, so one pass over a stream too large to hold can
leave a Gram for each week. Every fit after the pass reads those Grams and
never the rows. It can pool any set of weeks, fit a ridge with its t, walk
a lasso path, or count how often each column is chosen on halves of the
weeks. The module `po.gram` does each of these.

`bank.gram()` and `po.gram` need numpy, an optional extra, which
`pip install "polars-online[numpy]"` adds. The reference of
[`ModelBank.gram`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.gram)
states each array and the identities between them.

Each task on a set of Grams has a call in `po.gram` or the bank, and a
how-to that runs it:

| to | call | how-to |
|---|---|---|
| keep a Gram per week in one pass | `po.spec.ewridge(..., group="week")`, `bank.gram()` | [Keep a Gram per block in one pass](how-to/fit-weekly-grams-in-one-pass.md) |
| pool any weeks, and score a fit on others | `po.gram.merge` | [Pool Grams](how-to/pool-grams.md#pool-blocks-and-score-a-fit-on-blocks-held-out) |
| pool decayed weeks, each aged first | `po.gram.merge`, `bank.groups()` | [Pool Grams](how-to/pool-grams.md#age-decayed-blocks-before-you-pool-them) |
| pool weeks and leave their levels out | `po.gram.merge(..., within=True)` | [Pool Grams](how-to/pool-grams.md#pool-within-blocks-to-leave-their-levels-out) |
| fit a ridge and read its t | `po.gram.solve`, `po.gram.coef_stats` | [Fit a ridge from a Gram](how-to/fit-a-ridge-from-a-gram.md) |
| know whether a fit is determined, and fit on some of the columns | `po.gram.condition`, `po.gram.vif`, `po.gram.correlation`, `po.gram.subset` | [Check that a fit is determined](how-to/check-a-fit-is-determined.md) |
| choose columns | `po.gram.lars_path`, `po.gram.lasso_path` | [Choose columns on a Gram's lasso path](how-to/choose-columns-on-the-lasso-path.md#choose-columns-on-a-grams-lasso-path) |
| choose columns that hold across the weeks | `po.gram.lars_paths`, `po.gram.solve_subsets(..., path=...)` | [Select columns that hold across the weeks](how-to/select-columns-stably-over-weeks.md) |
| know how often a selection chooses a column with no effect of one sign | `po.gram.solve_subsets(..., within=True, target_signs=...)` | [Set the threshold from null runs, by sign flips](how-to/select-columns-stably-over-weeks.md#set-the-threshold-from-null-runs-by-sign-flips) |
| screen the columns again for each half, target and null run | `po.gram.solve_subsets(..., jobs=...)` | [Screen the columns again for each half and each null run](how-to/select-columns-stably-over-weeks.md#screen-the-columns-again-for-each-half-and-each-null-run) |
| see whether a coefficient holds from week to week | `po.gram.solve_subsets` | [Fit per-block coefficients](how-to/fit-per-block-coefficients.md) |
| read Grams back from files | `po.gram.from_row`, `po.ModelBank.load`, or numpy alone | [Read Grams from saved files](how-to/read-grams-from-saved-files.md) |
| halve or quarter a Gram's bytes | `bank.gram(dtype="float32", layout="packed")` | [Store a compact Gram](how-to/store-a-compact-gram.md) |
| keep a Gram of hundreds of features fast, in blocks of rows and on several cores | `gram_block_rows`, `gram_threads` | [Fit hundreds of features](how-to/fit-hundreds-of-features.md) |
| fit Huber's loss, reading the rows twice a pass and the Gram once | `po.gram.solve`, two Polars queries a pass | [Fit a Huber regression on a Gram](how-to/fit-huber-on-a-gram.md) |

The library's words, as they are used for Grams:

| word | what it means here |
|---|---|
| spec, bank | a spec is one model's description: the model, the columns it reads, and how it treats time. `po.ModelBank` fits a set of specs together over the same rows |
| `ewridge`, `lasso`, `ew_cov` | the three models whose state holds a Gram: a ridge regression, a lasso, and the moments of columns with no target. [MODELS.md's model table](MODELS.md#the-models-at-a-glance) lists every model |
| Gram | the sums a least-squares fit reads, as `bank.gram(spec)` returns them: one dict per group, whose keys the next table lists |
| column | one axis of a Gram: `"intercept"` first when the spec fits one, then the spec's features |
| block | the rows of one value of a spec's `group` column, and the Gram the bank keeps for them. The examples here make a week a block |
| pool | merge the Grams of disjoint rows into the Gram of all of them, with `po.gram.merge` |
| clock, `gap_cap` | the column a spec's `half_life`, `gap_cap` and `solve_every` are measured on, as `clock="ts"`, or the row count for a spec without one. `gap_cap` is the longest step between two rows that the bank ages a Gram by, and a spec with a clock needs one |
| `half_life` | each row's weight halves every `half_life` of the clock. `float("inf")` weighs every row alike, so a Gram holds plain least squares' sums |
| `weight_sum`, `n_kish` | the weight behind a Gram, and `weight_sum² / Σw²`, the number of equally weighted rows that weight is worth. Both are the row count when every row weighs 1 |
| active set, knot | on a lasso path, the columns whose coefficient is not 0, and a penalty at which that set changes |
| half | 20 of the 40 weeks, drawn at random, and their pooled Gram |

Each key of a Gram, and the symbol the formulas give it:

| key | what it holds |
|---|---|
| `group` | the block's key, as text: here the week's Monday |
| `instance` | the instance (the part of the model that fits one half-life): its suffix, such as `"@h500"`, or `""` for a spec with one half-life |
| `columns`, `targets` | the names along each axis |
| `weight_sum`, `n_kish` | the weight behind the Gram, `W`, and its Kish size |
| `means` | the columns' weighted means, `m` |
| `comoments` | the columns' centered co-moments, `C`, a `k × k` matrix |
| `cross_moments` | per target, the uncentered moments `E[z y]`, with `z` the columns of a row of data: one row of the array per target. At the intercept, a constant 1, the moment is the target's mean `ȳ` |
| `cross_centered` | per target, the same moments centered at the columns' means and the target's own mean: `c`, the vector a fit is solved from |
| `means_by_target` | per target, the columns' means over the rows the target was present on |
| `target_weights`, `target_means`, `target_vars`, `target_n_kish` | per target, its weight, its mean, its variance `V` and its Kish size |
| `lags`, `lag_comoments` | for an `ew_cov(lags=[...])`, the lags and the co-moments of each column with every column that many rows back. `None` for any other spec |
| `within_parts`, `target_within_parts` | only on a Gram from `po.gram.merge(..., within=True)`: the number of parts with weight, and per target the number with weight for it. The fits read them to leave the intercept out and to take a degree of freedom per part ([Pool within blocks to leave their levels out](how-to/pool-grams.md#pool-within-blocks-to-leave-their-levels-out)) |

## How a Gram stands in for the rows

A fit can read a Gram in place of its rows for two reasons. Every
least-squares fit on the Gram's columns is a function of the Gram, and
the Gram's moments are centered, so they keep their digits however far a
column sits from zero. A Gram holds the rows its spec weighs, as of the
last row it learned. Some questions need each row, and no Gram answers
them.

### Every least-squares fit is a function of the Gram

**Every fit to a weighted sum of squared residuals reads its rows only
through the Gram.** Take weights `w_i` that sum to `W`, the column means
`m` and the centered co-moments `C = Σ w (x − m)(x − m)' / W`. For a target
`y`, take its mean `ȳ`, its variance `V` and its centered cross-moments
`c = Σ w (x − m)(y − ȳ) / W`. For any intercept `b_0` and slopes `b`, the
mean squared residual is

```text
Σ w_i (y_i − b_0 − x_i'b)² / W  =  V − 2 b'c + b'C b + (ȳ − b_0 − m'b)²
```

Nothing on the right reads a row. Least squares minimizes it, a ridge
adds `λ b'b` to it, and a lasso adds `λ Σ |b_j|`. Best subset, stepwise
selection and an information criterion over sets of columns compare its
minimum from one set to another. Each of them is therefore computed from
the Gram. Under the Gaussian linear model this is Fisher's (1922)
sufficiency: the Gram is a sufficient statistic for the coefficients and
the noise variance.

**To fit on some of the columns, take their sub-block of the Gram.**
`po.gram.subset(g, ["intercept", "x0", "x5"])` selects those columns'
rows and columns of `C` and their entries of each target's moments. A fit
on the subset is the fit a pass over those columns alone would give on
the same rows. Keep `"intercept"` among the columns, since a subset
without it is fitted through the origin.
[Check that a fit is determined](how-to/check-a-fit-is-determined.md)
fits a subset. Every fit
in `po.gram` also takes `features=`, which fits the same columns with the
intercept kept, and returns a coefficient for every column of the full
Gram, 0 for each column left out.

**The same identity scores a fit on rows it was not fitted on.** Put a
held-out Gram's moments into the right side, and it gives the fit's mean
squared error over the held-out rows, without reading them. The target's
moments and the columns' must cover the same rows, as they do when the
target is present on every row of the Gram.
[Pool Grams](how-to/pool-grams.md#pool-blocks-and-score-a-fit-on-blocks-held-out)
scores a fit on the last ten weeks this way.

**The moments are centered, so they keep their digits when a column's
level, its typical size, is far from zero.** A raw second moment of a
column at a level `L` holds `L²` plus its variance. A variance read back
from it as `E[x²] − m²` loses the digits that `L²` takes up. A double
carries about sixteen, and for a unit-variance column at 1e8 the
difference keeps none of them. Each row's update adds the row's deviation from the running
means to the co-moments, as Welford's (1962) recursion does, with weights
as West (1979) gives them. Each target's cross-moments are kept
centered at its own means, as `cross_centered`, for the same reason (Chan,
Golub and LeVeque 1983).

**`weight_sum` counts weight, and `n_kish` counts rows.** With
`half_life=float("inf")` and every row's weight 1, both are the row
count. Under a decay, `n_kish = W² / Σ w²` is the number of equally
weighted rows the moments are worth (Kish 1965), and a standard error
divides by it.

### Which rows a Gram holds, and as of when

**A Gram holds the rows its model learned, at the weights its decay gave
them.** Under `half_life=float("inf")` that is every row since the group
began, each weighing the same. Under a finite half-life the older rows
weigh less, and a gap between two rows ages every row before it, by no
more than `gap_cap`.

**Under a `window_size`, a Gram holds only the hard window.** Every array
is the hard window's, the target moments included, so `po.gram.solve` on
it fits the hard window as it stood after the last row.

**`bank.gram()` is as of the last row, and `bank.coef()` as of the last
solve.** A model solves on the schedule its `solve_every` sets, so between
two solves its coefficients lag the Gram. Solve the Gram with
`po.gram.solve` for the fit as of the last row.

### What a Gram cannot answer

A question that needs each row's residual, or a column the Gram does not
hold, needs the rows, or a model that ran over them:

| question | why a Gram cannot answer it | where it is answered |
|---|---|---|
| a fit whose loss is not a sum of squares: robust, quantile or logistic | its minimum depends on each residual, so no sum of moments holds it | the robust and quantile regressions `huber` and `quantile`, and the online logistic regression `ftrl`, run over the stream; or, for Huber's loss, passes over the rows that share one Gram ([Fit a Huber regression on a Gram](how-to/fit-huber-on-a-gram.md)) |
| the residuals' tails, autocorrelation and spread, and which row moved the fit | each reads a row's own residual | the diagnostics a spec switches on, which judge each row's residual as the bank runs ([Is the model working?](DIAGNOSTICS.md)) |
| standard errors robust to heteroskedasticity or autocorrelation | their middle term sums each row's `x x' e²`, or products of residuals rows apart | the spec's `emit_robust_se` on the stream ([Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted)), or the spread of [per-block coefficients](how-to/fit-per-block-coefficients.md) |
| a column the Gram does not hold: a square, a product or a lag | its moments with the other columns are sums the pass never took | make the column with Polars' `with_columns` in the query before the bank, and run the pass again |
| a lagged co-moment across two pooled parts | a part holds only the pairs of its own rows | `ew_cov(lags=...)` over the whole stream. `po.gram.merge` returns `lags` as `None` |

## Pooling blocks

`po.gram.merge` pools the Grams of disjoint rows into the Gram of all of
them, so one pass serves every choice of weeks a fit makes. The parts'
rows must be disjoint: a row in two parts counts twice, and
`po.gram.merge` cannot tell.

To pool blocks' Grams into one, aged to a common row, with or without their levels, see [Pool Grams](how-to/pool-grams.md).

**A pooled Gram is formed from the parts' moments and the gaps between
their means, and never from a difference of two large sums.** This is
Chan, Golub and LeVeque's (1983) pairwise update. For parts `a` and `b`,
with weights `W_a` and `W_b`, means `m_a` and `m_b`, co-moments `C_a` and
`C_b`, and the gap `Δ = m_b − m_a` between their means, the union has

```text
W = W_a + W_b
m = m_a + (W_b / W) · Δ
C = (W_a · C_a + W_b · C_b) / W + (W_a · W_b / W²) · Δ Δ'
```

Each target's moments pool the same way over the rows it was present on,
by its `target_weights`. Every term is a weighted average of the parts'
moments, or a product of the gap. None is a difference of two large
cumulative sums, which is where pooling raw sums loses its digits. The
forty weeks of [Keep weekly Grams of hundreds of
features](how-to/fit-weekly-grams-in-one-pass.md#keep-weekly-grams-of-hundreds-of-features)
pooled to least squares on their rows within 1e-12.

**A block's weights are relative to its own last row.** So under a finite
half-life, a week that ended a month before the stream did is weighted as
if it ended with the stream. It weighs too much, by exactly the decay
between its last row and the stream's, unless it is aged to the stream's
last row before it is pooled ([Age decayed blocks before you pool
them](how-to/pool-grams.md#age-decayed-blocks-before-you-pool-them)).
Blocks run under `half_life=float("inf")` pool as they are.

**`po.gram.merge(blocks, within=True)` pools each block's moments about
its own means, and leaves out the spread between the blocks' means.** The
union's update adds `(W_a · W_b / W²) · Δ Δ'`, the co-moment of the gaps
between the parts' means. The within form drops that term, so the pooled
co-moments are the blocks' own, averaged by weight:

```text
W = Σ W_i        C = Σ W_i · C_i / W
T = Σ T_i        c = Σ T_i · c_i / T        V = Σ T_i · V_i / T
```

Each target's weights `T_i`, centered cross-moments `c_i` and variances
`V_i` pool over the rows it was present on. A week with no rows of a
target adds to `C` and not to that target's moments. Where the blocks'
means are equal, the two forms give the same co-moments, and so the same
slopes.

**The pooled Gram is that of the rows less their week's means, and a fit
on it has an intercept for each week.** A regression on the rows less
their weekly means gives the slopes that the raw rows give with a constant
column for each week (Frisch and Waugh 1933, Lovell 1963). That is the
within, or fixed-effects, estimator (Wooldridge 2010, chapter 10). It asks
only how the columns and the target move together inside each week, as
each week's own fit does, and the weekly levels drop out of it. Under
`target_gaps="pairwise"`, a target missing on some rows pools its
cross-moments over the rows it is present on, and the co-moments over
every row of the week
([Keep weekly Grams of hundreds of
features](how-to/fit-weekly-grams-in-one-pass.md#keep-weekly-grams-of-hundreds-of-features)).
So its
slopes can differ from the fixed-effects regression on its rows
alone, and a target present on every row gets that regression exactly.

**So the fit has no intercept of its own, and loses a degree of freedom
for each week.** The pooled Gram's `means` are 0, beside the intercept's
1, and the pooled Gram holds none of the weeks' intercepts. It counts its
weeks with weight in `within_parts`, and each target's in
`target_within_parts`. `po.gram.coef_stats` reads a target's
`target_within_parts` as `B` in `n − p − B`, with `n` the target's Kish
count (`target_n_kish`) and `p` the features that have variance.

**Pooled as one union, a chance correlation of the weekly means reads as
a correlation of the rows.** Take a feature whose level wanders from week
to week, such as an exponentially weighted mean at a half-life near a
week, and a target whose weekly level wanders apart from it. Their forty
weekly means correlate by chance, at about ±1/√40 = ±0.16 for independent
series. The union's co-moments hold that correlation, and its standard
errors divide by every row, as if each row had confirmed it. It
rests on forty weekly means. Pooled within weeks, that correlation drops
out ([Pool within blocks to leave their levels
out](how-to/pool-grams.md#pool-within-blocks-to-leave-their-levels-out)
measures it).

## A ridge and its t

`po.gram.solve` fits a ridge on a Gram, for a grid of penalties in one
call, and `po.gram.coef_stats` gives any fit its residual variance,
standard errors and t. The ridge solves the same system the `ewridge`
model solves. The t is least squares', with least squares' assumptions.
[Fit a ridge from a Gram](how-to/fit-a-ridge-from-a-gram.md) chooses a
ridge on weeks held out and reads its t.

**The ridge solves the centered system, so the intercept is not
penalized.** The slopes solve `(C + λI) b = c`, and the intercept is
`b_0 = ȳ − m'b` (Hoerl and Kennard 1970). With `standardize=True` the
system is solved in correlation form, so one `λ` means the same whatever
the columns' units. A grid of penalties shares one eigendecomposition:
with `C = Q diag(e) Q'`, each penalty's slopes are
`Q diag(1 / (e + λ)) Q' c`. That holds where `C`'s diagonal is even, as in
correlation form. An eigendecomposition is accurate only relative to the
largest eigenvalue, so on columns of unequal scale `solve` scales `C` by
its diagonal first, as a Cholesky solve in effect does, and the features'
units then cost no digits. A uniform `λ` is not uniform on the scaled
matrix, so there each penalty takes a solve of its own.

**`coef_stats` computes least squares' standard errors from the Gram:**

```text
resid_var = V − 2 b'c + b'C b
sigma2    = resid_var · n / (n − p)      n = target_n_kish; p, the columns with variance and the intercept
se_j      = sqrt((C⁻¹)_jj · sigma2 / n)
se_0      = sqrt((1 + m'C⁻¹m) · sigma2 / n)      the intercept's, m the column means
t_j       = b_j / se_j
```

With `half_life=float("inf")` and every row's weight 1, `n` is the row
count, and these are the classical standard errors: the recipe of [Fit a
ridge from a Gram](how-to/fit-a-ridge-from-a-gram.md) gets the ones numpy
computes from the rows, to within 1e-9. The intercept's is
least squares' too: `ȳ` and the slopes are uncorrelated, so the variance
of `b_0 = ȳ − m'b` is `Var[ȳ] + m' Var[b] m`.

**Without an intercept the statistics are a fit through the origin's,
read from the raw moments.** There `C` is `E[z z']`, `c` is `E[z y]` and
`V` is `E[y²]`. So `resid_var` is the residuals' mean square, and `r2` is
uncentered, as statsmodels reports an OLS fit with no constant.

**The t carries least squares' assumptions, whatever fit it is given:**

| it assumes | so |
|---|---|
| independent errors of one variance | correlated or heteroskedastic errors leave the standard errors too small. Standard errors robust to both need each row's residual: `emit_robust_se` on the stream ([Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted)), or the spread of [per-block coefficients](#per-block-coefficients) |
| one coefficient for all the pooled rows | a coefficient that changes from week to week reads as firm a t as a constant one: the pooled t of 20.3 for `x25`, whose coefficient is drawn again each week, beside the 20.3 of `x15` in [Keep weekly Grams of hundreds of features](how-to/fit-weekly-grams-in-one-pass.md#keep-weekly-grams-of-hundreds-of-features) |
| least squares' covariance, `sigma2 · C⁻¹ / n` | a ridge's standard errors are that covariance at the ridge's residual variance. They leave out the shrinkage and its bias, so a ridge's t compares its coefficients with one another, and is no test |
| `n` independent rows | under a decay the rows weigh unequally and are not independent, and `n_kish` stands in for `n`. Read the t as a comparison there too |
| one intercept for all the pooled rows | on a Gram pooled within parts each part had its own, so the intercept's standard error and t are `nan` |

## Is the fit determined?

When columns are nearly dependent, the rows fix only their combined
effect, and the split among them moves with the noise. The fit then
predicts well, and its coefficients mean nothing. `po.gram.vif` and
`po.gram.condition` read this from a Gram, and `po.gram.correlation`
gives the matrix behind them. [Check that a fit is
determined](how-to/check-a-fit-is-determined.md) reads both on a Gram
with a planted dependency, and refits without it.

**A variance inflation factor says how far a column's variance is
inflated by the other columns.** `VIF_j = 1 / (1 − R²_j)`, where `R²_j`
is the R² of column `j` regressed on the rest (Marquardt 1970). It is the
diagonal of the inverse correlation matrix. A coefficient's standard
error is `√VIF_j` times what it would be if its column were uncorrelated
with the rest, so a VIF of 10 makes it 3.2 times as wide. Read a VIF past
1e10 as an exact dependency: rounding can leave it finite, as
[`vif`'s reference](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.vif)
explains.

**Belsley's condition indexes say which columns are involved.**
`po.gram.condition` scales the design to unit column length, without
centering it, and takes its singular values `s`. Each index is
`s_max / s_j`. The variance-decomposition proportions give each column's
share of its coefficient's variance in each component. Belsley, Kuh and
Welsch (1980) read an index above 30, with two or more columns' shares
above 0.5 in its component, as a dependency among those columns. The
design is left uncentered so that a dependency on the intercept shows,
such as a column whose level is far above its spread.

## Choosing columns on the lasso path

The lasso adds `λ Σ_j |b_j|` to the squared loss (Tibshirani 1996). At a
large enough `λ` every slope is 0, and as `λ` falls, columns enter the
fit. `po.gram` follows that path two ways. `lars_path` walks it knot by
knot from its start, by least angle regression. `lasso_path` solves it at
the penalties given, by coordinate descent.

**Both solve the lasso on the Gram's correlation form.** With `R` the
features' correlation matrix and `d` their standardized cross-moments
with the target, centered at its own means, the lasso minimizes

```text
f(b) = ½ b'R b − d'b + λ Σ_j w_j |b_j|
```

with `w_j` a scale on column `j`'s penalty, 1 unless `penalty_weights=`
gives it. The coefficients come back in the columns' own units, with the
intercept recovered. The solution is piecewise linear in `λ`. A knot is a
`λ` at which the active set changes, as columns enter it, or leave it
when their coefficients reach 0 (Efron, Hastie, Johnstone and Tibshirani
2004).

**Least angle regression walks the path from its start, and coordinate
descent solves a grid of penalties.** `lars_path` stops after `max_iter`
knots past its first, or at the first knot with at least `max_active` columns active, so
it suits a selection: the first columns to enter. `lasso_path` takes what
a path of knots cannot: a weight `w_j` of 0, which keeps a column in the
fit at every penalty, and an elastic net (`l1_ratio` below 1), whose path
is not piecewise linear (Zou and Hastie 2005). [Choose columns on a
Gram's lasso
path](how-to/choose-columns-on-the-lasso-path.md#choose-columns-on-a-grams-lasso-path)
compares the two, with their times, and walks a path on four weeks.

**At a knot, LARS gives the lasso's coefficients exactly, and coordinate
descent reaches them to its tolerance.** `lasso_path`'s `tol` bounds the
largest move of a sweep, as a share of the target's spread: its standard
deviation, read from the Gram's target moments, or its root mean square
without an intercept. The standardized coefficients a sweep moves are in
the target's units, so the rule has none: a target scaled by a power of
two, with the penalties, gives the same path scaled, to the bit. On a poorly
conditioned Gram the rule says little about the distance from the
solution: at a condition number of 1e8, a descent at `tol=1e-14` stopped
with its intercept 0.48 from the knot's ([`lars_path`'s
reference](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.lars_path)). The
descent follows Friedman, Hastie and Tibshirani (2010).

**A penalty whose descent runs out of sweeps is named in a warning.** Its
row is where the descent stopped, and a `ReadinessWarning` lists each such
penalty. A cyclic descent shrinks the error along two columns of
correlation `ρ` by `ρ²` a sweep. Their correlation matrix's condition number
is `κ = (1 + ρ)/(1 − ρ)`, so `ρ² = ((κ − 1)/(κ + 1))²`, about `1 − 4/κ`, and
each e-fold of the error takes about `κ/4` sweeps. On columns that nearly
repeat each other that is many: give a larger `max_iter`, or take the knots from
`lars_path`.

## Stability selection over halves of the weeks

A lasso path on one sample chooses a set of columns, and another sample
can choose another set. Stability selection runs the choice on many
halves of the data and keeps the columns chosen on most of them
(Meinshausen and Bühlmann 2010). With a Gram per week, a half of the
weeks is one merge. So a hundred halves are a hundred merges and paths,
and no pass over the rows.

**Each half chooses every column that is active at a knot of its path, up
to the path's stop.** That is Meinshausen and Bühlmann's choice over a
range of penalties: the union of the active sets, which a path stopped at
`max_active` gives. For column `j`, `π_j` is the share of the halves that
chose it, and the stable set is every column with `π_j` at or above a
threshold `π_thr`.

**The threshold bounds the expected number of false selections.** Let
`q` be the average number of columns a half chooses, and `p` the number
of columns. The count `F` of columns with no effect in the stable set
then has an expected value of at most

```text
E[F] ≤ q² / ((2 π_thr − 1) p),      for π_thr above 1/2
```

Meinshausen and Bühlmann prove it when the columns with no effect are
chosen exchangeably, and the choice does no worse than guessing at
random. Shah and Samworth (2013) proved a bound of the same form with no
exchangeability, when each half is drawn beside its complement. Theirs
counts the columns whose chance of being chosen on a half is at most
`q / p`. [Select columns that hold across the
weeks](how-to/select-columns-stably-over-weeks.md) draws its halves that
way. At `q = 8`, `p = 200` and `π_thr = 0.75`, the bound is 0.64.

**The halves are drawn from whole weeks.** The theorem's halves are halves of
independent observations, and the rows of a stream are rarely
independent of their neighbors. A half of the weeks keeps each week's
rows on one side, so the weeks become the units being sampled, and the
assumption becomes that the weeks are alike. A run of weeks in another
regime from the rest breaks it.

**Stability selection bounds false selections, and says nothing about
whether a chosen coefficient holds.** [Select columns that hold across
the weeks](how-to/select-columns-stably-over-weeks.md) runs the selection
on a hundred halves of forty weeks. There `x25` has a coefficient that is
a different number each week around a nonzero average, and every half
chooses it. [Per-block coefficients](#per-block-coefficients) is the
reading for that.

### A null for the selection, by sign flips

**A null run negates each week's target at random and runs the selection
again, to see what it chooses where no effect keeps its sign across the
weeks.** Stability selection's bound, in Shah and Samworth's form, counts
only the columns with no effect whose chance of being chosen on a half is
at most `q / p`. A path that always stops at eight columns, among columns
that share families, can choose some of them more often than that. A null
run measures the selection instead. It draws a sign for each week,
negates that week's target, and reruns every half. A column whose
effect keeps its sign across the weeks loses that in a null run. The run
keeps everything else: each week's columns, their correlations, and the
size of each week's effects. Over many draws, the largest selection
frequency in each null run gives a threshold. Only 5% of the null runs
choose any column on more halves than its 95th percentile. That is the
maximum statistic of a permutation test by sign flips (Winkler et al.
2014), with a week as the unit that is flipped.

**Pooled within weeks, a negated target changes only the cross-moments,
so a null run needs no new merge of the co-moments.** Negating a week's
target negates its centered cross-moments `c_i`. Its co-moments, its target
weight and its variance stay as they were. So a draw of signs `s_i` gives
`c = Σ T_i · s_i · c_i / T`, and nothing else in the pooled Gram moves.
`po.gram.solve_subsets(..., within=True, target_signs=signs)` takes the
signs as an array of weeks by draws, and returns one run of every subset
per draw. Pooled as a union, a negated week's target mean would also move
the gaps between the weeks' means, so `target_signs` is refused without
`within=True`. [Set the threshold from null runs, by sign
flips](how-to/select-columns-stably-over-weeks.md#set-the-threshold-from-null-runs-by-sign-flips)
runs a hundred null runs. With each week's sign drawn at random, the most
chosen column of a null run there is still chosen on 0.81 of the halves at
the median draw. Each half chooses eight columns whatever the signs, and
a column with a large effect in some weeks keeps being chosen when its
sign changes from week to week. So the threshold of 0.75 that the
selection without null runs used passes such a column in 76 of the 100
null runs. The null runs' 95th percentile, 0.95, is the threshold the
selection earns on those weeks.

### A screen of its own for each path

**A selection that screens its columns first must screen again in every
half and every null run.** A screen keeps the columns most correlated with
the target, and a path then chooses among them. The screen is part of the
selection, so a half's screen reads that half's rows. A null run screens
under its own signs too, since a screen read from the real target would
carry the target's signs into the run that is meant to have none.

**A screen pooled within the weeks reads a vector per week, so a screen
for every half and draw merges no matrix.** A column's correlation with
the target, pooled within a half's weeks, needs only each week's weight,
its cross-moments under the draw's signs and its columns' variances. So
`po.gram.solve_subsets(..., jobs=...)` takes each path's own columns, and
still merges each half once, on every column its jobs read, and re-pools
its cross-moments once a draw. [Screen the columns again for each half
and each null
run](how-to/select-columns-stably-over-weeks.md#screen-the-columns-again-for-each-half-and-each-null-run)
runs 2,100 paths, each on a screen of its own.

## Per-block coefficients

Fitted alone, each week's Gram gives that week's coefficients. A pooled t
assumes one coefficient for all the pooled rows, and the spread of the
weekly coefficients shows whether there is one.

**Fama and MacBeth's (1973) t reads the weekly coefficients as a
sample.** With `b_w` the coefficient of week `w` out of `B`, it is the
mean of the `b_w` over their standard error, `sd(b_w) / √B`. That
standard error grows with the coefficient's own variation from week to
week, which a pooled standard error leaves out, and with any correlation
of the errors within a week. It assumes the weekly coefficients are
independent of one another. When they drift slowly, take Newey and
West's (1987) long-run variance of their series in place of `sd(b_w)²`.

**A week fits only the columns already chosen.** A week of 504 rows fits
201 columns poorly, so each week is fitted on a few columns chosen first.
[Fit per-block coefficients](how-to/fit-per-block-coefficients.md) fits
six chosen columns each week alone. There `x25`, whose coefficient is
drawn again each week, has a pooled t of 26.2, as firm as `x15`'s 26.3,
and a t from the weeks of 4.1.

## A compact Gram

`comoments` (and, for an `ew_cov` with `lags`, `lag_comoments`) are the
arrays of a Gram that grow as `k²`: `comoments` is 800 MB a block in
float64 at 10,000 columns. `bank.gram()` can return them in float32
(`dtype="float32"`), which rounds each entry to within a relative `2⁻²⁴`,
about 6e-8, and `comoments` as its upper triangle (`layout="packed"`),
which drops the lower one. Each takes about half the bytes of
`comoments`, and both together a quarter. `lag_comoments` is never
packed, since a lagged matrix is not symmetric, so with both options together it takes half its bytes, from float32 alone.
The state itself stays float64 and whole. To choose a form and check what
it cost a fit, see [Store a compact Gram](how-to/store-a-compact-gram.md).

**A float32 Gram is enough when `κ` times 6e-8 is below the precision
the fit needs.** A system whose entries carry a relative error `δ` gives
a solution whose relative error is of the order of `κδ` (Higham 2002).
Here `κ` is the condition number of the matrix solved, its largest
eigenvalue over its smallest. For a Gram that matrix is the features'
correlation matrix, which a standardized solve works on:
`np.linalg.cond(po.gram.correlation(g)[1:, 1:])`. Rounding moves a
solve by about `κδ` at the worst, and usually by far less, since the
errors of many entries offset one another. On the stream of [Store a
compact Gram](how-to/store-a-compact-gram.md), a float32 Gram moved the
fit by 7e-9 of its largest coefficient, where its condition number of 7.8
allowed 4.7e-7.

**A packed Gram solves to the same bits as the full one,** because the
model keeps only the lower triangle and reads the upper one as its
mirror, so the matrix is symmetric by construction.

## One wide Gram on several threads

Updating a Gram takes `O(k²)` time a row, and at a few thousand columns
that update is the whole pass. `gram_threads=n`, on `ewridge` and
`ew_cov`, runs it on up to `n` threads, and the Gram comes out the same
to the bit at every `n`. The default, `"auto"`, picks `n` from the bank's
threads and the groups it runs at once. To set it, with `gram_block_rows`,
for a fit over hundreds of features, see [Fit hundreds of
features](how-to/fit-hundreds-of-features.md).

**The bits do not move, because each entry is summed by one thread in
one order.** The update is cut into pieces fixed by `k` alone: diagonal
blocks of 256 columns, and tiles of 128 to 255 columns a side. Each piece
is one sequential call on one thread, whatever the number of threads. A
fit's factorization and solves are cut by shape the same way, and its
eigendecomposition runs at a fixed parallel degree of eight. So a
coefficient's bits do not follow the number of the bank's threads either.
Tests hold both: the Gram's pieces against one call at widths 1 to 520,
and each solve in sets of 1, 3, 8 and 14 threads.

**`gram_threads` takes its threads from the bank's own, the threads it
runs its groups on.** The bank's `threads` sizes them, or, without it,
`POLARS_ONLINE_MAX_THREADS`, one thread per core by default. A count caps
how many of them one spec's update may use. A bank whose groups already
keep every core busy gains nothing from a split, and loses the cost of
handing each row's pieces out. So `"auto"` splits only where the bank has
idle threads. On each
[chunk](GLOSSARY.md#the-model-bank-and-what-it-reads) the bank counts the
pieces of work it runs side by side, one per spec and group, `T`, on `P`
threads. When `2T ≤ P`, each piece's update may use `2P / T` threads,
rounded down and at most `P`. Otherwise it uses one. Twice the even share
lets a thread whose group is done take pieces of another's. The rule reads
the two counts and nothing else, so a run splits the same way whatever the
timing. With fourteen groups of 1,000 columns, eight threads of a separate
set took 28% longer than eight of the bank's own, since they added eight
threads to a machine whose cores were already busy. A bank of weekly
blocks fed a stream in time order often reads a chunk that holds one
week's rows, and so has one group to run. There `"auto"` gives that group
all the bank's threads.

The table sets `"auto"` against `gram_threads=1`, on `ewridge` unless
named. Each cell is the time of one thread over the time of `"auto"`: the
median of 11 interleaved rounds of `fit_predict` by a bank that starts
with nothing learned, in each of two runs. They were measured on an Apple
M4 Pro of 14 cores (ten performance, four efficiency), at a load average
of 7 to 12 (the number of processes running or waiting to run, averaged
over a minute):

| bank | `T` | `"auto"` splits into | `1` / `"auto"`, two runs |
|---|---|---|---|
| one group, 2,000 features, 400 rows | 1 | 14 | 2.93, 3.00 |
| the same with `gram_block_rows=256`, 1,024 rows | 1 | 14 | 3.56, 3.51 |
| 4 groups, 800 features, 8,000 rows | 4 | 7 | 1.66, 1.58 |
| 7 groups, 800 features, 7,000 rows | 7 | 4 | 1.09, 1.00 |
| 14 groups, 800 features, 14,000 rows | 14 | 1 | 0.98, 1.00 |
| 28 groups, 800 features, 28,000 rows | 28 | 1 | 1.01, 1.00 |
| the same on `ew_cov` | 28 | 1 | 0.99, 1.00 |
| 56 groups, 800 features, 28,000 rows | 56 | 1 | 1.00, 0.99 |
| 64 groups, 500 features, 64,000 rows | 64 | 1 | 1.01, 0.99 |
| 64 groups, 20 features, 1.6 million rows | 64 | 1 | 1.05, 0.99 |

Where `"auto"` splits into one thread it runs what `gram_threads=1` runs,
so those rows measure the noise, a few percent either way. Taking all the
bank's threads at any `T` instead, 28 groups at 800 features ran 0.89
times as fast as one thread on `ewridge` and 0.91 on `ew_cov`. At eight to
13 groups every count from 2 to 14 was even or up to 10% slower, which is
why the rule stops splitting at `2T > P` rather than at `T ≥ P`.

**A split helps past 256 columns with `gram_block_rows`, and past 725
without.** Up to 256 columns, a block's product is one piece. Without a
block, a row's update is bound by memory, and handing it out costs
microseconds a row, so below 725 columns it stays on one thread. A whole
bank gains less than its Gram's update, since the rest of a row's work,
`O(k)`, stays on one thread.

## Huber regression on a Gram

A Huber fit down-weights the rows that miss by far, so its weights depend
on each row's residual, and no single Gram holds it: it reads the rows
again, twice a pass, once for the scale and once for the moments. It does not need a Gram for each pass or for each
target. Huber's algorithm H (1981) reuses one Gram, that of the rows at
their own weights, in every pass. To run it over a file, see [Fit a Huber
regression on a Gram](how-to/fit-huber-on-a-gram.md).

**Weighting only the targets' cross-moments is not weighted least
squares.** Giving each row its Huber weight `w_i` in `x·y` and not in
`x·x'` solves `b = (X'X)⁻¹ X'W y`. A row it down-weights is not pulled
toward the fit: its target is shrunk toward 0, by the whole level of a
target that has one. On the stream of [Fit a Huber regression on a
Gram](how-to/fit-huber-on-a-gram.md), a target at 1,000 with a slope of 2,
that iteration reaches a slope of −59.4 after 30 passes. Weighted least squares weights `x·x'`
too, so it needs a Gram per target in every pass.

**Algorithm H moves the target instead of the weights.** With the fit `b`,
each row's residual `r = y − x'b`, a scale `s` and the cut `c`, the
pseudo-response is

```text
y* = x'b + s · ψ(r / s)        ψ(u) = u clipped to [−c, c]
```

which is the target where the residual is inside the cut, and the fit
plus or minus `c·s` beyond it. Its least-squares fit, on the rows at
their own weights, is the next `b`. That fit's matrix is the Gram of the
rows, which no pass changes, so it is factored once. A pass reads the
scale in one query over the rows, and `y*`'s mean and its centered
cross-moments with the columns in a second. At a fixed point the fit of `y*` is `b`, so
`Σ x ψ(r / s) = 0`, which is Huber's estimating equation. Each target has
its own `y*`, and all of them share the one Gram.

**Start from a fit that has the target's level,** such as least squares,
which the Gram gives, or the target's median alone, an intercept and no
slopes. From `b = 0` every residual is
the level. A scale read again from those residuals is the level's too, so
nothing is clipped and the first pass is least squares, a pass lost. A
scale taken from elsewhere, the noise's or a `huber` spec's
`huber_scale_<t>`, clips every row instead, and each pass then moves the
fit by at most `c·s`: from 0 at `s = 0.5`, a level of 1,000 takes about
1,500 passes.

**Read the scale again in each pass.** The MAD,
`median(|r|) / 0.6745`, is the noise's standard deviation when the
errors are normal, and holds until half the rows are outliers. A median
is a query of its own over the rows, which holds a value a row while it
runs, and two blocks' medians do not combine into their union's. Huber's Proposal 2 is a scale built from
sums, which do combine: the `s` that solves `mean(ψ(r/s)²) = E[ψ(Z)²]`,
0.7102 at `c = 1.345` for a standard normal `Z`. It holds nearly as well as
the MAD up to a fifth of the rows wild, and breaks sooner. The stream is
`y = 1000 + 2x + 0.5ε` over 20,000 rows, with +50 on a share of the rows
with `x` above 0. Each fit is statsmodels' `RLM` with `HuberT(1.345)`,
iterated to a tolerance of 1e-10, with `scale.mad` for the MAD and
`HuberScale(d=1.345)` for Proposal 2, at the cut `c = 1.345`. Measured
on 2026-10-24 with statsmodels 0.15.0, as the median slope over seeds 0 to
9 (the truth is 2):

| rows wild | least squares | MAD | Proposal 2 |
|---|---:|---:|---:|
| 12.4% | 6.960 | 2.124 | 2.129 |
| 20% | 9.975 | 2.317 | 2.392 |
| 25% | 11.978 | 2.889 | 11.825 |

At statsmodels' default cut for the scale, `HuberScale()`'s `d = 2.5`,
Proposal 2 breaks before 20%: 8.556 there.

**Algorithm H converges to statsmodels' Huber fit, in more passes than
weighted least squares.** On the stream of [Fit a Huber regression on a
Gram](how-to/fit-huber-on-a-gram.md), with 5% of the rows wild, its slope
reaches 2.0393 in ten passes from least squares or from the median, the
slope of statsmodels' `RLM` at the same scale. Weighted least squares, a
new Gram in each pass, reaches it in five. A `huber` spec, which weighs
each row once in one pass over the stream, gives 2.0394 there.

## Sources

Each claim of the explanations that is not measured rests on one of
these.

| source | for |
|---|---|
| Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). *Regression Diagnostics*. Wiley. | condition indexes and variance-decomposition proportions |
| Chan, T. F., Golub, G. H. and LeVeque, R. J. (1983). Algorithms for computing the sample variance: analysis and recommendations. *The American Statistician* 37(3), 242–247. | the pairwise update that pools two sets of moments, and the precision a raw sum loses at a level |
| Efron, B., Hastie, T., Johnstone, I. and Tibshirani, R. (2004). Least angle regression. *The Annals of Statistics* 32(2), 407–499. | least angle regression and its lasso modification, and the lasso path's knots |
| Fama, E. F. and MacBeth, J. D. (1973). Risk, return, and equilibrium: empirical tests. *Journal of Political Economy* 81(3), 607–636. | a coefficient's t from its estimates period by period |
| Fisher, R. A. (1922). On the mathematical foundations of theoretical statistics. *Philosophical Transactions of the Royal Society of London, Series A* 222, 309–368. | sufficiency |
| Friedman, J., Hastie, T. and Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software* 33(1), 1–22. | coordinate descent for the lasso and the elastic net, warm-started along a path |
| Frisch, R. and Waugh, F. V. (1933). Partial time regressions as compared with individual trends. *Econometrica* 1(4), 387–401. | a regression's slopes with a constant per block, from the rows less their block's means |
| Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. | the error a perturbation of a linear system's entries carries into its solution |
| Huber, P. J. (1981). *Robust Statistics*. Wiley. | algorithm H, the pseudo-response iteration on a fixed matrix, and the scale of Proposal 2 |
| Hoerl, A. E. and Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics* 12(1), 55–67. | the ridge |
| IEEE (2019). *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019. | float32's rounding and range |
| Kish, L. (1965). *Survey Sampling*. Wiley. | the effective sample size of unequal weights |
| Lovell, M. C. (1963). Seasonal adjustment of economic time series and multiple regression analysis. *Journal of the American Statistical Association* 58(304), 993–1010. | the general form of that theorem: the slopes with any set of columns partialled out first |
| Marquardt, D. W. (1970). Generalized inverses, ridge regression, biased linear estimation, and nonlinear estimation. *Technometrics* 12(3), 591–612. | the variance inflation factor |
| Meinshausen, N. and Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society, Series B* 72(4), 417–473. | stability selection and its bound on false selections |
| Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708. | the variance of a series of weekly coefficients that drift |
| Shah, R. D. and Samworth, R. J. (2013). Variable selection with error control: another look at stability selection. *Journal of the Royal Statistical Society, Series B* 75(1), 55–80. | halves drawn beside their complements, and the bound without exchangeability |
| Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B* 58(1), 267–288. | the lasso |
| Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics* 4(3), 419–420. | the centered update of each row |
| West, D. H. D. (1979). Updating mean and variance estimates: an improved method. *Communications of the ACM* 22(9), 532–535. | the same update with weights |
| Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press. | the within, or fixed-effects, estimator (chapter 10) |
| Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M. and Nichols, T. E. (2014). Permutation inference for the general linear model. *NeuroImage* 92, 381–397. | sign flipping as a permutation test, flipping whole blocks, and the maximum statistic over many tests |
| Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society, Series B* 67(2), 301–320. | the elastic net |