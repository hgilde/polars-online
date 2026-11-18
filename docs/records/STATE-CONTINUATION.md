<!-- kind: record -->
# The exceptions to state continuation (CLAUDE.md hard rule 5)

Each is the user's decision, numbered in the order given; a code comment or
test that cites "rule 5's Nth exception" means the Nth below. Moved
verbatim from CLAUDE.md on 2026-11-14 (the user: cut the size of the docs
read most). A new exception is added here, numbered next, and counted in
CLAUDE.md's rule 5.

One exception, the user's (2026-10-24, task 275): a *windowed* state of
schema 61 or 62 continues within the golden tolerance, not to the bit,
once a snapshot taken after its load is the window's edge (those states
never recorded the per-row decay factors); its first read is exact.
A second, the user's (2026-10-25, task 278): a `kmeans` state of schema
61-63 saved before it seeded seeds by the new rule after loading -- the
same centres and distances to the bit, the cluster labels possibly
permuted, held to one consistent relabelling.
A third, the user's (2026-10-25, task 278): a `lasso` state of schema
61-63 continues within 1e-9 relative, not to the bit (the stopping rule
became relative to the target's spread), and still loads and re-saves the
same bytes.
A fourth, the user's (2026-10-25, task 278): a diagnosed state of schema
61-63 continues to the current values in its windowed `cusum_*` and
`cusum_sq_*` fields while their sums are young (their null's share is now
read at the sums' age); every other field continues to the bit.
A fifth, the user's (2026-10-26, task 278): a `holt` state of schema
61-63 loads with no line fit and continues with the textbook gains it
was fitted under, within 1e-14 in its predictions and state, and its
diagnostics derived over many rows within the golden 1e-12 (to the bit
until task 291a's kept reciprocals, the user's, 2026-11-01); its re-saved bytes cannot be the current
fixture's (which carries the line fit), so it is held, beside `kmeans`,
to re-saving bytes that load back the same.
A sixth, the user's (2026-11-01, task 294): the co-moment matrix every
`EwCov` keeps is now stored as its lower triangle and read through its
mirror, so a state of schema 61-63 continues within rounding, not to the
bit, where it read an upper entry: an `ew_cov` state's covariances and
correlations within 1e-15 by |Δ|/(1+|x|), and a diagnosed state's
specification, calibration, break and `ic` statistics (and a closed
group's exported `comoments`) within the golden 1e-12; every prediction
continues to the bit, `lasso` stays inside its own 1e-9, and each state
still re-saves the current fixture's bytes exactly.
A seventh, the user's (2026-11-03, task 311): an `ewridge` state saved
by an earlier build no longer runs a solve its weight schedule brings due
while the weight is below `(1 − s)(1 − 2⁻¹⁶)` of its lowest floor. It
continues with every residual, spread, weight and Gram to the bit, and
every prediction to the bit except where a target's weight has decayed to
exactly 0 below the floor or a fit's factorization fails at every rung:
there a row may read the last fit solved rather than the skipped one.
Until the next solve run, its `coef`, `support_coef` and `se_coef` report
the file's fit and `solve_failures` counts only the solves run, so the
state it re-saves after those rows is not the old build's. No frozen
fixture reaches it.
An eighth, the user's (2026-11-05, task 323; amended 2026-11-05, task
340): a state of schema 64 or earlier that holds a MAD scale sketch
(`huber`, `quantile`, `sgd(loss="huber")`, `marginal`'s huber bins)
keeps the rule it was saved under -- a sketch awaiting its anchor takes
its first value, a warm-up with no held residuals reads its sketch --
marked in its bytes (`anchor_by_first_value`, `before_65`). It continues
to the bit, but re-saves to its own marked bytes, not the current
fixture's, so it is held, beside `kmeans` and `holt`, to re-saving bytes
that load back the same (`kept_scale_rule`). A converted `marginal`
state re-saves so only until its histogram is built: `before_65` acts at
the freeze and is cleared there (schema 67), so from the freeze it
re-saves the current form; a state of 65 or 66 saved past its freeze
still carries the mark and loads without it, converting to the current
fixture's bytes exactly (`marginal_bins_refit_first_value_frozen`, kept
under `v66/` by the 66 build).
A ninth, the user's (2026-11-05, task 328): a `corrchange(kind="window")`
state of schema 65 or earlier loads numbered from its ring's oldest row
(it never recorded `ring_rows`), and where its windows span four blocks
or more its statistic is merged from block moments rather than read from
the rows, so `stat` continues within the merged form's derived rounding
bound (held at 1e-14 of `1 + |stat|`, measured 4e-16), not to the bit;
every other output continues to the bit, a kept `permute_every_rows`
stays as saved, and its re-saved state equals the current fixture's but
for `ring_rows` (`RENUMBERED`).
A tenth, the user's (2026-11-05, task 327): a windowed `ew_class` under
`covariance_type="diag"` now counts each snapshot class at its k
variances, not k² co-moments, against `window_budget`, so a state of
schema 65 or earlier that had reached its budget thins or refuses later
than it would have, and from that row does not continue to the bit. No
frozen fixture reaches it; every state below its budget continues to the
bit.
An eleventh, the user's (2026-11-05, task 331): `refit_budget`'s
default rose from 256 to 320 MiB, and the spec keeps only a budget the
user gave, so a bank saved under the old default reads its saved
reservation against 320 MiB when loaded. The groups it gave up stay
given up and every group it holds continues to the bit, but a group it
first sees after the load refits until 320 MiB is reserved, where the
old build would have given it up. No frozen fixture reaches the
default budget.
A twelfth, the user's (2026-11-08, task 364): an `ewridge` state
saved by an earlier build no longer runs a solve its row cap brings due
under a weight share (the default schedule under a finite half-life)
while the weight is below `(1 − s)(1 − 2⁻¹⁶)` of its lowest floor, as it
already ran none the weight brought due there. It continues with every
residual, spread, weight and Gram to the bit, and every prediction to
the bit except in the seventh exception's two cases. Until the next
solve run, its `coef`, `support_coef` and `se_coef` report the file's
fit and `solve_failures` counts only the solves run, so the state it
re-saves after those rows is not the old build's. A given
`solve_every`'s clock and row cap still run every solve they bring due.
No frozen fixture reaches it.
A thirteenth, the user's (2026-11-09, task 368): a conformal tracker
now keeps a multiplier `m` of the residual spread, the interval being
`pred ± m·σ`, where a state of schema 66 or earlier holds the radius
`q`. Such a state loads with `q` kept, re-saves it, and shows it to the
bit on its next row; that row, if it reads an error scale, converts it
to `m = q / σ` and steps `m`, so from there its intervals and coverage
go on under the multiplier, not to the old build's, and the state it
re-saves is not the old build's. Every other field continues to the
bit. No frozen fixture holds a conformal tracker;
`a_conformal_radius_of_schema_66_is_shown_then_converted` holds the form,
written from a current bank, and `tests/test_released_state.py` each
release's file (`CONVERTED`: each group's first row to the bit).
A fourteenth, the user's (2026-11-15, task 405): a `lasso` coordinate
descent that reaches `max_iter` while its Gram has too few effective rows
(Kish n) for its features to settle in that many sweeps on any data -- the
smallest eigenvalue such rows can give, by the Marchenko-Pastur edge, is
below what `max_iter` sweeps need to reach `tol` -- is no longer counted in
`solve_failures`. A `lasso` state saved by an earlier build loads and
continues with every prediction, coefficient and other field to the bit;
its `solve_failures` grows by the counted cap hits only, so where a later
descent of its warm-up runs out of sweeps the count it re-saves is lower
than the old build's would be. The current fixture `lasso_window`
regenerated by one byte (its count 2 to 0); no schema bump.
A fifteenth, the user's (2026-11-17, task 457, landed in task 461): a
window operator's held mass -- the time a row's value is held over, `c (1 -
2^-x)` of the gap `x` half-lives before it, the partner of that row's
discount -- is now taken from the discount's own `exp2` for a gap of at
least a hundredth of a half-life (one `exp2` for a row's step and its mass),
where it was `exp_m1` at every gap; below a hundredth of a half-life it is
`exp_m1` still, so every row of such a gap is as it was. The two forms agree
to a few ulps of the mass (at most `2^-52 / (1 - d)`, 1.6e-14 at the edge and
less above), which is the whole difference: outputs of `ewm_mean`,
`ewm_std`, `ewm_var` and the mean and variance of a windowed operator move by
a median of a few ulps over the rows that move, and by at most about 1e-13
relative (1.05e-13 in the landing's re-measurement over 188 cases and 90M
rows, at a floor of 0.01 on the value, the largest at a 50 ns half-life and
1 ns steps; 52 cases move, and none flips a null, a non-finite value or a
decision), where `ewm_sum`, `ewm_rate` and `ewm_quantile` do not move.
A window run's state, or a bank's holding a window formula, saved by an
earlier build loads and re-saves the same bytes; its queue's sums hold the
bits the old form made for the rows before the save, so it continues within
that bound of the old build's run, not to the bit, once a row after the
load has a gap of a hundredth of a half-life or more, and the sums it
re-saves after that row are not the old build's. The user took the change
for its speed on 2026-11-17 (task 457: a window operator on an irregular
clock 14-16% faster on arm64 with one operator, 0.83-0.96 of the time on
x86; outputs moving by a median of 1-2 ulps and at most about 7e-14), and
the regeneration below is under this exception. The window digest's pinned
bits and the current `with_windows` and `with_windows_int` fixtures were
regenerated (no layout change, no `SCHEMA_VERSION`); no other fixture moved
(`with_windows_quantile` and every bank fixture re-save as before). The kept
previous-schema fixtures of window runs (`v62/`, `v63/`, `v66/`'s
`with_windows*`) are held to the golden 1e-12 against their frozen frames
and to re-saving bytes that load back the same, where the current set stays
to the bit (`moved_by_the_held_mass` in `state_fixtures.rs`).

A sixteenth, the user's (2026-11-17, task 457; built in task 462):
`bocpd` under the diagonal covariance, `covariance_type = "diag"` (with
`robust_beta` as well), and two features or more, takes the logarithms of
its predictive differently. It adds the features' `ln ψₙ` and
`ln(1 + z²/νₙ)` as the logs of products of blocks of 8 values (a block's
logarithm once, where its product is a normal number far from overflow, else
one at a time), and holds `ln(π(κₙ+1)/κₙ)`, the term `ln(νₙπ·var)` splits
into beside `ln ψₙ`, once for each run length (keyed by the run's weight),
where it took two logarithms for each feature of each run. The forms are
equal in exact arithmetic and differ by rounding, which the weight
`(νₙ+1)/2` of the second term scales. A diagonal `bocpd` state of two
features or more saved by an earlier build loads with every stored number as
it was and continues with `loglik`, `p_change`, `run_mean` and the
predictions parting from the old build's in their last digits, not to the
bit. Measured on 16 diagonal cases (2 to 200 features, 1, 100 and 10,000
groups, up to a million rows, a save and load, nulls) against the build
before: `pred` moved at most 2.0e-6 relative (a changepoint row; p99.9 at
most 5.0e-10, median at most 6e-14), `loglik` at most 2.0e-10 (median at
most 9.7e-14), `p_change` and `run_mean` at most 9.3e-10, with `robust_beta`
at most 1.1e-7 (p99.9 3.4e-10); no `run_mode`, `weight_sum`, withheld reason
or null changed, and no row crossed `p_change > 0.5` or `> 0.1`. The state it
re-saves is not the old build's. `bocpd` with one feature, with
`covariance_type = "full"`, and every other model continue to the bit, and
their states are the old build's byte for byte. The current fixtures `bocpd`
and `bocpd_warming` regenerated (their continuations moved by at most
1.4e-15 relative, each saved state by the last bit of one float); no schema
bump. The kept sets' two cases are held to that bound by `MOVED`
(`crates/online-core/tests/state_fixtures.rs`).

A seventeenth, the user's (2026-11-17, task 457: "keep the corrchange
improvement", after the per-seed difference below was put to them; built in
task 464): `corrchange(kind = "window")` draws the swaps of its permutation
critical value by Lemire's multiply-shift reduction (the high word of a
64-bit draw times the bound, with a draw whose low word is under
`2^64 mod n` redrawn, so it is exactly uniform) where it took the draw
modulo the bound (`x % n`, a 64-bit division a draw, biased by up to
`n / 2^64`). The generator is called once a draw as before, and a redraw has
probability under `n / 2^64`, so a state's saved generator is the old
build's after the same draws and nothing about the layout changes. What
moves is which permutations a seed visits: for a given `seed` and rows, every
redrawn critical value is a different draw from the same distribution, so it
parts from the old build's, and a flag whose statistic sits near the old or
the new critical value can differ. A window-kind state saved by an earlier
build loads with every stored number as it was, including the critical value
it holds until its next redraw, and its next redraw, from its saved
generator, takes the new draws; so it continues to the bit until that
redraw and parts from the old build's critical values from there, and the
state it re-saves is not the old build's. `corrchange` with `kind = "monitor"`
or `"sequential"`, which draw nothing, and every other model continue to
the bit. Measured against the build before (2026-11-18, arm64, the harness of
task 452: a window of 20, 100 and 400 rows, 4
columns, `n_perm = 200`, `perm_block` 1 and 10, 8 pools of correlated
Gaussian rows with 2,000 seeds each, 16,000 critical values a case): the
distribution of the critical value over seeds is the same (two-sample
Kolmogorov-Smirnov distance median 0.020-0.030 and at most 0.044 where the
5% critical value at 2,000 and 2,000 draws is 0.043; mean difference at most
0.076 of a standard deviation; standard deviation ratio 0.96-1.05; each
0.13-0.15 standard deviations above the critical value of 20,000 draws, the
bias of an order statistic of 200 draws, equal for both); a given seed's
critical value, paired against the old build's, moves by a median of 3.6% to
4.1% of its value (none identical), p99 15% to 16%, at most 32%. Over 3,000
streams of 3,000 rows (3 columns, correlation 0.3, and 0.8 from the middle for
the alternative): per-row flag rate 0.0511 against 0.0512 at window 100 (null,
standard error 0.0005), 0.0935 against 0.0933 (alternative); 0.0522 against
0.0522 and 0.0685 against 0.0685 at window 50; 0.0566 against 0.0567 and
0.0928 against 0.0928 at window 100 with `perm_block` 10; the median delay to
the first flag after the change 33 against 33 rows (22 against 22, 32 against
31). 1.60% to 1.71% of rows carry another `flag` than the old build gave in
the same stream; no stream changes whether it ever flags. The current fixtures
`corrchange_window` (core) and `bank_diagnosed` (bank), whose saved critical
values and redrawn ones moved, regenerated, with the two oracle hash guards
`corrchange.n_perm` and `truth_recovery.corrchange_break` re-run and
re-pinned; no schema bump. The kept sets are held by bounds: core's
`corrchange_window` in `v65/` and `v66/` is in `MOVED` at 0.25 of `1 + |x|`
(measured: the saved `perm_crit` 0.115, the redrawn `crit` 0.140; a flipped
`flag` would read 0.5, so none may), and the bank's `corr_window` `crit` in
`v65/` and `v66/` `bank_diagnosed` is in `CONTINUES_WITHIN` at 0.25 (measured
0.063), every other field staying to the bit.
