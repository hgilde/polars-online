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
