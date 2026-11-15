<!-- kind: record -->
# The plan archive, part 5 of 8

Moved out of `docs/PLAN.md` on 2026-11-14 (task 398), text unchanged: the plan had
grown to 1.7 MB, more than a reader can hold, and most of it was finished work.
This part holds the full entry of each finished task and the text of each closed
section, in the order they stood in the plan. `docs/PLAN.md` keeps one line per
finished task, linking the entry here (`#task-N`), and every section heading with
a line saying where its text went (`#sNN`).

Earlier text is in [part 4](PLAN-ARCHIVE-4.md).

Later text is in [part 6](PLAN-ARCHIVE-6.md).

<a id="s11a"></a>
## 11a. Decisions made while implementing

- *Line 6 of Algorithm 1 was implemented wrongly first, and it matters.*
  `ν⁽ʳ⁺¹⁾_{t+1} = ν⁽ʳ⁾_t + u(xₜ)` with `ν⁽⁰⁾_{t+1} = ν_prior`: slot `j`
  holds exactly the `j` rows the hypothesis `rₜ = j` says precede the next
  row, so the slot pushed at the front holds **nothing**. Letting it take
  the row too puts one row of the old regime inside every "brand new run".
  The symptom was subtle — every run's estimated start was one row early,
  and a 20-σ row produced `p_change = 0.09` instead of `0.91`, because the
  hypothesis that it started a run was evaluated with the row before it
  already in the run. The longhand oracle had been written from the code and
  so agreed with it; the ANSWERS quotation of line 6 is what settled it.
  Both the Rust and the numpy oracles now write line 6 out.

- *Each run carries its own length.* The slot index stops being the run
  length the moment `truncate` drops a run from the middle of the vector or
  `max_run` folds the tail, and it never was one under a fractional row
  weight or the robust emission's. `Run.len` counts rows; `run_mode` and
  `run_mean` read it.

- *`min_periods` gates the report, never the update.* The first version
  returned early before computing the recursion, so row one of every group
  was silently dropped from the model. The default is `1.0`, which nulls
  exactly that row: `P(r ≤ 1)` is 1 there whatever the data, since no run is
  older than one.

- *The robust emission tempers the message as well as the statistics.*
  Weighting only what a run learns leaves the outlier declaring a
  changepoint (measured: `p_change` 0.85) with clean statistics behind it,
  which is the worst of both. Using the same `w(x) = (π/π(mode))^β` on the
  likelihood in the recursion makes a row that is atypical for *every* run
  multiply every joint by about 1, so nothing moves at all. That is the
  plan's acceptance test, and it now passes.

- *`robust_beta` is a trade, and 0.1 is the measured default.* A whole new
  regime is a run of individually forgiven rows, so tempering too hard blinds
  the model: at 0.05 and 0.1 a four-sigma shift is still found within five
  rows and dated to within one, at 0.3, 0.5 and 1.0 it is never found at
  all. Both halves are pinned, so the note cannot go stale.

- *`prior_scale` too large is silence, not false alarms.* The plan said to
  set it from the data's scale; measured, the failure mode is that a
  predictive that wide finds no row surprising, and a real break is never
  detected (a four-sigma break on variance-1e-6 data: dated exactly at
  `prior_scale = 1e-6`, invisible from 1e-3 up).

- *`hazard_col` is explicit.* The hazard rides in the target slot, and a
  `hazard_from_row` flag in the config says to read it — without one, a
  `bocpd` in a bank with a regression target read the target as a hazard and
  poisoned itself with a NaN.

- *The default emission is `"diag"`.* `O(runs·d)` against `O(runs·d²)`, and
  `"gaussian"` is what to reach for when the break is in the correlation
  with the marginals unchanged, which is a test in `tests/test_bocpd.py`.

*Task 56 — the schema-5 fixture and the close of the batch.*

- *Fixture.* `crates/online-polars/tests/state_schema5.rs` in
  `state_schema4.rs`'s form — hex bytes in the source, a `print_fixture`
  generator, the "what it says" / "loads with everything" / "continues to
  the bit" / "re-saves byte-identically" tests — from a bank whose specs
  cover the batch: a `group_close = "monotone"` spec with a closed row
  undrained and a `high_water`, an `ew_cov` with `lags` and a partially
  filled ring, and one of `deco`, `rcov`, `hmm`, `corrchange`, `bocpd`
  each mid-stream. `state_schema4.rs` stays untouched and must load through
  the schema-5 reader (rule 5's loader).

- *Docs.* Every §10 row gets its task number and "done" as §9's did; the
  README's model table and a `### <name>` per new model with a runnable
  block; `docs/REGIMES.md` complete with the three experiments (task 53's
  recovery, task 54's size/power/ARL, task 52's Epps curve); CHANGELOG
  `[Unreleased]` listing the schema bump first; `docs/RELEASE-READINESS.md`
  for the new surface. E63 stays a note: the `weight=` recipe it describes
  is already expressible, and nothing in this batch adds `weight_from`.

*Task 56 as built, 2026-09-06.*

- *The fixture.* `crates/online-polars/tests/state_schema5.rs`, six specs on
  a 60-row stream with a **monotone** group key: an `ew_cov` with
  `lags = [1, 12]` and `group_close = "monotone"`, an `rcov` (whose whole
  output is a closed group's block), a `deco`, a sessioned `hmm`, a
  `corrchange` mid-span and a `bocpd`. Two groups have closed and nobody
  read them, so the file carries the queue and the high-water key; the third
  group holds ten rows, which is fewer than the lag ring wants, so the ring
  is frozen partly filled with the long lag still empty. `state_schema4.rs`
  is untouched and still loads, continues to the bit and re-saves as 5.

- *One departure from `state_schema4.rs`'s shape.* Its `assert_same_state`
  compares coefficients with `assert_eq!`, which cannot work here: a gated
  instance's coefficient is a NaN and `NaN != NaN`. Schema 5 compares the
  debug rendering, where a NaN in the same slot is a match.

- *`docs/REGIMES.md` and `scripts/regime_experiments.py`.* Five experiments,
  35 seconds, committed rather than gate-regenerated (the
  `docs/CLUSTERING.md` §7 precedent). Three of them are the ones §11a asked
  for; the fourth compares `corrchange(window)` and `bocpd` on the same
  break, which is where the two detectors' trade shows, and the fifth is the
  Epps curve with an `L` sweep of the lag inversion.

- *And the size study is why the experiments were worth running.* The first
  draft compared Gaussian draws against WKD's `t₅` table and concluded that
  this implementation is more powerful than the paper. It is not: under
  either reading of their DGP it is *less* powerful once the size is
  adjusted, and its size at `|ρ| = 0.5` swings from `.075` to `.025`
  depending on which bivariate `t₅` is meant. Task 54's note and its gate
  test are corrected; the document reports all three DGPs side by side.

- *A sixth experiment, `dhat`, turns that from a hypothesis into a
  measurement.* `D̂` is recovered from the reported statistic (the longhand
  numerator divided by it) and compared with the closed-form elliptical
  value it estimates. Exact on Gaussian pairs, 15 % low and wildly scattered
  on a `t₅`, and no better at four times the sample. Both symptoms in the
  two sections above are that one number, and the document says so instead
  of calling them unexplained.

- *The seeds are `zlib.crc32`, not `hash()`.* Python randomises string
  hashing per process, so the first draft's numbers changed run to run. An
  experiment in a committed document has to give the reader the numbers it
  claims.

- *What the experiments found, beyond the tables.* `hmm`'s default seeding
  (k-means over the rows) **cannot find a regime that lives in the
  covariance**: it splits two zero-mean states by direction and lands at
  chance, where the same filter given the covariances is at 98 %. And a
  `precision_prior` of 1.0 on data of variance 1.0 halves every correlation
  and collapses the accuracy — `bocpd`'s `prior_scale` lesson in a second
  model. Both are in the document and in `hmm`'s docstring.

- *ENHANCEMENTS §10 needed nothing.* Every row already carried its task
  number and its departures, written as each task landed; E63 stays a note
  by design.

*Answers folded in, 2026-09-05.* `docs/ANSWERS-E54-E64.md` was read against
every "confirm from the paper" above, and the blocks were edited in place
rather than annotated, so a task block is still read top to bottom. What
it changed, in order of consequence:

- **Task 54 is re-scoped.** The `kind="monitor"` of the first draft — a
  training span, a sequential boundary, an open-ended alarm — was written
  as Wied–Krämer–Dehling 2012 and that paper has no such boundary: its test
  is a *closed-sample* fluctuation test on a span of known length `T`, and
  the sequential detector is Wied & Galeano 2013, which nobody has read.
  The monitor now runs the WKD test **span by span** (`horizon` required,
  `train` gone), with the paper's `D̂` (three-way delta method over the five
  raw moments, Bartlett long-run covariance, `γ_T = ⌊ln T⌋`), critical values
  from the Kolmogorov series (`1.358` at 5 %), Bonferroni over pairs, and
  its size and power *pinned to WKD's Table 1* — `.040/.035/.041` at `T =
  500`, `.587` power on a `0.5 → 0.7` step — instead of "within Monte-Carlo
  error" of a nominal figure. The `train` semantics and the ARL experiment
  move to the `window` kind, which is unchanged (ANSWERS confirms the
  sign-flip null has zero spread, so the row permutation stands). The
  Wied–Galeano sequential form is an ENHANCEMENTS §10 follow-up, not a
  blocker.

- **`epps_invert` gets `L` and the triangular weights.** Tóth–Kertész
  eq. 12 weights the numerator *and both denominators* by `(L − |x|)`; the
  flat sum the draft had is its `L → ∞` limit. The signature is now
  `epps_invert(gram_or_row, *, L)`, and the input must carry lags `1 …
  L−1`.

- **`rcov`**: `kₙ = ⌊θ√n⌋` (floor, as CKP write it), the finite-sample `ψ`
  and `Φ` sums as the paper has them, and the auto-bandwidth's two grids
  named as BNHLS use them (`iv_stride` for the sparse `IV̂`, `noise_stride`
  for `ω̂²`, with its deliberate upward bias stated). The one thing ANSWERS
  did not read — CKP §3's longer window for `psd=True` — is marked as the
  implementer's read; the draft's `δ = 0.1` is Hautsch–Podolskij's and
  stands only if §3 agrees (it does: task 114, 2026-09-28).

- **`po.corr`**: Higham's examples, distances, null vector, rank and
  iteration count are now the paper's, not memory's, and `nearest` is
  pinned to Algorithm 3.3 with his convergence test 4.1 and the three
  bound tests his theorems give; `shrink` carries the Ledoit–Wolf `π̂`, `ρ̂`,
  `γ̂` formulae as its oracle; `mp_edge` is verified and gains
  `mp_density`; `fisher_se`'s AR(1) factor is derived from Bartlett's
  formula and its two assumptions are named.

- **`deco`**: the correlation-targeted intercept and `α + β < 1` are
  recorded as *our* departures from Engle–Kelly, with the paper's
  downward-bias remark and its unoffered `u^var` alternative noted.

- **`bocpd`**: the variance-step test uses Adams–MacKay's own finance
  fixture shape (`"diag"`, gamma prior `a = 1`, `b = 1e-4`, `hazard = 250`);
  the ABK 2023 robust posterior stays an implementer's read.

- **`refresh_time`**: the `N = 7`, `21/27` example is verified but its tick
  times are only a figure, so the test constructs a stream with those
  counts; the staleness caveat of BNHLS §2.1 goes in the docstring.

- **`hmm`** (task 53) and the batch-wide decisions: confirmed as written,
  nothing changed.

Two reads remain with the implementer — ABK 2023 (open access) for
`bocpd`'s robust emission and CKP §3 for `rcov`'s PSD window — and one
paper is deferred, Wied & Galeano 2013. Neither read blocks a task from
starting; each is a named paragraph in its block with the acceptance test
that pins it. The batch is ready to implement.

**The chunk plan, revisited: P9–P11 and a fan-out floor, 2026-09-04.**
Asked whether the per-chunk parallel plan could be faster without
compromise. Sectioned first: at 14 threads on a 64-group chunk the
per-group tasks were 15 ms and the phases around them 21, each with a
single-threaded stretch. Three changes, every golden number unchanged:
columns gathered once, group after group, so a stream reads a contiguous
run (P9); one job per output field, `Vec<f64>` + bitmap into
`from_vec_validity` (P10, which reverses the 2026-09-02 "typed builders
not worth it" — right for the chunk it measured, wrong at 14 threads and
for a grid); columns read in parallel, multi-chunk columns copied per arrow
chunk, integer keys bucketed by value with `integer_groups` pinned to the
`String` path by a test (P11). Wall on the 64-group chunk at 14 threads
37 → 17 ms; the 12M-row README workload 3.25 → 2.48 s. Two findings worth
more than the speed: the matrix's 4× interleaved-vs-blocked gap was
`solve_every` cadence — an index clock over interleaved groups hits
`max_dclock` every row and re-solves ten times as often — so the layout's
true cost was 14%/38%, and the "artifact" is the documented semantics of a
clock-unit solve schedule, which any such benchmark pays; and the gate's
memory test caught the fan-out of 30-row groups (the plugin under
`.over()`) doubling RSS *wobble*, not growth, for no speed, hence
`PAR_MIN_ROWS` = 4096 below which a chunk's columns and fields are done on
the calling thread. `docs/PERFORMANCE.md` §12 has the tables. Not merged
before `v0.1.0` unless the user says so: it moves the tag target and needs
another rehearsal. **Decided 2026-09-04: `v0.1.0` is tagged on `d6370d9`
(main, rehearsal and CI green there) and this branch is 0.1.1 material.**
The same day the README's Parallelism section gained a *Chunk size*
subsection — the knob is `chunk_rows` on every streaming surface, the
numbers never depend on it (only where `coef` lands), the default 100k is
right for interleaved groups (50k–500k within 0.4 s on 12M rows), a
group-sorted file wants a few × rows-per-group (8.1 → 4.2 s at 1M here),
and very large chunks cost memory and the read/fit/write overlap (2M: 2.4 GB
and nearly twice the default's time). Its memory column, and the two-knobs
paragraph's memory numbers, are **peak footprint** (`/usr/bin/time -l`),
one run per process; the two-knobs numbers had been peak RSS, which counts
the memory-mapped input (~0.7 GB on a 712 MB file) and read 1.8 GB where
the footprint is 1.1. The other numbers in that section are still main's
and are regenerated when the branch lands (`scripts/benchmark.py`,
`scripts/scaling_bench.py`, the grid timings). PERFORMANCE §12 has the
sweep on both builds.

**The x86 control, the merge, and 0.1.1 prepared, 2026-09-04.** Asked
whether the layout's gain was this machine's, the answer came from
GitHub's `ubuntu-latest` (4 vCPU, x86) with the same workflow file on both
commits — `benchmark.yml` gained a `ref` input for exactly that (a
dispatch measures any tag, branch or full SHA as a control) and its
artifact now carries both tables. k=20 over 64 groups: 389k / 704k / 788k
rows/s on `v0.1.0` against 421k / 807k / 890k on the branch at 1 / 2 / 4
threads (+8 / +15 / +13%), so the stride was not an Apple artefact; the
single-group table is at or above `v0.1.0` on every row within the
runner's own ±10% (two runs of `v0.1.0` code differed by that much). The
branch was merged fast-forward and prepared as **0.1.1**: version bumped,
CHANGELOG cut, `docs/VALIDATION.md` regenerated (the version line and
nothing else), the README's measured numbers regenerated on this build.
What remains is CI and a Release rehearsal on the merged head, then
`v0.1.1` on the user's word.

**The chunk plan's edge cases, 2026-09-04.** Asked to make sure every edge
case of the parallel implementation is tested. The plan's three pieces
had been proved by golden numbers on the benchmark frames and by the
existing 400-row bank tests, which never reach `PAR_MIN_ROWS`; what was
missing was the same claims *above* the floor and on the awkward inputs
(null keys, one-row groups, nulls in every column, sessions, a column in
many arrow chunks, non-Float64 dtypes, chunk sizes of 4095/4096/4097,
errors under a permuted layout). `crates/online-polars/tests/chunk_plan.rs`
now holds them, `PAR_MIN_ROWS` is public so the test can straddle it, and
`tests/test_portability.py`'s thread-determinism case runs above the floor
too, with a halflife grid and every output on. Nothing was found; the
tests were mutated by hand to check they could find something, and did.
`docs/TESTING.md` T-E13 has the list.

**`v0.1.1` released, 2026-09-04.** The user fixed the PyPI trusted
publisher (the `v0.1.0` publish job, re-run, put 0.1.0 on PyPI) and said
"tag for 0.1.1 when ready". Ready meant CI 33871208866 and the
user-dispatched rehearsal 33872948555 green on `e552176`, so the annotated
tag `v0.1.1` went on that commit — not on the edge-case tests that landed
on `main` the same hour, which are test-only and would have meant another
rehearsal. Release run 33879445306: every job green; the GitHub release
carries the six wheels, the sdist and five CLI binaries; PyPI has all
seven files, and a clean venv's `pip install polars-online==0.1.1` (polars
1.44.1) ran a 5000-row grouped fit. The next release is whatever
`[Unreleased]` gathers.

**The bank's pool is its own, named for what it is, 2026-09-04.** The
bank fanned out on rayon's global pool, so its one knob was
`RAYON_NUM_THREADS` — a name that, next to `POLARS_MAX_THREADS`, said
nothing about which pool it was. Asked whether two pools could interact
badly, measured first: no correctness hazard (the wait graph is one-way —
py-polars' pool → our pool → our polars copy's pool — and a bank task never
takes the GIL or calls back into polars' pool), and no speed hazard either
(28 + 28 threads on 14 cores ran the grid in the same time as 14 + 14;
7 + 7 was slower). The two counts do different things: polars' also sizes
its reader's prefetch, so on 12M rows over 64 groups `POLARS_MAX_THREADS=4`
with the bank on 14 was 3.0 s at 1.4 GB against 3.2 s at 1.8 GB, while one
shared count of 4 would have been 4.5 s. Decisions: keep two pools and
two knobs; rename ours **`POLARS_ONLINE_MAX_THREADS`** and build it
ourselves (`crates/online-polars/src/pool.rs`: `OnceLock<ThreadPool>`,
built at the first bank call, per-core default spelled out so
`RAYON_NUM_THREADS` reaches nothing, a non-count refused by name);
`Bank::fit_predict`/`predict` run under `pool().install`, which also
carries the per-instance `par_iter`s in `Stream`; the runner's parquet page
encoding and NDJSON slices move onto polars' pool
(`polars_core::runtime::THREAD_POOL`, a direct `polars-core` dependency
already in the tree, nothing new linked) so `POLARS_MAX_THREADS` is
polars' readers *and* writers in every form; `po.thread_pool_size()`
mirrors `pl.thread_pool_size()`. The README's parallelism example is a
grid over factor sets (one spec per set — its own accumulator,
standardization and null handling — with the halflife/ridge grid inside),
and a second example shows the two knobs set apart, with the numbers.
Not done before `v0.1.0` would have been a released knob to rename later.

**Boosted trees: investigated, prototyped, not built, 2026-09-03 (task 21).**
Asked to dig into XGBoost — the papers and the code — for how gradient-boosted
trees could be made as online as possible, fit in parallel, and use less
memory. The finding, in `docs/BOOSTED-TREES.md`: the boosting math is
already online-shaped (leaf values and split gains are functions of per-node
gradient sums, which are additive, mergeable and decayable), and what is
*not* online in XGBoost is everything around the sums — a cut pre-pass over
all the data, O(n) gradient/position arrays, histograms that exist only
while a tree is built. A design that keeps every rule of the contract was
prototyped in numpy and measured: it matches XGBoost's batch fit on the
warm-up buffer to 0.02, then ties an 8 000-row refit window on stationary
data and beats it by 2–12 MSE under drift, in 12–16 k doubles of state
against the window's 80 000 rows. Chunk invariance is exact and per-tree
sums are additive over threads; both are measured, not argued. The ideas
that did not survive measurement are recorded with their numbers (§7.3) so
they are not re-tried. Decisions: the prototype and its experiment script
are committed under `scripts/` as source, not wired into the gate or CI,
with nothing added to `pyproject.toml` (XGBoost is an optional `uv run
--with` overlay for the baseline rows only, so rule 12 is untouched);
downloaded sources, clones and notes stay under the gitignored
`.cache/research/`; the exclusion of trees in `ENHANCEMENTS` §4 and
`BEYOND-O-STATE` is reassessed — its three technical grounds (unbounded
state, nondeterminism under resampling, no clock-decay semantics) are
answered by the design; the cost of a second model family is not, and is
the decision — and both documents point here without rewriting their
history; nothing goes
into the Rust crates until the user decides, and the doc's §9 says what
that would take (a `gbt` module in `online-core`, `ModelState::Gbt`, a
`SCHEMA_VERSION` bump, the `EXTENDING.md` list) and what should come first
(real data, §8 idea 10).

**One error contract, stated once and documented at every entry point,
2026-09-03.** Audit of every public docstring and Rust doc comment for its
failure modes found the contract mostly there and the gaps all of one kind:
a failure that was typed wrong (`OSError` for a spec mismatch), silent (an
unknown key in a spec or config took the default), or late (`save_state`'s
directory checked after the run it would lose). Each was fixed rather than
written up: `deny_unknown_fields` on `Spec`, `ModelKind` and `RunConfig`
(the state-file loader reads the envelope first, so a newer build's file
with keys this build lacks says "newer", not "not a bank file");
`ModelBank.load` splits file errors (`OSError` subclass) from content errors
(`ValueError`); `po.run` checks `save_state`'s directory before the run.
The contract is one paragraph in `polars_online.__doc__`; each docstring
says only what *it* raises and when, and `tests/test_runner.py`,
`test_bank.py`, `test_eval.py` pin the types and messages. `cargo doc
--workspace --no-deps` builds clean and is the Rust reference.

**The API reference is Sphinx, and a bad docstring fails the build,
2026-09-03.** Asked for a doc builder with no preference between them:
Sphinx, because the docstrings were already written in its dialect
(`:class:`/`:func:` roles, `::` literal blocks) and pdoc would print those
roles as text. `docs/reference/` holds four pages — the package, `spec`
with the typed keyword sets, the three `online` namespaces, `eval` — all
autodoc; nothing is written twice. It builds with `-W` in the gate and in
CI's ubuntu test job (it imports the package, so it needs the built
extension that job already has; no second Rust build, no new job on the
other runners), and a `docs` job publishes the HTML to GitHub Pages from
`main`, skipping with a notice if Pages is ever switched off. Pages was
enabled the same day, Source "GitHub Actions" and nothing else — the
Jekyll / static-HTML starter workflows GitHub offers there would each add
a second deploy of the repository tree to the same site — and the run in
flight deployed <https://hgilde.github.io/polars-online/> at once; the URL
is `Documentation` in `[project.urls]`. The first `-W` build found four
docstrings that were not valid RST
(a `*by` read as emphasis, `|r|` as a substitution, a table column one
character narrow, ``` ``TypedDict``s ``` with the plural glued to the
literal) — the class of defect that had no test before, which is the
argument for `-W`. The `docs` dependency group (`sphinx`, `furo`) is
installed by `uv run --group docs`, so a plain `uv sync` stays as it was.

**The betas are a frame, not a detour, 2026-09-03.** Asked whether a saved
state can be introspected -- the coefficients of a linear model read back
from a file -- the answer was "in three roundabout ways": `predict` one row
and read the output's `coef` field, hand-solve `gram()`'s moments, or take
the last `coef` from the run that made the file. None names the terms.
`ModelBank.coef(spec, group=None)` now returns one frame -- `group`,
`instance`, `n_eff`, then `spec.coef_index`'s columns, then `coef` -- so
`bank.coef("ols").pivot("term", index=["group", "instance"], values="coef")`
is the wide table people expect, from a live bank or `ModelBank.load(path)`.
It is the same `AnyModel::coefficients()` call the output's `coef` field
makes, so the two agree by construction (`tests/test_coef.py`). Two facts
the docstring states because they surprised: `coef` is the last *solve*, not
gated by `min_periods` as `pred` is (so it can exist, jittered, over fewer
rows than terms -- `n_eff` is there to say how much is behind it), and under
the default `solve_every = halflife/50` it is that stale. Asked separately
whether an EW-OLS is a local regression: measured, yes -- with `ridge=0.0`
and `solve_every=0`, `pred[t]` is the weighted least-squares fit of rows
`< t` with weights `0.5**((t-1-i)/halflife)` to 1e-14, `coef[t]` the same
over rows `<= t`; the kernel is one-sided (causal), `n_eff` saturates at
`1/(1-0.5**(1/halflife))` ~ `1.44*halflife`, so `min_periods` must sit below
that, and the plan / `po.run` stream it in O(chunk) memory. The test pins
the statement against `numpy.linalg.lstsq`.

**The output comes apart with the coefficients named, 2026-09-03.** The
polars-native way to the betas was `coef_index(spec)["term"]` fed to
`list.to_struct(fields=...)` and an `unnest` -- correct for one instance
and one combo, wrong for a grid (one term list, several blocks), and with
bare names (`x0`) that collide with the feature columns. Asked for a helper
that covers whatever else the output carries, not just the coefficients:
`lf.online.unnest(specs)` (and `df.online.unnest`, `po.unnest(frame, ..)`)
is polars' `unnest` for a bank's output -- every scalar field becomes a
column of its own name, and each `coef` list becomes one column per
coefficient, named on the field grammar with the term after the target
(`coef_y_x1__r0.5@h500` beside `pred_y__r0.5@h500`). The names come from
`spec.coef_fields(spec)`, a Rust-rendered table (`online_polars::coef_fields`,
next to `output_index`) of every coefficient with its `field`, `position`,
`name`, target, halflife/lam, ridge, feature set, lambda and term;
`coef_index` is now that table's first instance. `unnest` takes the specs,
a bank, or a state path, replaces each struct in place, leaves unnamed
columns alone, and reports a spec that does not match the frame while the
plan is built; two specs with the same field names are polars'
`DuplicateError`, as with polars' own `unnest`. The test that matters
(`tests/test_unnest.py::test_named_coefficients_predict_the_next_row`) pins
the names against the models rather than the strings: with a solve on
every row, `pred[t+1] == coef_intercept[t] + sum_j coef_xj[t] * xj[t+1]`
for all 16 (target, feature set, ridge, halflife) columns of a grid, which
a wrong slot order, a swapped feature, a wrong ridge or a wrong instance
each break by 0.03 to 7 (checked by breaking them). Not added: a SQL
`where=` filter for the CLI (the user declined it: a `polars-sql`
dependency for a filter polars' own plan already has).

**Any row order, 2026-09-03.** The README said the library was "built for
ordered event data", which undersold it: a bank is sufficient statistics,
so row order reaches the fit only through decay, and with decay off it is
a regression library that never holds the data. Measured rather than
argued, 6M iid rows x 20 features in 12 parquet parts: `ewridge` with
`ridge=0` and `halflife=inf` (or `lam=1.0`) matches `numpy.linalg.lstsq`
to 2e-13 fed forwards or backwards, at 1.4 GB peak RSS against 3.97 GB
for `lstsq` on the same rows (10.7 s solving every row, the default for
`inf`; 1.4 s with `solve_every=1000`, the coefficients 2e-6 off). A finite
row halflife is the weighted least squares of the order given (halflife
1e6 rows: 4e-4 from OLS, which is itself 5e-4 from the truth; reversed,
the same distance). Two things came out of it. (1) The docs now lead with
both shapes: the README intro, a new "Any row order" section, "What this
is not", the `halflife` row and the `ewridge` solve sentence; and
`tests/test_row_order.py` pins no-decay == lstsq in four row orders with
two interleaved groups, through the bank and the chunked plan, and the
finite-halflife-reverses-the-weights statement. (2) A trap, known here as
the `solve_every` gotcha further down in this section and now documented
in the README and pinned, but not fixed: the solve cadence defaults to `halflife/50` for any
*finite* halflife and to every row only for `inf`/`lam`
(`Spec::solve_every_default`), and `max_rows_between_solves` defaults to
unlimited, so `halflife=1e12` solves once at `min_periods` and never again
-- 0.3 to 0.4 off OLS on the same 6M rows, where `inf` or
`solve_every=1000` is 2e-6 off. Not changed now because the cadence is
part of every golden and any rule that scales with the accumulated weight
needs the weight at the last solve in the state (a layout change:
`SCHEMA_VERSION` bump plus a loader for the old one). The candidate rule,
for the user's decision: solve when the weight added since the last solve
exceeds ~2% of the total, which reproduces `halflife/50` at steady state
(2% of a saturated `1.44*h` is `h/35`), solves densely while the weight
is still growing (early rows, and a warm-up after `reset_state`), and
makes `inf` cost O(log n) solves instead of one per row while bounding
the staleness to 2% of the weight. Until then the README says: `inf` is
the no-decay setting, `solve_every` is the throttle, and a huge finite
halflife is neither.

**The Polars concern stated up front, and the canary made honest,
2026-09-03.** Asked for a note at the top of the README on the moving
polars API and how the canary handles it. Writing it meant checking what
the canary does, and it was not what the README said: its Python unpin
regex still matched `"polars==..."`, which the range `polars>=1.34.0,<2`
no longer is (so it tested the newest 1.x only by the range's grace and
would have excluded a 2.0 silently), and its Rust unpin -- `version = "*"`
plus `cargo update -p polars` -- would have put two polars in the tree
the day polars 0.56 ships, because pyo3-polars 0.28 requires `^0.55.1`
(its Cargo.toml) and `crates/online-polars` pins polars-arrow,
polars-parquet and polars-utils to `=0.55.2`: a red canary for a reason
that is not "Polars broke us". Today both are harmless (0.55.2 is the
newest crate, `cargo search`; 1.44.1 the newest wheel), so the job has
never misfired -- it has also never run, the schedule being Mondays on a
repo public since 09-02. Now: the canary moves py-polars only, the copy
that can break a user (the Rust copy is the wheel's own and never meets
it), drops the range rather than widening it so a 2.0 is tested the week
it appears, asserts the dependency line was found, and upgrades polars
alone (`uv sync --upgrade-package polars`, checked locally to move
nothing else) so one thing varies per run. The README's top note and
"How the pin will move" state the policy for a red canary: cap the range
at the last release that passed, in a patch release, then fix and widen;
look at `ModelBank` first, the IO-plugin tests second, the plugin last.
"What the pin costs you" was rewritten too -- it still said a different
polars could not be installed at all, which the range made false.
Dispatched by hand after the push, the job's first run ever was red on
exactly one of 1162 tests, with the same 1.44.1 CI uses:
`test_scaffold.py::test_the_declared_range_brackets_what_we_build_against`
reads pyproject and asserts the range the canary had just removed. That
test and its neighbour (`pl.__version__ == BUILT_AGAINST`, which would
have gone red on the first newer wheel) are about our pins, not polars;
they now carry a `pins` marker and the canary runs
`-m "not soak and not pins"`. Note `-m` replaces pyproject's addopts
`-m 'not soak'` rather than adding to it.

**State leaves a streamed plan through a file, and only a file, 2026-09-03
(task 20).** `lf.online.fit_predict(specs, load_state=, save_state=)` — the
runner's two keywords on the plan, so the fourth step of the state workflow
(load, learn on, save) is written the same way on every surface. The plan
stays pure: `load_state` is read when the plan is built and the plan carries
the bytes, as `df.lazy()` carries a frame (the same for `predict(path)`), so
collecting twice gives the same frame and `load_state=p, save_state=p` used
twice in one query cannot race the second run's load against the first
run's write. `save_state` is written when the source has fed the bank its
last row — the stream's end, or the `n` rows of a pushed `head(n)`, which
the source now applies to the *input* chunk so the bank learns exactly those
rows — never in `finally`, never after a bank error; a plan used twice in
one query runs twice on two threads (measured: no common-subplan
elimination reaches a Python source) and writes the same bytes twice. That
second point turned up a hole older than the feature: `atomic.rs` named its
temporary by pid alone, so two threads saving one path in one process wrote
*one* temporary and published a mixture — `ModelBank.save` from two threads
had the same hole. Fixed there, with a process-wide counter in the name,
rather than with a Python-side lock: the root, and it covers `po.run`'s
output file too. The memory side — a plan that updates a `ModelBank`
object, or `load_state=bank` — is declined: the user's call, and the right
one, because a plan is re-executed (twice, concurrently, when a query uses
it twice) and an object it mutates has no single "after"; the file has, and
a user reads it without knowing any of this. One documented gap (R6 in the
research): a node *after* the bank failing does not stop the bank, so the
state is written although the query failed — `po.run` saves only after its
output is committed, and a dated `save_state` per batch keeps a rerun from
learning it twice. `coef` moved one row in `head(n)` results: it is reported
on each chunk's last row, and the `n`th row is now that row.

**The expression form stays and warns, 2026-09-03 (task 19).** Two forms
carried one set of numbers and two memory profiles — `df.with_columns(pl.col
("y").online.ewridge(..))` at 7.3 GB against `lf.online.fit_predict([spec])`
at 1.35 GB on 12M rows — and a user who wrote the natural expression inside a
lazy query got the O(data) one. The cause is polars' contract for a stateful
user expression (the entry below), which we cannot change from inside a
plugin. The first cut of this task removed the expression form: the wheel was built
without an `expr-plugin` cargo feature, `pl.Expr.online` went unregistered and
the README showed only surfaces that stream. Reconsidered the same day, before
the next commit: a user who writes the expression then gets polars' bare
`AttributeError` with no rationale and no pointer, the in-memory use (features
as expressions, `.over`) is lost for nothing, the one interface with a polars
stability guarantee leaves the wheel, and the plugin's runtime tests stop
running in CI. So the expression stays and *teaches* instead: every call issues
`InMemoryExpressionWarning` — a `UserWarning`, because a `DeprecationWarning`
is hidden outside `__main__`, i.e. in the pipeline module where it matters —
with the reason, the plan to write instead, and the filter for someone who
means it; the README's closing note shows the two forms side by side with
the numbers. Not "deprecated": it would become a streaming form too if
polars ever ran a user expression per morsel with state (§6). The feature
gate, `has_expr_plugin()` and the `requires_expr_plugin` marker are gone.

**The expression form is in-memory; the streaming query form is the bank as a
source, 2026-09-02 (ENHANCEMENTS E33).** `lf.with_columns(pl.col("y").online
.ewridge(..)).sink_parquet(..)` collects the whole input in either engine,
because polars' streaming engine has no ordered, stateful node for a user
expression — a plugin is a `columnar-function` node (collect, call once,
re-emit), and the only per-morsel path for user code is elementwise, which
is unordered. Measured 7.3 GB at 12M rows (`docs/PERFORMANCE.md` §11). That
cannot be fixed inside the plugin, so it is fixed at the plan level:
`lf.online.fit_predict(specs)` registers the bank as a polars IO-plugin
source (`register_io_source`) and returns a `LazyFrame` that streams the
input through a fresh bank when it runs — O(chunk), bit-identical to
`po.run`, composing with polars' filters, selections, joins and sinks. Rules
adopted: the plan is *pure* (a fresh bank per execution, or `load_state`;
no `bank=` the plan would mutate — `save_state=` came with task 20, and
purity is what makes it safe); a filter after the bank
never changes what it learns from; polars does not re-apply the pushdowns
it hands a Python source, so the source honours projection, predicate and
slice itself, slice counted before predicate (polars' optimizer order). The
interface is polars' documented-but-`@unstable` IO plugin (CLAUDE.md rule
13), and it reads with `LazyFrame.collect_batches` (py-polars 1.34.0), which
`po.run` already did since E32 — the declared floor moved from 1.28.1 to
1.34.0 to say so. The Rust API has no twin: polars-stream 0.55.2 lowers
`AnonymousScan` to `todo!()`; Rust callers use `run(.., Output::Batches)`.

**Clock dtype decision, 2026-08-30.**

`clock` must be a **numeric** column; temporal dtypes (`Datetime`, `Date`,
`Duration`, `Time`) are rejected with an error naming the column, its dtype and
the fix. Casting a temporal column to f64 exposes its internal representation,
so identical wall-clock data yields deltas differing by 10^3-10^6 depending only
on the column's time unit, and `halflife` / `max_dclock` / `session_gap` inherit
those units. `halflife = 600` against a microsecond-backed `Datetime` therefore
meant 600 microseconds: every row decays to nothing and the output is finite,
non-null, plausible-looking garbage — the worst failure shape available, since
none of the existing guards catch it. Rejecting costs one expression at the call
site (`pl.col("ts").dt.epoch("s")`) and makes the intended scale explicit,
consistent with the null-clock error and hard-rule bias toward loud failure.
Auto-converting to seconds was considered and declined: it would make the
meaning of `halflife` depend on the input dtype, which is the same class of
implicitness that caused the problem.

**Superseded in intent by task 88 (2026-09-23), not yet built.** A third option
was not weighed here: give the time parameters units too, as durations. That
removes the objection above, because a `timedelta` halflife states its own
unit and so no longer depends on the column's. Until task 88 lands, this
refusal stands exactly as written.

**Tasks 15-17 (CLI, release CI, README), 2026-08-30.**

- The streaming runner lives in `online-polars` (`RunConfig` / `run_config`), not
  in the CLI crate, so the same code path is testable without spawning a process
  and could back a Python streaming API later. (It did: E8, and E32 on
  2026-09-02 made the reader pluggable — `run(bank, Input::Lazy(plan) |
  Input::Batches(frames), Output::File | Output::Batches, ..)` — so `po.run`
  reads with py-polars and hands frames in, and any of parquet / ipc / csv /
  ndjson goes in or out. The Python API is not bound to parquet, and neither
  is the Rust one; only the CLI's inputs are files, because it is a binary.)

- Output is written with polars' batched writers: one row group (or record
  batch, or slice of text) per chunk, so memory stays O(state + chunk) end to
  end.

- The cross-OS state test (§9 class 7) is one test file driven by two env vars:
  `ONLINE_WRITE_STATE` writes the hand-off artifact, `ONLINE_FOREIGN_STATE`
  loads one. CI writes on macOS and reads on Windows and Linux; without the env
  vars the test still checks the round trip locally, and `save_bytes` is
  asserted deterministic (which is what makes the hand-off meaningful).

- Benchmarks (Apple M-series, 200k rows, best of 3): ew_ridge 2.27M rows/s at
  k=5, 1.59M at k=20, 0.75M at k=50; 10 targets cost ~1.5x one target (shared
  S), while 5 halflives cost ~2.9x one (separate accumulators, as documented).
  ftrl is the fastest model, kalman the slowest.

- Fixed a real bug the README examples caught: `eval.unpack` collided when a
  target column was literally named `y`; reserved output names are now dropped
  from the passthrough columns.

**Tasks 13-14 (robust + logistic), 2026-08-30.**

- The robust models expose two spec types, `huber` and `quantile`, over one core
  `Robust` model (they differ only in the IRLS weight function).

- `huber_delta` default 1.5 kept: a sweep is only meaningful against a
  contamination model, and the unit test confirms 1.5 recovers the clean slope
  under 3% gross outliers where least squares does not.

- Quantile weights are scaled by the residual std so they are O(1) rather than
  O(1/sigma); `quantile_eps` (default 1e-3, in units of that std) floors |r| so a
  near-zero residual cannot produce an unbounded weight.

- FTRL defaults `alpha=0.1, beta=1.0, l1=0.0, l2=1.0` kept (McMahan et al.).
  Note FTRL's L1 zeroes a coordinate only while `|z_i| <= l1`, and `z_i` grows
  with accumulated gradient, so a moderate `l1` shrinks rather than permanently
  pins - the unit test asserts that behaviour, not exact sparsity.

- **Gotcha worth knowing**: `solve_every` defaults to `halflife/50`, so a very
  large `halflife` (e.g. 1e9, meaning "never forget") means the model solves
  once and never again unless `max_rows_between_solves` is set. Left as-is
  (it is the documented default), but every long-halflife test sets an explicit
  cadence.

**Task 12 (evaluation + [validate] items), 2026-08-30.**

Full numbers in `docs/VALIDATION.md` (regenerate with
`uv run python scripts/validate.py > docs/VALIDATION.md`). Data: 10 days of
BTCUSDT 1-minute rows (14,336 rows) from Binance's public dump, features = past
returns / volume / trade-count z-scores, targets = strictly future returns.

- **`solve_every` = halflife/50 is confirmed as the default.** Sweeping
  halflife/d for d in {1, 5, 10, 50, 200, 1000}: d = 50 gives the lowest MSE
  (1.16172e-06). Solving every row (d = 1) is *worse*, not better -- the extra
  responsiveness is noise. Kept as-is.

- **`standardize` default (false for ridge) confirmed**: on this data plain and
  standardized ridge are within 0.3% MSE of each other, so the default stays
  off for ridge (cheaper) and on for lasso (required by the algorithm).

- **`l1_ratio`: no evidence elastic net is needed.** At matched penalties,
  l1_ratio 1.0 / 0.5 / 0.1 are within ~1% MSE. The parameter is kept (it is
  ~free) but the default stays 1.0 = pure lasso.

- **Kalman `share_p`: not recommended as a default.** With two targets of very
  different noise levels, sharing P helped the short-horizon target slightly
  (-0.049 vs -0.060 R2) but hurt the long-horizon one (-0.085 vs -0.011 R2).
  Default stays `share_p = false`; it remains available.
  *Reads differently since task 212 (2026-10-08):* `validate.py` then
  learned each target at its own row, `y1` five rows before it was known,
  and ran `kalman` at a coefficient half-life of 200 beside the others' 500.
  Under each target's embargo (one spec holding both waits 5 rows) and
  matched memory, VALIDATION §4's shared `P` is the better of the two on
  both targets: `y0` −0.0102 against −0.0173, `y1` −0.0590 against −0.0608
  (after task 211: −0.0138 against −0.0157, −0.0725 against −0.0849). The
  default is not revisited there; the kalman review of 2026-10-08 (T8)
  measured `share_p`'s `se_coef` off by 0.019 and 1.93 against a calibrated
  1, where one `P` per target gives 0.98 and 0.98 -- fixed by task 211
  (0.983 and 0.975), so the default is a question for the user.

- Solve-schedule sweeps really are free: 6 schedules over 14k rows in 0.06s,
  because they share one accumulator.

- `tests/data.py` now downloads N days (`public_intraday(dates)`), cached per
  day, so the validation set is 10x bigger than one day.

- Note: negative out-of-sample R2 on this data is expected and not a bug -- a
  1-minute crypto return is close to unpredictable from these features.

**Tasks 4-5 (EW-ridge), 2026-08-30.**

- Grid combos are ordered target-major then (feature_set x ridge); output slots
  `n_targets * n_combos`. Coefficient vectors are always full `k_total` length with
  zeros outside a combo's feature set.

- The forced solve "after any capped/session gap" is implemented as: the accumulated
  clock-since-solve includes the gap, so any gap >= `solve_every` triggers a solve on
  the next row. There is no separate force flag.

- `sigma2_j` uses the first-combo (primary) prediction's residual.
- ridge_decay (the decaying-prior / RLS-equivalent mode) refuses grids and
  standardization at validation time.

- Standardized solves drop a feature when its centered variance is < 1e-10 x its raw
  second moment (cancellation noise scales with the raw moment).

- Solve failure even after jitter keeps the previous coefficients and increments
  `solve_failures` (never NaN silently).

**Task 1 (scaffold), 2026-08-30.**

- Version pins (`Cargo.toml` workspace deps ↔ `pyproject.toml`, kept in sync by hand and
  asserted in `tests/test_scaffold.py`): py-polars **1.44.1** ↔ rust polars **=0.55.2** ↔
  pyo3-polars **0.28** ↔ pyo3 **0.29**. py-polars 1.44.1 is built from rust polars 0.55.1;
  0.55.2 is the same minor and is what pyo3-polars 0.28 resolves its sub-crates to, so the
  facade is pinned there to keep one polars version in the graph.

- Rust edition 2024, `rust-version = 1.95`, `rust-toolchain.toml` pins the stable channel.
  It said 1.85 until task 160 (CI4), while the locked tree needed 1.95 (`sysinfo` 0.39.6);
  `tests/test_ci_cost_policy.py` now holds the declaration to the lock. Declaring 1.95
  raised clippy's MSRV, which turned on `manual_is_multiple_of`, `collapsible_if` over
  let-chains and `as_chunks`; the code takes them.

- Two non-obvious feature flags on the polars/pyo3-polars side, both needed to compile at all
  (upstream feature-unification gaps in 0.55.2), both commented at the call site:
  - `polars` needs `object`: `pyo3-polars/derive` turns on `polars-plan/python`, which turns on
    `polars-core/object`, and `polars-ops` then fails an exhaustive `DataType` match.
  - `pyo3-polars` needs `lazy` alongside `derive`: only `polars-lazy/python` propagates
    `python` to `polars-mem-engine`, which otherwise fails an exhaustive `DeletionFilesList`
    match.

- `pyo3` is declared **without** `extension-module`; maturin adds it via `features` in
  `pyproject.toml`. With it always on, plain `cargo build`/`cargo test` fails to link.

- `online-py` builds with `abi3-py312`, so the Rust build needs a >= 3.12 interpreter present.
  Everything (including `cargo`) therefore runs under `uv run`, which exports `VIRTUAL_ENV`;
  CI does the same. This is why CI runs `uv sync` before any cargo step.

- `online-core` sets `unsafe_code = "forbid"` at the crate level (hard rule 6).
- `doc/` was renamed to `docs/` to match `CLAUDE.md` and task 12's `docs/VALIDATION.md`.
- CI (`.github/workflows/ci.yml`): `lint` and `test` jobs, each a matrix over
  ubuntu/macos/windows. Lint = `cargo fmt --check`, `cargo clippy -D warnings`,
  `ruff format --check`, `ruff check`. Test = `cargo test --workspace`, `maturin develop`,
  `pytest`. Wheel/binary release jobs are task 16.

**Fixing the review of 45–56 (task 57), 2026-09-06.** Read
`docs/REVIEW-E54-E64.md` for the items themselves. Where the review asked for
a decision rather than a repair, this is what was decided.

- *A clock break inside an `rcov` block splits it into **stretches** (R2,
  R10).* The alternative was to refuse `clock`/`max_dclock`/`session` on an
  rcov spec, which would have made the bug unreachable and the model less
  useful -- tick data is exactly where a clock belongs. Each stretch is closed
  the way the group's last one is (leading jitter, interior, trailing
  jitter) and `Γ̂_h` is the sum over stretches, so no product pairs two
  returns across the break. A stretch too short to fill its tail contributes
  what it already emitted. The rewrite carries the phase in `head` rather
  than in a count against `n`, so **no state field was added** and an
  unbroken block is unchanged to the bit.

- *`predict` gets the targets slot, rather than the plumbing refusing it
  (C1).* `OnlineModel::predict_with(x, y, d_clock)` is a defaulted trait
  method that ignores `y`; `bocpd` and `hmm` override it. That keeps
  `predict` out of sample for every model that regresses its targets -- the
  default *is* the old `predict` -- while the two that read a parameter out
  of that slot answer for the row's value. `AnyModel::predict` takes `y` and
  the contract's parity probe calls `predict_with`.

- *A hazard column value `<= 1` is an error, not a null (B1).* It is a
  parameter, not a target: null and non-finite still mean "no value here"
  and fall back to `hazard`, and a finite value that is not a hazard is
  refused naming the row, the way a negative weight is. A row whose
  predictive cannot be evaluated is counted in `solve_failures` instead of
  vanishing.

- *`corrchange` reports a zero-weight row as if it would be learned (CC2).*
  A row is always part of the span reported *on* it -- that is what makes
  the flag out of sample -- and `predict` cannot know the weight, so the
  parity contract fixes the answer. The dead `weight <= 0` branch in `read`
  went; the behaviour is in the docstring.

- *Ties in `refresh_time` are broken by row order (RT4).* Buffering a
  completed grid point until a strictly greater timestamp arrived would cost
  the chunk-invariance the sampler has now, and tick data carries its own
  sequence. Documented, with a test that pins it.

- *No schema bump.* `bocpd.solve_failures` and `BankFile.key_integer` are
  both skipped when they carry nothing, so a state that never hit either
  writes the bytes it always did -- the rule `SCHEMA_VERSION`'s own doc
  records for task 38's sums. The schema-5 fixture was re-frozen anyway,
  because the `hmm` state in it was written by the model H1 corrected; the
  layout did not move and the file exercises the same loader.

- *B4 was reverted, and the revert is the finding.* Renormalising `bocpd`'s
  `logjoint` per row is exact on paper and free -- every output is a
  difference against `z` -- but `z` comes out of `ln`, and libm's last bit
  is not the same on glibc and Apple's. Feeding it back into the **state**
  made a bank saved on macOS continue differently on Linux:
  `state_schema5.rs`'s frozen file stopped reproducing and its three tests
  failed on `ubuntu-latest` alone (CI 34037072202), where the same fixture
  had been green on all three OSes before. That is the general rule worth
  keeping: **a libm result may be compared or reported, but putting one into
  the state costs cross-platform reproducibility.** The joint stays
  unnormalised; `prune`'s docstring and the review item both say why, and
  the drift it would have fixed is ~1.4 nats a row, so it costs nothing
  before about 1e9 rows.

- *Three goldens moved, all deliberately*: `GOLDEN_HMM` (H1),
  `state_schema5.rs` (H1), and the `rcov` rows of the pipeline golden (R2,
  27 → 21 effective returns over three capped gaps). Each carries a comment
  saying which fix moved it.

**State compatibility was dropped once, deliberately, 2026-09-07.**
Renaming spec keys with no aliases ends backward compatibility outright, not
partially: a spec denies unknown fields, so a file naming `coef0` is refused
rather than losing that one value. Rather than hide that behind six
deserializer aliases, `MIN_SCHEMA_VERSION` moved 1 → 6, the four fixtures
that proved older files load were deleted, and with them the schema-1 and
schema-2 conversion paths in `rls`, `sgd` and `kalman` — three untagged wire
enums, their old layouts, `factor_of_inverse`, and the three tests whose
`const assert!(MIN_SCHEMA_VERSION <= 2)` was written to fail exactly when
this happened. A state saved by 0.2.0 must be refit.

This is an exception to hard rule 5, not a repeal of it: the library is days
old, pre-1.0, and getting the names right was judged worth more than the
compatibility. `state_schema6.rs` is the frozen file the *next* layout change
will be held to, and the rule applies from here.

**The documentation pass (task 58), 2026-09-06.** Three rules, so the next
pass does not have to rediscover them:

- *Files and section numbers stay where they are.* The code and the README
  cite `docs/PERFORMANCE.md §11`, `docs/PLAN.md §4.4`, `E56`, `C5`, `T-D4`
  and the like, and a renumbering would silently break every one of those
  pointers. Reorganisation happens *inside* a file — a guide-first section
  in front of a research record (`STATE-WORKFLOW.md`), a reader's map under
  a status line (`PERFORMANCE.md`, `TESTING.md`), two sections swapped into
  numeric order (`PERFORMANCE.md` §5/§6) — and `docs/README.md` is the map
  across files.

- *A record is not rewritten when the code moves on.* Every document under
  `docs/` opens with a dated status line that says what became of it; the
  body below keeps its date. So `CLUSTERING.md` still says "nothing in the
  crates" in a 2026-09-04 sentence, and its status line says `kmeans` and
  `micro` shipped.

- *The README is the guide and the docstrings are the reference; neither
  repeats the other.* Each model's README section states the recursion and
  links to its builder (every keyword with its default) and to its Rust
  module (the recursion as code, with the module comment stating it). A
  default lives in the docstring and in `stream.rs`, and the docstring was
  checked against `stream.rs` for every builder.

**`v0.6.0` tagged, 2026-09-16.** The user asked to push and tag for the next
minor bump. `CHANGELOG.md`'s `[Unreleased]` was cut as `0.6.0`, the version
moved in `pyproject.toml`, `Cargo.toml`, `polars_online.__version__` and both
lockfiles, and `docs/VALIDATION.md` was regenerated on it — it differs from
the committed document in its version line alone, the one other changed line
being a timing its test normalises away. A minor twice over by this package's
own pre-1.0 rule: numbers move in many models, and the state schema is 10.

The tag went on `9f6a5f9`, not on the release-preparation commit `bba23ed`.
Between the two the user read the rewritten README and objected to its first
example: it fitted a *decayed* regression over a folder of files and then
served from the final state, when a decayed fit is local and that state holds
only the last few hundred seconds, so the example taught the wrong habit. The
introduction now leads with a no-decay fit — ridge regression over every row
in bounded memory, where the saved state really is a summary of those rows and
saving and serving is the point — and follows it with a local fit whose
`coef_every=1` coefficient path is read as a time series, which is what a
local fit is for. The paragraph after them lost the sentence restating
no-decay convergence, and the README's running python-block count moved to 59.

That move cost the first rehearsal: `release.yml` dispatched on `bba23ed`
(35103650703) was cancelled once `main` passed that commit, because a
rehearsal certifies only the commit it ran on. The precondition was met again
on the new head — CI 35111714290 and rehearsal 35111713356 both green on
`9f6a5f9` — and `v0.6.0` is annotated there. The tagged release run is
35119720969; its `publish to PyPI` job parks on the owner's approval, which is
the one step nothing here can take, and `v*` tags are immutable, so a tag that
turns out wrong is left unapproved rather than moved.

<a id="s13"></a>
## 13. `window`: an EW accumulator with a hard cutoff (2026-09-06)

An exponentially weighted mean never forgets. A halflife of `h` leaves
`2^-3 = 12.5%` of the weight on data older than `3h`, `1.6%` older than `6h`,
and nothing is ever exactly zero. Some questions need the other thing: *no
information from before this point*, as a guarantee rather than an
approximation — a compliance window, a regime you believe began at a known
time, a backtest that must not see beyond its own horizon.

### 13.1 The identity

For any accumulator that is a sum of per-row contributions decayed
multiplicatively — `A(t) = Σᵢ wᵢ λ^(t−tᵢ) aᵢ` — splitting the sum at an
earlier time `u` gives

```
A(t) = λ^(t−u) · A(u)  +  (everything after u)
```

The first term is *exactly* the part to discard, and it is the accumulator's
own past value decayed forward. So a truncated accumulator is a subtraction,
not a recomputation:

```
A_window(t) = A(t) − λ^(t−u) · A(u),   u = the boundary
```

Verified in the small before any of this was designed: against a row-by-row
weighted sum over 800 rows on an irregular clock, the identity agreed to
**1.4e-14**, and with every row outside the window set to `1e6` the output
moved by `1.2e-09` where the ordinary EW mean moved by `1.3e+05`.

### 13.2 Which models can honour it, and which cannot

| exact | `ew_cov`, `marginal`, `ew_class`, `ewridge`, `lasso` |
|---|---|
| exact in the accumulator, not in the fit | `huber`, `quantile` |
| impossible | `rls`, `kalman`, `holt`, `sgd`, `pa`, `ftrl`, `kmeans`, `micro`, `hmm`, `bocpd`, `seqtest`, `corrchange`, `rcov` |

The first row is every model whose state is a sum: moments, or a Gram that is
solved *after* the subtraction. The second row is the honest awkward case —
`huber` and `quantile` accumulate rows already multiplied by an IRLS weight
that was computed from a state including rows the window now drops, so the
subtraction gives the accumulator those weights imply, not the one refitting
the window from scratch would produce. The third row is recursive filters,
path-dependent updates, and assignment or test models, where no such
decomposition exists.

`window` is therefore **not** a common parameter. A common parameter in this
library means the same thing in every model (the principle behind hard rule
8), and this one is meaningless in sixteen of twenty-one. It is a per-model
key, refused elsewhere by name, as `seqtest` already refuses `weight`.

### 13.3 The design

- **A view, never a destructive edit.** The model keeps its ordinary EW state
  and a ring of past snapshots; the truncated accumulator is computed at
  report time. Subtracting from the live state would break the recursion and
  compound its own error.

- **The ring holds the window.** Each entry is `(clock, snapshot)`. Before
  reporting, entries older than `t − window` are dropped from the front; the
  boundary is then the front entry. Memory is `O(rows in window × state)`,
  which for `ew_cov` at `k` features is `k²+k+2` doubles per snapshot.

- **The guarantee is one-sided, and the rounding follows it.** Subtracting the
  state as of `u` removes every row at or before `u`, so honouring "nothing
  older than `window`" requires `u ≥ t − window`: the *oldest* snapshot still
  inside the window. Coarse snapshots therefore discard slightly more than
  asked, never less. `window_every = m` snapshots every `m`th learned row and
  divides the memory by `m`; the effective window is then in
  `[window − m·spacing, window]`.

- **Snapshots are keyed by clock and cadenced by the stream, not by the
  chunk.** A cadence counted per chunk would make the boundary depend on how
  the data arrived, and chunk invariance is not negotiable (hard rule 3).

- **The ring is state**, so a bank saved mid-stream resumes with its window
  intact, and `SCHEMA_VERSION` rises to 6. The model's own fields skip when
  absent, so an unwindowed accumulator writes the bytes it always did — but
  spec fields serialize their nulls like every other spec key, so every
  spec's bytes move, which is the same reason 4 and 5 bumped. (Written first
  as "no bump", on the model fields alone; the schema-5 fixture's
  byte-identity test is what caught the spec half.) The bump is worth having
  on its own terms: an older build loading a windowed file would ignore the
  ring and report untruncated statistics.

- **A zero-weight window is a null, not a zero.** When the window holds no
  rows — a clock gap longer than `window` — the denominator is `0/0`, which
  hard rule 9 says to guard rather than propagate.

### 13.4 The oracle

Every claim above is testable without reference to the implementation, and
each test is written from the definition rather than the code:

1. **Against brute force.** For a seeded irregular stream, compare each row's
   truncated moments to a direct weighted sum over the rows inside the
   window. Agreement to `1e-12`.

2. **The guarantee itself.** Replace every row older than the window with
   `1e6` and require the output to move by less than `1e-8`. This is the
   test that would fail if the boundary rounded the wrong way.

3. **Chunk invariance**, as for every model: one chunk against a thousand,
   bit for bit.

4. **Save and resume** mid-window: a bank saved and reloaded continues
   identically, which is what pins the ring into the state.

5. **`window = inf`** reproduces the untruncated model exactly, so the
   feature cannot change what existing specs do.

6. **Refusal**: every model in the third row of §13.2 rejects `window` with a
   message naming the model.

### 13.5 What this must not do to the reader

The risk here is not the arithmetic, it is a user who believes the wrong
thing about the number in front of them. Four ways that happens, and what
the documentation owes each:

- **"Windowed" reads as flat.** A reader who sees `window=3h` may assume a
  boxcar: every row inside counted equally. It is an *exponential* weight
  inside a hard cutoff — the newest row still dominates. Every place the
  keyword is documented must say so in the same breath, and the README's
  entry should show the weight function, not just name it.

- **The guarantee is one-sided and approximate in the other direction.** With
  `window_every > 1` the effective window is shorter than asked, by up to one
  cadence. Documenting `window` as "exactly 3h" would be false; the promise
  is *"nothing older than `window`"*, and the shortfall is stated with it.

- **The boundary is a discontinuity.** A row aging out of a `3h` window at
  `h` removes 12.5% of the weight in one step, so the series has small jumps
  that an EWMA does not. Anyone plotting the two together will see it and
  should have been told first. The continuous alternative — the kernel
  `λ^u − λ^W`, which tapers to zero at the edge — is worth naming in the
  docs even though it is not what ships.

- **Subtraction is not the same arithmetic as accumulation.** The result is a
  difference of two positives, so it loses precision in proportion to what is
  discarded: negligible at `window = 3h` (a 12.5% correction), and worse the
  shorter the window is relative to the halflife. Below `window = h` the
  documentation should say plainly that the mean form is being reconstructed
  from a cancellation, and `docs/PERFORMANCE.md` should carry the measured
  error against brute force at a few ratios rather than a rule of thumb.

Two more the reference has to carry because no test can: that `n_eff` under a
window is the windowed weight, so `min_periods` now gates on a quantity that
stops growing; and that a windowed `ew_cov` is *not* a rolling covariance in
the polars sense — polars' `rolling_*` recomputes each window and costs
`O(n·W)`, this is `O(n)` and matches it to `1e-14`, measured at 24× to 1100×
faster as the window grows from 74 to 4,680 rows. Users who only need moments
on data that fits in memory should be told polars already does this; the
reason to reach for the spec is a stream, a saved state, or a regression.

<a id="s14"></a>
## 14. Review round R1 (2026-10-03), after tasks 143, 144, 104 and 150

Five read-only reviewers, one per area, each finding re-derived here and
pinned by a test that failed before its fix (the review protocol of
2026-09-14). Committed in three batches; the finding IDs are in the commit
messages.

### Batch A+B: the names (144) and the formula layer (143)

| ID | Finding | Fix | Test |
|---|---|---|---|
| 144-F1 | `with_windows(like=dict)` read an old clock key as unset, and a keyword under an old name was refused as a stray expression | both refused naming the new name (`stream.py`) | `test_renames.py::test_with_windows_refuses_an_old_clock_name_in_like_and_in_a_keyword` |
| 144-F2 | the restart boundary stated as "at least that large" in five live places (code: a step back equal to it is a late row) | wording (README ×2, `stream.py` ×2, CHANGELOG, the plan's decision row) | the existing `test_restart_after_step_back_is_one_rule` |
| 144-F3/F4/F7/F8 | hmm `p_<j>`, PCA `pc<j>_<feature>`, `lam_selected`, `n_eff²` in docstrings and the README | the new names | `test_outputs_doc`, `test_api_surface` (unchanged: the fields were right) |
| 144-F5/F6 | the Unreleased changelog and two README passages in the old vocabulary | rewritten | -- |
| 144-F9 | `EwRidgeCfg.ridge_scale` is a bool under the enum's name | not changed: internal | -- |
| 143-B1 | the increments' state advanced over the whole chunk where a slice or a refusal stopped the core short, so a state saved under `head` depended on `chunk_rows` | snapshot and rewind to the rows consumed | `test_windows.py::test_a_state_saved_under_a_slice_resumes_alike_at_any_chunk_size` |
| 143-B2 | a formula's `cast` was rebuilt non-strict (Polars' default is strict), so an overflow became a silent null | the tree carries `"non_strict"`; `strict_cast` otherwise; a wrapping cast refused | `test_a_cast_is_strict_as_in_polars_unless_told_otherwise` |
| 143-B3 | a long multibyte name panicked the message's cut | cut on a character boundary | `formula.rs::a_long_name_is_cut_on_a_character_boundary` |
| 143-S4 | a step back on the stream's clock restarted every group's windows but not their increments | the runner keeps the stream's clock and restarts every group's increments | `test_a_restart_on_the_streams_clock_restarts_every_groups_increments` |
| 143-S5 | an increment skips a null input, undocumented | decided: kept, as the operators hold a value from the last valued row; documented | `test_an_increment_skips_a_null_input` |
| 143-S6 | std's `DefaultHasher` persisted in the windows state | the bank's `fnv1a` (windows state v2 is unreleased) | -- |
| 143-N7/N8/N9/N10 | one operator rule on both sides; the reserved prefixes refused on inputs; the "already a column" message names `.alias()`; `from_tree`'s log base | as listed | `windows_frame.rs::each_formula_holds_an_operator_and_reads_no_reserved_column`, `test_the_operators_prefix_is_reserved_on_inputs`, `test_a_positional_expression_named_after_a_column_is_told_about_alias`, `test_from_tree_takes_a_log_base_that_is_not_a_literal` |

### Batch C: the window core (143)

| ID | Finding | Fix | Test |
|---|---|---|---|
| C1 | without a clock every backward `"right"`/`"both"` operator was null on every row (the row never waited for a next stamp, so nothing read it) | the row's own windows are read at the end of its push | `test_a_row_count_clock_reads_every_backward_window` |
| C2 | the policy time was a sum of rounded steps, so a row exactly one window later fell on either side of the edge by the sum's noise (999 against Polars' 1000 at 1 ms steps; a row wrongly in at 60 000 ms) | measured from the stretch's first row, one subtraction exact in nanoseconds; the brute forces likewise | `test_a_row_exactly_one_window_later_lands_on_the_edge_at_any_age` |
| C3 | forward `"left"`/`"both"` gave rows at one stamp different windows (a sequence rule) | decided: a window is a set of timestamps -- every row at the stamp, the row itself included, the mirror of backward `"right"`; both oracles follow | `test_forward_left_and_both_hold_every_row_at_the_stamp` (with the mirror identity over repeated stamps) |
| C4 | `partial="keep"` held the last value `gap_cap` past the last row, so a kept rate's span grew with the cap | decided: a cut window ends at the last row seen | `test_a_cut_window_ends_at_the_last_row_it_saw` |
| C5 | a windowed mean on a sparse input walks the queue per read, O(n · window): 0.04 s dense, 0.24 s at one value in 5 000 (300k rows) | not changed; PERFORMANCE §35 | -- |
| C6 | a row exactly one window old under `"left"`/`"both"` counts for `min_samples` and weighs nothing | decided: by the definition, its held interval ends at its row, before the window; a mean with no other value is null (Polars' row-weighted `rolling_mean_by` says the value); documented in `ops.py` | `test_a_row_on_the_far_edge_counts_but_weighs_nothing` |
| C7 | the core took a mean on `half_life = inf` with no window (NaN forever) | refused in `Windows::new` | `windows.rs::a_mean_with_no_decay_needs_a_window` |

### Batch D: the formula targets (104)

| ID | Finding | Fix | Test |
|---|---|---|---|
| D1 | `drop_groups` left the group's window core, so the group fed again met a stale clock | the cores go with the stream | `test_a_dropped_group_starts_cold_in_the_window_core_too` |
| D2 | a boolean column was cast to a number and then refused by the resolver's dtype check | a boolean form in the chunk, for a formula target's columns | `test_a_boolean_column_reaches_a_formula_target` |
| D3 | `rows_learned` counted a formula row at arrival whatever its window gave | counted at release, when a target resolved with a value; chunk-invariant | `test_rows_learned_counts_a_formula_row_once_its_target_resolved` |
| D4 | `partial="drop"` on one operator nulled every formula of the row | a formula is null of its own operator's null; nothing else nulled | `test_drop_on_one_target_leaves_the_rows_other_targets` |
| D5 | the resolver snapshot cloned every core each chunk | taken only when a ring of the spec's streams could refuse the chunk | -- (the clone is gone from the default path; a budget that refuses is the exception) |
| D6 | per-group serial Polars plans per chunk | not changed; PERFORMANCE §35 | -- |
| D7 | a formula's spans outside `clock_spans`, so the embargo check compared unlike units | the operators' spans are the spec's clock spans | `test_a_formulas_spans_are_the_specs_clock_spans` |
| D8 | a `pl.Expr` in a raw spec dict died in `json.dumps` | normalised in `ModelBank.__init__` | `test_a_raw_spec_dict_takes_a_window_expression` |
| 150-* | the Kalman reviewer: no code defect; four stale equations fixed inside task 150 | -- | -- |

### Round two (the same day): three reviewers over the fixes and the surfaces

| ID | Finding | Fix | Test |
|---|---|---|---|
| P1 | `fit`'s learn-only flag on the bank could be left set when a concurrent `predict` held the bank; `fit_predict` then ran in sample | learn-only is a parameter of each call (`Bank::fit_predict_from_with`, the binding, `ModelBank._feed`); the bank keeps no flag | `test_learn_only_is_the_calls_not_the_banks` |
| P2 | `_order_free` exempted a formula target from the row-order warning | a target that reads the rows ahead is never order-free | `test_a_formula_target_is_never_order_free` |
| P3 | TOML has no null, and a `when/then` without `otherwise` carried one | the tree writes a null literal as `["lit"]`; `["lit", null]` still reads | `test_a_null_literal_has_a_form_toml_can_carry`, `formula.rs::every_node_kind_round_trips` |
| P4 | `--no-output` is the command line's `fit` but ran `fit_predict` | it runs learn-only | `test_no_output_is_the_command_lines_fit` |
| P5/F5 | a `pl.Expr` in a raw spec dict died in `json.dumps` on every surface but the constructor, and a tuple of targets | `_json` writes it as the builders would | `test_every_surface_takes_a_raw_dict_with_a_window_expression` |
| P6/P8 | `rows_learned`'s docstring said "counted as handed over"; `n_eff`/`window` leftovers | reworded | -- |
| P7 | `--predict` kept the config's `closed_groups`, which the scoring run refused | dropped with `save_state` | `test_no_output.py::test_predict_drops_the_closed_groups_sidecar` |
| P9 | `rolling_metrics` compared a duration window with a missing clock's dtype first | the column is named first | `test_eval.py::test_rolling_metrics_names_a_missing_clock_before_reading_the_window` |
| P10 | `ridge_scale="foo"` was refused by serde with no parameter name | checked by value in the builder | `test_error_messages.py::test_ridge_scale_is_checked_by_name` |
| F1 | (D5) with two formula specs, the second's refusal left the first's core fed | every spec's frame and cores are built before any is fed; the snapshot is taken wherever a feed can still fail after rows went in (a cast, a second formula spec, a refusing ring) | `test_a_refused_chunk_leaves_every_specs_core_as_it_was` |
| F2 | (B2) `to_json` dropped `"non_strict"`, so a saved spec ran a strict cast | the third argument is written | `test_a_non_strict_cast_survives_the_specs_round_trip`, the round-trip list |
| F3 | (D2) a boolean a feature also read reached the formula as a number (the first form found) | the boolean form is looked up first, and a formula's columns before the clock's and the session's | `test_a_boolean_also_read_as_a_number_reaches_the_formula_as_a_boolean` |
| F4 | a plain target under an embargo counts at arrival and a reset discards the pending rows, so `rows_learned` counts rows no model saw (pre-existing) | not changed; recorded here | -- |
| W1 | edge decisions subtract two rounded clocks, so a row exactly one window after any row but the stretch's first can land on the wrong side (`[0, 100, 400]` ms, `300ms`, `"left"`: null against Polars' 1) | (round three) an edge between two rows is decided from the difference of their raw clocks, exact in nanoseconds, compared as integers with the window's own nanoseconds (`gap`, `KernelDef::window_ns`; a window given as a number keeps the seconds path); the queues and the waiting rows carry each row's raw clock, one `i64` a row; windows state version 3 | `test_a_row_exactly_one_window_from_another_lands_as_in_polars` (against `rolling_sum_by`, at epoch 0 and at 2024), `windows.rs::an_edge_is_decided_from_the_two_rows_clocks` (the brute force decides its edges the same way) |
| W2 | under `session_gap="reset"` a session change is a reset, so it discards (ignores `partial`) | decided: that is what `"reset"` asks for; documented. Under `group`, a silence past `gap_cap` cuts the group's windows as the stream's clock passes the cap, before its next row can say its session changed, so a reset there discards only what is still open (round four, A1) | `test_a_silent_groups_session_reset_after_a_capped_gap_is_a_cut`, `windows.rs::a_silent_groups_reset_after_a_capped_gap_is_a_cut` |
| W3 | the stream's clock cut every group at a session change, so groups with sessions of their own restarted at every row | (round three) with a group column the stream's clock takes no session (`Windows::set_grouped`); without one the stream is the one group and its session is the stream's, which is what the bank's per-group resolvers (one core per group, no group column) need to see a session change as the bank's own clock does | `test_with_groups_a_session_is_each_groups`, `windows.rs::with_groups_a_session_is_each_groups` |
| W4 | a state saved under a slice holds rows read past the last emitted, which a resume from the emitted count feeds again | (round three, redone in round four after B1 to B3) the state records the rows of its input the run consumed, skipped or fed (`WindowsRun::consumed`, `save_bytes_with`; `with_windows` passes it under a slice), and the input's first clock; a run resumed on the same input, unsliced, skips them, one on an input starting elsewhere skips nothing; an error's row number counts the skipped rows | `test_a_state_saved_under_a_slice_resumes_on_the_same_input` (a chain of six, with and without a `"drop"`), `test_rows_held_from_an_earlier_input_are_not_this_inputs`, `test_a_state_saved_under_a_slice_skips_nothing_of_another_input`, `windows_frame.rs::a_state_saved_under_a_slice_resumes_on_the_same_input` |
| W5 | a v2 windows state written before round one loads with defaults and misbehaves | (round three) `WINDOWS_VERSION` 3; the magic and the version are read before the rest, so an old state is refused by number and not by a field it lacks | `test_a_windows_state_of_another_version_is_refused_by_its_version`, `windows_frame.rs::a_state_of_another_version_is_refused_by_its_version` |

**Round three (the same day)**: W1, W3, W4 and W5 built as the table says,
each with its failing test first. W3 was refined while building: a session
is each group's only where there is a group column, since without one the
stream is the one group, and the bank's per-group formula resolvers (one
core per group, no group column) must see a session change exactly as the
bank's own clock does. A clock that starts over with a session is then a
step back on the stream's clock under `group`, which the docstring says.
The cost of W1, measured (PERFORMANCE §36): the first build converted every
gap to seconds and cost 12% of the window core's time at 16M rows (7% on
four kernels); comparing integers with the window's own nanoseconds
(`KernelDef::window_ns`) leaves 3%, the raw clock written and moved with
each queued row, and memory unchanged. The converting build was never
committed.

### Round four (the same day): two reviewers over rounds two and three

| ID | Finding | Fix | Test |
|---|---|---|---|
| B1 | the skip a sliced run saved counted this run's fed rows less the rows returned with new sequence numbers: rows a loaded state held and returned were not counted, nor the rows the run skipped, so a chain of sliced runs broke on its third run (a step back refused) | the state records the rows of its input the run consumed, skipped or fed (`consumed`), with no count of rows returned; the contract is a resume on the same input, unsliced | `test_a_state_saved_under_a_slice_resumes_on_the_same_input` (six sliced runs, then the rest), `windows_frame.rs::a_state_saved_under_a_slice_resumes_on_the_same_input` |
| B2 | a run on the next file returns the rows the state held from the file before; counted as this input's, a slice over them saved a skip that dropped never-fed rows, and under the `df.slice(n)` contract no skip could be right once rows returned exceeded rows consumed | as B1: consumed rows only, and the same-input contract | `test_rows_held_from_an_earlier_input_are_not_this_inputs` |
| B3 | a slice that lands on the input's last output row saves a skip meant for the same input, which a run on the next file would apply to the file's first rows | the state remembers the input's first row's clock; a run on an input that starts elsewhere skips nothing; without a clock column the docstring says to resume on the same input | `test_a_state_saved_under_a_slice_skips_nothing_of_another_input` |
| B4 | a state whose magic and version matched but whose body did not read was reported as "not a state it saved" | it says the state is damaged, or written by a build that changed a field without bumping the version | -- |
| B5 | the wording of W4 (docstrings, the binding, this table, the changelog) stated the count as rows read past the last returned | reworded with the contract | -- |
| A1 | under `group`, a session change under `session_gap="reset"` that follows a silence past `gap_cap` is a cut, not a discard: `end_silent` cut the group at another group's row before the group's own reset fired; without `group` the one clock sees both at one row and the reset comes first | recorded as the rule (the cut cannot be undone once rows drained, and delaying it would unbound how long a group holds output); the W2 row, the `with_windows` docstring | `test_a_silent_groups_session_reset_after_a_capped_gap_is_a_cut`, `windows.rs::a_silent_groups_reset_after_a_capped_gap_is_a_cut` |
| A2 | `WINDOWS_VERSION` 2 to 3 changed the bytes a bank file embeds per formula target while `SCHEMA_VERSION` stayed 23, so a 23 bank holding one passed the bank's check and failed at a group's first chunk | `SCHEMA_VERSION` 24, the bank's minimum 24 (23 unreleased, pre-1.0) | `test_a_bank_state_from_before_the_windows_state_changed_is_refused_by_number` |
| A3 | `a_saved_state_resumes_at_any_row` built its fresh core on a grouped stream without `set_grouped`, passing by coincidence | the flag set | the test |
| A4 | `KernelDef::check` did not tie `window_ns` to `window_size`, so a direct caller could split membership from the far edge and `complete` | refused unless `seconds_of_ns(window_ns) == window_size`; the frame's increments-only kernel carries its window in nanoseconds so a temporal clock converts nothing | `windows.rs::a_kernels_window_ns_must_be_its_window_size` |

### Round five (the same day): one reviewer over round four's changes

| ID | Finding | Fix | Test |
|---|---|---|---|
| C1 | the saved skip left out a loaded skip not yet applied: a resumed run that satisfied its slice from held rows while the skip still spanned chunks saved too small a count, so the chain depended on `chunk_rows` (hard rule 3); a run that saw no row dropped the loaded identity | `consumed` = skipped + the skip still pending + fed; a run that saw no row saves the identity it loaded | `test_a_state_saved_under_a_slice_resumes_on_the_same_input` over `chunk_rows` 1, 7 and 100,000; `windows_frame.rs::a_resumed_run_fed_one_row_at_a_time_saves_the_whole_count` |
| C2 | a sliced run whose input ended before the slice was satisfied saved a skip of 0, so the chain's next run fed the input again | the count is saved whenever a slice was asked for | the chain run to exhaustion in the same test |
| C3 | the identity by the first clock alone took another input that starts at the same clock (a clock that starts over each day) for the same one, and skipped its rows silently | the state knows its input by the rows it holds -- the unresolved tail of the consumed prefix, the input's own rows -- compared with the rows skipped at those positions when the skip completes; where it holds none, by the last skipped row's clock against the last clock it saw; an input that ends before the skip completes is refused at `finish`, and at `save` when the caller says the input ended (`input_ended`), since a slice satisfied mid-skip is a legitimate save | `test_a_state_saved_under_a_slice_refuses_another_input`, `windows_frame.rs::a_sliced_state_refuses_another_input_and_a_damaged_file_says_so` |
| C4 | the old `df.slice(n)` contract was silent without a clock column, with tied stamps, or under `restart_after_step_back` | refused by the same identity, with and without a clock column; a stream whose rows can repeat (a daily grid with the same values) is documented as resuming on the same input only | the same tests |
| C5 | round four changed what `resume_skip` counts and added `resume_first` under `WINDOWS_VERSION` 3 | `WINDOWS_VERSION` 4 | the version tests read 4 |
| C6 | a state cut short failed in the header read and was reported as "not a state it saved" | a header that cannot be read says "cannot be read: not a state it saved, or damaged"; B4's message stays for a header that read and a body that did not | `test_a_damaged_windows_state_says_so`, the Rust test above |
| C7 | the plan's task 104 still said schema 23 | amended | -- |
| C8 | the `with_windows` docstring and the changelog promised the chain under every chunk size and slice | true now; the identity and its one limit stated | -- |

### Round six (the same day): one reviewer over round five's additions

| ID | Finding | Fix | Test |
|---|---|---|---|
| D1 | the last-clock identity was dead code: the held snapshot was set even when empty, so a sliced state holding no rows (a backward operator under `closed="left"` or `"none"`, any operator on a row-count clock) had no identity, and another input or `df.slice(n)` was skipped silently; the changelog's "with and without a clock column" was false there | the state keeps the last row it read (windows state version 5), and the identity is the last rows read: the rows it holds that are the input's, or the last row where it holds none -- one path, no fallback | `test_a_sliced_state_that_holds_no_rows_knows_its_input_by_its_last_row`, `windows_frame.rs::a_sliced_state_that_holds_no_rows_knows_its_input_by_its_last_row` |
| D2 | a run on the next file under a slice still holds the previous file's unresolved rows ahead of its own (rows go out in order, so none of its own went out while one of theirs waited), and the identity took them all for this input's: the held rows outnumbered the rows consumed, and a resume on the second file was refused as another input (traced by the author before the report, confirmed by a failing test) | the identity is the last `consumed` held rows at most | `test_a_sliced_run_on_the_next_file_resumes_with_the_first_files_rows_held`, `windows_frame.rs::a_sliced_run_on_the_next_file_resumes_with_the_first_files_rows_held` |
| D3 | an input starting at the last stamp the state read -- a file boundary inside a tied stamp, ordinary with coarse stamps -- was refused as "at or before the last row", words that did not say a split inside one stamp is "at" | kept, by decision: a tie cannot be told from the same input sliced inside its last stamp, and a loud refusal beats a silent double feed; the message and the docstring say so, and that the next file starts after the last stamp read | `test_a_next_file_starts_after_the_last_stamp_the_state_read`, the Rust test under D4 |
| D4 | a clock that starts over each day, under `restart_after_step_back` or with a `session` column, could never present an accepted next-day file after a sliced run: the rule refused every step back before the policy spoke, where an unsliced state took the file | the first row is put to the core's clocks without taking it (`Windows::peek`, the two advances `push` makes, shared): a step forward or a new start by the policy's word (a restart, a new session) is the next file; a step back the policy refuses, or the same stamp, is refused | `test_a_sliced_state_takes_a_new_start_by_the_policys_word_for_the_next_file`, `windows_frame.rs::a_sliced_state_takes_a_new_start_by_the_policys_word_for_the_next_file` |
| D5 | `WINDOWS_VERSION` 3 to 4 moved alone, round four's A2 again: a 24 bank holding a version-4 core failed at a group's first chunk | `SCHEMA_VERSION` 25, the bank's minimum 25, and a tripwire pairing the two numbers so the next move of either fails until both move | `windows_frame.rs::a_windows_state_version_moves_the_banks_schema_with_it`, `test_a_bank_state_from_before_the_windows_state_changed_is_refused_by_number` (23 and 24) |
| D6 | the binding's `Windows.save` doc, the changelog, and the docstring's error paragraph: what the run refuses while the plan runs surfaces as `polars.exceptions.ComputeError`, not `ValueError` | amended; the docstring states the rule for an input that starts elsewhere as a list | -- |
| D7 | two of round five's claims had no test: a save under `input_ended` with a skip pending, and a run that saw no row saving the identity it loaded | pinned | `windows_frame.rs::a_resumed_run_that_saw_no_row_saves_the_identity_it_loaded`; the `input_ended` leg of `test_a_state_saved_under_a_slice_refuses_another_input` |

### Round seven (the same day): one reviewer over round six's additions

Folded into round six's commit: round seven adds a field to the windows state, and no commit should carry version 5 without it.

| ID | Finding | Fix | Test |
|---|---|---|---|
| E1 | `push` lost its doc comment to the new `step_clocks`, inserted between the comment and the function | the comment moved back | -- |
| E2 | a clock that starts over at the *same* stamp each day gives the next file the saved input's first clock, so the rule never reached the policy and the identity refused the file; round six's test dodged it with a start of 0.5 | the state knows its input by the first row's session beside its clock, so with a session column the file differs at its first row and is put to the policy; without one it is the saved input until the rows differ, refused by name -- the limit the docstring states | the D4 tests with day 2 at the same first stamp, both legs |
| E3 | the last row's columns check ran on every load, so an unsliced state holding no rows refused the next file with a column more, which it took before; the bank wrote the row for every core on every save and never read it | the row is written and read under a skip only; the columns message pinned | `windows_frame.rs::the_last_row_binds_a_sliced_state_only` |
| E4 | the last row went out of step with `consumed` on two paths: a skip still pending recorded the last skipped row while the count included the rest, and a refusal mid-chunk held rows without recording the last | by construction: the row is set when the skip completes (the loaded row until then) and in the refusal branch | round eight's F3: the first path is reachable at the Rust API (a limit of 0 on a chunk shorter than the skip), pinned there |
| E5 | the bare leg of the no-rows test was refused by the count (three rows against a skip of four), never the compare; the no-row re-save test pinned the count, not the identity | another input of the same length; a `closed="left"` leg resuming the re-saved state on another input | the same tests |
| E6 | the docstring lost R4-B3's "without a clock column, resume on the same input", and did not say a hand slice starting after the last clock read (possible after a restart) is taken as the next file | both stated | -- |

### Round eight (the same day): one reviewer over round seven's additions

| ID | Finding | Fix | Test |
|---|---|---|---|
| F1 | without a clock column, a session column made any hand slice (or overlapping file) that starts in a later session the next file, silently: the rule fired on the differing first session, and on a row-count clock every row is a step forward, so the policy's "forward" was evidence of nothing | without a clock column only a new start (a new session by the policy's word) is the next file; a step forward there is refused by name, and the docstring says a next file on a row-count clock begins with a new session | `test_without_a_clock_the_next_file_begins_with_a_new_session`, `windows_frame.rs::without_a_clock_the_next_file_begins_with_a_new_session` |
| F2 | where the policy refused the first row (under `group` a next-day file's step back is refused on the stream's clock whatever its session), the message dropped the policy's own words, which name `restart_after_step_back` as the unsliced path does | the refusal appends the policy's message | `windows_frame.rs::a_refused_first_row_carries_the_policys_words` |
| F3 | round seven's E4 row said the pending-skip path was unobservable; it is reachable at the Rust API (a limit of 0 on a chunk shorter than the skip) and from the generator (`.head(0)` with `chunk_rows` below the pending skip), and the fix had no test | pinned; the E4 row corrected | `windows_frame.rs::a_save_inside_a_pending_skip_keeps_the_loaded_identity` |
| F4 | the `save_bytes_with` and binding docs named neither the session nor the last row; the docstring's "as one can after a restart" read as the only way a hand slice starts after the last clock read | amended | -- |
| F5 | a file with a skip, no held rows and no last row (which this build cannot write) would carry an identity of zero rows and pass the check vacuously | refused at load as damaged | `windows_frame.rs::a_sliced_state_without_its_identity_is_damaged` |
| F6 | the E3 test asserted the next file's height only, not that its extra column was carried | asserted | the same test |

### Round nine (the same day): one reviewer over round eight's additions

Folded into round eight's commit. The reviewer found no wrong verdict or number in the new paths; the iteration ends here.

| ID | Finding | Fix | Test |
|---|---|---|---|
| G1 | under `group` without a clock column the no-clock refusal said "starts in the session the state last read" of a group the state never read (a fresh group's clock sees no session change, so the verdict is a step forward, refused by F1's reasoning) | the message says what is true in both cases: the first row continues the session last read, or is of a group the state has not read; the docstring says the next file begins with a new session of a group the state has read | the same tests, which now pin the words |
| G2 | the docstring's no-clock sentence lacked the qualifier the clock sentence has: a next file whose first row's session is the saved input's first session is taken for the saved input until the rows differ (cyclic session labels) | stated: the next file begins with a new session and not with the saved input's first | -- |
| G3 | the binding's `Windows.save` doc still said a step forward skips nothing; the docstring's refusal list omitted the no-clock step forward; the changelog said "a later session" where any session but the saved input's first was the bug | amended | -- |
| G4 | the F1 tests asserted "another input" alone, which an arm that let the skip apply would also give | the F1 words asserted | `test_without_a_clock_the_next_file_begins_with_a_new_session`, `windows_frame.rs::without_a_clock_the_next_file_begins_with_a_new_session` |
| G5 | the F3 row understated the reach of the pending-skip path: the generator reaches it too (`.head(0)` with `chunk_rows` below the pending skip) | the row amended | -- |
| G6 | the policy's message appended by F2 carried its own `with_windows:` prefix inside the rule's | stripped | `windows_frame.rs::a_refused_first_row_carries_the_policys_words` |

<a id="s15"></a>
## 15. Review of `v0.13.0..a7a8f3c` (2026-10-05): the 53 commits since the tag

Six read-only reviewers, one per area of the production diff, each finding
re-derived here and pinned by a test that failed before its fix (the review
protocol of 2026-09-14). The reproductions are in the session's scratchpad
(`review/<area>/`). Committed in three batches; the finding IDs are in the
commit messages. Severity: high = wrong numbers silently; medium = wrong
numbers in a narrow case, or a wrong refusal or broken contract, loudly;
low = a doc or an edge.

| ID | Sev | Finding | Fix | Test |
|---|---|---|---|---|
| B1 | med | task 158's same-stamp refusal (E13) fired inside the bank's formula-target resolver, so a bank with a window target saved between two rows at one stamp refused to resume, in every entry point; the column form resumed | the check runs only where `with_windows` feeds the core, not where the bank's resolver does (`resolving`): the bank's own clock has refused any real step back first | `test_formula_targets.py::test_a_state_saved_inside_a_tied_stamp_resumes_as_one_run` (an f64 clock, three rows a stamp; a `Date` clock, grouped, four rows a day) |
| F1 | med | `feed_inner` marked the run started, and kept its input's first row, before the identity verdict, so a refused first chunk was taken whole on the next call; with two groups the retry wedged the bank | the record is written after the verdict | `windows_frame.rs::a_refused_first_chunk_is_refused_again` |
| D1 | high | `rcov`'s pre-averaged estimate forms its first `Ȳ` from `Δ₂..Δ_{k}` (CKP's `Ȳ₁`), never `Ȳ₀`, one term fewer per stretch, and scales by `n/(n−k+2)` as if it had them all; every oracle shared the code's indexing | `Ȳ` is formed once the ring holds `k − 1` returns, from them; the scale is `n / pre_count`, the terms summed | `rcov.rs::the_preaveraged_estimate_is_its_definition` and `::the_scale_counts_the_terms_summed_across_a_break`, `test_rcov.py::test_the_preaveraged_estimate_is_its_definition`, all from Eq. 9; `GOLDEN_RCOV_PREAVG` re-pinned |
| R1 | med | `ewridge(ridge_scale="sum")`'s prior is not decayed across a zero-weight row while `w_sum == 0`: 6.8e-2 from its definition after two such rows, where `rls` matches | `EwCov::update` at `w == 0` with `lam·W == 0` is the decay, head row included; the Gram split for that row goes | `ewcov.rs::a_zero_weight_head_row_ages_the_prior`, `gaps.rs::a_row_of_no_weight_parts_no_targets`, `test_rls.py::test_leading_zero_weight_rows_age_the_prior_as_rls_does` |
| R2 | med | `seconds_of_ns` rounds a negative sub-second delta at the scale of a second, so on a `Datetime` clock a step back of exactly `restart_after_step_back` at 1-3 ms restarts the model instead of being a late row | `seconds_of_ns` takes the magnitude and mirrors the sign | `clock.rs::a_step_back_is_the_mirror_of_the_step_forward`, `test_renames.py::test_the_restart_edge_is_inclusive_at_a_millisecond_on_a_temporal_clock` |
| W3 | med | on an f64 clock a window edge is decided from two origin-subtracted policy times, not the rows' raw clocks, so a row exactly one window back lands outside where Polars keeps it | a number clock's policy time is the clock itself; windows state 6, schema 26 | `test_windows.py::test_a_number_clock_decides_an_edge_from_the_two_rows_clocks` |
| W2 | med | `KernelDef::mass` cancels at long half-lives: 6e-5 at `h/t = 1e12`, every mean null from 1e17 | `mass` by `exp_m1`, and both oracles with it | `windows.rs::mass_at_a_long_half_life_is_the_even_kernels`, `test_windows.py::test_a_long_half_life_agrees_with_the_definition_and_tends_to_the_even_one` |
| F2 | med | group and session keys are the value's string form and the state carries no dtype, so an `Int64` file followed by a `Float64` one starts every group cold, and a session column's dtype change cuts the windows | the windows state carries the group and session columns' dtypes, by role, and a resumed input of another is refused by name (windows state 7, schema 27) | `windows_frame.rs::a_state_refuses_a_group_column_of_another_dtype`, `test_windows.py::test_a_state_refuses_a_group_or_session_column_of_another_dtype` |
| W1 | med | a stream restart discards every group's windows but not their clocks, so a group's next row within `restart_after_step_back` of its stale clock is refused as a late row | a stream restart starts every group's clock over, the row's own group's from a fresh one; the brute force too | `windows_frame.rs::a_stream_restart_restarts_every_groups_clock`, `test_windows.py::test_a_restart_on_the_streams_clock_restarts_every_groups_clock` |
| P1 | med | a window target without `.alias()` named after its own input column passes the builder and can never run; the bank's refusal is worded for `with_windows` | the builder refuses a target named after a column its formula reads, naming `.alias`; a bank's refusal of a formula target drops the core's `with_windows` prefix | `test_formula_targets.py::test_a_target_named_after_its_own_column_is_refused_at_the_builder` |
| F4 | low | `inf` and `nan` serialize as `{"Dyn": {"Float": null}}`, so `clip(0, inf)` is refused as "this literal (…null)" | a null value under a float's type is refused as a literal that is not finite, naming inf and nan and the one-bound clip | `test_windows.py::test_a_literal_that_is_not_finite_is_refused_by_name` |
| D2 | low | `bocpd`'s `prior_nu > 1` floor for `diag`/`robust` buys a finite mean; the message says variance | the message names the mean or the variance by emission | `bocpd.rs`, the floor test |
| B2 | low | `predict` without a clock column writes the stream's fed-row count as `scored_clock`; the docs say the row's index in its group | `predict` without a clock column writes the row's index in its group, counting on from the rows learned | `test_clocks.py::test_predict_without_a_clock_counts_each_row_in_its_group` |
| B3 | low | the unset-policy refusal advises setting `restart_after_step_back` "to the smallest one that does", where equal is a late row | the message says to set it below the step | `test_renames.py::test_restart_after_step_back_is_one_rule` |
| W4 | low | at a tied stamp `ewm_sum` gives every tied row the stamp's total, where Polars' `ewm_sum_by` is a running sum; four places say the latter | the four places say: on distinct stamps; at a repeated stamp every row carries the stamp's total | `test_windows.py::test_at_a_repeated_stamp_every_row_carries_the_stamps_total` |
| W5 | low | a subnormal `half_life` passes `check` and emits `inf` rates | `check` refuses a half-life below `f64::MIN_POSITIVE` | `windows.rs::a_subnormal_half_life_is_refused`, `test_windows.py::test_a_subnormal_half_life_is_refused` |
| W6 | low | an operator input that is NaN, ±inf or above 1e100 reads as missing; the docs say only a null is skipped | `ops.py` and the README say a NaN, infinite or beyond-1e100 value is read as missing | -- |
| F3 | low | a typed literal is rebuilt dynamic, so a formula's dtype and last bits can differ from Polars' | a typed literal is carried as a cast of the plain one, so its dtype and bits are Polars' own | `test_windows.py::test_a_typed_literal_keeps_its_dtype` |
| F5 | low | the Rust tree writes a clip's absent bound as a bare `null`, where Python writes `["lit"]` | the Rust tree writes `["lit"]`, reading a bare `null` still | `formula.rs::every_node_kind_round_trips`, `test_formula_targets.py::test_a_clip_bound_left_out_is_the_null_literal_in_a_saved_spec` |
| P2 | low | a chain of two plan forms over an empty frame raises `ConsumedSourceWarning` | a source that got no rows from a Python scan warns only when none of this package's own sources ran under it | `test_consumed_source.py::test_a_chain_of_two_plan_forms_over_an_empty_frame_does_not_warn` |
| P3 | low | `output_index`'s `dtype` vocabulary omits `enum`, `i32`, `i64` and `clock` | the docstring lists all eight | `test_model_registry.py::test_output_index_names_every_dtype_it_declares` |

What held, measured: hard rules 2, 3, 8 and 9 across every model and
both plan forms (chunk invariance bit-identical over 1,008 window
configurations and 198 pushdown cases; the embargo's release against an
oracle from its definition; zero-weight first rows NaN-free in 18 kinds);
every renamed name refused by name in builders, dicts and TOML; save/load
at 159 cuts equal to one run; `bocpd`, the sequential `corrchange`,
`kalman`, `holt` and `hmm` against oracles from their papers; the `.pyi`,
`po.eval` and 99 docstring defaults.

<a id="s16"></a>
## 16. Whole-project review at `e6b70a2` (2026-10-05): every file but Markdown

Sixteen read-only reviewers, one slice each (six over `online-core`, three
over the polars layer, two over the Python package, three over the tests,
one over the scripts, one over CI), every finding re-derived from its
reproduction (`review2/<slice>/` in the session's scratchpad) and recorded
in one list before any fix. The fixes: four workers in worktrees and the
coordinator, each test shown failing on the old code, each branch reviewed
and merged under its own gate (task 160). Severity as in §15: high = wrong
numbers silently, or a check that could not fail where it was the only one;
medium = wrong numbers in a narrow case, a crash, or a wrong refusal;
low = a doc, a message or an edge.

| ID | Sev | Finding | Fix | Test |
|---|---|---|---|---|
| SC1 | high | `compare_release.py`'s verdict ignored the specs the new build refuses, so a refused spec dropped out and "identical" exited 0 | a spec the old build ran and the new one did not is a difference, naming the refusal | `test_release_compare.py::test_a_spec_this_build_refuses_is_a_difference_not_a_silence` |
| TA1 | high | the one universal rule-2 test perturbed the first non-null target, a row every model withholds: 0 of 1,200 comparisons saw a prediction | the row is drawn from those the model scored; at least 10 streams must compare a prediction that was there (ewridge's inflation gate off in this test, since it withheld 87% of the streams) | `test_properties.py::…::test_prediction_never_depends_on_the_current_target` (no rule-2 failure found) |
| TB1 | high | the kmeans/micro oracle is the Rust transcribed bit for bit; the independent checks were coarse | kept as a regression check, beside checks from the definitions: centres, weights and radii from the raw assigned rows; far, merge and dead decisions row by row; micro's admissions and summaries; scikit-learn `KMeans` (ARI 0.99995) | `test_kmeans.py`, `test_micro.py` `TestDefinitions`; TESTING.md's table |
| TC1 | high | `kalman_ref` and `robust_ref` restate the implementation; the library second opinions covered kalman only unstandardized and huber only at delta 1e9 | the standardized kalman held to filterpy on rows standardized in numpy (1e-9) and `coef·x[t+1] = pred[t+1]` (1e-12); the quantile fit to its smoothed check loss's stationarity (1e-15); the nudge to its purpose, which found TC1b | `test_second_opinion.py`, `test_robust.py::TestTheQuantileFitsDefinition` |
| TC1b | high | found by TC1 (iii): `quantile`'s nudge was bounded by the band Gram's diagonal leverage; on correlated features the next solve moved the row by 6 to 127 times its residual, past its target | the full leverage `1 + u'A⁻¹u` against the band's own system, from a kept `SpdFactor` that is `solve_spd`'s to the bit; `GOLDEN_QUANTILE` and four golden-pipeline values moved; the reference oracle and the stationarity test's reference take the full leverage too. Costs half the default schedule's throughput (2.6M against 5.0M rows/s, 10 features; 8% solving every row): a decision | `robust.rs::a_nudge_never_moves_its_row_past_its_residual`, `::the_full_leverage_is_the_diagonal_one_on_uncorrelated_features`, `test_robust.py::…::test_a_nudge_never_moves_its_row_past_its_residual` |
| PB1 | high | under an embargo the drift detector ran on the learn pass, so `drift_<t>` was never true and `drift_action="reset"` restarted silently | the flag goes on the row that released the label that tripped it (`RowPlan::drift_ri`), the restart before that row is scored | `test_label_delay.py::TestDriftUnderAnEmbargo` (8) |
| PB2 | high | `gap_cap` without a clock passed and capped the row-count step: `gap_cap=0.5` halved every decay, marked every row capped, cleared lags every row | refused, in specs and `with_windows` | `spec.rs::gap_cap_needs_a_clock`, `test_gap_cap_without_a_clock_is_refused_by_name` |
| PC2 | high | a forward `right`/`both` window whose far edge is exactly the stretch's last row before a break was cut as partial: null, or dropped under `partial="drop"` (2,309 of ~22k fuzz rows) | whole, as its mirror is. A reset still discards it, as documented: a decision | `test_windows.py::test_a_forward_window_whose_far_edge_is_the_last_row_before_a_break_is_whole`, the Rust mirror test across breaks |
| PA1 | high | `load_bytes` indexed specs by the states' count: a file with an extra states entry panicked, one with fewer lost a spec's groups | refused as damaged | `bank/damaged_file_tests.rs::a_file_whose_states_do_not_match_its_specs_is_refused` |
| PA2 | high | closed rows restored unchecked (`closed_groups()` panicked, `save_bytes` re-saved the poison); counts at `u64::MAX` wrapped | every queued row checked against its spec at load; counts saturate, and a summary whose counts wrap is refused | `::a_closed_row_that_does_not_fit_its_spec_is_refused`, `::a_row_count_at_the_top_of_its_range_saturates`, `::a_summary_count_that_wraps_is_refused` |
| PA3 | high | an `Int128` key past `i64` was cast non-strict to a null group, merged with real nulls | read as its text, marked an integer so `"monotone"` orders it as a number | `integer_group_keys_match_the_string_cast` (Int128), `a_monotone_int128_key_is_ordered_as_a_number` |
| PA4 | high | `refresh_time`'s group keys round-tripped through String: `Datetime`/`Time`/`Struct` came back null, `Boolean` and zoned failed | the input's own values at the completing ticks; a zoned key by its instant | `test_refresh_time.py::test_the_group_column_is_the_inputs_at_the_completing_ticks` (8 dtypes), `a_zoned_datetime_group_comes_back_as_it_went_in` |
| CI1 | med | `dtolnay/rust-toolchain@stable` is a branch, at 13 sites, the release's wheel build among them, against `.github/dependabot.yml`'s rule that actions are pinned to commits | pinned to the `stable` branch's head commit, the version in a comment | `test_ci_cost_policy.py::TestActionsArePinnedToCommits` |
| CI3 | med | three of the six wheels (Intel macOS, aarch64 Linux, musl) were built and uploaded without ever being imported, and none was run as the file that ships | each build installs its wheel into a fresh environment, dependencies from PyPI, and runs `scripts/wheel_smoke.py` (the version, a fit, a state round trip, the streaming plan); the musl wheel in `python:3.12-alpine` | `test_release_workflow.py::test_every_wheel_is_installed_and_run_where_it_belongs` and the two smoke tests; run here on a built arm64 wheel in a clean venv |
| SC2 | med | REGIMES.md §8's prose kept 0.151 where its table read 0.150 after task 159 | 0.150, and the section dated | `test_regimes_doc.py`: sections 1, 5, 7 and 8 against the experiments that print them (macOS) |
| SC3 | med | no benchmark printed its commit, versions, platform or load: a 576 ms reading at load 524 stood against the README's 155 | `scripts/bench_header.py`, printed first by the seven benchmarks | `test_bench_scripts.py` |
| SC4 | med | `parallel_bench.py` ran BSD `time -l`, and `windows_bench.py` imported `resource` at the top: neither ran on Linux or Windows | memory by platform: `time -l` on macOS, GNU `time -v` on Linux, NaN elsewhere | `test_bench_scripts.py::test_the_platform_specific_benchmarks_import_everywhere`, `::test_the_memory_measure_runs_on_this_platform` |
| YB3, YB10 | med | `po.stream.embargo` with a float delay on an integer clock, the docstring's own `delay=5.0`, died in the merge of the two copies (`SchemaError`) | a whole delay is added in the clock's dtype; a fractional one, and a frame already holding `role + "_weight"`, are refused by name (YB10) | `test_label_delay.py::TestRefusals::test_a_whole_delay_keeps_an_integer_clocks_dtype`, `::…_cannot_hold_is_refused_by_name`, `::test_embargo_names_the_weight_column_it_would_add` |
| YB4 | med | `po.eval` dropped null targets but not the NaN ones the bank predicts: `r2`, `ic`, `mse` NaN, `hit_rate` quietly lowered (0.889 against 0.914) | rows whose target or prediction the bank reads as missing (null, NaN, infinite, beyond 1e100) are dropped, as `eval.seqtest` already did | `test_eval.py::test_what_the_bank_reads_as_missing_is_missing_here_too` |
| YA2 | med | a broken bank's `save` raised a bare `OSError` where `save_bytes` and pickle raise `ValueError`, and `to_json` exported the state `save` refuses | `save` maps the refusal (`io::Error::other`, a kind std never produces, by its docs) to `ValueError` and leaves the file alone; `to_json`/`save_json` refuse it | `test_window_budget.py::test_a_broken_bank_names_itself_before_a_bad_column` |
| PB7 | med | `lasso(max_iter=0)` passed the builder and `validate`: no descent, every solve a failure, output that reads like a fit | the builder refuses `max_iter < 1` (the Rust side is worker 2's) | `test_error_messages.py::test_a_bad_value_names_the_parameter[max_iter0/1]` |
| YB1 | med | a `Date`/`Time` literal was read as its bare integer: a cast to String compared false silently, and a date less it was refused | the literal is carried as a cast to its dtype, read on both sides | `test_windows.py::test_a_date_literal_keeps_its_dtype`, `formula.rs::a_date_or_time_literal_is_its_own_dtype` |
| YB2 | med | a cast to a parametrized dtype crashed with `TypeError: unhashable type: 'dict'` before the refusal | refused by name (`a cast to Datetime is not read`) | `test_windows.py::test_a_cast_to_a_parametrized_dtype_is_refused_by_name` |
| YB5 | med | `partial` was accepted without a `window_size`, and did nothing | refused, saying a window must exist to be cut short | `test_windows.py::test_partial_needs_a_window_size` |
| YA1 | med | `features=[]` raised `IndexError` in the eight builders of models with no target | `features must be non-empty`, by name | `test_error_messages.py::test_an_empty_features_list_is_named_by_a_model_with_no_target` (8) |
| TA2 | med | drift fired in none of the four legs, so the reset leg and the river comparison tested nothing | a level shift makes it fire; the reset leg's `weight_sum` drops to 0 after each flag; river's PageHinkley agrees flag for flag at `half_life=inf`, where its plain mean is ours | `test_second_opinion.py::TestAZeroWeightRowIsNotSeen` |
| TA3, TB2 | med | an HTTP 404/403/500 was an `OSError`, so "offline" and a skip; `test_windows` caught `OSError` where offline raised `RuntimeError` (TB2) | a 4xx raises at once; 5xx, 429 and the network retry, then raise `data.Offline`, the one thing a test skips on (also `test_released_state`) | `test_data_and_reference.py::test_only_the_network_is_offline` |
| TB5 | med | `N_EFF_MODELS` ("every model that emits weight_sum") missed eight kinds; `CLOCK_MODELS` missed marginal | both built from `MINIMAL` and the Rust clock-field table, exemptions checked against the builders' refusals; corrchange, deco, hmm, kmeans and micro hold the exact recursion; the marginal leg found defect 2 (`specs` round trip) | `test_properties_temporal.py` |
| TB7 | med | the ew_cov lag oracle was the recursion written out | the lagged co-moment equals the unrolled sum over the raw rows (AR(1), zero weights) to 1e-9 | `test_ew_cov.py` |
| TC4 | med | the docstring-block test claimed a NameError on an unshown name while handing every block 16 names; `skip_learned`'s example read `rerun`, built nowhere | the example builds what it reads; a rule test: a docstring block reads only names its docstring builds and the Example-data frames (29 docstrings fixed, defect 3) | `test_production_hardening.py` |
| TC5 | med | README blocks ran on fixture objects that differ from the README's own (`grid`'s `min_weight`, `now`) | blocks run on names the README's own latest defining block builds (83 substitutions); the fixture's copies pinned | `test_production_hardening.py` |
| TA4 | med | every simulator block was state 0, and `continue` skipped ρ = 0.7 silently | `durations=[4, 4]`; each state must occur | `test_sim.py` |
| TB4 | med | corrchange's break sat on a span boundary, its one flag a false positive, and the restart check behind a false condition | the break mid-span at row 150; `since_flag` 200 then 100 | `test_corrchange.py` |
| TB6 | med | two kalman tests asserted only a finite prediction | the outputs differ | `test_kalman.py` |
| TC2 | med | `test_ew_variance_tracks_river` never called the library | `ew_cov(stats=["var"])` against river's EWVar, 1.5e-14 in the limit | `test_river.py` |
| TC3 | med | a value compared with itself; a module-level rng made data depend on order | captured before, compared after; a seed per test | `test_ffi_memory.py` |
| CC2 | med | ftrl without a rate guard in the non-forgetting arm: one subnormal gradient at `beta = l2 = 0` made a coefficient infinite and every later row null | no evidence and no prior is no fit (0), as under a half-life | `ftrl.rs::a_rate_of_zero_is_no_fit` (both decays), `::a_row_too_light_to_square_leaves_the_rows_after_it_learning` |
| CC3 | med | an overflow-skipped ftrl row still aged its target's sums and moved the scale: the next prediction moved 2.2% | `Ftrl::teach` returns before the owed decay, `W*`, the scale and the target's weight move | `ftrl.rs::a_row_that_teaches_nothing_leaves_the_fit` (its overflow twin) |
| CB2 | med | the window's target moments lacked the feature side's floor: a target held at 1e8 exported `target_vars = 2.5e-9` | the 64ε floor, the means' low parts kept in snapshots; the held-run rule is not built (the targets keep no runs: new state and a schema bump) | `gaps.rs::a_target_held_over_the_window_has_no_spread_there` |
| CE2 | med | `marginal`'s bin `m2` overflowed for a target spread above ~1e79 once the scale neared its fold | the scale folds at 1e-50; ordinary scales unchanged (2.7e-13 against 2.8e-13) | `margbins.rs::a_target_spread_at_the_bound_survives_the_folds` |
| CE3 | med | a window of rows each ≤ 1e-12 of the decayed weight but together above it reported the last heavy row's value held (var = cov = 0) | every row with a weight counts for the runs, in `marginal` and `EwCov`; the window alone says what is nothing | `marginal::a_window_of_light_rows_reads_their_moments`, `ewcov::…` |
| CE4 | med | no ceiling on rcov's `bandwidth`, `max_bandwidth`, `preavg_rows`, `theta`, `jitter` or strides: "capacity overflow", or an abort at `theta = 1e15` | at most `block_rows` where given and 2^20 in any case; `theta`'s window read as a double | `rcov.rs::a_bad_configuration_is_refused_by_name` |
| CA1 | med | `ewridge`'s restore checked neither `beta`, `ready` nor the twin: `beta = []` loaded and predict panicked | refused at load | `ewridge::a_state_whose_fit_or_readiness_is_the_wrong_shape_is_refused` |
| PB3 | med | `half_life=[]` passed the builder and validate: zero instances, an empty struct | refused | `test_an_empty_grid_is_refused_by_name`, `an_empty_half_life_grid_is_refused` |
| CD1 | med | `boundary_gamma` up to 0.5 admitted; the solve's work ∝ 1/(½ − γ): 32 s at 0.499, an hour at 0.49999 | at most 0.49, refused with the reason | `corrchange.rs::a_bad_sequential_configuration_is_refused_by_name`, `test_corrchange.py::test_the_boundary_exponent_is_at_most_0_49` |
| PA5 | med | `RefreshTime::load_bytes` checked nothing: a narrow grid panicked | refused; grid ticks saturate | `a_state_whose_grids_do_not_fit_its_series_is_refused` |
| PA6 | med | `time_refresh` was the ns integer as Float64 (…2001 → …2048) | the completing tick's clock in the clock's dtype, exact | `test_time_refresh_is_the_completing_ticks_clock_in_its_own_dtype` |
| PA7 | med | `pca_prev` grew per key forever under `group_close="session"`; `drop_groups` left it | dropped with the group | `drop_groups_drops_the_groups_pca_continuity` |
| PA8 | med | `POLARS_ONLINE_MAX_THREADS=100000` panicked where `ComputeError` is promised | the documented error | `a_pool_the_system_will_not_start_names_the_variable` |
| PC3 | med | `Duration`'s equality included its text: a state saved with `"5s"` refused `"5000ms"` | equal by length; a bank loaded under another spelling restores every stream under the caller's spec and renames its PCA continuity (PC3b) | `a_state_resumes_another_spelling_of_one_length`, `bank.rs::a_bank_loaded_under_another_spelling_keeps_its_pca_continuity` |
| PA4b | med | found by worker 2: a zoned `Datetime` group or session column was refused by the bank with polars' inner message (no `timezones` feature) | keyed by its instant as UTC wall time (`arrow::key_text`) in the bank, `with_windows`, `refresh_time` and `skip_learned` | `bank.rs::a_zoned_datetime_group_is_keyed_by_its_instant`, `test_error_messages.py::…`, `test_windows.py::…` |
| CF1 | med | the ew_cov contract probe asserted rows 0-1 only | the decayed row and the gap asserted | `model_contract.rs` |
| MUT | med | `target_n_eff_into` in ftrl, pa and sgd had no Rust test (6 survivors); `clock.rs`'s `<`→`<=` survived | tests kill them; the clock mutant is not an equivalent (zero recursed until the stack overflowed) | `::the_trait_reports_each_targets_own_weight` (3), `clock.rs::no_time_is_positive_zero` |
| TC11 | low | `test_soak.py` ran in no workflow, and its resume test had failed since task 120: its tail restarted the clock at 0 | the weekly leak-check job runs `-m soak`; the tail continues the clock | `test_ci_cost_policy.py::TestEveryDeselectedMarkerRunsSomewhere`; the soak itself |
| CI4 | low | `rust-version = "1.85"` while the lock needs 1.95 (`sysinfo` 0.39.6), so an sdist build on 1.85-1.94 failed inside a dependency | 1.95 | `test_ci_cost_policy.py::TestTheDeclaredRustVersionBuildsTheLock`, from `cargo metadata --locked` |
| CI8 | low | ci.yml's concurrency group lacked the event, so the Monday run and a push to main could cancel each other, and only a push publishes the reference | the group carries `github.event_name`. The red docs job on `a7a8f3c`'s run was GitHub failing to acquire a runner (zero steps), not a superseded deploy | `test_ci_cost_policy.py::test_a_scheduled_run_cannot_cancel_a_push` |
| SC5 | low | `coverage.sh` built `online-py`, so every test binary linked libpython | `--exclude online-py` | `test_ci_cost_policy.py::test_the_coverage_script_leaves_it_out_too` |
| SC6 | low | three scripts said scikit-learn was not installed, or to add it with `--with`, where the dev group has it | they say `uv sync` | `test_dependency_policy.py::test_no_script_tells_the_reader_to_install_a_dev_library` |
| SC7 | low | `doc_review.py`'s bold-lead split missed a lead holding a code span with a star | a code span is one token of the lead | `test_doc_review.py::test_a_bold_lead_holding_a_code_span_with_a_star_is_split_too` |
| TC7 | low | the hygiene test missed the suite's own state formats and any binary file | `.state`, `.msgpack`, `.mpk` and `.bin` are data, and no tracked file may be binary (a NUL in its first 8,000 bytes, git's test) | `test_repo_hygiene.py::test_no_binary_file_is_tracked`, `::test_the_checks_know_the_suites_own_formats` |
| TC8 | low | the release test pinned `publish`'s literal `needs`, which passed with a new job left out | every job but the three after the upload is upstream of `publish`, transitively | `test_release_workflow.py::test_publishing_waits_for_every_job_and_the_tag_waits_for_publishing` |
| YA4 | low | `targets` was an invariant `list`: mypy rejected six documented forms | a covariant `Sequence`; a bare `str` now type-checks and is refused at run time by name (a trade-off: no annotation excludes `str` from `Sequence[str]`) | `test_kwargs_typing.py::test_every_documented_target_form_type_checks` (mypy over the forms) |
| YA5 | low | `targets=` on `deco`, `bocpd`, `corrchange`, `hmm`, `rcov` raised `_common() got multiple values` | refused as `ew_cov` refuses it | `test_error_messages.py::test_a_model_with_no_target_refuses_targets_by_name` (5) |
| YA6 | low | a word such as `"inf"` given to `rolling_metrics`/`embargo` on a temporal clock lost the `who: key` prefix | the parse error carries the helper's and the parameter's name | `test_temporal_clock.py::…::test_what_is_not_a_duration_is_refused_by_name_by_a_helper` |
| YA7 | low | `half_life="inf"` did not round-trip: `ModelBank([s]).specs[0] != s` | an infinity word is kept as `float("inf")` | `test_temporal_clock.py::TestADurationSurvives::test_save_load_and_the_specs_it_was_built_from` |
| YA8 | low | `sgd` accepted `huber_delta`, `quantile`, `eps` and `power` beside a loss or schedule that does not read them | the builder refuses each by name; Rust `validate` too, for a raw dict or TOML (YA8b, worker 2) | `test_sgd.py::TestPlumbing::test_a_parameter_of_another_loss_or_schedule_is_refused_by_name` (8) |
| YA9 | low | `format_of_path`'s doc comment sat on `RefreshTime` | moved | `test_kwargs_typing.py::test_each_native_item_carries_its_own_docstring` |
| YA10 | low | spec.py said every field is computed before the row (`coef` and `support_coef` are after it); a stale error example; 19 builders' Output rubrics omitted `settled_frac`/`withheld_reason`; "largest first" in duration text, which Polars does not require | the docs say each; `coef` on row t is the fit row t+1 is predicted with (measured: exact for ewridge and rls under the default cadence, 8.9e-16 for sgd's lane order) | `test_model_registry.py::test_every_builders_output_rubric_names_the_fields_its_plainest_spec_writes`, `::test_the_field_grammar_names_every_field_a_grid_writes` |
| YA11 | low | an expression under any key of a raw dict was read as a formula | a duration expression or `timedelta` under a clock parameter is converted as the builders convert it; an expression is a formula under `targets` alone, refused by name elsewhere | `test_formula_targets.py::test_a_raw_dict_takes_a_duration_under_a_clock_parameter` |
| YB6 | low | `fit`/`fit_predict_batches` over this package's own plan form on an empty input warned `ConsumedSourceWarning` | the source-run counter is read before the plan is collected | `test_consumed_source.py` (the chain test) |
| YB7 | low | a `save_state` that is a directory failed after the whole stream | refused when the plan is built, `IsADirectoryError` on every OS | `test_frame.py::test_a_save_state_that_is_a_directory_is_refused_before_the_stream` |
| YB8 | low | `po.gram.merge([g])` dropped `group`, the instance and the lags | one part is returned as it is | `test_gram_module.py::TestMerge::test_merging_one_gram_returns_it_unchanged` |
| YB9 | low | `metrics` gave `r2 = -inf` and `ic = NaN` at zero variance where `from_sums` gives null | null where undefined, as `from_sums` | `test_eval_sums.py::…::test_metrics_is_null_where_from_sums_is` (4) |
| YB11 | low | `rolling_metrics(window_size=inf)` gave one bucket with `window_start = NaN` | refused, pointing at `po.eval.metrics` | `test_eval.py::test_rolling_metrics_refuses_a_window_that_is_not_finite` |
| YB14 | low | `_frame.py` said a join above a bank is never reported; on 1.44.2 it is | the doc says what is measured | `test_order_hazards.py::test_a_join_above_a_bank_is_reported_where_polars_serializes_the_plan` |
| CI5 | low | `po.ReadinessWarning` and `po.FormulaTarget` were exported and on no reference page | added | `test_api_links.py::test_every_name_the_package_exports_is_in_the_reference` |
| CI6 | low | llms.txt, the bug template, the Pathway example and two README sentences said `coef` is the one field that follows the chunking (`support_coef` too) | each names both | `test_llms_txt.py::test_it_names_every_field_whose_schedule_follows_the_chunking` (the README sentences included) |
| CI7 | low | llms.txt showed a clocked spec without `gap_cap`, which the bank refuses | the rule is stated | `test_llms_txt.py::test_it_says_a_clock_needs_a_gap_cap` |
| TC6 | low | sgd's `clip_gradient` bound in none of 18 replay cases; a test restated `sgd.rs` | a case where the clip binds on more than 1,000 coordinates under every schedule, held to a reference from the per-coordinate rule | `test_oracles_gradient_paths.py` |
| TA5 | low | the filterpy oracle's process noise was linear in d, passing at d ∈ {0, 1} | d², on a clock with steps 1, 4 and 0.5 (149 of 300 rows differed before) | `test_second_opinion.py` |
| TA6 | low | the CLI resume test read files the previous test wrote | one module fixture runs the CLI for both | `test_examples.py` |
| TA7 | low | a refusal test skipped when the scorers finished first: a race, not a platform guard | five rounds, then `pytest.fail` | `test_predict.py` |
| TA8 | low | "no embargo, the same rows learned" asserted `rows_seen == height`, true of any fit | `rows_learned` (7) and the whole Gram, bit for bit, against `embargo=W`; `W + 2.5` gives 6, so it can fail | `test_formula_targets.py` |
| TA9 | low | the psd form's test asserted only a smaller `rcov_n` | held to its definition (k_n = 37) to 1e-9 | `test_rcov.py` |
| TA10 | low | a skipped row's test checked `weight_sum` alone | every field null | `test_properties.py` |
| TA11 | low | two tests slept 0.2 s before asserting a source closed | wait for it (gc plus a polars call, since pyo3 releases the source at polars' next call), bounded by the engine's measured read-ahead (6 streaming, 4-7 in memory; bound 8) | `test_frame.py` |
| TB8 | low | the conformal module docstring left out `w/w̄` | fixed | -- |
| TB9 | low | `… or dropped.height < 11` passed for any drop; two lines asserted a constant | the precise claim (height 4, first t 33.0); the constant gone | `test_windows.py` |
| TB10 | low | `TestClockColumnTypes` said a temporal clock is refused | re-documented to what it tests | `test_edge_cases.py` |
| TB11 | low | a negative-weight test duplicated another for 7 of 10 models | deleted | -- |
| TB12 | low | deco's embargo test asserted only that it was accepted | the embargoed run differs from the plain one and equals the delay applied by hand | `test_deco.py` |
| TB13 | low | the null-target and null-weight tests were met by the stream layer whatever the model learned | per model (11): after a null target every later row equals the weight-0 stream, `weight_sum` up by the row's decayed weight; a null weight equals the row deleted | `test_semantics_all_models.py` |
| TC9 | low | four assertions that could not fail | one deleted, three compare with something real | -- |
| TC10 | low | "four threads never overlapped" could fail on a fast machine | retry rounds ending in `pytest.fail` | `test_error_messages.py` |
| CE6, CE6b | low | windowed Kish sizes cancel: 2% off at 1e-8, `inf` at 1e-11 in `marginal`; 10.75 against 11 at 1e-6 in `ew_cov`, the regressions and the target moments (CE6b, found by worker 1) | null where the window's Kish sum is within 64ε of the history's | `kish_size_inside_a_window_is_the_rows_inside_or_nothing` (marginal, ewcov), `gaps.rs::kish_sizes_inside_a_window_are_the_rows_inside_or_nothing` |
| CC5 | low | a zero-weight quantile row outside the band moved σ² by 1 ulp | it leaves before the σ² update | `robust.rs::a_zero_weight_row_keeps_the_residual_variance_to_the_bit` |
| CB3 | low | `pca_every`'s docs said learned rows; the code counts every stepped row | the docs say every row, weight-zero rows included: C7's cadence rule (the code is the released behaviour) | `ewcov.rs::a_zero_weight_row_counts_toward_the_pca_cadence` |
| CE5 | low | `preavg_rows=2` without `psd` gave the zero matrix | refused; `theta`'s window at least 3 without `psd` | `each_stride_is_refused_at_zero_and_a_window_of_two_is_allowed` |
| CA5, CA6 | low | a failed first solve left zeros and a share of 1.0; a combo skipped for no weight kept stale shares | NaN `coef` and a null prediction; no shares | `ewridge::a_first_solve_that_fails_leaves_no_fit`, `::a_combo_skipped_for_no_weight_reports_no_shares` |
| CB6 | low | ew_lagcov's only definitional test was the recursion | an oracle from the unrolled definition | `ewlagcov.rs::the_recursion_is_its_unrolled_definition` |
| CF5 | low | no `Decay::check` in the Rust API | every decay-carrying `validate` checks first | `model_contract.rs::every_model_refuses_a_decay_it_cannot_run_on` (17), `clock.rs::a_decay_is_checked_at_its_edges` |
| CF6 | low | `holds()` could pass vacuously | the slope is an `Option`; the windowed ridge test tightened | the held-value tests (`0 of 3000` rows predicted fails) |
| CF7 | low | EXTENDING's hook table missed `set_solve_share` and the readiness hooks | rows added | `exactly_the_scheduled_solvers_report_a_solve_share` |
| PB4, PB5 | low | wrong-length `coef_min`/`coef_max` dropped when trivial; the formula-target name check was the builder's alone | refused; `validate_spec`, `ModelBank` and the CLI dry run refuse it | `test_the_rust_side_names_the_offence`, the builder test extended |
| YA8b | low | found by worker 3: Rust `validate` accepted the sgd parameters the builder refuses (YA8), from a dict or TOML | refused in the builder's words | `sgd_tests::…`, `test_a_raw_spec_with_a_parameter_nothing_reads_is_refused` |
| PC5 | low | the brute force shared the forward closing rule with the core | it decides completeness from the stretch, on half-unit grids reaching the edge exactly | the brute force (`seed 5: drop flags differ` on the old core) |
| CD2, CD3, CD4, CD5, CE7, CE8, CA4, CB4, CB5, CC6, CC7, CF2, CF3, CF4, PA9, PB8, PB9, PB10 | low | a message naming no parameter (`truncate`); a vacuous hmm test; bocpd's row-one sentence (`min_weight` gates it); `alpha_adjust` doing nothing under the window kind (now refused); doc drift in kmeans, rcov, lasso, ewcov, window, sgd, conformal, clock, model, bank, spec | each fixed, each message or claim pinned where a test can read it | `bocpd::a_bad_configuration_is_refused_by_name`, `hmm::a_uniform_chain_is_the_classifier`, `test_a_parameter_of_another_kind_is_refused`, `test_a_deco_block_refusal_reads_as_one_sentence`, `test_a_zero_delay_is_told_to_leave_it_out` |
| YB12 | low | `online --dry-run` said "config OK" for an input that is not there | the dry run opens the input and names a missing one | `a_dry_run_names_an_input_that_is_not_there` |
| YB13 | low | `--no-output --predict` was refused with a message pointing at fixes that cannot apply | refused, naming `--predict` | `no_output_with_predict_is_refused_naming_predict` |

**Decisions left for the user.** Each has its evidence in the review's
scratchpad and is unchanged in the code:

- **CC1**: `drift_threshold` is in σ times clock units; on a temporal clock
  `d_clock` is seconds, so the default 20 fires on noise at a row a minute
  (237 of 3,000 rows) and a row a day (435), and under
  `drift_action="reset"` the model restarts on noise. Which unit?
  *Decided 2026-10-06, the user's word ("make drift_threshold a clock
  parameter as you recommend"); built as task 168.* The seconds were the
  scale every temporal clock is read on, not a unit anyone chose, so the
  threshold became a clock parameter.

- **CE1**: `rcov`'s pre-averaged estimate under `psd=False` omits CKP's
  footnote rescaling `1/(1 − ψ₁/(θ²ψ₂)/(2n))`: 0.842 of the integrated
  variance at `k_n = 6`, 0.941 at 10, 0.985 at the default. Applying it
  moves the golden. *Decided 2026-10-06, the user's word ("follow your
  recommendations for all remaining review decisions"): applied, task 171.*

- **CC4**: `kalman`'s observation and process noise are the literal 1.0 in
  target units until the first residual, so the warm-up's gains depend on
  the target's units (a 100% difference between scales of 1e-6 and 1e6).
  *Decided 2026-10-06, the user's word ("follow your recommendations for
  all remaining review decisions", then "p0 relative"): the noise from the
  first innovation and the prior as `p0` times it, task 172.*

- **CB1**: a model's window edge is decided on its own summed `d_clock`, so a
  row exactly `window_size` old is dropped on some rows and kept on others;
  the window operators were fixed in task 159 (W3), the models' windows
  were not. Exact edges need the raw clocks in the models. *Decided
  2026-10-06, the user's word ("Go with option 1"): stamps on the decayed
  clock held exactly, task 175.*

- **PC1**: a silent group on a row-count clock with a forward window holds
  every later row of every group until the end (500,001 rows held, 117 MiB
  over 2M rows), where the docs promise memory of one window. Refuse it, or
  bound it? *Decided 2026-10-06, the user's word ("follow your recommendations for all remaining review decisions"): refused, task 173.*

- **CA3**: `lasso`'s selection starts counting errors at the model's own
  first prediction, gated by the list's smallest `min_weight`, so
  `penalty_selected` differs on 84 of 240 rows between `[0, 60]` and
  `[60, 60]`. When should the selection start counting? *Decided 2026-10-06, the user's word ("follow your recommendations for all remaining review decisions"): from each
  target's own `min_weight`, task 174.*

- **CI2**: run 35508619563 (v0.8.1, 2026-09-20) still waits at "publish to
  PyPI", approvable with one click. Cancelling it is a write. *Cancelled
  2026-10-06 on the user's word ("Yes to the three items"), with the push
  of task 167 and the merges of Dependabot's #3, #4 and #5.*

- **PC2, a reset**: a cut now keeps a forward window whose far edge is the
  last row before it whole; a reset still discards it, as its docs say ("a
  reset discards them"). *Decided 2026-10-06, the user's word ("follow your recommendations for all remaining review decisions"): kept whole on a reset too, task 173.*

- **TC1b's cost**: the full-leverage bound halves `quantile`'s throughput on
  the default solve schedule (2.6M against 5.0M rows a second, 10
  features; 8% solving every row). Accept it, or skip the leverage where
  the bound provably cannot bind (bit-identical and resume-safe, but it
  needs a floating-point margin argument and a review)? *2026-10-06: the
  user asked instead for the Cholesky update, in one place; built as task
  170 (`solve.rs`), with bit-identical savings that take back about a
  sixth; its use in `robust` is parked, below.*

**Left as they are, with the reason.** CI9: CLAUDE.md says the Polars pins
in `pyproject.toml` and `Cargo.toml` "must match", where they are two
numbering schemes kept in step; CLAUDE.md is the user's to edit. YA4's
`Sequence` typing takes a bare `str` as a `Sequence[str]`; no annotation
excludes it, and the builders refuse it by name at run time. CB2's held-run
rule for targets needs per-target runs in the state, a schema bump; the
64ε floor covers the case found. `bank.rs` stood at 249,165 bytes against
the 250 KB cap; its four inline test modules moved to files of their own
under `src/bank/`, as `damaged_file_tests` had (2026-10-06, on the user's
word), leaving 242,524 bytes of production code.

**What held, measured.** No rule-2 failure under the corrected property
(TA1); every other kind round-trips its spec dict (only marginal did not);
corrchange, deco, hmm, kmeans and micro hold the exact `weight_sum`
recursion (TB5); kmeans and micro against their definitions and
scikit-learn (ARI 0.99995); the standardized kalman against filterpy
(1e-9); the quantile fit stationary for the loss it smooths (1e-15).

<a id="s17"></a>
## 17. Parameters counted in rows, where a clock would fit (2026-10-06)

The user, after task 161: "Find more example where a parameter references
only rows when it should also be a clock", then "Do both these and document
the rest for review". Every integer parameter of every builder was read
against what it counts (`_spec.py`, `spec.rs`, the models). Two are built on
the regressions' pattern -- clock units in the parameter, a row cap beside
it, whichever comes first, the old default kept: `micro`'s pruning (task 163)
and the window snapshots (task 162). The rest are listed here for the
user's review, each with what it counts, what a clock form would mean, and a
recommendation. Nothing below is built.

**Cadences that could take the clock, as `solve_every` does.**

| Parameter | Counts | A clock form | Recommendation |
|---|---|---|---|
| `coef_every` (every model with coefficients) | each group's accepted rows, rows of weight zero and rows with a null target included (its doc said learned rows until task 164) | "write `coef` every five minutes": an output cadence, so no number moves | build it, on the pattern; one decision first: `coef_every = 0` means "only each group's last row in a chunk", where `0` in the clock parameters means "every row", so the clock form needs its own spelling of that default |
| `kmeans`' `update_every`, `split_merge_every` | learned rows | batch the centres' updates, and look for splits and merges, every so long | leave as rows unless a stream needs it: both are closer to batching knobs than to a span of time |

**Dynamics stated per row, a model change rather than a cadence.**

| Parameter | Today | Clock-aware | Recommendation |
|---|---|---|---|
| `bocpd`'s `hazard` | "the per-row chance of a break", `1/hazard` | the chance grows with the time since the last row: `H(d) = 1 - exp(-d / tau)`, `tau` the expected time between breaks, in clock units | worth building as an option (a duration `hazard` on a temporal clock): on irregular rows the per-row hazard says breaks follow the row rate. `hazard_col` can carry such a per-row value today, computed from the clock -- from the step *after* its row, since a row's hazard is the chance of a break after it (corrected 2026-10-06: task 179's worker found the step into the row dates every break a row late) |
| `hmm`'s transition matrix | a per-row Markov chain | a continuous-time chain, `P(d) = exp(Q d)` | record only: estimating `Q` from soft counts over irregular gaps is a larger change, needed only for irregular data whose regimes switch on the clock |
| `deco`'s `alpha`, `beta` | per-row dynamics | dynamics per unit of time | record only, as for `hmm` |
| `sgd`, `pa`, `ftrl` coefficients | learn per row; `n_eff` decays on the clock (hard rule 8) | -- | by design, documented |

**Rows by definition, to keep.**

- `lags`, `cross_lags` (`ew_cov`, `marginal`) and `resid_autocorr_lag`: a lag
  of `l` rows is the estimator. A lag in time on irregular data is another
  estimator. The standard route is to put the series on a common clock first
  -- a fixed grid with the last value, or refresh times
  (`po.stream.refresh_time`) -- after which a lag in rows is a lag in time.
  For asynchronous ticks the established estimators are Hayashi and Yoshida's
  (2005) for the covariance and its time-shifted form for the lead-lag
  (Hoffmann, Rosenbaum and Yoshida 2013), which avoid the bias previous-tick
  sampling adds (the Epps effect). Recommendation: document the grid route
  beside `lags`; an HRY lead-lag model would be a new model, as `rcov` is, if
  lead-lag on ticks becomes a use.

- `corrchange`'s `span_rows`, `monitor_rows`, `permute_every`, `perm_block`:
  the tests' statistics are defined over row counts (Wied and Galeano; a
  permutation in blocks of rows).

- `rcov`'s `block_rows`, `preavg_rows`, `bandwidth`, `max_bandwidth`,
  `jitter`, `noise_stride`, `iv_stride`: tick-count estimators (BNHLS, CKP);
  the block's edges already come from `group_close`.

- `bocpd`'s `max_run`: a run length is a count of observations.
- `kmeans`' and `hmm`'s `warm_rows`, `marginal`'s `bin_warm_rows`: sample
  sizes for seeding and for quantile edges.

- Mechanical: `gram_block_rows`, `shards`, `bins`, `max_clusters`,
  `max_iter`, `n_perm`, `k`, `pca`, the seeds.

**Decisions for the user.**

1. **Two rules for counting a cadence's rows.** `window_every`'s row cap,
   `max_rows_between_pca` and `coef_every` count every row, rows of weight
   zero included (review C7); `kmeans`' `update_every` and
   `split_merge_every` and `micro`'s `max_rows_between_prunes` count learned
   rows only. A test docstring of task 160 (CB3) says "the library has one
   cadence rule". Narrow that sentence, or move `kmeans` and `micro` to the
   shared rule, which moves their numbers on a stream with rows of weight
   zero.

2. **`micro`'s default cadence.** Every 100 learned rows, kept by task 163;
   DenStream's is `Tp = ceil(h log2(beta_mu / (beta_mu - 1)))` clock units.
   Keep it, or default to `Tp`, which moves every default `micro`'s numbers.

3. **`coef_every` on the clock**, and the spelling of its "last row of a
   chunk" default (above).

4. **`bocpd`'s hazard in clock units** (above).

*Decided 2026-10-06, the user's word ("Follow your reco on all"):* (1) the
sentence narrowed, in `ewcov.rs`'s cadence test: the cadences that refresh a
readout count every row, the ones that act on what new rows taught count
learned rows; (2) `micro` keeps every 100 learned rows; (3) built, task 178
(unset is the old default, `0` every row, `max_rows_between_coefs` the row
cap); (4) built, task 179, as a duration `hazard` applied before the row it
leads into (a row's hazard is the chance of a break after it).

<a id="s18"></a>
## 18. Whole-project review before 1.0 (2026-10-06): everything, with a 1.0 lens

The user: "Deep code review of everything in this project. We are getting
close to version 1.0; find everything we should do before then." Nineteen
read-only reviewers at `8e28c1f`, one slice each (six over `online-core`,
four over the polars layer, one over the three surfaces, two over the Python
package, two over the tests, one over CI and release, two over the
documents, one cross-cutting over names, units, promises and policy), 257
findings. Two lenses: **defect**, as §15 and §16; and **1.0**, anything cheap
to change now and breaking after the promise (a name unlike Polars' or the
library's own, a default with units in it, an undocumented semantic, a file
format, a promise no test pins). Severity as §16; a 1.0 finding takes the
severity of what it would cost after 1.0. Every reproduction was re-run by a
verification agent: all 13 Rust tests confirmed (the load succeeds or the
next row panics, as stated), all 27 Python scripts reproduce. The per-slice
reports, reproductions and the consolidated list are in the session's
scratchpad (`review4/`); this section keeps what must outlive it.

The user then: "Do everything that does not need me." Tasks 183-193 carry
the findings whose fix the code, the docs, an existing rule or a precedent
settles; the decisions below wait for the user.

**High defects** (each in a task): `huber`/`quantile` and `ewridge` predict a
target with no rows at a solve as exactly 0.0 until the next scheduled solve
(CC1, task 186); a windowed `ew_class`'s `coef` is the whole history's
(CE1, 186); `po.eval` scores a relative target against its raw column (YB1,
188). Two clusters: loaders that trust the bytes (twelve findings, 185) and
spec values that size an allocation with no ceiling (CD10, CF2, CE9; 193).

**Decisions for the user**, with the recommendation given:

| # | Decision | Recommendation |
|---|---|---|
| D1 | Hard rule 5 returns at 1.0 with no loader and no fixture; `MIN_SCHEMA_VERSION = 14` is false (layouts since fail in serde before the gate); the released-state test asserts refusal only (CF1 AP10 CI3 DB1 TB3 CC8 DA3) | 1.0.0's schema is the floor; from there every layout change ships a loader and a frozen fixture per `ModelState` variant (embedded bytes), checked three ways (loads, continues to the bit, re-saves byte-identically); raise `MIN_SCHEMA_VERSION` to the shipped schema; build the harness now with HEAD's state; freeze the variant names; delete the dead `#[serde(default)]` repairs |
| D2 | No deprecation policy; a renamed name is refused outright (AP9 CI4 DA16) | warn-and-forward with a `DeprecationWarning` subclass, removed at the next major; `RENAMED` becomes the forwarding table at 1.0 |
| D3 | What a numbers-moving fix is after 1.0 (DA2) | a fix whose old numbers were wrong against the stated definition is a minor, declared with `compare_release`'s difference; a changed default or semantic is a major |
| D4 | The Polars floor 1.34.0 has not run since the window operators; the canary tests the ceiling (AP7 CI2 AP11) | a blocking floor leg in `release.yml` and a monthly canary run; after 1.0, raising the floor is a minor |
| D5 | No "unstable" marker (AP12 DA4 YB11) | Polars-style label and opt-in warning on the windows state format, the formula tree's written form, `fit_predict_arrow`, `po.sim`, `po.corr`; promise the rest |
| D6 | Pins for the stable surface | *not a decision: task 190* |
| D7 | The stream-state tidy (S6) | *not a decision: task 194* |
| D8 | Task 116 leaves readiness defaults provisional (DB7) | build the parts that move no default; declare the current floors final |
| D9 | New users get polars 2.0.0; the goldens and the lock are on 1.44.2 (CI18) | move the dev pin to 2.0.x once the canary has passed on it. *Done 2026-10-08*: the canary passed on 2.0.0 (run 37718689527); the lock, `BUILT_AGAINST` and VALIDATION.md are on 2.0.0, the goldens unchanged |
| D10 | The changed-lines mutation job cannot finish a normal push (CI1) | shard it as the weekly job is |
| D11 | `rust-version = 1.95` is compiled by no job (CI11) | a weekly `cargo check --locked` on 1.95 |
| D12 | SECURITY.md's support line after 1.0 (CI15) | the latest minor receives fixes |
| D13 | 21 of 32 `docs/` files are records (DB14) | `docs/records/` for the uncited ones; PHRASING and README-ITERATIONS filed as records |
| D14 | No payload hash; a flipped bit loads (PA14) | no hash at 1.0; the limit stated in `ModelBank.load`'s docstring |
| N1 | `po.spec.ewridge` vs `type = "ew_ridge"` vs the `ewridge:` message prefix (AP4 CF5) | tag `"ewridge"`; `huber:`/`quantile:` prefixes |
| N2 | `chunk_rows` vs Polars' `chunk_size` (AP5) | `chunk_size` on all three surfaces |
| N3 | `po.eval`'s `min_obs`, `by=` (a bare string iterates characters) (AP6 YB16) | `min_samples`, `group` |
| N4 | `rolling_metrics` is tumbling (Polars' `group_by_dynamic`) (YB5) | `window_metrics(every=)` |
| N5-N9 | `corr.shift`; `po.gram.lasso_path(lambdas=)`; `dist2`; `bank.marginal()`'s `t`; `rows_seen()` (YB10 YB20 TA7 CD13 PA11) | `absorption_shift`; `penalties=`; `dist_second`; `t_stat`; `rows_fed()` |
| N10 | `lagcorr` at the surface, `lag_corr` in the state (CB3) | `lag_corr` everywhere |
| N11 | `rls.ridge` is `ewridge`'s `ridge_scale="sum"`, not its `ridge` (CA7) | rename `delta` |
| N12, N13 | `conformal=<level>`; `increment` vs `diff` (CA9 PD8) | keep both |
| N14 | `update_every`, `split_merge_every`, `permute_every` count rows (AP15) | `*_every_rows` |
| N15, N16 | `--resume` vs `load_state`; holt's `level_half_life` beside `half_life` (SF7 AP8 TA6) | `--load-state`; drop `level_half_life` |
| N17 | the models' `window_size` keeps a row exactly `W` old; the operators and Polars drop it (CB1 PD2) | a `closed` parameter on the windowed models, Polars' `"right"` |
| N18, N19 | clock range as Float64 seconds in three frames; `rows_fed` Int64 in one frame (SF1 AP13 PA3) | the clock's own dtype; UInt64 everywhere |
| N20-N22 | rcov's closed-group block half-prefixed; the null group unreadable alone and text key order; a key column changing dtype splits groups silently (PA13 PA10 PC1 PA7) | prefix all; `group` takes `str \| None` lists and integer keys sort as numbers; the dtype recorded in the state and a change refused |
| N23-N26 | `sim`'s status; five constants outside `__all__` and the `embargo` columns; `clock`/`group`/`lam`; `--skip-learned` (YB11 YB9 AP19 PC4 PC5 AP22) | unstable label; add to `__all__` and name the columns stable; keep (task 144); add it |
| U1 | `sgd.huber_delta` 1.0, `sgd.eps` and `pa.eps` 0.1 in target units (AP3 CC4 PC2) | σ-relative, as `huber`'s |
| U2 | `sgd.standardize` False; `pa` has none (CC6) | True for sgd; add to pa, default True |
| U3 | `huber_delta` 1.5 unmeasured; the textbook constant is 1.345 (TA8) | 1.345 |
| U4, U5 | `bocpd.prior_scale` is the identity in target² units; `prior_nu = d + 2` under every emission (CE5 CE2) | data-relative from the first rows; per-emission ν |
| U6 | `rls.ridge`, `learning_rate`, `clip_gradient`, `pa.c`, ftrl's constants | keep; state the units; pinned by task 190 |
| U7 | a row cap of 0 refused for `max_rows_between_coefs`, "every row" for four others; `pca = 0`, `stats = []` accepted (PC3 CB4 YA2 PC9) | refuse 0 and the empties everywhere |
| S1 | `permute_every` redraws every `n + 1` reports (CD11) | the code matches the docs |
| S2 | `coef` is skipped when a chunk's last row of a group is skipped (PB2) | the last accepted row |
| S3 | under an embargo `predict` learns no matured label (PB3 TB6) | document and pin it |
| S4-S6 | `sgd` logistic labels outside {0, 1}; a Poisson `hit_rate` reads 1.0; a prediction of exactly 0 a hit in the bank and a miss in `po.eval` (CC9 CC5 YB8) | ftrl's clamp and `strict_binary`; null; exclude `pred == 0` on both sides |
| S9b | a target's own first solve when its weight first reaches its `min_weight` (CC1's second half) | build it |

*Settled without the user, as following a rule or the docs (in tasks
186-193):* S7 (`shrink`'s identity target, the paper's), S8 (CE1, the
docstring's), S9a (NaN for an unsolved target, CA5's rule), S10 (a
parameter a mode ignores is refused, YA8's rule), S11 (every way in checks
a spec the same way), S12 (`refresh_time`'s value read by the `usable`
rule), S13 (`pins` markers), S14 (`increment` of a `Time` gives seconds, as
its docstring says), S15 and S16 (documentation of the current behaviour).

*Decided 2026-10-06, the user's word ("Your reco all"): every recommendation
in the table above, and task 183's cost accepted.* Built as tasks 194-199
once tasks 183-193 merge, since they touch the same files. D9 waited for the
weekly canary to pass on polars 2.0.0: its run of 2026-10-05 failed on the
unwrapped `ValueError`, fixed in `a7a8f3c`, and it passed on 2026-10-08
(below).

**Marked for the user's review (2026-10-07).** Two decisions taken after §18,
on the user's word, to be looked at again before 1.0: the unit of an
insensitivity band (`pa`'s `eps`, `sgd`'s under `epsilon_insensitive`) -- the
target's own EW spread rather than the residual's (task 202) -- and its
default, 0.01 (task 203). **Reviewed 2026-10-07 (the user: "Yes" to both
kept, two wordings changed).** A third rule was measured beside them, the
residual's std capped by the target's (`eps·min(σ_resid, σ_y)`), which
removes task 195's trap and shrinks with the fit: no rule wins across R²
(`$S/eps/bands.py`; at `c = 1`, excess OOS MSE over the noise variance,
0.01 of σ_y / 0.1 of σ_y / 0.3 of the cap: 0.18 / a lottery / 0.44 at R²
0.99998, 1.16 / 0.59 / 0.79 at 0.978, 0.81 / 0.68 / 0.59 at 0.80). The
lottery (review 5, G2): at 0.1 the band is 20 noise stds wide there, so
the fit freezes wherever it first lands inside it, and over 20 seeds
through the bank the excess runs from 0.58 to 278 (median 35). The replica
drew 54 for its seeds; task 203's table holds the bank's 5-seed median,
2.86. At 0.01 the same 20 seeds give 0.16 to 0.20. The tension is `pa`'s:
at `c = 1` the tube is its only damping against noise. 0.01 stays by regret
(its worst case 1.2× the noise variance, 0.1's between 0.6× and 280×
wherever its fit freezes, on the near-deterministic target a user has
before differencing it); for `sgd` 0.01 loses nowhere. Changed: "inside a good fit's errors" became "below a
good fit's errors on most targets" (at R² 0.9998 the band is two-thirds of
the noise std), and `pa`'s docstring and the README carry the trade-off
(2.2× / 1.6× / 1.3× the noise at the default / `eps=0.1` / `c=0.1`, at R²
0.978, `test_pa.py::test_the_tube_is_pas_only_damping_on_a_target_predicted_less_well`).
*Review 5 (F2, E5, A3, G4):* the four places had stated those figures for
"R² 0.97 to 0.998" with `c=0.1` as the better lever, from that one point.
At R² 0.9975 the bank gives 2.00 / 1.18 / 1.70 and at 0.990 2.12 / 1.42 /
1.46 (median of 3 seeds), so `eps=0.1` is the smaller loss above about R²
0.99 and `c=0.1` damps best where the noise is large. They now say so, at
R² 0.98 and without the range.

**Decided 2026-10-07 (the user: "Your reco all except 116").** On the open
items after the round:
- **The next release** (0.14.0) is rehearsed before it is tagged: this
  commit benchmarked against the cached 0.13.0 wheel, then the release
  workflow dispatched with publishing off, then the release, each dispatch
  on the user's word. Five workflows changed (`release.yml`, `ci.yml`,
  `polars-canary.yml`, `mutants.yml`, and `msrv.yml`, new).

- **D9**: the canary ran on demand (run 37715703707, on `9a67dd8`) rather
  than on Monday, and passed on 2.0.0 final on 2026-10-08 (run 37718689527,
  on `9b23ae6`: 4,920 passed, 1 failed on the floor-leg workflow test, which
  reads the pin's form the canary rewrites, since marked `pins`). The dev
  pin and the lock moved to polars 2.0.0 the same day (worker
  `d9-polars-2`): `uv.lock` (polars and polars-runtime-32 alone),
  `BUILT_AGAINST`, `docs/VALIDATION.md` regenerated on it, the README's
  matrix and RUNNER's pin line; the goldens were checked on it. The Rust
  crate stays at 0.55.2, crates.io's newest that day, and the range
  `>=1.34.0,<3` is unchanged. The record is in RELEASE-READINESS, beside
  the 2.0.0rc1 measurement.

- **`kalman`'s `share_p`**: fixed, task 204.
- **YB3 closed, not changed.** `embargo` over this package's own plan form
  reads its input twice, and so runs that fit twice: the warning is true.
  A `pl.scan_parquet` input does not warn.

- **Tasks 117, 86, 118 and 127 stay as they are**: 117 waits on 3.15.0 and
  its wheels; 86 and 118 are parked by the user; 127 waits on a
  measurement nothing asks for.

- **Task 116**: the user wants its changes, defaults included, made before
  1.0; its scope is decided from a summary of it (2026-10-07).

<a id="s19"></a>
## 19. Review round 5 (2026-10-08): tasks 183-204, code and theory

Round 4 read the code at `8e28c1f`; its decisions produced 33 commits
(tasks 183-204, `8e28c1f..9b23ae6`) that no one but their authors had read.
Seven read-only reviewers read them at `9b23ae6`, each with a code lens and
a theory lens (A models, B clocks and windows, C bank/stream/spec, D state,
E Python and tests, F docs/CI/policy, G the theory of the decisions alone);
briefs, findings and probes in the session scratchpad `review5/`
(`CONSOLIDATED.md` is the table). Every finding is a probe output or a code
path; the coordinator re-ran the three high ones through the bank before
the fixes were briefed.

What held: hard rules 2, 3, 8 and 9 over 17 specs and 159 fields,
bit-identical at 1/7/600 chunks with unusable values on the chunk
boundaries; every re-pinned golden reproduced from its docstring replica;
every rename refused with a test; task 189's oracles fail on a 1e-7 nudge;
the share_p filterpy oracle fails on the old rule; the release, CI, canary,
mutants and MSRV workflows; the 1.0 policy's tables against each other.

**Fixed in task 205** (each test failing on `9b23ae6` first):

| id | sev | finding | fix |
|---|---|---|---|
| C1 | high | under an embargo the "cannot be met" notices paired the held-inclusive `settled` with a weight that had not seen the held rows: "tops out near −0.0462, withheld for good", then 374 of 400 rows predicted | both notices read the learned rows' settled fraction (trigger, ceiling, projection); the gate and the row field keep the held-inclusive one |
| C2 | high | a damaged `emit_metrics` state loaded and the next row panicked (`ewcov.rs:838`) | `SlotMetrics::has_shape()` at load |
| C3 | med | `--skip-learned` dropped a NaN-clock row in silence | `Some(Less \| Equal)`; the bank refuses the row |
| B2 | med | `embargo` wrapped at an integer clock's top: the learn copy first in the stream | widen before the add, strict cast back: polars raises |
| E1 | med | `po.eval(spec=)` gave a Poisson fit `hit_rate` 1.0 | null, as the bank |
| E2, C5, E7 | med | the snapshot pinned refused `closed` words, pre-144 clock names, and unstable names unlabelled | a two-word `ModelClosed`, `taken_words` requires acceptance, `restart_after_step_back` rendered, `# unstable` labels |
| B1, D3 | med | no fixture held an integer clock, `sgd`'s per-loss state, `ewridge`'s kept systems or `bocpd`'s warm rows | cases added, a decoding test holds each form non-empty; existing fixtures byte for byte |
| D2 | med | the from-1.0 half of the harness was unbuilt | `previous![..]`, `v<N>/` includes, the convert-to-current check, coverage proven on a temp dir |
| D1 | med | the promise's "re-save byte for byte" cannot hold for a loaded previous schema | stated as conversion to the current schema's bytes (README, RR, rule 5) |
| F1, F2, G2, A2, A5, A6, B4, E6, F5, F6, F8, D5, D6, C6, C7, A4, D7, E3, E4, F4b | low-med | wording, counts, a NaN cfg through the Rust API, `pow(lam, 1)`, `to_json`'s label, the property's draw, the floor test's mark | as the CHANGELOG says |

**Decisions, the user's (open 2026-10-08; decided the same day, below the table):**

| id | finding | options | reco |
|---|---|---|---|
| G1 (high) | `standardize=True` (sgd, pa since U2; kalman always) holds coefficients in standardized coordinates and reads rows and coefficients off the moving EW moments: at half-life 10 and R² 0.978 `sgd` is 190× worse than unstandardized and its `coef` reads [0.83, 3.03] for [0, 2]; at half-life 50 and R² 0.99998 pa/sgd/kalman are 33/237/213× worse; `ewridge` re-solves and is immune; every U2/203 measurement was at no decay | (a) keep the fit in the caller's units and use the scaler for the step only, mapping β (and kalman's P) through the affine change when the moments move; (b) revert U2's default, document kalman | (a) before 1.0 |
| G3 (med) | the eps regret argument compared values inside the σ_y rule only, and "the band must sit below a good fit's errors" is wrong for PA at c = 1: a band of 0.5 σ_y (3.3 noise stds) gives excess 0.07 at R² 0.978 against 1.16 at 0.01 -- the tail rows pull the fit to within band − 3σ_noise; the best band is about three noise stds, which no σ_y rule gives across R², and under no decay a residual σ keeps the start-up | a task: the band in noise units with a start-up cap, swept under decay, levels, c and mode, `eps·min(σ_resid, σ_y)` at 0.5 the candidate (worst case 0.61 vs 1.16); or keep the shipped rule with honest docs | the task, before 1.0 |
| G5 (med) | `c` is in the target's units, so "the tube is pa's only damping" holds at spread 2 only | `c` as a multiple of σ_y as `eps` is; or document spread 2 | σ_y-relative |
| A1 (med) | share_p's mean noise averages over all m targets, zeros included, so a target with no residual variance yet counts as noise 0; task 204's "number of targets" claim holds for targets on the same rows | (a) average over targets with σ² > 0 in `step` and `pred_var`, as `shared_first_noise` does; (b) keep, narrow the docstring | (a) |
| B3 (med) | an `Int128` column is "integer" to the docs and a double to the code (`po.increment` loses steps near 2^63) | refuse as a clock by name; read it in `po.increment` | both |
| C4 (med) | `summary().settled_frac` leaves the held rows' clock out; the row field adds it | (a) summary held-inclusive, `weight_sum_settled` on the learned fraction; (b) document | (a) |
| D4 (med) | the policy promises forwarding for every renamed stable name; the mechanism forwards parameters and keys | (i) narrow the promise, a forwarding stub when a function or word is first renamed; (ii) build now | (i) |
| F3 (med) | the "cannot be met" notice also fires falsely on a rising row rate (the README's kernel example: 381 of 400 predicted, weight 103 against a projected 7.7) | conditional wording, fire only after the weight stayed below the floor a further half-life past 95% settled, pin the example | as stated |
| F7, F9 (low) | U7's `stats=[]` exception recorded only in a task note; PERFORMANCE §11 cells rewritten where the rule was a pointer | annotate U7; accept the cells | as stated |
| D8 (note) | `test_released_state.py` asserts `refused == {}` per release: after 1.0.0 a new kind fails 1.0.0's leg | filter per release on release day | -- |
| 116-K | the "never met" notice for `kalman`'s per-row gate: `P` settles on `coef_half_life`'s clock and the rows spread 1.02-1.11 around the average 1.07 at `coef_half_life` 50, so a notice tested at ±5% cannot hold (task 116) | judge the best row (`√(1 + 1/(P⁻¹)₀₀/R)` with an intercept), the average row, or stay off; evidence WARMUP §7.11 | stay off until a 1.x user asks |

**Decided 2026-10-08 (the user: "Your reco all").** Every row above as
recommended:
- **G1** (option a): the fit is held in the caller's units and the scaler
  shapes the step only -- task 206. *Revised the same day* when the first
  build broke short histories: option (e2), today's design for a warm-up
  of 22 rows, then (a); measured on both regimes before the user's "Go"
  (task 206's entry).

- **G3 and G5**: a measured task -- the band in noise units with a start-up
  cap, swept under decay, levels, `c` and `mode`, `eps·min(σ_resid, σ_y)`
  at 0.5 the candidate; `c` a multiple of the target's spread -- task 207,
  built on task 206's code, since the scaler's wobble dominated every
  decayed measurement before it.

- **A1** (option a), **B3** (both), **C4** (option a), **D4** (option i),
  **F3** (as stated) -- task 208 (A1 inside task 206's branch, kalman).

- **F7**: U7's `stats = []` exception is annotated in §18's U7 row's
  reading here: U7 holds except `stats = []`, a documented accumulate-only
  mode the user kept (task 196's note).

- **F9**: PERFORMANCE §11's rewritten cells are accepted.
- **116-K**: kalman's "never met" notice stays off.
- **Ubuntu 26.04** (`ubuntu-latest` moves on 2026-10-19): let it move; the
  CLI's glibc floor is manylinux2014's, not the runner's.

- **D8** is a release-day step: filter `test_released_state.py`'s refused
  kinds per release once 1.0.0 is in `RELEASES`.

The coordinator's own correction, recorded here: the eps review of
2026-10-07 said "nothing else in the eps chain needs reopening"; G3's
measurements and the 0.5-band probe (`review5/verify/bands_h.py`) show the
theory the docs state is wrong for PA at `c = 1`, and the chain is open.

<a id="s20"></a>
## 20. Review round 6 (2026-10-09): tasks 221-231, code and concepts

The user, 2026-10-08: "After implementing everything do a code and concepts
review and fix any obvious bugs." Seven read-only reviewers over
`f54d1e5..integ23` (A: 221 (a)-(c); B: 221 (d)-(h); C: `check` and `audit`;
D: tasks 225 and 231; E: tasks 226/227/229; F: cross-cutting; G: concepts
and the two pages), every finding shown by a probe; task 222's recipes had
found F1-F6 first. Findings and probes: the session scratchpad,
`review6/FINDINGS.md`.

**Fixed (one right answer), each with a test that failed on the base:**
B-3 (the windowed Ljung-Box is Box-Pierce at Kish's size; Ljung-Box's
factor passed 15.8% of clean streams at half-life 10), B-5 (the tails keep
central moments: kurtosis read -129,065 after a far first residual), A-6
(under `embargo` a row keeps the inflation it was scored with, schema 57),
F4, F6, F-9, F5 and G-6/G-10 (docs), F-12; C-2 (an exact trend's
Dickey-Fuller is ±inf, not null), C-4 (`vif` inf only for the constant
column), C-7, C-8, D-1 (`gram_threads` is configuration: a save resumes
under another count), D-3, E-1 (NaN, not 0, standard errors where the
system is not positive definite), E-2, E-4, E-5 (no knot for a column set
aside), E-6, E-7, F-6 (first half), F-10, F-11, G-9, G-11, G-12.

**Decisions put to the user (14, merged across reviewers), with the
coordinator's recommendation:** (1) warm-up residuals dominate the
diagnostics on `rls`/`kalman` at a level (F1, F-1, G-6): fold from
readiness; (2) the memory beside `window_size`/`kalman` (A-2, A-3, B-4, G-4,
F-5): derive it from the fit's; (3) look-ahead targets (B-1, A-4, F3, B-2,
G-5, F-4, A-8): one horizon, Newey-West forms under it; (4) the windowed
nulls and `cusum_sq`'s Gaussian assumption (A-5, G-3, F2, A-7): standardize
by the measured fourth moment, rescale; (5) `break_wald`'s shared Gram
(A-1): the exact sandwich; (6) quantile fits (G-1, F-2): coverage and a
CUSUM of `1{y<pred}-q`; (7) `frozen` on persistent-discrete columns (C-1):
the zero-change atom test; (8) `group_close` (C-3, F-3): refuse on `audit`;
(9) `collinear` uncentred (G-2): the centred index for the centring models;
(10) `leakage` on a long random walk (G-7): a warning; (11)
`level_over_spread` on a decaying `rls` (G-8): no decay only, scaled; (12)
float32's range (E-3): refuse; (13) the audit's clock (C-5, F-6): uncapped
steps, Durations; (14) Newey-West lag cost, embargo notes, feature health's
refusal, `vif`'s docstring (F-7, B-6, B-7, F-8, C-6).

**Decided 2026-10-09** (the user: "Your suggestion for all the fixes"):
all fourteen as recommended. Built as task 232 (the diagnostics: 1-6 and
14's diagnostic parts) and task 233 (`check`, `audit` and the Gram: 7-13
and `vif`'s docstring).

<a id="task-232"></a>
- [x] 232. **Review 6's diagnostic decisions** (§20, 1-6 and 14): fold from
      readiness; the memory from the fit's (window, `coef_half_life`); one
      horizon with Newey-West forms under it; nulls standardized
      (`cusum_sq` by the measured fourth moment); `break_wald`'s exact
      sandwich; quantile forms; the Newey-West lag cap, the embargo notes,
      feature health's refusal.
      *Done 2026-10-09* (schema 60). (1) `min_weight` already gated every
      fold, so a diagnostic also waits for the stream to be 95% settled on
      the fit's memory, and `sgd`/`pa` for their scaler: `rls` at a level,
      Jarque-Bera 100% -> 3.2%, run-once RESET beside a decaying fit 25.8% ->
      4.8%; what remains is the prior's real bias, which the calibration
      rightly flags. (2) the memory from the fit's: a window's Kish half-life
      (`ln 2 / ln((W+1)/(W-1))` rows), `kalman`'s `coef_half_life`;
      `se_coef_hc0` beside a window 0.0495 against a true 0.0498 (3.1x too
      small before). (3) `horizon_rows`; Newey-West Wald forms of the
      calibration, Breusch-Pagan and RESET, the long-run variance in the
      CUSUMs and `break_wald`: 4-5% where 10-63% before. (4) `cusum_sq` over
      the measured fourth moment, windowed CUSUMs over their derived null
      (spread 1.04-1.10). (5) `break_wald`'s exact sandwich: a feature going
      quiet 85% -> 0%. (6) quantile fits: `calibration_coverage` and a CUSUM
      over indicators, 100% -> 0% at q 0.9. (14) Newey-West lags from
      `embargo` capped at 64.

<a id="task-233"></a>
- [x] 233. **Review 6's data-check decisions** (§20, 7-13, C-6): `frozen`'s
      zero-change atom; `group_close` refused on `audit`; the centred
      `collinear`; `leakage` a warning; `level_over_spread` for `rls`
      without decay; float32 refused out of range; the audit's clock in
      uncapped Durations; `vif`'s docstring.
      *Done 2026-10-09* (schema 59). `frozen` a zero-change atom test (the
      six persistent-discrete clean columns silent, every planted case found;
      25 stuck rows no longer); `audit` refuses `group_close`; `collinear`
      centred for the centring models (G-2's pair 5,061 -> 1.004); `leakage`
      a warning; `rls`'s level limit `0.9*sqrt(W/delta_left)` from the prior
      it still holds (the measured R² loss falls on one curve in that ratio;
      wider than the brief's no-decay rule, which missed a short stream under
      a long half-life); float32 refused out of range; the audit's clock
      uncapped, in Durations; `vif`'s docstring.

<a id="s21"></a>
## 21. Review round 7 (2026-10-11): everything since round 6, code and concepts

The user, 2026-10-11: "Review the recent changes including Int64 and the
bocpd fold. Are these the right changes, is there something smarter to be
done? Review the code and test case coverage." then "review other recent
changes since the last review". The coordinator reviewed the fold, Int64
and the formula list itself (the fold's defect at `prune_below = 0`, task
243b, was found there and built on the user's "We will make this change";
Int64's declared-vs-realized gap closed by a registry-wide test, `8aa55f6`);
then five read-only reviewers over `926ee17..8aa55f6` (213 commits: A the
diagnostics after round 6, B Grams/LARS/the sharded marginal, C `po.gbdt`,
D infrastructure, E concepts with a 1.0 lens), every finding shown by a
probe. Findings and probes: the session scratchpad, `review7/FINDINGS-*.md`
and `review7/<slice>/`.

**Found:** no high. A: A-1 (the CUSUM's null share beside a `kalman` with a
`coef_half_life` list read the shortest memory; a level is absorbed at the
intercept's: spread 1.40), A-2 (a quantile fit's `break_wald` under a
horizon scaled by the indicators' long-run variance, not the residuals':
10.8% past 7.815 on quiet streams against 5.9%), A-3 (a no-decay window's
share read as an exponential at its Kish half-life: spread 1.19). B: none
(within fits to statsmodels at 1e-14 under weights, decay, singleton and
zero-weight parts; LARS ties at the KKT conditions to 2e-15 on 800 planted
knots where scikit-learn misses by 0.30; the sharded marginal to the bit at
every shard count and chunking). C: C-1 (a `ref` on a `pl.Enum` of other
categories accepted, codes meaning other levels), C-2 (the null-handling
tests put nulls on a feature the target ignores: unexercised), C-3 (the
XGBoost floor is 2.1, `XGProxyDMatrixSetDataColumnar` absent from 2.0.3),
C-4 (bin cuts unweighted under `weight=` where XGBoost's sketch is
weighted: 2 of 15 cuts above 1.0 against 14). D: D-1 (the message scanner
cut a file at its *first* `#[cfg(test)] mod`: `spec.rs`'s 621 shipped
refusals never read, a planted citation passed), D-2 (TESTING's 226 was
225), D-3 (the shard size 30 in the workflow, 40 in every test). E: E-1
(two settled clocks -- the notices and `settled_frac` on the model's own
decay, the diagnostics on the longest memory -- stated as one in
DIAGNOSTICS.md:226 and a `stream.rs` comment), E-2 (two guides list three
`withheld_reason` values), E-3 (fourteen `po.gbdt` refusals untested).

**Fixed, one right answer:** D-1..D-3 (`9b20cc5`); A-1, A-2, E-2
(`review7-fix1`, schema 61; merged with fix3 2026-10-11); C-1 (a `ref` built here records each feature's Enum
categories and refuses a mismatch by name), C-2, C-3, E-3 and `in_memory=1`
(`review7-fix2`; stopped once by a permission prompt, resumed on the user's
"Resume"); merged 2026-10-11, full tier 6302 passed).

**Decided 2026-10-11** (the user: "Your reco all", with "we are not
worried about doing more work now and want the best outcomes"; the fixes
delegated at the user's word to Opus at high effort): A-3 (a) the window's
own form of the null, derived in closed form and held to spread 1.0 (what a
window beside a decay reads is measured and stated); E-1 (a) keep both
clocks, state them, test holt's two rows (`review7-fix3`); done and merged 2026-10-11: `cusum_null_window`, a closed form held to quadrature at 9.7e-5, the window form beside a decay too, spread 1.00-1.02; holt's notice at row 268 and first diagnostic at 866); C-4 (b) weights
carried into the sample and weighted quantiles, so the cuts are XGBoost's,
plus the docstring sentence for LightGBM's unweighted binning
(`review7-fix5`, after fix2); done and merged 2026-10-11: probe (a) 2 -> 14 of 15 cuts above 1.0 (XGBoost 14), heaviest bin 35.5% -> 6.26% (XGBoost 6.47%), unweighted cuts bit-identical, scoped mutants 122 caught and the last survivor closed by hand); B-D1 the behaviour stays (the pairwise
convention), one sentence in GRAM.md; the README's *Versions* rewritten for
1.x beside the user's 1.0 note (`review7-fix4`). Not changed: the Int64
choice (Polars' data integer; its index type UInt32 is for positions), the
pairwise convention, the two settled rules.

<a id="s22"></a>
## 22. Review round 8 (2026-10-16): everything since round 7, code and concepts

The user, 2026-10-16: "Review code changes since the last review". Six
read-only reviewers over `93d0ef4..bb69654` (229 commits, 453 files; tasks
244-268 and the releases 1.0.0, 1.0.1, 1.1.0), one slice each in a worktree
slot: A the Gram numerics (256, 256 (b), 257, 258's oracle, 265), B the
bank's plumbing (250, 253, 255 (a), 257 (b)/(c), 265), C the models'
rewrites and diagnostics (250, review 7 fix 5, 252 core, `breaks.rs`), D
the Python package, bindings and API (244, 249, 252 python, 253, 256), E
the infrastructure and build (245, 248, 251, 254, 255 (b), 258-268, the
releases, the schema 62 fixtures), F the concepts and documents with a 1.x
lens. Every finding shown by a probe; the coordinator reproduced D-1, D-3
and A-1 on the main checkout, and audited the locks: no locked crate (467)
or PyPI package (88) is yanked, and task 268's manifest entry leaves the
x86 crate-feature sets identical (1,282 Linux, 1,369 Windows pairs).
Findings and probes: the session scratchpad, `review8/FINDINGS-*.md` and
`review8/<slice>/`.

**Found: 1 high, 11 medium, 24 low.**
- High. D-1: `closed_groups="*.csv"` crashes every Python surface after the
  whole fit (each closed row has a `List` column; Polars' CSV writer refuses
  it), and the plan form then writes no `save_state`; fca6699 flattened the
  runner's CSV only (`csv_flat`).

- Medium. A-1: under `target_gaps="pairwise"` a gappy target's block mean is
  `(Σ_all u·a − M)/U_j`, a subtraction whose rounding is the whole block's
  amplified by `U_all/U_j` (300 to 506,290 ulps for a one-row light target
  in a 256-row block; the per-row path exact), so the CHANGELOG's "within
  the rounding bound of a sum in any order" fails there; `own_rows`
  unaffected. D-2: a TOML float `inf` beside a deprecated model key refuses
  the config ("invalid type: null"; the forwarding round-trips through
  JSON). D-3: a column named twice in `features=` or a job is not refused;
  `solve_subsets` reports coef 0 for it beside the fit's own `r2`. D-4:
  `test_every_word_valued_parameter_is_probed` collects bare-`str`
  annotations, none since task 244's `Literal`s, so it cannot fail. E-1:
  `slot.sh take` checks out a branch another worktree holds (`checkout -B`
  skips git's guard). E-2: `budgets.py`'s tighten-only history is bypassed
  by deleting a key and re-adding it higher, or renaming it. E-3:
  `clear_caches.sh --worktrees` run from a slot treats every other slot and
  the main checkout as other worktrees (185 GB of warm builds). E-4: the RSS
  budgets' 30% two-sided tolerance admits up to +86% over the last tightening
  on every push (`rss.streaming_bounded.linux` passes at 284.6 MiB against a
  release run of 154.2). F-1: CHANGELOG 1.1.0/1.0.1 and PERFORMANCE §43 say
  `gram_threads=` bounds the Windows commit; a solve spreads over the whole
  pool (8 of 8 threads allocate in a k=1,500 factorization with
  `gram_threads=1`), so only `threads=` bounds it. F-2: RELEASE-READINESS
  says every release's state files are refused and `RELEASES` stops at
  1.0.0, against its own 1.x rule and the test. F-3: RUNNER.md states the
  py-polars floor as 1.34.0.

- Low. A-2 (a blocked merge's `W'²` overflows from weights ~1e154: NaN for
  good); A-3 (the trisolve test's ill leg is full rank below n = 7, κ 2.1e9
  at n = 4 where the PLAN says 4e12); A-4 (three spots still say the factor
  is dropped at a run's end); A-5 (pre-existing: one constant column makes
  every `se`/`t` NaN, `inverse_factor` factoring the dropped slot); B-1 (a
  formula-target spec's state bytes depend on chunking: the resolver's held
  rows written as one IPC batch per chunk fed); B-2 (eight `subprocess(...,
  text=True)` without `encoding=`); C-1 (`cusum_null_window`'s decayed form
  loses ~1.5ε/(u·v) where the window is far shorter than both memories,
  2.3e-5 relative at the worst grid point); D-5 (a non-integer TOML
  `threads` refused by serde, not the shared message); D-6 (numpy ints
  refused for `threads=` and a hand-built job's `target`/`features`); D-7 (a
  parse error after a forwarded key quotes the rewritten line); E-5 (every CI
  leg writes `rss-<platform>.json`; `apply` must be given the floor's); E-6
  (a sign-off of any past version silences the push warning; `check_file`
  ignores a sign-off's key); E-7 (a corrupted v61 map key whose field
  defaults to the frozen value passes the previous-schema check: state
  structs do not `deny_unknown_fields`); E-8 (the gate accepts an empty
  `serial` selection in the extended tier); E-9 (the gate's cargo steps lack
  `--locked`); F-4 (TESTING says released states are refused); F-5 (the
  1.1.0 `corrchange` figures transposed); F-6 (PERFORMANCE §41's Windows and
  24 MiB sentences after §43); F-7 (a four-thread instruction case claimed
  twice); F-8 (CHANGELOG 1.0.0's 1.34.0 remnants); F-9 (task 258's Done
  paragraph displaced under the 1.1.0 record); F-10 (README:249 omits
  `se_coef`); F-11 (STATE-WORKFLOW names `po.run`); F-12 (the API policy has
  no row for a change that moves last bits within rounding, which 1.1.0's
  256 (b), 257 and the kernel did).

**Checked and found right** (the reviewers' lists): every task 250 rewrite
bit-identical to `93d0ef4` (128 digests over bocpd, kalman, sgd, kmeans,
micro, EwCov); the fold within the exact oracle on 1,728 edge cases and the
same bits at every thread count; a blocked fit with held targets chunk
invariant through 14 save cuts and a real 1.0.1 blocked state continuing
within the fold's bound; the typed copy at every integer width and float32
edge with null and NaN apart; threads identical at every pool size and the
Sink's error paths; every rename forwarded on four Python surfaces and the
CLI; `jobs=` refused by name and bit-equal to the per-job recipe; the
budgets' 1.1.0 apply; the kernel entry; the 1.1.0 Fixed entry against the
probe record; 400 doc tests and the Sphinx build.

**Decided** (the user, 2026-10-16: "Yes and keep the remaining list in the
project", to fixing every finding, the decisions as recommended): A-1
option 1, the target's block mean accumulated over its counted rows (no
subtraction; blocked-pairwise bits move; cost measured with
`examples/gram_targets_bench.rs`, raised if large); D-3 a refusal by name;
E-4 an RSS budget tightened whenever a run is under it by more than a
quarter of the tolerance; A-5 the standard errors from the factor of the
kept columns (NaN becomes numbers where the doc promises them); B-1 the
held rows rechunked before the IPC write (fixtures regenerated where they
move); E-7 the limit recorded beside the test (no `deny_unknown_fields`:
a field removed since a previous schema would refuse that schema's files);
F-12 a policy row for last-bits moves within rounding, declared with their
size; C-1 the bound documented. Built as task 269 in four batches.

<a id="task-269"></a>
- [x] 269. **Review round 8's fixes** -- the user, 2026-10-16 (above).
      Batches, one worker each in a slot, test first (each new test fails
      on `bb69654` where its finding says): (a) Python, bindings and CLI:
      D-1, D-2, D-3, D-4, D-5, D-6, D-7; (b) numerics: A-1, A-2, A-3, A-4
      (code comments), A-5, B-1, C-1; (c) infrastructure: E-1..E-9, B-2;
      (d) documents: F-1..F-12, A-4 (PERFORMANCE). Workers do not edit the
      CHANGELOG's `[Unreleased]` or this plan; the coordinator adds both at
      merge. Then a second look over the merged diff before the push.
      **Remaining, kept here for later work** (not in this batch): a
      `with_windows`/bank fixture freezing `Held` with a NaN target value
      and several held rows per target, and a blocked spec in
      `release_probe.py`'s workload so `test_released_state` loads a
      released blocked file (E); every shape of a schema 61 blocked state
      continuing within the fold's bound, `lam = 0` mid-block among them
      (E); two pools of `n` threads alive when `Threads::new(Some(n))` runs
      before the process pool (B, documented, unmeasured); a window past a
      refusing budget under `drift_action = "reset"` on a sink'd spec,
      driven (B); `gram_threads=n` pieces wandering across the pool over a
      long run (F); the release benchmark's raw rows kept in the repository
      so its CHANGELOG figures can be traced (F); `po.gbdt`'s `threads`
      (D); TOML datetimes through the CLI's forwarding (D); a budget key
      renamed in one commit still enters the history with nothing to
      compare against, so tighten-only does not hold across a rename
      (E-2's second path: a key new to the history could be held to
      `budget == release` or a `[[raised]]` entry; `check_history` over the
      16 committed versions finds no such entry today) (E, the second look).
      *Done 2026-10-16*, four batches merged into `main`, each new test
      failing on `bb69654` where its finding says. (a) `fdb514c`: D-1 the
      closed-groups CSV through the runner's `csv_flat` (`_native.write_frame`);
      D-2/D-7 forwarding over `toml::de::DeTable`; D-3 a duplicate column
      refused by name in Python and Rust; D-4 the Literal collector; D-5/D-6
      `threads` and job fields through one integer reader. (c) `825a6be`:
      E-1..E-9 and B-2 (E-2's delete-then-re-add only; its rename path is
      under Remaining) (slot holder refused, budgets' last-seen history, the
      main checkout from the git common dir, `TIGHTEN_SHARE = {"rss": 0.25}`
      with six RSS budgets tightened, the RSS file only where held, sign-off
      rules, E-7's limit recorded beside the test, `--locked`, an empty
      extended `serial` refused, 29 subprocess sites given `encoding=` and an
      AST check). (d) `57861b2`: F-1..F-12 and A-4 in PERFORMANCE, with tests
      holding the policy page's releases and every guide's floor to the
      declared ones. (b) `447c8d1`: A-1 `G` accumulated over the counted
      rows in the same pass as `S'` (worst cost +4.6% at ten gappy targets,
      k=100; `own_rows` bit-identical over every case of a dump); A-2 a shared
      `between_share` for a `W'²` that is not normal, and A-2b found beside
      it (a subnormal `W'` made `1/W'` infinite and the fold NaN: the rows
      weighted by `uᵢ/W'`, `fold_tests::worst` now counts NaN as a failure);
      A-3 `lu_rows`; A-4 comments; A-5 the standard errors over the columns
      with variance in `ridge_fits` and in Python's `coef_stats` alike; B-1
      a rechunk before the IPC write (no frozen fixture moved); C-1 the bound
      stated and held by the source's test. The second look over
      `bb69654..0b6ced8` found no high or medium defect; of its three lows,
      the CHANGELOG's numpy-integer scope and `clear_caches.sh --worktrees`
      sparing the main checkout from another worktree were fixed, and the
      budget-rename path went to Remaining.
      The push (`24114c5`, 2026-10-17) failed two checks, fixed after it:
      Windows' `exp` put C-1's new bound test at 1.7 · 2⁻⁵²/v (Apple and
      glibc 0.9), so the test allows 4 and the doc states each; and the
      changed-lines mutants left five survivors in (b)'s code -- the
      subnormal shares path in `EwCov::flush_riding` and the fold (`/` read
      as `*` or `%`; the tests read the moments only after rows of weight 1
      washed them out, now also over the subnormal rows alone, held to the
      rows at 1e-9 since no path meets the oracle's bound at 44 bits), and
      the fused pass's skip at a four's third row (target 2's second row
      moved to 246). Each mutant applied by hand fails its test.
      *Released as 1.2.0 on 2026-10-19* (the user: "What is the next
      version number" -> 1.2.0, a minor by the policy's last-bits and
      wrong-numbers rows; "Go", then "Release" twice for the dispatch, which
      the classifier refused once). First run 39152507210 on `b496262`
      failed one check and was cancelled before the approval, nothing
      uploaded: `rss.ew_cov_wide.linux` read 416.78 MiB, 7.54% under its
      budget, past the quarter line this task set (the push's CI on the same
      sha read inside it); `a5e2f8c` tightened it and the case's macOS and
      Windows legs (7.15%, 6.83% under) to their 1.1.0 readings. Second run
      39163116713 on `a5e2f8c`, every job green, PyPI approved by the user,
      `v1.2.0` and the GitHub release by the workflow. Verified from a clean
      venv: 1.2.0 native 1.2.0 schema 62, a fit tracks its target. Before
      the dispatch: compare_release identical to 1.1.0 (297 fields of 31
      specs); benchmark.py against 1.1.0 built here, six interleaved rounds,
      37 of 38 medians within 2%; `rcov` 0.91 in every round, bisected to
      batch (b), whose code it does not run, and 0.99 with the symbols kept:
      code layout, declared; the PyPI wheel at a load of 53 read 0.98
      (median, too noisy to settle); wide_benchmark.py medians 0.98-1.01;
      budgets drift at most 0.4%; the floor Polars 1.37.0 suite green.
      Recorded after: budgets.toml's release counts from the run's artifacts
      (`--release 1.2.0`), RELEASES gains 1.2.0 (its states load and go on
      as this build's own), RELEASE-READINESS's list follows.

<a id="s23"></a>
## 23. Review round 9 (2026-10-23): the whole repository, code, theory, performance and memory

The user, 2026-10-23: "Do a deep review of the code and theory", then "Be
sure to consider performance and memory use". Twelve read-only reviewers at
`cb230ef` (1.2.0 and its record), one slice each in a worktree slot, four at
a time: A the Gram foundations, B the Gram-backed and dynamic regressors, C
the online learners and robust fits, D covariance structure, regimes and
change points, E marginals, classification and clustering, F the core's
foundations and state, G the bank and stream, H the window operators,
formula targets, GBDT adapter, runner and command line, I the Python package
and bindings, J tests, scripts and CI, K the documents and the theory they
state, M performance and memory end to end. Three lenses: code (as rounds
4-8), theory (every update equation, weight, null, critical value and
stated bound against its paper, with a second opinion from a library
wherever one computes the quantity) and performance and memory (every
per-row path's cost by its order, allocations per row, state kept, peak
memory against the stream, with a measurement and the load logged). Every
finding shown by a probe; the coordinator reproduced A-1, B-1, C-1,
D-1/D-2, M-1, M-2, M-3 on the main checkout and confirmed F-1's and H-1's
mechanisms in the source, and audited the locks: no yanked crate (467) or
package (88); OSV advisories against `rustls`, `rustls-webpki`, `tokio`,
`quick-xml` (the wheel's graph through polars' cloud reader), `certifi`
(docs group) and `pillow` (dev group). Findings and probes: the session
scratchpad, `review9/FINDINGS.md` (consolidated), `FINDINGS-<slice>.md`,
`<slice>/`.

**Theory: nothing wrong against a paper.** Every recursion, merge
identity, null distribution, critical value and documented constant checked
in the code (A-F, H, I) and the documents (K) is the paper's or a recorded,
stated departure; the second opinions taken anew (sklearn PA, Lasso,
ElasticNet, QuantileRegressor, LDA, GaussianNB, k-means step, stump;
statsmodels OLS, acf, jarque_bera; scipy quadrature and distributions;
Polars' own window kernels) agree to rounding or to the stated statistical
tolerance. Where theory parts from the code: a stated scale that is not the
paper's robust one (C-1), and stated bounds or orders the mechanism does not
give (A-4, D-3, M-2, M-3, M-4).

**Found: 0 high, 14 medium, 41 low.**
- Medium, numbers and guards. C-1: `sgd(loss="huber")` scales its cut by
  the plain EW RMS of every residual, so on a target with a level and no
  decay the start-up residuals set the scale for good and the cut never
  binds (squared loss's slope 3.563 at level 1,000; the robust fit's
  2.189). B-1: `kalman`, standardized without `obs_var`, predicts exactly
  0.0 on the rows past `min_weight` before its prior is sized, and that 0
  seeds the residual variance with the level squared (sigma 106 at row 2
  for a noise of 0.5). A-1: the `gram_block_rows` budget counts one block;
  `own_rows` holds one per pattern of missing rows (ten targets, ten blocks,
  2.5 GiB accepted as 256 MiB). F-1: the core clock takes a NaN value as a
  step of `gap_cap` on that row and the next and poisons every later stamp
  (the bank refuses it first). H-1: the command line hands a URL input to
  polars' cloud reader (an HTTP HEAD over `rustls 0.23.45`, ten retries),
  so the advisories X-1 found are reachable from a user-supplied input.
  J-1: the `streaming_bounded` RSS budget cannot tell a bounded stream from
  a whole-frame read at its 1M-row size (collect-then-plan 185.8 MiB, under
  the streaming run's 192.8). J-2: the release comparison runs 135 of the
  346 output field names and never reaches 20 paths, against "every output
  field".

- Medium, performance and memory. M-1: a wide spec's struct output costs
  7.6-15x in Polars' parquet writer what the same values cost as flat
  columns (`ew_cov` k=50 through `sink_parquet` 47,649 ns/row, the bank's
  row 2,900). M-2: a windowed `ewm_mean`/`var`/`std` costs O(rows in the
  window) per row where `ewm_sum` is O(1) (161 to 36,995 ns/row for 1,000
  to 100,000 rows). M-3: a formula target or any `embargo` pays O(held
  rows) per row (536 to 20,078 ns/row for 10 to 10,000 held; a `Vec`
  shifted from its front). D-1: `corrchange`'s monitor closes a span in
  O(T²) per pair. D-2: the window kind clones its whole ring every row (204
  allocations a row at w = 100). G-1: a (spec, group) task carries ~3.6 µs
  of fixed work per chunk, so a bank of many small groups runs 5-18x below
  one stream (M-6: half of the `ewridge` figure is the young streams' solve
  cadence). M-4: the bank's row is 2-3.5x the model's step for the narrow
  shapes the README leads with, against the headline's claim. M-5
  (IMPROVEMENTS C8, measured): the CLI on the system allocator is 1.3x a
  row and 1.8x a task slower than under jemalloc. X-1: the advisories above
  and no advisory job.

- Low (41): B-2 (`sandwich.rs` maps a NaN variance to `0.0`); B-3/I-2
  (`coef_stats`' dof counts a dropped column, Rust and Python); A-2, A-3
  (measured proposal), A-4; C-2; D-3..D-8; E-1..E-4; F-2, F-3; G-2; H-2..H-4;
  I-1 (`po.eval`'s long form 7x the frame), I-3; J-3..J-7; K-1..K-8 (stale
  numbers: the README throughput table three releases old, the install size
  0.13.0's, the windows state version); M-7..M-9. Each in
  `review9/FINDINGS.md` with its probe.

**Decisions for the user** (recommendations in `review9/FINDINGS.md`):
D1 C-1's scale (recommended Huber Proposal 2 in `sgd` and `robust`, a
numbers-moving minor); D2 A-3 (measure end to end first); D3 advisories
weekly or gating (recommended weekly, failing on the shipped graphs); D4
M-1's CLI output layout (recommended an `--output-layout` flag, the
default kept, `unnest` documented for the Python sink); D5 M-5 (rule 12: a
`#[global_allocator]` in the CLI alone; recommended jemalloc/mimalloc as
py-polars). Everything else needs none. Proposed batches (a) numerics,
(b) the window and target paths, (c) the bank and output path, (d) the
CLI, locks and CI, (e) tests, budgets and documents, in
`review9/FINDINGS.md`.

**Decided** (the user, 2026-10-23: "Fix all using opus high workers then
review changes with fable high and fix any findings", to the five decisions
as recommended): D1 Huber Proposal 2's scale in `sgd` and `robust` (the
clipped squared residual over `E[ψ²]`), a numbers-moving minor declared with
its measured size; D2 the lower-triangle per-row update measured end to end
first and adopted only at ≥ 1.5× for an unblocked wide `ewridge`, otherwise
recorded with the measurement; D3 a weekly `advisories.yml` over both locks
failing only on a crate in the shipped graphs, the lock bumps and the URL
refusal now; D4 an `--output-layout` flag on the command line (the struct
kept as the default) and `unnest` before `sink_parquet` documented; D5
(rule 12, raised and approved) a `#[global_allocator]` in the command line
alone, jemalloc on unix and mimalloc on Windows, the wheel unchanged. Built
as task 270.

<a id="task-270"></a>
- [x] 270. **Review round 9's fixes** -- the user, 2026-10-23 (above).
      Five batches, one worker each in a slot, test first (each new test
      fails on `cb230ef` where its finding says; a performance fix shows its
      measurement before and after, with the load, and keeps every output's
      bits unless the finding says otherwise): (a) numerics: B-1, B-2 (+
      `kalman.rs:1894`), B-3 + I-2, A-1, F-1, C-1 (D1); (b) the window and
      target paths: M-2, M-3, D-1, D-2, E-2, with the instruction cases that
      pin them; (c) the bank and output path: G-1 (M-6's cadence caveat in
      its measurement), M-4's `Step::pred`, I-1, E-1, E-3, A-2, A-3 (D2),
      D-5..D-7, F-3, H-3, C-2; (d) the command line, the locks and CI: H-1,
      H-2, X-1's bumps, the advisories job (D3), J-4, M-1 (D4), M-5 (D5),
      H-4; (e) tests, budgets and documents: J-1, J-2, J-3 and slice M's
      proposed cases, J-5..J-7, A-4, D-3/D-8/M-9, D-4, E-4, F-2 (+ its
      case), G-2, I-3, K-1..K-8, M-4's headline, M-7, M-8. Workers do not
      edit the CHANGELOG's `[Unreleased]` or this plan; the coordinator adds
      both at merge. Then a review of the merged diff (Fable, high effort)
      and the fix of its findings before the push.
      *Done 2026-10-24*, eleven batches merged into `main`, each worker an
      Opus at high effort in a slot, each new test failing first on the old
      code: (d) `b418fc3` H-1/H-2 (the CLI refuses URLs by name), M-1
      (`--output-layout flat`), M-5 (the CLI's own allocator, jemalloc off
      Windows and mimalloc on it: rule 12, raised and approved as D5), X-1
      (rustls, rustls-webpki, tokio bumped; certifi, pillow), D3 (a weekly
      OSV advisories job, quick-xml accepted until object_store moves), J-4,
      H-4; (a) `777f3ee` B-1, B-2, B-3 + I-2, A-1, F-1 (C-1 moved to task
      271); (c) `97bec0c` G-1, E-1, E-3, A-2, D-5..D-7, F-3, C-2, H-3, I-1,
      A-3 measured and not adopted (1.14-1.20x at the default schedule, below
      D2's 1.5x), the ChunkOut move; (e) `66b214c` J-1..J-7, F-2, the
      documents; (f) `fbb4604` M-4's code half, `OnlineModel::step_into`;
      (g) `558864c` `break_wald`'s 2k-row rule (its variance is singular in
      exact arithmetic below 2k rows), D-4 measured and declined (a layout
      change for a path no budget runs); (h) `e0accf8` `sgd`'s `step_into`;
      (b) `798b380` M-2, M-3, E-2, D-1 (declared last bits), D-2 in place
      (the O(pairs) form declined: a layout change and drift at a level).
      **Task 271 (b)'s squash reverted (f)** in twelve files (its commit had
      (f)'s merge as parent and a tree from before it); found at (h)'s merge,
      repaired in `92ce27b` by restoring the twelve files from `fbb4604`;
      workers forbidden to squash, every merge checked against its report.
      Then the review (Fable, high effort, four slices R1-R4 over
      `a0923f6..20a8ece`): no high finding; R1-1 (sgd's level-shift cost
      under the MAD), R1-2 (`break_wald` after a wiping gap), R4-1 (a
      release-comparison sentence claiming bit identity) medium; the rest
      low. Fixed by (i) `a9b95e6` (the two pre-existing sgd/pa defects batch
      (h) found: a scaler history below the normal range forgotten,
      `inv_scaling`'s rate on the decayed weight, a numbers move declared;
      `break_wald`'s wait restarted after a wiping gap; sgd's figures and
      the `huber_scale_half_life` lever documented; a `bank_autocorr`
      fixture) and (j) (the documents, the policy page's record of the
      `huber_scale` default, the RSS placeholders unmeasured, the run-buffer
      decision from the headers, the release comparison reaching kalman's
      unsized prior). The disk filled once during the review (450 GB of
      slot builds); cleared. Follow-ups: bound `ew_cov`'s tail-quantile
      sketches as the MAD's are (a layout change), the per-group bank
      constant's remaining part (group keys), D-4 and D-2's O(pairs) form if
      a budget ever runs those paths.

<a id="s24"></a>
## 24. Huber regression, from first principles (2026-10-23)

The user, after task 270 (a) was handed C-1 under decision D1 (Huber's
Proposal 2 scale): "Does ridge support Huber loss", "How many passes would
it take for full convergence", then "Do a deep analysis of Huber make a
complete plan of what we should do from first principals". The analysis is
`docs/HUBER.md`; its measurements are `scripts/huber/m1_breakdown.py` to
`m5_request3.py`. In brief: the scale is the estimator's robust part, and the
plain RMS has breakdown 0; Proposal 2 holds to about 12% one-sided
contamination in batch and, online, does not repair the inflated start that
is C-1's cause (started from the RMS it stays at the RMS over 0.71), so
**D1 is withdrawn**; the EW-MAD (one `EwQuantile` at 0.5 per target) needs
no start rule, and one online pass with it equals the converged batch fit
at 1% and 5% contamination (2.008 / 2.040 against 2.008 / 2.040), with
re-weighting passes closing under half of the gap that opens at 12.5%; a
robust scale slows adaptation after a regime shift (mean error 19.5 against
least squares' 13.2 in rows 200-400) unless the scale has a shorter memory
than the fit (14.4 at a fifth). The plain RMS is the *stated* definition
(docstring, MODELS.md), so the policy makes a replaced default a major: the
plan adds `huber_scale` with the present behaviour as the default.

<a id="task-271"></a>
- [x] 271. **`huber_scale` and the Huber documents** -- the plan in
      `docs/HUBER.md` §7, P1-P6: `huber_scale="rms"|"mad"` on `huber` and
      `sgd(loss="huber")` (an `EwQuantile` per target, schema 63 with the
      v62 sets kept and a loader, fixtures for both forms);
      `huber_scale_half_life` defaulting to the model's; `sgd.rs`'s module
      doc corrected; MODELS.md's `huber` section rewritten from first
      principles with the measured tables; the two specs in the release
      probe's workload; the 2.0 default recorded. C-1 moves here from task
      270 (a) (held, not built there). Tests: statsmodels `RLM` with the MAD
      scale as the second opinion at the measured tolerance, the level-1,000
      no-decay stream for `sgd`, the breakdown table at 1% and 5%, the shift
      test, fixtures, chunk invariance. P7's three items (Mallows weights, a
      shift-aware scale, `sgd`'s intercept start) wait for their own
      measurements. **Decided** (the user, 2026-10-23: "Build this using
      opus high and then review with fable high and fix findings. Move away
      from rms now. Your reco on the rest."): (a) the default is `"mad"`
      now, `"rms"` kept as an option, the change declared under *Changed*
      with its size and the way back (the user's call against the policy's
      changed-default row); (b) `huber_scale_half_life` built now; (c) P7
      out. Then a review of the change (Fable, high effort) and the fix of
      its findings.
      *Done 2026-10-24*: `77ab85c` (`huber_scale` "mad" default with "rms"
      kept, `huber_scale_half_life`, C-1, schema 63 with the v62 sets and a
      loader reading a v62 state as "rms"; slopes 2.008 / 2.041 / 2.128 at 1%
      / 5% / 12.6% outliers against statsmodels' converged 2.008 / 2.039 /
      2.124; sgd's C-1 3.751 → 2.016), and 271 (b) `063c0c8` (the MAD's
      sketch bounded to 16 octaves around its median, 13.9 → 9.3 KB per
      target and a wild residual no longer 680 KB; a select for every
      sketch's per-row branch, 10-12 → 7-9.5 ns), reviewed with task 270.

<a id="task-272"></a>
- [x] 272. **Huber through the bank, `po.eval`, `po.gram`, `marginal` and
      the window operators** -- a downstream project's five requests, the
      user, 2026-10-23: "Analyze these recos from another project and build
      them after the other parts are done. Build with opus high and review
      with fable high." The analysis is `docs/HUBER.md` §8. After task 270
      and task 271 merge. Batches: (a) `huber`: `feature_sets` (the weight
      from the largest set's residual, stated), `ridge_scale` (the
      `rls`-equivalent Huber), an emitted robust scale read before the row,
      and the delayed-label rule tested and stated; no `loss=` on `ewridge`
      or `rls`, their docs pointing to `huber`; (b) `po.eval`: `huber_sse` /
      `huber_mse` with a causal scale column or the emitted scale; `po.gram`:
      `pseudo_response(y, pred, scale, c)` and a GRAM.md section with
      algorithm H's passes table; request 3's per-target cross-moment
      weights refused with the measurement (slope -59.4 against 2.0393); (c)
      `marginal(bin_stat="huber")`, the online M-location per cell with a
      per-target scale, default `"mean"`; (d) `po.ewm_quantile` on the
      window machinery, the EW-MAD recipe and the label-delay rule in
      WINDOWS.md, the trees' objectives in GBDT.md. Each with its second
      opinion (statsmodels `RLM` / Huber location, the batch definition of
      the loss, numpy's weighted quantile) and, where a path is new, a
      budget case. Then a review (Fable, high effort) and the fix of its
      findings.
      *Revised 2026-10-23* after the requester's reply (relayed by the
      user): request 3 withdrawn (they checked it: slope 1.92 for 2.00 with
      outliers independent of `x`, and the pseudo-response matched a direct
      reweighted fit to 5e-15); request 2 is their own scoring code; request
      1 is `po.spec.huber`, already built, with the robust scale arriving in
      task 271; requests 4 and 5 conveniences. So batch (b) shrinks to the
      GRAM.md section on algorithm H with its passes table (no helper, no
      `po.eval` metric), and (a), (c), (d) stand, in that order of value.
      *Done 2026-10-24*: (a) `c3622cd2`, (c) `6bcc3144`, (d) `dffcabcf`, the
      algorithm H section in (a). The Fable review (R5, R6; scratchpad
      `review9/fixreview272/`) found no defect in code: R5-1 (the largest
      set is the plain fit to the bit only at every feature or the same
      `min_weight`), R5-2 (GRAM.md against `RLM(scale_est="mad")`, 1.2e-6 at
      ten passes, 3e-7 = 0.6745 against Φ⁻¹(3/4)), R5-3 (the emitted scale
      null on an unread row), R6-1 (the step bound `c·w/(λΩ+ω)`), R6-2 (the
      recovery law `E₀·exp(−c·n/(Ω₀E₀))`), R6-3 and R6-4 were documentation;
      R5-4, a 3.5% dearer solve, was a shares base allocated at every solve,
      now filled only when read (1,960 → 1,923 ns a row solving every row;
      about 1.5% above the pre-sets row at the bench's cadence, left for
      CI's instruction count to judge). Fixes in (e), merged with task 270
      (l) (the clock guard's second difference, both clock cases now below
      their pre-guard counts).

<a id="s25"></a>
## 25. Windows' math library and hardware in the tests (2026-10-24)

<a id="task-274"></a>
- [x] 274. **Linear-algebra oracle bounds derived** -- from task 273's Rust
      audit: with every tolerance literal scaled by 1e-5, 30 of the 42
      `crate::oracle` comparisons still pass, so a small-width kernel 100×
      past its rounding would pass a 1e-9 tolerance, and the weekly SDE
      legs could run exactly such a kernel. Add an `oracle::forward_bound`
      helper (γ_n·κ, as `trisolve_tests.rs`' `forward_bound` does) and hold
      each comparison to it, with the measured headroom in a comment.
      *Done 2026-10-24*: `oracle.rs` gained γ_n, dot and running-error
      bounds, normwise and componentwise forward bounds (Higham 7.2, 7.11),
      the library Cholesky's backward bound (Thm 10.4) and a posteriori
      errors of the oracle itself; two test-only replays (`ewcov/rounding.rs`,
      `sandwich/rounding.rs`). About 32 comparisons sit at 0.01-0.31 of
      their bounds or agree to the bit, and fail at 100× their rounding;
      eleven dominated by long accumulations sit at 1e-4 to 1e-2 (the
      worst-case per-step bound grows with N, the realized error as √N),
      still 10³-10⁶ tighter than the tolerances they replaced. Definitions
      summed about 0 at a level are now summed about the level (their own
      rounding was 1e3-1e5 times the library's). No library defect.

<a id="task-275"></a>
- [x] 275. **A window's boundary factor from the product of the per-row
      factors** -- from task 273 (w)'s finding 1; the user, 2026-10-24,
      chose option A. Under a half-life the live moments decay by each
      row's `factor(d_i)` and the window subtracts a snapshot scaled by one
      `factor(Σd_i)`; with a per-row factor e ulps off (a platform `exp2`
      unlike Apple's) they part by about d·e ulps, and a held target over a
      199- or 500-row window read a variance near 1e-15 instead of 0 (140
      of 490 perturbed cases). Fix by representation: `Snapshots` keeps a
      running product of the per-row factors (with a power-of-two exponent
      beside it, so it never underflows), each ring entry the product at
      its snapshot, and the boundary factor is their ratio; rounding then
      random-walks (√d·ε) instead of drifting. Every windowed model
      (ewclass, ewridge, ewcov, lasso with a second product for its
      selection decay, marginal). A layout change inside the unreleased
      schema 63: fixtures regenerated, 1.2.0's loader seeds each entry's
      product so the first read after loading is unchanged. Built on a
      branch with its measured memory and speed cost, which go to the user
      before it merges (CLAUDE.md, Budgets).
      *Done 2026-10-24*: a ring entry 48 → 96 bytes in memory, +12 bytes a
      saved snapshot per decay (23 for `lasso`), states +42 to +75 bytes, a
      bank +378; speed within noise (windowed cases 0.98-1.06, budget cases
      0.992 geometric mean); 19 of 1,299 probe fields move, all windowed, at
      most 3.3e-13 relative; all 490 perturbed held-target cases read 0
      (main: 140 nonzero). *Decided by the user*: "Merge, accept the
      relaxation" -- a windowed state from 1.2.0 (schemas 61, 62) loads and
      its first read is exact, but once a snapshot taken after the load is
      the window's edge it continues within the golden tolerance (1e-12),
      not to the bit, since the old state never recorded the per-row
      factors; unwindowed states stay exact. Recorded in rule 5
      (CLAUDE.md) and the `previous!` docs of both fixture files.

<a id="task-277"></a>
- [x] 277. **Huber's converged warm-up, on by default, for `huber` and
      `sgd(loss="huber")`** -- the user, 2026-10-25: "Fix the cpu paths and
      check the Huber warmup", then, on the measurement, "1.4.0, on by
      default" and the `sgd` analogue "same release". Measured (scratchpad
      `huber-warmup/`): one online pass is *captured* in some streams -- the
      outliers pull the first solves toward least squares, the MAD sketch
      learns that fit's residuals, the cut is drawn wide and the outliers
      keep entering at nearly full weight. Medians hide it (0.001 at 12.5%);
      seeds do not: 13 of 40 captured at 12.5% (worst 0.43 in slope), 25 of
      40 at 20% (worst 4.3), and without decay it persists (200,000 rows,
      20%: 4.67 against the oracle's 2.34). The level makes no difference.
      A refit of the coefficients alone is pulled back; refitting and
      rebuilding the scale from the converged residuals fixes it: warm-up
      200 captures 0 / 0 / 4 of 40 at 12.5 / 15 / 20% (1,000: 0 / 0 / 2), a
      one-off 0.07-5 ms, per-row speed unchanged, about 8 KB a target held
      until the refit. `sgd(loss="huber")` is captured too (20 of 20 seeds
      at 20%). Ship in 1.4.0: schema 64 (v63 sets to `v63/`, a loader that
      reads a v63 state with the warm-up off), fixtures freezing a mid-buffer
      and a post-refit state, a clearer public name than `warm_rows`
      (`hmm`/`kmeans` have one), default 200 on, `0` restoring 1.3.0's
      numbers to the bit, declared in the CHANGELOG and in the release
      policy as a second default moved in a minor (beside `huber_scale`),
      MODELS.md and docstrings with the capture tables, the `sgd` analogue,
      the refit row solving only its own target, budgets and the release
      comparison declared. Then a review and the fix of its findings.
      *Reviewed (two Fable reviews, 2026-10-25)*: a high finding -- the
      buffer held every ageing row at full width, unbounded -- fixed by runs
      of `(λ, λ_s, count)` with a cap past which a target gives its refit up;
      the rest fixed (passes uncounted as failures, `sgd`'s `l2` in the
      refit, the resumed 1.3.0 bank documented and tested, capture counts
      with their stream lengths, a reproducing script). *Decided by the user,
      2026-10-25*: a per-spec refit-buffer budget of 256 MiB, past which a
      target gives up its refit (counted, with a notice); the default count
      500 ("500"), scaling as `max(500, 10·k)` up to 100 features and off by
      default above 100 (an explicit value still taken); and
      `marginal(bin_stat="huber")`, captured by the same mechanism (2,000
      rows at 12%: 13 of 40 seeds with a bin more than 0.1 scale off, worst
      4.1), fixed in 1.4.0 too.

<a id="task-278"></a>
- [x] 278. **Convergence tests of every kind, before 1.4.0** -- the user,
      2026-10-25: "What additional tests should we do to catch all kinds of
      convergence issues", then "Do them all before 1.4". Task 277's capture
      was invisible to medians (0.001 at 12.5% while 13 of 40 seeds were
      captured, worst 0.43). Rules for every class: judge by the worst seed
      or a high quantile over 50+ seeds, never a median; the oracle is a
      third-party batch fit on the same weighted rows; measure the rate
      (error against n, or against half-lives), not only the endpoint; print
      evidence the hard path activated. Classes: (1) consistency on
      stationary known-truth streams, no decay -- the worst seed's error
      shrinks with n toward the oracle; (2) start independence -- the same
      data from a wrong state (a loaded state from another regime, extreme
      coefficients, a large level) reaches the same fit; (3) front-loaded
      adversity -- outliers, a shift or heavy tails in the first rows; (4)
      contamination to breakdown -- zero captures below each robust model's
      documented breakdown (huber, quantile, sgd's losses, the MAD
      sketches); (5) re-convergence after a mid-stream break, within a
      stated number of half-lives of the decayed batch oracle; (6)
      equivariance -- y scaled by 1e±6, x affine, the level shifted, the fit
      transforms as theory says; (7) conditioning -- near-collinear,
      rank-deficient and 1e-10..1e10 features, predictions within task 274's
      derived bounds; (8) inner-iteration honesty -- lasso's coordinate
      descent, hmm's EM, kmeans, the warm-up refit: the stopping rule met
      (KKT for lasso, residual otherwise), every iteration-limit hit
      counted; (9) long-run stability -- 1e8-row soaks with periodic
      from-scratch recomputes, kalman's P positive definite and bounded; (10)
      learning-rate models -- sgd, pa, ftrl: excess loss falls under each
      schedule, a large rate fails loudly; (11) truth recovery for kmeans,
      micro, hmm, bocpd on known synthetic regimes, by the worst seed. A
      harness in the extended tier: each model registers a generator, an
      oracle and the classes that apply, so a new model must opt in
      (EXTENDING.md). Heavy grids (4, 9) weekly. Library defects the classes
      find are reported and fixed before 1.4.0.
      *Widened 2026-10-25* (the user: "We should use this chance to test all
      parameters that can't be compared to third party libraries", and to run
      as much as possible in parallel): (12) every public parameter with no
      third-party oracle gets oracle-free tests -- limits reducing it to an
      oracled model, equivalent spellings, monotonicity, the invariances it
      must keep, boundaries, chosen interactions, a definition written by
      hand where no library computes it -- and an activation test showing it
      takes effect where it should and nowhere else; a parameter no test shows
      taking effect is a dead-parameter suspect. A read-only inventory
      (scratchpad `t278/PARAMS.md`) groups the missing tests into batches;
      heavy grids run on CI (free, 20 concurrent jobs) rather than locally.
      *Inventory 2026-10-25* (`t278/PARAMS.md`, 436 parameters: 176 against
      a third-party oracle, 222 shown taking effect, 38 with no test showing
      an effect, of which probes show 37 act): three inert keys, four
      defects, invalid values accepted; 33 batches that do not collide on
      files. *Decided by the user, 2026-10-25*: (i) the inert keys act or
      are deprecated -- `marginal`'s `min_settled_frac` and `rcov`'s
      `min_weight` withhold as their names say, `with_windows` no longer
      requires a numeric `session_gap` and warns that a number has no effect
      (`"reset"` kept); (ii) `session_shrink`: "Do the right thing do not
      worry about moving results as long as we can test" -- a blend leaves
      the solve schedule alone, as its docstring says, so the output is
      continuous in the factor and 0 is no shrink to the bit, the move
      declared and pinned; (iii) the nonsense values (`min_weight=inf`,
      `bin_edges=[[]]`, kmeans `dead_frac >= 1`, `emit_averaged` on one
      slot, corrchange `bandwidth` unbounded) refused by name as validation
      fixes. Plain fixes without a decision: `rls` under
      `drift_action="reset"` matches a fresh bank on the next row as every
      other kind does (it differed by up to 0.017); `with_windows` refuses
      `session_gap` without `session` and `restart_after_step_back` without
      `clock`; `bocpd`'s refusal names its current keywords. The user, 2026-10-25,
      asked for as much parallel work as possible and approved six builders
      at once for this effort (CLAUDE.md's four otherwise):
      `POLARS_ONLINE_SLOTS=6`.
      *Classes 6-8 merged (278 (b))*, finding seven defects, each pinned as
      a labelled test of today's behaviour; fixed before 1.4.0 under the
      user's direction ("do the right thing ... as long as we can test"),
      each declared: D4 (high) a numerically singular normal-equations solve
      is reported nowhere -- estimate the condition from the Cholesky factor
      and withhold and count above a threshold, a new `withheld_reason`; D5
      `gram.solve(standardize=False)` rescales the Gram by its diagonal
      before `eigh`; D1 `sgd`'s `clip_gradient` measured in the residual
      scale's units (as built, in `s_y`, the target's EW spread, not the
      residual scale; task 282, S3-2); D2 AdaGrad's epsilon relative; D3
      `lasso`'s `tol` relative to the target's spread (and `gram.lasso_path`'s); D6
      `gram.lasso_path` counts and warns on its sweep cap; D7 `kmeans`/`hmm`
      seeding iterates to a fixed point under a cap and counts cap hits; K1
      `kalman`'s noise estimate inflated while the intercept absorbs a large
      level. D1/D2 wait for task 277 (both touch `sgd`).
      *Classes 9-11 merged (278 (c))*: no drift at 1e8 rows (NIST NumAcc,
      the decayed and windowed moments, ewridge against an fsum fit, kalman's
      P, both clocks); every learning-rate schedule's worst seed converges to
      the batch optimum; bocpd and corrchange miss no break over 50 seeds.
      Found, to fix before 1.4.0: D8 `sgd` at too large a learning rate
      diverges quietly (the clip keeps predictions finite and wrong: MSE 1e6
      against 1.07) -- detect divergence and withhold with a reason and a
      notice, after task 277; D9 `hmm` at its default `warm_rows` (50) never
      recovers sticky regimes (worst 0.54-0.65 in the true state; 0.94 at
      2,000) -- a better default or a re-seeding rule, measured; F3
      `kmeans`' seeding ends in a local minimum in most orders of Fränti's
      A-sets (median 1.26x the optimal inertia) -- greedy k-means++, with
      D7; F4 the standardized metric stretches isotropic blobs, a documented
      trade-off (docs only).
      *Classes 1-5 (278 (a))*, held back from merging until its heavy grids
      leave the local gate for a CI job (task 280's first piece; they added
      ~22 min to `gate.sh --extended`). Found, to fix before 1.4.0: D-Q
      `quantile` draws its band in the residuals' RMS, not robust (50 of 50
      seeds captured at 5% one-sided outliers) -- a MAD-type scale, a layout
      change that joins unreleased schema 64 after task 277; D-K `kalman`
      with a learned R and no drift is captured by an early burst (R
      inflates, the gain vanishes; slopes [-25.5, 33.6] against OLS [4.1,
      -1.4]), overlapping K1; D-H `holt` without decay: the level error grows
      like log n (both gains 1/n, the trend error summing into the level);
      D-S `sgd` does not centre its target (a level L moves every coefficient
      by about lr·L; at 1e6 the clip means it is never learned), with the
      `sgd` group after 277; D-F `ftrl` under a half-life tracks a break at
      AdaGrad's pace (a limit to decide: document or forget the
      accumulators). Defect cells are strict xfails running the real bound;
      none may remain at 1.4.0.
      *Seeding (D7, F3, D9), decided by the user 2026-10-25*: `kmeans`/`hmm`
      seeding runs Lloyd to a fixed point (cap 300 passes, counted in
      `solve_failures`); greedy k-means++ behind `init="k-means++"` and
      `lloyd`'s starts (Fränti S1 worst 2.19 -> 1.001 of the optimum);
      `hmm`'s `warm_rows` default 50 -> 1,000 ("1,000 rows": worst seed 0.46
      -> 0.95 in the true state; each group's first 1,000 rows report null,
      a shorter group never seeds). Rule 5's second exception ("Accept the
      relabelling"): a 1.3.0 `kmeans` state saved mid-warm-up seeds by the new
      rule after loading, the same centres and distances to the bit with the
      labels possibly permuted, held to one consistent relabelling; a state
      saved after seeding continues to the bit.
      *ftrl and holt (2026-10-25)*: D-F is a limit of the method, not a
      defect -- `ftrl` already decays `n`, `zz` and `d`; AdaGrad's normalised
      step caps a coefficient's move at about `2·alpha·√(1 − λ)` a row;
      documented and pinned, defaults kept (a weaker logistic prior measured
      better under a half-life, recorded as a trade-off). D-H is real and
      wider: `holt`'s trend is pinned to the first observation (gains
      `c_g = c_h = 1`), so its level error grows like ln n without decay and
      during every warm-up (worst 3.4-6.3 against the oracle's 0.03-0.2).
      Approved design: per row the larger of the exact weighted line fit's
      gains and the textbook gains, three new f64 a target (`S0`, `ā`, `Q`)
      joining unreleased schema 64 after task 277; statsmodels WLS to 2e-13
      without decay and statsmodels `Holt` to 1e-12 in steady state.
      *lasso (D3, D6), 2026-10-25*: `tol` relative to the target's spread
      (bitwise scale equivariance at 2^±20, 2^±30; outputs move only where
      the spread is not 1, at most 7.8e-10 on the probe); `gram.lasso_path`
      warns naming each penalty that ran out of sweeps. Rule 5's third
      exception, the user's ("Accept within 1e-9"): a `lasso` state of schema
      61-63 continues within 1e-9 relative (measured 5.8e-11, windowed
      1.7e-11), and loads and re-saves the same bytes. The 798-of-799 cap
      hits at defaults are the descent's rate, not the rule; `max_iter` kept.
      *Windowed CUSUMs (D-B5), 2026-10-25*: the windowed `cusum` and
      `cusum_sq` are divided by their null's share of the plain variance at
      the sums' age, read from the sums' own weights (`S1/w4`), not the filled
      share from the first row (first 150 rows past 1.96: 6.5-34% -> 6.1-6.4%
      for `cusum`, 11-16% -> 6.2-6.7% for `cusum_sq`; filled values unchanged
      to the bit under unit weights, within 1e-6 under uneven weights; no
      layout change). Rule 5's fourth exception, the user's ("Accept"): a
      state of schema 61-63 continues to the current values in its `cusum_*`
      fields while its sums are young (`MOVED_SINCE` in the fixture test);
      every other field and the state bytes stay exact. Left: the young
      scale's ~1 point over 5% (needs a new accumulator), and the quantile
      fit's filled CUSUMs oversized at q = 0.9 (8-15%), pre-existing.
      *holt (D-H), built 2026-10-26*: each row takes the larger of the
      textbook gains and the exact weighted line fit's (`fit_weight`,
      `fit_age`, `fit_spread`, schema 64); without decay it is statsmodels'
      WLS line to 1e-15 (1.3.0: up to 3.4 off), decayed it is statsmodels'
      `Holt` to 1e-12 once settled; warm-up worst 3.4-6.3 -> 0.03-0.17. Rule
      5's fifth exception, the user's ("Accept"): a 61-63 `holt` state
      continues to the bit with the textbook gains and is held, beside
      `kmeans`, to re-saving bytes that load back the same.
      *Robust decisions, the user's (2026-10-26)*: (i) decimal-scale
      equivariance of the MAD sketch -- first chosen as a continuous read
      within the bucket, which a proof showed cannot be equivariant (the
      bucket edges map onto themselves only under powers of two; measured, the
      worst case did not move); then the user chose an anchored grid ("Anchored
      grid"): each `EwQuantile` buckets `v / a`, `a` its first usable value
      (one f64 in schema 64), behind a cfg flag a spec sets, for `huber`,
      `quantile`, `sgd(huber)` and `marginal`'s Huber bins (prototype:
      huber's decimal tail 3.9e-3 -> 5.2e-14); `ew_cov`'s `mahal_quantiles`,
      `resid_quantiles` and `ewm_quantile` keep the plain grid; (ii) `quantile` gets
      task 277's converged refit of its first rows, on by default under
      `refit_budget` (D-Q2: no-decay front-loaded wild 0.20 off after the
      MAD fix); (iii) `min_weight` withholds output only in every model --
      `kalman` and `lasso` lose their documented exceptions, `lasso` counting
      its selection errors from its own floor (a row per coefficient)
      ("Both output-only"; kalman at min_weight 50 was 100x off least
      squares on clean data).
      *First period of an explicit cadence (the user, 2026-10-26, "Default
      cadence first")*: with `min_weight` output-only, a model first solves at
      its own floor, and an explicit long `solve_every` then held that fit a
      whole period (`solve_every=500`, `min_weight=50`: served MSE over the
      first 600 rows 1.89e-6 -> 1.58e-5). Until the first explicit period has
      passed, the model solves on the default weight-share schedule; then the
      explicit cadence takes over. Learning stays independent of
      `min_weight`. (The "100x" kalman figure above did not reproduce:
      `noise_units` had already removed it; the real move was through an
      explicit `q`.)
      *"Never solve" (the user, 2026-10-26, "inf = never; huge = never
      too")*: the first-period rule made `solve_every=1e18` (GRAM.md's,
      README's and MODELS.md's "collect the sums, solve nothing") solve on the
      default schedule all stream (a Gram pass 27x slower at 1,000 features).
      `solve_every=inf` now means never solve with no first period, the docs
      use it, and a finite value past a threshold no stream reaches counts as
      never too, so existing `1e18` code keeps its speed.

<a id="task-280"></a>
- [x] 280. **An oracle ledger: no drift from the oracles and datasets, no
      longer gate** -- the user, 2026-10-25: "After all the fixes we need a
      way not to drift from all the oracles and data sets for every
      parameter test. But we do not want to run more very long tests in the
      gate as it's already long." Before 1.4.0, after task 278's fixes. (1)
      A committed ledger, as `budgets.toml` is for speed: per oracle and
      dataset case (every task 278 class, every parameter test with an
      oracle), the error against the oracle (worst seed, p95) and the
      oracle's version; it only tightens -- a larger error fails unless a
      `[[raised]]` entry gives the reason and a CHANGELOG line, a smaller one
      asks to be tightened, as `scripts/budgets.py` does. (2) The oracle side
      (the batch fits over many seeds, the downloads) runs on CI on every
      push, sharded across parallel jobs (free, 20 concurrent), and its
      conclusion is read after each push with the others. (3) The gate keeps
      a cheap guard: a hash of our outputs on each case's inputs, seconds in
      all; unchanged means the comparison still holds (the oracle's answer is
      fixed by its pinned version); changed names the cases whose oracle runs
      must re-run (one command locally, or CI on push), and `quick.sh`-style
      selection runs a changed model's cases only. (4) A weekly full sweep,
      and any bump of an oracle library or dataset in `uv.lock` re-runs
      everything.
      *Done 2026-10-26*: `oracles.toml` (59 groups, 755 cases; counts held
      exactly), `scripts/oracles.py check|apply|raises|rerun`, the Oracles
      workflow's ledger job and weekly sweep, the gate's hash guard (~11 s);
      NIST StRD and TCPD in the grids' CI job; seed-grid twins of five
      single-seed second opinions. Open until the first CI run: whether Linux
      and Windows share the macOS hashes.

<a id="task-282"></a>
- [x] 282. **Review round 10's fixes, before 1.4.0** -- the user, 2026-10-28:
      "Review the 1.4 changes theory and code". Nine read-only reviewers over
      `v1.3.0..main` (92d5bd41..f36a4a6c), code and theory (§26). Theory
      clean everywhere; 1 high, 7 medium, 32 low. Decisions, the user's
      (2026-10-28): the six 20%-outlier strict xfails are reclassified as a
      documented bound (assert the measured capture share, a Known issue),
      not fixed; the `unstable_rate` gate bounds each row's `q` before it
      enters the sums AND derives its memory from the fit (the model's decay
      or the rate's relearning time) rather than the 100-row constant --
      "consider whether the 100 row half-life is fitting the library features
      to the test data". Four slot workers, two at a time: W1 sgd + the Huber
      refit (H1 the refit ridge under standardize, M1 the gate), W2 robust +
      linear + regimes + diagnostics (M2 the one-coefficient returning
      target), W3 the stream and bank (M3 the reset under an embargo) + the
      xfail reclassification, W4 the ledger's enforceability (M4, M5), the
      released-state continuation test (M6), the readiness policy (M7) and
      the documents. `.claude/coord/review10/FINDINGS.md` is the list.
      *Built 2026-10-31* in four slot branches merged on `integration` (one
      essentials gate PASS at a44f3498): W1 H1 (the sgd refit's ridge is
      `1e-6 s_i^2` under standardize; the b19 strict xfail gone), M1 (each
      gain bounded at max(2, 4 x the running mean); the 100-row memory KEPT,
      measured: the fit's own memory, a handful of rows near the bound, made
      the verdict noise -- 6-11% of a stable rate's rows withheld, a third
      to a half of a diverging one's served; the reason is the sample size
      of an expectation, now in the docs), S3-5/6/7, S1-2/3/4, S9-3; W2 M2
      (the floor solve on the post-reset count), S4-4 (a jittered factor
      judged as A + delta*I), S6-2, S2-2..6, S4-1..3, S5-1, S6-3/4, S9-6; W3
      M3 (the reset marked under an embargo; `front_cut`, `d_fresh`), S7-2..5,
      S7-7/8, the six 20%-outlier cells a documented bound (Known issues);
      W4 M4/M5 (the ledger: floor, tolerance, ideal, `[[retired]]`, `fewer`),
      M6 (every moved spec of every 1.x release held to its release's own
      continuation; the young CUSUM rows held to an exact rescaling by a
      share in (filled, 1]), M7 (five changed defaults in RELEASE-READINESS;
      a 1.3.0 hmm state saved mid-warm-up keeps its 50), S8-6..9, S9-1/4/5,
      Q1 (quick maps an online-core change to the bank fixtures), Q2 (36,000).
      Open, the user's decision: N1 -- the quantile refit runs to its pass cap
      at ridge >= 1e-2 (6/40 at 1e-2, 35/40 at 1e-1; 0/40 at the default),
      the penalty's weight W_h/4h re-read from each pass's fit, so each pass
      descends a different F (`.claude/coord/review10/FINDINGS.md`, N1).
      Inner/outer (h and W_h held within a solve) was built and failed on a
      held-out grid (W5, not merged). DECIDED 2026-10-31, the user ("Your
      reco"): option 1. Quantile's penalty is weighed by the target's total
      decayed weight W, not the band's W_h, in the stream's solve and the
      refit alike (the band diagonal carries lambda*W/W_h). That is
      sklearn's QuantileRegressor (alpha*sum w) and conquer's form. W does
      not move with beta, and h (a weighted MAD) is continuous in it, so
      the refit's F is continuous. States before schema 64 keep the old
      rule. Built as W6 (`task-282-n1b`): converged refits agree with scipy
      to 1.4e-9, and caps fell 2522->453 at 1e-1 and 614->0 at 1, but they did
      not end. The rest are 2-cycles of h (a weighted MAD re-read each pass),
      15/5760 even at the default ridge, and W2's probe rose 6->20/40 at 1e-2.
      Variant P (h and the penalty held at the stream's own values at the
      refit row) ends the cycling but fits wild first rows worse than no
      refit (2.86 off QuantReg at tau 0.8, vs 1.58 without the refit and
      0.0075 for main): the stream's scale there is 15-88x inflated.
      DECIDED 2026-10-31, the user: a Known issue for 1.4, main's refit
      unchanged (CHANGELOG); N1 stays open after 1.4. The W5/W6 patches are
      in `.claude/coord/review10/`.
