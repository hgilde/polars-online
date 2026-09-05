# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/) — while pre-1.0, the minor version
carries breaking changes.

## [Unreleased]

### Added

- **`kmeans`: exponentially weighted k-means** (`po.spec.kmeans`,
  `docs/PLAN.md` §11a, task 23). The first model with no target: every
  column of interest goes in `features`, and each row's outputs are read
  from the centres *before* the row is learned — `cluster` (the nearest
  centre's index, `i32`), `dist` and `dist2` (the distance to it and to the
  runner-up), `n_eff`, and the centres as `coef` (`k` slots `cluster{j}`,
  one coordinate per feature; `po.spec.coef_index` lays them out and
  `unnest` names them `coef_cluster0_x1`). Distances are in units of each
  feature's EW standard deviation unless `standardize=False`. Seeding waits
  for `warm_rows` learned rows, then `lloyd` (default; ten restarts of ten
  iterations over the buffer), `kmeanspp`, `farthest` or `first`; centres
  update every `update_every` rows. Halflife, clock, weight, group and
  `min_periods` mean what they mean everywhere else, and a null or
  non-finite feature row is skipped with its clock tick folded into the next
  row's, as for every other model.
- **A split–merge move for `kmeans`**, on by default (`split_merge=0.5`,
  `sm_every=100`, `dead_frac=0.05`; `split_merge=0` gives plain k-means).
  Rows farther from every centre than `1 + 4·sqrt(2/p)` typical radii are
  summarised per cluster instead of learned, so an outlier neither drags a
  centre nor widens its radius. At each check the two closest clusters merge
  when their centres are within `split_merge` summed radii, and a cluster
  lighter than `dead_frac·n_eff/k` is declared dead; the freed centre goes
  to the far rows' mean, once those are at least three rows and five per
  cent of the window's weight. Measured on a blob born after seeding: tail
  ARI 1.000 after `log2(1/dead_frac)` halflives (4.3 by default, 2 at
  `dead_frac=0.25`), where plain k-means never recovers (0.71–0.73). Five
  per cent uniform outliers: ARI 0.984, no spurious move. What it repairs
  and what it costs is in the README's `kmeans` section.
- Rust: `online_core::{KMeans, KMeansCfg, SeedRule, ClusterSummary,
  FeatureMoments, SplitMix64, dist2}`; `ModelState::KMeans`. The state
  schema stays at 2 — a new variant, not a new layout — so a 0.2 bank that
  holds a `kmeans` model fails to load on 0.1 at deserialization rather than
  by version.
- `tests/reference_cluster.py`: a numpy oracle for the whole recursion
  (seeding, standardization, the batch update, far rows, split–merge), held
  bit-exact by `tests/test_kmeans.py`.

### Changed

- **Residual diagnostics are refused for a model that predicts no target.**
  `ew_cov` already refused `emit_selected` and `emit_averaged`; it and
  `kmeans` now refuse `emit_sigma`, `emit_resid_z`, `emit_metrics`,
  `resid_quantiles`, `emit_autocorr` and `emit_drift` too, by name
  (`"emit_sigma does not apply to ew_cov (it has no predictions, so no
  residuals)"`), where `ew_cov` used to accept the flag and silently emit
  nothing for it. A spec that set one of them on `ew_cov` must drop it.
- `Decay::factor` computes `exp2(-d/h)` rather than `0.5.powf(d/h)`. A
  release build already did (LLVM rewrites the one into the other), so no
  released number moves; a debug build now agrees with it bit for bit, and
  so can a reference in another language.

## [0.1.1] — 2026-09-04

A faster chunk plan and a guide to the chunk size. Every number a model
produces is unchanged.

### Changed

- **The chunk plan: every phase parallel, and no stride** (`docs/PERFORMANCE.md`
  §12, P9–P11). A spec's columns are gathered once, group after group, so
  each stream reads a contiguous run; output fields are built one job each
  into `Vec<f64>` + validity; columns are read in parallel, multi-chunk
  columns copied per arrow chunk, and an integer group key is bucketed by
  its value rather than cast to text (the keys and output are identical, and
  a test says so). Measured on a 400k-row chunk over 64 groups at 14
  threads: 37 → 17 ms; the README's 12M-row grouped workload 3.25 → 2.48 s.
  Below 4096 rows a chunk's columns and fields are done on the calling
  thread, so the expression plugin's small `.over()` groups do not fan out
  for nothing. Every output is bit-identical; the golden, chunk-invariance
  and oracle suites are unchanged.
- **README: a *Chunk size* guide** under Parallelism — what `chunk_rows`
  does and does not change, and a sweep from 20k to 2M rows on interleaved
  and group-sorted data. The section's memory numbers are now peak
  footprint (`/usr/bin/time -l`), where they had been RSS with the
  memory-mapped input counted in. The README's measured numbers are
  regenerated on this build.
- **`benchmark.yml` keeps both tables in its artifact** (throughput and
  thread scaling), so two runs can be compared without the browser.

## [0.1.0] — 2026-09-03

First release.

### Models

Ten online regression models plus streaming moments, all on exponentially
weighted **mean-form** accumulators with centered (Welford) co-moments:
`ewridge`, `rls`, `lasso`, `kalman`, `huber`, `quantile`, `sgd`, `pa`, `ftrl`,
`holt`, and `ew_cov`.

### Interfaces

Three, with identical numerics: a Polars **expression plugin**
(`pl.col("y").online.ewridge(...)`; in-memory only, and it warns so since —
see *Changed* below), a chunk-fed **`ModelBank`** with O(state)
memory that reports what it holds (`groups()`, `rows_seen()`) and can forget
stale groups (`drop_groups()`), and a standalone **CLI** (parquet in, parquet
out, TOML config). The Python surface is typed: PEP 692 keywords on the
builders and the namespace, and `po.online(expr)` for type checkers, which
cannot see a registered namespace.

### Parallelism

One task per (spec × group) per chunk on the bank's own thread pool, sized
by **`POLARS_ONLINE_MAX_THREADS`** (unset: one per core; read when the pool
is built, at the first bank call; `po.thread_pool_size()` reports it).
Polars' readers and writers stay on polars' pool, `POLARS_MAX_THREADS`,
which also sizes what its reader holds in flight — so a run can keep polars
small for memory and give the bank every core, and the README's
*Parallelism* section measures why. A value that is not a count is refused
by name. Thread count changes speed and nothing else; a test runs the same
stream at 1 and 8 threads and requires identical output.

### Guarantees

- Predictions are out-of-sample by construction.
- Chunk invariance: 1 chunk or 1000 produces identical output, as does saving
  state mid-stream and resuming. (`coef` is a reporting cadence and excepted.)
- `n_eff` means the same thing in every model, which is what makes
  `min_periods` portable across a bank.

### Diagnostics

`emit_sigma`, `emit_resid_z`, `emit_drift` (Page-Hinkley), `emit_metrics`
(ic / r² / hit rate), `emit_autocorr`, `resid_quantiles` (P²),
`emit_selected` and `emit_averaged` for online model selection and averaging.

### Verified against [river](https://riverml.xyz)

FTRL's z/n recursion to 1e-12; Kalman ≡ `BayesianLinearRegression` to 3.6e-15;
`EwCov` ≡ river's Welford statistics exactly. Two documented places where the
libraries legitimately differ are pinned by tests rather than left as
surprises.

### Known limitations

- `polars>=1.34.0,<2`. The floor is measured (`LazyFrame.collect_batches`,
  which the streaming paths read with, arrived in py-polars 1.34.0; see
  *Changed* below); the ceiling is a bet that 1.x keeps the interface, hedged
  by a version-negotiated plugin ABI that refuses to load rather than
  misbehave and by a weekly canary against the latest polars. The Rust
  `polars` inside the wheel is pinned exactly, but it never meets the user's
  copy — data crosses on the Arrow C Data Interface. The README's
  *Versioning and the Polars pin* has the matrix.
- Requires Python 3.12+ (`abi3-py312`).
- Wheels for macOS (arm64, x86_64), Windows x64 and Linux (x64 glibc and
  musl, aarch64 glibc); anything else builds from the sdist with a Rust
  toolchain.

### Before the release

*The entries below were written as the code evolved, before anything was
published; they describe changes relative to earlier development snapshots,
not to a released version, and stay because they record why things are the
way they are.*

#### Added

- **`online.unnest(specs)`: a bank's output as flat columns, the
  coefficients named.** `lf.online.unnest(specs)`, `df.online.unnest(specs)`
  and `po.unnest(frame, specs)` take each spec's struct column apart in
  place — scalar fields under their own names, each `coef` list as one
  column per coefficient named on the field grammar (`coef_y_intercept`,
  `coef_y_x1__r0.5@h500` beside `pred_y__r0.5@h500`). `specs` may be the
  spec dicts, a `ModelBank`, or the path of a saved state; a parquet the
  CLI wrote reads back flat through `pl.scan_parquet(..).online.unnest(..)`.
  The names come from the new **`polars_online.spec.coef_fields(spec)`**:
  every coefficient with the `coef` field it sits in, its `position` there,
  its column `name`, and `target`, `halflife`/`lam`, `ridge`,
  `feature_set`, `lambda`, `term` — rendered by the same Rust code as the
  field names (`online_polars::coef_fields`, `CoefField`). `coef_index` is
  unchanged and is now derived from it.
- **A weekly native leak check in CI** (`.github/workflows/leakcheck.yml`,
  PLAN task 18): `scripts/leakcheck.sh` under `leaks` on macOS and valgrind
  on Linux, Mondays and on demand; nothing gates on it, a red scheduled run
  is the report. Wiring it showed the script's earlier "0 leaks" was a blind
  check — pymalloc's arenas are invisible to both tools — so it now runs
  under `PYTHONMALLOC=malloc`, counts differentially (1 iteration against
  1000) and has a control mode that leaks one object per iteration and must
  be caught; the job runs the control too. What it cannot see, and says so:
  memory from polars' allocator, i.e. the Rust side, which
  `tests/test_ffi_memory.py` covers by RSS.
- **An API reference, built from the docstrings with Sphinx** (`docs/reference/`;
  `uv run --group docs sphinx-build -W docs/reference docs/_build/html`), in
  the gate and CI with warnings as errors, and published to GitHub Pages
  from `main`: <https://hgilde.github.io/polars-online/>. The `docs`
  dependency group (`sphinx`, `furo`) is separate from `dev`. The first
  build found four docstrings that were not valid reStructuredText; fixed.
- **`ModelBank.coef(spec, group=None)`: the coefficients behind a fit, as a
  frame**, from a live bank or one loaded from a state file — one row per
  `(group, instance, position)` with `coef_index`'s `target`, grid values
  and `term`, so `bank.coef("ols").pivot("term", index=["group",
  "instance"], values="coef")` is the betas per group. The values are what
  the output's `coef` field reported on the last row each group learned
  from (the fit after that row, which the next `pred` is computed from);
  null before the group's first solve — the solve schedule's decision, not
  `min_periods`', which gates `pred` alone, so the frame carries `n_eff`
  for how much weight is behind each fit; an empty frame for a group the
  bank has never seen; `ValueError` for `ew_cov`. Rust: `Bank::coef` and
  the `Coef` row. Before this, the betas of a saved state were reachable
  only through `predict` on a one-row frame or a hand solve over `gram()`.
  The README's new *Saving, loading and reading a model* section shows
  save/load and the coefficients each in both the bank's form and the
  query's (`coef_index` + `list.to_struct` for the per-row path in polars).
- **`save_state=` on the plan: `lf.online.fit_predict(specs, load_state=,
  save_state=)`**, and on `df.online.fit_predict` and `po.fit_predict`. The
  state the execution ends in — after the last row the source fed the bank:
  the stream's end, or the `n` rows of a `head(n)` — is written to the path
  when the run ends, atomically (`ModelBank.save`), as the same bytes a bank
  fed those rows saves and `po.run(save_state=)` writes. `load_state` and
  `save_state` may be the same path, for a resume in place. The plan stays
  pure, and that is what makes the write safe: polars runs a plan's source
  once per use in a query — twice, on two threads, under a self-join,
  `pl.concat` or `pl.collect_all` of two sinks — and every run writes the
  same bytes. Nothing is written by a run the caller abandons or one the
  bank ended with an error; a node after the bank failing does not stop the
  bank, so the state is written then (`po.run` saves only after its output
  is committed). `docs/STATE-WORKFLOW.md` has the measurements and the
  rules.

- **`lf.online.fit_predict(specs)` — the bank as a polars source.** A
  `LazyFrame` in, a `LazyFrame` out: executing it (`collect`,
  `collect_batches`, `sink_parquet`, …) streams the plan's rows through a
  fresh `ModelBank` in `chunk_rows` chunks, so a query with the bank in it is
  O(chunk) in memory — where the expression plugin in the same query is
  O(data) in either engine, because polars calls a user expression once with
  its whole column. Bit-identical to `po.run`'s output; 12M rows in 2.8 s at
  0.78 GB live (the plugin: 14.4 s, 7.3 GB). Filters, selections and `head`
  after the bank are pushed into the source and honoured there — a filter
  after never changes what the bank learns from — and a selection reaches
  the input scan. The plan is pure: every run starts from the specs' state
  or `load_state` (and `save_state=`, below, writes where it ends). Also
  `lf.online.predict(bank)` to
  score against a bank or a state file, the eager twins
  `df.online.fit_predict(specs)` / `df.online.predict(bank)`, and
  `po.fit_predict(frame, …)` / `po.predict(frame, bank)` for type checkers.
  Rides on polars' IO-plugin interface (`register_io_source`), documented
  but marked unstable by polars (`docs/RELEASE-READINESS.md`).
- **`ModelBank.predict(df)`** scores a frame against the bank exactly as it
  stands and updates nothing: no clock advance, no decay, `n_eff` frozen. Row
  `i` carries what `fit_predict` would have reported had it been the next row
  of the stream — the same fields, the same values — every row scored from the
  same state, with the clock distance measured from the last row the bank
  learned from. The target column is optional (then `resid` is null), `weight`
  is not read, unknown groups score null, and the stream's session and clock
  policies still hold. Concurrent `predict` calls are fine; a `fit_predict`
  racing one is refused. Also from the runner, as `po.run(predict=True)`, the
  TOML key `predict = true`, and the CLI flag `--predict` — each needs a
  loaded state and refuses `save_state` (the keyword and the flag drop a
  config's own `save_state`, so one TOML serves both the learning and the
  scoring run). Roughly twice the throughput of `fit_predict`.
- **`OnlineModel::predict`** (Rust, `online-core`): the step without the
  step, implemented by every model and held to `predict == step` row by row
  in `tests/model_contract.rs`.
- **The runner reads and writes parquet, ipc, csv and ndjson**, told from
  the extension (`.parquet`/`.pq`, `.ipc`/`.arrow`/`.feather`, `.csv`,
  `.ndjson`/`.jsonl`) or named with `input_format=` / `output_format=` (TOML
  keys of the same names, CLI `--input-format` / `--output-format`). CSV
  cannot hold the bank's struct columns, so there each spec is flattened to
  `<spec>.<field>` columns and `coef` is a JSON list;
  `pl.col("ridge.coef").str.json_decode(pl.List(pl.Float64))` reads it back
  bit-exact.
- **`po.run(input=...)` takes any source py-polars can stream**: a path
  (globs and cloud URLs as `pl.scan_*` takes them), a `LazyFrame` — any query,
  including one with a Python UDF — a `DataFrame`, or any iterable of
  `DataFrame`s in stream order. The reading is py-polars' own
  (`collect_batches`), so a CSV streams through the wheel's SIMD parser;
  frames handed in are taken as they come, `chunk_rows` chunking what polars
  reads. Also `keep_columns=` to select input columns before the bank (and
  before the scan reads them), and `progress(rows, chunks)`, called after
  each chunk; an exception raised in it or in the input iterator surfaces as
  itself and no output is published.
- **Rust:** `online_polars::run(bank, Input, Output, RunOptions, progress)`
  is the pipeline the CLI and `po.run` share — `Input::Lazy(LazyFrame)` or
  `Input::Batches { frames, schema }` in, `Output::File { path, format }` or
  `Output::Batches(callback)` out — with `run_config_on(cfg, Input, ..)` for
  a `RunConfig` over an input the caller already has, `Format`
  (`from_path`, `scan`, `name`, `ALL`) and `DEFAULT_CHUNK_ROWS`.
  The three stages now overlap and a plan is read in one streaming pass
  instead of a `slice().collect()` per chunk, so on parquet the same run is
  **1.6–2.7× faster than before** (`po.run`, 3M rows: 1.93 → 0.72 s with
  groups interleaved, 3.30 → 2.12 s group-sorted; the CLI 2.14 → 0.83 s and
  3.84 → 2.55 s). The extension grows 4% for the three extra formats
  (gzipped 18.8 → 19.8 MB; wheel 19.8 → 20.8 MB; CLI 51 → 53 MB); no new
  dependency outside polars. Measured in `docs/PERFORMANCE.md` §10.

#### Changed

- **The bank's thread pool is its own, sized by `POLARS_ONLINE_MAX_THREADS`**
  (`crates/online-polars/src/pool.rs`), where it used to be rayon's global
  pool and `RAYON_NUM_THREADS` — a name that said nothing about which pool
  it was next to `POLARS_MAX_THREADS`. `RAYON_NUM_THREADS` now reaches
  nothing here: the per-core default is spelled out rather than left to
  rayon, and `tests/test_portability.py` checks that neither it nor
  `POLARS_MAX_THREADS` sizes the bank's pool. The runner's parquet page
  encoding and NDJSON serialization moved the other way, onto polars' pool
  (`polars_core::runtime::THREAD_POOL`, already in the tree), so that
  `POLARS_MAX_THREADS` is polars' readers *and* writers in every form.
  `po.thread_pool_size()` is new, the mirror of `pl.thread_pool_size()`.
- **Every public entry point documents its failure modes, and they follow
  one contract** (`polars_online.__doc__` states it): a file problem is
  the `OSError` subclass for what went wrong, naming the path; a parameter,
  spec or column problem is `ValueError` naming the spec, parameter or
  column; the wrong kind of object is `TypeError`; a name or position that
  is not there is `KeyError`/`IndexError`; a bank used from two threads is
  `RuntimeError`; inside a plan, a run-time error is polars'
  `ComputeError` carrying that message. Rust: `# Errors` on `Bank::new`,
  `fit_predict`, `predict`, `run_config`, `run_config_on` and `run`;
  `cargo doc` builds without a warning. What did not fit the contract
  was changed to:
  - **An unknown key in a spec, a `po.run` config or a CLI TOML is
    refused**, naming the keys there are (the CLI with the line), where a
    misspelt `halflfe` used to fall silently back to the default.
    `Spec`, `ModelKind` and `RunConfig` are `deny_unknown_fields`.
  - `ModelBank.load` raises `FileNotFoundError` (or the `OSError` subclass
    the platform gives, e.g. `PermissionError` for a directory on Windows)
    for a path it cannot read and `ValueError` for a file that is not a bank, a
    newer build's file (now told from garbage by its envelope), or a spec
    mismatch — it raised `OSError` for all of them.
  - `po.run` raises the `OSError` subclass for an unreadable `load_state`,
    an unwritable output or `save_state`, each naming the path, and checks
    `save_state`'s directory *before* the run, so a typo there no longer
    costs the run and the state. `config=` that is not a dict, a path or
    `None` is `TypeError`; `chunk_rows < 1` is `ValueError` on every surface.
  - A spec position out of range (`ModelBank.gram(3)`, `groups(3)`, …) is
    `IndexError`, not `ValueError`; every bank method, not just
    `fit_predict`, gives the "bank is busy" `RuntimeError` when another
    thread holds it.
  - A chunk the bank refuses leaves no empty new groups behind: `groups()`
    lists only what the bank has learned from.
  - `eval.unpack` on a struct with no `pred_*` fields is `TypeError` naming
    the fields it found; `rolling_metrics(window=0)` is `ValueError`; a
    non-numeric `clock` there is `TypeError`.
  - Rust: `online_polars::Gram` is exported like the other bank types.
- **`load_state=` on the plan, and `predict(path)`, read the file when the
  plan is built**, not each time it runs: the plan carries the state, as
  `df.lazy()` carries a frame, so a plan collected twice gives the same
  frame whatever happened to the file in between, and `load_state=p,
  save_state=p` used twice in one query cannot race one run's load against
  the other's write. Build the plan again to pick up a newer file.
  `predict(bank_object)` still scores the bank as it stands when the plan
  runs.
- **`head(n)` on the plan feeds the bank exactly `n` rows.** The source
  applied polars' pushed slice to its output and fed the bank the whole
  chunk the `n`th row fell in; it now trims the input chunk, so the state
  after a `head(n)` is the state after `n` rows. The numbers are unchanged
  except `coef`, reported on each chunk's last row, which now lands on the
  `n`th row.

- **The expression form warns on every use** (`docs/PLAN.md` §6). Each
  `pl.col("y").online.<model>(...)` call now issues
  `polars_online.InMemoryExpressionWarning`, new and exported: polars hands
  a stateful user expression its whole column in either engine, so that
  form is O(data) — 7.3 GB at 12M rows against 1.35 GB for
  `lf.online.fit_predict([spec])` in the same query — and a reader who took
  it for the streaming form learned otherwise from a memory profile. The
  warning says why, names the plan to write instead, and gives the one-line
  filter for a frame in memory on purpose; it is a `UserWarning` because a
  `DeprecationWarning` is hidden outside `__main__`, which is exactly the
  pipeline module where it matters. The README shows the two forms side
  by side in a closing note. Nothing else moves: the expression still runs,
  `po.online` is still exported, and the numbers are the same bits.
- **`polars>=1.34.0,<2`** (was `>=1.28.1`). `po.run` over a path or a plan
  has read with `LazyFrame.collect_batches` since the runner became
  format-agnostic, and so does `lf.online.fit_predict`; py-polars added it
  in 1.34.0, and on 1.28.1–1.33 those calls failed with an `AttributeError`
  the latest-only canary could not see. The whole suite passes on 1.34.0,
  1.38.1 and 1.44.1 with identical numbers; `ModelBank` and the expression
  plugin alone still work from 1.28.1.
- **`lasso`'s `lam_selected_<target>`** is reported as it stood *before* the
  row — the λ the row was scored with — rather than after the row's error
  joined the selection. A one-row shift in that column, which makes it
  identical between `fit_predict` and `predict`.
- **`Step.coef`** (Rust) is gone: it was never `Some`; coefficients are read
  through `coefficients()`.
- The busy-bank message now says the bank "is in use on another thread" and
  that concurrent `predict` calls are fine.
- **Rust runner API:** `run_lazy` is `run` and takes an `Input`; the progress
  closure returns `PolarsResult<()>` (an `Err` stops the run) instead of
  `()`; the native module's `run_config` is `run_config_frames` (the Python
  side reads, Rust fits and writes). `po.run`'s signature and TOML keys are
  unchanged, with the new keywords optional.
- **`rls` is 20% faster at k=20 and 57% faster at k=50** (model arithmetic;
  1.63M → 1.93M rows/s through the bank at k=20). The per-row
  back-substitution summed each row in the one order that serialized it on
  the coefficient just solved; it now sums from the far end. A summation
  order is a rounding-level change: the golden signatures moved by at most
  1.2e-15, inside their 1e-12 tolerance.

#### Documented

- **The docs say what the bank is for a table in any row order.** It was
  introduced as "built for ordered event data"; row order reaches a bank's
  fit only through decay, and with decay off (`halflife=inf`, `lam=1.0`)
  `ewridge` is least squares over every row seen — `numpy.linalg.lstsq` to
  2e-13 in any order, 1.4 GB against `lstsq`'s 3.97 GB at 6M rows × 20
  features from parquet. The README leads with both shapes and has an "Any
  row order" section; `tests/test_row_order.py` pins the claim. Documented
  with it, and pinned, a trap that is not fixed: a huge *finite* halflife
  inherits the `halflife/50` solve cadence and so solves once and stays
  there — `inf` is the no-decay setting and `solve_every` the throttle.
- **How to score without learning**: give the rows weight `0`, which freezes
  the coefficients bit for bit, rather than nulling the target, which does
  not. The README also states the cost — a zero-weight row still advances the
  clock, so `n_eff` decays while scoring and `min_periods` can blank the
  output.
- **An upstream `filter` costs a bounded window, not the data**
  (`docs/PERFORMANCE.md` §11). The plan form reads its input with polars'
  `collect_batches`; the streaming engine bounds what is in flight in
  morsels per thread, and for a predicate pushed into a parquet scan the
  morsels are whole row groups of what the filter keeps: the reader applies
  the predicate itself and the scan then restores the column order through
  a 7-slot-per-thread pipeline — 2.5 GB at 12M rows and 3.1 GB at 36M on
  14 threads when the filter keeps every row, 1.2 GB when it keeps half,
  0.7 GB on 2 threads, 0.38 GB with the predicate column first in the
  projection (the stage becomes a no-op; an accident, not a recipe) —
  against 0.65 GB with nothing upstream and 0.78 GB for the same filter
  written *after* the bank, where it is pushed into the source and applied
  per chunk. Filter after unless the model should skip those rows, and then
  give them weight 0 through `when/then/otherwise` (1.3 GB; 1.1 with
  `pl.Config.set_streaming_chunk_size(25_000)`, which shrinks any
  `with_columns` window, and the filter's only once such a node above the
  scan makes the reader split its row groups) rather than filtering.
  `.over()` and `sort` upstream are O(data), and `sink_batches` with the
  default engine collects its input on polars 1.x (`engine="streaming"`
  streams; 2.0 makes it the default). The
  engine's own map of which is which:
  `lf.show_graph(engine="streaming", plan_stage="physical")`. A filter run
  inside the source would be 0.81 GB with identical output; not added — a
  second `filter` differing only in memory, for a cost that is
  polars' column-reorder stage to remove.
- **Which surface is O(data)**, measured (`docs/PERFORMANCE.md` §11): the
  expression plugin — 2.0 GB at 3M rows, 7.3 GB at 12M, in either engine,
  because polars' streaming engine collects the input of any user
  expression before calling it. The bank, `po.run` and the CLI are flat at
  ~0.75 GB from 3M to 12M rows, nearly all of it polars' parquet
  read-ahead; `POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes the CLI to 0.15 GB at
  the same speed.
- **The README is written to be read.** It opens with a summary of what the
  package does — the model table, the ways to run a bank, the stream
  semantics, the two guarantees, state as a file, diagnostics — and then
  keeps like with like: the loop, the query, the job and the expression form
  under one heading, the stream parameters under another, the coefficient
  and field-name material together. The facts are the previous README's;
  measurements stay where they decide something and otherwise point at
  `docs/`. Every python block still runs under the README harness in
  `tests/test_production_hardening.py`.

#### Fixed

- **Two writers of one state file in one process no longer share a
  temporary.** `atomic.rs` named its temporary sibling by pid alone, so two
  threads saving the same path at the same moment — `ModelBank.save` from
  two threads, or now a plan with `save_state=` used twice in one query —
  created and wrote *the same* temporary and the rename published a
  mixture. The name now carries a process-wide sequence number, so the
  destination is always the old file or one writer's whole file; the
  runner's output file is written the same way and is covered too. Held by
  a two-thread, fifty-round test that fails under the old name.

- **The runner (`po.run` and the CLI) no longer panics on a parquet with
  more than one row group.** A chunk that spanned a row-group boundary
  arrived as a multi-chunk frame, the bank's outputs are single-chunk, and
  the batched parquet writer handed arrow a record batch of mismatched arrays
  (`RecordBatch requires all its arrays to have an equal number of rows`).
  Polars writes 262,144-row groups by default, so any file longer than
  `chunk_rows` was affected; the tests had written every input in one row
  group. The chunks are now aligned before writing.

- **`coef` is null, never an empty list, before a model's first solve.** Rows
  between `coef_every` snapshots were already null; warmup rows were an empty
  list, which made `coef.list.get(position)` — the documented way to read one
  coefficient — raise "index out of bounds" instead of returning null.
- **`holt` accepts `level_halflife` on its own.** For Holt the level halflife
  is the spec's halflife — one knob under two names — but a spec that gave
  only `level_halflife` was refused with "one of halflife/lam is required",
  including the example in this project's own README.
- **Every polars dtype can now cross into the model bank.** A `Decimal` or
  `Int128` column *anywhere* in the frame — even one no spec named — aborted
  the process with `activate 'dtype-decimal' feature`, and `Int8`/`UInt8`/
  `Array` columns failed with a polars error naming neither the column nor the
  fix. The missing dtype features are enabled, so unused columns are carried
  through whatever they are, and narrow numeric columns (`UInt8`, `Decimal`,
  …) are usable as features, cast to `f64` and bit-identical to the `Float64`
  columns they came from. The extension grows 7% (gzipped 17.6 → 18.9 MB); no
  new dependency.

- **State and output files are written atomically** — a temporary sibling,
  then a rename into place. `ModelBank.save` used to truncate the destination
  and write into it, so an interrupted save (a kill, a full disk, a quota)
  left a truncated file *and* destroyed the last good state; a `--resume` loop
  then started the stream over. The CLI's output parquet is published the same
  way, so a run that fails halfway leaves the previous output intact instead
  of a headless parquet under its name. Saving now costs a filesystem sync
  (~4 ms on macOS, where `sync_all` is `F_FULLFSYNC`); save less often if that
  matters more than surviving a crash.
