# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/) — while pre-1.0, the minor version
carries breaking changes.

## [Unreleased]

### Added

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
  or `load_state`, and nothing is saved. Also `lf.online.predict(bank)` to
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

### Changed

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

### Documented

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
  a 6-slot-per-thread pipeline — 2.5 GB at 12M rows and 3.1 GB at 36M on
  14 threads when the filter keeps every row, 1.2 GB when it keeps half,
  0.7 GB on 2 threads, 0.38 GB with the predicate column first in the
  projection (the stage becomes a no-op; an accident, not a recipe) —
  against 0.65 GB with nothing upstream and 0.78 GB for the same filter
  written *after* the bank, where it is pushed into the source and applied
  per chunk. Filter after unless the model should skip those rows, and then
  give them weight 0 through `when/then/otherwise` (1.3 GB; 1.1 with
  `pl.Config.set_streaming_chunk_size(25_000)`, which shrinks any
  `with_columns` window and not the filter's) rather than filtering.
  `.over()` and `sort` upstream are O(data), and `sink_batches` with the
  default engine collects its input (`engine="streaming"` streams). The
  engine's own map of which is which:
  `lf.show_graph(engine="streaming", plan_stage="physical")`. A filter run
  inside the source would be 0.81 GB with identical output; not added — a
  second spelling of `filter` differing only in memory, for a cost that is
  polars' column-reorder stage to remove.
- **Which surface is O(data)**, measured (`docs/PERFORMANCE.md` §11): the
  expression plugin — 2.0 GB at 3M rows, 7.3 GB at 12M, in either engine,
  because polars' streaming engine collects the input of any user
  expression before calling it. The bank, `po.run` and the CLI are flat at
  ~0.75 GB from 3M to 12M rows, nearly all of it polars' parquet
  read-ahead; `POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes the CLI to 0.15 GB at
  the same speed.

### Fixed

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
  is the spec's halflife — one knob spelled two ways — but a spec that gave
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

## [0.1.0] — unreleased

First release.

### Models

Ten online regression models plus streaming moments, all on exponentially
weighted **mean-form** accumulators with centered (Welford) co-moments:
`ewridge`, `rls`, `lasso`, `kalman`, `huber`, `quantile`, `sgd`, `pa`, `ftrl`,
`holt`, and `ew_cov`.

### Interfaces

Three, with identical numerics: a Polars **expression plugin**
(`pl.col("y").online.ewridge(...)`), a chunk-fed **`ModelBank`** with O(state)
memory that reports what it holds (`groups()`, `rows_seen()`) and can forget
stale groups (`drop_groups()`), and a standalone **CLI** (parquet in, parquet
out, TOML config). The Python surface is typed: PEP 692 keywords on the
builders and the namespace, and `po.online(expr)` for type checkers, which
cannot see a registered namespace.

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

- `polars` is pinned exactly (see the README's *Version pins*). The pyo3-polars
  plugin ABI is negotiated and a mismatch produces a clear error, but the pin
  means this package cannot currently coexist with a different polars.
- Requires Python 3.12+ (`abi3-py312`).
