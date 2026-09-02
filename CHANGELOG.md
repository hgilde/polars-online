# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/) — while pre-1.0, the minor version
carries breaking changes.

## [Unreleased]

### Added

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

### Changed

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
