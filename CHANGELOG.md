# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/) — while pre-1.0, the minor version
carries breaking changes.

## [Unreleased]

### Fixed

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
