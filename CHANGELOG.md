# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/) — while pre-1.0, the minor version
carries breaking changes.

## [Unreleased]

Nothing yet.

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
out, TOML config).

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
