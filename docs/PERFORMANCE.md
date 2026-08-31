# Performance: measurements and the parallelism plan

Status as of 2026-08-31: baseline measured, plan defined, nothing below
implemented yet. Tick tasks here as they land (CLAUDE.md rule 7 applies — one
commit per task, `P<n>` in the message).

Machine for every number in this file: Apple M-series, 10 performance + 4
efficiency cores, release build (thin LTO, `codegen-units = 1`), single
process, `ONLINE_TIMING=1` for the section rows. Regenerate the raw numbers
with `cargo run --release -p online-core --example core_bench` and
`ONLINE_TIMING=1 uv run python scripts/benchmark.py`.

## 1. The measured baseline

**The pure core** (`EwRidge::step` in a loop, no Polars anywhere —
`core_bench`):

| configuration | rows/s |
|---|---|
| k=5, 1 target | 11,804,355 |
| k=20, 1 target | 5,734,052 |
| k=50, 1 target | 1,987,873 |
| k=20, 10 targets | 3,968,505 |
| k=20, solve **every** row | 468,455 |
| k=20, solve every 25 rows | 3,941,742 |

**The bank on the same arithmetic** (200k rows, ewridge, clock column,
`ONLINE_TIMING` sections in ms):

| case | extract | group | process | assemble | total | rows/s |
|---|---|---|---|---|---|---|
| k=5, 1 group | 3.8 | 0.1 | 87.6 | 4.1 | 95.5 | 2,093,719 |
| k=20, 1 group | 12.2 | 0.2 | 133.6 | 6.9 | 152.8 | 1,308,792 |
| k=20, 8 groups | 12.2 | 14.8 | 46.3 | 8.8 | 82.3 | 2,431,070 |
| k=20, 64 groups | 13.0 | 15.3 | 40.6 | 8.0 | 76.9 | 2,601,719 |

**Thread scaling**, 400k rows, k=20, 64 independent groups (`RAYON_NUM_THREADS`):

| threads | rows/s | speedup |
|---|---|---|
| 1 | 435,792 | 1.0× |
| 2 | 702,049 | 1.6× |
| 4 | 1,021,413 | 2.3× |
| 8 | 1,363,018 | 3.1× |
| 10 | 1,414,097 | **3.2×** |

**The expression API**, 400k rows, k=20:

| path | rows/s |
|---|---|
| bank, 1000 groups | 1,522,227 |
| `.online.ewridge(...).over("g")`, 1000 groups | 511,237 |
| expression, single stream | 1,022,170 |

## 2. What the numbers say

1. **The integration layer costs 3–5× the model itself.** k=20 single stream:
   the core needs ~35 ms for 200k rows; the bank's `process` section takes
   133.6 ms. The difference is per-row heap traffic: two `Vec` allocations for
   `x`/`y` per row, a `RowOut` with ~11 `Vec` fields per row, plus per-model
   `resid`/`sigma`/`resid_z` vectors — ~10–20 allocations per row, ~25M/s at
   these speeds.

2. **The same traffic is why threads don't scale.** 64 independent streams
   reach 3.2× on 10 performance cores; embarrassingly parallel work stalling at
   3× is the signature of allocator contention and memory bandwidth, not of
   compute. Fixing (1) is most of fixing this.

3. **The serial sections put an Amdahl cap on top.** `extract` (12 ms),
   `group` (15 ms) and `assemble` (8 ms) are all single-threaded; at 64 groups
   they are already 45% of wall. `group` casts the key column to String and
   clones a `String` per row into a HashMap. `extract` materializes
   `Vec<Option<f64>>` — 16 bytes and a branch per value — even for columns with
   no nulls.

4. **Structural serialism the benchmarks don't show.** Specs are processed one
   after another (parallelism is only across groups *within* a spec), and the
   model instances of a grid are stepped serially inside one stream: a
   5-halflife grid on a single stream runs at 292,573 rows/s with 9 cores idle,
   though the 5 instances are independent given the same rows.

5. **The expression path pays per group.** Under `.over()` the plugin is
   invoked once per group, and each invocation parses the spec JSON, builds a
   `Bank`, and re-extracts its columns: 3× slower than the bank on identical
   work at 1000 groups.

6. **Solving is already amortized.** Solve-every-row is a 12× hit
   (5.7M → 468k); the default `solve_every = halflife/50` sits within 1.5× of
   never-solve. No work needed beyond not regressing it.

7. **Within one model instance, one stream, the recursion is sequential by
   construction** — state at row *i* depends on row *i−1*. There is no
   parallelism to extract there, and nothing below attempts it.

## 3. The plan

Ordered by measured impact per unit of risk. Every task keeps the two
guarantees (out-of-sample, chunk invariance) bit-for-bit — the golden tests
(`golden.rs`, `test_golden_pipeline.py`) are the regression net, and any task
that moves a golden number is wrong by definition.

- [x] **P1 — Columnar hot path.** *Done.* `RowOut` (a struct of ~11 `Vec`s per
  row) is replaced by `ChunkOut`: flat slot-major `Vec<f64>` buffers, one
  allocation per output column per (stream, chunk), with `processed` marking
  skipped rows and NaN meaning null. `Stream::process_chunk` owns the row loop
  and reuses scratch (`xs`, `ys`, `r_buf`, `sig_buf`, `zs_buf`, and `pred_buf`
  catching the `Vec` each `Step` hands back), so the loop allocates nothing.
  Measured: **k=5 single stream 2.09M → 5.12M rows/s, k=20 1.31M → 2.40M,
  k=20/64 groups 2.60M → 3.32M**; the `process` section fell 87.6 → 30.7 ms
  (k=5) and 133.6 → 67.1 ms (k=20). Thread scaling went **3.2× → 4.8×** at 10
  threads (628k → 2.99M rows/s). Short of the ≥3.5M and ≥7× targets, and the
  reason is now visible in the sections: at 64 groups the serial `extract`
  (11.6 ms) and `group` (15.2 ms) are 44% of wall, so P3 is what unlocks the
  rest. Every golden number unchanged.
  <details><summary>original plan</summary> Replace per-row `RowOut` with
  `Stream::process_chunk`: the stream walks its row indices writing directly
  into preallocated flat output buffers (one `Vec<f64>` per output slot per
  chunk, NaN as null, bitmaps built at assembly). Reuse one scratch `xs`/`ys`
  buffer per stream. This deletes the per-row allocations of (2) and most of
  `assemble`'s scatter. Target: k=20 single stream ≥ 3.5M rows/s (from 1.31M);
  grouped scaling ≥ 7× at 10 threads (from 3.2×).</details>
- [ ] **P2 — One flat task pool.** Fan out over (spec × group × model-instance)
  in a single `par_iter`, not per-spec loops: a bank of N single-group specs
  currently uses one core; a grid on one stream uses one core. Instances need
  their per-instance diagnostics (`resid_var`, `drift`, …) split into a
  per-instance struct so rayon tasks own disjoint `&mut` — mechanical, the
  indexing is already `[mi]`-major. Target: 5-halflife single stream ≥ 4×
  itself; N-spec banks scale with specs.
- [x] **P3 — Extraction and grouping without materialization.** *Done.*
  Columns extract to plain `Vec<f64>` with **NaN for null** instead of
  `Vec<Option<f64>>` — half the bytes, no per-value branch, and a `memcpy` via
  `cont_slice()` for a null-free contiguous column. Sound because every
  consumer already collapsed the two (a feature or weight counts only when
  `is_finite`, a target only when finite); the clock is the one column where
  null is an error, and that check now catches NaN with it. Group keys are
  bucketed by a 64-bit hash of the value, so a `String` is allocated once per
  distinct group rather than cloned three times per row. Extract and group run
  per spec in parallel. Measured: **extract 11.6 → 2.1 ms, group 15.2 → 4.6 ms**
  (at k=20/64 groups), taking that case **3.32M → 5.49M rows/s** and thread
  scaling **4.8× → 6.3×**. Just short of the ≤4 ms target; what is left is real
  work (the cast and the copy), not overhead.
  <details><summary>original plan</summary> Borrow value
  slices + validity bitmaps from the (rechunked) columns instead of building
  `Vec<Option<f64>>`; null-free fast path is a borrow, not a copy. Group and
  session keys: hash the physical values row-wise (as `session_hash` now does)
  — no String cast, no per-row `String` clone, and run it per spec in parallel
  with extraction. Target: extract+group ≤ 4 ms at k=20/64 groups (from 27 ms).</details>
- [ ] **P4 — Assembly into typed builders.** Write `Float64Chunked` from
  `Vec<f64>` + computed validity instead of `Vec<Option<f64>>` series; assemble
  specs in parallel. Mostly falls out of P1's flat buffers.
- [ ] **P5 — Expression path parity.** Thread-local cache of parsed
  `Spec`/plan keyed by the kwargs JSON, so 1000 `.over()` groups parse once;
  skip re-validation per group. Re-measure after P1 — the remaining gap should
  be per-group extraction only. Target: within 1.3× of the bank at 1000 groups.
- [ ] **P6 — Runner pipelining.** `run_config` currently alternates
  read-chunk / compute-chunk. Double-buffer with a bounded channel (read row
  group *n+1* while computing *n*); parquet decode is already internally
  parallel, so this overlaps the two pools. Target: CLI wall time ≤
  max(io, compute) + ε on a large file.
- [ ] **P7 — Build flags, measured not assumed.** Try `lto = "fat"` and (CLI
  and local dev only, never wheels) `-C target-cpu=native`; keep each only if
  ≥ 3% on `core_bench`. Verify the k-loops in `ewcov::update` auto-vectorize
  (`cargo asm` spot check) before considering any manual SIMD — at k ≤ 50 the
  compiler usually already does this.
- [ ] **P8 — Re-baseline and lock.** Re-run `core_bench`, the timing matrix,
  scaling and `scripts/benchmark.py`; update this file and the README table;
  extend `benchmark.yml`'s job summary with the scaling row so the CI history
  carries it. Golden tests must be untouched throughout.

**Rejected, with reasons** — so the omission is a decision (the ENHANCEMENTS
§4 convention): custom global allocator in the Python extension (Python and
polars own that arena; revisit only if post-P1 profiles still show allocator
time — the CLI could adopt mimalloc independently); GPU/BLAS batching (k ≤ 50
solves are too small to amortize a dispatch); speculative/parallel prefix tricks
for the recursion itself (breaks exactness, see (7)).

## 4. Bugs found by this review

- **Null-session sentinel collision** (fixed alongside this document): null
  session values were hashed as the string `"\0<null>"`, so a session literally
  named that was *the same session* as null — the T-E2 bug one layer down.
  Null now hashes to a value no string can produce (colliding strings are
  nudged), state files resume unchanged, and
  `test_a_session_named_like_the_null_sentinel_is_not_null` pins it.
- Two 64-bit hash-collision residues remain by design and are recorded here:
  two distinct session names, or one name against null, collide with
  probability ~2⁻⁶⁴ per pair (FNV-1a). Accepted; a collision merges two
  sessions, it does not corrupt state.
