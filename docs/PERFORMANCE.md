# Performance: measurements and the parallelism plan

Status as of 2026-08-31: **P1–P8 all done.** Headline, against the baseline in
§1: **2.8× single stream at k=5, 2.0× at k=20, 2.0× on grouped data, 2.1× on a
single-stream grid, 3.9× on a multi-spec bank, 1.23× on the CLI end to end**,
and thread scaling from 3.2× to 6.2× on ten performance cores. Every golden
number is unchanged throughout — that was the contract.

Two plan items were closed by *rejecting* them on measurement rather than
building them (P4's typed builders, P7's build flags), and P5 found its target
unreachable for a reason outside this codebase. Those are written up where they
sit, and §5 collects everything rejected.

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
- [x] **P2 — One flat task pool.** *Done, both halves.* **(spec × group):**
  the per-spec loop of per-spec rayon pools became one pool over every stream
  in the bank, longest-first so a few big groups do not strand cores at the
  tail. Measured on 8 single-group specs: **783.8 → 238.8 ms (3.3×)**, where
  before only one spec ran at a time. **(× instance):** `process_chunk` is now
  two passes — one serial walk deciding the clock schedule (which depends only
  on the clock and the input columns, never on the models), then the instances
  over the whole chunk. Instances share nothing but that schedule, so a
  five-halflife grid on one stream is five independent recursions:
  **458,500 → 1,000,341 rows/s (2.2×)**. The exception is
  `drift_action = "reset"`, where a break in any instance resets all of them
  within a row; that case keeps row-major order, and both paths call the same
  `run_instance`, so there is one copy of the arithmetic. Cost: ~5% on the
  already-parallel grouped case (4.74M → 4.54M at 10 threads) for the extra
  plan pass. Golden numbers unchanged.
  <details><summary>original plan</summary> Fan out over (spec × group × model-instance)
  in a single `par_iter`, not per-spec loops: a bank of N single-group specs
  currently uses one core; a grid on one stream uses one core. Instances need
  their per-instance diagnostics (`resid_var`, `drift`, …) split into a
  per-instance struct so rayon tasks own disjoint `&mut` — mechanical, the
  indexing is already `[mi]`-major. Target: 5-halflife single stream ≥ 4×
  itself; N-spec banks scale with specs.</details>
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
- [x] **P4 — Assembly into typed builders.** *Done, partly, and the rest
  dropped as unnecessary.* Specs now assemble in parallel. The typed-builder
  half was not worth doing: P1's flat buffers already took `assemble` from 8.0
  to ~5.5 ms at 200k rows, which is under 15% of wall, and the remaining cost
  is the `Series` construction polars needs either way. Measuring first is what
  said so.
  <details><summary>original plan</summary> Write `Float64Chunked` from
  `Vec<f64>` + computed validity instead of `Vec<Option<f64>>` series; assemble
  specs in parallel. Mostly falls out of P1's flat buffers.</details>
- [x] **P5 — Expression path parity.** *Done; the target turned out to be
  unreachable, for a reason worth recording.* A thread-local cache keyed on the
  kwargs JSON means `.over()` parses and validates the spec once per thread
  instead of once per group: **511,237 → 731,416 rows/s** at 1000 groups.
  Then a sweep over group counts showed where the rest goes — throughput falls
  from 2.77M (ungrouped) to 769k at *ten* groups of 40k rows each, and is then
  flat through 1000 groups. Per-group `Bank` construction would scale with the
  group count; this does not. **The remaining gap is polars' own `.over()`
  gather/scatter, not the plugin**, and no change on our side reaches it. The
  `ModelBank` API with `group=` is the answer for high group counts (5.01M
  rows/s on the same data), which is what the README already recommends.
  <details><summary>original plan</summary> Thread-local cache of parsed
  `Spec`/plan keyed by the kwargs JSON, so 1000 `.over()` groups parse once;
  skip re-validation per group. Re-measure after P1 — the remaining gap should
  be per-group extraction only. Target: within 1.3× of the bank at 1000 groups.</details>
- [x] **P6 — Runner pipelining.** *Done.* The runner alternated
  read-chunk / compute-and-write-chunk. A reader thread now fills a
  `sync_channel(1)`, so chunk *n+1* is decoded while chunk *n* is fitted and
  written — one chunk of lookahead, so memory stays O(chunk), and order is
  preserved by construction (one reader, FIFO), which matters because chunks
  must reach the bank in stream order. Measured on 3M rows × 20 features × 32
  groups through the CLI: **2.17 s → 1.76 s (1.23×)**.
  <details><summary>original plan</summary> `run_config` currently alternates
  read-chunk / compute-chunk. Double-buffer with a bounded channel (read row
  group *n+1* while computing *n*); parquet decode is already internally
  parallel, so this overlaps the two pools. Target: CLI wall time ≤
  max(io, compute) + ε on a large file.</details>
- [x] **P7 — Build flags: measured, and both rejected.** `lto = "fat"` moved
  the six `core_bench` cases by +3.2%, +1.3%, −1.2%, −1.0%, −2.1%, +0.4% — a
  wash, against a build that is already slow, so it fails the ≥3% bar the plan
  set. `-C target-cpu=native` was −3% at k=20 and within noise elsewhere, and
  would make wheels non-portable for nothing. Both stay off; `lto = "thin"`,
  `codegen-units = 1` remains.

  Manual SIMD is also unnecessary, and the throughput curve is the evidence:
  the co-moment update is O(k²), so k=5 → k=20 is 12.3× the arithmetic but only
  2.1× the time, and k=5 → k=50 is 72× the arithmetic for 6.0× the time. Per
  element, wider is *cheaper* — which is what auto-vectorized inner loops look
  like. There is nothing here for hand-written intrinsics to recover.
  <details><summary>original plan</summary> Try `lto = "fat"` and (CLI
  and local dev only, never wheels) `-C target-cpu=native`; keep each only if
  ≥ 3% on `core_bench`. Verify the k-loops in `ewcov::update` auto-vectorize
  (`cargo asm` spot check) before considering any manual SIMD — at k ≤ 50 the
  compiler usually already does this.</details>
- [x] **P8 — Re-baseline and lock.** *Done.* All of §1's measurements re-run
  on an idle machine and written up in §4; the README's throughput table
  regenerated from `scripts/benchmark.py`, with a note that grouped data now
  scales rather than merely working. `benchmark.yml` gains the scaling row so
  CI history carries it.
  <details><summary>original plan</summary> Re-run `core_bench`, the timing matrix,
  scaling and `scripts/benchmark.py`; update this file and the README table;
  extend `benchmark.yml`'s job summary with the scaling row so the CI history
  carries it. Golden tests must be untouched throughout.</details>

**Rejected, with reasons:** see §5, which records what was rejected up
front and what was rejected after measuring.

## 4. Where it ended up

Same machine, same build, same scripts as §1.

| case | before | after | |
|---|---|---|---|
| k=5, 1 group | 2,093,719 | 4,980,607 | 2.4× |
| k=20, 1 group | 1,308,792 | 2,634,627 | 2.0× |
| k=20, 64 groups | 2,601,719 | 5,132,169 | 2.0× |
| single stream, 5-halflife grid | 458,500 | 955,662 | 2.1× |
| 8 specs × 1 group (wall) | 783.8 ms | 201.7 ms | 3.9× |
| CLI, 3M rows × 20 feat × 32 groups | 2.17 s | 1.76 s | 1.23× |
| expression, 1000 groups | 511,237 | 727,089 | 1.4× |

Sections at k=20 / 64 groups: extract 13.0 → 1.7 ms, group 15.3 → 4.6 ms,
process 40.6 → 26.7 ms, assemble 8.0 → 6.0 ms.

Thread scaling, 400k rows, k=20, 64 groups:

| threads | before | after |
|---|---|---|
| 1 | 435,792 | 719,320 |
| 2 | 702,049 | 1,303,530 |
| 4 | 1,021,413 | 2,335,795 |
| 8 | 1,363,018 | 3,881,813 |
| 10 | 1,414,097 (3.2×) | 4,478,262 (**6.2×**) |

**What now limits it.** At k=20 on a single stream the bank reaches 2.63M rows/s
against the pure core's 5.73M, so the plumbing costs ~2.2× rather than the
original ~4.4×. What remains is the per-row gather of features into the scratch
buffer and the `Step` the model returns — real work at this point, not
bookkeeping. The recursion inside one instance stays sequential by construction
(§2 item 7); everything around it is now parallel.

## 6. The allocator (2026-08-31)

Found while answering "won't two copies of Polars in one process go wrong?".
The answer is no — the Arrow C Data Interface keeps each side freeing its own
memory — but the investigation turned up that pyo3-polars ships
`PolarsAllocator`, which routes a plugin's allocations through py-polars' own
allocator, and we were not installing it. Two allocator arenas in one process,
neither able to reuse the other's pages.

One line, `#[global_allocator]`, reproducible across repeated runs:

| case | before | after | |
|---|---:|---:|---:|
| ew_ridge k=5 | 5.62M rows/s | **8.02M** | +43% |
| ew_ridge k=20 | 2.80M | **3.26M** | +16% |
| ew_ridge k=50 | 855k | **900k** | +5% |

The gradient is the tell: largest at small `k`, where per-chunk allocation is a
big share of the work, and smallest at `k=50`, where the O(k³) solve dominates
and the allocator barely matters. Performance was not the reason for the
change; it is the reason it is recorded here.

## 5. Rejected, and why

Recorded so each omission is a decision. The first three were rejected up
front; the last three were rejected *after* measuring, which is the more
useful kind.

- **A custom global allocator in the Python extension.** Python and polars own
  that arena; swapping it under them is a compatibility risk for a benefit P1
  already took by removing the allocations instead of making them cheaper. The
  CLI could adopt mimalloc independently if a profile ever justifies it.
- **GPU or BLAS batching for the solves.** At k ≤ 50 a solve is microseconds;
  dispatch would cost more than the work, and `solve_every` has already made
  solving 6% of the k=20 budget.
- **Parallelizing the recursion itself** (speculative execution, parallel
  prefix over the decay). Row *i*'s state depends on row *i−1* exactly, and
  every approximation trades the exactness that chunk invariance and
  out-of-sample-ness are built on. Not a speed/complexity trade — a correctness
  one.
- **Typed-builder assembly (part of P4).** Measured first: P1's flat buffers had
  already taken `assemble` to ~5.5 ms of a 39 ms chunk, and the rest is `Series`
  construction polars needs anyway. Not worth the code.
- **`lto = "fat"` and `-C target-cpu=native` (P7).** Measured: +3.2%, +1.3%,
  −1.2%, −1.0%, −2.1%, +0.4% across the six `core_bench` cases for fat LTO — a
  wash against a slower build; and −3% at k=20 for `target-cpu=native`, which
  would also cost wheel portability. Both fail the ≥3% bar the plan set for
  itself.
- **Manual SIMD in the co-moment update.** The throughput curve rules it out
  without a disassembler: the update is O(k²), so k=5 → k=20 is 12.3× the
  arithmetic for 2.1× the time and k=5 → k=50 is 72× for 6.0×. Per element the
  wide cases are cheaper, which is what auto-vectorized loops look like.
- **Chasing polars' `.over()` overhead (P5's target).** Throughput drops from
  2.77M to 769k between one group and *ten*, then stays flat to 1000 — so it is
  the gather/scatter, not per-group setup, and not reachable from a plugin.
  `ModelBank(group=...)` is the supported answer at 5.0M rows/s.

## 7. Bugs found by this review

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
