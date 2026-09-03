# Performance: measurements and the parallelism plan

Status as of 2026-09-02: **P1–P8 all done**, numbers refreshed in §8. Headline, against the baseline in
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
with `cargo run --release -p online-core --example core_bench` (and
`--example rls_bench` for the `rls` A/B in §8) and
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
  and reuses scratch (`xs`, `ys`, `r_buf`, `sig_buf`, `zs_buf`), so the loop
  itself allocates nothing. (The `pred` `Vec` inside each `Step` is still
  allocated per row — see §4; measured at 14 ns against a 95 ns non-solve
  step in docs/IMPROVEMENTS.md P2, and left alone.)
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
  **Revisited (docs/IMPROVEMENTS.md P1): the conclusion above was wrong.** The
  gap was not gather/scatter but polars evaluating a *multi-input* group-aware
  function group by group on one thread (`apply_multiple_group_aware` in
  polars-expr has no parallel branch; the single-input path does). Packing
  every input column into one struct moved the plugin onto the parallel path:
  **4.2M → 21.4M rows/s** at 1000 groups (k=5, 2M rows) and **0.94M → 6.4M**
  on the k=20 table above — at parity with, or ahead of, the bank.
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

One line, `#[global_allocator]`. Attributed by A/B/A on an otherwise identical
tree, because two changes had landed between the first two measurements and the
gain was too large to assign by assumption: **5,716,246 → 8,165,418 →
5,614,830** rows/s at k=5, bounding machine drift at ~2%.

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
- ~~**Chasing polars' `.over()` overhead (P5's target).**~~ Reopened and
  closed by docs/IMPROVEMENTS.md P1: the "flat from ten groups on" curve was
  the signature of serial per-group evaluation, and one packed struct input
  puts the plugin on polars' parallel path. See P5 above.

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

## 8. Refresh (2026-09-02), and what `rls` costs now

§4's table was the state after P1–P8; §6's allocator fix landed *after* it, and
the README's table predates both. Re-measured with the same commands
(`scripts/benchmark.py --markdown`, `scripts/scaling_bench.py`), two runs
agreeing within 2%:

| configuration | README claimed | measured now | |
|---|---:|---:|---:|
| ew_ridge k=5 | 5,823,285 | **8,961,460** | +54% |
| ew_ridge k=20 | 2,775,086 | **3,620,024** | +30% |
| ew_ridge k=50 | 846,000 | **960,926** | +14% |
| ew_ridge k=20, 10 targets | 1,391,097 | **1,906,032** | +37% |
| ew_ridge k=20, 5 halflives | 1,328,381 | **2,158,579** | +62% |
| rls k=20 | 3,131,992 | **1,629,801** (1,927,999 after the solve reorder below) | **−48%** |
| kalman k=20 | 1,345,525 | **1,661,270** | +23% |
| lasso k=20 | 1,503,706 | **1,878,137** | +25% |
| huber k=20 | 2,852,545 | **3,680,996** | +29% |
| ftrl k=20 | 4,814,801 | **6,288,122** | +31% |

Grouped, k=20 over 64 groups: 5.13M → **6.04M rows/s**. Thread scaling on the
same workload now runs to every core the machine has (the script stopped at 8,
which on a 14-core box hides the row that matters): 916k / 1.65M / 2.86M /
4.75M / 6.03M at 1, 2, 4, 8, 14 threads — **6.6×**. Eight single-group specs
over 300k rows: 201.7 ms → **118 ms** against 685 ms run one at a time.
Expression under `.over(g)` at 1000 groups: **12.2M rows/s**.

**`rls` is the exception, and it is C5's bill.** Attributed by A/B on the model
arithmetic alone — `crates/online-core/examples/rls_bench.rs`, which compiles
unchanged against `50c1a38^` (the commit before the square-root rewrite),
so nothing but `Rls::step` differs:

| k | covariance form | square-root (QR) form | |
|---|---:|---:|---:|
| 5 | 13,915,222 | 7,226,608 | 0.52× |
| 20 | 4,390,434 | 1,700,789 | 0.39× |
| 50 | 996,592 | 491,013 | 0.49× |

Both forms are O(k²) per row; the QR form does more of it — `k` Givens
rotations, each with a square root — and buys the thing C5 was after: the
covariance form died deterministically on one extreme row. Worth paying, and
now stated rather than implied.

**Where the QR form's time actually went (2026-09-02).** Skipping the
per-row back-substitution — timing only, the numbers are meaningless without
it — took k=20 from 1.75M to 3.15M rows/s and k=50 from 511k to 1.26M: the
solve was *half* the row, though it is a quarter of the flops. It was
latency-bound. `beta_i = (u_i − Σ_{j>i} R_ij β_j) / R_ii` summed the row from
`j = i+1` upward, so every row's chain began with the coefficient that had
just been solved and could not start until it had; `k` chains of `k/2`
dependent subtractions, queued. Summing from the far end instead (`j = k−1`
downward) makes every term but the last independent of the previous row, so
the chains overlap in the pipeline. One `.rev()`:

| k | before | after | |
|---|---:|---:|---:|
| 5 | 7,244,780 | 7,360,936 | 1.02× |
| 20 | 1,753,624 | 2,118,547 | 1.21× |
| 50 | 510,829 | 799,909 | 1.57× |

Bank level, `rls` k=20: 1.63M → **1.93M rows/s** (README table). It is a
rounding-level change — a different summation order — measured on the golden
signatures at 1e-15 relative (the Rust one moved by one ulp in two of three
values; the Python pipeline by up to 1.2e-15 absolute), inside both tests'
1e-12 tolerance, so no pinned value was regenerated. The same experiment
rejected two others: a column-oriented solve (the ideal chain, but strided
loads; 2.25M / 774k, no better) and two interleaved partial sums (2.19M /
831k, within noise of the one-line version), and a decay pass fused into the
rotations, which is bit-identical and gained 8% at k ≤ 20 but lost 4% at
k=50 for a longer skip path — not worth its shape. What remains is the
rotation chain itself: `k` dependent `sqrt`-and-divide pairs per row, which
is the QR form's structure and not a codegen artefact; a square-root-free
(fast) Givens variant would shorten it and is the obvious place to look if
`rls` throughput ever becomes the constraint, at the cost of a rescaling
step whose stability would have to be argued as carefully as C5 was.

## 9. `predict` (E31, 2026-09-02)

Scoring is the learning loop with `learn = false` — the same `run_instance`,
the same per-row arithmetic up to the model call, and then `predict` in place
of `step`. Measured on the bank benchmark's million-row single-stream case,
`ewridge`, the two paths on the same frame:

| k | `fit_predict` | `predict` | |
|---|---:|---:|---:|
| 5 | 8,003,000 | **14,270,000** | 1.8× |
| 20 | 3,300,000 | **9,430,000** | 2.9× |

The gap is the update: at `k=20` a step is a rank-one update of the `k × k`
co-moments plus a solve every `solve_every` rows, and a prediction is a dot
product. The learning path itself did not move — `scripts/benchmark.py` gives
8.99M / 3.60M / 990k rows/s at k=5/20/50 against §8's 8.96M / 3.62M / 961k —
because the only change on it is that `ewridge`'s `step` now gets its
prediction from `predict` instead of an inlined copy of the same loop.

## 10. The runner: every format, every source (E32, 2026-09-02)

The runner is a three-stage pipeline — a reader thread, the bank on the
calling thread, a writer thread — with one chunk in flight per stage. E32
made the reader pluggable: `Input::Lazy` is a polars plan read by the
streaming engine in `chunk_rows` batches (the CLI, Rust callers), and
`Input::Batches` is an iterator of frames the caller already has, which is
how `po.run` feeds it: py-polars reads (`collect_batches`), Rust fits and
writes. Measured on the P6 file — 3M rows × 20 features × 32 groups, `ewridge`
k=20 with a clock, weights and `min_periods`, `chunk_rows=100k`, best of 3,
through `po.run`:

| input → output (same format) | rows sorted by clock (groups interleaved) | rows sorted by group |
|---|---:|---:|
| parquet | 0.71 s | 2.15 s |
| ipc | 0.59 s | 2.05 s |
| csv | 0.90 s | 2.15 s |
| ndjson | 1.14 s | 2.33 s |

`ONLINE_TIMING=1` prints one line per run beside the per-chunk ones —
`read_wait` and `write_wait` are the bank thread's slack, `writer_busy` the
writer's own time — and it says where each case sits:

- **Interleaved groups are I/O-bound, and the I/O is overlapped.** parquet:
  `read_wait=0.07s bank=0.67s write_wait=0.00s writer_busy=0.66s total=0.77s`
  — the parquet writer is as busy as the bank and hidden behind it. ipc:
  `writer_busy=0.11s`, total 0.62 s, the floor. csv:
  `bank=0.70s write_wait=0.13s writer_busy=0.86s` — the one write-bound case;
  the CSV *reader* (`read_wait=0.02s`) is no longer the problem. ndjson: the
  bank reads 1.04 s against 0.64 s alone, because the parallel NDJSON
  serializer competes for the same rayon pool as the bank's per-group tasks
  — contention, not a defect, and the total still beats serial.
- **Group-sorted rows are bank-bound in every format**: `bank=2.07s` of
  2.16 s, `read_wait=0.05s`. A 100k chunk of group-sorted rows holds one or
  two groups, so the bank's per-group parallelism has nothing to spread —
  the same file interleaved is 3× faster through the same code. The fix is
  a chunk that spans many groups: `chunk_rows=500_000` takes the sorted
  parquet from 2.15 s to **1.01 s**. The trade is memory (three chunks in
  flight), which is what `chunk_rows` was always for.

**Why py-polars reads for the Python path.** The first cut scanned in Rust
for every caller. On the CSV that was 1.72 s with `read_wait=1.06s`: the
bank sat waiting for the parser. polars-io's SIMD CSV parser is behind a
feature that needs a nightly compiler; py-polars' wheels have it and a
stable toolchain cannot, so the Rust-side `scan_csv` was ~6× slower than
`pl.scan_csv` on the same file. Reading with py-polars and handing frames
across (per-series Arrow C FFI export — cheap, though it rechunks, so a
120-chunk CSV batch arrives single-chunk; the bank is ~10% slower on
multi-chunk frames anyway) took the CSV run to 0.90 s and, as a side
effect, gave `po.run` every source py-polars can stream: globs, cloud URLs,
a query with a filter or a UDF, a `DataFrame`, an iterable of frames. The
GIL is released for the run and reacquired only to take each batch. The CLI
still scans in Rust, so on a stable toolchain **a large CSV is faster
through `po.run` than through `online`**. On parquet the CLI is ~15% behind
too (0.83 s / 2.55 s against `po.run`'s 0.72 s / 2.12 s), and its timing
line says why it is not the reader: `read_wait=0.06s bank=2.42s` against
`bank=2.07s` in Python, for the same code on the same chunks. Rechunking the
engine's two-piece batches on the reader thread was tried and changed
nothing. What differs between the two processes is the allocator — the
extension allocates through py-polars' jemalloc (`PolarsAllocator`, §6,
which measured it at +16% for exactly this `k`), the binary through the
system malloc — and giving the CLI its own would statically link one,
which rule 12 keeps a decision rather than a tweak.

**What the custom parts are worth.** Asked whether the runner's hand-written
pieces could go, each was measured against the plain polars call it
replaces, in the CLI on the interleaved file: the three-stage pipeline
against C7's one-thread loop, below (1.6–2.7×); `ParquetSink`'s parallel
page encoding against `BatchedWriter::write_batch` (0.86 s against 1.55 s —
the serial encode becomes the pace); and `ndjson_write`'s slice-per-thread
against polars' NDJSON `BatchedWriter`, where the custom part is a 4× win
under jemalloc (`po.run`: 1.04 s against ~4 s) and a defect under the
system allocator (the CLI: 4.8–54 s against a steady 4.1 s) — see
docs/IMPROVEMENTS.md C8 for the diagnosis and the two fixes.

**Why the bank is not a polars node.** The pipeline is what polars-stream
builds for its own ordered, stateful operators and does not offer to a user
function. Checked in polars-stream 0.55.2 (`physical_plan/lower_ir.rs`,
`lower_expr.rs`, `lower_group_by.rs`, `nodes/`): `df.rolling(index_column,
period)` and `expr.rolling(index_column=..)` with no `group_by` keys lower to
a `RollingGroupBy` node that receives morsels serially, insists the index is
sorted, keeps only the rows still inside the lookback (`buf_df` is sliced as
windows retire) and hands each batch of windows to parallel evaluators — a
serial fold feeding a parallel stage, which is this runner's shape.
`ewm_mean`/`ewm_var`/`ewm_std`/`ewm_sum`, `cum_*`, `forward_fill`, `rle`
and a group-by over sorted keys have nodes of the same kind. Everything else
collects first: `rolling(.., group_by=..)` and `group_by_dynamic(.., group_by=..)`
(`lower_group_by.rs:737`, an in-memory fallback), `rolling_mean(window_size)`
and any other non-elementwise function, `.over` with an `order_by`, and
every plugin or Python UDF expression — those become a `columnar-function`
node, an in-memory sink per input that calls the function once on the whole
column and becomes a source (`nodes/columnar_function.rs`, flagged
`is_memory_intensive_pipeline_blocker`). A user function that *is*
elementwise gets a `Map` node, per morsel and concurrent, which is why it
cannot carry state. So `online.ewridge(..)` as an expression is O(data) in
either engine, and an O(state) pass over a stream needs a fold the engine
does not expose — the reader → bank → writer pipeline here, which also gets
what `RollingGroupBy` gets, a serial stage and a parallel one, with the
parallel one across groups.

**Before and after.** The C7 runner ran `lf.slice(offset, chunk_rows)
.collect()` per chunk on the calling thread, re-planning the scan thirty
times and overlapping nothing. Same machine, same file, same spec, best of
3, the C7 wheel and binary built from `62d74a1` in a worktree:

| parquet → parquet | C7 `po.run` | E32 `po.run` | C7 `online` | E32 `online` |
|---|---:|---:|---:|---:|
| groups interleaved | 1.93 s | **0.72 s** | 2.14 s | **0.83 s** |
| group-sorted | 3.30 s | **2.12 s** | 3.84 s | **2.55 s** |

The bank, the plugin and `online-core` are untouched by E32 (`git diff
--stat 62d74a1 HEAD -- crates/` names only the runner, the CLI and the
bindings), so the `fit_predict` numbers in §8 stand. The price is size:
the extension 59.3 → 61.9 MB (+4.4%; gzipped 18.8 → 19.8 MB), the wheel
19.8 → 20.8 MB, the CLI 51.0 → 53.1 MB, all of it polars' ipc, csv and
ndjson readers and writers — no new crate outside polars and no new `-sys`
crate (`Cargo.lock`'s are the four C7 had).

**Writers.** Parquet through `BatchedWriter` with parallel page encoding —
one row group per chunk, so 30 row groups regardless of the input's layout.
IPC batched, one record batch per chunk. CSV batched, structs flattened to
`<spec>.<field>` and lists to JSON text (`format!("{v}")`, shortest
round-trip, so `str.json_decode` gives the bits back). NDJSON serialized in
parallel slices of each chunk and written in order. Output goes through a
temporary sibling and a rename, as `save` does.

## 11. Memory: which surface is O(data) (2026-09-02)

The claim on the README's first line is that the bank and the runner run on
data that does not fit in memory. §10's note that the plugin is O(data)
raised the question of whether everything is. Measured, on the §10 file and
the same file doubled and quadrupled by appending itself with the clock
shifted — same 32 groups, so the state is identical and only the data
grows — `ewridge` k=20 with clock, weights and `min_periods`, parquet in,
parquet out, `chunk_rows=100k`, 14 threads:

| peak physical footprint | 3M rows (0.54 GB parquet, 0.59 GB in memory) | 6M rows | 12M rows |
|---|---:|---:|---:|
| plugin, `sink_parquet(engine="streaming")` (`collect` is the same) | 1.98 GB | 3.73 GB | 7.35 GB |
| the same query without the plugin (`pl.col("y0") * 2`) | 0.81 GB | — | 2.08 GB |
| `online` CLI | 0.73 GB | 0.72 GB | 0.75 GB |
| `po.run` | 0.95 GB | 1.29 GB | 1.41 GB |
| `ModelBank` loop over `collect_batches` | 0.80 GB | 1.09 GB | 1.24 GB |
| `po.run`, jemalloc told to release freed pages at once | — | 0.87 GB | 0.86 GB |
| `ModelBank` loop, the same | — | 0.73 GB | 0.74 GB |
| `online` CLI, `POLARS_ROW_GROUP_PREFETCH_SIZE=1` | — | — | 0.15 GB |
| `po.run` / bank loop, prefetch 1 and pages released | — | — | 0.46 GB / 0.31 GB |
| **`lf.online.fit_predict`, `sink_parquet(engine="streaming")`** (E33) | 0.90 GB | 1.13 GB | 1.35 GB |
| the same, pages released / plus prefetch 1 | — | — | 0.78 GB / 0.37 GB |

**The plugin is O(data); nothing else is.** (Which is why, since 2026-09-03,
every expression call warns with `InMemoryExpressionWarning` and names the
plan — PLAN §6.) O(data) here means what it says:
the whole input is resident at once, because polars calls a plugin function
once with the entire column and has to have the column first (§10). The
sharpest form of the measurement is the same query with one expression
swapped — `scan_parquet(f).with_columns(<expr>).sink_parquet(out,
engine="streaming")`, prefetch pinned to 1 row group so the reader is not
part of the number:

| | 3M rows | 12M rows |
|---|---:|---:|
| `<expr>` = `pl.col("y0") * 2` | 0.51 GB | 0.51 GB |
| `<expr>` = `online.ewridge(...).over("group")` | 1.85 GB | 7.30 GB |

Same engine, same file, same sink: one expression streams flat, the other
holds three times the frame (2.4 GB at 12M rows) — the collected input, the
packed struct the plugin is handed, the `.over` gather, and the output
column. `collect()` and `sink_parquet(engine="streaming")` are within 1% of
each other, and pinning the prefetch, which takes the non-plugin query from
1.86 GB to 0.51 GB, does nothing for the plugin. Its time is linear too
(3.2 / 6.7 / 14.4 s). The CLI is flat
at 0.73 GB from 3M to 12M rows. `po.run` and the bank loop *report* a
creeping number, and the creep is the allocator, not live data: the
extension allocates through py-polars' jemalloc, which keeps freed pages
for ten seconds (`dirty_decay_ms`), longer than these runs, so the peak is
the high-water mark of everything ever allocated — with
`_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0` the same runs are flat
(0.87 / 0.86 GB) and about 20% slower for the purging; the CLI's system
malloc returns pages at once, which is why its trace drains and Python's
does not. Not a knob to set in production — it is the explanation of the
reported number, and the reason the CLI and `po.run` differ here.

**What the constant is.** Almost none of it is ours. A 100k-row chunk of
this file is ~20 MB and the pipeline holds three, the bank's state for 32
groups at k=20 is under a megabyte, and the output writer holds one chunk.
The rest is polars' parquet reader: the streaming engine prefetches
row groups ahead of the consumer — `row_group_prefetch_size: 96` on 14
threads, held back only by a 448 MB byte budget — and this file's row
groups are 262k rows, so the reader front-loads ~0.7 GB of decoded rows
before the bank has consumed one chunk (the CLI's trace peaks at the start
and drains from there). `POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes the
CLI on 12M rows from 0.75 GB to **0.15 GB at the same 3.2 s** — the bank,
not the reader, is the bound, and a local SSD needs no read-ahead; `=2` is
0.28 GB. The bank loop and `po.run` keep a few more chunks in flight in
py-polars' `collect_batches` and the FFI hand-off (0.31 / 0.46 GB with
prefetch 1). `POLARS_MAX_THREADS` scales the same term (4 threads: 0.45 GB,
1 thread: 0.18 GB), because the prefetch is sized from the thread count.
The byte budget (`POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET`) counts
compressed bytes, which for a memory-mapped local file cost nothing, so
lowering it changed nothing here. CSV and NDJSON have their own
(`POLARS_CSV_CHUNK_PREFETCH_LIMIT`, `POLARS_NDJSON_CHUNK_PREFETCH_LIMIT`),
not measured.

`chunk_rows` is the term the caller owns: `po.run` at 500k is 1.38 GB on 3M
rows against 0.95 GB at 100k, three chunks in flight being five times
larger. §10's advice to raise it for group-sorted input is a memory trade,
which is what the knob was for.

**The fix for the query-shaped trap (E33).** The plugin row above is what a
user gets for writing the natural thing — a `LazyFrame`, the expression,
`sink_parquet` — and expecting online processing, and it cannot be fixed in
the plugin: polars' contract for a user expression is the whole column (or
elementwise, which is stateless and unordered), and an ordered, stateful
per-morsel node is something only polars-stream itself can add
(`AnonymousScan` on the Rust side is `todo!()` in 0.55.2's `lower_ir.rs`).
What can be fixed is where the bank sits in a plan: `lf.online.fit_predict
(specs)` registers the bank as a polars **IO-plugin source**
(`register_io_source`), the kind of node the engine pulls batches from,
which runs `collect_batches` over the input and `ModelBank.fit_predict` per
chunk. The last two rows are that: **bit-identical to `po.run`'s output**,
12M rows in 2.8 s (`po.run` 2.9 s, the bank loop 2.5 s, the plugin 14.4 s),
and flat — the reported creep is jemalloc's again, 0.78 GB live at 12M rows
and 0.37 GB with the prefetch at 1 — with the plan's filter, projection and
`head` pushed into the source and a selection reaching the input scan. The
engine reads a few morsels ahead of the bank (7 of 100 input batches were
requested before a `head(10)` stopped the plan) and tears the input query
down with the plan.

**What comes before the bank: an upstream `filter` or `with_columns`
costs a bounded window, not the data (2026-09-02, and corrected three
times the same day from the polars-stream source: the stage, then what
fills its slots, then who decides what a slot holds).** The source reads its input with
`LazyFrame.collect_batches`, so the input plan runs in the streaming engine
and its memory is polars'. Measured on the same files, prefetch 1,
`sink_parquet(engine="streaming")`, the input plan being what feeds
`lf.online.fit_predict([spec])`:

| peak footprint | 3M rows | 12M rows |
|---|---:|---:|
| `scan` — nothing upstream | 0.65 GB | 0.65 GB |
| `scan.with_columns(<elementwise>)` | 0.92 GB | 1.52 GB |
| **`scan.filter(..)`** | 0.87 GB | **2.59 GB** |
| `scan.filter(..).with_columns(..)` | 0.92 GB | **2.85 GB** |
| `scan.with_columns(mean().over("group"))` | 0.97 GB | 3.16 GB |
| `scan.sort("t")` | 1.74 GB | 6.73 GB |
| the same filter **after** the bank (pushed into the source) | — | **0.78 GB** |

`sort` and `.over` are O(data) — pipeline breakers; the whole frame exists
before the first row comes out (`.over` measured on: 8.4 GB at 36M rows).
The `filter` and `with_columns` rows are something else, and the first
reading of them here — "`collect_batches` stops applying backpressure once a
filter is in the plan and buffers the filtered result" — was wrong. What
they hold is a **window that is bounded in morsels, not bytes**, and on this
machine the window is bigger than the 12M-row file. Isolated with no bank in
the process — `scan.<shape>.collect_batches(chunk_size=100_000)` iterated
with a `time.sleep` per chunk, 14 threads unless noted, a 36M-row file that
is the 12M one three times over:

| peak footprint | 12M rows | 36M rows |
|---|---:|---:|
| plain scan, `sleep(0.02)` | 0.67 GB | 0.64 GB |
| `filter`, `sleep(0.02)` | 2.54 GB | **3.11 GB** — a flat plateau, then a drain |
| `filter`, `sleep(0.02)`, `POLARS_MAX_THREADS=2` | 0.71 GB | 0.74 GB |
| `with_columns`, `sleep(0.1)` | 2.60 GB (the whole file) | **4.70 GB** (the file is 7.8 GB) |
| `mean().over("group")` | 2.97 GB | 8.39 GB — a straight ramp |

and the filter at 12M rows against the thread count: 1 → 0.46, 2 → 0.71,
4 → 1.08, 8 → 1.89, 14 → 2.54 GB, **0.2 GB per thread**, flat plateaus at
≤ 4 threads and a draining profile above, where the window exceeds the file
(96 row groups of 26 MB; at 14 threads the window is 98).

Where the window comes from (polars-stream 0.55.2). Backpressure in the
streaming engine counts *morsels per pipe*, and the count is multiplied by
the number of pipelines (= threads) at every serial → parallel → serial
transition (`pipe.rs`: a distributor with 4 slots per lane, one morsel in
flight per lane, a linearizer with 4 slots per lane when order is kept — 9
per lane through a parallel compute node such as `with-columns`). That is
the `with_columns` row, and it shrinks with the morsel:
`pl.Config.set_streaming_chunk_size(25_000)` — public API; the env var is
`POLARS_STREAMING_CHUNK_SIZE`, and `POLARS_IDEAL_MORSEL_SIZE` is silently
overwritten by the unset legacy name in 0.55.2, pola-rs/polars#29021 —
takes the isolated
`with_columns` probe from 2.56 to 1.32 GB, and to 1.03 at 10,000 rows.

The pushed-down `filter` is not a filter cost at all. The parquet reader
applies the predicate itself (`FULL_FILTER`, `row_group_decode.rs`: the
predicate's columns are decoded first, the mask is built, the other columns
are decoded through it), so the reader's output carries the predicate
columns *first*. Unless those columns already lead the projection, that
order no longer matches it, so the scan's post-apply stage is
`Initialized` with column selectors instead of `Noop` (`apply_extra_ops.rs`:
`is_input_passthrough` is `input_index == output_index` for every column),
and every morsel goes through `distributor_channel(num_pipelines, 1)` → one
worker per lane → `MorselLinearizer::new(num_pipelines, 4)`
(`post_apply_extra_ops.rs`) to have its columns permuted — a zero-copy
`select`, through a pipeline holding about 7 morsels per lane: 1 in the
distributor's buffer, 1 in the worker, 4 in the linearizer's channel, 1 in
its heap. What a morsel is here is the planner's choice, not the reader's.
A sink directly above the scan — `collect_batches`, `sink_*` — sets
`disable_morsel_split` on it (`physical_plan/lower_ir.rs`), so the reader
emits whole row groups and the chunk size does nothing: 2.59 GB at 25,000
rows, 2.63 at 10,000. With a compute node above the scan the reader splits
to `ideal_morsel_size` rows *before* this stage (`parquet/init.rs`), and
the stage then costs 14 × 7 × 3.2 MB ≈ 0.3 GB at 25,000-row morsels and
0.12 at 10,000 — isolated, 8M rows × 16 columns, 50,000-row groups:
`filter(x1).with_columns(..)` 1.11 / 0.60 GB against 0.80 / 0.48 with the
predicate on the first column. 14 lanes × 7 × 26 MB ≈ 2.5 GB plus the
reader's own ≈ the 2.5–3.1 GB measured, 0.2 GB per thread. Three checks,
isolated probes at 12M rows: the predicate column moved to the front of
the projection — `scan.select(["vol", *rest]).filter(pl.col("vol") > 0)` —
makes the post-apply `Noop` (`POLARS_VERBOSE=1` says so) and the same
filter costs **0.38 GB**, 0.58 GB with the bank behind it, and a predicate
that is on the first column to begin with is `Noop` as it stands (the 8M-row
file, 14 threads: plain 0.30 GB, `filter(x0)` 0.25, `filter(x1)` 0.93);
the slots hold what the filter *keeps*, so a predicate keeping 100 / 99 /
90 / 50 / 10 / 1 % of the rows peaks at 2.52 / 2.48 / 2.31 / 1.23 / 0.23 /
0.12 GB — the 2.5 GB above is the keep-everything worst case; and with
predicate pushdown off the filter is an ordinary compute node, 1.96 GB,
1.33 with 25,000-row morsels. A plain scan has no parallel stage between the reader and the sink
(serial → serial, capacity 1 each way), hence 0.65 GB at any consumer
speed; the filter *after* the bank is applied inside the IO-plugin source
and never enters a `multi-scan` stage at all. The window is O(threads ×
row-group rows × kept columns) and O(1) in the data. That the post-apply
pipeline exists only to restore column order is polars' to fix — a reader
emitting its columns in projection order would make the stage `Noop` —
and reordering one's own projection to dodge it is not something to build
on. Upstream (checked 2026-09-02): the stage's memory is known from
pola-rs/polars#28912 — a multi-file scan with an out-of-order `select`,
which the maintainer traced to "the morsel distributor in
PostApplyExtraOps" — and PR #29049 (merged after 1.44.1) divides its lane
count by the number of concurrently scanned files, which for one file is 1:
`stage_pipelines = num_pipelines.div_ceil(max_concurrent_scans)`, so a
single-file scan is unchanged; the PR calls in-lining the permutation
"still worth pursuing". That a pushed-down predicate on any but the
leading column trips the same stage on a single file, with these numbers,
is not reported there; #28569 (accepted) asks more generally that the
parquet source keep a bounded number of decoded morsels outstanding under
sink backpressure. #25242 says `set_streaming_chunk_size` has no effect in
the new engine; on 1.44.1 it does (`ideal_morsel_size: 25000` in the
verbose log, and the numbers above).

What shrinks it: fewer threads (`POLARS_MAX_THREADS`); for the pushed-down
filter, keeping fewer rows and smaller parquet row groups (the slots are
row groups, and polars' own default when writing is 262,144 rows, twice
these files'); for compute nodes, `pl.Config.set_streaming_chunk_size` as
above — and through it the pushed-down filter too, once a compute node
above the scan makes the reader split — and
`POLARS_DEFAULT_DISTRIBUTOR_BUFFER_SIZE` /
`POLARS_DEFAULT_LINEARIZER_BUFFER_SIZE`, which did nothing for the
pushed-down filter: its 1 and 4 are literals. A narrower projection
(`keep_columns=`, or a `select` before the bank) shrinks every row group.
Not the allocator (2.44 GB with `dirty_decay_ms:0`), not
`maintain_order=False`, not `lazy=True`.
One more spelling that is not the engine at all: `sink_batches` with the
default `engine="auto"` runs in the *in-memory* engine — polars-lazy maps
`Auto` to `InMemory`, file sinks are handed to the streaming executor from
there but the callback sink is not (`polars-mem-engine/planner/lp.rs`), so
it collects its input and then chunks it: 2.77 GB on the plain scan,
ramping, against 0.49 GB with `engine="streaming"`. `collect_batches`
resolves `auto` to streaming itself, in py-polars. That is the 1.x line:
polars 2.0 (rc.1, 2026-09-02) resolves `auto` to the streaming engine for
every lazy plan (pola-rs/polars#27822), so there `sink_batches` streams
by default.

What to do about it, and it is not merely a workaround: **filter after the
bank when the semantics allow**. The predicate is pushed into the source and
applied per chunk (E33), which is 0.78 GB flat against 2.53 GB for the same
filter before, and the two mean different things anyway — before changes what
the model *learns from*, after changes only what comes out. When the model
*must* skip those rows, a zero weight is the streaming spelling:
`with_columns(pl.when(cond).then(pl.col("w")).otherwise(0.0).alias("w2"))`
with `weight="w2"` — `when/then/otherwise` is elementwise, 1.3–1.4 GB at
12M rows against the filter's 2.53, 1.08 GB with
`pl.Config.set_streaming_chunk_size(25_000)` and 0.98 at 10,000 (the plain
scan is 0.87) — at the documented cost that the rows still
come out (scored), the clock advances through them, so `n_eff` decays and
`min_periods` can blank output, and no `max_dclock` gap opens where a filter
would leave one. A branch holding an `.over()` drags the whole expression
onto a collecting node (3.18 GB). `po.run(input=<a filtered plan>)` reads
the same way and has the same window; `keep_columns=` does not (a
projection is not a filter).

The one spelling that gives a filter the plain scan's footprint is to run
it *inside* the source — read the plain scan, `chunk.filter(cond)`, feed
the bank — because the IO-plugin source is the only serial stage in the
graph and whatever runs there costs one chunk. Measured as a prototype
(2026-09-02): 0.81 GB at 12M rows against 2.54, the same wall time, and the
output identical to the upstream filter's except `coef`, which is
snapshotted per chunk by contract. It is not in the API, deliberately: it
would be a second spelling of `filter` whose only difference is memory —
the class of surprise this section exists to remove; the CLI could not say
it without an expression parser; a predicate that is not elementwise
(`x > x.mean()`) would silently mean something different per chunk, where
polars refuses to push such a predicate into a scan at all; and the cost it
avoids is the column-reorder stage above, which is polars' to fix, after
which `scan.filter(..)` is 0.58 GB as written. Lowering the pipeline count
for the input query alone is the other lever that keeps the syntax
(`polars_config` reads `POLARS_MAX_THREADS` at every query start, and
py-polars' private `config_reload_env_var` re-reads it): 0.65 GB at 2
pipelines, 0.29 at 1 — and 2.7 → 3.9 → 7.1 s, because the reader's
row-group parallelism follows the same number. Not taken either.

**Polars' own windowed operations do the same thing (2026-09-02).** Worth
knowing before concluding that this is a quirk of ours: it is polars' rule,
and the rule is *whether the streaming engine has a node for that spelling*.
Same file, `sink_parquet(engine="streaming")`, `POLARS_ROW_GROUP_PREFETCH_SIZE=1`,
one output column so the scan reads only what the expression needs — each
plan is `pl.scan_parquet(f).select(<the expression below>)`, run in its own
process:

| peak footprint | 3M rows | 12M rows | grows? |
|---|---:|---:|---|
| `pl.col("y0") * 2` (the floor) | 0.11 GB | 0.14 GB | no |
| `pl.col("y0").mean().rolling(index_column="ti", period="1000i")` | 0.16 GB | 0.25 GB | no |
| **the same, `.over("group")`** | 1.76 GB | **6.52 GB** | **3.7× on 4× data** |
| `pl.col("y0").rolling_mean(1000)` | 0.12 GB | 0.34 GB | barely |
| `pl.col("y0").ewm_mean(half_life=500)` | 0.14 GB | 0.19 GB | no |
| the same, `.over("group")` | 0.30 GB | 0.72 GB | yes |
| `lf.rolling(index_column="ti", period="1000i").agg(mean)` | 0.18 GB | 0.28 GB | no |
| **the same, plus `group_by="group"`** | 0.49 GB | **1.72 GB** | **3.5× on 4× data** |

The mechanism is visible in polars-stream 0.55.2
(`physical_plan/lower_expr.rs`), and it is a classification, not a
heuristic. Every expression lands in one bin. *Elementwise* —
`FunctionFlags::ROW_SEPARABLE | LENGTH_PRESERVING`, computable on any subset
of rows with one row out per row in — stays inside the `select` /
`with-columns` / `filter` node it appears in and streams per morsel.
`AExpr::Rolling` is lowered to a dedicated `RollingGroupBy` node, a real
streaming one — it keeps a `buf_df` with a `buf_df_offset` and drops rows
once the window has passed them (`nodes/rolling_group_by.rs`); `ewm_*`,
`cum_*`, `shift`, `interpolate`, `rle` and friends each have their own node
under `nodes/`. `AExpr::Over` without an `order_by` tries
`try_build_streaming_group_by`, which rewrites `mean().over("group")` as
`multiplexer → group-by → equi-join → zip` — streaming nodes, but the
multiplexer buffers the whole input while the group-by side finishes, so it
is O(data) all the same (2.97 GB at 12M rows, 8.39 GB at 36M) — and when
that returns `None` (a `rolling` or `ewm_mean` under `.over`) pushes the
expression into `fallback_subset` → `build_fallback_node_with_ctx` →
**`PhysNodeKind::InMemoryMap`**: collect the input, run the in-memory engine,
re-emit. A frame-level `rolling` takes the dedicated node only while
`keys.is_empty()`, and any group-by carrying `rolling`/`dynamic` options
returns `Ok(None)` twice over (`lower_group_by.rs:737`, `:1043`) and ends at
`build_group_by_fallback` — the same collect. A user expression not flagged
elementwise — a plugin — is the generic fallback for column UDFs: a
`columnar-function` node, one `InMemorySink` per input, `call_udf` once on
the whole column, an `InMemorySource` after (`nodes/columnar_function.rs`).
The engine draws exactly this classification, and it is the fastest way to
know which bin a query landed in:
`lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)`
marks streaming nodes ◯, memory-intensive ones (multiplexer, group-by, join,
sort) yellow, and in-memory fallbacks (`columnar-function`, `in-memory-map`)
red. Out-of-core spilling exists for the yellow nodes in 1.44.1 but is off
by default — `POLARS_OOC_MEMORY_BUDGET_MB` is `u64::MAX` and the `_FRACTION`
variable is parsed and never read (`polars-config-0.55.2`); with a 1.5 GB
budget the `.over` plan peaked at 2.47 GB instead of 2.97: it spills, it
does not cap.

So the trap is not "user code is O(data)". It is that an ordered, stateful
operation streams **only where polars has hand-written a node for it**, and
the plugin interface has no way to declare one: there is no "call me per
morsel, in order, and let me keep state" in the contract, which is why one
call with the whole column is what a plugin gets. Two consequences worth
stating plainly:
polars' own `ewm_mean` streams as an expression while the same EW mean
written as our plugin does not, for that reason alone; and *per-group*
windowing in polars is the collecting spelling (`.over`, `group_by=`),
while a bank's `group=` is O(state) — one accumulator per group, no
partitioning of the data at all.

**How it was measured, and why not RSS.** Peak *physical footprint* —
`proc_pid_rusage(RUSAGE_INFO_V4).ri_phys_footprint`, sampled every 20 ms
from outside the process; the same number `/usr/bin/time -l` prints as
"peak memory footprint". RSS is the wrong ruler for this question: polars
memory-maps a local parquet file, and the file's clean pages count in RSS
though the kernel drops them under pressure at no cost. The CLI's *RSS*
grew 1.37 → 2.26 GB from 3M to 6M rows while its footprint stayed at
0.75 GB; the first cut of this measurement used `ru_maxrss` and said every
surface was O(data). It is not.
