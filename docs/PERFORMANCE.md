# Performance: where the time and memory go

This document records where a bank's time and memory go, every measurement
behind that, and the settings that move each number. Each numbered section
records one measurement, most of them dated. The first four, §1–§4, are the
original 2026-08-30 baseline, plan and outcome, kept as the record. The
later sections measure each model and feature as it arrived. Section
numbers are cited from the code and the README, so they stay where they
are.

| section | read it when | subsections |
|---|---|---|
| [Reading this document](#reading-this-document) | you are new here: the words this document uses, how every number was measured, and the results | [words](#words-this-document-uses) · [how the numbers are made](#how-the-numbers-are-made) · [the headline](#the-headline) |
| [1. The measured baseline](#1-the-measured-baseline) | you want the numbers from before any change, 2026-08-30 | |
| [2. What the numbers say](#2-what-the-numbers-say) | you want why the plumbing around the models, not the models, was the cost | |
| [3. The plan](#3-the-plan) | the code cites one of P1–P11, or you want what each change did to the numbers | [P1](#p1--columnar-hot-path) · [P2](#p2--one-flat-task-pool) · [P3](#p3--extraction-and-grouping-without-materialization) · [P4](#p4--assembly-into-typed-builders) · [P5](#p5--expression-path-parity) · [P6](#p6--runner-pipelining) · [P7](#p7--build-flags-measured-and-both-rejected) · [P8](#p8--re-baseline-and-lock) · [P9](#p9--group-contiguous-layout-2026-09-04) · [P10](#p10--assembly-per-field-2026-09-04) · [P11](#p11--extraction-per-column-per-arrow-chunk-and-integer-keys-as-themselves-2026-09-04) |
| [4. Where it ended up](#4-where-it-ended-up) | you want the result of P1–P8, against §1 | |
| [5. Rejected, and why](#5-rejected-and-why) | you are about to try something, and want to know whether it was tried and rejected | [rejected later](#rejected-later-where-each-was-measured) |
| [6. The allocator](#6-the-allocator-2026-08-31) | you want why the extension installs `PolarsAllocator`, and what it measured | |
| [7. Bugs found by this review](#7-bugs-found-by-this-review) | you want the two session-hashing findings | |
| [8. Refresh, and what `rls` costs now](#8-refresh-2026-09-02-and-what-rls-costs-now) | you want what a row cost per model on 2026-09-02, and why `rls`'s square-root form is slower | [the refresh](#the-refresh) · [`rls`'s square-root form](#rls-and-its-square-root-form-c5) · [where its time went](#where-the-qr-forms-time-actually-went-2026-09-02) |
| [9. `predict`](#9-predict-e31-2026-09-02) | you score without learning, and want the speed against `fit_predict` | |
| [10. The runner](#10-the-runner-every-format-every-source-e32-2026-09-02) | you run the `online` command line on files: its formats, its timing line, and what its custom parts are worth | [the pipeline](#the-pipeline) · [every format](#every-format) · [why py-polars reads](#why-py-polars-reads-for-the-python-path) · [the custom parts](#what-the-custom-parts-are-worth) · [not a polars node](#why-the-bank-is-not-a-polars-node) · [before and after](#before-and-after-c7--e32) · [writers](#writers) |
| [11. Memory](#11-memory-which-surface-is-odata-2026-09-02) | memory is the question: which surface is O(data) and which O(state), with the numbers, and what Polars holds before the bank | [the plugin](#the-plugin-was-odata-nothing-else-is) · [the constant](#what-the-constant-is) · [the fix, E33](#the-query-shaped-trap-and-the-fix-e33) · [before the bank](#what-comes-before-the-bank) · [the window](#where-the-window-comes-from) · [what shrinks it](#what-shrinks-the-window) · [filter after the bank](#filter-after-the-bank) · [levers not taken](#two-levers-not-taken) · [Polars' own windows](#polars-own-windowed-operations-do-the-same-thing) · [how it was measured](#how-it-was-measured-and-why-not-rss) |
| [12. The chunk plan, revisited](#12-the-chunk-plan-revisited-2026-09-04) | you are choosing a chunk size and a thread count | [the plan](#the-plan-as-it-stood) · [the sections](#what-the-sections-said) · [the artifact](#the-benchmark-artifact) · [where it ended up](#where-it-ended-up) · [`chunk_rows`, swept](#chunk_rows-swept) · [the README's numbers](#the-readmes-numbers-regenerated) · [the row floor](#the-row-floor) · [what is left](#what-is-left-and-why-it-stays) |
| [13. The new families](#13-the-new-families-and-where-a-wide-row-goes-2026-09-05) | you run a family of tasks 23–30 (`kmeans`, `micro`, `ew_class`, `sgd`, `pa`, `seqtest`, `kalman`'s transition, conformal intervals), or a wide row | [the contract](#the-contract-bit-for-bit) · [the survey](#the-survey) · [`ew_cov`'s row](#where-ew_covs-row-went) · [the wall clock](#reading-the-wall-clock-not-the-profile) · [`ew_class`](#ew_class-one-factorization-per-learned-row) · [`kalman`, `sgd`, `pa`](#kalman-sgd-pa-allocations-and-a-closure) · [thread scaling](#thread-scaling) · [what is left](#what-is-left-and-why) |
| [14. The Gram update](#14-the-gram-update-measured-e48-2026-09-05) | you want why the Gram update is not computed as one triangle | [the variants](#the-variants) · [not bit-identical](#e48s-mirror-is-not-bit-identical) · [slower](#and-it-is-slower-by-a-lot) · [what shipped](#what-is-worth-having) |
| [15. The correlation families](#15-the-correlation-families-bocpds-prune_below-keeps-it-finite-and-rcovs-estimator-sets-its-cost-2026-09-06) | you run `bocpd`, `rcov`, `deco`, `hmm` or `corrchange` | [`bocpd`](#bocpd-prune_below-is-what-makes-it-finite) · [`rcov`](#rcov-the-cost-is-at-the-close-and-the-kernel-chooses-it) · [the other three](#the-other-three) · [going wide](#and-they-go-wide) |
| [16. What a `window` costs](#16-what-a-window-costs-2026-09-07) | you use `window`: what it costs, and what it does not | [the windowless path](#the-windowless-path-is-unchanged) · [the window itself](#what-the-window-itself-costs) |
| [17. `marginal`'s two views](#17-what-marginals-two-views-cost-and-two-costs-that-were-not-theirs-2026-09-07) | you use `marginal`'s lags or bins | [the plain path](#the-plain-path-and-two-regressions-no-test-could-see) · [the views](#what-the-views-themselves-cost) |
| [18. The blocked Gram update](#18-the-blocked-gram-update-e51-2026-09-08) | you have a wide `ewridge`: what `gram_block_rows` gains, and what a solve takes back | [the measurement](#the-measurement) · [three readings](#three-things-to-read-off-it) · [what it does not change](#what-blocking-does-not-change) |
| [19. Against `SGDRegressor`](#19-against-sklearnlinear_modelsgdregressor-task-72-2026-09-08) | you are comparing this library with scikit-learn | [the protocol](#the-protocol) · [accuracy](#accuracy) · [the first hundred rows](#the-first-hundred-rows-of-every-group) · [task 74](#where-sgd-lost-and-task-74) · [throughput](#throughput) · [a grid](#a-grid) · [a wide row](#a-wide-row) · [what it gave back](#what-the-comparison-gave-back) |
| [20. `sgd`'s per-feature cost](#20-sgds-per-feature-cost-task-75-2026-09-08) | you run `sgd` on thousands of features | [the 14 ns](#where-the-14-ns-went) · [the result](#the-result) · [what is left](#what-is-left-at-k--10000) · [the scaler](#the-scaler) |
| [21. What inspecting the plan costs `fit(lf)`](#21-what-inspecting-the-plan-costs-fitlf-2026-09-18) | `fit(lf)` seems slow on a small input | [the bisect](#the-bisect) · [what it is](#what-it-actually-is) · [the fix](#the-fix) · [the case the user has](#measure-the-case-the-user-has) |

## Reading this document

Every later section assumes three things: the words below, how the numbers
were made, and the results they add up to.

### Words this document uses

Besides the README's own words (bank, spec, chunk, state), these:

| word | meaning here |
|---|---|
| golden number | a value the golden tests pin: `golden.rs` in `online-core`, and `tests/test_golden_pipeline.py` to 1e-12. A task in §3 that moved one was wrong by definition. Later sections accept a reordering of the arithmetic that moves a value within that tolerance (§8, §20) |
| bit-identical, bit-exact | the same floating-point bits: §13 compares every float column as its `u64` bits, with its validity |
| stream | as the README's *Parallelism* uses it: one spec over one group's rows, in order, which is the bank's unit of parallel work (§15). A single stream is one spec over one group, or with no `group` at all |
| instance | one model inside a spec's grid, such as one halflife of a five-halflife grid. A stream steps its instances over the same rows (§2, P2) |
| slot | one value an instance writes for every row: a target's prediction, or one of `ew_cov`'s 230 statistics (§13) |
| section | one part of a chunk's work, as `ONLINE_TIMING=1` times it: `group` (row indices per key), `extract` (the columns), `process` (the models) and `assemble` (the output columns) (§12) |
| run | the rows a stream feeds through one set of output buffers, `ChunkOut::run_rows` of them (§13) |
| stride, gather | a stride reads one group's values from every n-th row of an interleaved column. A gather copies them into one contiguous run first (§12, P9) |
| the Gram | the `k × k` co-moment matrix a regression keeps (§14, §18) |
| solve cadence | how often a regression solves: every `solve_every` clock units, `halflife / 50` by default (§12) |
| O(state), O(data) | memory that grows with what the models keep, or with the rows that have passed. O(data) means the whole input is resident at once (§11) |
| peak footprint | the most physical memory a process held, as `/usr/bin/time -l` prints it. Not RSS, which also counts the pages of a memory-mapped file (§11) |
| row group, prefetch | a block of a parquet file. Polars' reader decodes row groups ahead of whatever consumes them, and that read-ahead is the prefetch (§11) |
| morsel | the batch of rows Polars' streaming engine passes between its nodes. §11 says what sets its size |
| pipeline, lane | the streaming engine runs a parallel stage as one pipeline, or lane, per thread (§11) |
| distributor, linearizer | the stages that hand morsels out to the lanes, and gather them back in order (§11) |

### How the numbers are made

**Every number here was measured on one machine:** an Apple M-series with
10 performance and 4 efficiency cores, a release build (thin LTO,
`codegen-units = 1`) and a single process. The chip, named in §19 and §20,
is an M4 Pro. `ONLINE_TIMING=1` gives the section rows. §11 says how
memory was measured, and why not by RSS.

The raw numbers regenerate with these commands, each run from the
repository root:

```sh
# The pure core, no Polars anywhere: EwRidge::step in a loop (§1)
cargo run --release -p online-core --example core_bench

# The rls A/B of §8, and the blocked Gram of §18
cargo run --release -p online-core --example rls_bench
cargo run --release -p online-core --example gram_block_bench

# The bank through Python, with the per-chunk section timings on stderr
ONLINE_TIMING=1 uv run python scripts/benchmark.py
```

The sections name the rest where they use them: `scripts/scaling_bench.py`
(§8), `marg_bench` (§17), `scripts/sklearn_comparison.py` (§19), and
`sgd_bench` and `sgd_signature` (§20).

`ONLINE_TIMING=1` prints two kinds of line to stderr:

| printed by | once per | fields |
|---|---|---|
| the bank | chunk | `extract`, `group`, `process` and `assemble`, the sections, in wall milliseconds; and `total`, the whole call, of which the four sections are consecutive parts (`crates/online-polars/src/bank.rs:2704`, `:2945`) |
| the command line's runner | run | `read_wait` and `write_wait`, the bank thread's waits on the reader and the writer; `bank`, its time in the bank; `writer_busy`, the writer's own time; and `total`, the whole run (`crates/online-polars/src/runner.rs:585`–`593`, `:748`) |

A profile says where inside `process` the time goes, and the wall clock
says how much: §13 has the recipe. [CONTRIBUTING.md](../CONTRIBUTING.md)
says how to measure a change against these numbers.

### The headline

**Every golden number was unchanged throughout: that was the contract.**
Status as of 2026-09-06: P1–P11 all done, the numbers refreshed in §8, the
chunk plan revisited in §12, the new families surveyed in §13, and the
correlation families of tasks 45–56 in §15. The later sections, §16–§21,
are each dated in their headings.

Against the baseline in §1:

| case | before | after | ratio | § |
|---|---:|---:|---:|---|
| single stream, k=5 | 2,093,719 rows/s | 4,980,607 rows/s | 2.4× | §4 |
| single stream, k=20 | 1,308,792 rows/s | 2,634,627 rows/s | 2.0× | §4 |
| grouped data, k=20 over 64 groups | 2,601,719 rows/s | 5,132,169 rows/s | 2.0× | §4 |
| a single-stream grid of five halflives | 458,500 rows/s | 955,662 rows/s | 2.1× | §4 |
| a multi-spec bank, 8 specs × 1 group (wall) | 783.8 ms | 201.7 ms | 3.9× | §4 |
| the CLI end to end, 3M rows × 20 features × 32 groups | 2.17 s | 1.76 s | 1.23× | §4 |
| thread scaling on ten performance cores | 3.2× | 6.2× | | §4 |
| a grouped chunk at 14 threads (wall) | 37 ms | 17 ms | 2.2× | §12 |
| the 230-statistic `ew_cov`, per 400k rows | 758 ms | 222 ms | 3.4× | §13 |
| the full-covariance `ew_class`, per 400k rows | 1510 ms | 808 ms | 1.87× | §13 |

**Two plan items were closed by *rejecting* them on measurement, rather
than building them:** P4's typed builders and P7's build flags. P10 later
reversed the first, on new measurements (2026-09-04). P5 found its target
unreachable, for a reason it placed outside this codebase.
[docs/IMPROVEMENTS.md](IMPROVEMENTS.md) P1 later showed that reason wrong,
and the plugin reached parity with the bank. Each is written up where it
sits, and §5 collects everything rejected, including what later sections
rejected.

## 1. The measured baseline

The numbers before any of §3's changes, measured on 2026-08-30.

**The pure core:** `EwRidge::step` in a loop, no Polars anywhere
(`core_bench`).

| configuration | rows/s |
|---|---|
| k=5, 1 target | 11,804,355 |
| k=20, 1 target | 5,734,052 |
| k=50, 1 target | 1,987,873 |
| k=20, 10 targets | 3,968,505 |
| k=20, solve **every** row | 468,455 |
| k=20, solve every 25 rows | 3,941,742 |

**The bank on the same arithmetic:** 200k rows, ewridge, a clock column,
and the `ONLINE_TIMING` sections in ms.

| case | extract | group | process | assemble | total | rows/s |
|---|---|---|---|---|---|---|
| k=5, 1 group | 3.8 | 0.1 | 87.6 | 4.1 | 95.5 | 2,093,719 |
| k=20, 1 group | 12.2 | 0.2 | 133.6 | 6.9 | 152.8 | 1,308,792 |
| k=20, 8 groups | 12.2 | 14.8 | 46.3 | 8.8 | 82.3 | 2,431,070 |
| k=20, 64 groups | 13.0 | 15.3 | 40.6 | 8.0 | 76.9 | 2,601,719 |

**Thread scaling:** 400k rows, k=20, 64 independent groups
(`POLARS_ONLINE_MAX_THREADS`, then `RAYON_NUM_THREADS`).

| threads | rows/s | speedup |
|---|---|---|
| 1 | 435,792 | 1.0× |
| 2 | 702,049 | 1.6× |
| 4 | 1,021,413 | 2.3× |
| 8 | 1,363,018 | 3.1× |
| 10 | 1,414,097 | **3.2×** |

**The expression API:** 400k rows, k=20, measured while it existed. The
expression form was removed in task 85 (2026-09-17). These numbers are part
of why, and are kept as the record of it.

| path | rows/s |
|---|---|
| bank, 1000 groups | 1,522,227 |
| `.online.ewridge(...).over("g")`, 1000 groups | 511,237 |
| expression, single stream | 1,022,170 |

## 2. What the numbers say

1. **The integration layer costs 3–5× the model itself.** At k=20 on a
   single stream, the core needs ~35 ms for 200k rows, and the bank's
   `process` section takes 133.6 ms. The difference is per-row heap
   traffic: two `Vec` allocations for `x`/`y` per row, a `RowOut` with ~11
   `Vec` fields per row, and per-model `resid`/`sigma`/`resid_z` vectors.
   That is ~10–20 allocations per row, ~25M/s at these speeds.

2. **The same traffic is why threads don't scale.** 64 independent streams
   reach 3.2× on 10 performance cores. Embarrassingly parallel work that
   stalls at 3× is the signature of allocator contention and memory
   bandwidth, not of compute. Fixing (1) is most of fixing this.

3. **The serial sections put an Amdahl cap on top.** `extract` (12 ms),
   `group` (15 ms) and `assemble` (8 ms) are all single-threaded, and at 64
   groups they are already 45% of wall. `group` casts the key column to
   String and clones a `String` per row into a HashMap. `extract`
   materializes `Vec<Option<f64>>`, 16 bytes and a branch per value, even
   for columns with no nulls.

4. **Structural serialism the benchmarks don't show.** Specs are processed
   one after another: parallelism is only across groups *within* a spec.
   The model instances of a grid are stepped serially inside one stream,
   though the 5 instances are independent given the same rows. A
   5-halflife grid on a single stream runs at 292,573 rows/s with 9 cores
   idle.

5. **The expression path pays per group.** Under `.over()` the plugin is
   invoked once per group, and each invocation parses the spec JSON, builds
   a `Bank`, and re-extracts its columns. On identical work at 1000 groups,
   that is 3× slower than the bank. (The expression path was removed in
   task 85; §1 keeps its numbers.)

6. **Solving is already amortized.** Solving every row is a 12× hit
   (5.7M → 468k). The default `solve_every = halflife/50` sits within 1.5×
   of never solving. No work is needed beyond not regressing it.

7. **Within one model instance, one stream, the recursion is sequential by
   construction.** State at row *i* depends on row *i−1*. There is no
   parallelism to extract there, and nothing below attempts it.

## 3. The plan

The plan was ordered by measured impact per unit of risk. **Every task
keeps the two guarantees, out-of-sample and chunk invariance,
bit-for-bit.** The golden tests (`golden.rs`, `test_golden_pipeline.py`)
are the regression net, and any task that moves a golden number is wrong by
definition. Each item below keeps its original plan, folded, under what was
done. Code cites these items by ID.

| ID | the change | status | measured |
|---|---|---|---|
| [P1](#p1--columnar-hot-path) | columnar hot path | done | k=20 single stream 1.31M → 2.40M rows/s; 10-thread scaling 3.2× → 4.8× |
| [P2](#p2--one-flat-task-pool) | one flat task pool | done, both halves | 8 single-group specs 783.8 → 238.8 ms; a five-halflife grid 458,500 → 1,000,341 rows/s |
| [P3](#p3--extraction-and-grouping-without-materialization) | extraction and grouping without materialization | done | extract 11.6 → 2.1 ms, group 15.2 → 4.6 ms; 3.32M → 5.49M rows/s |
| [P4](#p4--assembly-into-typed-builders) | assembly into typed builders | done, partly; the typed builders dropped, then taken up by P10 | `assemble` 8.0 → ~5.5 ms |
| [P5](#p5--expression-path-parity) | expression path parity | target first found unreachable; revisited, and reached | 511,237 → 731,416 rows/s; then 4.2M → 21.4M |
| [P6](#p6--runner-pipelining) | runner pipelining | done | the CLI 2.17 s → 1.76 s (1.23×) |
| [P7](#p7--build-flags-measured-and-both-rejected) | build flags | measured, and both rejected | fat LTO −2.1% to +3.2%; `target-cpu=native` −3% at k=20 |
| [P8](#p8--re-baseline-and-lock) | re-baseline and lock | done | §4 |
| [P9](#p9--group-contiguous-layout-2026-09-04) | group-contiguous layout | done, 2026-09-04 | the stride had cost 14% of `process` on one thread, 38% on fourteen |
| [P10](#p10--assembly-per-field-2026-09-04) | assembly per field | done, 2026-09-04 | `assemble` 39.6 → 3.5 ms on a five-halflife grid |
| [P11](#p11--extraction-per-column-per-arrow-chunk-and-integer-keys-as-themselves-2026-09-04) | extraction per column, per arrow chunk, and integer keys as themselves | done, 2026-09-04 | `group` 8.6 → 1.6 ms on the 64-group chunk |

### P1 — Columnar hot path

*Done.* `RowOut`, a struct of ~11 `Vec`s per row, is replaced by
`ChunkOut`: flat slot-major `Vec<f64>` buffers, one allocation per output
column per (stream, chunk). `processed` marks skipped rows, and NaN means
null. `Stream::process_chunk` owns the row loop and reuses scratch (`xs`,
`ys`, `r_buf`, `sig_buf`, `zs_buf`), so the loop itself allocates nothing.
The `pred` `Vec` inside each `Step` is still allocated per row (see §4). It
measured at 14 ns against a 95 ns non-solve step in docs/IMPROVEMENTS.md
P2, and was left alone.

| measured | before | after |
|---|---:|---:|
| k=5, single stream | 2.09M rows/s | **5.12M** |
| k=20, single stream | 1.31M | **2.40M** |
| k=20, 64 groups | 2.60M | **3.32M** |
| the `process` section, k=5 | 87.6 ms | 30.7 ms |
| the `process` section, k=20 | 133.6 ms | 67.1 ms |
| thread scaling, 10 threads | **3.2×** | **4.8×** (628k → 2.99M rows/s) |

That is short of the ≥3.5M and ≥7× targets, and the reason is now visible
in the sections. At 64 groups the serial `extract` (11.6 ms) and `group`
(15.2 ms) are 44% of wall, so P3 is what unlocks the rest. Every golden
number unchanged.

*Since P2 (2026-08-30)* the scratch is one `Scratch` per model instance,
whose fields are `ys`, `r`, `sig` and `zs`
(`crates/online-polars/src/stream.rs:3159`). *Since task 75 (2026-09-08)*
a row's features come from the chunk's row-major `FeatureRows` (§20), the
`features` argument of `run_instance` (`stream.rs:3290`), so there is no
`xs`.

<details><summary>original plan</summary>

Replace per-row `RowOut` with `Stream::process_chunk`: the stream walks its
row indices writing directly into preallocated flat output buffers (one
`Vec<f64>` per output slot per chunk, NaN as null, bitmaps built at
assembly). Reuse one scratch `xs`/`ys` buffer per stream. This deletes the
per-row allocations of (2) and most of `assemble`'s scatter. Target: k=20
single stream ≥ 3.5M rows/s (from 1.31M); grouped scaling ≥ 7× at 10
threads (from 3.2×).

</details>

### P2 — One flat task pool

*Done, both halves.*

| half | what changed | measured |
|---|---|---|
| **(spec × group)** | the per-spec loop of per-spec rayon pools became one pool over every stream in the bank, longest-first, so a few big groups do not strand cores at the tail | on 8 single-group specs, **783.8 → 238.8 ms (3.3×)**, where before only one spec ran at a time |
| **(× instance)** | `process_chunk` is now two passes: one serial walk deciding the clock schedule, then the instances over the whole chunk | a five-halflife grid on one stream, **458,500 → 1,000,341 rows/s (2.2×)** |

The clock schedule depends only on the clock and the input columns, never
on the models. Instances share nothing but that schedule, so a
five-halflife grid on one stream is five independent recursions. The
exception is `drift_action = "reset"`, where a break in any instance resets
all of them within a row. That case keeps row-major order. Both paths call
the same `run_instance`, so there is one copy of the arithmetic. The extra
plan pass costs ~5% on the already-parallel grouped case (4.74M → 4.54M at
10 threads). Golden numbers unchanged.

<details><summary>original plan</summary>

Fan out over (spec × group × model-instance) in a single `par_iter`, not
per-spec loops: a bank of N single-group specs currently uses one core; a
grid on one stream uses one core. Instances need their per-instance
diagnostics (`resid_var`, `drift`, …) split into a per-instance struct so
rayon tasks own disjoint `&mut` — mechanical, the indexing is already
`[mi]`-major. Target: 5-halflife single stream ≥ 4× itself; N-spec banks
scale with specs.

</details>

### P3 — Extraction and grouping without materialization

*Done.* Columns extract to plain `Vec<f64>` with **NaN for null** instead
of `Vec<Option<f64>>`. That is half the bytes, no per-value branch, and a
`memcpy` via `cont_slice()` for a null-free contiguous column. (Since task
86 the same borrow is `f64_values` over the Arrow array, in
`crate::arrow`.) It is sound because every consumer already collapsed the
two: a feature or weight counts only when `is_finite`, and a target only
when finite. The clock is the one column where null is an error, and that
check now catches NaN with it.

Group keys are bucketed by a 64-bit hash of the value, so a `String` is
allocated once per distinct group rather than cloned three times per row.
Extract and group run per spec in parallel. Measured at k=20 over 64
groups: **extract 11.6 → 2.1 ms, group 15.2 → 4.6 ms**, taking that case
**3.32M → 5.49M rows/s** and thread scaling **4.8× → 6.3×**. That is just
short of the ≤4 ms target. What is left is real work, the cast and the
copy, not overhead.

<details><summary>original plan</summary>

Borrow value slices + validity bitmaps from the (rechunked) columns instead
of building `Vec<Option<f64>>`; null-free fast path is a borrow, not a
copy. Group and session keys: hash the physical values row-wise (as
`session_hash` now does) — no String cast, no per-row `String` clone, and
run it per spec in parallel with extraction. Target: extract+group ≤ 4 ms
at k=20/64 groups (from 27 ms).

</details>

### P4 — Assembly into typed builders

*Done, partly, and the rest dropped as unnecessary.* Specs now assemble in
parallel. The typed-builder half was not worth doing. P1's flat buffers had
already taken `assemble` from 8.0 to ~5.5 ms at 200k rows, which is under
15% of wall, and the remaining cost is the `Series` construction polars
needs either way. Measuring first is what said so. *(2026-09-04: P10
reversed this on new measurements; §5 has both.)*

<details><summary>original plan</summary>

Write `Float64Chunked` from `Vec<f64>` + computed validity instead of
`Vec<Option<f64>>` series; assemble specs in parallel. Mostly falls out of
P1's flat buffers.

</details>

### P5 — Expression path parity

*Done; the target turned out to be unreachable, for a reason worth
recording.* A thread-local cache keyed on the kwargs JSON means `.over()`
parses and validates the spec once per thread instead of once per group:
**511,237 → 731,416 rows/s** at 1000 groups. Then a sweep over group counts
showed where the rest goes. Throughput falls from 2.77M (ungrouped) to 769k
at *ten* groups of 40k rows each, and is then flat through 1000 groups.
Per-group `Bank` construction would scale with the group count; this does
not. **The remaining gap is polars' own `.over()` gather/scatter, not the
plugin**, and no change on our side reaches it. The `ModelBank` API with
`group=` is the answer for high group counts (5.01M rows/s on the same
data), which is what the README already recommends.

**Revisited ([docs/IMPROVEMENTS.md](IMPROVEMENTS.md) P1): the conclusion
above was wrong.** The gap was not gather/scatter. It was polars evaluating
a *multi-input* group-aware function group by group on one thread:
`apply_multiple_group_aware` in polars-expr has no parallel branch, and the
single-input path does. Packing every input column into one struct moved
the plugin onto the parallel path. At 1000 groups (k=5, 2M rows) that is
**4.2M → 21.4M rows/s**, and at k=20 over 100 groups of 4000 rows it is
**0.94M → 6.4M** (the table in docs/IMPROVEMENTS.md P1). That is at parity
with, or ahead of, the bank. The expression form was later removed, in task 85 (§1).

<details><summary>original plan</summary>

Thread-local cache of parsed `Spec`/plan keyed by the kwargs JSON, so 1000
`.over()` groups parse once; skip re-validation per group. Re-measure after
P1 — the remaining gap should be per-group extraction only. Target: within
1.3× of the bank at 1000 groups.

</details>

### P6 — Runner pipelining

*Done.* The runner alternated read-chunk and compute-and-write-chunk. A
reader thread now fills a `sync_channel(1)`, so chunk *n+1* is decoded
while chunk *n* is fitted and written. That is one chunk of lookahead, so
memory stays O(chunk). Order is preserved by construction (one reader,
FIFO), which matters because chunks must reach the bank in stream order.
Measured on 3M rows × 20 features × 32 groups through the CLI: **2.17 s →
1.76 s (1.23×)**.

<details><summary>original plan</summary>

`run_config` currently alternates read-chunk / compute-chunk. Double-buffer
with a bounded channel (read row group *n+1* while computing *n*); parquet
decode is already internally parallel, so this overlaps the two pools.
Target: CLI wall time ≤ max(io, compute) + ε on a large file.

</details>

### P7 — Build flags: measured, and both rejected

| flag | measured on the six `core_bench` cases | verdict |
|---|---|---|
| `lto = "fat"` | +3.2%, +1.3%, −1.2%, −1.0%, −2.1%, +0.4%: a wash, against a build that is already slow | fails the ≥3% bar the plan set |
| `-C target-cpu=native` | −3% at k=20, and within noise elsewhere | would make wheels non-portable for nothing |

Both stay off; `lto = "thin"`, `codegen-units = 1` remains.

**Manual SIMD is also unnecessary, and the throughput curve is the
evidence.** The co-moment update is O(k²), so k=5 → k=20 is 12.3× the
arithmetic but only 2.1× the time, and k=5 → k=50 is 72× the arithmetic for
6.0× the time. Per element, wider is *cheaper*, which is what
auto-vectorized inner loops look like. There is nothing here for
hand-written intrinsics to recover.

<details><summary>original plan</summary>

Try `lto = "fat"` and (CLI and local dev only, never wheels)
`-C target-cpu=native`; keep each only if ≥ 3% on `core_bench`. Verify the
k-loops in `ewcov::update` auto-vectorize (`cargo asm` spot check) before
considering any manual SIMD — at k ≤ 50 the compiler usually already does
this.

</details>

### P8 — Re-baseline and lock

*Done.* All of §1's measurements were re-run on an idle machine and written
up in §4. The README's throughput table was regenerated from
`scripts/benchmark.py`, with a note that grouped data now scales rather
than merely working. `benchmark.yml` gains the scaling row, so CI history
carries it.

<details><summary>original plan</summary>

Re-run `core_bench`, the timing matrix, scaling and `scripts/benchmark.py`;
update this file and the README table; extend `benchmark.yml`'s job summary
with the scaling row so the CI history carries it. Golden tests must be
untouched throughout.

</details>

### P9 — Group-contiguous layout (2026-09-04)

*Done.* Every column of a spec is gathered once, group after group in
first-seen order (`layout_of`). A stream then reads its rows as one
contiguous run at `base + ri`, instead of gathering every column at a
stride, twice per row (`check_clock`, then `process_chunk`). The gather is
skipped when the chunk is already blocked: no group, one group, or groups
arriving as blocks. Measured with a *matched* clock (§12), the stride cost
14% of `process` on one thread and 38% on fourteen, at k=20 over 64
interleaved groups. Output rows keep their absolute index for `out.rows`
and error messages.

### P10 — Assembly per field (2026-09-04)

*Done, and it reverses §5's "typed builders are not worth it".* There is
one job per output field, each a scatter over every stream's run into a
`Vec<f64>` plus a `MutableBitmap`, finished with
`Float64Chunked::from_vec_validity`. A spec with a grid of instances used
to build its fields one after another through `Vec<Option<f64>>`.
`assemble` went 8–9 → 1.5–2 ms on the 64-group chunk, and **39.6 → 3.5 ms**
on a five-halflife grid, where it had been a tenth of the wall. Below
`PAR_MIN_ROWS` (4096) the fields build on the calling thread (§12).

*Since task 86 (2026-09-17)* a field finishes as an Arrow `Float64Array`,
built by `PrimitiveArray::new` over its values and packed validity bits
(`F64Column::finish_array`, `crates/online-polars/src/column.rs:137`–`140`),
not through `Float64Chunked::from_vec_validity`. §13 records the packed
bits.

### P11 — Extraction: per column, per arrow chunk, and integer keys as themselves (2026-09-04)

*Done.* Three changes:

| change | before | now |
|---|---|---|
| a spec's columns | one spec meant one thread copying every column | read in parallel from `PAR_MIN_ROWS` up |
| a column with more than one arrow chunk, which is what `collect_batches` hands over when a batch spans two parquet row groups | fell to the per-element path when `cont_slice` failed | copied chunk by chunk with `downcast_iter`. Since task 86 the adapter rechunks such a column once, measured at no cost ([docs/REVIEW-2026-09-17.md](REVIEW-2026-09-17.md) §4) |
| an integer group key | cast to `String` and hashed | bucketed by its value (`integer_groups`, `PlHashMap`: foldhash rather than SipHash), since its text *is* its decimal. `integer_group_keys_match_the_string_cast` pins the two paths to the same keys and output |

Measured: `group` 8.6 → 1.6 ms on the 64-group chunk. On the 12M-row README
workload, `group` went 0.20 → 0.09 s and `extract` 0.26 → 0.19 s.

*Since task 75 (2026-09-08)* a wide chunk also reads its columns in
parallel from 65,536 feature values, `PAR_MIN_CELLS`, whatever its height
(`crates/online-polars/src/bank.rs:302`, `:347`–`348`).

**Rejected, with reasons:** see §5, which records what was rejected up
front and what was rejected after measuring.

## 4. Where it ended up

Same machine, same build, same scripts as §1.

| case | before | after | ratio |
|---|---|---|---|
| k=5, 1 group | 2,093,719 | 4,980,607 | 2.4× |
| k=20, 1 group | 1,308,792 | 2,634,627 | 2.0× |
| k=20, 64 groups | 2,601,719 | 5,132,169 | 2.0× |
| single stream, 5-halflife grid | 458,500 | 955,662 | 2.1× |
| 8 specs × 1 group (wall) | 783.8 ms | 201.7 ms | 3.9× |
| CLI, 3M rows × 20 feat × 32 groups | 2.17 s | 1.76 s | 1.23× |
| expression, 1000 groups | 511,237 | 727,089 | 1.4× |

The sections at k=20 over 64 groups:

| section | before, ms | after, ms |
|---|---:|---:|
| `extract` | 13.0 | 1.7 |
| `group` | 15.3 | 4.6 |
| `process` | 40.6 | 26.7 |
| `assemble` | 8.0 | 6.0 |

Thread scaling, 400k rows, k=20, 64 groups:

| threads | before | after |
|---|---|---|
| 1 | 435,792 | 719,320 |
| 2 | 702,049 | 1,303,530 |
| 4 | 1,021,413 | 2,335,795 |
| 8 | 1,363,018 | 3,881,813 |
| 10 | 1,414,097 (3.2×) | 4,478,262 (**6.2×**) |

**What now limits it.** At k=20 on a single stream, the bank reaches 2.63M
rows/s against the pure core's 5.73M, so the plumbing costs ~2.2× rather
than the original ~4.4×. What remains is the per-row gather of features
into the scratch buffer, and the `Step` the model returns: real work at
this point, not bookkeeping. The recursion inside one instance stays
sequential by construction (§2 item 7), and everything around it is now
parallel.

## 5. Rejected, and why

Recorded so each omission is a decision. The first three below were
rejected up front, and the next three *after* measuring, which is the more
useful kind. The last was a target set aside, then reopened and met. Later
sections rejected more, each where it was measured, and the second table
indexes them.

| rejected | when | outcome | where |
|---|---|---|---|
| a custom global allocator in the Python extension | up front | rejected | |
| GPU or BLAS batching for the solves | up front | rejected | |
| parallelizing the recursion itself | up front | rejected | |
| typed-builder assembly (part of P4) | after measuring | reversed 2026-09-04 by P10 | P4, P10 |
| `lto = "fat"` and `-C target-cpu=native` | after measuring | rejected | P7 |
| manual SIMD in the co-moment update | after measuring | rejected | P7 |
| chasing polars' `.over()` overhead (P5's target) | set aside, then reopened | met: docs/IMPROVEMENTS.md P1 | P5 |

**A custom global allocator in the Python extension.** Python and polars
own that arena. Swapping it under them is a compatibility risk, for a
benefit P1 already took by removing the allocations instead of making them
cheaper. The CLI could adopt mimalloc independently if a profile ever
justifies it. (§6 installs `PolarsAllocator`, which routes the extension's
allocations through py-polars' own allocator: that arena, not a
replacement for it.)

**GPU or BLAS batching for the solves.** At k ≤ 50 a solve is
microseconds. Dispatch would cost more than the work, and `solve_every` has
already made solving 6% of the k=20 budget.

**Parallelizing the recursion itself** (speculative execution, parallel
prefix over the decay). Row *i*'s state depends on row *i−1* exactly, and
every approximation trades the exactness that chunk invariance and
out-of-sample-ness are built on. It is not a speed/complexity trade but a
correctness one.

**Typed-builder assembly (part of P4).** Measured first: P1's flat buffers
had already taken `assemble` to ~5.5 ms of a 39 ms chunk, and the rest is
`Series` construction polars needs anyway. Not worth the code. *Reversed
2026-09-04 (P10): at 14 threads the 5.5 ms had become a fifth of the wall,
and a grid of instances assembled on one thread for 40 ms. Per-field
scatters into `Vec<f64>` + bitmap took those to 1.5 and 3.5 ms. The
2026-09-02 verdict was right about the single-thread, single-instance chunk
it measured.*

**`lto = "fat"` and `-C target-cpu=native` (P7).** Measured across the six
`core_bench` cases: +3.2%, +1.3%, −1.2%, −1.0%, −2.1%, +0.4% for fat LTO, a
wash against a slower build. `target-cpu=native` was −3% at k=20, and would
also cost wheel portability. Both fail the ≥3% bar the plan set for
itself.

**Manual SIMD in the co-moment update.** The throughput curve rules it out
without a disassembler. The update is O(k²), so k=5 → k=20 is 12.3× the
arithmetic for 2.1× the time, and k=5 → k=50 is 72× for 6.0×. Per element
the wide cases are cheaper, which is what auto-vectorized loops look like.

~~**Chasing polars' `.over()` overhead (P5's target).**~~ Reopened and
closed by docs/IMPROVEMENTS.md P1. The "flat from ten groups on" curve was
the signature of serial per-group evaluation, and one packed struct input
puts the plugin on polars' parallel path. See P5 in §3.

### Rejected later, where each was measured

| rejected | verdict | § |
|---|---|---|
| a column-oriented back-substitution in `rls` | the ideal chain, but strided loads: 2.25M / 774k rows/s, no better | §8 |
| two interleaved partial sums in the same solve | 2.19M / 831k, within noise of the one-line version | §8 |
| a decay pass fused into `rls`'s rotations | bit-identical, and 8% faster at k ≤ 20, but 4% slower at k=50 for a longer skip path: not worth its shape | §8 |
| scanning in Rust for the Python path | replaced by py-polars' readers: the Rust-side `scan_csv` was ~6× slower than `pl.scan_csv` | §10 |
| rechunking the engine's two-piece batches on the reader thread | tried, and changed nothing | §10 |
| a filter run inside the IO-plugin source | 0.81 GB against 2.54 at 12M rows, but a second `filter` whose only difference is memory | §11 |
| fewer pipelines for the input query alone | 0.65 GB at 2 pipelines, 0.29 at 1, but 2.7 → 3.9 → 7.1 s | §11 |
| a row-major `ChunkOut` | the run split gets the same cache behaviour without changing the reader in `assemble` | §13 |
| a fused single pass to pack `F64Column`'s validity | measured the same within noise, so the simpler code stays | §13 |
| a zero-filled `pred` | NaN is the "never written" sentinel, and zero is a value | §13 |
| a two-pass moment per chunk for the data summary | ruled out by chunk invariance: different bits under different chunkings | §13 |
| E48's symmetric-half Gram update | not bit-identical, and 49% to 107% slower | §14 |
| a triangular Gram | would still be 430 MB at `k = 10,000`, so it does not change the answer; not done | §19 |
| comparing before storing `min` and `max` in the data summary | measured no change, and dropped | §20 |
| a column-wise layout for the data summary | a state layout change for perhaps 4 µs of 20; not done | §20 |

## 6. The allocator (2026-08-31)

**The extension installs pyo3-polars' `PolarsAllocator`, one line of
`#[global_allocator]`, and it measured up to 43% faster.** It was found
while answering "won't two copies of Polars in one process go wrong?". The
answer is no: the Arrow C Data Interface keeps each side freeing its own
memory. But the investigation turned up that pyo3-polars ships
`PolarsAllocator`, which routes a plugin's allocations through py-polars'
own allocator, and we were not installing it. That left two allocator
arenas in one process, neither able to reuse the other's pages. The line is
in `crates/online-py/src/lib.rs:28`.

It was attributed by A/B/A on an otherwise identical tree. Two changes had
landed between the first two measurements, and the gain was too large to
assign by assumption. The A/B/A read **5,716,246 → 8,165,418 → 5,614,830**
rows/s at k=5, which bounds machine drift at ~2%.

| case | before | after | change |
|---|---:|---:|---:|
| ew_ridge k=5 | 5.62M rows/s | **8.02M** | +43% |
| ew_ridge k=20 | 2.80M | **3.26M** | +16% |
| ew_ridge k=50 | 855k | **900k** | +5% |

The gradient is the tell. The gain is largest at small `k`, where
per-chunk allocation is a big share of the work, and smallest at `k=50`,
where the O(k³) solve dominates and the allocator barely matters.
Performance was not the reason for the change; it is the reason it is
recorded here.

## 7. Bugs found by this review

| finding | what it did | now | pinned by |
|---|---|---|---|
| **null-session sentinel collision**, fixed alongside this document | null session values were hashed as the string `"\0<null>"`, so a session literally named that was *the same session* as null: the T-E2 bug one layer down | null hashes to a value no string can produce (colliding strings are nudged), and state files resume unchanged | `test_a_session_named_like_the_null_sentinel_is_not_null` |
| two 64-bit hash-collision residues, which remain by design | two distinct session names, or one name against null, collide with probability ~2⁻⁶⁴ per pair (FNV-1a) | accepted: a collision merges two sessions, it does not corrupt state | |

## 8. Refresh (2026-09-02), and what `rls` costs now

### The refresh

§4's table was the state after P1–P8. §6's allocator fix landed *after* it,
and the README's table predated both. Everything was re-measured with the
same commands (`scripts/benchmark.py --markdown`,
`scripts/scaling_bench.py`), two runs agreeing within 2%:

| configuration | README claimed | measured now | change |
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

The rest of the refresh:

| measure | before | now |
|---|---:|---:|
| grouped, k=20 over 64 groups | 5.13M rows/s | **6.04M rows/s** |
| eight single-group specs over 300k rows, in one bank | 201.7 ms | **118 ms**, against 685 ms run one at a time |
| expression under `.over(g)`, 1000 groups | | **12.2M rows/s** |

Thread scaling on the same grouped workload now runs to every core the
machine has. The script stopped at 8, which on a 14-core box hides the row
that matters:

| threads | rows/s |
|---|---:|
| 1 | 916k |
| 2 | 1.65M |
| 4 | 2.86M |
| 8 | 4.75M |
| 14 | 6.03M: **6.6×** |

### `rls` and its square-root form (C5)

**`rls` is the exception, and the square-root form C5 introduced is the
cause.** An A/B on the model arithmetic alone attributes it.
`crates/online-core/examples/rls_bench.rs` compiles unchanged against
`50c1a38^`, the commit before the square-root rewrite, so nothing but
`Rls::step` differs:

| k | covariance form | square-root (QR) form | ratio |
|---|---:|---:|---:|
| 5 | 13,915,222 | 7,226,608 | 0.52× |
| 20 | 4,390,434 | 1,700,789 | 0.39× |
| 50 | 996,592 | 491,013 | 0.49× |

Both forms are O(k²) per row, and the QR form does more of it: `k` Givens
rotations, each with a square root. What that time gets is what C5 was
after: the covariance form died deterministically on one extreme row. It is
worth the time, and now stated rather than implied.

### Where the QR form's time actually went (2026-09-02)

**The back-substitution was half the row, though a quarter of the flops,
because it was latency-bound.** Skipping the per-row back-substitution took
k=20 from 1.75M to 3.15M rows/s, and k=50 from 511k to 1.26M. That was for
timing only: the numbers are meaningless without the solve.
`beta_i = (u_i − Σ_{j>i} R_ij β_j) / R_ii` summed the row from `j = i+1`
upward. So every row's chain began with the coefficient that had just been
solved, and could not start until it had: `k` chains of `k/2` dependent
subtractions, queued. Summing from the far end instead (`j = k−1` downward)
makes every term but the last independent of the previous row, so the
chains overlap in the pipeline. One `.rev()`:

| k | before | after | ratio |
|---|---:|---:|---:|
| 5 | 7,244,780 | 7,360,936 | 1.02× |
| 20 | 1,753,624 | 2,118,547 | 1.21× |
| 50 | 510,829 | 799,909 | 1.57× |

At bank level, `rls` k=20 went 1.63M → **1.93M rows/s** (README table). It
is a rounding-level change, a different summation order. Measured on the
golden signatures it is 1e-15 relative: the Rust one moved by one ulp in
two of three values, and the Python pipeline by up to 1.2e-15 absolute.
Both are inside the tests' 1e-12 tolerance, so no pinned value was
regenerated.

The same experiment rejected three other changes:

| variant | k=20 | k=50 | verdict |
|---|---:|---:|---|
| a column-oriented solve | 2.25M | 774k | the ideal chain, but strided loads: no better |
| two interleaved partial sums | 2.19M | 831k | within noise of the one-line version |
| a decay pass fused into the rotations | | | bit-identical, and gained 8% at k ≤ 20, but lost 4% at k=50 for a longer skip path: not worth its shape |

What remains is the rotation chain itself: `k` dependent
`sqrt`-and-divide pairs per row. That is the QR form's structure, not a
codegen artefact. A square-root-free (fast) Givens variant would shorten
it, and is the obvious place to look if `rls` throughput ever becomes the
constraint. It would add a rescaling step, whose stability would have to be
argued as carefully as C5 was.

## 9. `predict` (E31, 2026-09-02)

Scoring is the learning loop with `learn = false`: the same
`run_instance`, the same per-row arithmetic up to the model call, and then
`predict` in place of `step`. Measured on the bank benchmark's million-row
single-stream case, `ewridge`, the two paths on the same frame:

| k | `fit_predict` | `predict` | speedup |
|---|---:|---:|---:|
| 5 | 8,003,000 | **14,270,000** | 1.8× |
| 20 | 3,300,000 | **9,430,000** | 2.9× |

The gap is the update. At `k=20` a step is a rank-one update of the
`k × k` co-moments plus a solve every `solve_every` rows, and a prediction
is a dot product. The learning path itself did not move:
`scripts/benchmark.py` gives 8.99M / 3.60M / 990k rows/s at k=5/20/50,
against §8's 8.96M / 3.62M / 961k. The only change on it is that
`ewridge`'s `step` now gets its prediction from `predict` instead of an
inlined copy of the same loop.

## 10. The runner: every format, every source (E32, 2026-09-02)

*Read since task 83 (2026-09-17):* `po.run`, the Python entry to this
runner, was removed; the numbers below stand as the record of what they
measured. The pipeline they measured runs today as the `online` CLI (the
Rust reader, `Input::Lazy`) and, in Python, as `ModelBank.fit(lf)` /
`fit_predict_batches(lf)`. Those have the same `collect_batches` read and
the same bank, without the writer thread. `Input::Batches` remains for a
Rust caller with frames of its own.

### The pipeline

The runner is a three-stage pipeline, with one chunk in flight per stage: a
reader thread, the bank on the calling thread, and a writer thread. E32
made the reader pluggable:

| input | what it is | who fed it |
|---|---|---|
| `Input::Lazy` | a polars plan, read by the streaming engine in `chunk_rows` batches | the CLI, and Rust callers |
| `Input::Batches` | an iterator of frames the caller already has | `po.run`: py-polars read (`collect_batches`), and Rust fitted and wrote |

### Every format

Measured on the P6 file: 3M rows × 20 features × 32 groups, `ewridge` k=20
with a clock, weights and `min_periods`, `chunk_rows=100k`, best of 3,
through `po.run`:

| input → output (same format) | rows sorted by clock (groups interleaved) | rows sorted by group |
|---|---:|---:|
| parquet | 0.71 s | 2.15 s |
| ipc | 0.59 s | 2.05 s |
| csv | 0.90 s | 2.15 s |
| ndjson | 1.14 s | 2.33 s |

`ONLINE_TIMING=1` prints one line per run beside the per-chunk ones. In
it, `read_wait` and `write_wait` are the bank thread's slack, and
`writer_busy` is the writer's own time; *How the numbers are made* lists
every field. The line says where each case sits.

**Interleaved groups are I/O-bound, and the I/O is overlapped.**

| format | the run's timing line | what it says |
|---|---|---|
| parquet | `read_wait=0.07s bank=0.67s write_wait=0.00s writer_busy=0.66s total=0.77s` | the parquet writer is as busy as the bank, and hidden behind it |
| ipc | `writer_busy=0.11s`, total 0.62 s | the floor |
| csv | `bank=0.70s write_wait=0.13s writer_busy=0.86s`; the CSV *reader* at `read_wait=0.02s` | the one write-bound case; the reader is no longer the problem |
| ndjson | the bank reads 1.04 s, against 0.64 s alone | the parallel NDJSON serializer competes for the same rayon pool as the bank's per-group tasks: contention, not a defect, and the total still beats serial |

Since 2026-09-04 the serializer runs on polars' pool and the bank on its
own (docs/PLAN.md §11a), so they now compete for cores rather than for one
pool.

**Group-sorted rows are bank-bound in every format:** `bank=2.07s` of
2.16 s, `read_wait=0.05s`. A 100k chunk of group-sorted rows holds one or
two groups, so the bank's per-group parallelism has nothing to spread. The
same file interleaved is 3× faster through the same code. The fix is a
chunk that spans many groups: `chunk_rows=500_000` takes the sorted parquet
from 2.15 s to **1.01 s**. The trade is memory (three chunks in flight),
which is what `chunk_rows` was always for.

### Why py-polars reads for the Python path

**py-polars' CSV reader is ~6× faster than the one a stable Rust toolchain
can build.** The first cut scanned in Rust for every caller. On the CSV
that was 1.72 s with `read_wait=1.06s`: the bank sat waiting for the
parser. The SIMD CSV parser in polars-io is behind a feature that needs a
nightly compiler. The py-polars wheels have it, and a stable toolchain
cannot, so the Rust-side `scan_csv` was ~6× slower than `pl.scan_csv` on
the same file.

Reading with py-polars and handing the frames across took the CSV run to
0.90 s. The hand-off is a per-series Arrow C FFI export. It is cheap,
though it rechunks, so a 120-chunk CSV batch arrives single-chunk, and the
bank is ~10% slower on multi-chunk frames anyway. As a side effect, it gave
`po.run` every source py-polars can stream: globs, cloud URLs, a query with
a filter or a UDF, a `DataFrame`, an iterable of frames. The GIL is
released for the run, and reacquired only to take each batch.

The CLI still scans in Rust, so on a stable toolchain **a large CSV is
faster through `po.run` than through `online`**. On parquet the CLI is ~15%
behind too: 0.83 s / 2.55 s, against `po.run`'s 0.72 s / 2.12 s. Its timing
line says why it is not the reader: `read_wait=0.06s bank=2.42s`, against
`bank=2.07s` in Python, for the same code on the same chunks. Rechunking
the engine's two-piece batches on the reader thread was tried, and changed
nothing. What differs between the two processes is the allocator. The
extension allocates through py-polars' jemalloc (`PolarsAllocator`, §6,
which measured it at +16% for exactly this `k`), and the binary through the
system malloc. Giving the CLI its own would statically link one, which
rule 12 of [CLAUDE.md](../CLAUDE.md) keeps a decision rather than a tweak.

### What the custom parts are worth

Asked whether the runner's hand-written pieces could go, each was measured
against the plain polars call it replaces, in the CLI on the interleaved
file:

| custom part | replaces | measured |
|---|---|---|
| the three-stage pipeline | C7's one-thread loop (below) | 1.6–2.7× |
| `ParquetSink`'s parallel page encoding | `BatchedWriter::write_batch` | 0.86 s against 1.55 s: the serial encode becomes the pace |
| `ndjson_write`'s slice-per-thread | polars' NDJSON `BatchedWriter` | a 4× win under jemalloc (`po.run`: 1.04 s against ~4 s), and a defect under the system allocator (the CLI: 4.8–54 s against a steady 4.1 s); docs/IMPROVEMENTS.md C8 has the diagnosis and the two fixes |

### Why the bank is not a polars node

**The pipeline is what polars-stream builds for its own ordered, stateful
operators, and it does not offer that to a user function.** Checked in
polars-stream 0.55.2 (`physical_plan/lower_ir.rs`, `lower_expr.rs`,
`lower_group_by.rs`, `nodes/`):

| what a query uses | how polars-stream 0.55.2 lowers it | memory |
|---|---|---|
| `df.rolling(index_column, period)`, and `expr.rolling(index_column=..)` with no `group_by` keys | a `RollingGroupBy` node. It receives morsels serially, insists the index is sorted, keeps only the rows still inside the lookback (`buf_df` is sliced as windows retire), and hands each batch of windows to parallel evaluators | streams |
| `ewm_mean`/`ewm_var`/`ewm_std`/`ewm_sum`, `cum_*`, `forward_fill`, `rle`, and a group-by over sorted keys | nodes of the same kind | streams |
| `rolling(.., group_by=..)` and `group_by_dynamic(.., group_by=..)` | an in-memory fallback (`lower_group_by.rs:737`) | collects first |
| `rolling_mean(window_size)` and any other non-elementwise function, and `.over` with an `order_by` | | collects first |
| every plugin or Python UDF expression | a `columnar-function` node: an in-memory sink per input that calls the function once on the whole column and becomes a source (`nodes/columnar_function.rs`, flagged `is_memory_intensive_pipeline_blocker`) | collects first |
| a user function that *is* elementwise | a `Map` node, per morsel and concurrent, which is why it cannot carry state | streams |

`RollingGroupBy` is a serial fold feeding a parallel stage, which is this
runner's shape. So `online.ewridge(..)` as an expression was O(data) in
either engine. An O(state) pass over a stream needs a fold the engine does
not expose: the reader → bank → writer pipeline here. It also gets what
`RollingGroupBy` gets, a serial stage and a parallel one, with the parallel
one across groups.

### Before and after (C7 → E32)

The C7 runner ran `lf.slice(offset, chunk_rows).collect()` per chunk on the
calling thread, re-planning the scan thirty times and overlapping nothing.
Same machine, same file, same spec, best of 3, the C7 wheel and binary
built from `62d74a1` in a worktree:

| parquet → parquet | C7 `po.run` | E32 `po.run` | C7 `online` | E32 `online` |
|---|---:|---:|---:|---:|
| groups interleaved | 1.93 s | **0.72 s** | 2.14 s | **0.83 s** |
| group-sorted | 3.30 s | **2.12 s** | 3.84 s | **2.55 s** |

The bank, the plugin and `online-core` are untouched by E32:
`git diff --stat 62d74a1 HEAD -- crates/` names only the runner, the CLI
and the bindings. So the `fit_predict` numbers in §8 stand.

What E32 added is size, all of it polars' ipc, csv and ndjson readers and
writers:

| artifact | C7 | E32 |
|---|---:|---:|
| the extension | 59.3 MB | 61.9 MB (+4.4%) |
| the extension, gzipped | 18.8 MB | 19.8 MB |
| the wheel | 19.8 MB | 20.8 MB |
| the CLI | 51.0 MB | 53.1 MB |

No new crate came in outside polars, and no new `-sys` crate: `Cargo.lock`'s
are the four C7 had.

### Writers

| format | how it is written |
|---|---|
| parquet | through `BatchedWriter` with parallel page encoding, one row group per chunk, so 30 row groups regardless of the input's layout |
| ipc | batched, one record batch per chunk |
| csv | batched, with structs flattened to `<spec>.<field>` and lists to JSON text (`format!("{v}")`, shortest round-trip, so `str.json_decode` gives the bits back) |
| ndjson | serialized in parallel slices of each chunk, and written in order |

Output goes through a temporary sibling and a rename, as `save` does.

## 11. Memory: which surface is O(data) (2026-09-02)

*Read since 2026-09-17:* the `po.run` rows are the Python runner, removed
in task 83. Its in-process successor is `ModelBank.fit(lf)`, which reads
the same way as the bank loop measured beside it. The expression rows are
the plugin, removed in task 85 for exactly the O(data) this section
measured. Both stand as the record. The current guidance is the README's
[Tuning memory with Polars' own settings](../README.md#tuning-memory-with-polars-own-settings).

The claim on the README's first line is that the bank and the runner run
on data that does not fit in memory. The note in §10 that the plugin is
O(data) raised the question of whether everything is. **Measured, the plugin was
O(data), and nothing else is.** The measurement used the §10
file, and the same file doubled and quadrupled by appending itself with the
clock shifted. The groups stay the same 32, so the state is identical and
only the data grows. The spec is `ewridge` k=20 with clock, weights and
`min_periods`, parquet in, parquet out, `chunk_rows=100k`, 14 threads:

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

### The plugin was O(data); nothing else is

O(data) here means what it says: the whole input is resident at once.
Polars calls a plugin function once with the entire column, and has to have
the column first (§10). From 2026-09-03 every expression call warned with
`InMemoryExpressionWarning` and named the plan (PLAN §6); the warning went
with the plugin in task 85.

The sharpest form of the measurement is the same query with one expression
swapped:
`scan_parquet(f).with_columns(<expr>).sink_parquet(out, engine="streaming")`,
with the prefetch pinned to 1 row group so the reader is not part of the
number.

| `<expr>` | 3M rows | 12M rows |
|---|---:|---:|
| `pl.col("y0") * 2` | 0.51 GB | 0.51 GB |
| `online.ewridge(...).over("group")` | 1.85 GB | 7.30 GB |

Same engine, same file, same sink: one expression streams flat, and the
other holds three times the frame (2.4 GB at 12M rows). That is the
collected input, the packed struct the plugin is handed, the `.over`
gather, and the output column. `collect()` and
`sink_parquet(engine="streaming")` are within 1% of each other. Pinning the
prefetch takes the non-plugin query from 1.86 GB to 0.51 GB, and does
nothing for the plugin. Its time is linear too (3.2 / 6.7 / 14.4 s).

**The CLI is flat at 0.73 GB from 3M to 12M rows. `po.run` and the bank
loop *report* a creeping number, and the creep is the allocator, not live
data.** The extension allocates through py-polars' jemalloc, which keeps
freed pages for ten seconds (`dirty_decay_ms`), longer than these runs. So
the peak is the high-water mark of everything ever allocated. With
`_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0` the same runs are
flat (0.87 / 0.86 GB), and about 20% slower for the purging. The CLI's
system malloc returns pages at once, which is why its trace drains and
Python's does not. This is not a knob to set in production. It explains
the reported number, and why the CLI and `po.run` differ here.

### What the constant is

**Almost none of the constant is ours.** A 100k-row chunk of this file is
~20 MB, and the pipeline holds three. The bank's state for 32 groups at
k=20 is under a megabyte, and the output writer holds one chunk. The rest
is polars' parquet reader. The streaming engine prefetches row groups ahead
of the consumer: `row_group_prefetch_size: 96` on 14 threads, held back
only by a 448 MB byte budget. This file's row groups are 262k rows, so the
reader front-loads ~0.7 GB of decoded rows before the bank has consumed one
chunk. The CLI's trace peaks at the start and drains from there.

The settings that move it are all Polars' own:

| setting | what it sizes | measured |
|---|---|---|
| `POLARS_ROW_GROUP_PREFETCH_SIZE=1` | the row groups read ahead | the CLI on 12M rows, 0.75 GB → **0.15 GB at the same 3.2 s**: the bank, not the reader, is the bound, and a local SSD needs no read-ahead. `=2` is 0.28 GB. The bank loop and `po.run` keep a few more chunks in flight in py-polars' `collect_batches` and the FFI hand-off: 0.31 / 0.46 GB with prefetch 1 |
| `POLARS_MAX_THREADS` | the same term, because the prefetch is sized from the thread count | 4 threads: 0.45 GB; 1 thread: 0.18 GB |
| `POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET` | the byte budget, in compressed bytes, which for a memory-mapped local file cost nothing | lowering it changed nothing here |
| `POLARS_CSV_CHUNK_PREFETCH_LIMIT`, `POLARS_NDJSON_CHUNK_PREFETCH_LIMIT` | CSV's and NDJSON's own read-ahead | not measured |

**`chunk_rows` is the term the caller owns.** `po.run` at 500k is 1.38 GB
on 3M rows, against 0.95 GB at 100k: three chunks in flight, each five
times larger. The advice in §10 to raise it for group-sorted input is a
memory trade, which is what the knob was for.

### The query-shaped trap, and the fix (E33)

The plugin row above is what a user got for writing the natural thing, a
`LazyFrame`, the expression and `sink_parquet`, and expecting online
processing. It could not be fixed in the plugin. Polars' contract for a
user expression is the whole column, or elementwise, which is stateless and
unordered. An ordered, stateful per-morsel node is something only
polars-stream itself can add: `AnonymousScan` on the Rust side is `todo!()`
in 0.55.2's `lower_ir.rs`.

**What can be fixed is where the bank sits in a plan.**
`lf.online.fit_predict(specs)` registers the bank as a polars **IO-plugin
source** (`register_io_source`), the kind of node the engine pulls batches
from. The source runs `collect_batches` over the input, and
`ModelBank.fit_predict` per chunk. The last two rows of the table above are
that:

| | measured |
|---|---|
| output | **bit-identical to `po.run`'s output** |
| time, 12M rows | 2.8 s (`po.run` 2.9 s, the bank loop 2.5 s, the plugin 14.4 s) |
| memory | flat: the reported creep is jemalloc's again, 0.78 GB live at 12M rows and 0.37 GB with the prefetch at 1 |
| pushdown | the plan's filter, projection and `head` are pushed into the source, and a selection reaches the input scan |
| read-ahead | the engine reads a few morsels ahead of the bank (7 of 100 input batches were requested before a `head(10)` stopped the plan), and tears the input query down with the plan |

### What comes before the bank

*(2026-09-02, and corrected three times the same day from the
polars-stream source: the stage, then what fills its slots, then who
decides what a slot holds.)*

**An upstream `filter` or `with_columns` costs a bounded window, not the
data.** The source reads its input with `LazyFrame.collect_batches`, so the
input plan runs in the streaming engine, and its memory is polars'.
Measured on the same files, prefetch 1,
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

`sort` and `.over` are O(data): they are pipeline breakers, and the whole
frame exists before the first row comes out (`.over` measured on: 8.4 GB at
36M rows). The `filter` and `with_columns` rows are something else. The
first reading of them here was wrong: "`collect_batches` stops applying
backpressure once a filter is in the plan and buffers the filtered result".
What they hold is a **window that is bounded in morsels, not bytes**, and
on this machine the window is bigger than the 12M-row file.

To isolate it, there was no bank in the process:
`scan.<shape>.collect_batches(chunk_size=100_000)`, iterated with a
`time.sleep` per chunk, 14 threads unless noted. The 36M-row file is the
12M one three times over.

| peak footprint | 12M rows | 36M rows |
|---|---:|---:|
| plain scan, `sleep(0.02)` | 0.67 GB | 0.64 GB |
| `filter`, `sleep(0.02)` | 2.54 GB | **3.11 GB** — a flat plateau, then a drain |
| `filter`, `sleep(0.02)`, `POLARS_MAX_THREADS=2` | 0.71 GB | 0.74 GB |
| `with_columns`, `sleep(0.1)` | 2.60 GB (the whole file) | **4.70 GB** (the file is 7.8 GB) |
| `mean().over("group")` | 2.97 GB | 8.39 GB — a straight ramp |

The filter at 12M rows, against the thread count:

| threads | peak footprint |
|---|---:|
| 1 | 0.46 GB |
| 2 | 0.71 GB |
| 4 | 1.08 GB |
| 8 | 1.89 GB |
| 14 | 2.54 GB |

That is **0.2 GB per thread**. The profile has flat plateaus at ≤ 4
threads, and drains above that, where the window exceeds the file (96 row
groups of 26 MB; at 14 threads the window is 98).

### Where the window comes from

**Backpressure in polars-stream 0.55.2 counts *morsels per pipe*, and the
count is multiplied by the number of pipelines (= threads) at every serial
→ parallel → serial transition.** In `pipe.rs` that is a distributor with
4 slots per lane, one morsel in flight per lane, and a linearizer with 4
slots per lane when order is kept. Through a parallel compute node such as
`with-columns`, that makes 9 per lane. That is the `with_columns` row, and it shrinks
with the morsel. `pl.Config.set_streaming_chunk_size(25_000)` takes the
isolated `with_columns` probe from 2.56 to 1.32 GB, and to 1.03 at 10,000
rows. It is public API. The env var is `POLARS_STREAMING_CHUNK_SIZE`, and
`POLARS_IDEAL_MORSEL_SIZE` is silently overwritten by the unset legacy name
in 0.55.2 (pola-rs/polars#29021).

**The pushed-down `filter` is not a filter cost at all.** The parquet reader
applies the predicate itself (`FULL_FILTER`, `row_group_decode.rs`). The
predicate's columns are decoded first, the mask is built, and the other
columns are decoded through it. So the reader's output carries the
predicate columns *first*, and unless those columns already lead the
projection, that order no longer matches it. The scan's post-apply stage is
then `Initialized` with column selectors instead of `Noop`
(`apply_extra_ops.rs`: `is_input_passthrough` is
`input_index == output_index` for every column). Every morsel goes through
`distributor_channel(num_pipelines, 1)` → one worker per lane →
`MorselLinearizer::new(num_pipelines, 4)` (`post_apply_extra_ops.rs`), to
have its columns permuted. The permutation is a zero-copy `select`, through
a pipeline holding about 7 morsels per lane: 1 in the distributor's buffer,
1 in the worker, 4 in the linearizer's channel, and 1 in its heap.

**What a morsel is here is the planner's choice, not the reader's.** A sink
directly above the scan (`collect_batches`, `sink_*`) sets
`disable_morsel_split` on it (`physical_plan/lower_ir.rs`). The reader
then emits whole row groups, and the chunk size does nothing: 2.59 GB at
25,000 rows, 2.63 at 10,000. With a compute node above the scan, the reader
splits to `ideal_morsel_size` rows *before* this stage (`parquet/init.rs`).
The stage then costs 14 × 7 × 3.2 MB ≈ 0.3 GB at 25,000-row morsels, and
0.12 at 10,000. Isolated, on 8M rows × 16 columns with 50,000-row
groups, `filter(x1).with_columns(..)` peaks at 1.11 / 0.60 GB, against
0.80 / 0.48 with the predicate on the first column. And 14 lanes × 7 ×
26 MB ≈ 2.5 GB, plus the reader's own, is the 2.5–3.1 GB measured: 0.2 GB
per thread.

Three checks, as isolated probes at 12M rows:

| check | the probe | peak footprint |
|---|---|---|
| 1. the predicate column moved to the front of the projection | `scan.select(["vol", *rest]).filter(pl.col("vol") > 0)` makes the post-apply stage `Noop` (`POLARS_VERBOSE=1` says so) | **0.38 GB**, and 0.58 GB with the bank behind it |
| 1, on the 8M-row file at 14 threads: a predicate on the first column to begin with is `Noop` as it stands | plain scan, `filter(x0)`, `filter(x1)` | 0.30 GB, 0.25, 0.93 |
| 2. the slots hold what the filter *keeps* | a predicate keeping 100 / 99 / 90 / 50 / 10 / 1 % of the rows | 2.52 / 2.48 / 2.31 / 1.23 / 0.23 / 0.12 GB: the 2.5 GB above is the keep-everything worst case |
| 3. predicate pushdown off | the filter is an ordinary compute node | 1.96 GB, and 1.33 with 25,000-row morsels |

A plain scan has no parallel stage between the reader and the sink (serial
→ serial, capacity 1 each way), hence 0.65 GB at any consumer speed. The
filter *after* the bank is applied inside the IO-plugin source, and never
enters a `multi-scan` stage at all. **The window is O(threads × row-group
rows × kept columns), and O(1) in the data.**

That the post-apply pipeline exists only to restore column order is
polars' to fix: a reader emitting its columns in projection order would
make the stage `Noop`. Reordering one's own projection to dodge it is not
something to build on. Upstream, as checked 2026-09-02:

| upstream | what it says |
|---|---|
| pola-rs/polars#28912 | the stage's memory is known, from a multi-file scan with an out-of-order `select`, which the maintainer traced to "the morsel distributor in PostApplyExtraOps" |
| PR #29049 (merged after 1.44.1) | divides the stage's lane count by the number of concurrently scanned files, which for one file is 1: `stage_pipelines = num_pipelines.div_ceil(max_concurrent_scans)`, so a single-file scan is unchanged. The PR calls in-lining the permutation "still worth pursuing" |
| #28569 (accepted) | asks more generally that the parquet source keep a bounded number of decoded morsels outstanding under sink backpressure |
| #25242 | says `set_streaming_chunk_size` has no effect in the new engine. On 1.44.1 it does (`ideal_morsel_size: 25000` in the verbose log, and the numbers above) |

That a pushed-down predicate on any but the leading column trips the same
stage on a single file, with these numbers, is not reported there.

### What shrinks the window

| lever | what it does here |
|---|---|
| fewer threads (`POLARS_MAX_THREADS`) | takes 0.2 GB per thread off the filter's window, per the table above |
| keeping fewer rows, and smaller parquet row groups | shrinks the pushed-down filter's slots, which are row groups; polars' own default when writing is 262,144 rows, twice these files' |
| `pl.Config.set_streaming_chunk_size`, as above | shrinks a compute node's morsels, and through them the pushed-down filter's too, once a compute node above the scan makes the reader split |
| `POLARS_DEFAULT_DISTRIBUTOR_BUFFER_SIZE` / `POLARS_DEFAULT_LINEARIZER_BUFFER_SIZE` | nothing for the pushed-down filter: its 1 and 4 are literals |
| a narrower projection: `keep_columns=`, or a `select` before the bank | shrinks every row group |

`keep_columns=` was `po.run`'s keyword. Since task 83 it survives as the
command line's `keep_columns` setting (`crates/online-polars/src/runner.rs:172`,
`:319`–`324`), and in Python a `select` does the same. Three things did not
shrink it: the allocator (2.44 GB with `dirty_decay_ms:0`),
`maintain_order=False`, and `lazy=True`.

**On polars 1.x, `sink_batches` with the default `engine="auto"` is not the
streaming engine at all.** polars-lazy maps `Auto` to `InMemory`. File
sinks are handed to the streaming executor from there, but the callback
sink is not (`polars-mem-engine/planner/lp.rs`). So it collects its input
and then chunks it: 2.77 GB on the plain scan, ramping, against 0.49 GB
with `engine="streaming"`. `collect_batches` resolves `auto` to streaming
itself, in py-polars. That is the 1.x line: polars 2.0 (rc.1, 2026-09-02)
resolves `auto` to the streaming engine for every lazy plan
(pola-rs/polars#27822), so there `sink_batches` streams by default.

### Filter after the bank

**Filter after the bank when the semantics allow.** The predicate is then
pushed into the source and applied per chunk (E33): 0.78 GB flat, against
2.53 GB for the same filter before. This is not merely a workaround: the
two mean different things anyway. Before changes what the model *learns
from*; after changes only what comes out.

**When the model *must* skip those rows, a zero weight is the streaming
form.** `when/then/otherwise` is elementwise, so the query keeps
streaming:

```python
import polars as pl
import polars_online as po

cond = pl.col("vol") > 0      # the rows the model may learn from

# A zero weight instead of a filter: the other rows are still scored and still
# move the clock, but the model learns nothing from them, and every step streams.
lf = pl.scan_parquet("ticks.parquet")
lf = lf.with_columns(pl.when(cond).then(pl.col("w")).otherwise(0.0).alias("w2"))

spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t",
                       max_dclock=10.0, weight="w2", halflife=1000.0)
lf.online.fit_predict([spec]).sink_parquet("fit.parquet")
```

The spec's `weight="w2"` is what makes the zero count. At 12M rows this
peaks at 1.3–1.4 GB, against the filter's 2.53. It is 1.08 GB with
`pl.Config.set_streaming_chunk_size(25_000)`, and 0.98 at 10,000; the
plain scan is 0.87. The cost is the documented one:

| | what the model learns from | rows in the output | the clock | peak at 12M rows |
|---|---|---|---|---|
| filter before the bank | the kept rows | the kept rows | the filter leaves a `max_dclock` gap where the rows were | 2.53 GB |
| filter after the bank | every row | the kept rows | | 0.78 GB |
| a zero weight | the kept rows | every row, scored | advances through the skipped rows, so `n_eff` decays and `min_periods` can blank output; no `max_dclock` gap opens | 1.3–1.4 GB |

A branch holding an `.over()` drags the whole expression onto a collecting
node (3.18 GB). `po.run(input=<a filtered plan>)` read the same way and had
the same window. `keep_columns=` did not, because a projection is not a
filter.

### Two levers not taken

**One way gives a filter the plain scan's footprint: running it *inside*
the source.** Read the plain scan, `chunk.filter(cond)`, and feed the bank.
The IO-plugin source is the only serial stage in the graph, and whatever
runs there costs one chunk. Measured as a prototype (2026-09-02), it peaked
at 0.81 GB at 12M rows against 2.54, in the same wall time. The output was
identical to the upstream filter's, except `coef`, which is snapshotted per
chunk by contract.

It is not in the API, deliberately, for four reasons. It would be a second
`filter` whose only difference is memory: the class of surprise this
section exists to remove. The CLI could not say it without an expression
parser. A predicate that is not elementwise (`x > x.mean()`) would silently
mean something different per chunk, where polars refuses to push such a
predicate into a scan at all. And the cost it avoids is the column-reorder
stage above, which is polars' to fix, after which `scan.filter(..)` is
0.58 GB as written.

**The other lever that keeps the syntax is fewer pipelines for the input
query alone.** `polars_config` reads `POLARS_MAX_THREADS` at every query
start, and py-polars' private `config_reload_env_var` re-reads it. That
gives 0.65 GB at 2 pipelines and 0.29 at 1, but 2.7 → 3.9 → 7.1 s, because
the reader's row-group parallelism follows the same number. Not taken
either.

### Polars' own windowed operations do the same thing

**Polars' own windowed operations behave the same way (2026-09-02), so this
is polars' rule, not a quirk of ours.** The rule is *whether the streaming
engine has a node for that call*. The measurement used the same file,
`sink_parquet(engine="streaming")`, `POLARS_ROW_GROUP_PREFETCH_SIZE=1`,
and one output column, so the scan reads only what the expression needs.
Each plan is `pl.scan_parquet(f).select(<the expression below>)`, run in
its own process:

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
heuristic. Every expression lands in one bin. An *elementwise* expression,
`FunctionFlags::ROW_SEPARABLE | LENGTH_PRESERVING`, is computable on any
subset of rows with one row out per row in. The bins:

| expression | how it is lowered | memory |
|---|---|---|
| elementwise | stays inside the `select` / `with-columns` / `filter` node it appears in | streams per morsel |
| `AExpr::Rolling` | a dedicated `RollingGroupBy` node, a real streaming one: it keeps a `buf_df` with a `buf_df_offset`, and drops rows once the window has passed them (`nodes/rolling_group_by.rs`) | streams |
| `ewm_*`, `cum_*`, `shift`, `interpolate`, `rle` and friends | each has its own node under `nodes/` | streams |
| `AExpr::Over` without an `order_by` | `try_build_streaming_group_by`, which rewrites `mean().over("group")` as `multiplexer → group-by → equi-join → zip`: streaming nodes, but the multiplexer buffers the whole input while the group-by side finishes | O(data) all the same: 2.97 GB at 12M rows, 8.39 GB at 36M |
| `AExpr::Over` when that returns `None` (a `rolling` or `ewm_mean` under `.over`) | `fallback_subset` → `build_fallback_node_with_ctx` → **`PhysNodeKind::InMemoryMap`**: collect the input, run the in-memory engine, re-emit | collects |
| a frame-level `rolling`, which takes the dedicated node only while `keys.is_empty()`; and any group-by carrying `rolling`/`dynamic` options | returns `Ok(None)` twice over (`lower_group_by.rs:737`, `:1043`), and ends at `build_group_by_fallback` | the same collect |
| a user expression not flagged elementwise: a plugin | the generic fallback for column UDFs, a `columnar-function` node: one `InMemorySink` per input, `call_udf` once on the whole column, an `InMemorySource` after (`nodes/columnar_function.rs`) | collects |

The engine draws exactly this classification, and it is the fastest way to
know which bin a query landed in:

```python
import polars as pl

# The physical plan the streaming engine will run, as Graphviz source. Its legend:
# ◯ a streaming node; yellow a memory-intensive one (multiplexer, group-by,
# join, sort); red an in-memory fallback (columnar-function, in-memory-map).
lf = pl.scan_parquet("ticks.parquet").select(pl.col("y").mean().over("group"))
print(lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True))
```

Out-of-core spilling exists for the yellow nodes in 1.44.1, but is off by
default. `POLARS_OOC_MEMORY_BUDGET_MB` is `u64::MAX`, and the `_FRACTION`
variable is parsed and never read (`polars-config-0.55.2`). With a 1.5 GB
budget, the `.over` plan peaked at 2.47 GB instead of 2.97: it spills, it
does not cap.

**So the trap is not "user code is O(data)".** It is that an ordered,
stateful operation streams **only where polars has hand-written a node for
it**, and the plugin interface has no way to declare one. There is no "call
me per morsel, in order, and let me keep state" in the contract, which is
why one call with the whole column is what a plugin gets. Two consequences,
stated plainly. Polars' own `ewm_mean` streams as an expression, while the
same EW mean written as our plugin did not, for that reason alone. And
*per-group* windowing in polars is the collecting form (`.over`,
`group_by=`), while a bank's `group=` is O(state): one accumulator per
group, and no partitioning of the data at all.

### How it was measured, and why not RSS

**Memory here is the peak *physical footprint*:**
`proc_pid_rusage(RUSAGE_INFO_V4).ri_phys_footprint`, sampled every 20 ms
from outside the process. It is the same number `/usr/bin/time -l` prints
as "peak memory footprint". RSS is the wrong ruler for this question.
Polars memory-maps a local parquet file, and the file's clean pages count
in RSS, though the kernel drops them under pressure at no cost. The CLI's
*RSS* grew 1.37 → 2.26 GB from 3M to 6M rows, while its footprint stayed at
0.75 GB. The first cut of this measurement used `ru_maxrss`, and said every
surface was O(data). It is not.

## 12. The chunk plan, revisited (2026-09-04)

The question was whether the plan for one chunk could be faster without
giving anything up: what runs in parallel, in which order, over which
memory. The answer is the three items P9–P11 in §3. This section is the
measurement behind them, including the one that overturned the hypothesis
the work started from.

### The plan as it stood

Under the bank's pool, each chunk ran these phases in turn:

| phase | in the plan as it stood |
|---|---|
| `group` | indices per key, specs in parallel |
| `extract` | columns, one thread per spec |
| `check_clock` | |
| `process` | one task per spec × group, instances nested |
| `assemble` | specs in parallel, fields serial within one |

Every phase is a barrier, and chunks are strictly sequential. That is what
chunk invariance means, and it is not on the table.

### What the sections said

The per-group tasks had been made to scale (P1–P3), and nothing around them
had. On 400k rows × k=20 × 64 interleaved groups at 14 threads, `process`
was 15 ms, and `group` + `extract` + `assemble` were 21. The plumbing had
become the majority of the wall, and all three of those phases had a
single-threaded stretch inside them.

### The benchmark artifact

**Most of the gap between interleaved and blocked groups was the
benchmark's clock, not the layout.** The obvious suspect was memory layout. The same 64 groups
arriving as blocks ran the chunk in 27 ms, against 63 interleaved, which
read as "a cache line per gathered value, twice per row". Building the
group-contiguous layout (P9) took the interleaved `process` on one thread
from 400–420 ms (the run-to-run spread on `main`) to 399: nothing.

The gap was somewhere else. The matrix used `clock="t"` with `t` the row
index and `max_dclock=10`. `solve_every` defaults to `halflife / 50` clock
units (`spec.rs`, `solve_every_default`), which is 20 units at
`halflife=1000`. Blocked, a group's consecutive rows are 1 unit apart: a
solve every 20 rows. Interleaved, they are 64 apart, capped to 10: a solve
every 2 rows. Ten times the solves is the whole 300 ms. That is the
documented semantics of a clock-unit solve schedule, not a layout cost, and
any benchmark with an index clock over interleaved groups pays it.

Re-measured with a *matched* clock, each group's rows 1 unit apart in both
orders, the layout penalty on `main` was 14% at one thread and 38% at
fourteen. It is real, and worth P9, but a third of the story rather than
all of it.

### Where it ended up

Milliseconds per 400k-row chunk, k=20, 64 groups on an `Int64` key,
matched clock, best of three. `total` is the bank's whole call, of which
the four sections are consecutive parts, and `wall` is the Python-side
call.

| layout, threads | build | extract | group | process | assemble | total | wall |
|---|---|---|---|---|---|---|---|
| interleaved, 1 | main | 1.6 | 8.6 | 119.1 | 8.9 | 138.1 | 139.9 |
| | now | 7.3 | 1.6 | 102.3 | 3.7 | 114.9 | 116.4 |
| interleaved, 14 | main | 2.2 | 9.3 | 15.3 | 9.0 | 35.9 | 37.3 |
| | now | 1.6 | 1.6 | 11.1 | 1.8 | 16.0 | **17.3** |
| blocked, 1 | main | 1.4 | 6.2 | 105.0 | 8.0 | 120.6 | 122.3 |
| | now | 2.0 | 1.3 | 103.0 | 4.0 | 110.3 | 112.0 |
| blocked, 14 | main | 1.8 | 6.2 | 11.1 | 7.8 | 26.9 | 28.6 |
| | now | 0.8 | 1.3 | 11.5 | 2.1 | 15.8 | **17.1** |

At 14 threads that is 2.2× on the interleaved chunk and 1.7× on the blocked
one; at one thread, 17% and 8%. The gather moved the stride from `process`
into `extract` (1.6 → 7.3 ms at one thread). There it is paid once per
column instead of twice per row, and, from 4096 rows up, in parallel.
Interleaved and blocked now finish within noise of each other, which is
what P9 was for.

The rest of the matrix, at 14 threads, on 400k rows at k=20 with the index
clock of the artifact above, in ms:

| workload | measure | `main` | now | bounded by |
|---|---|---:|---:|---|
| one group | `total` | 119.8 | 109.9 | |
| 64 groups, interleaved | `total` | 62.7 | 44.8 | |
| 64 groups, blocked | `total` | 26.8 | 15.5 | |
| one group × five halflives | `assemble` | 39.6 | 3.5 | `process`, unchanged at ~415: a `halflife=100` instance solves every 2 clock units |
| 64 Zipf-sized groups | `process` | 117 | 110.5 | the biggest group |

The README's workloads, in seconds:

| workload | measure | `main` | now |
|---|---|---:|---:|
| 12M rows over 64 groups with a k=4 grid, 14 threads | `total` | 3.03 | 2.27 |
| | wall | 3.25 | **2.48** |
| | `assemble` | 0.84 | 0.10 |
| | `group` | 0.20 | 0.09 |
| | `extract` | 0.26 | 0.19 |
| the same, one thread | | 14.06 | 13.48 |
| the group-sorted file | `total` | 7.47 | 6.86 |
| the ticks file, one spec | | 0.74 | 0.64 |
| the ticks file, six specs | | 2.32 | 2.23 |

The group-sorted file has few groups per chunk, so §10's `chunk_rows`
advice stands. Every golden number, the chunk-invariance suite and the
oracle tests are unchanged, and the whole pytest suite passes on the
branch.

### `chunk_rows`, swept

The README's *Chunk size* subsection comes from this sweep (2026-09-04):
12M rows over 64 groups, one k=4 spec with two halflives, 14 + 14 threads,
one run per process. Each cell is wall time and peak footprint
(`/usr/bin/time -l`). RSS reads ~0.7 GB higher, because the memory-mapped
input counts there, which is why the README's two-knobs paragraph once said
1.8 GB where these say 1.1.

| `chunk_rows` | interleaved, this branch | sorted by group, this branch | interleaved, `main` | sorted by group, `main` |
|---|---|---|---|---|
| 20k | 2.73 s / 1.00 GB | 9.19 / 0.92 | | |
| 50k | 2.51 / 0.95 | 8.77 / 0.93 | 3.14 / 0.97 | 9.30 / 0.85 |
| 100k | 2.41 / 1.04 | 8.09 / 0.98 | 3.06 / 0.97 | 8.71 / 0.96 |
| 200k | 2.59 / 1.13 | 7.13 / 1.06 | 3.36 / 1.09 | 7.77 / 1.02 |
| 500k | 2.78 / 1.45 | 4.60 / 1.44 | 3.87 / 1.32 | 5.27 / 1.37 |
| 1M | 3.16 / 1.83 | 4.18 / 1.85 | 4.47 / 1.81 | 5.04 / 1.92 |
| 2M | 4.53 / 2.38 | 6.00 / 2.64 | 6.58 / 2.53 | 5.42 / 2.70 |

Interleaved, the bank's `total` is 2.2–2.7 s at every size, so what the
large chunks lose is the read/fit/write overlap. Sorted by group, `process`
falls 8.5 → 2.2 s from 20k to 1M, because a chunk runs only the groups it
holds, and this file has ~187k rows per group. On `main` the shape is the
same, with a slower assembly. Below the default the footprint barely moves:
polars' reader prefetch is most of the first gigabyte (0.46 GB at
`POLARS_MAX_THREADS=1`, 1.14 at 14, same 200k chunks).

The two-knobs matrix, re-measured the same way on this branch. The column
heads are Polars' threads / the bank's, as in the README's table:

| specs | 14/14 | 4/14 | 4/4 | 1/14 |
|---|---|---|---|---|
| one spec | 2.64 s / 1.14 GB | 2.65 / 0.76 | 3.87 / 0.61 | 7.35 / 0.46 |
| six specs | 10.30 / 1.52 | 11.83 / 1.18 | 16.63 / 1.04 | |

With `assemble` parallel, 4/14 no longer beats 14/14 on time; it still
takes a third off the memory. 28 + 28 on the ticks grid: 2.18 s, against
2.21 at 14 + 14.

### The README's numbers, regenerated

On this branch (2026-09-04), `scripts/benchmark.py`, 200k rows, best of 3:

| configuration | §8, rows/s | now, rows/s |
|---|---:|---:|
| k=5 | 8.96M | **11.05M** |
| k=20 | 3.62M | 3.92M |
| k=50 | 961k | 1.01M |
| 10 targets | 1.91M | 2.32M |
| 5 halflives | 2.16M | 2.50M |
| `rls` | 1.93M | 1.96M |
| `kalman` | 1.66M | 1.69M |
| `lasso` | 1.88M | 2.09M |
| `huber` | 3.68M | 4.09M |
| `ftrl` | 6.29M | 7.12M |

A 200k single-group run spends a visible share of its time in `assemble`,
which P10 made ~5× cheaper.

`scripts/scaling_bench.py`, k=20 over 64 groups:

| threads | rows/s |
|---|---:|
| 1 | 1.02M |
| 2 | 1.91M |
| 4 | 3.52M |
| 8 | 6.44M |
| 14 | **8.20M**: **8.0×** (was 6.6×) |

The ticks grid, and one spec of it:

| workload | 1 thread | 14 threads | before, 1 / 14 |
|---|---:|---:|---|
| the ticks grid, six specs over 2.56M rows | 12.3 s | 2.22 s | 13.1 / 2.35 |
| the three-factor spec alone | 2.47 s | 0.62 s | 2.65 / 0.72 |

Eight single-group specs, k=20 over 300k rows, one halflife each: 130 ms in
one bank, against 515 ms one at a time. The old 118 / 685 came from a
configuration nobody wrote down; this one is `halflife=1000·j`,
`max_dclock=10`. The `.over()` figure is untouched: the plugin's groups sit
below the row floor, and time the same on both builds.

### The row floor

**The gate caught what the wall clock did not.**
`tests/test_ffi_memory.py::test_plugin_over_groups` failed once, at
6.6 KB/iter against its 4.0 line. It was not a leak: 3000 iterations of the
same body drift by −0.07 and −0.37 KB/iter overall. But the per-block
wobble around that flat mean had doubled, ±4.6 KB/iter against ±2 on
`main`, and the test's 240-iteration window can now catch the wobble.

The cause is fanning a 30-row group (what the expression plugin hands the
bank under `.over()`) out across the pool. More threads' allocator caches
take part in every tiny call, for no speed at all. A call took 0.85 ms in
both builds at 1500 rows over 50 groups, and 15.0 vs 15.2 ms at 200k rows
over 1000. So the column reads and the field builds fan out only from
`PAR_MIN_ROWS` = 4096 rows up, where one task is a 32 KB copy, about a
rayon dispatch. With the floor the wobble is back at ±2.4, the tiny-group
timings are unchanged, and the 400k-row rows above are the same to the
tenth of a millisecond.

A threshold on fan-out is the usual answer to this (polars' own splits have
one). What is worth writing down is that the *memory* test found it, and
what it found was noise amplitude, not growth. Read the marks, not the
verdict, before touching that test's line.

*Since task 85 (2026-09-17)* the plugin is gone, and
`test_plugin_over_groups` with it. `test_many_tiny_groups` feeds the same
shape, fifty groups of thirty rows, through the bank's own group column,
with each mark averaged over 240 iterations
(`tests/test_ffi_memory.py:126`, `:149`). *Since task 75 (2026-09-08)*, a
wide chunk's column reads also fan out from 65,536 feature values
(`PAR_MIN_CELLS`, `crates/online-polars/src/bank.rs:302`, `:347`–`348`).

### What is left, and why it stays

| limit | why it stays |
|---|---|
| the recursion in a stream | it is the per-row cost, and cannot be split (§5, "parallelizing the recursion itself") |
| the biggest group | it bounds every Zipf-shaped chunk |
| a short halflife | it bounds a grid through its solve cadence |
| a group-sorted file | it gives a chunk few groups to spread, which is a `chunk_rows` decision, not a plan one |

The phases are now all parallel above the floor, the stride is gone, and
the barriers between phases are the ones chunk invariance requires. The
next factor would have to come from inside `process`, and §5 says why it
will not.

## 13. The new families, and where a wide row goes (2026-09-05)

Tasks 23–30 added two clustering models (`kmeans`, `micro`), and the
moments family's Mahalanobis distance and PCA. They also added
class-conditional moments (`ew_class`), two first-order learners with
constraints (`sgd`, `pa`), a transition on `kalman`, conformal intervals
and a sequential test (`seqtest`). Task 31 is the survey §8 gave the regressions: what each costs
on one thread, how each scales across the pool, and what could be had
without moving a number.

### The contract: bit for bit

**The contract is §12's: every golden bit is unchanged.** Each change below
either recomputes the same arithmetic on the same inputs, or touches no
arithmetic at all. Two dumps prove it against a build of the Task 30 commit
in a worktree:

| dump | size | specs |
|---|---|---|
| the first | 31 columns × 65k rows | fifteen: groups, weights, a null key, `predict`, and every new family |
| the second | 22 columns × 440k rows | ten, long enough that every chunk end and every run boundary of the split below falls inside the data |

They are compared bit for bit: float columns as `u64` bits plus the
validity, and lists and structs recursed. Both print `BIT-EXACT`.

### The survey

400k rows, k = 20, one group, one chunk, one process, best of 3, the Task
30 build against this one:

| workload | before, ms | after, ms | speedup |
|---|---|---|---|
| `ewridge` | 113.3 | 110.0 | |
| `ewridge` + conformal | 111.2 | 109.8 | |
| `ewridge` + `emit_sigma` + `emit_resid_z` | 111.3 | 107.8 | |
| `kalman` | 238.4 | 225.1 | |
| `kalman` + `revert_halflife` | 266.9 | 254.8 | |
| `sgd` | 41.7 | 39.9 | |
| `sgd`, box | 46.0 | 45.2 | |
| `sgd`, simplex | 179.1 | 157.1 | 1.14× |
| `pa` | 32.7 | 30.6 | |
| `pa`, simplex | 168.8 | 148.9 | 1.13× |
| `ew_cov` mean, std, corr (230 statistics) | 757.7 | 222.1 | **3.4×** |
| `ew_cov` mean only | 107.5 | 66.8 | 1.6× |
| `ew_cov` mean + mahal | 562.2 | 538.8 | |
| `ew_cov` mean + mahal + `mahal_q0.99` | 566.0 | 547.5 | |
| `ew_cov` pca 3, `pca_every=1` | 6399.6 | 6592.5 | |
| `ew_cov` pca 3, `pca_every=100` | 332.3 | 162.0 | **2.05×** |
| `ew_cov` k = 4, mahal | 115.5 | 108.8 | |
| `ew_class` full, 3 classes | 1509.7 | 808.3 | **1.87×** |
| `ew_class` shared | 697.3 | 707.3 | |
| `ew_class` diagonal | 160.4 | 155.3 | |
| `ew_class` k = 4, full | 315.2 | 199.4 | 1.58× |
| `kmeans` k = 4, K = 3 | 41.7 | 34.6 | 1.2× |
| `kmeans` k = 4, K = 8 | 68.3 | 58.9 | 1.16× |
| `kmeans` k = 20, K = 8 | 127.6 | 122.6 | |
| `micro` k = 4, eps 0.5 | 37.0 | 24.5 | **1.5×** |
| `micro` k = 20, eps 1.0 | 58.6 | 44.4 | 1.32× |
| `seqtest`, one column | — | 16.9 | |
| two `ewridge` sides | 246.9 | 257.4 | |
| two sides + `seqtest` compare | 271.0 | 265.6 | |

The rows without a ratio moved by less than the run-to-run spread (3–5% on
this machine at 400k rows). `pca_every=1` is the O(k³)-per-row cadence a
user asks for by name. The row is there to show what it costs against
every hundredth row.

### Where `ew_cov`'s row went

**The model was not the cost.** 758 ms for 400k rows is 1.9 µs a row, for
230 numbers that are each a load, a multiply and a store. Four things were,
in the order they were found.

1. *The pairwise square roots.* `corr(i, j) = cov / (std_i · std_j)` took
   both roots inside the pair loop: 380 `sqrt` per row at k = 20, 80% of
   the model's own time. Now twenty roots go into a `Vec`, then the same
   product of the same two roots per pair. That is bit-identical, since it
   *is* the same two operands (`ewcov.rs`, `EwCovStat::Corr`).
2. *Residual plumbing for a model with no residual.* `Stream::process_one`
   tracked `r`, `sigma`, `z` per slot for every model. That was a division
   by the residual variance per slot per row, and three per-row buffer
   fills, for the 230 slots of a model whose outputs are not predictions.
   An `Instance` now knows whether its model `predicts_no_target`, and the
   stream skips the tracking, the fills and the `resid` buffer itself
   (`Buffers::of(spec)` sizes it at zero). While there, the two integer
   divisions `slot / nc` per slot became a `chunks(nc)` pass. *(The name
   `process_one` is P1's: P2 had already moved that per-row code into
   `run_instance`, `crates/online-polars/src/stream.rs:3287`.)*
3. *The stores.* This one is the section's finding, and it is about layout,
   not arithmetic. `ChunkOut` is slot-major,
   `(mi·n_slots + slot)·n_rows + ri`, so one row is 231 stores, each
   `n_rows × 8` bytes apart. Over a 400k-row chunk those are 231 lines
   3.2 MB apart, a page per store. A third of the row went on stores to
   lines evicted before the next row came back to complete them. And when
   `n_rows` is a power of two, the stride is a multiple of the L1's way
   size, and all 231 stores of a row map to a handful of the 8-way sets.
   Measured, `process` alone, in ns per row of the default `ew_cov`:

   | chunk size | with the first two fixes in | after the run split | the final build |
   |---|---:|---:|---:|
   | 65 536 rows | **1302** | 437 | 423 |
   | 65 552 rows | **685** | 450 | 440 |
   | 4096 rows | **1256** | 438 | 424 |
   | 4112 rows | **435** | 446 | 421 |
   | the whole 400k in one chunk | 690 | 491 | 473 |

   A chunk sixteen rows longer ran twice as fast, and 65 536 is the natural
   streaming batch. The fix is to stop letting the caller's chunk size
   choose the stride. Each (spec, group) work item now feeds its rows
   through the stream in sequential *runs* of `ChunkOut::run_rows` rows,
   each run its own `ChunkOut`. A run is sized so its buffers fit in 2 MiB
   and the stride is an odd number of 128-byte lines:

   | model | values a row writes | rows per run |
   |---|---:|---:|
   | the default `ew_cov` | 231 | 1104 |
   | a one-target `ewridge` | three | 87 376 |
   | a model of width two | two | 131 056 |
   | anything wider than the budget | | a floor of 80 |

   The stream is the same object across runs, so the recursion does not
   see the boundary. The coefficient report follows the `last` run's final
   row, so `coef` still lands on the chunk's last row and on `coef_every`.
   After the split the time per row is flat across the five chunk sizes,
   and the power-of-two cliff is gone. Chunk invariance is what makes the
   split invisible. Three tests in `crates/online-polars/tests/bank.rs`
   pin it: 1 vs 5 vs 5000 chunks equal across runs for the wide model, the
   `coef` rows across run boundaries, and the odd-lines-within-budget
   arithmetic itself.
4. *The assembly.* `F64Column` now keeps its validity as a packed bitmap
   built eight flags at a time
   (`wrapping_mul(0x0102_0408_1020_4080) >> 56`), handed to arrow as a
   `Bitmap` instead of a `Vec<bool>` it would pack again. And a work item
   whose rows are one unbroken range (a single group, or a group that
   arrives in blocks) is copied with `copy_from_slice` rather than
   scattered a row at a time. `assemble` for the 400k default chunk went
   27 → 20 ms. The read-out on the final build is `process` 189 ms,
   `assemble` 21, `total` 214. *(`copy_from_slice` is not the code's name
   for that copy, and never was: it is `F64Column::run`, called at
   `crates/online-polars/src/bank.rs:4334`–`4335`.)*

### Reading the wall clock, not the profile

**The wall clock says how much, and a profile says only where inside
`process`.** The first `sample` of the default `ew_cov` put half the time
in `assemble`, and the list above nearly started at the wrong end. `sample`
counts threads: `assemble` runs on all fourteen, and `process` on one for a
single group, so a 14× weighting made a 20 ms phase look like 280.
`ONLINE_TIMING=1` prints `extract / group / process / assemble / total` per
chunk in wall milliseconds, and is the number to trust. The profile is for
*where in `process`*, at function granularity. Per-line attribution inside
an inlined recursion is not reliable: the `sqrt` above showed up on the
loop header. The recipe, for next time:

```sh
# 1. From the repository root, build the extension with symbols. The release
#    profile strips them (Cargo.toml), so this turns that off for one build.
CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_DEBUG=1 uv run maturin develop --release -m crates/online-py/Cargo.toml

# 2. Run the workload in a loop, in its own Python process, and sample that
#    process by its pid. Read "Sort by top of stack" in out.txt.
sample <pid> 8 -mayDie -file out.txt

# 3. When the function name is not enough, read the inner loop in the binary.
nm python/polars_online/_polars_online.abi3.so
objdump -d python/polars_online/_polars_online.abi3.so
```

### `ew_class`: one factorization per learned row

**The full-covariance classifier factorized every class on every row,
though a row updates only one.** It scored every row against every class
with a fresh Cholesky factor of each class's ridged comoment matrix. For
three classes that is three O(k³) factorizations per row, when a row
updates one class and leaves the other two matrices untouched. `solve::SpdFactor`
splits `quad_forms_logdet` into the factor (`of`, the same jitter ladder as
before) and its uses (`quad_forms`, `log_det`). `ewclass::Factors` keeps
one per class, `#[serde(skip)]`, invalidated for the class a row learns. A
`step` factorizes one class; `predict` (`&self`) factorizes what it needs
without keeping it. `solve.rs` has a test that the kept factor gives the
one-shot numbers to the bit.

The result is 1510 → 808 ms at k = 20, and 315 → 199 at k = 4. The
shared-covariance form learns *the* one matrix every row and gains nothing
(697 → 707, noise), which confirms the mechanism: it was already at one
factorization per row.

### `kalman`, `sgd`, `pa`: allocations and a closure

`kalman` built its per-slot standardizer scales and its process-noise
vector as fresh `Vec`s every row. They are `#[serde(skip)]` scratch now
(`sbuf`, `qbuf`), and a shared `revert_halflife` fills `phi` once, instead
of deriving a per-slot halflife a slot at a time: 238 → 225 ms.

**The constraint projection (`constraint.rs`) is the one place a first
attempt made things worse.** With `scales` (the standardizer's units), the
closure `bound(i)` recomputed `1 / scale_i`, `lo_i · scale_i` and
`hi_i · scale_i` at every one of the O(k log k) reads of the breakpoint
sweep. A `Scratch` that fills them once per projection is the fix. The
survey caught it making the *unscaled* simplex 9% slower (179 → 195 ms):
with `scales = None` the old closure folded to constants, and the new one
read a prepared `a[i]` per coordinate. So `project` is two instantiations
of one `project_with`: the unscaled closure `|i| (1.0, lo[i], hi[i])`,
whose multiplications by one fold away, and the scaled one over `Scratch`.
Multiplication and division by 1.0 are exact, so nothing moves: 179 → 157
and 169 → 149 ms. The rest of the projection is a sort of 2k breakpoints
per row (≈ 300 ns at k = 20), and stays. An O(k) selection would replace
it, but 157 ms for 400k constrained rows was not the problem this section
was solving.

### Thread scaling

`POLARS_ONLINE_MAX_THREADS` at 1, 2, 4, 8, 14 over 800k rows in 64
interleaved groups (`t = row // 64`, so a group's rows are one clock unit
apart; `max_dclock = 10`), best of 3. Wall in ms, and the 14-thread
speedup, as Task 30 build / this one:

| workload | 1 | 2 | 4 | 8 | 14 | speedup |
|---|---|---|---|---|---|---|
| `ewridge` | 230 / 238 | 128 / 130 | 74 / 76 | 46 / 46 | 40 / 42 | 5.8× / 5.6× |
| `ewridge` + conformal | 241 / 254 | 134 / 141 | 75 / 78 | 47 / 48 | 43 / 44 | 5.6× / 5.8× |
| `kalman` + revert | 552 / 532 | 287 / 276 | 158 / 153 | 89 / 87 | 75 / 74 | 7.3× / 7.2× |
| `sgd`, simplex | 346 / 340 | 181 / 181 | 105 / 102 | 61 / 58 | 54 / 51 | 6.4× / 6.6× |
| `ew_cov` default | 1641 / **753** | 890 / 389 | 514 / 246 | 320 / 150 | 276 / **152** | 5.9× / 5.0× |
| `ew_cov` mahal | 1170 / 1137 | 610 / 584 | 322 / 310 | 172 / 168 | 140 / 131 | 8.4× / 8.7× |
| `ew_cov` pca 3 / 100 | 769 / **460** | 407 / 258 | 229 / 150 | 142 / 94 | 121 / **85** | 6.4× / 5.4× |
| `ew_class` full | 3090 / **1655** | 1601 / 852 | 823 / 440 | 434 / 236 | 324 / **191** | 9.5× / 8.7× |
| `ew_class` k = 4, full | 662 / 419 | 374 / 220 | 189 / 124 | 114 / 76 | 85 / 65 | 7.7× / 6.4× |
| `kmeans` k = 4, K = 8 | 219 / 211 | 117 / 114 | 66 / 64 | 39 / 38 | 31 / 31 | 7.1× / 6.9× |
| `kmeans` k = 20, K = 8 | 463 / 461 | 242 / 244 | 131 / 132 | 74 / 72 | 63 / 62 | 7.4× / 7.4× |
| `micro` k = 4 | 92 / 73 | 51 / 41 | 33 / 27 | 22 / 19 | 20 / 17 | 4.6× / 4.2× |
| `micro` k = 20 | 133 / 117 | 75 / 70 | 47 / 41 | 30 / 29 | 31 / 31 | 4.3× / 3.8× |
| `seqtest`, one column | 57 / 44 | 33 / 28 | 21 / 18 | 15 / 13 | 14 / 13 | 4.2× / 3.4× |
| two `ewridge` sides | 731 / 747 | 387 / 388 | 209 / 215 | 119 / 121 | 96 / 98 | 7.6× / 7.6× |
| two sides + compare | 793 / 791 | 423 / 430 | 230 / 234 | 140 / 139 | 115 / 117 | 6.9× / 6.7× |

**Every gain on one thread carries to fourteen:** `ew_cov` 276 → 152 ms,
`ew_class` 324 → 191, the PCA cadence 121 → 85.

**The *speedup ratios* of the models that got faster went down, and that is
Amdahl, not a regression** (5.9× → 5.0×, 9.5× → 8.7×). The part that
scales, `process`, shrank, and the phases around it (`extract`, `group`,
the Python call) did not, so they are a larger share of a smaller wall.
The models that scale worst are the cheapest per row (`micro`, `seqtest`,
3–4×). At 13 ms for 800k rows, the per-chunk fixed cost is most of the
chunk, as §12 found for the plumbing around a k = 20 `ewridge`. The models
with the most arithmetic per row scale best (`ew_class`, `ew_cov` mahal,
8.7×). The ceiling in §5 stands: the recursion is the per-row cost and
cannot be split within a group, and 64 groups over 14 threads leaves the
biggest group's tail.

### What is left, and why

**`kalman`'s standardizer: done in task 33 (2026-09-05).** It was a full
`EwCov` over the features, of which the filter reads the diagonal: O(k²)
of comoment updates per row for k variances, and `sgd`'s `scale_features`
the same. It was deferred here as a serialized layout change, and done
once the user chose to take the schema bump this early. `EwDiag`
(`ewdiag.rs`) is `EwCov`'s diagonal, operation for operation;
`SCHEMA_VERSION` went 2 → 3, and schema-2 states convert on load by taking
the diagonal. It is bit-exact: the Task 31 dump recipe against a build of
the previous commit, plus a unit test that compares the two accumulators as
`u64` bits. Per 400k rows on one thread, the two builds run back to back:

| workload | k = 20, ms | k = 50, ms |
|---|---|---|
| `kalman` | 223 → 179 | 1065 → 908 |
| `kalman` with reversion | 250 → 211 | |
| `sgd` with `scale_features` | 109 → 62 | 279 → 121 |
| `sgd`, the scaled simplex | 266 → 227 | 657 → 504 |
| `sgd` without scaling | unchanged | |

The standardizer is updated on every row whether or not `standardize` is
on (it carries `n_eff`), so `standardize=False` gains the same.

**`pca_every=1`** is an O(k³) eigendecomposition per row, by request. The
cadence knob is the answer (40× at every hundredth row), and the docstring
says so.

**The simplex sort** (above): an O(k) selection is possible, and not done.

**A row-major `ChunkOut`** would make a row one contiguous store, and was
the obvious alternative to the run split. The split gets the same cache
behaviour without changing the reader in `assemble`, the `at(...)` layout
contract or the `scatter` fast path, so the layout stays.

**The predict-only path** (`predict_chunk`, `score`) is not split into
runs. The coefficient-on-the-last-row rule is what the split has to
preserve, `predict` reports none, and it is not the hot path.

**`F64Column::run`** packs its validity in a second pass over the flags
(eight per byte). A fused single pass measured the same within noise, and
`assemble` is 21 ms of 214 on the widest workload, so the simpler code
stays.

**Compare mode's second pass** (`seqtest` over two sides) costs 8 ms per
400k rows on top of the sides, 24 before. It is a second walk over two
residual columns, which the two-phase bank that compare mode runs on
requires.

**A zero-filled `pred`** (fresh pages from the allocator instead of a NaN
fill) was rejected. NaN is the "never written" sentinel the assembly turns
into null, and zero is a value.

**The data summary (task 35)** is one Welford step per input column per
row, always on, and costs about a nanosecond per column-row. From two
builds run back to back on the same machine, 400k rows, one group, best of
five:

| model | k = 5, ns/row | k = 20, ns/row |
|---|---|---|
| `sgd` | 53 → 59 | 95 → 112 |
| `ewridge` | 97 → 99 | 265 → 276 |
| `kalman` | 101 → 107 | 449 → 451 |

Chunk invariance rules out the vectorisable alternative: a two-pass moment
per chunk, merged with Chan's formula, gives different bits under different
chunkings. So the per-row form stays; an opt-out is the answer if a wide,
cheap model ever needs one.

## 14. The Gram update, measured (E48, 2026-09-05)

`EwCov::update` is the hottest loop in the library: every Gram model runs
it once a row, `O(k²)`. E48 proposed halving its flops by computing the
upper triangle and mirroring it, "bit-identical (the products commute in
IEEE)" with "the goldens, unchanged". **Both halves of that turned out to
be wrong**, and a different change to the same loop turned out to be worth
14% to 65%.

### The variants

Five variants, each run over the same rows at six widths, twice, on this
machine. The table gives ratios only, and every variant was checked
bit-for-bit against the current code before being timed.

| variant | what it changes |
|---|---|
| `hoisted` | the deviations computed once into a scratch |
| `zipped` | the inner loop over slice iterators instead of indices |
| `h+zip` | both |
| `upper` | E48's proposal: the upper triangle, mirrored |
| `both_sym` | the symmetric association `(a·b)·(dᵢdⱼ)` |

| k | hoisted | both_sym | zipped | **h+zip** | upper (E48) |
|---|---|---|---|---|---|
| 4 | −9% | −1% | −28% | **−14%** | −21% |
| 16 | −22% | −8% | −55% | **−63%** | −16% |
| 64 | −26% | −16% | −23% | **−45%** | **+64%** |
| 200 | −19% | −20% | −1% | **−24%** | **+54%** |
| 400 | −22% | −20% | +3% | **−27%** | **+53%** |
| 800 | −21% | −21% | +1% | **−22%** | **+107%** |

### E48's mirror is not bit-identical

**The products do commute, but the association does not.** The loop
computes `a*b*dᵢ*dⱼ`, which is `((a·b)·dᵢ)·dⱼ`, and the transposed entry
computes `((a·b)·dⱼ)·dᵢ`. The intermediate rounds differently, so today's
co-moment matrix is *not* exactly symmetric: the two triangles differ in
the last bit or two, and mirroring one onto the other changes every golden
value. `the_two_triangles_are_equal_but_not_bit_equal`
(`crates/online-core/src/ewcov.rs:3294`) pins this, so that the next
person to reach for the shortcut finds the reason it was not taken. Writing
it as `(a·b)·(dᵢ·dⱼ)` would make the matrix exactly symmetric. But that is
itself a different rounding from today's (`both_sym` above, "NO" on
bit-equality), so the goldens move either way.

### And it is slower, by a lot

The mirror store `c[j*k + i]` walks a new cache line for every `j`, so the
loop touches the whole matrix twice instead of once, defeats the
prefetcher, and cannot vectorise. The flops halve and the traffic does
not: +49% to +107% at every width where a triangle would be worth having.
E48's "0.20 ns per element per row, both triangles" was measured; the
conclusion drawn from it was not.

### What is worth having

**`h+zip` is bit-identical, and shipped (task 41):**

- the deviations `x − m` are computed **once** into a row scratch, where
  the old loop recomputed `x[j] − m[j]` inside every row of the matrix:
  `k²` subtractions where `k` will do;
- the inner loop runs over `c`'s row slice zipped with the scratch, so `c`,
  `x` and `m` are not indexed by `j`, and the bounds checks that were
  keeping the loop scalar are gone.

Same operations in the same order, so every golden value is unchanged and
no state file moves. The scratch is `#[serde(skip)]` and its `PartialEq` is
`true`, so it is not part of the state, and a round-tripped accumulator
still compares equal to the one that wrote it. It refills itself on the
first row after a load.

**At these widths the loop is bound by how it touches memory, not by how
many multiplies it does.** That is the lesson §12 and §13 keep teaching. A
change that halves the arithmetic and doubles the traffic is a
pessimisation, and the only way to know which one a change is, is to run
it.

## 15. The correlation families: `bocpd`'s `prune_below` keeps it finite, and `rcov`'s estimator sets its cost (2026-09-06)

Tasks 45–56 added five models. `scripts/benchmark.py` gained a row for
each, so the README's throughput table now covers them, and a regression
in one of them shows up where every other model's would. What follows is
what the measurements say beyond the table: Apple M-series, single process,
best of 2 or 3. Ratios are the part to read.

### `bocpd`: `prune_below` is what makes it finite

**The run vector grows by one entry every row, so without a bound on it the
cost of a stream is quadratic.** The cost of a row is `O(runs · d²)`. That
is not a subtlety; it is the whole performance profile:

| `prune_below` | `max_run` | rows | rows/s |
|---|---|---|---|
| 0 | ∞ | 5,000 | 1,897 |
| 0 | ∞ | 10,000 | 947 |
| 0 | ∞ | 20,000 | 472 |
| 1e-8 | ∞ | 20,000 | 203,699 |
| **1e-6** (default) | ∞ | 20,000 | 323,533 |
| 1e-4 | ∞ | 20,000 | 687,188 |
| 1e-6 | 200 | 20,000 | 338,783 |
| 1e-6 | 20 | 20,000 | 396,467 |

The first three rows halve as the stream doubles, which is the `O(rows²)`
written out. Turning truncation on is a 400–1,400× change at 20k rows, and
unbounded beyond it. The knob is then a direct dial on throughput: each
factor of 100 in `prune_below` is roughly a factor of 2 in rows/s, because
it is choosing how many runs stay alive. `max_run` barely moves anything at
the default `prune_below`: by the time the vector is 200 long, truncation
has already dropped everything below `1e-6`. So it is a backstop to
truncation, there for the case where the data keeps a long tail of runs
genuinely alive.

`tests/test_bocpd.py` measures what the knob *costs* in answers rather than
speed: `prune_below = 1e-4` moves `p_change` by less than `1e-3` against
`prune_below = 0` over 400 rows, and leaves `run_mode` identical.

**And `bocpd` is faster on data that breaks.** A changepoint collapses the
posterior onto a short run, so the vector shortens: 1.0M rows/s on the
benchmark's blob features, against 324k on i.i.d. Gaussian rows, with the
same parameters. A model that costs more when nothing is happening is an
odd shape, and it is the right way round: the interesting streams are
cheap.

### `rcov`: the cost is at the close, and the kernel chooses it

**Per row, `rcov` only accumulates.** Everything expensive happens when the
group closes, and which estimator is asked for decides how expensive.
100k rows, four features, blocks of the stated size, throughput over the
whole stream:

| rows per block | `plain` | `kernel` | `preavg` |
|---|---|---|---|
| 200 | 24.8M | 8.8M | 20.0M |
| 1,000 | 36.2M | 4.8M | 23.6M |
| 5,000 | 43.0M | 1.8M | 16.8M |
| 20,000 | 30.1M | 0.45M | 5.8M |

`plain` is free and flat: the close is an `O(k²)` read of an accumulator.
The other two are paid per block, and grow with it. Per close, `kernel`
costs 0.21 ms, 2.7 ms and 44 ms at 1,000, 5,000 and 20,000 rows: about
`n^1.6`, which is exactly `O(n·H)` with the BNHLS automatic bandwidth
`H ∝ n^{3/5}`. `preavg` is `O(n·k_n)` with `k_n = ⌊θ√n⌋` and a much
smaller constant, so it stays within a factor of 5 of `plain` until the
blocks are very large.

The practical reading: a session-length block of a few thousand rows
costs single-digit milliseconds a close under the kernel, which is nothing
beside the session. A block of tens of thousands is where the kernel
starts to be a choice rather than a default, and `preavg` is the estimator
to reach for.

### The other three

**`deco`** is `O(m)` a row by construction: it estimates one number, not a
matrix, and runs at `ew_cov`'s speed with 20 columns. Blocking it costs
about 40%, since it computes one `u` per block and per pair of blocks.

**`hmm`** factorizes a `k × k` covariance per state per row, so it is
`ew_class`'s cost with the classes hidden: 1.34M rows/s at four features
and two states, 353k at twenty features. `covariance="diagonal"` is the way
out at width, as it is for `ew_class`. (Both take `"full"`, `"shared"` or
`"diagonal"`: `crates/online-core/src/ewclass.rs:74`–`83`.)

**`corrchange`'s window kind is the slowest model in the library, and by
design.** The permutation null re-draws `n_perm` statistics every
`permute_every` rows, an `O(n_perm · window · k²)` job amortized over that
many rows. It runs at 187k rows/s at the default cadence, and a `crit`
given as a number skips the whole thing. The monitor kind pays only at a
span's close, where it walks the span once: 389k rows/s at
`span_rows = 500`.

### And they go wide

**The unit of parallel work is a stream, and the new models are streams
like any other.** 200k rows, four features, one group against 64, on a
14-core machine (10 performance + 4 efficiency):

| model | 1 group | 64 groups | speedup |
|---|---|---|---|
| `deco` | 3.22M | 20.1M | 6.3× |
| `hmm` | 1.32M | 10.8M | 8.2× |
| `bocpd` | 0.42M | 2.93M | 7.0× |
| `corrchange`, window | 0.19M | 1.97M | 10.5× |

Nothing here serializes. The slowest model in the library gains the most,
because its per-stream work is the largest thing the pool has to schedule.
`corrchange`'s permutation draws from a per-stream `SplitMix64` seeded from
the spec's `seed` alone, which is the library's convention (`kmeans` seeds
the same way). So every group runs the same permutation *pattern* over its
own rows. That is what makes a stream's output depend on its own rows and
nothing else, the property the chunk-invariance and group-independence
sweeps pin. What it gives up is only that two groups' critical values are
drawn with the same shuffles, rather than independent ones.

`ew_cov` with `lags = 1..5` runs at 1.15M rows/s, against 2.12M for the
same spec without them: five more `k × k` outer products a row, and the
ring of rows they need. The lag block is the whole cost of the Epps
inversion in [docs/REGIMES.md](REGIMES.md) §6, and it is a fifth of a
`mahal`.

## 16. What a `window` costs (2026-09-07)

Task 63 gave five models a hard cutoff ([docs/PLAN.md](PLAN.md) §13). Two
questions follow, and both are measured here rather than argued.

### The windowless path is unchanged

**The concern with a feature like this is that everyone pays for it. They
do not.** When `window` is unset, the added work is a single `Option` check
per call site per row (`self.win.as_ref()?`). The `None` branch *borrows*
the live accumulators rather than cloning them, and no accumulator
arithmetic moves. Best of three runs of
`cargo run --release -p online-core --example core_bench`, on the same
machine, against a build from before the feature (`7cbdaf7`, in a separate
worktree and target directory):

| case | before | after | ratio |
|---|---:|---:|---:|
| `ewridge` k=5 m=1 | 21,515,055 rows/s | 21,161,072 | 0.984 |
| `ewridge` k=20 m=1 | 9,768,511 | 9,427,411 | 0.965 |
| `ewridge` k=50 m=1 | 2,932,569 | 2,875,719 | 0.981 |
| `ewridge` k=20 m=10 | 5,488,848 | 5,428,447 | 0.989 |
| `ewridge` solve every row | 484,921 | 504,666 | **1.041** |
| `ewridge` solve every 25 | 5,478,246 | 5,506,259 | 1.005 |

The signs go both ways, from −3.5% to +4.1%, which is this benchmark's
noise on an unquiesced machine rather than a cost. Nothing here changes
complexity.

### What the window itself costs

200,000 rows, 8 features, one spec, `window = 500` against no window, best
of three through `ModelBank.fit_predict`:

| model | no window | `window=500` | ratio |
|---|---:|---:|---:|
| `ew_cov` (mean + corr) | 27.6 ms | 59.8 ms | 2.2x |
| `ewridge` | 49.9 ms | 123.3 ms | 2.5x |
| `lasso` (1 path point) | 55.4 ms | 137.5 ms | 2.5x |
| `marginal` | 9.5 ms | 26.2 ms | 2.8x |
| `ew_class` (`covariance="full"`) | 177.9 ms | 308.3 ms | 1.7x |

The 2.2–2.8x is the shape of the mechanism: one snapshot pushed per
learned row, and an `O(k²)` subtraction and re-centring at every read. What
that time gets is a hard cutoff, a guarantee an exponential weight cannot
give at any halflife.

**`ew_class` is the one with a structural penalty, and its 1.7x
understates it.** Pure decay leaves a covariance unchanged: means and
centred co-moments do not move under `lam`. That is exactly why the `full`
shape caches its Cholesky factor between rows, and only invalidates the
class a row updated. A *truncated* covariance moves every row, because the
decay carried to the boundary does, so every class's factor is stale every
row: the shape pays one `O(k³)` factorization per class per row. It reads
as only 1.7x because factorization already dominated its baseline (§13
measured the full-covariance `ew_class` at 1510 ms per 400k rows before
that work). `covariance="diagonal"` and `"shared"` do not factorize per
class, and are unaffected.

**Memory is the other axis, and it is the one the feature genuinely
changes:** `O(rows in the window x state)`, where every other model here
is `O(state)`. The settings that bound it:

| setting | effect on the window's memory |
|---|---|
| `window_every = m` | divides it by `m`, and shortens the effective window by at most one snapshot's spacing; it never lengthens it |
| `window_budget`, as `{"refuse": mib}` | a bound per ring, in MiB: the chunk that crosses it is refused |
| `window_budget`, as `{"thin": mib}` | a bound per ring, in MiB: every other snapshot is dropped and the spacing doubled, as often as it takes, which shortens the window the same way |
| no `window_budget` | a window refuses past 256 MiB |

The refusal dates from finding P4 of the 2026-09-12 review (not §3's P4):
at `k = 1000` over a 3,600-row window, the ring was about 29 GB per
instance, and nothing checked.

## 17. What `marginal`'s two views cost, and two costs that were not theirs (2026-09-07)

Tasks 65 and 66 gave `marginal` lagged pair moments and binned target
moments ([docs/MARGINAL-LAGS-AND-BINS.md](MARGINAL-LAGS-AND-BINS.md)). The
rule from §16 applies again: a stream that asks for neither must not pay
for the fact that they exist.

### The plain path, and two regressions no test could see

**It did pay, twice, and both were found by measuring rather than
reading.** `cargo run --release -p online-core --example marg_bench` runs
one million rows, eight features, one target, and no lags, bins or window.
It ran against a build from before either feature (`7c5327f`, separate
worktree and target directory), interleaved, best of the runs each prints:

| plain `marginal` | rows/s | vs before |
|---|---:|---:|
| before tasks 65 and 66 (`7c5327f`) | 69.2M | — |
| with both regressions | 58.0M | 0.84 |
| with the call hoisted out of `learn` | 64.1M | 0.93 |
| with the scan moved inside its guard | 72.2M | **1.04** |

Every row of that table is the same binary, the same harness and the same
sitting, with only the two lines under test moved. The fixes were measured
by putting each regression back, not by comparing against an older note.

**A call inside the hot loop, even one never taken.** The lag update was a
branch inside `learn`'s per-target loop. A call there makes the compiler
assume the callee could reallocate `mx`, `sxx` and `sxy`. So it reloads
their base pointers on every iteration and stops vectorizing, for a stream
with no lags at all. Moving the whole thing to its own pass over the
targets (`Marginal::learn_lags`, run before `learn`, recomputing `a` and
`b` from the same values) gets the loop back. The moments stay
bit-identical to `ew_cov(lags=)`, which is asserted by a test rather than
assumed.

**An `O(p)` scan outside its guard.** The lag ring only accepts rows whose
features are all finite, and the check `x.iter().all(|v| v.is_finite())`
sat *outside* `if let Some(lag)`. So every row of every `marginal` walked
all `p` features to decide whether to push into a ring that did not exist.

Neither is visible to a test, since both compute exactly the right answer.
Neither would have been found by reading the diff, since both look like
ordinary guard clauses. Only a measurement against a build from before the
feature finds this class of thing, which is the argument for keeping one
cheap enough to run.

The finished path lands slightly *above* where it started. That is this
benchmark's noise on an unquiesced machine rather than an improvement (§16
measured −3.5% to +4.1% run to run), so the claim is parity, not a
speed-up. The bank-level number below agrees with the one §16 recorded for
the same shape, which is the independent check.

### What the views themselves cost

200,000 rows, eight features, one target, through `ModelBank.fit_predict`,
best of three:

| spec | wall | vs plain |
|---|---:|---:|
| `marginal` | 9.9 ms | — |
| `bins=16`, edges given | 15.5 ms | 1.6x |
| `bins=16`, edges learned | 16.0 ms | 1.6x |
| `bins=64`, edges given | 19.5 ms | 2.0x |
| `lags=[1,2,3,5,8]` | 27.2 ms | 2.7x |
| `lags` + `serial_rule="geometric"` | 27.2 ms | 2.7x |

The 9.9 ms plain agrees with the 9.5 ms §16 recorded for `marginal` without
a window, which is the independent check that the path is where it was.

**Four times the bins costs 26% more, not four times more.** The per-row
work is a binary search over the edges and three adds, and the decay is
`O(1)` for the whole histogram however many bins it has (`margbins.rs`).
Had decay been the obvious loop over bins, 16 to 64 would have quadrupled
the added cost: 5.6 ms to 22 ms, rather than to 9.6 ms.

**Lags cost more than bins,** because five lags are five more pair updates
per row, each the same width as the contemporaneous one. At 5 lags, the
2.7x is the `1 + L` shape, slightly better than linear because the means
and weights are computed once. `serial_rule` is free: it is read-time
arithmetic over the lags already kept.

## 18. The blocked Gram update (E51, 2026-09-08)

Task 71 gave `ewridge` a `gram_block_rows`: hold `B` rows back, and bring
the `k×k` co-moment matrix up to date once per block with one matrix
product, instead of the rank-1 update §14 tuned. The rank-1 loop touches
the whole matrix every row, and is bound by memory. The product touches it
once per block, and is bound by arithmetic. [docs/PLAN.md](PLAN.md) task
71 has the merge, the two findings its review made, and why the product is
single-threaded. This section is the measurement.

### The measurement

`crates/online-core/examples/gram_block_bench.rs` times the whole
`EwRidge::step` (prediction, cross-moments and the Gram) on one core. It
has one target, an intercept and `halflife=5000`, on this machine, in rows
per second. `faer`'s product is called with `Par::Seq`: the bank
parallelises across groups, never inside a step.

| k | solve | per row | block 64 | block 256 |
|---|---|---:|---:|---:|
| 256 | never | 91,214 | 372,772 (4.1×) | 464,135 (5.1×) |
| 256 | every 512 rows | 85,578 | 288,352 (3.4×) | 347,461 (4.1×) |
| 1,000 | never | 5,650 | 28,593 (5.1×) | 37,167 (6.6×) |
| 1,000 | every 512 rows | 5,711 | 20,863 (3.7×) | 23,719 (4.2×) |
| 2,000 | never | 1,415 | 6,544 (4.6×) | 8,380 (5.9×) |
| 2,000 | every 512 rows | 1,309 | 4,435 (3.4×) | 5,644 (4.3×) |

### Three things to read off it

**The update itself is 5–6.6× faster with a 256-row block.** The ledger
row's 5–10× was right, once the probe's multithreaded 9.5×/11.2× had been
withdrawn (task 71). The block size matters at the low end: 64 rows leaves
the product's inner dimension short, and gives 4–5×. Larger than 256 gains
little more, and holds more rows in the state; 256 is what the docs
recommend.

**A solve dilutes it, and the dilution is the whole story at a real
cadence.** The Gram is brought up to date before every solve, and a solve
is `O(k³)` on both sides of the table. At `k = 2,000` it takes ~30 ms
whichever way the matrix was built (`1/1309 − 1/1415` seconds per row,
times 512). Once the update is six times cheaper, that 30 ms is a third of
the blocked path's time. So "every 512 rows" lands at 4.1–4.3× across the
widths, and a cadence of every 64 rows would land near 2×. The number a
bank sees is the update's ratio, times how rarely it solves, which for
anyone at `k ≥ 1,000` is already rarely.

**It is not free to switch on where it does not pay.** The parameter is
refused with `window`, which snapshots the matrix every row, and with a
solve every row: `solve_every <= 0` or `max_rows_between_solves <= 1`. The
default `solve_every` is `halflife / 50`, which is `0` for `lam` and for an
infinite halflife, so those need it set. In both cases a block could never
hold more than one row, and the product would be a slower rank-1 update.
It is not refused narrow, because it still gains there, less. With a
256-row block and no solve:

| k | blocked, against per row |
|---:|---:|
| 16 | 1.6× |
| 64 | 2.4× |
| 128 | 4.0× |

```sh
# From the repository root: the widths and row counts are pairs, k then rows.
cargo run --release -p online-core --example gram_block_bench -- 16 262144 64 131072 128 65536
```

The guess that the rank-1 loop is in cache at those widths, and has nothing
to gain, was wrong by that much. That is why the example takes widths on
the command line, rather than leaving the guess in the docs.

### What blocking does not change

**Blocking does not change `n_eff`, the timing of every prediction, chunk
invariance, or `gram()`.** The flush is a function of the learned-row count
and the solve schedule, never of a chunk ending, and `gram()` reads through
the held rows without moving the block. What it does change: the merged
matrix is a floating-point sum in a different order, so a blocked fit's
coefficients agree with the per-row fit's to rounding rather than to the
bit. And the product's kernel follows the CPU's vector width, so a blocked
Gram's last bits can differ between machines, where the rank-1 path's do
not. Frozen fixtures stay unblocked.

## 19. Against `sklearn.linear_model.SGDRegressor` (task 72, 2026-09-08)

"How does this compare to scikit-learn" is the first question a reader has,
and the answer was nowhere. [docs/PLAN.md](PLAN.md) task 72 has the design
comparison: what each side is, and where the two disagree about
forgetting, leakage and grids. This section is the measurement behind it.

### The protocol

Reproduce it from the repository root. It used scikit-learn 1.9.0,
installed into the environment; it is not a dependency of this project.

```sh
uv run python scripts/sklearn_comparison.py
```

**Every contender is held to the rule this library guarantees: a row is
scored from the state as it stands, and only then learned from.** For
sklearn that is the setup its own documentation prescribes for out-of-core
work: a `StandardScaler` fitted online, then `partial_fit`. It is run twice: row by
row, which is our semantics exactly, and in mini-batches of 1,000, which is
what people write. Each contender is swept over a small grid of its own
settings and reported at its best, so this compares designs and not
defaults.

### Accuracy

**Accuracy is not the difference, and the first version of this table said
otherwise by accident.** 100,000 rows, `k = 20`, noise `0.1`, scored from
row 1,000. The noise ceiling is the R² of the generating signal itself,
which is what a perfect model would score:

| out-of-sample R² | stationary | drifting coefficients |
|---|---:|---:|
| noise ceiling | 0.9831 | 0.9923 |
| `SGDRegressor`, batches of 1,000 | 0.9830 | 0.9820 |
| `SGDRegressor`, row by row | 0.9829 | 0.9899 |
| `po.spec.sgd` | 0.9826 | 0.9906 |
| `po.spec.ewridge`, refit every row | **0.9831** | **0.9907** |

On the stationary stream everything is at the ceiling. On the drifting
stream the three row-by-row contenders are within 0.001 of each other and
0.002 of the ceiling, and the batched one is 0.010 below. *That* gap is the
staleness of a prediction made up to 999 rows before its update, not a
difference of algorithm. The cleanest evidence that the algorithms are the
same: `SGDRegressor` row by row at `constant, eta0=0.003` scores 0.9899,
and `po.spec.sgd` at `learning_rate=0.003` scores 0.9899. Same recursion,
same number.

Two corrections to what this section said on the morning of 2026-09-08:

- **The sweep had an edge.** `ewridge` was swept over halflives of 500 rows
  and up, and reported 0.9878 on the drifting stream, 0.003 behind `sgd`:
  the best point was the shortest halflife tried. At `halflife=100` it
  scores 0.9907 and ties. A sweep whose best point is at its edge is a
  grid, not a result. The script now sweeps 50 to ∞, and prints every
  sweep, best to worst.
- **"A halflife beats a learning rate tuned in rows" was the wrong
  reading.** A constant learning rate **is** forgetting: an LMS step of
  `eta` remembers about `1/eta` rows. On a stream whose rows are evenly
  spaced, a halflife in rows and a halflife on a clock are the same thing.
  The clock earns its keep when rows are *not* evenly spaced: a gap of a
  week should forget a week, not one row. This stream cannot show that,
  and the earlier reading claimed it anyway. `SGDRegressor`'s row-by-row
  number moved from 0.9840 to 0.9899 once it was given the `constant`
  schedule, rather than its `invscaling` default, whose decay is about
  convergence, not recency.

`average=True` is the best `SGDRegressor` setting on the stationary stream
(0.9830, Polyak averaging of the iterates), and the worst under drift
(0.817: the average remembers everything). `sgd` here has no averaging. A
halflife-weighted average of the iterates would be the version that fits
this library, and is not built.

### The first hundred rows of every group

**Where the Gram wins: the first hundred rows of every group.** The exact
solve is right as soon as the Gram is full rank, about `k` rows in. A
first-order method needs about `1/eta` rows *per direction* to get there.
That only shows on a short history, so here is the short history. It has
500 groups of 200 rows, each group with its own coefficients, at `k = 20`,
scored by position in the group. Here sklearn gets one estimator and one
scaler per group in a dict, row by row, because there is no other way to
give it a group. The bank gets `group="g"`:

| contender | rows 25–50 | rows 50–100 | rows 100–200 | rows/sec |
|---|---:|---:|---:|---:|
| noise ceiling | 0.9896 | 0.9903 | 0.9900 | |
| `SGDRegressor` per group, `invscaling` (the default) | 0.2731 | 0.4389 | 0.6356 | 3,350 |
| `SGDRegressor` per group, `constant, eta0=0.03` (its best) | 0.7277 | 0.9242 | 0.9789 | 3,342 |
| `po.spec.sgd`, `learning_rate=0.01` (before task 74) | −6.9 | −2.4 | 0.1095 | 19,696,993 |
| `po.spec.sgd`, `learning_rate=0.01` | 0.4328 | 0.7006 | 0.9067 | 11,559,468 |
| `po.spec.sgd`, `learning_rate=0.03` | 0.7182 | 0.9213 | 0.9788 | 18,519,660 |
| `po.spec.ewridge`, refit every row | **0.9693** | **0.9860** | **0.9882** | 5,130,803 |

By row 50 of a group, `ewridge` is within 0.005 of the ceiling. The best
first-order contender is 0.26 below it, and at the default schedule 0.72
below. The throughput column is the group feature: one bank call at 5.1
million rows/second, against a Python loop over 500 estimators at 3,350.

### Where `sgd` lost, and task 74

**This library's `sgd` also lost to sklearn's in the same table, and was
fixed the same day (task 74).** `po.spec.sgd` diverged at the start of
every group: the struck row. Its `scale_features` standardised a row
against the running moments from *before* that row (ENHANCEMENTS E24 as
first written). While those moments are two or five rows old, the variance
estimate can be tiny by chance, and the standardised value huge. Then one
LMS step with `eta · |z|² > 2` throws a coefficient far enough that the
next hundred rows do not bring it back.

sklearn's recipe is `scaler.partial_fit(x)` *then* `transform(x)`: the
moments include the row they scale. That bounds a standardised value by
`√n_eff`, and it is not a leak. The features of the row being predicted
are known at prediction time; the rule is about the target. A pure-numpy
LMS switched between the two orders on this stream reproduces both:
`−64,309` against `0.4328` at rows 25–50, with `eta=0.01`. The model now
gives the replica's number to four places. The 100,000-row tables above did
not show it because they score from row 1,000, long after the moments have
settled.

The two `sgd` rows that stand are sklearn's step at the same constant rate,
at 3,500× to 5,500× the rows per second. That is again the group feature:
one bank call, against a Python loop over 500 estimators and 500 scalers.

| rate | `po.spec.sgd`, rows 25–50 | `SGDRegressor`, rows 25–50 |
|---|---:|---:|
| 0.01 | 0.4328 | 0.4501 |
| 0.03 | 0.7182 | 0.7277 |

**What is left between them is *where the prediction is standardised*,
checked by switching the replica.** In sklearn's loop the row is predicted
against the moments from before it (`predict(scaler.transform(x))` comes
before `scaler.partial_fit(x)`), and learned against the moments including
it: two standardisations of every row. A replica doing that gives sklearn's
0.4501 / 0.7119 / 0.9111 to four places. Here it is one `z` for both. A
row is standardised the way every row the coefficients were learned from
was, `predict` is bounded by `√n_eff` like the step, and the scaler is one
pass over the row rather than two.

The gap is an under-convergence artefact rather than a better estimate.
The moments that include a row shrink its standardised value by about
`1/n`. A fit that has not yet grown into its coefficients scores a little
higher when its predictions are inflated by that much. And it closes with
the fit: 0.0044 by rows 100–200 at 0.01, and 0.0001 at 0.03.

### Throughput

**Throughput is the difference, and it is a difference in semantics.** Rows
per second on the same stream:

| contender | rows/sec | what a prediction saw |
|---|---:|---|
| `SGDRegressor`, row by row | 3,400 | every row before it |
| `SGDRegressor`, batches of 1,000 | 2,200,000 | every row before its batch |
| `po.spec.sgd` | 6,000,000 | every row before it |
| `po.spec.ewridge`, refit every row | 575,000 | every row before it |
| `po.spec.ewridge`, solve every 100 rows | 5,000,000 | every row before it, coefficients ≤100 rows old |

sklearn's fast form and ours are not doing the same thing. `partial_fit` on
a mini-batch is one BLAS-shaped update for 1,000 rows, and the predictions
inside the batch are up to 999 rows stale. Our loop is one update per row
in Rust, so every prediction is fresh, at 6,000,000 rows/second. Ask
sklearn for our
semantics, `partial_fit` per row, and it runs at 3,400 rows/second, three
orders of magnitude down. That time is Python's per-row overhead rather
than anything about the algorithm.

### A grid

**A grid is nearly free on one side only.** Six penalties over the same
stream, `k = 20`:

| contender | one penalty | six | ratio |
|---|---:|---:|---:|
| `SGDRegressor` | 2,180,972 | 580,096 | 3.76× the work |
| `po.spec.ewridge` | 5,359,787 | 3,309,081 | 1.62× |

Six penalties are six estimators to sklearn, each with its own update per
batch. Here they are six solves off one accumulator, and the accumulator is
what the row cost is. (The first run of this table read 3.66× and 1.40×.
The ratios move by a few tenths between runs, and the shape does not.)

### A wide row

**Where sklearn wins: a wide row, against `ewridge`, and by batching.**
`ewridge` keeps a `(k+1)²` co-moment matrix and updates it every row, so
both its memory and its per-row cost are quadratic in the feature count.
The R² column is scored from row 50 (`min_periods`). With 0.2 rows per
feature at `k = 10,000` nothing can learn much, and it should read near
zero, which is what a *sane* run looks like at that width. The
learning-rate note below says what an insane one looked like. The table is
the run of 2026-09-08 after task 75 (§20). Its first version had `sgd` at
185,641 and 6,944 rows/second, and the R² column did not move.

| contender | k | rows/sec | R² | state |
|---|---:|---:|---:|---:|
| `SGDRegressor`, row by row | 1,000 | 2,980 | 0.8513 | 0.03 MB |
| `SGDRegressor`, batches of 1,000 | 1,000 | 175,298 | 0.8687 | 0.03 MB |
| `po.spec.sgd`, `scale_features=False` | 1,000 | 407,925 | 0.8516 | 0.09 MB |
| `po.spec.sgd`, `scale_features=True`, before task 74 | 1,000 | 264,444 | 0.8221 | 0.11 MB |
| `po.spec.sgd`, `scale_features=True`, after task 74 | 1,000 | 269,264 | 0.8512 | |
| `po.spec.ewridge`, solve every 1,000 rows | 1,000 | 6,236 | 0.9274 | 8.71 MB |
| `po.spec.ewridge`, `gram_block_rows=256` | 1,000 | 33,914 | 0.9274 | 10.27 MB |
| `po.spec.ewridge`, `gram_block_rows=1024` | 1,000 | 37,065 | 0.9274 | 16.89 MB |
| `SGDRegressor`, row by row | 10,000 | 2,206 | 0.0322 | 0.31 MB |
| `SGDRegressor`, batches of 1,000 | 10,000 | 20,007 | 0.0338 | 0.31 MB |
| `po.spec.sgd`, `scale_features=False` | 10,000 | 33,271 | 0.0319 | 0.89 MB |
| `po.spec.sgd`, `scale_features=True`, before task 74 | 10,000 | 21,682 | 0.0209 | 1.06 MB |
| `po.spec.sgd`, `scale_features=True`, after task 74 | 10,000 | 22,989 | 0.0321 | |
| `po.spec.ewridge`, solve every 1,000 rows | 10,000 | 53 | 0.0412 | 859.54 MB |
| `po.spec.ewridge`, `gram_block_rows=256` | 10,000 | 269 | 0.0412 | 875.16 MB |
| `po.spec.ewridge`, `gram_block_rows=1024` | 10,000 | 379 | 0.0412 | 941.10 MB |

*The rows marked "before task 74" were measured before that task. This
table was run for task 75, and task 74 landed after it the same day
(commits `072d42e` and `0343b24`), recording its own figures only in the
prose further down. The "after task 74" rows copy those figures; their
state was not measured.*

**An LMS step is stable only while `eta · |z|² < 2`, and a standardised row
has `|z|² ≈ k`,** so the rate has to fall as `1/k`. Both libraries run at
`learning_rate = 0.2 / k`. The first version of this table ran both at 0.01
and timed two fits that had diverged (R² −6.8e8 for `sgd`, −5.8e26 for
`SGDRegressor`). The timings were the same to within noise, because a
diverged LMS costs exactly what a converged one does. But the configuration
was not one to copy, and the table had no column that could have said so.
Now it does.

**Read like for like, the wide row is not where sklearn is faster.**
`SGDRegressor` **at this library's semantics**, row by row and
predict-then-fit, runs at 2,206 rows/second at `k = 10,000`, and
`po.spec.sgd` at 33,271, fifteen times faster. At `k = 1,000` it is 2,980
against 407,925. The two produce the same predictions. Unscaled, the
correlation between their predictions from row 50 is:

| k | correlation of the predictions | R², `po.spec.sgd` / `SGDRegressor` |
|---|---:|---|
| 20 | 0.999999 | 0.9809 both, a mean difference of 0.02% of a prediction |
| 1,000 | 0.999922 | 0.8516 / 0.8513 |
| 10,000 | 0.9954 | |

What sklearn's 20,007 rows/second comes from is its batch: `partial_fit` on
1,000 rows at a time, with every prediction inside the batch made from
coefficients up to 999 rows stale. And since task 75 that batch no longer
makes sklearn faster on the wide row either. `sgd` is 1.7× it at this
table's 2,000 rows per call, and 2.4× at 20,000 (§20), with every
prediction from the state as it stands. Before task 75 the batch was 2.7×
faster at this width, and the remaining factor was the inner loop. `sgd`
cost a flat 13–14 ns per feature per row from `k = 1,000` to `k = 10,000`,
where sklearn's batched Cython loop is about 5 ns. This section's first
version added "the gather out of a columnar frame is not it — `to_numpy` of
the entire 10,000-column frame is 8.4 µs/row against `sgd`'s 142". That
compared one transpose of the frame with a bank that gathered every row
again, from 10,000 separate columns at a stride. That walk *was* most of
the 142. The same day, §20 took the number apart: 2.2 ns per feature. The
table below is re-run from it.

**`scale_features=True` did *not* match sklearn at these widths before task
74.** The correlation with `SGDRegressor`'s predictions fell to 0.978 at
`k = 1,000` and **0.521 at `k = 10,000`**, and the R² with it (0.0209
against 0.0322). That was the short-history defect from the other side. The
scaler standardised a row against moments from *before* it, and what makes
those moments immature is **few rows per feature**; a wide fit is that on
every row. The scaled fit *is* sklearn's recipe, a scaler in front of the
step. With the row inside the moments (re-run the day it landed, same
stream, same rates), it now matches sklearn more closely than the unscaled
one does:

| k | correlation with `SGDRegressor`: unscaled | scaled, before task 74 | scaled, after | R²: unscaled | scaled, before | scaled, after | `SGDRegressor` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.999922 | 0.978 | **0.99999995** | 0.8516 | 0.8221 | 0.8512 | 0.8513 |
| 10,000 | 0.9954 | **0.521** | **0.999997** | 0.0319 | 0.0209 | 0.0321 | 0.0322 |

The rows/sec column below is the unscaled setting. Scaled, it is 269,264 at
`k = 1,000` and 22,989 at `k = 10,000`: the scaler's own two passes over
the row (§20).

Where `ewridge`'s time goes was measured rather than argued, at
`k = 10,000` (2,000 rows), because "is it the coefficients it emits every
row?" is the natural question:

| `ewridge` variant, k = 10,000 | rows/sec | µs/row |
|---|---:|---:|
| the table's row (solve every 1,000 rows, `coef_every=0`) | 53 | 18,770 |
| the same with `coef_every=1` (10,000 coefficients emitted per row) | 52 | 19,249 |
| the same with one solve instead of two | 53 | 18,731 |
| `gram_block_rows=256` | 269 | 3,718 |
| `gram_block_rows=512` | 306 | 3,264 |
| `gram_block_rows=1024` | 378 | 2,647 |
| `gram_block_rows=2048` | 376 | 2,661 |
| `po.spec.sgd`, same bank, same 10,005-column output | 6,217 | 161 |
| the same `sgd` after task 75 (§20) | 19,120 | 52 |

`coef_every` defaults to 0, so the table emits no coefficients at all.
Turning it on doubles the output frame from 153 to 302 MB, and costs 2.5%.
One solve instead of two costs 0.2%. The bank, the emission and the frame
are all inside `sgd`'s 161 µs, which is 52 since task 75. Of that the
emission is 22: 10,000 coefficients per row is a second 80 KB row written
per row.

**What is left is the Gram, and it is moving the matrix, not computing on
it.** A rank-1 update of an 800 MB matrix (`10,001² × 8` bytes) reads and
writes all of it once per row. That is 1.6 GB of traffic per row, and
1.6 GB in 18.8 ms is **85 GB/s**. A plain in-place `a += 1` over the same 800 MB in
numpy runs at 123 GB/s on this machine (M4 Pro). So the update is within
1.5× of a pure memory stream, and nothing about output shape changes that.
The blocked Gram (§18) holds `B` rows back and applies them as one
rank-`B` product, so the matrix is touched once per `B` rows. The cost
becomes about `2k²` flops per row at ~54 GFLOPS: compute-bound instead of
bandwidth-bound, 5.1× at `B = 256` and 7.2× at `B = 1024`, where it levels
off (2048 adds nothing). The buffer is `B · k · 8` bytes, 82 MB at 1024,
which is what the state column's 941 MB against 860 says.

**State is what has to be kept and moved:** `bank.save_bytes()` for a bank,
and a pickle of the fitted estimator and its scaler for sklearn. At
`k = 10,000` that is 860 MB against 0.31 MB. The block adds throughput and
none of the memory: that is what E51 is for, and it is not a fix for this.
The matrix is stored full (`EwCov::c`, row-major `k×k`). A triangle would
halve it, and halve the block product's flops, but would still be 430 MB at
this width. So it does not change the answer, and is not done. **At a wide
`k`, `sgd` is the answer here:** `O(k)` like sklearn's, and faster than
sklearn's at the same semantics. And at `k = 1,000` the R² column repeats
the short-history lesson at a different scale. With 20 rows per feature,
the exact solve is at 0.9274, where every first-order contender is at 0.85,
even predicting from coefficients up to 1,000 rows old.

One more thing the table cannot show: the ecosystem is sklearn's, with
pipelines, `GridSearchCV`, calibration, and far more use than this has.
What this has is the stream: a clock, a halflife, one state per group,
chunk invariance, and a state file that is not a pickle.

### What the comparison gave back

**What the comparison gave back, in one paragraph.** For `ewridge`, nothing
to borrow. sklearn's recipe is a scaler and a schedule, and `ewridge` has
the scaler folded into its solve (`standardize`), and no schedule to tune,
because nothing is being descended. Its limit is the one the wide table
shows, and that limit is the Gram itself. For `sgd`, one thing, and it is
the order of two lines: scale the row against moments that include it. That
was done as task 74 the same day, with the row admitted at unit weight, so
`predict` still returns exactly the step's number. A halflife-weighted
Polyak average could be added as an option, if the stationary gap of 0.0005
ever matters to anyone. For the measurement, two lessons already recorded
in [docs/PLAN.md](PLAN.md) task 72: print the ceiling, and never report a
sweep whose best point is at its edge.

## 20. `sgd`'s per-feature cost (task 75, 2026-09-08)

The number §19 left standing as a target was `sgd` at 13–14 ns per feature
per row, against about 5 for `SGDRegressor`'s batched Cython loop. That is
why sklearn's batch was faster on a wide row. This section is that number
taken apart. **None of it was the arithmetic.** The step itself cost
2.25 ns per feature; the other 11 were how the row reached it, and how the
row was summed.

### Where the 14 ns went

Four places, measured one at a time on an M4 Pro, with `ONLINE_TIMING=1`
(per-phase milliseconds on stderr) and
`crates/online-core/examples/sgd_bench.rs` (the core step alone):

1. *The columns.* The bank held each feature as its own `Vec<f64>`, and a
   row was a walk across all of them at a stride of `n`. That is one cache
   line and one TLB entry per feature per row, with nothing reused. At `k = 10,000`
   that walk touched 10,000 pages per row, and the row cost 78–130 µs
   depending on the chunk's height (the stride). That is the "erratic with
   chunk size" the earlier tables showed. The row is now gathered once per
   chunk into a row-major buffer (`crates/online-polars/src/rows.rs`,
   `FeatureRows`). It is a tiled transpose, 128 rows by 64 columns at a
   time, so both the source lines and the destination lines stay in L1. The
   layout permutation is folded into the gather, which runs in parallel
   above 65,536 cells. The transpose moves 1.6 GB in and 1.6 GB out for
   20,000 × 10,000 in 60 ms: 3 µs per row, against the 75 it replaced.
2. *The step's own loop.* `Sgd::step` copied the row into `[1, x]`, then
   updated `beta[j][i]`, `zbuf[i]` and `g2[j][i]` by index through two
   levels of `Vec`, with a bounds check on each, per feature. It chose the
   schedule by a `match` inside that loop, and under `inv_scaling` raised
   `(1 + n)^power` once per *feature*. The scaler path allocated three
   vectors per row on top. Now the intercept is folded into the dot product
   as its coefficient (no copy), and the loops are zipped slices. The rate
   is raised once per row, and the scaler's buffers persist. That took
   2.25 → 0.93 ns per feature, and `scale_features=True` 14.8 → 2.6. It is
   bit-identical: a signature of every `step` and `predict` over 48
   configurations hashed the same before and after
   (`crates/online-core/examples/sgd_signature.rs`). The configurations are
   intercept × scaler × three schedules × two losses × two penalties, over
   400 rows with zero weights, clock gaps and a missing target.
3. *The dot product.* One running sum is a chain of dependent additions,
   three or four cycles each, whatever the core could do alongside: 10,000
   of them are 7 of the 9.3 µs the step still took. It is now eight
   interleaved partial sums, folded pairwise (`DOT_LANES` in
   `crates/online-core/src/sgd.rs`): 0.93 → 0.49 ns per feature. **This one
   changes bits.** The order is fixed by the code, the same on every
   platform, but it is not 0.3.1's, so `sgd` predictions differ from the
   last release's at rounding level. That is 1e-16 relative on squared
   loss, and 1.5e-11 at worst in the 48-configuration stress signature. The
   worst is under Huber at a constant rate, whose clipped gradient neither
   damps a perturbation nor amplifies it, over 400 rows of zero weights and
   gaps.
   The goldens (`tests/test_golden_pipeline.py`, 1e-12) pass unchanged;
   reordered arithmetic is what their tolerance is for.
4. *The accept walk.* `Iterator::all` over the row short-circuits, which
   keeps it scalar. A row is nearly always usable, and a fold over the
   compare vectorises: 1–2 µs per row at `k = 10,000`.

### The result

One bank, one spec, `learning_rate = 0.2 / k`, `scale_features=False`.
Learn is `fit_predict` on a fresh bank, and predict is `predict` on a
fitted one. "8 chunks" is the same stream fed as eight frames, which is
where the stride used to show:

| k | rows | learn, µs/row: before → after | ns/feature | rows/s | predict, µs/row | 8 chunks, µs/row |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 400,000 | 0.11 → 0.09 | 4.6 | 10,790,000 | 0.10 → 0.07 | 0.10 → 0.09 |
| 100 | 200,000 | 0.45 → 0.26 | 2.6 | 3,818,000 | 0.25 → 0.15 | 0.41 → 0.29 |
| 1,000 | 40,000 | 4.60 → 2.06 | 2.1 | 486,000 | 2.37 → 0.87 | 5.65 → 2.74 |
| 10,000 | 2,000 | 137.9 → 30.4 | 3.0 | 32,900 | 81.5 → 15.8 | 111.7 → 65.7 |
| 10,000 | 20,000 | 76.5 → 22.2 | 2.2 | 45,200 | 49.5 → 9.4 | 127.1 → 26.4 |

At `k = 10,000` that is 2.2 ns per feature, against sklearn's batched 5.0,
and 45,000 rows per second against its 20,007 (§19's run of the same day;
18,980 the day before). Every prediction is made from the state as it
stands, which sklearn's batch does not do. The core step alone, `sgd_bench`,
in µs per row:

| configuration, k = 10,000 | before | bit-identical | with the 8-lane dot |
|---|---:|---:|---:|
| constant, unscaled | 22.5 | 9.3 | 4.9 |
| constant, unscaled, `l2=1e-4` | 22.7 | 9.3 | 4.8 |
| constant, `scale_features=True` | 147.9 | 25.7 | 19.9 |
| `inv_scaling`, unscaled | 65.7 | 9.4 | 4.8 |
| `adagrad`, unscaled | 32.1 | 13.8 | 9.0 |

### What is left, at `k = 10,000`

**At `k = 10,000` and 20,000 rows, 20 µs per row is left:**

| part | µs per row |
|---|---:|
| the step | 4.9 |
| the transpose | 3 |
| the data summary (task 35) | about 7 |
| the plan, the output row, and the per-call constant of the frame hand-off | the rest |

The data summary is a Welford update of a 48-byte record per feature per
row, measured by switching it off. It is now the largest item. Making it
cheaper means laying the records out column-wise, so the update vectorises.
That is a state layout change, either a `SCHEMA_VERSION` bump or a serde
mirror of the wire form, for perhaps 4 µs of the 20. Not done. The cheap
version, comparing before storing `min` and `max`, measured no change and
was dropped.

The frame hand-off is about 8 ms per call at 10,000 columns, 0.8 µs per
column in pyo3-polars' `PyDataFrame` extraction (`get_columns`, then one
`_export` per Series), none of it this library's. It is what separates the
2,000-row line from the 20,000-row one (4 µs per row against 0.4). It also
separates the "8 chunks" column at 2,000 rows from the rest: 250 rows per
call carry 32 µs each of it. **Feed wide frames in tall chunks.**

### The scaler

**And the scaler.** `scale_features=True` is 20 µs per row in the core,
against 5 unscaled. The standardised row is `(x - mean) / sqrt(var)` per
feature, one square root each, and the scaler's own moment update is a
second pass over the row. That is the cost of the feature. Task 74 changed
which moments it uses, not how many. The row is standardised against the
moments with itself admitted, read off the accumulator as one more Welford
step per feature, without writing it back. `sgd_bench` measured 20.7 µs a
row before it and 21.3 after, at `k = 10,000`, within the run-to-run spread
(the table above is the earlier run).

## 21. What inspecting the plan costs `fit(lf)` (2026-09-18)

Asked whether `fit` had become slower on small inputs: it had not. What is
true is structural, and it has been true since `fit` existed.

### The bisect

Each tag's `python/polars_online/` was run against **one** compiled
extension, so only the Python changed: 100 rows, `min` of 50 runs of 20
calls, a fresh `ModelBank` per call.

| version | `fit(DataFrame)` | `fit(LazyFrame)` | plan overhead |
|---|---:|---:|---:|
| v0.6.0 | — | — | `ModelBank.fit` did not exist |
| v0.7.0 | 0.102 ms | 0.371 ms | +0.269 ms |
| v0.7.1 | 0.103 ms | 0.381 ms | +0.278 ms |
| v0.7.2 | 0.102 ms | 0.376 ms | +0.274 ms |
| v0.7.3 | 0.102 ms | 0.382 ms | +0.280 ms |
| **after** | 0.104 ms | **0.164 ms** | **+0.060 ms** |

0.371 → 0.382 across four releases is ~3% drift, inside the run-to-run
spread, and `fit(DataFrame)` never moved. `ConsumedSourceWarning`'s
`explain` call, added in 0.7.1 and predicted to cost ~0.1 ms, cost about
0.01: measuring `explain` alone had caught first-call warmup, not the
steady state. **`fit` is new in 0.7.0**, so there is no earlier number. A
caller who moved to `fit(lf)` from `fit(df)` or `fit_predict_batches` meets
this cost for the first time, and reads it as a slowdown.

### What it actually is

`fit(lf)` inspected the plan twice: `explain` for the Python-scan test,
and `serialize(format="json")` plus a walk for the order hazards. That is a
fixed ~0.27 ms, against ~0.10 ms of fitting. On a small input the
inspection *is* the call: 3.7× `fit(df)`, unchanged since 0.7.0.

### The fix

**The fix is not to make the scan faster.** Read `explain` once and share
it, then use its text as a filter: no `JOIN`, `AGGREGATE` or `UNIQUE`
means no node the walk can report. That skips the JSON entirely on the
plans that have no hazard, which is most of them. Sharing is what makes it
a saving: reading `explain` twice costs more than the JSON scan it avoids
on a small plan (0.105 ms against 0.074 ms). Both are in
`python/polars_online/_frame.py`: `_plan_text` reads the plan once, and
`_order_hazards` walks the JSON only when that text names one of the three.

The JSON path's own shape says the same thing. By plan, `serialize` /
`json.loads` / walk:

| plan | JSON bytes | serialize | loads | walk | total |
|---|---:|---:|---:|---:|---:|
| trivial scan | 2,176 | 0.190 ms | 0.024 | 0.037 | 0.074 ms |
| with a join | 3,876 | 0.017 ms | 0.033 | 0.060 | 0.115 ms |
| 50 `with_columns` | 13,106 | 0.027 ms | 0.061 | 0.123 | 0.210 ms |
| 200-column schema | 74,337 | 0.137 ms | 0.667 | 1.280 | **2.13 ms** |

*The trivial-scan row's parts (0.190 + 0.024 + 0.037 ms) exceed its
0.074 ms total, so one of its numbers is misrecorded. Both are kept as
recorded.*

It scales with schema width, and the **walk** dominates there (1.28 ms of
2.13), not the serialization, so trimming the serializer would have been
optimising the wrong half. The filter removes all of it: `explain` on that
200-column plan is 0.001 ms, because a plain frame's explain is one line
whatever the width, while `serialize` dumps the whole schema.

### Measure the case the user has

**Measure the case the user has, not the one that flatters.** The first
pass of this used a `lam=1.0` spec and reported no gain. That spec is
order-free, so since 0.7.2 it already skipped the order check, and the
benchmark was measuring a fast path built two releases earlier rather than
the common one. A `halflife` spec never qualifies, and that is where the
2.3× is.
