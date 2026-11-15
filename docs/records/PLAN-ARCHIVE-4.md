<!-- kind: record -->
# The plan archive, part 4 of 8

Moved out of `docs/PLAN.md` on 2026-11-14 (task 398), text unchanged: the plan had
grown to 1.7 MB, more than a reader can hold, and most of it was finished work.
This part holds the full entry of each finished task and the text of each closed
section, in the order they stood in the plan. `docs/PLAN.md` keeps one line per
finished task, linking the entry here (`#task-N`), and every section heading with
a line saying where its text went (`#sNN`).

Earlier text is in [part 3](PLAN-ARCHIVE-3.md).

Later text is in [part 5](PLAN-ARCHIVE-5.md).

<a id="s11"></a>
## 11. Task list

<a id="task-251"></a>
- [x] 251. **Performance and memory budgets that only tighten** -- the user,
      2026-10-11: "How can we find these problems sooner next time", then
      "The instruction count and memory budget are good but we want to
      avoid slipping by 5% repeatedly". Built right after task 250's fixes,
      so the first budgets are set on the faster build. (1) A committed
      budget file: an instruction count per hot-path case (one row of
      `ewridge`, `sgd`, `pa`, `seqtest`, `bocpd`, `kmeans`, `micro`, the
      clock step, the bank's per-row path), measured under Valgrind with a
      test-only harness (`iai-callgrind` or equivalent) on CI's Linux
      runners, and a bytes-per-row budget per output field plus peak RSS for
      a few standard workloads. (2) Each push is measured against the
      budget, not against the previous run, so small regressions add up
      against one fixed number; the tolerance is set from the measured
      repeatability (instruction counts repeat to well under 1%: about 1-2%,
      where timings needed 5-10%). (3) A ratchet: a case clearly faster or
      smaller moves its budget down in the same change. (4) Raising a budget
      needs a stated reason in the file and a CHANGELOG *Performance* entry,
      and a test holds both. (5) A backstop: each case's total drift since
      the last release is reported on every push, and past about 10% needs
      an explicit sign-off before a release. Measure the harness's own
      repeatability on CI before trusting a tolerance; Valgrind on CI is a
      test tool, nothing in the wheel.
      **Before 1.0** (the user, 2026-10-11: "Build the budgets before 1.0"):
      the order is task 250's fixes, then 251, then the release (task 247).
      *Built 2026-10-12* (`0790424`, `b1e587f`, merged): `crates/online-bench`
      (14 Callgrind cases through Gungraun 0.20.0, iai-callgrind's renamed
      successor; unix-only dev-dependency; 1,000 rows after 2,000 of
      warm-up; the bank case on its own one-thread pool); `budgets.toml`
      (toolchain, runner and harness pinned; tolerances 2% instructions, 2%
      memory, 15% RSS; sign-off past 10% drift); `scripts/budgets.py`
      (check, raises, apply, release-check); `budgets.yml` (gating, the
      bench run twice, the runs must agree within a quarter of the
      tolerance), `release.yml`'s `publish` needs it; memory: 890 field
      budgets set locally from `estimated_size`, three RSS runs per
      platform. **Waiting on CI's first push:** the instruction and RSS
      budgets (read the Budgets run's second-run column and that no case is
      zero, `apply measured.json`, `apply rss-*.json`, commit, and the next
      push must pass with no new rows); at release, `apply ... --release
      1.0.0` (RELEASE-READINESS step 12). Also in that commit: CI's mutants
      under nextest (`--run-ignored=all`, 1,840 tests per mutant, checked
      against cargo-mutants' source) and task 246's cap replaced by a 12G
      systemd scope -- RLIMIT_DATA counted virtual memory and failed 4 of
      32 baselines -- proven on every run by a `memory-cap` job.
      *Measured on the way, not changed:* wide `ew_cov` (20 features, 200k
      rows) peaks +830 MiB for a 368 MB output (2.25x); a 5-feature ewridge
      streaming 1M rows from a parquet scan to a sink peaks +290 MiB.

<a id="task-253"></a>
- [x] 253. **Thread count through the API (before 1.0; the user: "Go" on
      the design below, 2026-10-12)** -- the user, 2026-10-12: "We should control the
      number of threads through the api not env variables, can we fit that
      into 1.0?" Today the bank's pool is one process-wide `OnceLock`,
      sized once from `POLARS_ONLINE_MAX_THREADS` at the first call and
      ignored afterwards; `po.thread_pool_size()` only reads it; the CLI has
      no flag; the in-API knobs are per spec (`gram_threads`, `shards`) and
      `po.gbdt`'s `threads`. Polars 2.0 has no per-call thread parameter
      either (`pl.thread_pool_size()` and `POLARS_MAX_THREADS` only), so the
      name is ours. Design recommended: `threads=` on `ModelBank(...)`,
      `ModelBank.load(...)`, `po.fit_predict`/`predict`, the frame
      namespaces, the pool-using `po.gram` functions, `--threads` on the CLI
      and a `threads` key in the TOML; `None` is the process pool (the
      variable stays as its default); an int is a pool of that size from a
      size-keyed registry of weak references, so a hundred banks asking for
      8 share one pool; not state (a loaded bank takes its own); outputs
      are independent of it (task 231: solve bits do not depend on the
      pool), held by a test at 1, 2 and 8. Only five call sites enter the
      pool (`bank.rs` fit and predict, `gramfit.rs` x2, `runner.rs`).
      Adding a keyword is non-breaking, but the names in the snapshot are
      not: `gram_threads`/`shards`/`po.gbdt`'s `threads` are decided with it.
      *Decided 2026-10-12* (the user: "Gram threads auto should be the
      default", after asking why `gram_threads` differs from `threads`:
      `threads` sizes the pool, `gram_threads` caps how much of it one
      spec's `k x k` update may use): `gram_threads` accepts `"auto"`, as
      `shards` does, and `None` resolves to `"auto"` (it was 1), decided
      before 1.0 because a changed default after it is a major (D3). The
      output is the same to the bit at every count, so only speed moves;
      the worker verifies "loses nothing" first (a many-group bank at 800+
      features, a many-group narrow bank, a single wide group) and leaves
      the default at 1, reporting, if any realistic configuration is slower.
      *Built 2026-10-12, the default held at 1 for the user's call.*
      `pool::Threads`: `None` is `pool()` as before; `Some(n)` comes from
      a `Mutex<HashMap<usize, Weak<ThreadPool>>>`, the lock held across a
      build so two asks build one pool, dead entries swept on each build,
      and the process pool itself when it is already built with `n`
      threads. A pool lives while a bank (or a gram call) holds its `Arc`;
      `Some(0)` and a pool the system will not start are `ComputeError`s
      naming `threads`. The runner's pool site (`runner.rs`) is Polars'
      pool, not ours, so the runner reaches the bank's through
      `RunConfig::threads` and `open_bank`. `with_windows` runs no bank
      and takes no `threads`. Pickle keeps the count beside the state.
      `gram_threads` takes `"auto"` (`CountOrAuto`, shared with `shards`;
      the layout is unchanged, a count serializes as before, schema stays
      61), and a count is now capped at the pool's size
      (`Threads::workers` in `ewcov.rs`, `usize::MAX` being `"auto"`). The
      default stayed 1: on 14 cores, load 2-6, 7 and 11 interleaved rounds,
      `"auto"` against 1 was 1.00x on 64 groups at k=20, 1.01x on 64 at
      k=500, 1.02x on 14 groups at k=800, 1.6x on 4 groups at k=800, 3.0x
      on one group at k=2,000 and 3.7x with `gram_block_rows=256`, and
      0.89-0.91x on 28 groups at k=800 (10 of 11 rounds below 1): with
      every thread holding a group, a row's split adds only its hand-out.
      Flipping it, or making `"auto"` split only when the pool has idle
      threads, is the user's decision.
      *Part 2, 2026-10-12: `"auto"` adaptive, and the default.* The
      coordinator's brief asked for `max(1, P / T)` per task, `T` the
      (spec x group) tasks of a phase (`process`), `P` the bank's pool.
      Measured, that floor lost the four-group win (1.29x, against 1.57x
      for the whole pool): with an even share a thread whose group is done
      has nothing to steal. A curve of every count at 2-13 groups of 800
      (14 cores, ten fast) showed a split paying up to seven groups and
      costing 2-10% at every count from eight to 13. Built instead:
      `gram_share(T, P)` = 1 when `2T > P`, else `min(P, 2P / T)` (twice
      the even share), in `pool.rs`; `process` sets it on each task's
      stream per segment (`Stream::set_gram_threads`, not state); a count
      is unchanged. Two runs of 11 interleaved rounds, load 7-12, `"auto"`
      / 1: one group k=2,000 2.93/3.00x, block 256 3.56/3.51x, 4 groups
      1.66/1.58x, 7 groups 1.09/1.00x, 14, 28 (ewridge and ew_cov), 56 and
      64 groups 0.98-1.05x (the one-thread path, so noise). Accepted, so
      `None` resolves to `"auto"` (`gram_threads_sharing`, `defaults.rs`,
      the snapshot's two `stream.gram_threads` lines). A load given no
      specs runs the saved spec's setting, which for a spec that left the
      key out is now `"auto"`; the bits are the same either way. Held by
      `bank/threads_tests.rs` (the rule's values, and the share each task
      is given at 1, 4, 14, 28 and 64 groups on 14 threads) and
      `test_auto_shares_the_pool_and_moves_no_bit` (1, 4, 28, 64 groups x
      pools 2, 8, default x 1/7/600 chunks, to the bit).

<a id="task-259"></a>
- [x] 259. **Release 1.0.1** -- the user, 2026-10-13: "We want not only
      tests of identity but tests of performance across thread counts. Then
      the rest." A patch: task 255 (a)'s peak cut and speed-up, the three
      fixes task 252 found, the unused window bindings removed; no output
      changes. Before it: identity at 1, 2, 4, 8 threads and the default,
      and performance measured at each (every count faster than 8950c46;
      `ew_cov` +16% at one thread); instruction cases `bank_ew_cov_wide_t1`
      and `_t4` (Callgrind's two runs of t4 agree to 0.02%); a scoped
      mutation run over `bank/assembly.rs` (233: 178 caught, 12 unviable,
      43 missed -> 34 caught by fields held to their parts, 9 equivalent);
      the suite on py-polars 1.37.0 (6,861 + 21 passed); `compare_release`
      against 1.0.0: 297 fields identical; `benchmark.py` against 1.0.0
      built here from its tag: `ew_cov` +6.8%, `corrchange` +3.3%, the rest
      within scatter (against PyPI's wheel a few rows read 2-7% slower, the
      build's provenance). Steps after the bump: full gate, push, CI green,
      dispatch, the user's PyPI click, verify, budgets `--release 1.0.1`,
      `RELEASES` += 1.0.1.
      *Done 2026-10-13*: the first dispatch (38443180802) failed a Windows
      leg on faer's per-thread buffer and was cancelled (task 258); the
      second (38455505928, `92554fe`) passed every job, the user approved,
      PyPI has 1.0.1, verified from a clean venv (1.0.1 / 1.0.1 / 61, a
      fit). Its tag job was refused ("refusing to allow a GitHub App to
      create or update workflow `.github/workflows/benchmark.yml` without
      `workflows` permission"): task 258's push had changed that workflow
      on `main` while the run waited. The tag (`v1.0.1` on `92554fe`, the
      run's notes) and the GitHub release (the run's 12 files, the same set
      as 1.0.0's) were made by hand; RELEASE-READINESS step 9 now says to
      push no workflow change until the tag exists. Step 12: budgets
      recorded `--release 1.0.1` (without the four-thread case `main`
      dropped); `RELEASES` += 1.0.1.

<a id="task-258"></a>
- [x] 258. **On x86, faer's matrix product commits about twice the L3
      cache per thread** -- found 2026-10-13: CI 38410183746 (`b796502`, a
      plan-only push) failed its Windows Python 3.14 leg in the Rust tests
      of online-core: "memory allocation of 1006632960 bytes failed" on a
      rayon worker in `pieces::syrk_lower_cut` -> `faer::linalg::matmul` ->
      `private_gemm_x86::gemm`. The same code passed every leg one push
      before. `private-gemm-x86` 0.1.20 (a dependency of faer 0.24 through
      its `std` feature, so not switchable off) keeps a thread-local packing
      buffer of `2 x L3` bytes, the L3 size read from CPUID, made at a
      thread's first product (lib.rs ~1286: `Vec::with_capacity(lhs_size +
      rhs_size)`); on that runner 960 MiB a thread. Linux only backs the
      pages touched and Apple Silicon takes another path, so it never shows
      there; Windows commits the reservation against the page file, so
      many threads multiplying at once -- the parallel Rust tests, or a
      wide Gram on many `gram_threads` -- can exhaust the commit limit.
      Present since faer 0.24 was adopted, 1.0.0 included; not from
      1.0.1's changes. **Next:** measure on Windows (a CI job: commit
      charge per thread for a wide `gram_threads` fit); options, cheapest
      first -- run the Gram's products on fewer threads on Windows, call
      the product so that a reused thread is the one that multiplies, or
      ask faer upstream for a bounded buffer (an upstream report needs the
      user's word, and is filed by the user).
      *Decided 2026-10-13* (it failed a second Windows leg, in the 1.0.1
      release run 38443180802; the user: "Your reco on 1.0.1 and the
      subsequent fix as long as we can validate the code against an
      oracle"): 1.0.1 ships with CI's Windows Rust tests at
      `--test-threads=2` and a CHANGELOG known issue; the fix is 1.1.0's,
      a Gram product whose memory is bounded, validated against an
      independent oracle (`crate::oracle`, faer's LU / a plain sum), and
      measured on Windows CI. It may move last bits on x86, hence a minor.
      *Built 2026-10-13* (branch `task-258-bounded-product`): faer without
      its `std` feature, the only one that links `private-gemm-x86`, with
      the rest of what `std` turns on (`faer-traits`, `pulp`, `nano-gemm`
      `std`, `libm` `arch`; `gemm`'s through `rayon`) named in the
      workspace so Cargo's unification keeps it. Chosen over calling the
      `gemm` crate from `pieces.rs` and `ewcov/par.rs` (or a kernel of our
      own) because faer's Cholesky, triangular solves and eigensolver also
      call `private_gemm_x86` on x86 (`internal/mod.rs`, `triangular.rs`,
      `has_spicy_matmul`), on the same pool threads, which no change to our
      own calls reaches; and because aarch64 already ran exactly this path,
      so nothing there moves (release_probe's 297 fields and 31 states,
      `examples/product_digest.rs`, golden and fixtures, all to the bit; speed
      the same within scatter, PERFORMANCE §41). Nothing new linked; faer's
      `npy` default dropped (needs `std`); `rand` and `sparse-linalg` kept,
      unused, because without them a 20-feature `ew_cov` (no faer code)
      measured 5.5% slower on aarch64 -- code generation of the changed
      crate graph; with them it is the old build's speed. Oracle:
      `oracle::dot`/`dot3` (Dekker products, Shewchuk partials, rounded
      once), `pieces/oracle_tests.rs` and `ewcov/par/tests.rs`'s merge test,
      held to `γ₂ₙ₊₂ (|C| + |α|Σ|dd|)` (`γ₂ₙ₊₆` for the merge); a 1e-12
      scale or a dropped row fails them. Memory: `online-bench`'s
      `product_memory` test, a per-thread counting allocator, every thread
      under 24 MiB (aarch64: 6.1-9.6 MiB). CI's Windows `--test-threads=2`
      removed; `benchmark.yml` times `gram_threads_bench` and
      `solve_threads_bench` at `k` = 1,000 on its x86 runner. **Left:** on
      CI, the memory test passing on the Linux and Windows legs, and a
      Benchmark dispatch on the branch against one with `ref` = `main`, for
      x86's speed (not measurable on this machine, which has no x86 nor
      Rosetta); the to-the-bit pieces tests stay guarded off x86 until a
      CI run shows the `gemm` crate's x86 kernels sum a tile as the whole.
      *Done 2026-10-13, measured on CI* (`e157e46`, run 38466643836;
      reversed for x86 by 264 and 267, which put faer's x86 kernel back on
      every x86 build, Windows included, with CI's Windows Rust tests at
      `--test-threads=2` and `product_memory` off x86-64; this paragraph
      sat under the 1.1.0 record until task 269 moved it back): every
      leg green, the Windows legs' Rust tests at full parallelism again (no
      `--test-threads=2`), and `product_memory` holding each thread under
      24 MiB on Linux and Windows x86. x86 throughput, CI's Benchmark on
      `e157e46` against `92554fe` (one run each, 4 cpus, load 5.2): 35 of
      38 configurations 1.00-1.31 of the old (the wide ones 1.12-1.15),
      the rest 0.95-0.99, inside one run's scatter; Budgets green.

<a id="task-260"></a>
- [x] 260. **faer's x86 kernel back on Linux and Intel macOS, off Windows
      only** -- found 2026-10-14 by the user's Linux session (Intel Xeon):
      `po.gram.solve_subsets` at one thread 15-41% slower than 1.0.1 at
      `03abe0e`, traced to task 258 (faer without `std`, so no
      `private-gemm-x86`). Measured with the kernel (B) against without (A),
      interleaved on one machine each: the Xeon's solves 0.67-0.75 of A's
      time (1.0.1's speed); wide `ewridge` (200-1,000 features) on the
      default schedule 8-38% faster on five x86 processors (CI's AMD Zen 3,
      Zen 4, Zen 5, Intel Ice Lake; the user's Xeon), blocked 1-52%;
      `benchmark.py`'s 38 narrow configurations level (median 1.00);
      `lasso` within 5%. A microbenchmark found a thread 3x slower at the
      rank-one update right after the kernel's blocked product on Zen 4/5
      (its hand-written assembly returns without `vzeroupper`, checked in
      `private-gemm-x86` 0.1.20's code generator; our SSE2 code pays for
      the dirty AVX state), none on Zen 3 or the Xeon, and no spec showed
      it on any processor; left as measured (PERFORMANCE §43). Memory on
      Linux is a reservation, backed where touched; strict overcommit
      counts it, as in 1.0.1. *Decided 2026-10-14* (the user: "Yes" to
      landing it): `online-core`'s manifest turns faer's `std` on for
      `cfg(all(target_arch = "x86_64", not(windows)))`; aarch64's and
      Windows' feature graphs identical to before (`cargo tree -e
      features`); `product_memory` runs on Windows and off x86-64 only.
      CHANGELOG's Fixed and Changed entries now say Windows; PERFORMANCE
      §43 has the tables. CI runs 38610653398, 38625365356, 38626170815,
      38645950981, 38658964449; the user's artifact
      claude.ai/artifact/Tb3tizyExD1PLqpxnwknDe.

<a id="task-263"></a>
- [x] 263. **Why the bounded product costs twice as much on Windows** --
      found 2026-10-14 (CI 38658964449, a Windows runner, Intel Xeon
      8573C, against 1.0.1's wheel): wide `ewridge` 0.74-0.88 of 1.0.1's
      speed on the default schedule and 0.37-0.71 with `gram_block_rows`
      256, where the same processor family without the kernel on Linux
      loses 1.34 at most; commit for a 4-thread wide fit 374 MiB against
      2,382. Hypothesis: the `gemm` crate allocates each product's packed
      right-hand side (`gemm-common` 0.19 `gemm.rs` ~437,
      `MemBuffer::new` per call; 15 allocations, 16 MiB a flush at 1,000
      features), and Windows returns blocks that large to the system on
      free, so each product pays page faults again; Linux's allocator keeps
      them. **First check:** `online-bench/examples/alloc_probe.rs` on a
      throwaway branch, the blocked Gram timed with and without a
      per-thread cache of freed blocks of 1 MiB or more, Windows against
      Linux. **Then:** if it is the allocation, a fix that keeps the
      memory bounded (a bounded per-thread reuse of the packing, in
      `online-py`'s allocator path or upstream in `gemm`), measured on
      Windows against 1.0.1; if not, the user chooses for Windows between
      the bounded product and 1.0.1's kernel. Before 1.1.0 (the user,
      2026-10-14: "Yes"). The CHANGELOG's Windows performance entry comes
      with it.
      *Found 2026-10-15* (`online-bench/examples/alloc_probe.rs` on the
      throwaway branches `win-alloc-263`/`-263b`, since deleted; CI
      38664722353, 38665929118, 38667980193): (1) **allocation**: a
      per-thread cache of freed blocks of 1 MiB or more made the blocked
      Gram 18-21% faster at 1,000 features on a Windows runner (AMD Zen 5;
      15 large allocations, 23.5 MiB a flush), nothing on Linux; (2)
      **cache detection**: `gemm-common` 0.19 reads cache sizes from sysfs
      on Linux and CPUID elsewhere, and under the runners' hypervisor an
      AMD EPYC 7763 reports L1/L2/L3 32 KiB / 512 KiB / **0** on Windows
      against 16 KiB / 256 KiB / 32 MiB on Linux (`gemm_common::cache::
      CACHE_INFO`); `gemm` sizes its blocks from it, and there its path was
      30-55% slower than the kernel where Linux has them level; an Intel
      Xeon 6973P read 48 KiB / 2 MiB / 480 MiB on Windows, so (2) does not
      explain (3) **the rest** of the Python-level 2.7x on the Intel 8573C.
      A physical Windows PC may not show (2). Commit model of the kernel:
      2 x L3 per thread that has ever run a large product, held until the
      thread exits (desktop L3 16-64 MB: 32-128 MB a thread; the runners'
      260-480 MiB: 520-960 MiB a thread); in a bank with many groups every
      pool thread can multiply, not only `gram_threads`.
      **Options recorded, none chosen:** (a) ship the bounded path on
      Windows as on `main` since task 260 (commit 374 MiB against 2,382 for
      a 4-thread wide fit; wide `ewridge` 12-26% slower on the default
      schedule, 29-63% with `gram_block_rows`, narrow specs level);
      (b) a bounded per-thread reuse of freed large blocks in a
      `#[global_allocator]` that forwards to `PolarsAllocator` (~15-20%
      back; changes the line CLAUDE.md says must stay, so the user's word);
      (c) 1.0.1's kernel on Windows too (one path and the same last bits on
      all x86; the wide speed back), with CI's Windows Rust tests at
      `--test-threads=2` again and a CHANGELOG known issue plus a docs note
      giving the commit rule and the knobs (`threads=`, `gram_threads=`,
      the pool size) -- the user's suggestion, 2026-10-15: "we could just
      reduce the thread count on windows ci runners and such"; (d) (c) plus
      a Windows-only cap on the threads that run products, holding 2 x L3 x
      threads under a share of the commit limit; (e) upstream asks, the
      user's to file: a bounded or reused packing buffer in
      `private-gemm-x86`, cache detection under hypervisors in `gemm`.
      *Deferred by the user 2026-10-15*: "Let's not make changes only for
      windows for now. Record everything and we will pick later." Then,
      the same day: "Keep the kernel for every build for now, record all
      the findings for later work" -- task 264. The options above stay
      open for later work, measured against that build.
      *Closed 2026-11-03, the user: "263 seems like we currently have the
      best option."* faer's kernel stays on every x86 build (task 264); the
      Windows commit of 2 x L3 a multiplying thread is documented in
      PERFORMANCE §43 and repeated as a Known issue in 1.4.0's CHANGELOG
      (pointing to 1.1.0's); `threads=` bounds it. The upstream asks of (e)
      stay the user's to file.

<a id="task-264"></a>
- [x] 264. **The kernel on every x86 build** -- the user, 2026-10-15:
      "Keep the kernel for every build for now, record all the findings for
      later work." `online-core`'s manifest turns faer's `std` on for
      `cfg(target_arch = "x86_64")` (task 260 had `not(windows)`), so
      Windows runs 1.0.1's kernel and its last bits again, with its commit
      of 2 x L3 per thread that multiplies (2,382 MiB for a 4-thread
      1,000-feature fit on a runner with 260 MiB of L3; 374 MiB without the
      kernel, which ran wide `ewridge` 12-63% slower there). CI's Windows
      Rust tests at `--test-threads=2` again, as in 1.0.1; `product_memory`
      off x86-64 only; aarch64's feature graph unchanged. CHANGELOG:
      task 258/260's Windows Fixed entry and the x86 last-bits entry become
      a Known issue with these figures; PERFORMANCE §43 says every x86
      build. All findings and the Windows options are task 263's, for later
      work.

<a id="task-265"></a>
- [x] 265. **A wide `ewridge` at 0.56 of 1.0.1's speed, from one settle a
      run** -- found 2026-10-15 by the 1.1.0 release benchmark on an Apple
      M4 Pro, both builds compiled here: 1,000 features, 2,000 rows, a solve
      every 256 rows, 38.6k rows/s on 1.0.1 against 21.9k with
      `gram_block_rows` 256, 13.2k against 11.4k without; every
      `benchmark.py` row unchanged (50 features at most) and
      `compare_release.py` identical. Bisected to task 257 (b) (`28a1525`;
      `63c2495` before it 37.3k/12.6k): its run budget counts the run's
      feature rows (`ChunkOut::run_rows`, `+ spec.k()`), so a 1,000-feature
      chunk became about nine runs of 240 rows where it had been one, and
      each run's end settled the readiness shares (task 140,
      `AnyModel::flush` -> `settle_readiness`), which takes a solve's
      `inverse_diagonal`, a solve against the identity, about `k³`
      (sampled: `ewridge::settle` -> `shares` -> `inverse_diagonal` the
      largest share of the profile). **Fixed** by settling once a
      stream's rows of a chunk are learned (`Stream::settle_readiness`,
      called after the bank's run loop); a run's `flush` keeps
      `marginal`'s held rows only. The shares are the same bits whenever
      they are taken, so no output moves, and no factor outlives the chunk
      that solved it (`readiness_settle_tests`, now also holding that a
      run's `flush` settles nothing). Measured after: 34.9-36.6k and
      13.7-14.3k at a load of 5-7, against 20.6-22.4k and 11.1-11.4k for
      `main` in the same rounds. `scripts/wide_benchmark.py` (the wide
      configurations that found it) joins step 2 of the release steps
      (docs/RELEASE-READINESS.md).

<a id="task-266"></a>
- [x] 266. **Windows back on the `gemm` crate: the kernel's products were
      wrong there on a processor that reports no L3** -- found 2026-10-15:
      CI on task 264's push (`5959531`, run 38720754422) failed its Windows
      Python 3.14 leg in `gramfit::trisolve_tests::the_solve_is_the_lu_solve_
      within_its_forward_error_up_to_1500`: n = 1,500, `2.69e-4 > 2.25e-6`
      (κ ≤ 1,501). Rerun on eight Windows runners (run 38731522819): all
      four AMD EPYC 7763 failed with that same value, all four EPYC 9V74 /
      9V45 passed; the Linux legs (EPYC 7763, the kernel since task 260)
      pass it, and every Windows leg on the `gemm` crate (tasks 258-260)
      did. `private-gemm-x86` 0.1.20 reads cache sizes from sysfs on Linux
      and CPUID elsewhere, and sizes its blocks from the L3 (`lib.rs`
      ~1278-1464: chunks of `2 * l3`, `l3 / 2`, strides zeroed past
      `l3 / 2`); on Windows the runners' EPYC 7763 reports no L3 to CPUID
      (the `gemm` crate read 32 KiB / 512 KiB / 0 there, task 263). 1.0.0
      and 1.0.1 built the kernel for Windows, so they have it on such a
      machine (that test arrived with task 256 (b), after 1.0.1). *Decided
      2026-10-15* (the user: "Your reco windows"): task 264 reverted --
      faer's `std` for `cfg(all(target_arch = "x86_64", not(windows)))`
      again, `product_memory` on Windows again, CI's Windows Rust tests at
      the default parallelism again; the CHANGELOG's Fixed entries give the
      wrong products and the commit, its Changed entry Windows' last bits;
      PERFORMANCE §43 the finding. Windows' speed is task 263's numbers
      (wide `ewridge` 12-63% slower than 1.0.1 on the runners). An upstream
      report (the kernel's cache detection under a hypervisor, and its
      products with no L3) is the user's to file.

<a id="task-267"></a>
- [x] 267. **The kernel on Windows after all, the wrong products a known
      issue** -- the user, 2026-10-15, after task 266 merged: "Something
      is wrong windows should not be bounded memory it should have the
      kernel", "When the windows tests run they should reduce the threads
      to stay within the memory bounds for ci"; told that the failure was
      a wrong answer, not memory, and that two test threads had not
      changed it (run 38731522819 ran at two), with A (bounded, correct)
      against B (kernel, the wrong products a known issue): "B". Task 266
      reverted to task 264's build: faer's `std` for every x86-64 target,
      CI's Windows Rust tests at `--test-threads=2`, `product_memory` off
      x86-64. `trisolve_tests::the_solve_is_the_lu_solve_within_its_
      forward_error_up_to_1500` held off Windows by `#[cfg(not(windows))]`,
      a platform guard with its reason (the smaller-order test still runs
      there and passed on every runner). CHANGELOG: two Known issues (the
      wrong products where no L3 is reported, the commit); PERFORMANCE §43
      the finding. Upstream reports (the kernel's cache detection and its
      products with no L3) are the user's to file.

<a id="task-268"></a>
- [x] 268. **`private-gemm-x86` 0.1.22: the kernel's wrong products fixed
      upstream** -- the user, 2026-10-15: "Is there a new version of [faer]
      that would fix it", then "Yes" to updating and verifying. faer 0.24.4
      is the latest; its kernel `private-gemm-x86` 0.1.20 (our lock) is
      yanked, and 0.1.22 (2026-10-05) changes exactly the block sizing:
      `lib.rs` ~1438 now clamps a chunk to the L3 before rounding it up to a
      multiple of the chunk nested in it (0.1.20 rounded first, so a clamp
      to `l3 / 2` could break the tiling), and three triangle tests
      (`tril && pos.row + mr > pos.col`) compare against `row_chunk`. The
      user's Linux session (artifact claude.ai/artifact/WYiVhsfUJZhRWBYzbNLV83)
      proved the cause on 0.1.20: the kernel's used L3 forced to 2 MiB fails
      the order-1,500 solve test exactly as the Windows EPYC 7763 runners
      do (their CPUID reports no L3, floored to 4 x L2 = 2 MiB), 4/8/33 MiB
      pass. With 0.1.22 (CI 38804802757) the product and solve tests pass on
      ten runners -- four Windows EPYC 7763 (0.1.20: four of four failed),
      Windows Zen 4/5 and Xeon 8573C, Linux Xeon 8573C and Zen 4. Built:
      the lock bumps only `private-gemm-x86` (version, checksum, and
      online-core's dependency list), `online-core` names it on x86-64 for
      its minimum (`>= 0.1.22`), task 267's Windows test guard removed;
      CHANGELOG's known issue becomes a Fixed entry (1.0.x's wrong
      products); PERFORMANCE §43. The width probe (the user's Linux
      session, artifact claude.ai/artifact/JQv5vS19ZiDKFm2uNC3sBz, the
      kernel's used L3 forced): 0.1.20 failed 29 (n, t) cases -- 2 MiB at
      n = 700 and 1,350-1,540, 4 MiB at 1,800-1,920 and 2,600-2,650, 8 MiB
      at 3,700 and 3,900 (errors up to 1.3e-2 against bounds of 1e-6 to
      4e-5), none at the detected 33 MiB up to 4,097 -- in bands, not at
      `2 x L3 / 4096`; 0.1.22 passed all 29. So 1.0.0 and 1.0.1 could be
      wrong on ordinary x86 machines (an 8 MiB L3 at about 3,700-3,900
      features), not only on VMs that hide their L3; the CHANGELOG's Fixed
      entry says so and tells 1.0.x users on x86 to upgrade.

**1.1.0** (the user, 2026-10-13): tasks 258's fix, 256 and 257; 260 (2026-10-14); 264, the kernel on every x86 build; 266 reverted it (wrong products on Windows), 267 restored it at the user's choice; 268, the kernel's fix (0.1.22); 265 (2026-10-15); 263's options for later work.
      *Released 2026-10-15* (the user: "Do the release", then "Release
      1.1" for the dispatch): run 38832066738 on `7fa8ee0`, every job green
      (CI on all three OSes, six wheels, the sdist, the CLI binaries, the
      state hand-off, the floor / newest / next-major Polars and NumPy legs,
      the budgets), PyPI approved by the user, `v1.1.0` and the GitHub
      release made by the workflow. Verified from a clean venv: 1.1.0
      native 1.1.0 schema 62, a fit recovers its coefficients. Before the
      dispatch: compare_release identical (297 fields), benchmark.py 18 of
      38 faster by 1-8% and `corrchange`'s monitor 3.1% slower (declared),
      wide_benchmark.py level, the floor Polars 1.37.0 suite green, kernel
      0.1.22 the speed of 0.1.20 on three CPUs (same-runner, CI 38821326415).
      Recorded after: budgets.toml's release counts from the run's artifacts
      (`--release 1.1.0`; the floor-Python peaks fell, e.g. 155 -> 123 MiB
      and 452 -> 419 MiB on Linux), RELEASES gains 1.1.0 (its states load
      and go on as this build's own).

<a id="task-256"></a>
- [x] 256. **Per-job feature lists in `po.gram.solve_subsets` and
      `lars_paths`** -- requested 2026-10-13. Today one call takes a single
      `features` list for every subset, target and draw, so a workload
      whose jobs each use their own columns (a screen per subset, per
      target and per draw, as complementary-pairs stability selection
      runs) cannot use it: it falls back to one `po.gram.lars_path` call
      per job from Python, after a `merge(parts, within=True)` and a
      `subset` per subset, also from Python. **Requirement:** one call
      carries the per-job feature lists, as either `features` nested
      `[subset][draw][target] -> columns` (`None` at any level meaning
      every feature, as today) or a flat `jobs=[{"subset", "target",
      "draw", "features"}, ...]` returning one result per job in order.
      Everything else as today: the Grams read once, each subset merged
      once (within, the draws' signs on the cross-moments), every job's
      path on the pool. **Check:** per job, the same `penalties`, `coef`,
      `active` and `stop`, to the bit, as `lars_path` on
      `subset(merge(parts, within=True), ["intercept", *features])` with the
      draw's signed cross-moments. **Potential gain** (measured by the
      requester on 0.13.0; re-measure at HEAD first): a workload of 11,000
      such paths per run spends most of its remaining time in the per-call
      crossing from Python into Rust and the Gram's conversion on each
      call -- per subset 4 ms at 500 columns (150 screened) and 49 ms at
      1,500 (400) -- which one call removes. An API addition (a minor
      release); the field and argument names follow the existing
      `solve_subsets`.
      *Built 2026-10-13* (branch `task-256-per-job-features`). **Design:
      the flat `jobs=`**, a list of mappings or a polars DataFrame of them
      (rows as `to_dicts()` gives them, Polars' own form for records),
      keys `subset`, `target`, `draw`, `features`, `penalty_weights`; one
      result per job in order. Chosen over the nested `features` because
      (1) it adds a keyword and changes no existing one's type: a nested
      `features` would have to be told from today's list of names by its
      depth, and `None` at three levels; (2) the nested form's order,
      `[subset][draw][target]`, is not the return's `[draw][subset][target]`,
      and it needs a draw level even without signs; (3) a flat table runs
      only the jobs named, any subset of the grid, and is what a screen
      table joined to the subsets and draws already is. `targets=` is
      refused beside it; `features=` is the default for a job without its
      own, and so is `path`'s `penalty_weights`, held to each job's length.
      `lars_paths` takes no `jobs`: its per-Gram form is
      `solve_subsets([[0], [1], ...], jobs=...)`, a merge of one Gram being
      the Gram, and a test holds that to `lars_path` to the bit.
      **Native:** `online_polars::gramfit::fit_jobs` (column sets deduped,
      a job = subset, target, draw, set); per subset one merge on the
      intercept and the union of its jobs' columns in the Grams' order, one
      `within_cross` per draw used, one design per (draw, set), the targets
      on the pool; `fit_subsets` is now a wrapper over it. The merge is
      entry for entry, so the union of columns moves no bit. **A ridge's
      targets are now solved one column at a time** on the shared factor
      (`ridge.rs`): faer's blocked multi-column solve moved a target's last
      bit with its neighbours (found by the Rust job test), so a ridge job
      depended on which other jobs shared its columns. **Also fixed:**
      `_col_index` scanned the column list per name (`list.index`), 3.2 ms
      for a `subset` of 400 of 1,500, 0.56 ms by a table. **Measured**
      (M4 Pro, load 7-8; 38 synthetic Grams, 100 subsets of 19, 11 draws,
      10 targets, `max_active=10`, `within=True`; scratchpad `bench.py`):
      per subset, the loop of `merge` + signed cross-moments + `subset` +
      `lars_path` per job took 65 ms at 500 columns (150 a job) and 409 ms
      at 1,500 (400) at HEAD, 24 and 115 ms with the `_col_index` fix, and
      the one call 3.3 and 11.2 ms (of which the native call 8.8 ms and the
      jobs' resolution in Python 1.7 ms at 1,500). The requester's premise
      (4 and 49 ms of crossing per subset on 0.13.0) holds in kind: at HEAD
      the loop's time was `subset`'s name lookup and the Python merge, the
      paths themselves 0.1 ms a job. **Tests:** `tests/test_gram_jobs.py`
      (every job to the bit against `lars_path(subset(merge(signed parts,
      within=True), ["intercept", *features]))`, union, `None` = the call's
      features, penalty weights, subsets of one = `lars_path`, the jobs of
      every fit = the nested call, a DataFrame = its rows, threads 1 = 4,
      ridge jobs = the call per job, twenty refusals by name);
      `crates/online-polars/tests/gram_jobs.rs` (the same against
      `merge`/`merge_within`/`within_cross` + `lars_paths`/`ridge_fits` on
      the job's columns alone, `fit_subsets` = its jobs, threads, every
      index refusal of `check_jobs`).
      *256 (b), 2026-10-14* (branch `task-256b-ridge-solve`): the
      one-column-at-a-time solve cost 6-12% at 50 targets (0.76 -> 0.82 ms
      at 200 x 10, 13.8 -> 15.5 at 500 x 50, 190 -> 202 at 1,500 x 50):
      each target read the factor twice alone and paid a scratch and a
      dispatch. **Kernel** (`gramfit/trisolve.rs`): forward then back
      substitution on the factor where it lies (column-major), every
      column's arithmetic the textbook recurrence -- `s = s - L_ij x_j`, `j`
      ascending forward and descending back, one rounded product and one
      rounded difference a term (Rust never contracts to an FMA), then
      `/ L_ii` -- so a column's bits cannot depend on its neighbours, the
      tile shapes or the platform's tile choice. Panels of 16 rows: every
      term from outside the panel in register tiles (4 rows x 8 targets, 8
      x 4 and 16 x 1 on aarch64; 4 x 4 and 16 x 1 elsewhere), the targets of
      a tile sharing each load of `L`, then the panel's own triangle row by
      row. No packing copy (an `n²` transpose would be 800 MB at 10,000).
      **The statistics were the larger cost:** `sample` at 1,500 x 50 put
      the per-target `b' C b` (a `k²` chain of dependent adds) and
      `quadratic`'s `m' C⁻¹ m` (`k²/2`, a strided row of `L⁻¹`) at about
      half the call, the solve at 4%. `gramfit/colstats.rs` runs both for
      every target at once in the same tiles, each sum the left fold from
      `-0.0` that `Iterator::sum` is over the same terms in the same order:
      the bits of the loop they replace. **Tests:**
      `gramfit/trisolve_tests.rs` -- a column alone = beside others, random
      subsets in random orders, n = 1-65 x T = 1-64 across the panel and
      tile edges, to the bit; each column = the scalar recurrence to the
      bit; each substitution's componentwise backward error within `γ_n`
      by an exactly rounded residual (`oracle::dot`), well and
      ill-conditioned; the solve against faer's partial-pivot LU
      (`oracle::solve_columns`, new) within Cholesky's forward bound
      `2κ n γ_{3n+1}`, κ as large as that bound allows (4e12 at n = 4, 2e9 at
      200, 3e7 at 1,500) at n = 1-200, and 511 and
      1,500 in the extended tier; and a target's whole `RidgeFit` the same
      bits alone or in any subset and order of 13 or 21 targets, with and
      without intercept and standardize (fails on faer's blocked solve:
      checked). `gramfit/colstats_tests.rs` -- both forms = the old per-target
      expressions to the bit, zeros' signs included. **Bits:** single-target
      fits move too (faer's one-column solve is not the recurrence: 3 of 3
      entries differ at n = 3), within `6e-15` of `coef`'s largest entry and
      `6e-14` of `se`'s against 1.0.1; `release_probe.py`'s 297 fields and
      31 states byte-identical to main's; `solve.rs` untouched. FMA was
      left out on purpose: it would fix a column's bits only per platform,
      and x86-64's baseline has none. **Speed** (M4 Pro, load 6-20; one
      subset of a Gram on 2,000 rows, ridge 1; 1.0.1 / main / branch, ms,
      `threads=1`, the pool the same within noise):

    | k | T = 1 | T = 10 | T = 50 |
    |---|---|---|---|
    | 200 | 0.53 / 0.54 / 0.52 | 0.78 / 0.85 / 0.62 | 1.88 / 2.16 / 1.00 |
    | 500 | 4.43 / 4.42 / 4.39 | 6.10 / 6.41 / 4.83 | 13.6 / 15.4 / 6.76 |
    | 1,500 | 95.1 / 93.0 / 89.0 | 112 / 114 / 93.3 | 195 / 209 / 109 |
    | 3,000 | 253 / 252 / 247 | 300 / 318 / 258 | 513 / 609 / 309 |
    | 5,000 | 1,039 / 1,017 / 1,013 | 1,164 / 1,193 / 1,047 | 1,770 / 2,137 / 1,207 |
    | 10,000 | 6,907 / 6,894 / 6,846 | 7,417 / 7,588 / 6,991 | 9,607 / 10,658 / 7,601 |

      Best of 15, 9, 5, 3, 2 and 1 calls at 200 to 10,000 columns, in 3, 3,
      3, 3, 2 and 2 rounds of fresh processes, the builds in rotated order;
      load 6-20; peak RSS the same for all three (5.2 GB at 10,000). One
      target ties 1.0.1 (a rerun at 200 and 500: 40 and 20 calls x 5 rounds,
      medians 0.487 vs 0.485 ms and 4.23 vs 4.33 ms at `threads=1`, 0.510 vs
      0.505 and 4.20 vs 4.29 on the pool): its call is the merge, two
      Cholesky factorizations and a triangular inverse, and the kernel's one-column solve costs 8 us
      against faer's 7 at 200 (a Rust micro-bench, `tribench` in the
      scratchpad), 2.9 ms against 3.4 at 3,000. x86-64 is not measured
      here: its tiles are 4 x 4 and 16 x 1 under SSE2 (no AVX at the
      baseline target), where faer's products dispatch to AVX2 at run time,
      so `examples/ridge_solve_bench.rs` measures it: `ridge_fits` against
      a copy of 1.0.1's (faer's blocked solve, each target's forms summed
      alone) in one process, rounds of the shared part (design, two
      factorizations, triangular inverse), 1.0.1's call and today's in turn,
      after asserting a target's fit is the same bits alone, among 10 and
      among 50 (and printing that 1.0.1's was not). Its Gram is positive
      definite with a Kish size past `k`, so the standard errors run at every
      `k`: the Python table above fits 2,000 rows, so from 3,000 columns its
      `sigma2` is NaN (no degrees of freedom) and it times the solve, `b' C b`
      and one factorization only. `benchmark.yml` runs it at 500-5,000 beside
      the wide products, for the x86 numbers. Local run (M4 Pro, load 4-5),
      ms, 1.0.1 / now (ratio), and the solve and statistics alone (the
      call less the shared part):

    | k | T | call | solve + statistics |
    |---|---|---|---|
    | 500 | 1 / 10 / 50 | 4.06 / 4.05, 5.84 / 4.50, 13.3 / 6.15 (1.00, 0.77, 0.46) | 0.96, 0.33, 0.25 |
    | 1,500 | 1 / 10 / 50 | 80.2 / 79.1, 98.7 / 83.6, 177 / 98.8 (0.99, 0.85, 0.56) | 0.70, 0.32, 0.22 |
    | 3,000 | 1 / 10 / 50 | 576 / 574, 663 / 609, 1,028 / 659 (1.00, 0.92, 0.64) | 0.85, 0.46, 0.21 |
    | 5,000 | 1 / 10 / 50 | 2,571 / 2,560, 2,904 / 2,627, 4,246 / 2,845 (1.00, 0.90, 0.67) | 0.70, 0.26, 0.18 |
    | 10,000 | 1 / 10 / 50 | 20,377 / 20,020, 21,049 / 19,961, 26,781 / 20,914 (0.98, 0.95, 0.78) | one round: noise |

      At 10,000 one round took 265 s and 5.0 GB, so CI stops at 5,000 (a
      16 GB, 4-cpu runner; about 2 minutes here at 500-5,000).

<a id="task-257"></a>
- [x] 257. **The targets inside the blocked Gram product** -- requested
      2026-10-13. Measured by the requester on 0.13.0 (M4 Pro, load 4-9;
      re-measure at HEAD first, since `gram_threads="auto"` and task 250
      changed the bank's costs): a bank of `ewridge` Gram accumulators
      (groups of 200 features, 10 targets, `target_gaps="pairwise"`,
      `gram_block_rows=256`, never solved) costs 19-20 ns of CPU per
      feature-row, of which the blocked Gram product is about 5.7 ns. With
      one target it is 11.5 ns: each target adds about 0.7 ns per
      feature-row, as if its cross-moments were updated row by row beside
      the blocked product (`pairwise` and `own_rows` cost the same). About
      6 ns per feature-row more has no identified source (perhaps the
      per-column summary statistics). **Requirement:** fold the targets'
      cross-moments into the blocked product -- the block of rows as
      `[X | Y]`, one product giving `X'X`, `X'Y` (and `Y'Y` where wanted);
      under `target_gaps="pairwise"` a target missing on a row must not
      enter as 0, so a mask per target (`X' diag(m_t) y_t`) or the block
      split at the target's gaps keeps each target's own rows. And find
      where the other ~6 ns goes. **Check:** `cross_moments`,
      `cross_centered`, `means_by_target` and `target_weights` the same bits
      as today's at every `gram_block_rows` and `gram_threads`; the cost
      per feature-row against the target count falling toward the
      product's; the instruction-count budgets for the Gram path.
      **Potential gain:** at 10,000 features over 9.5 million rows, the
      targets alone are about 660 CPU-seconds of a pass, and the unexplained
      6 ns about 570 more.
      *Re-measured 2026-10-13 at `e157e46`* (M4 Pro, load 2.5-6;
      `examples/gram_targets_bench.rs`, scratchpad `bank_bench.py`). One
      `EwRidge` of 200 features, 256-row block, `pairwise`, never solved,
      ns per feature-row on one thread: 6.6-6.9 at 1 target, 7.0-7.3 at 2,
      8.0-8.4 at 5, 9.7-10.2 at 10 -- 0.35 ns a target, with gaps or
      without, `own_rows` the same; `gram_threads` 14 the same (a 201-wide
      merge is one piece). `sample` of the 10-target case: the product
      (gemm kernels, packing, the merge) 50% of the samples, `Acc::learn`
      37% (each target's `Cross::learn` and the all-row mean's
      `Cross::advance`, `k` compensated steps each), the flush's centring
      5%. A bank of 5 such specs (1,000 float32 columns, 20,000 rows, 10%
      nulls a target): CPU ns per feature-row 13.3-16 at 1 target and
      16.8-18.8 at 10 on the default pool of 14, 9.4 and 12.9 on a pool of
      one -- the requester's numbers. **The unexplained part is the bank's,
      not the model's:** on one thread the bank adds ~3 ns over the model
      (the chunk's transpose into rows 0.9, the per-row loop around `step`
      ~2); on the pool, 2-5 more of CPU without wall time: the transpose
      (`FeatureRows::from_columns`, 27% of the bank's samples on the pool)
      runs on every thread, memory-bound (a 160 MB block per chunk written
      and read again), and the pool's idle threads spin. Run on each spec's
      own thread (no nested `par`) it took 2-2.5 ns less CPU at the same
      wall (G = 5: 15.1-16.8 against 16.6-19.2; G = 1: 15.1-17.1 against
      17.0-24.4, wall 12.9 either way).
      **The fold needs a state field.** Each target's cross-moments, its
      own feature means and the all-row mean are up to date per row in
      the state (`gaps::Cross`), so a save mid-block resumes to the bit.
      Deferred to the block's flush, the targets' values on the held rows
      must be in the state, as the Gram's held rows already are: there is
      no other copy of them. Proposed: a trailing `#[serde(default)]`
      field beside the Gram's block, the held targets (`rows x T`, NaN where
      absent) and per target its weight and mean at the block's start; a
      1.0 state mid-block loads with none, its targets already current, so
      the held targets are a suffix of the Gram's held rows. Upper bound of
      the gain, measured with the per-row `Cross::learn` skipped and a
      `k x T` (`2T` with the gap masks) product in the flush: 10 targets
      9.9-10.2 -> 7.2-7.4 ns (7.6-8.1 with masks), 5 targets 8.3-8.5 ->
      6.9-7.2; a target then costs ~0.08 ns, where it costs 0.35.
      *Decided 2026-10-13* (the user, on both recommendations): the field
      and the fold, and (a) the per-spec transpose; (b) later, its own step.
      *Built 2026-10-13* (branch `task-257-targets-in-product`). **The
      fold:** `EwCov::flush_riding` hands a `Rider` the block between its
      centring and its scaling (`ewcov::Block`: deviations `a = x − m_B`,
      weights `u`, `m_B`, `δ` and `Σ uᵢaᵢ`), and is told of each row it
      holds; `gaps::Grams::update` gives each Gram a `Fold` of its readers
      (on a row where targets part, the absent ones' copy folds them).
      `gaps::Held` (`acc.held`, schema 62, trailing, written only while a
      target holds values) keeps per target its values on its Gram's last
      held rows (NaN where it learns nothing), its history's weight decayed
      through them, its weight after the last and its mean pair before the
      first; a decay that forgets its history ([`decay_into`]) restarts
      them. `gaps::fold_block` mirrors `EwCov::flush` for `(z, y_j)` over the
      target's rows: `S' = Σ (uᵢeᵢ)aᵢ` summed row by row (four rows a
      register pass, the same additions in order; a zero coefficient adds
      nothing since a sum from `+0` is never `−0`), `ε` the Gram's `δ` where
      the target has every row of weight, else `(Σuᵢaᵢ − M)/U` with `M` over
      the rows it misses; bands of 256+ columns on `gram_threads`, each
      entry the same operations. Not faer's product: its kernel's order for
      a column could follow the other columns beside it (task 256's lesson),
      and the sum by rows is the same on every platform. The weight, `ȳ_j`
      (`Cross::learn_mean`), the target moments, `nj` and the all-row mean
      stay per row (the all-row mean, ~0.35 ns, is a possible follow-up:
      under `own_rows` no Gram need hold every row). Readers mid-block take
      `Acc::flushed` (`gram_parts`, `cross_moments`); a solve and a blend
      `Acc::flush`. Unblocked fits run the old code (`Acc::learn_per_row`).
      **Bits:** blocked `ewridge`'s `c_j` and `m_j` (so `cross_centered`,
      `cross_moments`, `means_by_target` under `pairwise`, coefficients and
      predictions): 3 specs x 40 features x 4 targets x 3,000 rows against
      `e157e46`, outputs within 2e-12 relative, `cross_moments` 2e-16 and
      `cross_centered` 6e-15 of their largest entry; `target_weights`,
      `ȳ`, the target moments, unblocked fits, `release_probe.py`'s 297
      fields (no blocked spec there) and `benchmark.py`'s rows the same
      bits. Under `own_rows` a target's `m_j` is its Gram's mean to the bit
      (the per-row path's promise, which blocking had broken). A share
      `U/W'` rounds, so a block's merge leaves a mean a rounding of its
      level short, as the Gram's own always has: an intercept's centred
      cross-moment reads ~1e-19 where the per-row path wrote 0.
      **Tests:** `gaps/fold_tests.rs`: every entry of `c_j` and `m_j`
      within `γ_{8R+32}` of the magnitudes (plus the level's resolution) of
      `oracle::dot`/`dot3`'s exactly rounded definition, blocks 0-257, 1/3/8
      threads, both gaps, features at 0 and 1e3 with a constant column,
      zero weights, decays of 1/2 and 0, a target-only reset; a dropped row
      and a 1e-12 scale fail it (worst case 0.42 of the bound); thread
      counts and neighbours to the bit at 600 columns; `own_rows` means = the
      Gram's; a 61-style state (per-row targets, held Gram rows) going on
      within the bound; mid-block round trips at six offsets (named and
      compact) to the bit, the field absent between blocks. `ewridge/tests`'
      all-row-mean test restated for blocking. `tests/test_gram_block.py`:
      three targets with gaps, both `target_gaps`, per-row fit to 1e-9,
      chunks of 1/7/133 and saves at six rows to the bit with `gram()`.
      **Fixtures:** schema 62, `MIN_SCHEMA_VERSION` and the bank's minimum
      stay 61; 61's sets kept (`online-core` every case, `online-polars` the
      bank files and `refresh_time`), `previous![61]`, each loading, going
      on and re-saving as 62's bytes (no kept case is blocked, so the
      conversion is exact: every unblocked file's bytes moved only in the
      schema byte); new case `ewridge_blocked`, saved mid-block, its
      `acc.held.y` and `w_end` held non-empty. **Transpose (a):**
      `transposes_in_parallel`: a chunk's transpose spreads over the pool
      only when one spec has features. **Measured** (M4 Pro, load 2-4,
      three rounds each build in turn; PERFORMANCE §42): one fit, ns per
      feature-row, 1/2/5/10 targets 7.1-7.4/7.4-7.7/8.5-8.7/10.2-10.6 ->
      6.9-7.1/7.1-7.3/7.4-7.6/8.1-8.3 (gaps at 10: 10.0-10.3 -> 8.5-8.8);
      the bank of 5 x 200 at 10 targets CPU 17.0-20.4 -> 13.4-13.7, wall
      3.3-3.8 -> 2.9-3.2, at 1 target 13.9-17.5 -> 11.1-12.5; one spec of
      1,000 features unchanged at 1 target and 29.3 -> 26.5-27.6 ns of wall
      at 10 (8.4-8.9 against 11.6-11.8 on `"auto"`); `benchmark.py` six
      rounds each, median ratio 0.999, every row within scatter. New
      instruction case `ewridge_gram_t10` (50 features, 10 targets, block
      64), its budget from CI's first run.
      **Per-row cost** (2026-10-13, branch `task-257-per-row-cost`). CI's
      counts on `b440a24` put a fixed ~44-52 instructions a row on the
      *unblocked* path (`ewridge_k5` 2,961,513 -> 3,013,824, +1.77%;
      `bank_ewridge_k5` +0.9%; `ewridge_k20` +0.54%, the same amount at
      k = 5 and 20): inside the tolerance, but a slip. Cause: the rider was
      an `Option<&mut dyn Rider>` threaded through every row, and with two
      callers (`learn_per_row`, `learn_held`) `Grams::update` and
      `TargetMoments::learn` left `Acc::learn` for calls of their own -- a
      call per Gram and per target each row, with their spills. Fix: the
      rider a type parameter (`ewcov::NoRider`, whose `RIDES = false`
      picks the merge's one-loop pass at compile time; `gaps::Ride` with
      `NoRide`, `Grams::update` over `update_with`), so `EwCov::update`,
      `skip` and `flush` monomorphise to `733e3f4`'s code (its `update` and
      `update_riding::<NoRider>` differ in register allocation alone), and
      `TargetMoments::learn` `#[inline(always)]`. What is left on the
      unblocked row is `Acc::learn`'s `blocked()` test, a few instructions.
      Measured natively (M4 Pro, load 2.5-5) as instructions retired
      (`/usr/bin/time -l`, 1M rows of the bench cases' workloads through
      `EwRidge::step` less a run of none, median of 7): k = 5 3,990 /
      4,064 / 3,972 a row at `733e3f4` / `main` / this; k = 20 8,087 /
      8,167 / 8,074; the blocked `ewridge_gram_t10` workload 14,135 ->
      13,987 (-1.0%) against `main`. Six interleaved timing rounds, ns a
      row, k = 5 134.8 / 138.8 / 133.1, k = 20 270.5 / 271.1 / 270.0.
      **Bits:** `release_probe.py`'s outputs (31 specs) and `--states`
      files byte-identical to `main`'s wheel, and so are blocked fits
      (both `target_gaps`, blocks 5/64/256, 1 and 4 Gram threads, states
      saved mid-block at every 333-row chunk); `online-core`'s every test
      with `--include-ignored`, the fold tests and the frozen fixtures
      pass. CI's counts on the push are the check.
      **Next (b)**, after CI passes on this: transpose tile by tile inside
      each stream, cache-resident, so a chunk's row block (160 MB at 20,000
      rows of 1,000 features) is never written and read back; the
      remaining ~3 ns of a bank's own per feature-row (0.9 the transpose on
      one thread, ~2 the loop around `step`) is its target.
      *(b) built 2026-10-13* (branch `task-257b-tile-transpose`, the user's
      approval). Re-measured at `63c2495` (M4 Pro, load 2.5-4.5, scratchpad
      `t257b/bank_bench.py`, one process per pool size): 5 x 200 float32
      features, 20k rows, 10% nulls a target, `fit`, CPU ns per feature-row
      9.5-10.1 at 1 target and 11.7-12.3 at 10 on a pool of one, 11.4-11.7
      and 13.9-14.6 on the default (wall 2.8-3.2, 3.2-3.4); the lone spec of
      1,000 features 25.0-25.8 / 26.3-27.2 on one thread, 41.5-46 CPU at
      8.0-9.4 wall on the default. `sample` of the 10-target bank:
      `FeatureRows::from_columns` 7.7-7.9% of the busy samples on one thread
      (~0.9 ns), 9.4% on the pool; `process_chunk`'s own loop 6.6%,
      `schedule` 1.5-3.1%. **Built:** `rows::FeatureCols` (the chunk's
      feature columns, each its `Float64Array`, shared where it has no
      nulls, the filled copy otherwise, and the layout) replaces the
      chunk's row block in `SpecColumns`; `process` keeps one
      `FeatureRows` per task, filled by `FeatureCols::rows_into` for each
      run's positions just before `process_chunk` (same tiled loop, the
      layout gather folded in, positions read through `row(i)` with the
      window's start subtracted); `ChunkOut::run_rows` counts `k` f64s a
      row in its 2 MiB, so a run's rows stay in cache (976 at the old
      1,040 for `ew_cov` 20; ~1,000 at 200 features, 240 at 1,000).
      `last_accepted` reads `FeatureCols::usable_at` (the same fold of
      `usable` over the row, from the columns); scoring and the window
      pre-pass take the task's or segment's rows at once
      (`FeatureCols::rows`, spread over the pool for a lone spec), the
      pre-pass only past its early returns -- built first before them, it
      copied every segment of every chunk for nothing and the bank ran
      0.8 ns slower. Targets, weights, clock and groups unchanged.
      **Bits:** `release_probe.py` outputs and `--states` byte-identical;
      its workload on a 203-feature stream (float32/float64, nulls, NaN,
      +-inf, 1e150; interleaved groups, embargo, window budget, blocked
      Gram, ungrouped): 4,500 rows fit in 1, 6 and 8 chunks, 1,500 more
      scored, 316 files byte for byte; `online-polars`' tests (run_rows
      pins moved with the width), the gate. **Measured** (PERFORMANCE §42,
      *the copy, a run at a time*): CPU 11.1-11.6 (pool 1) and 12.8-13.4
      (default) at 10 targets, 9.0-9.4 and 10.4-11.0 at 1; wall 2.9-3.2;
      the lone 1,000 same wall, CPU 1-7 ns less on the pool; the copy 5.2%
      of the samples on one thread, 5.9% on the pool, `schedule` 1.0%.
      Peak RSS: the five specs' `fit` 395 -> 231 MiB, a 1,000-feature
      `sgd` 165 -> 14, `rss.ewridge_10_targets` 145-152 -> 122,
      `rss.ew_cov_wide` 444-446 -> 418, streaming unchanged.
      `benchmark.py` six rounds, median ratio 1.002, `sgd`/`pa` +4-5%.
      Native instruction counts of the bank cases (proc_pid_rusage, median
      of 7): `bank_seqtest` 2,204k -> 2,213k (+0.4%, within its run-to-run
      scatter of 2,199k-2,357k), `bank_ewridge_k5` 7,390k -> 7,373k,
      `bank_ew_cov_wide_t1` 143.49M -> 142.94M. **Next:** the float32 ->
      float64 cast when the chunk is read (`crate::arrow`) is now the one
      whole-chunk copy of the features (160 MB for the five specs); the
      run's copy could read the float32 column and cast in the tile.
      *(c) built 2026-10-14* (branch `task-257c-cast-in-tile`). **Built:**
      `Want::Feature` in the adapter: a column only features read is kept
      in its own type, `ArrowCol::Num(NumArray)` (Float32, Int8-64,
      UInt8-64; `NumArray::of` rechunks only a kept dtype), and is read by
      `ArrowChunk::feature`; a column another role reads as a number (the
      `form_of` "a number" set, e.g. a weight that is also a feature) is
      cast once to Float64 and shared, and Boolean, Decimal, Int128 and
      Float16 still cast. `rows::FeatureCol` holds the array (any of the ten
      types) and its validity where it marks a null; `f64_filled` is gone.
      The copy (`FeatureRows::fill`), per 128-row tile: up to `WIDE` = 448
      features, `by_groups` -- eight columns at a time, each read for the
      tile's rows into an 8 KiB scratch (one type, one validity branch, `as
      f64`, `NULL` for a null), then a fixed-length copy of a cache line
      per row; past 448, runs of plain columns (no nulls, all f64 or all
      f32) `by_rows`, the old row-by-row loop over `COL_TILE` = 64, the
      rest by groups. Measured (scratch `tile_bench`): the scratch's first build 0.46-0.83
      ns a value at 20-200 features against 0.55-1.0 row by row, but
      1.2-1.3 at 1,000 and 1.8-2.0 at 1,024 (rows 8 KiB apart, one cache
      set); crossover 400-500. Two slips found by measuring: a runtime-length
      `copy_from_slice` per row and group was a `memcpy` call (a third of
      the copy at 1,000), and `NumArray::of` rechunked a multi-chunk
      Float64 column before declining it (+1.6 MiB on `rss.ew_cov_wide`).
      **Bits:** `release_probe.py` outputs and 31 `--states` byte-identical;
      the 203-feature probe plus five integer feature types (Int8, Int32
      with nulls, Int64 past 2^62, UInt16 with nulls, UInt64 past 2^63) and
      a Float32 weight that is also a feature (shared path), 1/6/8 chunks
      then scored, 325 files; the same at 608 features with 150-column runs
      of plain f32 and f64 (the row path), 46 files; an `audit` of
      float32/int16/float64 columns with NaN, +-inf and nulls (counts kept
      apart); float fields equal across chunkings. Tests: `rows` each kept
      type, NaN payloads (signalling, negative), +-inf, extremes, subnormal,
      integers past 2^53, nulls, against polars' cast and fill, by `at`, the
      run copy, a layout and the pool, at 36 and 538 features;
      `arrow_chunk`: Float32/Int32 features and a shared Float32 weight
      learn and save what the frame cast to Float64 does, at 600 and 97-row
      chunks, and a target lookup on a kept column names its type.
      A third slip: the scratch zeroed per tile and a short last group
      copied in a loop made 4 features 0.97 ns a value against 0.33; now
      the scratch is made once a thread (`for_each_init`) and each group
      copied out by a fixed-width copy of its width (`out_of_scratch::<N>`).
      **Measured** (PERFORMANCE §42, *the cast, in the copy*; M4 Pro, load
      4-6): peak RSS the five specs' `fit` 230 -> 67 MiB, a 1,000-feature
      float32 `sgd` 179 -> 14, the `rss.*` cases unchanged; the five specs'
      wall on the default pool 2.44-2.54 -> 2.08-2.14 (T = 1), 2.89-3.01 ->
      2.55-2.66 (T = 10), CPU 0.1-0.4 ns less at pool 1 and default; the
      lone 1,000 the same within scatter (the first build, scratch at every
      width with the `memcpy`, was +1-1.5% on one thread and +4-5% wall on
      the pool); the copy alone 0.31-0.67 ns a value at 4-200 features
      against 0.32-0.95 row by row, 0.99-1.03 at 1,000-1,024 against
      1.01-1.04. `benchmark.py` eight ABBA rounds each, median ratio 1.001,
      `sgd`/`pa` +1.6%. Native instruction counts (median of 7):
      `bank_seqtest` 2,481k -> 2,485k, `bank_ewridge_k5` 7,665k -> 7,631k,
      `bank_ew_cov_wide_t1` 143.24M -> 142.36M.
      *Survivors* (2026-10-14, branch `task-257-survivors`): CI's
      changed-lines pass on `b440a24` (run 38490073010) left 42 survivors
      and a timeout in `gaps.rs`. Why they lived: the fold tests read `c_j`
      and `m_j` within `γ_{8R+32}` at levels where a block mean `ȳ_B` taken
      about the wrong centre, or its residue `δy` dropped, costs only
      roundings (the merge is exact in any centre); nothing compared the
      per-row numbers of a blocked fit with the unblocked fit's (`nj`, the
      target moments' `a`, `ȳ_j` on a zero-weight row); nothing read the
      held state's values, its equality, its empty shape or the borrow of
      `Acc::flushed`; and no case reached a row its block decays to `u = 0`
      or a damaged held weight. Caught now by `fold_tests`' far target
      (`1e8`, an outlier first row of `2^-40`: the fold at under 0.04 of the
      bound, each mistake 45 to 2.8e5 times it), per-row bits against the
      unblocked fit, a zero-weight row's `ȳ_j`, a row decayed to nothing
      and damaged blocks, and `held_tests`. Equivalent (recorded): `gappy`
      turned true where it was false (4; the gappy `ε` is the Gram's `δ` to
      the bit there) and `accumulate`'s band width (7; a column's sum is the
      same operations in any band). Tolerated, `loop`: `i += 4` -> `*=`.

<a id="task-255"></a>
- [x] 255. **The memory findings of the 1.0 budgets** -- measured
      2026-10-13 on this machine (py-polars 2.0.0, `a2f794a`):
      (a) **A wide eager `fit_predict` peaks at twice its output.** `ew_cov`
      at 20 features over 200,000 rows returns 393 MiB (233 fields) and
      peaks 797 MiB above the baseline: each stream writes its outputs row
      by row into its `ChunkOut` buffers, and `assemble` (bank.rs) copies
      them into one Arrow array per field, so both exist at the peak. Not
      a leak: after `del` the process holds 748 MiB, and five calls level
      off near 1.1 GB, which is the allocator keeping freed pages. Ten-target
      `ewridge` (25 fields) peaks at 0.9x its output. A fix writes each
      field straight into its final column (column-major), a hot-path
      refactor; it needs its speed measured against the current path, and a
      streamed run already peaks at a chunk's worth, so the gain is the
      eager call's.
      (b) **The `streaming` RSS budget measures Polars' read-ahead more than
      this library.** Its peak grows with the rows (286-292 MiB at 1M rows,
      511-531 at 3M) and so does a plain `scan_parquet(...).with_columns(...)
      .sink_parquet(...)` with no model (227 -> 473; peak footprint 246 ->
      477, so not mapped pages); with `POLARS_MAX_THREADS=4
      POLARS_ROW_GROUP_PREFETCH_SIZE=1` that drops to 95 -> 110, so the
      growth is Polars' default row-group prefetch reading ahead of the
      writer, bounded by that setting, and this library's own addition is
      about 50-65 MiB at either size. *(b) done 2026-10-13*: the budget's
      case is now `streaming_bounded`, run with the prefetch at 1 on four
      threads (203 MiB at 1M rows, 287 at 3M here; Polars alone 96 and 126);
      the old `streaming` budgets removed, a stale-budget check added, and
      the new case's budgets applied from the first CI run. Checked that
      nothing leaks: a bank's state is 4,460 bytes after 1M rows and after
      3M; fifteen repeated runs over 1M rows level off at 299 MiB for
      Polars' `collect_batches` alone, 355 MiB with the bank (a constant 56
      above it) and about 700 MiB eagerly -- the allocator's ramp, not a
      slope.
      *(a) done 2026-10-13* (`crates/online-polars/src/bank/assembly.rs`,
      PERFORMANCE §40): a spec that one stream serves in the chunk (no
      group, or one group in it) gets a `Sink` before `process`, its output
      columns allocated and NaN-filled in parallel, and each run is copied
      into it as the run ends, on another thread while the stream learns the
      next run (`rayon::join`); the run after that reuses the freed buffers.
      A spec of many groups keeps the old order (every run, then one job per
      field), now through the same sink, which allocates, copies and
      finishes each field in one job as before. Measured interleaved against
      `8950c46` (M4 Pro, load 2.1-4.7): `ew_cov` 20 features x 200k peaks
      443-446 MiB above the baseline (803-831 before; output 393), ten-target
      `ewridge` 144-153 (168-210), `ew_cov` over 8 groups unchanged at
      813-838. Speed: `ew_cov` +8%, `ew_cov` lags +6%, the light rows
      (`sgd`, `seqtest`, `ewridge` k=5) and the grouped rows 0 to -2%;
      `scripts/benchmark.py` geomean +0.2% over 37 rows, none below -2%.
      Outputs and states bit-identical: `scripts/release_probe.py` (both
      modes, 31 specs) byte-identical parquet and state files, and 97
      outputs and 95 states over every model at its fullest, grouped and
      single-stream, 1 and 7 chunks, byte-identical; a Rust test feeds one
      bank both ways and compares every field bit for bit. No state layout
      change. Not built: a sink for a spec of many groups (one column
      written from several threads at interleaved rows) and for `predict`,
      which keeps one buffer a stream for the whole chunk; both still peak
      at about twice their output. **Next**: on CI, `rss.ew_cov_wide` will read
      under its budget by more than the 30% tolerance (about 0.55 of it, by
      the local ratio) and `rss.ewridge_10_targets` near the edge (about
      0.8); tighten them from that run's `memory.json` with
      `scripts/budgets.py apply`. The
      bank instruction counts (`bank_seqtest`, `bank_ewridge_k5`, one run of
      1,000 rows, now through the sink) were not measurable here (no
      Valgrind on macOS); the work per row is the same copy, so a move
      should be small.
      *Across thread counts, 2026-10-13* (the user, before 1.0.1: "tests of
      performance across thread counts"): measured `8950c46` against
      `b796502` at `threads=1, 2, 4, 8` (PERFORMANCE §40, "Across pool
      sizes"; four rounds alternating, load 3.3-5.3): no count is slower,
      one thread included -- `ew_cov` 20 x 200k +16% at 1 thread and +20-29%
      at 2-8, ten-target `ewridge` +3-4%, `ewridge` k=5, `sgd` and
      `seqtest` over 1M rows +2-13%, their peaks down 29-63%; so no
      one-thread fallback was built. Held by
      `a_run_at_a_time_is_the_same_at_every_thread_count` (extended; its
      fastest form in the essentials) and
      `test_a_spec_one_stream_serves_is_the_same_at_every_count`, every
      field bit for bit and the state byte for byte at 1, 2, 4, 8 and the
      default, with the sink's runs counted; and by two instruction cases,
      `bank_ew_cov_wide_t1` and `bank_ew_cov_wide_t4` (8,192 rows; the
      four-thread one counted on every thread from a Callgrind client
      request), in `budgets.toml` with no budget yet. **Next**: apply their
      first CI counts; if the four-thread case's two runs disagree by more
      than a quarter of the tolerance (its threads' scheduling under
      Valgrind), `check` fails the job even with no budget: take the case
      out, or count it another way, rather than widen the tolerance.

<a id="task-254"></a>
- [x] 254. **The FFI leak check on Linux (after 1.0)** -- the user,
      2026-10-12, offered a Linux session. `test_ffi_memory.py`'s
      `test_output_outliving_its_input` failed on both Ubuntu legs of CI
      38315781585 (`7613560`; marks rising 20-50 KB an iteration over seven
      blocks, e.g. [615212 .. 634756] KB) and passed on their re-run and on
      the commits before; its third flake under load (after 2026-10-08/09).
      macOS cannot reproduce glibc's arenas. **Brief for a Linux session**:
      `uv sync`, `uv run maturin develop --release -m
      crates/online-py/Cargo.toml`; (1) run that test 20 times in a row in a
      bash loop, alone and after the whole suite, recording the marks each
      failure prints; (2) run its loop body (`fit_predict` of 400 rows,
      3,000+ iterations) in a plain script printing RSS every 240
      iterations, plain, with `MALLOC_ARENA_MAX=2` and with
      `POLARS_MAX_THREADS=1`, to tell arena growth from a leak. Report the
      RSS series, not a verdict; then decide: a fix if it leaks, else a
      measurement that tells arena growth from a slope.
      *Measured 2026-10-13 by the user's Linux session* (Ubuntu 24.04.5,
      glibc 2.39, x86_64, 4 cores, Python 3.13.16, py-polars 2.0.0,
      `7613560`): the test passed 41 of 41 times (20 alone, once inside
      the whole serial suite, 20 after it). The loop body as a plain script
      for 3,120 iterations steps about 25.5 MB in its first 240 and then
      holds within 0.5 MB: -0.02 to 0.14 KB an iteration, plain, with
      `MALLOC_ARENA_MAX=2` and with `POLARS_MAX_THREADS=1` alike, against
      the test's limit of 4. No leak. The CI failures measured a pytest
      process near 615 MB on a loaded shared runner. **Next**: run each
      plateau check in a fresh child interpreter (a baseline near 73 MB,
      nothing else in the process), as the plain script did.
      *Done 2026-10-13*: `in_a_fresh_interpreter` re-runs each of the seven
      `TestNothingLeaksAcrossTheBoundary` tests as its own pytest process
      and requires the child to report exactly one pass (a child that runs
      nothing exits 4 and fails); the three plateau-statistic tests, fed
      scripted readings, stay in-process.

<a id="task-248"></a>
- [x] 248. **A task worktree rebuilds only what it changes (after 1.0)** --
      the user, 2026-10-11: "For after 1.0 what can we do so that a task
      worktree only has to rebuild the artifacts it is working on?" Task
      245 measured why it cannot today: each task gets a fresh worktree and
      so an empty `target/` (66 s for the Rust tests, 241 s for the release
      wheel, 130 s for `dev-wheel`), while a warm one rebuilds after an edit
      in 13 s and 4.5-37 s; sccache missed a third of the crates from a new
      path because their build scripts' absolute `OUT_DIR` enters its hash.
      Options, the coordinator's recommendation first: (1) a pool of
      long-lived worktree slots (about four, the concurrency cap), each with
      its own warm `target/`, reused by `git checkout -B <task> <base>`, so
      cargo's incremental build recompiles the workspace crates the task
      changed and not Polars and its dependencies; no shared `target/`, so
      the main checkout is never clobbered; `clear_caches.sh` learns to keep
      the slots and clear the rest; check how workers are pointed at a slot
      when the agent tool's isolation makes fresh worktrees; (2) a fixed
      target path per slot, which also makes sccache's hashes repeat; (3)
      Cargo's build-directory options for sharing dependencies across
      checkouts -- read Cargo's current reference before relying on them
      (CLAUDE.md rule 10). Measure each against task 245's numbers.
      *Done 2026-10-13* (option 1): `scripts/slot.sh take|give|list`, at
      most four slots under `.claude/slots/<n>` (gitignored), the holder in
      the common git dir so a slot's status stays clean; `take` refuses
      uncommitted work, `give` detaches so the branch can go. Measured, a
      slot built at `94e6c55` and moved to `fca6699` (40 files, +5,120
      lines) against itself from nothing, load 10-20: Rust tests 79 -> 23 s,
      `uv sync` 259 -> 0 s, dev wheel 145 -> 6 s; the release wheel 198 s
      warm, since thin LTO over one codegen unit relinks it whole (only the
      gate builds it). 28 GB of `target/` a slot. `clear_caches.sh
      --worktrees` keeps slots, `--slots` clears them. Workers start without
      `isolation`, pointed at their slot (CLAUDE.md "Iterating"). Options 2
      and 3 not taken: a fixed path per slot is what this gives already,
      and a shared build directory is the clobbering a separate `target/`
      avoids.

- The Arrow tests still open: capsules dropped unread in the leak check,
  counting batches to show a reader streams, input in several chunks.

- ARROW-SOURCES' follow-ups: pandas conformance, the SQL order guidance,
  ADBC on Postgres, DuckDB's memory growth, the unmeasured producers, and
  the DataFusion spike for the CLI.

- Licensed libraries in tests: Pathway, under the Business Source License
  (open-source test libraries are task 121).

- C8's mimalloc option (a new statically linked library, rule 12).
- Reports to other projects, which wait on the user in any case: pyarrow
  25.0.1's cast bug (pinned by a test), river's `EpsilonInsensitiveHinge`.

<a id="s11a"></a>
## 11a. Decisions made while implementing

**A wide `marginal`'s shards take a batch of rows, not one (task 126),
2026-09-25.** The plan's design forked and joined the pool once per row,
on the estimate that a row at `p = 10,000` and nine targets was 2 ms of
work. Task 124 had taken that row to about 100 µs, and a rayon fork-join
costs about 10 µs here, since idle threads sleep between rows. A stand-in
for the pair loop ran at best 0.6× and 1.7× split per row, at one and nine
targets, and 3.9× and 6.5× split once per 64 rows (PERFORMANCE §25). So
the split holds rows back: every target's own numbers advance as the row
arrives, since nothing a row reports reads a pair, and each pair's share
of the row is kept until a flush steps the pairs through the held rows in
order. That is what the E73 ask itself described (`MARGINAL-AT-WIDTH.md`,
"over the chunk's rows in order"). It costs a copy of each held row's
features, 1.6 µs at 10,000, which bounds the split of the moments alone.
Borrowing the caller's rows instead would remove it, at the price of an
API that names rows by index across `online-core`'s boundary; not taken.
`"auto"` sizes from a cost model fitted to eleven shapes, up to twice the
pool's threads (ten fast cores and four slow ones here), and `shards`
stays off by default: with more groups than threads the pool is already
full. Whether `"auto"` should become the default is the user's call.

**Readiness gates (task 87), 2026-09-21.** The user's standard: a setting
rests on theory, not on a sweep, and states the intent rather than a number
that needs a formula in the user's head. So the noise gate is
`sqrt(1 + edf / n_kish)` -- the effective degrees of freedom the solve used
over Kish's effective sample size, both read from state the model already
keeps (`EwCov::n_kish`, and the diagonal of the ridged system's inverse on
the solve schedule) -- and its default is the equal-parts point, estimation
variance equal to the noise. `min_settled_frac` defaults *off* on a theory
argument, not a scar: the mean-form fit is unbiased from its first row when
the process is stationary, so the gate guards a representativeness bias only
the user can size (doc §4.1.1). `full_rank` as a boolean with a tolerance
was dropped for the exact, threshold-free `support_coef`; `G₂`, the
squared-decay Gram, was dropped because the gate's average does not need it
-- and the identity tests then measured exactly what it would buy (doc
§2.1). `min_periods` stays for every model without the statistic rather
than being aliased away: removing it would have gated `sgd` at row one.
`withheld_reason` is a dictionary-encoded Arrow array (polars reads it as a
categorical): a String column costs 16 B/row even when every value is null.

**Building the clustering (tasks 23–24), 2026-09-04.** The user chose
`kmeans` + `micro` (CLUSTERING §0's exposure), all of E36–E42, a branch
(`clustering-build`) pushed after each task for CI with `main` fast-forwarded
at the end, and a prepared 0.2.0 that they tag. Decisions taken against the
prototypes and the doc, recorded so the numbers can be re-derived:

- *Unsupervised is one thing.* `ModelKind::is_unsupervised()` (`ew_cov`,
  `kmeans`, `micro`) replaces the `ew_cov`-only exemptions: the target/feature
  leak check, the `emit_selected`/`emit_averaged` refusals, the plugin's input
  names and the expression's packing. The residual diagnostics (`emit_sigma`,
  `emit_resid_z`, `emit_metrics`, `resid_quantiles`, `emit_autocorr`,
  `emit_drift`) are **refused** for all three; `ew_cov` used to accept and
  silently ignore them (CHANGELOG).

- *Scope of `kmeans`.* Hard assignment, split–merge on a slower clock, the four
  seeding rules. The prototype's Huber weights, spherical distance, fuzzy
  memberships and stand-alone reseed rule are not built (§7 measured none of
  them earning a place). Parameters: `k`, `warm_rows` (500), `seed_rule`
  (`lloyd`; `first | farthest | kmeanspp | lloyd`), `seed` (0), `update_every`
  (1), `split_merge` (0.5), `split_merge_every` (100), `dead_frac` (0.05),
  `standardize` (true, a metric — never the coordinates, §10).

- *One accumulator, always in mean form.* A cluster is `(n, c, R)`; rows since
  the last checkpoint accumulate into a per-cluster **batch** summary of the
  same shape (`W`, mean of `z`, mean of `d²`), and the checkpoint merges batch
  into cluster: `n' = n + W`, `c' = c + (W/n')(z̄ − c)`, `R' = R + (W/n')(d̄² − R)`.
  With `update_every = 1` the batch is one row and this *is* MacQueen's step;
  no sum ever exceeds the largest input, so the bound rows of the contract
  test cannot overflow it (the prototype's `(n·C + S)/(n + W)` can). `R` for
  `kmeans` is the EW mean of each row's squared distance to the centre it
  was assigned to, *at assignment* — out-of-sample, like `sigma`. For
  `micro` it is Welford's centred radius², DenStream's definition.

- *The metric.* Diagonal EW moments (`FeatureMoments`, O(p)) rather than the
  full `EwCov`: `mw_i = 1/v_i` where `v_i > 0` and finite, else 1, read from
  the moments *before* the row. Distances are `Σ mw_i (x_i − c_i)²`; a row at
  the input bound against a variance at the opposite scale gives `d² = ∞`
  rather than NaN, and an infinite `d²` is not learned into `R` (the centre
  still moves; the radius learns nothing from a row it cannot measure).

- *Seeding.* Buffer `warm_rows` rows (the buffer is capped at
  `max(warm_rows, 1000)`, where duplicates are allowed as seeds); every buffered
  weight is multiplied by each row's `lam` (the product form, exact in both
  implementations; the prototype's `exp(L − L_row)` agrees to ~1e-9 at a
  finite halflife). `kmeanspp`/`lloyd` draw from **splitmix64** seeded by
  `seed`, `u = (x >> 11)·2⁻⁵³`, weighted choice = first index whose cumulative
  weight exceeds `u·total`, uniform `⌊u·n⌋` when the weights sum to zero; Lloyd
  is 10 weighted iterations, first minimum wins, a cluster with no weight
  keeps its centre. The same generator is written out in
  `tests/reference_cluster.py`, so the Python reference is bit-exact.

- *The far row (final design below, 2026-09-05).* A check runs every
  `split_merge_every` learned rows at a checkpoint: merge the closest pair when
  `d_ij / (r_i + r_j) < split_merge` and re-place the freed centre on the
  heaviest far summary; **else** if the lightest cluster is dead
  (`n_j < dead_frac · n_eff / k`) re-place it the same way. The dead rule is
  what recovers a centre parked at the input bound with nothing to win; the
  prototype's ratio condition (`reseed_factor`) never fires on a single
  uniform blob and was dropped. `k ≥ 3` for the merge, as measured.

- *`micro` decides at unit weight.* The absorption test (merged radius ≤ `eps`)
  is made with weight 1 whatever the row's weight, and the update then applies
  the weight. `predict` has no weight, so this is what makes `predict` the
  step without the step for a model whose label *is* the decision; a heavy row
  can push a radius past `eps` once, and the next rows see the larger radius.

- *`micro`'s threshold.* `macro_link = None` derives the linkage threshold at
  every checkpoint as 1.5× the p90 (nearest rank) of the nearest-neighbour
  spacing among the potential micro-clusters — §6.5's rule; a value is
  `macro_link · eps`, the prototype's constant. Default `eps` is `0.4·√p` in
  the standardized metric (§7.8: the clean-mixture setting; shapes need
  0.07–0.1·√p and a larger `max_clusters`). `beta_mu` 3, `max_clusters` 200,
  `prune_every` 100. Age decay per micro-cluster is a product of `lam`s.

- *Outputs.* `kmeans`: `cluster` (i32, null before seeding or under
  `min_periods`), `dist`, `dist2` (second-nearest; null at `k = 1`), `n_eff`,
  `coef` = the `k × p` centres flat, named `coef_cluster{j}_{feature}` by
  `coef_fields`. `micro`: `cluster` (i32 macro label of the nearest potential
  micro-cluster), `dist`, `micro` (i32, the id the row would join, or the id a
  new one would get), `outlier` (bool), `n_clusters` (i32), `n_micro` (i32),
  `n_eff`, `coef` = per potential micro-cluster in id order
  `[id, label, weight, radius, centre…]` — ragged, so `coef_fields` is empty
  and `coef_index` refuses it as it does `ew_cov`. Two new `Source`s carry the
  integer and boolean columns out of the `pred` buffer.

- *Contract tests.* `kmeans`/`micro` get the shared probe, the `PROBED` entry,
  a golden signature, and a recovery criterion of their own under the names
  the parity scanner requires (`Recovery::Fit`): after the bound rows, the
  tail's outputs are finite and its mean `dist²` is within the tolerance of
  a twin that never saw them — 1e-6 as the margin, measured 2.9e-15
  (standardized) and 0 (raw).

**Task 23's deep testing: what the split–merge move can and cannot repair,
2026-09-05.** Every claim below was measured through the Rust bank on
`scripts/clustering_experiments.py`'s fixtures (N = 20000, p = 4, k = 5,
halflife 3000, 20 seeds, last-quarter ARI) and the stranded fixture in
`tests/test_kmeans.py`; the numbers are what the tests pin.

- *The doc's regime claim was a seeding artefact.* CLUSTERING §7.6 reported
  "split–merge recovers a regime change in 1500 rows" because the prototype's
  one k-means++ start had put two seeds in one blob, which the merge then
  freed. With `lloyd` seeding the plain model scores 1.000 / 1.000 / 0.926 /
  1.000 over the four segments of that fixture with no move at all. The real
  case is a *stranded* centre: a blob dies and another is born far from every
  centre (fixture `stranded`, 4 blobs, halflife 1000). The move recovers it
  to a tail ARI of 1.000 against 0.71–0.73 without.

- *Far rows are summarised, never learned.* A row is far when `d² > f · R̃`,
  `f = 1 + FAR_SIGMAS · sqrt(2/p)` (`FAR_SIGMAS = 4`: `d²/R̃ ~ χ²_p/p` has sd
  `sqrt(2/p)`; a Gaussian blob crosses the cut 0.7% of the time in 2-D, 0.4%
  in 4-D), `R̃` the mean of the trusted radii (`RADIUS_ROWS = 10` learned rows
  and `r2 > 0`) leaving out the largest, or that one alone; with no trusted
  radius nothing is far (or everything would be). A far row goes to its
  cluster's far summary (Welford) and nowhere else — not the centre, the
  radius or the weight. Two designs in between failed the regime unit test:
  far rows moving the centre but not the radius drag a centre off its own
  blob and break the closest-pair ratio; far rows counting in `n` but not
  the centre keep a jumped blob's centre alive forever.

- *The winsorized radius is what makes the cut safe.* At each check
  `r2_j ← (n_j r2_j + F_j · cut) / (n_j + F_j)`: far rows count in the
  radius as if they sat at the cut. A burst of outliers widens it a little
  (steady state ≈ 1.17× at 5% far); a cluster whose rows are all far widens
  by `(n + f F)/(n + F)` per check — the contract test's bound rows had left
  the cut at 1.3e-149 with every radius at 2.7e-150 and `n_dead = 406`, a
  trap that never opened until this. The per-cluster ratchet cut tried
  before it was dropped: it blocks the wide-cluster split a freed centre
  needs.

- *Where a freed centre goes.* The heaviest far summary's mean, with the
  typical radius `R̃` and half the source's weight (a newborn with the far
  weight dies at the next check at `dead_frac = 0.25`). A merge, which costs
  a live cluster, is gated: the source must hold at least `FAR_ROWS = 3` rows
  weighing `FAR_SHARE = 5%` of the window's learned weight `V`, the pair's own
  summaries pooled. Without the gates newborns placed on 1–9 outliers with
  `r2` 10–30 poisoned the ratio and merged real blobs (min ARI 0.051 on the
  outlier fixture; now 0.768, the same symmetric split-blob miss as without
  outliers). A dead centre, already lost, takes any far rows, its own
  included.

- *Seeding trims by the same rule.* Rows whose `d²` to the EW mean exceeds
  `f` times the buffer's weighted mean `d²` do not choose the seeds (the
  whole buffer does when the rest cannot give `k` distinct seeds) and are
  replayed as far rows. Outliers 5% + drift: 0.984 mean ARI, the plain 0.749
  before.

- *Latency, measured.* A stranded centre is re-placed `log2(1/dead_frac)`
  halflives after its blob vanished: 4500 rows at the default 0.05 and
  halflife 1000 (formula 4320), 2000–2500 at 0.25 (formula 2000); at
  halflife 3000 the default does not fire within the 10000 rows the regime
  fixture leaves (0.784, 18/20 misses, against 0.824 without the move, whose
  nearest centre at least drifts toward the new blob — far rows do not drag)
  while 0.25 scores 0.933 with 1/20 misses. The price of `dead_frac`: a blob
  lighter than `dead_frac / k` of the stream loses its centre whenever any
  row is far. The default stays 0.05; the README says when to raise it.
  With `k = 1`, or when every cluster sees far rows, the typical radius
  widens with the cut and the rows are learned again after
  `log(D²/r2) / log((n + f F)/(n + F))` checks (659 rows for a 20-sd jump
  at halflife 200). One jumped blob among `k ≥ 2` waits for the dead rule:
  its cluster's widening radius is the largest, which `R̃` leaves out.

- *Blind spot.* A cluster owning two blobs: all its rows are within its own
  radius, and their far mean is its own centre. `lloyd` seeding is what
  prevents it; the drift fixture's 1/20 miss is such a pair at ratio ≈ 0.6
  (> 0.5 by design: a legitimate wide cluster looks the same).

- *Rejected on measurement, not to be retried:* D²-weighted reservoir of far
  rows (outlier-prone, random); the max-ratio far row (picks outliers by
  construction); a recent-share dead test over `split_merge_every` windows (a second
  time scale; kills quiet clusters; outlier trickles defeat ratio tests);
  ISODATA per-feature variance split (k·p state, blind to bimodality in
  general position); in-place coincident split as a trigger (splits wide
  clusters); self re-placement when far weight exceeds own intake
  (ping-pong). A possible follow-on, not built: a cohesion-gated fast
  re-placement (a far summary whose `r2` is blob-like against `R̃`) would cut
  the stranded latency from halflives to one check.

- *Oracle.* `tests/reference_cluster.py` mirrors every operation in order and
  the 22 oracle tests are bit-exact; 61 kmeans tests in all.

**Task 24's decisions: what `micro` admits, links and prunes, 2026-09-05.**
Measured through the oracle (`tests/reference_cluster.py`, n = 6000, halflife
3000, `prune_every` 100) on the shapes of `scripts/clustering_experiments.py`,
then through the Rust bank at 20k–200k rows; ARI against the truth, DBSCAN on
a 3000-row sample as the ceiling. What the tests pin is what is written here.

- *The label and the id are two columns.* §6.5's rule 2 ("a row's label is
  the id it would be absorbed by") is kept for `micro`; `cluster` is the macro
  label of the nearest *potential* summary, null while there is none, so a
  row that opens a summary still reads the cluster it sits next to. The
  doc's ARI complaint about variable-`k` (purity 1, ARI 0.4–0.8) was about
  ids; on labels the built model scores 1.000 where DBSCAN does.

- *The admission rule is DenStream's, at unit weight, with a capped radius.*
  A row is admitted where a unit row would be (`merged_radius2(λn, r2, d2, 1)
  ≤ E`, `E = eps²p`), potential summaries first, then outlier ones, else it
  opens a summary; it is then absorbed with its full weight and `r2 ← min(r2,
  E)`. Without the cap a heavy row overshoots the bound and the summary
  admits nothing — not even a row at its centre — until decay brings `n`
  under `E/(r2 − E)`, halflives later; capped, it is merely full. The
  alternative measured and rejected, admission by distance to the centre
  (`d2 ≤ E`, weight-independent): ARI radius / distance — moons .05
  0.999/0.670, .07 0.999/0.997, .1 1.000/0.999; rings .05 0.788/0.512, .07
  1.000/0.741, .1 1.000/0.834, .14 0.000/1.000; varied .05 0.866/0.700, .07
  0.950/0.790, .1 0.972/0.934, .14 0.088/0.559; highdim20 .2 0.875/null, .25
  1.000/1.000, .4 0.000/1.000. The distance rule fragments at the working
  `eps` (twice the live summaries, 3–5× the outlier share) and only wins
  where the radius rule has already bridged. `predict` decides with
  `factor(d_clock)` applied to every `n`, bit-exact with `step`.

- *The derived link.* `L = max(2·eps√p, 1.5 × p90 of the nearest-neighbour
  spacing among potential summaries)`, nearest rank, recomputed at every
  checkpoint; `macro_link` given makes it `macro_link·eps√p` (0 links
  nothing, 2 only summaries that touch). A sweep of (quantile, factor) put
  q0.9 / 1.5 best overall: moons 0.996–1.000 at eps .05–.14 (ceiling
  0.999); rings 0.787 / 1.000 / 1.000 / 0.000 at .05 / .07 / .1 / .14 (the
  ring gap is 3.55·eps√p, under `L` = 4.55 at .14); varied 0.86 / 0.95 /
  0.972 / 0.088 (the varied-density limit of one global threshold: `L` 4.0
  > gap 2.9 at .14); aniso ≈ 0.57 (ceiling 0.68, not density-separable);
  highdim20 null at eps ≤ .14, 0.997 / 1.000 at .25 / .3 (`L` = the floor),
  0.000 at .4 (one summary per cluster). A promoted summary attaches to the
  nearest potential one within `L` at once (attach-on-promotion), so a new
  cluster has a label before its first checkpoint.

- *Pruning is DenStream's ξ, on a learned-row schedule, with no grace.* An
  outlier summary is dropped at a checkpoint when `n < ξ(age) = (λ^age λ^Tp −
  1)/(λ^Tp − 1)`, `Tp = ⌈h log2(β/(β − 1))⌉`: ξ = 1 at birth and rises to
  β, so a lone-row summary is dropped at the first checkpoint after the one
  it was born on (born on the checkpoint row itself it survives that one:
  age 0, weight 1 not below 1). A potential summary is dropped under β and
  lingers `h log2(n₀/β)` after its rows stop. Zero-weight rows do not count
  toward a checkpoint. `prune_every` 25 fragments sparse shapes (rings@.1 →
  0.059) because a one-row summary never sees a second checkpoint; a
  one-checkpoint grace repaired that and bridged `varied` (0.972 → 0.771) —
  no grace, default 100. An infinite halflife prunes nothing; only the cap
  applies, evicting the lightest outlier summary, else the lightest
  potential one.

- *Two failure regimes, both readable off the outputs, neither guarded.*
  `eps` too small: every row an `outlier`, `cluster` null, `n_micro`
  cycling — no summary reaches β before ξ takes it. `eps` too coarse for
  the derived link: `n_micro ≈ k`, each cluster one summary, the p90 spacing
  *is* the inter-cluster spacing and everything bridges into one cluster
  (4-D blobs sd 0.6 at eps .3, constant-feature fixture at .1, highdim20 at
  .4). A regime guard (refuse to link when `n_micro` is small) was measured
  and rejected: it fragments the shapes the link exists for. Rule of thumb,
  in the README: `eps` ≈ the within-cluster spread per standardized
  coordinate, 0.07 for 2-D shapes, 0.3 for separated Gaussians in 20-D.

- *At scale (Rust bank, halflife 3000).* 20k rows: moons .07 → 1.000, rings
  .1 → 1.000, varied .1 → 0.984 (sample ceiling 0.956), highdim20 .3 →
  1.000 (0.728). 200k rows of 4-D blobs, halflife 20000: eps .2 / .25 →
  1.000 / 0.999 with ≤ 200 live summaries. Stranded fixture, halflife 1000:
  the newborn cluster is labelled 31 rows after its first row, the dead one
  lingers 5.5 halflives (`h log2(n₀/β)`), then `n_clusters` returns to 4;
  tail ARI 1.0. 5% uniform noise at eps .07: 94% of noise rows flagged, 0.3%
  of real rows, ARI on the real rows 1.0; at .14 the noise bridges the
  clusters. sd-0.8 blobs in 4-D: .25 → 0.995, but .2 and .3 → four clusters
  (two bridged) — the working band narrows as clusters approach.

- *Oracle.* `reference_cluster.py`'s `Micro` mirrors every operation in
  order; 17 bit-exact cases; 63 micro tests in all.

**Task 25's decisions: the conformal recursion, 2026-09-05.**

- *The rule.* Per slot, `q` is a tracked quantile of the conformity score
  `s = |resid|`: read `lo = pred − q`, `hi = pred + q` before the row, then
  `err = 1{s > q}`, `q ← max(0, q + η·w·(err − α))`, `α = 1 − coverage`. This
  is the P step of Angelopoulos, Candès & Tibshirani (2023) applied to the
  score's quantile, i.e. online gradient descent on the pinball loss, and it
  telescopes: for scores in `[0, B]`, `|Σ η_t (err_t − α)| ≤ B + max η`
  (the clamp at zero can only add coverage), so the average miss rate tends
  to `α` at rate `1/T` on *any* residual sequence. No distribution, no
  stationarity, no split: the score comes from a model that has not seen the
  row, which is the property the library already guarantees everywhere.

- *The step is in sigma units.* `η_t = conformal_rate · sigma_t`, the slot's
  EW residual standard deviation before the row. A fixed `η` would need a
  scale from the user; scaling by `sigma` makes 0.05 a sensible default on
  every stream and lets the radius follow a scale shift at the speed `sigma`
  does. The bound holds in σ-weighted form with `η_t` in place of `η`. With no
  usable `sigma` (`emit_sigma` is not required: the tracker reads the
  internal one, which exists for every regression model) the step is 0 and
  the radius holds.

- *The warm start.* `q` is undefined until the first scored row that has a
  finite positive `sigma`; then `q = sigma · Φ⁻¹(1 − α/2)`, the Gaussian
  radius, and that row is not scored. `Φ⁻¹` is Acklam's rational
  approximation evaluated in a fixed order, so the Python replay in
  `tests/test_conformal.py` is bit-exact. The alternative, `q = 0` and let it
  grow, wastes `B/η` rows widening from nothing; starting at the Gaussian
  radius is right on Gaussian residuals and a few steps off otherwise.

- *`coverage` is the EW hit rate* on the model's own clock (`cov_w` decays
  by `lam`, a row adds `w`), read before the row, so it says what the
  interval has delivered recently, not over all time. A null target or a
  zero-weight row ages it and moves nothing else; the warm-start row is not
  counted.

- *Measured (200k rows, halflife 2000, rate 0.05).* Coverage at target
  ±0.01 on Gaussian, t(2.5), `exp(x₁)·N(0,1)` and slope-flip + noise-×3
  residuals; the Gaussian `pred ± 1.645·sigma` covers 0.942–0.951 on the
  last three. The radius follows the noise ×3 shift to within 10% of the
  new Gaussian radius. Levels 0.5, 0.8, 0.99 are met within 0.012 on fat
  tails. Cost: three f64s of output and five of state per slot; no
  measurable throughput change on `ewridge`.

**Task 26's decisions: `mahal` and EW-PCA on `ew_cov`, 2026-09-05.**

- *`mahal` is a distance, in σ units.* ENHANCEMENTS E37 sketched the
  quadratic form `δᵀ Σ⁻¹ δ`; the field is its square root, so at k = 1 it is
  `|z|` and it reads like `resid_z` — the feature-side twin it was proposed
  as. `mahal²` is then χ²_k on Gaussian columns; the README says so and
  `tests/test_ew_cov_scores.py` holds it at 200k rows. The matrix solved is
  `C + s·prior·I`, the same fading ridge `partial_corr` reads, hence the
  `precision_prior` requirement; NaN until the prior is set, before
  `min_periods`, or when the solve fails. One `solve_spd` per row, not a
  tracked inverse (the E2 finding still stands: a tracked inverse cancels to
  zero under a dominant row and never recovers).

- *`mahal_quantiles` are unweighted P² trackers* of the emitted score, like
  `resid_quantiles`: a zero-weight row still adds its score, a `predict`
  pass does not, and they are read before the row. They lag the score by
  the five rows P² needs to start.

- *The PCA refresh runs after the row's update*, not before it, so
  `predict` (no update) and `step` read the same frozen loadings and the
  score `Σ v_j,i (x_i − m_i)` uses the live mean with loadings from the
  last refresh. The first refresh waits for `min_periods`; then every
  `pca_every` learned rows. A zero-weight row advances the cadence counter
  like any other learned row, since it advances the clock.

- *Signs follow the previous refresh.* E38 proposed largest-magnitude-entry
  positive. On `gaussian(k=5, seed 10)` that rule flipped `pc1` between two
  refreshes (dot with the previous loading −0.99999) when two loadings of
  near-equal size traded the lead. The rule built: sign each new loading so
  `v_new · v_old ≥ 0` with the previous refresh's loading for that
  component; fall back to largest-entry-positive when there is no previous
  one or the dot is exactly 0. The Python oracle carries the same rule
  (`pca_oracle(c, r, prev)`), and a test proves the max-abs rule would have
  flipped on the same stream.

- *`pc<j>_share`, not `explained`.* One number per emitted component (its
  eigenvalue over the trace) rather than the `k`-vector E38 sketched; the
  trace is the sum of the diagonal, so the shares of the emitted components
  need not sum to 1. NaN when the trace is ≤ 0 (a constant stream).

- *Measured (200k rows, Mrows/s, k = 4 / 8 / 16 / 32).* `mean,std,corr`
  7.44 / 2.85 / 0.86 / 0.21; `+ mahal` 3.86 / 2.21 / 1.07 / 0.35; `+ 2
  quantiles` 3.32 / 2.03 / 1.03 / 0.34; `pca=2, pca_every=1` 0.98 / 0.35 /
  0.10 / 0.03; `pca_every=100` 6.52 / 3.74 / 1.92 / 0.78; `partial_corr`
  2.92 / 1.21 / 0.39 / 0.09. The eigendecomposition is ≈ 1 µs at k = 4 and
  ≈ 30 µs at k = 32, which is what `pca_every` amortizes; `mahal`'s solve is
  cheaper than `partial_corr`'s because it is one right-hand side.

**Task 27's decisions: `ew_class`, 2026-09-05.**

- *A kind of its own, not an `ew_cov` option.* ENHANCEMENTS E39 sketched a
  class-conditional `ew_cov`. Built as `ew_class` because nothing of
  `ew_cov`'s surface carries over: no `stats`, no pairs, a label column
  instead of a target and a String output. It reuses `EwCov` whole — one per
  class, `with_precision_prior` — and the covariance shapes are views of the
  same state: `full` reads each class's `C_c + r_c I`, `shared` pools them
  by the class weights `Σ π_c (C_c + r_c I)` (one factorization, all the
  quadratic forms at once through `quad_forms_logdet`), `diagonal` reads
  the variances, clamped at 0. The ridge `r_c = precision_prior ·
  precision_scale_c` fades per class as `partial_corr`'s does, so a class
  is scoreable from its first row and the prior is gone once it has data.

- *`n_eff` counts every accepted row, labelled or not.* It is the stream's
  weight, the quantity `min_periods` compares against everywhere else; the
  class weights `n_c` are the labelled weight per class and a class's share
  `π_c` is read off them. A row before `min_periods`, or before any class
  has been seen, is null. Hard rule 8 holds: `n_eff` is reported before the
  row's own update and decay.

- *A null label scores and does not learn; an undeclared one is an error.*
  Null is the late-label case (score now, learn when the label comes back
  in a later stream) and mirrors a null target. A value not in `classes` is
  neither a class nor "unknown": a static schema needs the class set
  declared, and silently dropping the row would hide a typo, so the bank
  raises with the row, the value and the class list, and says to null the
  rows that should only be scored. The column is read as a key (cast to
  String, like `group`), which is what makes integer, boolean and
  categorical label columns work through their text.

- *An unseen class has `p = 0` exactly and null means.* Its log-likelihood
  is −∞, so the softmax gives exactly 0 rather than a tiny positive number,
  and it is never the argmax; its `coef` entries are null (the list builder
  and `ModelBank.coef` both map NaN to null, which every `coef` list now
  honours: finite or null, inside the list too). The first maximum wins a
  tie, so two classes with identical states resolve to the first declared.

- *Plumbing predicates.* `is_unsupervised()` (ew_cov, kmeans, micro: no
  target column at all — the leak-check exemption, the expression's input
  packing) is now distinct from `predicts_no_target()` (those plus
  ew_class: no residual, so every residual diagnostic is refused by name and
  the per-model slot count comes from the schema). `ew_class`'s label
  travels as `targets[0]`, so every place that reads the target column by
  name — `keep_columns`, the projection the lazy source pushes, the
  expression's packing — works unchanged.

- *Fields.* `class`, `p_<class>`, `n_eff`, `coef`, with the halflife suffix
  after each (`class@h50`, `p_a@h50`); `output_index` kinds `class` (dtype
  `str`) and `p`; `coef_fields` one slot per class named by the class, so
  `coef_index`'s `target` column is the class and `term` the feature.

- *Measured (200k rows, 3 classes, Mrows/s, k = 2 / 4 / 8 / 16 / 32).*
  `full` 1.63 / 1.33 / 0.78 / 0.38 / 0.13; `shared` 3.22 / 2.55 / 1.58 /
  0.81 / 0.28; `diagonal` 6.92 / 5.88 / 4.51 / 2.84 / 1.57. `full` pays `C`
  Cholesky factorizations per row, `shared` one, `diagonal` none; the update
  itself is `ew_cov`'s O(k²) on one class and an O(1) decay on the others.

**Task 28's decisions: constrained coefficients, 2026-09-05.**

- *One projection, not two.* ENHANCEMENTS E40 sketched a box (clamp) and a
  simplex (the sorting algorithm). Built as one operator over
  `{lo ≤ b ≤ hi, Σb = s}` with any of the three optional, because the
  portfolio ask is usually all three at once (long-only, capped per name,
  fully invested) and the sort covers only `lo = 0, hi = ∞, s = 1`. The
  Lagrangian `b_i(μ) = clamp(v_i − μ, lo_i, hi_i)` has a sum that is
  piecewise linear and non-increasing in `μ`; the root lies between two of
  the `2k` breakpoints `v_i − hi_i`, `v_i − lo_i`, found by a binary search
  over the sorted breakpoints and one linear solve on the segment. `O(k)`
  for a box alone (no sort), `O(k log k)` with a sum. `constraint.rs` checks
  it against the sort formula on the simplex and against the KKT
  conditions on random boxes-with-a-sum; the Python side checks it against
  a bisection that shares no code.

- *Where the projection runs under `scale_features`.* The bound is a
  promise about the coefficient the caller reads, `c_i = b_i / scale_i`, so
  in the standardized space it is `b_i ∈ [lo_i·scale_i, hi_i·scale_i]` and
  the sum is `Σ b_i / scale_i = s`. The nearest point in the *standardized*
  metric (the metric the gradient step is taken in) is the box-with-a-sum
  projection with weights `a_i = 1/scale_i` — the same breakpoint search
  with `a_i` in the sums. So `sgd` projects `beta` in place with the
  scales, not the reported coefficients. Consequence, measured: with
  features a million apart and a sum on the caller's coefficients, the sum
  goes to the coefficient whose unit is cheap in the standardized metric,
  and the well-determined slope keeps its truth — not the corner a
  clamp-then-renormalize would give.

- *When.* `pa` projects right after each target's update, inside the row.
  `sgd` projects at the end of `step`, after the scaler has moved, for the
  targets that learned this row — or every target when the scales moved
  (a weight > 0 with `scale_features`), because a moved scale changes what
  the stored `beta` means in the caller's units even for a target that saw
  a null. `predict` never projects; a zero-weight or null-target row moves
  nothing. The initial zero is projected in `new` so the first prediction
  and the first `coef` are feasible: uniform weights on a simplex, the
  nearest corner of a box that excludes zero.

- *`pa` under a constraint.* Its step is the smallest change that meets
  the row's margin; the projection then takes part of it back, so the
  margin is not met and a truth outside the set is never reached (a wall
  is approached, not sat on, as `pa.rs` measures: 4,000 of 5,000 rows
  touching it). Documented as "keep `c` small"; not fixed by projecting
  inside the step, which would be a different (constrained-QP) update.

- *Refusals.* Lengths by name, NaN and the wrong infinity by index (`+inf`
  as a floor or `−inf` as a cap pins nothing and means a typo), a floor
  above a cap, a non-finite sum, and a sum outside `[Σlo, Σhi]` — with
  rounding slack of `1e-12` relative, so `[0.1, 0.2, 0.3]` (which sums to
  `0.6000000000000001`) accepts a sum of `0.6` and projects onto the
  floors. Python's `_INF_OK` admits `inf` for `coef_min`/`coef_max` and
  refuses it for `coef_sum`, matching the Rust parser
  (`tests/test_error_messages.py`).

- *Measured (200k rows, one bank, Mrows/s, k = 2 / 4 / 8 / 16 / 32).*
  `sgd` 22.2 / 20.2 / 17.8 / 13.0 / 7.4; with a box 21.0 / 18.7 / 15.6 /
  11.4 / 6.5; on the simplex 17.0 / 12.8 / 8.2 / 3.9 / 1.7; the simplex
  under `scale_features` 8.5 / 6.5 / 4.6 / 2.5 / 1.1. `pa` 22.2 / 20.2 /
  19.5 / 14.5 / 10.0; on the simplex 19.3 / 15.9 / 12.6 / 7.0 / 3.7. The
  box is `k` clamps (5–12%); the sum is the sort and `log2(2k)` sweeps of
  `k` clamps, about 0.4 µs a row at `k = 32`, which is what `O(k log k)`
  costs and is unchanged by an unstable sort. A first cut allocated a
  `learned` flag per target per row and cost the box 28% at `k = 2`; it
  is a `#[serde(skip)]` buffer now. The `O(k)`-expected simplex
  projections (Condat 2016) would be the next step if a bank ever ran
  hundreds of constrained slopes; none does.

**Task 29's decisions: coefficient reversion, 2026-09-05.**

- *Where the transition sits.* At the top of `step`, before the row's
  prediction and before the process noise, so `P ← ΦPΦ + Q·d` is the
  textbook order and the prediction a row sees is the coefficient after
  the decay over the gap since the last accepted row. That makes a
  null-target or zero-weight row advance it too — it is clock, like the
  `Q·d` it precedes — and it makes `predict(x, d)` a closed form,
  `Σ z_i·(β_i·φ_i(d))`, bit-identical to the step because `β·φ == φ·β`;
  the contract test (`predict_is_the_step_without_the_step`) runs with
  and without reversion. The bank's `predict` already passes the distance
  to the last learned row, capped by `max_dclock`, for `holt`; `kalman`
  now reads it.

- *Toward zero, not toward a prior.* ENHANCEMENTS E41 offered both. Zero
  in the standardized coordinates is the one target that means the same
  thing at every row — "no effect" for a slope, "the target averages zero"
  for the intercept — while a fixed caller-unit prior would be a moving
  target as the scaler moves, and `ewridge`'s E15 warm prior solves from
  accumulated statistics the filter does not keep. No `coef_prior` on `kalman`.

- *A scalar broadcasts to every slot, intercept included*, as
  `coef_halflife` does; `[inf, r, r]` exempts it. Nothing is applied when
  no slot is finite, so the default is bit-identical to the previous
  filter (checked against HEAD in a worktree, and by the golden bank).

- *Prior variance.* A reverting slot's stationary variance is
  `q_i·d/(1−φ_i²)` instead of unbounded; measured after 200k null-target
  rows, the gain of the next update matches that to 1e-6 relative (and the
  random walk's `n·q` to 1e-3).

- *Refusals.* Length 1 or `k_total`; NaN, zero and negative refused (`inf`
  is the walk); Python's `_INF_OK` admits `inf`; `rls`/`ewridge` do not
  take the argument.

- *Measured.* 300k rows, exact-Bayes process noise (`q = [0, 1−φ², σ_w²]`,
  `obs_var = σ²`, `standardize=False`): a slope active 2% of the time is
  tracked at 0.51× the random walk's error (0.86× dense at `H = 20`,
  `σ = 2`; 0.93× at `H = 40`, `σ = 1.5`); a slot left at `inf` within 0.2%
  of the unmodified filter; a random-walk truth under a reverting filter
  (`r = 30`) more than 2× worse. A first cut with `coef_halflife=40` and
  the default process noise had the reverting filter *worse* (0.72 vs
  0.53): `q` was far below the truth's innovation variance and the shrink
  dominated, so the test was rewritten around the exact-Bayes noise — the
  prior has to match the world, and the docstring says so. Throughput
  (200k rows, one bank, `k = 2 / 4 / 8 / 16 / 32`): the transition is `k²`
  multiplies on a `~3k²` update, 8 / 14 / 16 / 15 / 14 % slower, nothing
  when off. The step's pre-existing per-row `scales()` and `q_vec()`
  allocations (one `Vec` each, per target for `q_vec`) are the first thing
  task 31 should remove.

**Task 30's decisions: `seqtest`, 2026-09-05.**

- *A model, not a diagnostic flag.* ENHANCEMENTS E42 asked for a test
  between two specs. Built as a spec kind (`type = "seqtest"`) rather than
  an `emit_*` flag on the sides, because the test has its own targets (the
  residual fields it compares), its own warm-up and reset policy, its own
  state to save, and a second use the flag could never have: on its own it
  tests the sign of any column. `targets` name the columns (column mode) or
  the residual suffixes `t` of `resid_<t>` (compare mode).

- *The bet.* Per target two one-sided e-processes, `E⁺` for "positive" and
  `E⁻` for "negative", each `E ← E·(1 + λ s)` with the Krichevsky–Trofimov
  stake on the counts *before* the row, `λ⁺ = max(0, (n⁺ − n⁻)/(n + 1))` —
  the Beta(½,½) posterior mean `2p̂ − 1`, clipped at zero so a side never
  bets against its own lead. Two consequences the tests pin: the losing
  side's `log_e` is *exactly* 0 while it has never led (assert `<= 0`, not
  `< 0`), and `1 − λ ≥ 1/(n + 1)` bounds a lost bet at `ln(n + 1)`, so the
  wealth is finite forever. Where the clip never binds the wealth is the KT
  mixture `2ⁿ B(n⁺ + ½, n⁻ + ½)/π` in closed form; a `+ + −` repeating
  pattern keeps `n⁺ ≥ n⁻` on every prefix, so its `+` side is the exact
  Beta integral and its `−` side is 0 — the oracle for the recursion, held
  through `math.lgamma` (no scipy). The two-sided e-value is `(E⁺ + E⁻)/2`,
  emitted as its parts. Not the Waudby-Smith–Ramdas aGRAPA stake: KT is
  parameter-free, has the closed form, and its regret to the best fixed
  stake is `½ ln n + O(1)`, which is what the power measurements show.

- *A trial is a row.* Ties (zero), nulls, NaN and beyond-bound values bet
  nothing and count nothing, but the row is seen (`n_eff` advances, so
  `min_periods` means what it means everywhere). No `weight` (the stake is
  a function of counts; a weighted bettor's e-process is not the KT
  mixture) and no `halflife`/`lam` (a product of bets cannot decay and stay
  an e-value); both refused by name, as are `features` and every residual
  diagnostic (`predicts_no_target`). `session_gap="reset"` and
  `on_clock_reset="reset_state"` restart the test; a clock otherwise only
  admits or refuses rows.

- *Compare mode is column mode on `|resid_b| − |resid_a|`*, positive when
  `a` came closer — the same sign under any loss that grows with `|resid|`
  (squared, absolute, Huber), so the test is of "`a` beats `b`" under all of
  them at once. A row where either side is null (warm-up, a skipped row) is
  a tie. `a_suffix`/`b_suffix` pick a grid instance (`resid_y__r0.5@h20`);
  one spec against itself is refused only with equal suffixes, because two
  instances of one grid are a comparison like any other. Tested by
  computing the difference as a column and running column mode on it: the
  fields agree bit for bit, including `n_eff`.

- *The two-phase bank.* `fit_predict` runs `check_clock` on every stream
  first (so a refused chunk still updates nothing, in either phase), then
  every non-comparison spec in parallel as before into `out:
  Vec<Option<Column>>`, then each comparison reading its two sides' residual
  fields from the structs in `out` (`compare_targets`), and returns the
  columns in spec order whichever phase produced them. A comparison
  therefore never needs the sides to be listed before it; `resolve_compare`
  at `Bank::new` refuses a side that is not in the bank, is itself a
  `seqtest`, or lacks the residual field, naming the fields it has.
  `predict_on_pool` does the same with the scored sides, so scoring reads
  the state before the chunk and compares what it scored. The comparison's
  own `group` is independent of the sides' (grouped sides can be compared
  pooled).

- *`RunConfig::validate` builds the bank.* The CLI's `--dry-run` used to
  validate specs one by one and would have said "config OK" to a
  comparison whose side was not in the config; it now runs `Bank::new` and
  reports duplicate names and comparison mismatches before opening the
  input.

- *The expression form.* `pl.col("d").online.seqtest()` is column mode on
  its column (`extra_targets` for more); `a`/`b`/`a_suffix`/`b_suffix` are
  omitted from `SeqTestKwargs` (`EXPR_OMITS` in the typing test) and
  refused at runtime with the way to write it: the difference as a column,
  or the spec in a bank. The kwargs TypedDict says so in its docstring.

- *`po.eval.seqtest`* mirrors the builder's keyword shape (`targets`, `a`,
  `b`, `a_suffix`, `b_suffix`, `by`, `min_periods`) and is the same
  recursion in polars expressions (`cum_sum().shift(1)` under `over`),
  bit-identical to the bank in both modes. Two gotchas it absorbs: polars
  orders NaN above every float, so `NaN > 0` is `True` and the sign must
  be masked by the bank's admission rule (`is_finite & abs <= 1e100`)
  first; and the bank emits `n_eff` through warm-up while the other fields
  are null, so the twin gates every field but `n_eff`.

- *Validity is a rate.* `E[E_t] ≤ 1` is not a test of an e-process (a
  process that never bets passes it); the guarantee is
  `P(sup E ≥ 1/α) ≤ α`, so the tests run twenty thousand streams as groups
  of one bank and count the ones whose peak crosses. Fair coin, 300 rows:
  2.2% at α = 5%, 0.4% at 1% (a lower bound of 0.5% / 0.05% catches a
  bettor that never bets); a dependent null with `P(+ | last +) = 0.3`,
  `P(+ | last −) = 0.5` — not i.i.d., still never favouring +; the two-sided
  average under 5%. Power, 2000 streams: 60% positive crosses 20 within
  1000 rows in 99.9%; 55% in 12% / 59% / 100% at 200 / 1000 / 4000 rows.
  Six million rows in half a second, so these run in the default suite.

- *What it does not claim.* A test of the *sign*, by design and by
  docstring: 60% gains of 1e-9 against 40% losses of 1e6 is "positive". A
  test of the mean would need bounded losses or clipping, which E42 ruled
  out; the sign-conditional null is the one that holds for dependent rows.

**Task 31's decisions: the performance survey, 2026-09-05.**

- *Bit-exact or not at all.* Every change recomputes the same arithmetic
  on the same inputs (`k` square roots reused across the pairs; a kept
  Cholesky factor of the same matrix; a projection's bounds prepared once;
  multiplications by exactly 1.0) or is plumbing (scratch buffers, skipped
  residual tracking for a model with no predictions, a packed validity
  bitmap, a contiguous copy). Proved, not argued: two output dumps over
  every family, with groups, weights, a null key, `predict` and chunk
  ends, compared as `u64` bits plus validity against a build of the
  Task 30 commit. A survey row that moved the wrong way was a real
  regression (the constraint scratch made the unscaled simplex 9%
  slower) and was fixed by keeping the folded instantiation, not
  accepted as noise.

- *The stride is the bank's decision, not the caller's.* `ChunkOut` is
  slot-major and a row is one store per slot at stride `n_rows × 8`
  bytes; a power-of-two chunk put every slot of a 231-wide `ew_cov` row
  in the same L1 sets (65 536 rows: 1302 ns/row; 65 552: 685), and a
  400k chunk paid a page per store. Each (spec, group) work item now runs
  its rows through the stream in runs of `ChunkOut::run_rows` — buffers
  within 2 MiB, stride an odd number of 128-byte lines, floor five lines
  — each run its own `ChunkOut`, the coefficient report on the last run's
  final row. Chunk invariance is what makes it invisible; three bank
  tests pin it. Rejected alternatives: a row-major layout (changes the
  reader, the `at` contract and the scatter path for the same cache
  effect) and splitting the predict-only path (the coef-on-last-row rule
  is the thing to preserve, and `predict` reports none).

- *Wall clock over profile.* `sample` weights a phase by its thread
  count; `assemble` on fourteen threads looked like half the chunk when
  `ONLINE_TIMING=1` said 20 ms of 300. The per-chunk section rows are the
  truth for *where*; the profiler is for *what*, at function granularity.
  Written up as a recipe in `docs/PERFORMANCE.md` §13.

- *One factor per learned row.* `ew_class` full covariance cached the
  factor per class (`solve::SpdFactor`, `ewclass::Factors`,
  `#[serde(skip)]`, invalidated for the class a row learns) rather than
  refactorizing all classes every row; the shared form was already at one
  factorization per row and did not move.

- *Deferred, with the reason.* `kalman`'s standardizer is a full `EwCov`
  read on its diagonal — a diagonal accumulator is O(k) instead of O(k²)
  but is a serialized layout change (rule 5, schema bump), left for a
  release with another reason to bump. `pca_every=1` is O(k³) per row by
  request. The simplex sort of `2k` breakpoints could be an O(k)
  selection. Compare mode's second pass is 8 ms per 400k rows.

- *Documented ratios, not a loaded machine.* The README's two tables are
  from one quiet run of `scripts/benchmark.py --markdown` on the final
  build; the per-row and thread-scaling tables in `docs/PERFORMANCE.md`
  §13 carry both builds so the ratio, not the absolute, is the claim.

**Task 32's decisions: 0.2.0 prepared, 2026-09-05.**

- *What ships.* Tasks 23–31 on `clustering-build`, fast-forwarded onto
  `main`: `kmeans` and `micro`, conformal intervals, `mahal` and EW-PCA
  on `ew_cov`, `ew_class`, constrained `sgd`/`pa`, `kalman` reversion,
  `seqtest`, and the bit-exact performance work. Version 0.1.1 → 0.2.0 in
  `Cargo.toml` (workspace and the two path dependencies), `pyproject.toml`,
  `__init__.py` and both lock files — the same six files as the 0.1.1 bump;
  `CHANGELOG.md` cut with `[Unreleased]` left at "Nothing yet."

- *Minor, not patch.* Pre-1.0 the minor carries breaking changes, and
  there is one: a residual diagnostic on a model with no predictions is
  refused by name where `ew_cov` accepted it silently. Everything else is
  additive. The README's pin advice moves to `~=0.2.0`.

- *The 0.1 numbers are the 0.1 numbers.* `docs/VALIDATION.md` regenerated
  on this build differs from 0.1.1's in the version line and one timing;
  `golden.rs` and `test_golden_pipeline.py` gained entries for the new
  models and lost no value (the diff since `main` removes three type
  signatures and nothing numeric); Task 31's two dumps compare bit for bit
  against the Task 30 build. The README's two *Performance* tables and the
  test-count line (466 Rust tests, ~1,700 pytest cases from 920-odd
  functions) are regenerated; `docs/TESTING.md`'s status line with them,
  its coverage figures left dated.

- *Where it stops.* Gate PASS on every commit; CI green on the Task 31
  commit (33960229948) and on the prepare commit (33962266816, superseded
  by the run on this head); `main` fast-forwarded to this head and pushed
  once that run is green. The Release rehearsal (`release.yml` dispatched
  without a tag) and the annotated tag `v0.2.0` are the user's steps, as
  for 0.1.1: the tag goes on the commit whose rehearsal and CI are green,
  and the `v*` ruleset makes it permanent.

**Task 33's decisions: the diagonal standardizer and schema 3, 2026-09-05.**

- *The deferral was the user's to overturn, and they did.* Task 31 left
  `kalman`'s standardizer alone because a diagonal accumulator is a
  serialized layout change (rule 5). Asked, the user said a schema change
  is fine this early in the project; the same waste in `sgd`'s
  `scale_features` scaler goes in the same bump rather than a later one.

- *Bit-exact by construction, then proved.* `EwDiag::update` is the
  diagonal path of `EwCov::update` with the same operation order (`a·c +
  a·b·d·d`, the deviation against the old mean, the mean updated after);
  Rust does not reassociate or fuse, so the bits agree. A unit test
  compares the two accumulators as `u64` bits over an adversarial stream
  (zero-weight first row and run, weights 0/1/2.5, two decay factors,
  offsets and scales apart by orders of magnitude, k = 1, 2, 5), and the
  Task 31 dump recipe — twelve configurations, groups, weights, two
  targets, a save/load mid-stream, `predict`, coefficients — compares
  bit-identical against a wheel built from the previous commit.

- *Migration by field name, not by luck.* A compact msgpack struct is an
  array, so `#[serde(untagged)]` layouts are newtype variants (the `rls`
  v1→v2 precedent). `kalman` renamed `cov` → `stats`, so the two layouts
  are told apart by name. `sgd`'s scaler is `Option` with
  `#[serde(default)]`: renaming it would make every schema-2 file load
  with `None` — silently, the file's scaler discarded — so the name stays
  and the layouts are told apart by `EwDiag`'s `deny_unknown_fields`
  (an `EwCov` map carries three names it does not know) plus a shape
  check (an `EwCov` array has `k²` entries where `k` are expected), each
  tested in both encodings. The conversion is `EwDiag::diagonal_of`: the
  numbers the model was already reading, so a schema-2 state continues
  the stream identically to one that never left schema 3.

- *A real 0.1 file, frozen before the change.* `state_schema2.rs` carries
  a 4126-byte bank written by the pre-change build (a hex constant, rule
  1) — `kalman` standardized with intercept and `sgd` with
  `scale_features`, a group column — and checks it loads, continues the
  stream to the bit against a fresh bank, and re-saves as schema 3. The
  bank `format_version` stays 2: the envelope did not change.
  `tests/api_surface.txt` records `schema_version = 3` (a deliberate,
  reviewed diff). The JSON-mutation refusal test lives in `online-core`,
  not the bank test: `serde_json` writes `inf` as `null`, so a bank
  file's `revert_halflife = [inf]` cannot survive that round trip.

- *`standardize=False` gains too.* The standardizer is updated on every
  row regardless (it carries `n_eff`), so the raw `kalman` runs 213 → 172
  ms as well. Reported as before/after from two wheels run back to back
  on the same (loaded) machine — the ratio is the claim.

**Task 34's decisions: the last row in the bank file, 2026-09-05.**

- *It is one row of `ChunkOut`, not a second set of diagnostics.* Every
  output buffer keeps the row index innermost (`(mi*n_slots+slot)*n_rows
  +ri`, and the same shape for metrics, conformal, quantiles, `n_eff`,
  `lam_selected`), so `LastRow::take` is one strided gather per buffer
  and `to_chunk` puts the row back into a 1-row `ChunkOut` that the
  ordinary `assemble` turns into the struct column. The saved row is
  therefore the `fit_predict` row field for field — `emit_selected`,
  `emit_averaged` and the rest are re-derived by the same code. A new
  `ChunkOut` buffer has to be added to `take` and `to_chunk` by hand, and
  switched on in `tests/last_row.rs`'s rich spec, whose field-for-field
  comparison against the frame is what catches the omission
  (`docs/EXTENDING.md`).

- *The last row **learned from**, not the last row fed.* `remember_last`
  runs after every accepted run in `process` and keeps the run's last
  *processed* row, so a chunk that ends in skipped rows (a null feature
  or weight) steps back to the row before them, and a group that never
  learned reports a null row. `predict` and `score` never touch it: the
  saved row describes the state, and the state did not move.

- *`coef` when the frame's row carried it.* The row is faithful to the
  frame, so `coef` is present exactly when that row had it — a chunk's
  last accepted row does, an interior row only on the `coef_every`
  cadence. `coef()` remains the way to the coefficients otherwise;
  duplicating them into every saved row would have made the file's row
  disagree with the frame's.

- *Additive, so no format bump.* `StreamState.last_row: Option<LastRow>`
  with `#[serde(default)]`, map-encoded like `rows_fed` and the residual
  statistics before it: files from 0.1.x load and report a null row
  (`state_v1.rs`, `state_schema2.rs` check this on the frozen fixtures).
  `Stream::restore` checks the row's shape against the spec — twelve
  lengths — so a file whose spec no longer matches its row is refused,
  not assembled out of bounds.

- *Python stacks specs `diagonal_relaxed`.* `ModelBank.last_row(spec=None,
  group=None)` returns `spec`, `group`, then the struct's fields unnested,
  one row per (spec, group); specs with different fields stack with nulls,
  which is what a table of many fits needs. An unseen group is an empty
  frame, not an error, so a glob over saved files can ask for one group
  everywhere. The CLI is untouched: it exposes neither `coef` nor an
  inspect facility, and the data is read through `ModelBank.load`.

**Task 35's decisions: the training data in the bank file, 2026-09-05.**

- *Fed after the clock commits, so a refused row feeds nothing.*
  `Stream::process_chunk` plans every row on a copy of the clock state
  and commits when the whole chunk is accepted; `DataSummary::feed_row`
  runs on the committed plans, one call per row, so a chunk that a
  backwards clock refuses under `on_clock_reset = "error"` leaves the
  summary where it was, as it leaves the models. No per-chunk clone of
  the summary is needed for that.

- *Undecayed, in row order, one Welford per column.* The summary answers
  "what was this trained on", and a decayed count would answer "what does
  it remember" — `n_eff` already does. Row-order Welford with a shared
  reciprocal per row (`inv = 1/rows_fed` when the column has no nulls, its
  own `1/count` otherwise — bit-identical to plain Welford, proved by a
  unit test) makes the numbers a function of the row sequence alone, so
  chunking cannot move a bit; that is asserted with `to_bits` in
  `tests/summary.rs` and `tests/test_summary.py`. A two-pass-per-chunk
  merge (Chan) would vectorise but is not chunk-invariant, so it was not
  used. Measured cost: about a nanosecond per input column per row (`sgd`
  at k = 20 from 95 to 112 ns/row, `kalman` unchanged within noise), always
  on; an opt-out is a candidate if a wide, cheap model needs it.

- *The models' notion of a usable value.* A column's `count` is the rows
  `online_core::INPUT_BOUND` admits — finite and at most `1e100` — and
  everything else is `null_count`, so `describe`'s counts are the counts
  the models saw. A `seqtest` comparison's one "target" is its own
  difference of residuals, an `ew_class` label is its class index with
  counts only, and an unsupervised model's mirrored target is dropped from
  `describe` (its features are the data). Events are the clock schedule's
  own: `session_changed` when both sessions are known and differ,
  `backwards` when the clock fell within a session (a new session owns its
  clock), `reset` when either policy reset the model.

- *Additive, so no format bump; a legacy file never grows one.*
  `StreamState.summary: Option<DataSummary>` with `#[serde(default)]`;
  `Stream::restore` keeps `None` for a file written before it, and `None`
  stays `None` through more rows and a re-save, so the frame reports nulls
  rather than a count that began partway. `rows_processed` and
  `last_clock`, which the stream always kept, are filled in either way
  (`state_v1.rs`, `state_schema2.rs` check the frozen fixtures for this).
  A saved summary is validated against its spec and its own arithmetic —
  column count, counts against `rows_seen`, finite moments, a range that
  is a range, events no more than rows — and refused with the spec named.

- *Testing the file, not the round trip.* Beyond every facet equal across
  save/load and a byte-for-byte re-save, `tests/summary.rs` truncates a
  real file at every length and flips bits through it, requiring each
  result to be refused or to load into a bank whose every accessor works
  (most flips land in f64 payload and load; the test asserts both
  outcomes occur). `state_schema3.rs` freezes a 0.2.0 file with a last
  row and a summary as a hex constant, pins its counts by hand, and
  requires the next layout to load it, continue it to the bit and, while
  the writer is unchanged, re-save it byte for byte.

- *Python stacks specs with a `spec` column.* `ModelBank.summary(spec=None,
  group=None)` and `describe(...)` return every spec by default with `spec`
  first; the schemas are fixed across specs so plain `concat` stacks them.
  An unseen group or a fresh bank is an empty frame of the right schema,
  not an error. The CLI is untouched, as for `last_row`.

**Task 36's decisions: `stats=[]` on `ew_cov`, 2026-09-05.**

- *One line.* The whole change to the model is the removal of the
  `validate()` clause that refused an empty `stats`. `n_outputs()` was
  already the sum of the statistic slots, the Mahalanobis quantiles and
  the PCA slots, so an empty list gives 0 + those; the stream already
  tolerated a model with zero slots (`nc = n_slots / m_targets`, floored
  at 1, with `m_targets = 0` for `ew_cov`); the output struct is then
  `{n_eff}` alone. Nothing in `online-polars` or `online-py` changed.

- *`None` and `[]` differ.* `stats=None` (Python) and a missing `stats`
  (TOML) still default to `["mean", "std", "corr"]`; only the explicit
  empty list means accumulate only. Changing the default would turn every
  existing `ew_cov` spec silent.

- *The extra slots do not need a statistic.* `pca=r` and
  `mahal_quantiles=[…]` add their outputs on `stats=[]` as on any list —
  they were never tied to an entry in it — except that `mahal_quantiles`
  without `"mahal"` still errors, because a quantile of a score that is
  not computed is a configuration mistake, not a request.

- *Same state to the bit.* The core test drives a bare and a full spec
  over the same weighted, clocked rows and asserts `cov()` equal, `n_eff`
  bit-equal, and the bare state restoring to itself; the Python test
  checks `gram()` of the bare spec against the full spec and against
  `np.cov(bias=True)`, then runs the empty list through every surface —
  `fit_predict`, chunked, `LazyFrame.online.fit_predict`, `po.run` with
  `save_state` and `ModelBank.load`, the expression plugin under its
  warning, and the CLI from TOML (`[specs.model] type = "ew_cov"`,
  `stats = []`).

**Task 37's decisions: `marginal`, 2026-09-05.**

- *`ew_cov`'s arithmetic, operation for operation.* `Marginal::learn` is
  the weighted-Welford mean form of `EwCov::update` — `a = λW/W'`,
  `b = w/W'`, `S' = a·S + a·b·dᵢdⱼ` with deviations from the *old* means,
  then `m += b·d` — applied to each pair's three moments, and `pair()`
  derives `corr` with the `ew_cov` reader's formula (`√var_x·√var_y`,
  clamp to ±1, NaN on a zero variance). That is what makes a pair's
  correlation bit-identical to an `ew_cov` over the two columns, which a
  core test and a Python test hold; a mathematically equal but
  differently ordered update would agree to 1e-15 and no further.

- *Per-target `min_periods` lives in the core config.* Every other model
  takes one threshold and the stream gates each target's fields; a
  `marginal` emits nothing per target, and its pairs are read from the
  state by whoever holds it, so the model gates `corr`/`beta`/`t`
  itself, per target, from `MarginalCfg.min_periods: Vec<f64>`
  (`spec.min_periods_per_target()`). The default is 3, not
  `k + intercept`: a pair is over two columns whatever the feature
  count, two rows give ±1 whatever the data, and three is the first
  count with content.

- *Not unsupervised.* A `marginal` has real targets — the columns its
  pairs are against — so it is not in `is_unsupervised()` (whose targets
  mirror `features[0]`), and the leak check (no column on both sides)
  applies. It is in `predicts_no_target()`, with `ew_class` and
  `seqtest`, so the residual switches are refused by name and no `resid`
  buffer is kept.

- *A null target ages its pairs.* `W_t ← λW_t`, `Q_t ← λ²Q_t`, moments
  untouched; the row is still learned by the other targets and by
  `w_sum` (the struct's `n_eff`, over every processed row whatever its
  targets). A zero-weight or zero-`W'` row is skipped per target, so a
  zero-weight first row leaves the pairs bit-equal to a stream without
  it (rule 9).

- *Kish's `n` for the t-statistic.* `t = corr·√((n_kish − 2)/(1 − corr²))`
  with `n_kish = W_t²/Q_t`, the count of equally weighted rows carrying
  the same information ((1+λ)/(1−λ) for unit weights, held by a test);
  documented as a scale for comparing pairs, not a p-value, since the
  rows are neither independent nor Gaussian. `n_kish ≤ 2`, a constant
  column and a perfect correlation give NaN in the core and null in the
  frame — the frame never carries NaN, as the output structs never do.

- *A long frame, sorted by group.* `Bank::marginal` returns one row per
  (group, instance, feature, target), groups in the order `gram()` and
  `summary()` use, targets outer and features inner in spec order; a
  group never seen is an empty frame of the same schema, a non-`marginal`
  spec a `ValueError` naming its kind (an `ew_cov`'s moments are
  `gram()`'s). The frame is built on the Rust side; no numpy.

- *The golden pipeline pins the state.* Every other model's golden
  values are struct fields at three rows; a `marginal`'s struct is
  `n_eff`, so `signature()` also reads `bank.marginal()` after the last
  row and pins `n_eff`, `n_kish`, `corr`, `beta` and `t` per (group,
  feature) — the same numbers on three operating systems.

- *A `ModelState` variant appended, no schema bump.* `ModelState::Marginal`
  is added at the end of the enum; the file format encodes a variant by
  name, so schema-3 files written before it still load (rule 5's
  appended-variant precedent, as for `seqtest`).

**Task 38's decisions: the complete Gram, 2026-09-05.**

- *`Option`, not a sentinel, and skipped when absent.* `EwCov::q_sum`,
  `EwRidge::tm` and `Lasso::tm` are `Option`, `#[serde(default,
  skip_serializing_if = "Option::is_none")]`. A state written before this
  task deserialises to `None`, keeps streaming, and re-saves **byte for
  byte** — the schema-3 fixture's `a_schema_3_state_re_saves_byte_identically`
  still passes, which is why the field is skipped rather than written as
  nil. No `SCHEMA_VERSION` bump: additive defaulted fields are the
  `rows_fed` precedent (rule 5), as an appended variant was for task 37.

- *A legacy state reports `None` for good, not from the resume point.* The
  tempting alternative — start accumulating `Q` on load — pairs a `Q` over
  the rows since the resume with a `W` over the whole stream, and reports an
  effective sample size too large by the length of the history. That is a
  wrong number where `None` is the true answer, and it would be wrong
  silently. `a_schema_3_state_reports_no_kish_size_and_no_target_moments`
  feeds twenty more rows and asserts it is still `None`.

- *The target moments ride the cross-moment update's own `a` and `b`.*
  `TargetMoments::learn` takes them rather than recomputing from `W_t`, so
  the arithmetic is `EwCov::update`'s operation for operation and a target's
  variance equals an `ew_cov` over that column to the bit — asserted in the
  core and again through `gram()` in Python. It is the same reasoning as
  task 37's, and the same test.

- *Kish's `n` is scale-free, and that is a feature.* `W` decays by `λ` and
  `Q` by `λ²`, so `W²/Q` is unchanged by pure decay: a target that stops
  arriving keeps the sample size its moments earned, and `target_weights` is
  what collapses. A first draft of the test asserted the opposite and was
  wrong, not the code. The docstring and the README now say which number
  means "how many rows" and which means "how recent".

- *A blend mixes `Q` by the moments' coefficients, not as a union.* The
  slow twin sees the *same* rows under a longer halflife, so summing the two
  `Q`s (the exact `Σw²` of the re-weighted union) reports a blend of a state
  with itself as twice as informative as the state. Mixing by `af`/`as`
  keeps the invariant the means, co-moments and weights already have: an
  identical twin blends to the identity. Documented on `TargetMoments::blend`
  as an approximation for overlapping histories, which it is.

- *Empty is not `None`.* `ew_cov` has no targets, so its three target lists
  are empty; `None` is reserved for "this state cannot tell you". Two
  different answers, two different values.

**Task 39's decisions: `po.gram`, 2026-09-05.**

- *The mapping names its own axes.* `gram()` gained `columns` and `targets`,
  derived on the Python side from the spec the bank already holds — no Rust
  change. Without them `subset(g, ["x0", "x2"])` and `solve(g, target="y")`
  could only take positions, and a caller would have to rebuild
  `["intercept"] + features` by hand and get `ew_cov`'s missing intercept
  wrong. `"intercept"` is the name `coef_index` already gives that slot.

- *Held against the models, not against a second copy of the formula.*
  `solve` is checked against `bank.coef()` over both standardization paths,
  three ridges and with/without an intercept; `lasso_path` against the
  `lasso` model's own path; `merge` against the Gram of the whole stream.
  A test that re-derived the same algebra in numpy would only prove I can
  write it twice.

- *Not bit-identical, and the docs say so.* E46 asked for "to the bit". The
  models factorize with `faer`'s Cholesky and numpy with LAPACK's LU; the
  measured gap is ~1e-16 relative on a well-conditioned system, and the
  assertions sit at 1e-12. Claiming equality would have been false, and the
  first draft of the docstring did claim it.

- *A grid of ridges rides one eigendecomposition where the penalty is
  uniform.* That is the standardized path always, and the unstandardized one
  without an intercept. With an intercept the model leaves that column
  unpenalized, so the penalty is not a multiple of the identity and each
  value costs a factorization. The centred reduction that would restore the
  shortcut is algebraically identical but rounds differently from the
  model, and fidelity to the model is worth more here than the speed.

- *`merge` is for parts that share a weighting.* Shards of a pass, groups
  being pooled, workers. Two halves of a decayed stream in time order are
  *not* the stream: each part's weights are relative to its own last row.
  The docstring gives the rescaling (`n_eff * lam**dt`, `Q * lam**(2*dt)`,
  the means and co-moments untouched), and a test does exactly that and
  recovers the whole — the caveat is held by a test, not only by prose.

- *`coef_stats` divides by `n_kish`.* A weighted stream's `n_eff` is not a
  count, and using it would report standard errors too small by however
  unequal the weights are. A test puts ten times the weight on every row and
  asserts the errors do not move. The intercept's standard error is `nan`:
  it depends on the design's centring, which the Gram has already absorbed.

- *Two disagreements that are not bugs, both now tests.* `bank.coef()` is as
  of the model's last *solve* (its `solve_every` / `max_rows_between_solves`
  schedule), while `gram()` is as of the last row — the first draft of the
  test read that gap as a defect in `solve`. And the tests use `lam=1.0`, no
  decay at all, wherever they compare a merge against a whole: `halflife=1e12`
  is nearly but not exactly no decay, and at 2400 rows the difference shows
  at 1e-6.

- *`penalty_weights` is the one thing the models do not also do.* Offline the
  path is re-walked rather than carried, so a per-feature penalty costs
  nothing; online it would be another vector in the state for a knob nobody
  has asked for. Listed in E46, added here, and not added to the model.

**Task 40's decisions: `label_delay`, 2026-09-05.**

- *Two virtual rows, not two code paths.* A delayed row is scored where it
  sits and stepped later, so the plan list is rewritten into
  (release, …, score this row) and `RowPlan` gained `emit` and `learn`.
  `run_instance` already guarded every diagnostic fold behind `learn` for
  the scoring path, so the fold deferral came almost free; the new work is
  guarding the output writes behind `emit`. A spec without a delay pays two
  predictable branches a row and nothing else.

- *Maturity is measured on the model's clock.* Not the raw column: the
  delta the models are given, capped by `max_dclock`, with skipped rows'
  time folded in. That is the only clock that is chunk-invariant, and it
  makes `label_delay` mean the same thing as `halflife` does. With no
  `clock` column it is one unit per accepted row, so `label_delay = 20` is
  twenty rows. Each buffered row counts down by every later row's delta
  rather than holding an absolute deadline, so a long stream cannot lose
  precision in a running clock.

- *A released row replays the delta it arrived with.* The models therefore
  see exactly the gap sequence they would have seen without the delay, just
  later — which is why the state at the end is bit-for-bit a plain bank fed
  only the matured rows, and a test says so.

- *A reset drops the buffer and applies at its row; a session change
  releases it.* A clock reset means "this state is no longer about this
  stream", so deferring it would leave the old model predicting the new
  regime; the rows waiting to teach that state go with it. A session change
  keeps the model, but one session's clock does not measure time in the
  next, so a row still waiting would wait for a deadline that never comes:
  it is released in order at the boundary. Both were arrived at by writing
  the test and watching the first version get 66.9 where 51.8 was right.

- *The doubled stream is the oracle, and it disagrees in exactly one
  place.* `po.prep.embargo` builds E47's recipe and the native path matches
  it field by field, bit for bit, at three delays — once `embargo` was
  fixed to put the lesson *before* the prediction at a tied clock, which is
  what "the clock has reached `t + delay`" means. The exception found by
  the test: `resid_quantiles`, `emit_autocorr` and `emit_drift` take no row
  weight (a P² estimator counts samples), so a zero-weight predict row
  feeds them as much as its learn copy and every residual lands twice. The
  native path feeds them once. That is the better answer, and it is now a
  test and a line in the README rather than a surprise.

- *A skipped row agrees to a ulp, not a bit.* A null feature's clock time
  folds into the next *accepted* row, and the doubled stream's accepted
  rows are not the same rows, so the two partition the same elapsed time
  differently. The decay factors multiply to the same number to within
  rounding, which is all that can be asked.

- *`SCHEMA_VERSION` 3 → 4.* Not because a model's state moved — none did —
  but because every spec's bytes moved: the spec each bank file carries
  gained `label_delay`. Rule 5 asks to be told about a layout change, and
  `state_schema3.rs` was written to become the upgrade test the moment the
  writer changed, which is exactly what happened. `MIN_SCHEMA_VERSION` is
  still 1 and every older fixture still loads. Task 38's additions rode
  into schema 3 without a bump because `skip_serializing_if` kept an old
  file's bytes where they were; a spec field cannot do that without making
  `bank.specs` disagree with the dict it was built from.

- *A frozen schema-4 fixture is owed once 41–43 have landed*, so it is
  frozen once rather than three times. Task 44 does it.

**Task 41's decisions: E48 measured and rejected, 2026-09-05.**

- *The premise was wrong, and measuring is what showed it.* E48 said
  mirroring the upper triangle is "bit-identical (the products commute in
  IEEE)". The products do commute; the association does not.
  `c[i][j] = a*b*d_i*d_j` is `((a·b)·d_i)·d_j` and the transposed entry is
  `((a·b)·d_j)·d_i`, whose intermediates round differently — so the matrix
  the library has always kept is symmetric to about 1e-16 and *not* to the
  bit, and mirroring moves every golden value. Writing it as
  `(a·b)·(d_i·d_j)` would make it exactly symmetric, and is itself a
  different rounding from today's. Either way E48's "the goldens, unchanged"
  cannot hold.

- *And it is slower, by 49% to 107%.* The mirror store `c[j*k+i]` walks a
  new cache line for every `j`: the loop touches the whole matrix twice
  instead of once, defeats the prefetcher and cannot vectorise. Halving the
  flops and doubling the traffic is a pessimisation at every width where a
  triangle would be worth having. Five variants, six widths, two runs, each
  checked bit-for-bit before being timed (docs/PERFORMANCE.md §14).

- *What shipped instead is on the same loop and is bit-identical.* The
  deviations are computed once into a row scratch — the old form recomputed
  `x[j] - m[j]` once per *row of the matrix*, `k²` subtractions where `k`
  will do — and the inner loop runs over `c`'s row slice zipped with that
  scratch, so `c`, `x` and `m` are not indexed by `j` and the bounds checks
  that kept it scalar are gone. −14% at k = 4, −63% at 16, −45% at 64,
  −22% to −27% from 200 up, and the goldens do not move.

- *The scratch is a buffer, not state.* `#[serde(skip)]` and a `PartialEq`
  that always returns true, so a state file carries none of it and a
  round-tripped accumulator still compares equal to the one that wrote it.
  It refills itself on the first row after a load; a test asserts both.

- *The rejection is kept as a test, not only as prose.*
  `the_two_triangles_are_equal_but_not_bit_equal` fails if the two ever
  become bit-equal, and its message points at §14 — so the next person to
  reach for the shortcut finds out why it was not taken, in the place they
  would reach for it.

**Task 42's decisions: mergeable evaluation sums, 2026-09-05.**

- *Centred sums, not the raw ones E49 asked for.* The row lists `Σy`, `Σy²`,
  `Σŷ²`, `Σyŷ`, which make `merge_sums` a plain addition. They also lose the
  variance entirely at a large offset: a target around 1e8 with unit spread
  has `var / E[y²] ≈ 1e-16`, and `Σy² − (Σy)²/n` has nothing left after the
  subtraction. That is E11b's finding, in the same library, and taking the
  raw form here would have re-made the mistake the accumulators exist to
  avoid. The stored fields are the weighted means plus centred second
  moments, and the merge pays a parallel-axis term for them — a multiply per
  key per part. Still ten doubles, still O(state).

- *An n-way merge, not a fold of pairs.* `merge_sums(*parts)` concatenates
  and reduces in one `group_by`: the pooled mean is the weight-weighted one,
  and each part's centred sum picks up `w_g·(mean_g − mean)²`. Every part
  enters as a sum and never as a difference of running totals, so a hundred
  parts lose no more than two — asserted at 2, 5 and 97.

- *Held against `metrics`, not against a rewrite of it.* The test is
  `from_sums(sums(df)) == metrics(df)`, column for column, grouped and
  ungrouped. A test that re-derived R² in numpy would only show I can write
  the formula twice.

- *`rmse` beside `mse`, and nulls where a metric is undefined.* E49 named
  `rmse`; `metrics` reports `mse`. Both are here, so the two frames line up
  and the extra column is the one the row asked for. A key whose target
  never varied gets `null` for `r2` and `ic` rather than an infinity that
  would read as a number.

**Task 43's decisions: a quiet run, and a spec that leaves things out,
2026-09-05.**

- *No output is a destination, not a missing one.* `Output::Discard` sits
  beside `File` and `Batches`, so `deliver` drops the frame and no writer
  thread is spawned at all. The empty-input path — which exists to write a
  valid empty frame of the right schema — is skipped too: collecting the
  plan only to drop it would be a read of the source for no one.

- *`save_state` is required in its place.* A run that writes nothing and
  saves nothing has done nothing, and silently succeeding at it is worse
  than refusing. The message names both ways out.

- *`output=None` already meant this; `no_output=True` is for the other
  case.* Leaving `output` out of both the call and the config is E50's
  `po.run(output=None, save_state=)` and needed no sentinel. The flag is
  for clearing an `output` a config carries, and mirrors the CLI's
  `--no-output`, which `conflicts_with` `--output`.

- *Filling happens where a spec enters, and on both sides of the load
  check.* `Spec::fill_defaults` is called by `Bank::new` (so every surface
  that runs a spec, and every state file, carries the filled form), by
  `RunConfig::fill_defaults` (the CLI and `po.run`, before validation), and
  on both the expected and the saved specs in `Bank::load_bytes` — the last
  because a file written before the filling existed carries the unfilled
  form, and refusing to load it would be a regression for the sake of a
  field that means the same either way.

- *`drift_action` came out of the byte-identity requirement.* E53 asks for
  a TOML spec and a Python spec to be byte-identical. `targets` was one
  difference; `drift_action` was the other — the builders write `"flag"`
  and a TOML author would not, and `None` and `"flag"` already meant the
  same thing to every reader. The test that pins this compares the two
  state files, not the two dicts, so any third such field will be found the
  same way.

- *A better error for the case that is still wrong.* An unsupervised spec
  with no features and no targets used to say "targets must be non-empty",
  which points at the field the author was right to omit. It now says
  features must be non-empty, and why.

**Task 44's decisions: the schema-4 fixture, 2026-09-05.**

- *Frozen after the layout stopped moving, not when it first moved.* Tasks
  40–43 each touched what a bank file holds; freezing after 40 would have
  meant regenerating three times, and a fixture regenerated is a fixture
  that proves nothing.

- *Four specs, two of them there for the new fields.* `d` is an `ew_ridge`
  with `label_delay = 8`, so the file carries a **non-empty** pending
  buffer — the part of schema 4 that a converter would be most likely to
  drop. `u` is an `ew_cov` written with no `targets` at all, so the file
  carries the ones `fill_defaults` put there, and a test asserts both that
  and the filled `drift_action`. The other two are schema 3's, so the two
  fixtures are comparable.

- *The other three fixtures stay exactly as they are.* Their module docs
  already say so, and `state_schema3.rs`'s byte-identity test has taken its
  upgrade branch — which is what it was written to do.

**Preparing E54–E64 for implementation (tasks 45–56), 2026-09-05.** The
eleven items of ENHANCEMENTS §10 were read against the code they land in —
`model.rs`, `clock.rs`, `ewcov.rs`, `ewdiag.rs`, `ewclass.rs`,
`cluster/kmeans.rs`, `stats.rs`, `drift.rs`; `stream.rs`, `bank.rs`,
`spec.rs`, `runner.rs`, `summary.rs`; `_bank.py`, `_frame.py`, `_runner.py`,
`_spec.py`, `_kwargs.py`, `prep.py`, `gram.py`; `EXTENDING.md` — and each
became a task above. What follows is what the implementer needs beyond the
§10 row: the decisions the row left open, resolved; the places the row was
wrong about the code, corrected; and the exact seams each task touches.
Where a choice is genuinely free it is marked *implementer's call*; nothing
else is. The points this block leaves to be confirmed from the papers —
the log base and `D̂` of the constancy monitor and its size/power tables,
Higham's example matrices, the Ledoit–Wolf intensity formulae, the
pre-averaging constants, the kernel bandwidth rule, the Epps-inversion
weights, the LDECO conventions — are answered in `docs/ANSWERS-E54-E64.md`,
which also says which items it could *not* verify.

*Batch-wide.*

- *Order.* 45 → 46 → 47 → 48 → 49 → 50 → 51 → 52 → 53 → 54 → 55 → 56, as
  §10 gives it, with 47 inserted before the first model that needs it and 52
  (`po.sim`) before the three detectors whose experiments need a stream with
  a known truth. 47 is small and must land before 48; 49 is independent of
  everything and can be built at any point.

- *One schema bump for the batch.* Task 45 adds a field to `Spec`, and every
  bank file carries its specs, so every file's bytes move: `SCHEMA_VERSION`
  4 → 5 there, with a loader that is the existing one (`#[serde(default)]`
  fields; a schema-4 file loads, continues to the bit and re-saves as 5),
  and the `lib.rs` history entry. Everything after 45 rides on 5 without a
  further bump: appended `ModelState` variants (`Deco`, `Rcov`, `Hmm`,
  `CorrChange`, `Bocpd`), `Option` fields with `skip_serializing_if` on
  `EwCovModel` and `StreamState`, and new map keys on `BankFile` with
  `#[serde(default)]` (the file is written with `to_vec_named`, so a key is
  additive). Task 56 freezes `state_schema5.rs` once the layout has stopped
  moving — Task 44's reason: a fixture regenerated proves nothing.

- *The model shape.* 46, 50, 53, 54 and 55 are `ew_cov`'s shape: `n_targets()
  = 0`, `is_unsupervised` and `predicts_no_target` both true, `fill_defaults`
  writing `targets = [features[0]]`, the residual-side flags (`emit_sigma`,
  `emit_resid_z`, `emit_metrics`, `conformal`, `resid_quantiles`,
  `emit_autocorr`, `emit_drift`, `emit_selected`, `emit_averaged`) refused
  by name with `ew_cov`'s messages, `label_delay` refused (nothing to hold
  back), the leak check exempted where `ew_cov` is (grep `ew_cov` in
  `tests/` and `scripts/leakcheck.sh`), non-`f64` outputs through the
  `Source` variants that exist (`Cluster`, `Flag`, `Id`). Each needs ≥ 2
  features except `bocpd`; refuse fewer by name.

- *The clock-rescaling test, once per new model.* `Decay::factor` is
  `exp2(−d/h)`; scaling the clock column and the halflife by the same power
  of two leaves `d/h` bit-identical, so the output must be bit-identical at
  `c = 2^k` and equal to `1e-12` at `c = 3`. Put it in each model's
  `tests/test_<model>.py`; it is the "clocks are numbers" property §10 asks
  for, stated so it can fail.

- *Where things go.* Core: `crates/online-core/src/{deco,rcov,hmm,corrchange,
  bocpd}.rs`. Stream layer: `refresh.rs` in `online-polars` (task 49), the
  close path in `bank.rs`/`stream.rs` (45). Python: `corr.py`, `sim.py`, a
  second function in `prep.py`, `from_row` in `gram.py`. Docs: a `docs/
  REGIMES.md` for the detector experiments (53, 54), generated by
  `scripts/regime_experiments.py` and committed, *not* regenerated by the
  gate — the `docs/CLUSTERING.md` §7 precedent, chosen over
  `docs/VALIDATION.md` because `test_validation_doc.py` regenerates that file
  on every run and a Monte-Carlo experiment does not belong in a 0.4 s test.

- *Dependencies.* None new in Rust (`faer` already does the eigenwork;
  nothing is statically linked that was not). `numpy` only in Python, and
  only in `corr.py`, `sim.py` and the tests — the `gram.py` import guard
  with the install hint. No scipy, no scikit-learn.

- *Every task ends the same way.* `scripts/gate.sh` unpiped; `api_surface.txt`
  regenerated with `UPDATE_API_SURFACE=1` and the diff read; Sphinx `-W`
  over `docs/reference` (new modules need a page there); README `### <name>`
  heading for a model, whose code blocks the tests execute; CHANGELOG
  `[Unreleased]`; the §10 row's status; one commit per task with its number.

*Task 45 — closed-group emission (E54).*

- *Native order, not `GroupKey` order.* `GroupKey` holds integer keys as
  decimal strings and its derived `Ord` is lexicographic, so `"10" < "9"`;
  a monotone close on that order would close group 10 before 9 arrived.
  Add `fn key_cmp(a: &GroupKey, b: &GroupKey, integer: bool) -> Ordering`
  in `bank.rs`: integer compare (`i128` parse, the keys are what
  `integer_groups` formatted) when the group column's dtype is integer,
  bytewise on the string otherwise (String and Categorical, which `extract`
  already renders as strings). Any other dtype under `"monotone"` is refused
  by name at the first chunk (`group_indices` knows the dtype). A null key
  under `"monotone"` is refused naming the row (a null has no order); under
  `"session"` it is an ordinary key.

- *The pre-check is what makes interleaving an error.* `group_indices`
  partitions a chunk by first appearance and would silently gather the rows
  of `A, B, A` into one run per key. So, before any model is touched and
  beside `check_clock` (the refusal must leave the bank as it was; the
  `forget` path takes back the streams the chunk materialised), walk the
  key column in row order: it must be non-decreasing under `key_cmp`, and
  its first key `≥` the spec's high-water mark. The first offending row is
  named with both keys: `row 1234: group 7 after group 9 — group_close =
  "monotone" needs keys in non-decreasing order`. `label_delay` with
  `group_close` is refused by name at `validate`: a closed group cannot
  release the rows it is holding, and the closed row would then differ from
  the `gram()` a driver reads — the acceptance equality would be false by
  construction.

- *The monotone close batch.* After both phases and `rows_fed`, per spec
  with `"monotone"`: `max_key` = the chunk's largest key; every stream whose
  key is `< max_key` closes, in `key_cmp` order; then `high_water =
  Some(max_key)`. Content is chunk-invariant because every row of a closing
  group precedes the first row of a greater key, whatever the chunking;
  order is chunk-invariant because a batch closes in key order and a group
  never closes before a smaller one. The high-water mark is per spec, saved
  as a `GroupKey` string in a new `BankFile` key (`#[serde(default,
  skip_serializing_if = ...)]`, written only when some spec has one), and
  compared under the *current* column dtype on load — a stored key that no
  longer parses under it is an error naming both.

- *The session close is a split of the run.* `Stream::process_chunk` learns
  whether the spec closes on session (`ClockCfg` or a parallel flag; the
  stream already builds every `RowPlan` in pass 1). When a plan carries
  `session_changed` at `ri`, the run is processed as two segments: rows
  `< ri` as today, then `close_into(&mut self.closed)` — one `ClosedRow` per
  instance from the stream as it stands, then `*self = Stream::new(spec)` so
  the new session's first row is a first row (Δ = 0, fresh clock, fresh
  summary) — then rows `≥ ri`. Chunk-invariant because a session boundary is
  a property of two consecutive accepted rows. `Stream::closed` is
  transient (`#[serde(skip)]`) and is drained by the bank at the end of
  every `fit_predict`. `"session"` requires `session`; with `session_gap` it
  is refused by name (two prescriptions for one event — the close *is* the
  reset, with an emission). Sessions are stored hashed (`prev_session:
  Option<u64>`), so the closed row's `session` value needs the last session
  *value* kept: a `StreamState` field `#[serde(default, skip_serializing_if
  = "Option::is_none")] last_session: Option<String>`, written only under
  `"session"` so no other spec's bytes move.

- *Queue order.* One `Bank::closed: Vec<ClosedRow>`, appended per chunk in
  the order `(global index of the closing row, key_cmp)`: for a session
  close the closing row is the first row of the new session, for a monotone
  close it is the first row of the greater key; the global index is
  `rows_fed` before the chunk plus the row's position in it. Streams are
  processed in parallel, so this sort is what makes the queue's order a
  function of row order alone.

- *`closed_groups(spec=None, *, drop=True)`.* Returns the queue as one long
  frame, narrowed by `spec`; `drop=True` removes what it returned from the
  queue, `drop=False` peeks. Streams are dropped at close time, never at
  the call — a bank's memory must not depend on the caller polling. The
  undrained queue **is** saved (`BankFile` key, skipped when empty): a
  driver that saves between chunks without draining would otherwise lose
  rows silently. `Bank::predict` never closes anything; `closed_groups=`
  with `predict` is refused in `RunConfig::validate`, the CLI and
  `po.run`.

- *The row.* One frame schema per bank: the common columns always, a
  kind's block when any spec of that kind in the bank has `group_close`,
  null on other kinds' rows. Common: `spec` str, `group` str (null for a
  null key), `instance` str (the decay suffix, `""` for one instance),
  `session` str (the closed span's session value under `"session"`, null
  under `"monotone"`), `n_eff` f64, `n_kish` f64 (null before a weighted
  row), `rows_fed` i64, `rows_learned` i64, `clock_min` f64, `clock_max`
  f64 (null on a row-count clock). The last four are `DataSummary`'s
  fields, already computed and saved; §10's `clock_first`/`clock_last` are
  renamed to them rather than defined a second time — the two differ only
  when a clock ran backwards inside a group, and the summary's names are
  what the state already reports. Gram block (`ew_ridge`, `lasso`,
  `ew_cov`): `columns` list[str], `means` list[f64], `comoments` list[f64]
  (vech of the upper triangle with the diagonal, row-major, `k(k+1)/2`),
  `targets` list[str], `target_means`, `target_vars`, `target_n_kish`
  list[f64], `cross_moments` list[f64] (targets × k, row-major), `lags`
  list[i64] and `lag_comoments` list[f64] (task 48; `L × k × k`
  row-major; null without `lags`). `coef` list[f64] for every kind that
  reports one, laid out as `coef()` reports it — **as of the last solve**,
  the meaning `coef()` already has; the exact solve on the closed Gram is
  `po.gram.solve(po.gram.from_row(row))`, one line, and the docstring says
  so. `eig_vals` list[f64] (r, descending) and `eig_vecs` list[f64]
  (`r × k` row-major) for `ew_cov(pca=r)`: `Pca::of` on the row's own
  `comoments` with `prev` = the last closed row's vectors for the same
  (spec, instance), kept in a `BankFile` map key so continuity survives a
  save/load; `Pca`'s own first-refresh sign rule. Marginal block: every
  column of `marginal()` as a list in `marginal()` order, prefixed `pair_`.
  `rcov` block: task 50's. The other kinds (`rls`, `kalman`, `holt`,
  `kmeans`, `micro`, `ew_class`, `seqtest`, `sgd`, `pa`, `ftrl`, `robust`)
  emit the common columns and `coef` where they have one; `group_close` is
  still worth having on them for the memory bound.

- *One builder for the row and for `gram()`.* Lift the `match model` in
  `Bank::gram` into `fn gram_of(key, label, model) -> Option<Gram>` and call
  it from both; the acceptance equality then holds by construction and the
  test is the tripwire that keeps it so.

- *Sidecar.* `RunConfig.closed_groups: Option<PathBuf>`, format from the
  extension as `output`'s is, CSV flattening as `output`'s; refused with
  `predict`, and refused when no spec has `group_close` ("closed_groups
  names a file but no spec closes groups" — the file would be empty by
  construction). Written atomically through a second writer thread on the
  `write_file` path — temp sibling, rename, published only when the run
  completes, before `save_state`. It was first written once, at the end
  (E35's rule, one code path); since the code review's P5 (2026-09-15) the
  bank is drained after every chunk and each drain goes to that writer as
  it comes, so the rows no longer wait in the bank for the run.
  `output=None` with `closed_groups=` is
  the accumulate-only pass. A run in which nothing closed writes an empty
  frame with the schema, as the empty-output rule does. The IO plugin
  (`lf.online.fit_predict(closed_groups=path)`) drains `closed_groups()`
  after every chunk into a list and writes the one file when the source
  reaches the last row, under `save_state`'s rules and caveats
  (`_frame.py`, `docs/STATE-WORKFLOW.md`). CLI: `--closed-groups PATH`.

- *Python.* `group_close` is `CommonKwargs`-only, beside `group`; the
  expression namespace has no groups and does not take it (a test says so).
  `_common` writes `None` when unset, `fill_defaults` writes nothing, so a
  TOML spec and a Python spec serialise to the same bytes.
  `po.gram.from_row(row)` takes a one-row frame or a dict, expands the
  `vech` and the row-major lists, and returns a `gram()` dict that
  round-trips through `solve`, `correlation`, `subset`, `coef_stats`; the
  test is `from_row(closed_row) == bank.gram(...)[i]` field for field.

- *Tests, beyond §10's list.* `A, B, A` in one chunk refused naming row 2;
  `"10"` after `"9"` accepted on an integer column and closed in the order
  9, 10; a Categorical key closed bytewise; save/load mid-stream then a
  smaller key refused from the loaded bank; `closed_groups(drop=False)`
  twice returns the same frame; the union schema with an `ew_cov` and a
  `marginal` spec in one bank; the sidecar equal to `pl.concat` of the
  driver's `closed_groups()` frames; `predict=True` with `closed_groups`
  refused in all three entry points.

*Task 45 as built, 2026-09-06.* The design above stood; five things were
decided at the keyboard and are here so the next reader does not re-derive
them.

- *`group_close = "session"` replaces `session_gap`, it does not sit beside
  it.* `clock_cfg()` has always required `session_gap` whenever `session` is
  given -- a session change has to say what the delta is -- and the block
  above refuses the pair. Both cannot hold, so the close is now itself the
  prescription: `session_gap` is required for a session column *unless* the
  spec closes on it, where the answer is "emit the span and start over".

- *The session split lives in `process`, not in `process_chunk`.* The bank
  already cuts a stream's rows into cache-sized runs, each with its own
  `ChunkOut`; a session segment is one more cut of the same kind, so
  `Stream::process_chunk` is untouched and the buffers, the assembly and the
  `last`-row coefficient report all keep working as they did. The boundaries
  come from `ClockState::prev_session()` and the session hashes the columns
  already carry, walked in row order -- a property of two consecutive rows,
  which is what makes the split chunk-invariant.

- *The PCA is computed when the row is queued, not when the group closes.*
  Streams close in parallel, and the loadings are signed for continuity with
  the previous closed row of the same (spec, instance) -- so "previous" has
  to mean previous in the queue, or the signs would depend on which thread
  got there first. `Bank::queue_closed` sorts by `(closing row, spec, key,
  instance)` and then fills `eig_vals`/`eig_vecs` in that order.

- *`from_row` mirrors the packed half, and the mirror is not bit-exact.* The
  row carries `vech` of the upper triangle, half the bytes; `po.gram.from_row`
  reflects it. The reflected lower triangle differs from `gram()`'s in the
  last bit or so, because the accumulator adds the same two products in the
  opposite order for `C[i][j]` and `C[j][i]` and IEEE multiplication of a
  three-term product does not commute -- E48's finding, in the same library.
  The acceptance therefore holds bit for bit on the half the row carries and
  to `1e-15` on the reflection, and the docstring says so. Storing the full
  `k*k` would make the mirror exact at twice the size; the half was chosen.

- *`rows_fed` and `rows_learned` are nullable.* They come from the stream's
  `DataSummary`, which is `None` for a stream restored from a file written
  before task 35. Null is the true answer there; a zero would read as a
  count.

*Task 46 — `deco` (E55).*

- *Standardisation is `kalman(standardize=True)`'s.* One `EwDiag` over the
  features, decayed on the model's clock; the row is standardised against
  the **pre-row** means and variances (`r_i = (x_i − m_i)/√v_i`), then the
  diag is updated. `u` is NaN until every feature has a positive pre-row
  variance; a zero-variance feature thereafter is NaN for that row (no
  substitution of 1, unlike `kalman`, whose coefficient scale is what that
  rule protects).

- *`u` and its block form.* With `S₁ = Σᵢ rᵢ`, `S₂ = Σᵢ rᵢ²` over `n`
  features: `u = (S₁² − S₂)/((n − 1)·S₂)`, Lemma 2.3's
  `Σ_{i≠j} rᵢrⱼ/((n−1)Σrᵢ²)`, in `(−1/(n−1), 1)` except at `S₂ = 0`
  (NaN). Blocks: `u` is the ratio of the mean off-diagonal product to the
  mean squared entry, so within block `A`: `u_A = (S₁ₐ² − S₂ₐ)/((n_A −
  1)·S₂ₐ)`, and between `A` and `B`: `u_AB = S₁ₐ·S₁_B / √(n_A·n_B·S₂ₐ·S₂_B)`
  — the same ratio with the cross term `(Σ_A r)(Σ_B r)/(n_A n_B)` over the
  geometric mean of the two blocks' mean squares, in `[−1, 1]` by
  Cauchy–Schwarz. A block of one feature has no within term: refuse blocks
  of size `< 2` by name. Every feature must be in exactly one block.
  ANSWERS confirms the closed forms against Engle–Kelly (2012), and two
  things the paper says that the docstring repeats: `uₜ` is a downward
  biased estimate of the equicorrelation (their §2.2), and an alternative
  `u^var = 1 − (1/(n−1))·Σᵢ(rᵢ − r̄)²` exists — the variance-of-standardised
  returns form — which this task does **not** offer (one estimator, the
  Lemma 2.3 one, until a use asks for the other).

- *Dynamics.* `"ew"`: `rho' = a·rho + b·u` with `a = λW/W'`, `b = w/W'`,
  `W' = λW + w` — `EwCov::update`'s mean form; at `W = 0` that is `rho = u`;
  `W' = 0` (a zero-weight first row) skips. `rho` is NaN before its first
  update. `"linear"` (LDECO eq. 21 with correlation targeting): `rho' = (1 −
  α − β)·rho_bar' + α·u + β·rho`, where `rho_bar` is the `"ew"` recursion run
  alongside as the target and `rho` starts at the first `u`; `α, β ≥ 0`,
  `α + β < 1`, refused otherwise and refused under `"ew"`. `halflife`/`lam`
  are required under both (the standardiser and `rho_bar` decay on them).
  Reported `rho` is the level **before** the row. Two departures from the
  paper, recorded here so nobody "fixes" them back (ANSWERS, verified
  against Engle–Kelly): eq. 21 has a free intercept `ω`, and the paper
  applies correlation targeting to the DECO-DCC `Q` recursion (its eq. 5),
  not to the linear one — writing the intercept as `(1 − α − β)·rho_bar`
  is *our* reparameterisation, chosen because it removes a free parameter
  that has no sample to be fitted on in a streaming model; and the paper
  lets `α + β` sit slightly above 1 under numerical bounds, where our `α +
  β < 1` is the stricter, stationary choice. The docstring says both.

- *`loglik`.* The row's Gaussian log-density under the pre-row `rho`, in
  standardised coordinates: `−½·[n·ln 2π + ln det R + r'R⁻¹r]` with `det R
  = (1−ρ)^{n−1}(1 + (n−1)ρ)` and `r'R⁻¹r = (S₂ − ρ·S₁²/(1 + (n−1)ρ))/(1−ρ)`.
  `ρ` is clamped into `(−1/(n−1) + 1e-9, 1 − 1e-9)` **for the density
  only**; the emitted `rho` is not clamped. With blocks, `R = D + U·P·U'`
  (`D = diag(1 − ρ_{A(i)A(i)})`, `U` the `n × K` block indicator, `P` the
  `K × K` matrix of block correlations): Woodbury and the matrix
  determinant lemma give `R⁻¹r` and `ln det R` from a `K × K` solve, `O(n +
  K³)`; the test is a dense `numpy.linalg.slogdet`/`solve`. NaN whenever
  `rho` is.

- *Outputs and state.* `u`, `rho`, `loglik`, `n_eff`; with blocks `u_<A>`
  per block then `u_<A>_<B>` per pair in `blocks` order, `rho_*` likewise,
  one `loglik`. `coef` = the `rho` values in that order, one slot per
  value, term `rho` (`coef_fields` gets a `Deco` arm). `ModelState::Deco {
  diag, rho: Vec<f64>, rho_bar: Vec<f64>, w_sum, q_sum, dynamics, alpha,
  beta, blocks }`, appended. `n_outputs()` = `3` unblocked, `2·(K +
  K(K−1)/2) + 1` blocked.

- *Tests.* §10's, plus: `rho` under `"ew"` with one pair against
  `ew_cov(stats=["corr"])` on the columns standardised by the same
  `EwDiag` **to the bit** — same `a`, `b`, same order of operations, or the
  test says which differs; the clock-rescaling test; `MINIMAL["deco"] =
  {"features": ["x0", "x1"]}` in `test_model_registry`.

*Task 46 as built, 2026-09-06.* The design stood; four corrections and one
departure, all measured.

- **The plan's `ew_cov` equality is false.** It said a one-pair `"ew"`
  `deco` equals `ew_cov(stats = ["corr"])` on the same standardised columns
  *to the bit*. It does not, and not by a little: on a two-column stream
  with a true correlation of 0.6, `rho` settles near **0.32** and `corr` near
  **0.60**. The EW mean of a ratio is not the ratio of EW means, and `u` is
  the downward biased estimator Engle & Kelly themselves flag (`E[u]` is
  0.20 at a true 0.30 with six columns, 0.60 at a true 0.80 — measured by
  Monte Carlo, and reproduced by the model to 0.02). What *is* bit-identical,
  and is what the test asserts, is `rho` against an `ew_cov(stats = ["mean"])`
  over the `u` sequence at the same halflife.

- *Which is why the recursion is written `ρ + b·(u − ρ)`* rather than the
  algebraically equal `a·ρ + b·u`: that is the order `EwCov::update` writes
  its mean in, and the two differ in the last bit. Written the other way the
  agreement above was 7e-16, not exact.

- *One code path for blocked and unblocked.* The unblocked model is one block
  holding every feature, and the `K × K` Woodbury form reduces to the closed
  `(1−ρ)^{n−1}(1+(n−1)ρ)` at `K = 1` — a unit test asserts that reduction, so
  there is one implementation of the density and two tests of it. A single
  *named* block is the unblocked model under a name: same three slots,
  labelled `u_<A>`, `rho_<A>`, `loglik`.

- *One shared weight for the level.* `rho`'s accumulated weight is not the
  standardiser's: a row whose `u` is not finite (a zero-variance feature)
  teaches the level nothing and must not decay it either. All the values of
  a blocked model advance together, on that one weight; a row where any
  block's `u` is undefined moves none of them.

- **Departure: `label_delay` is accepted, not refused.** The batch-wide rule
  says the new no-target models refuse it, "nothing to hold back". `ew_cov`,
  whose shape `deco` copies, accepts it today, and it does hold something
  back — the accumulator update. Refusing it for one and not the other would
  be a surprise, so `deco` accepts it and a test says so. The same reasoning
  will apply to 50, 53, 54 and 55 unless one of them has a real reason to
  differ.

*Task 47 — the ring-clearing signal.*

- *What is missing.* `ClockAdvance` has `d_clock`, `reset`, `accepted`,
  `backwards`, `session_changed` — nothing says the cap was hit — and a
  model is told nothing about a session change unless it is `EwRidge`
  (`AnyModel::blend_toward_long_run`). §10's rule "cleared by a
  `max_dclock` breach or a session change" therefore has no carrier.

- *The carrier.* `ClockAdvance::capped: bool`, true iff `max_dclock` is set
  and the gap-adjusted raw delta (pending time folded in) exceeded it, so
  `d_clock == max_dclock` was substituted; set in `ClockState::advance`,
  never on a row-count clock. `RowPlan::capped` copies it (false on a
  replayed `label_delay` row, as `session_changed` is). `OnlineModel::
  clear_lags(&mut self) {}` — a default no-op every model inherits; the
  stream calls `inst.model.clear_lags()` in pass 2 for `plan.session_changed
  || plan.capped` when `!plan.reset` (a reset already rebuilds the instance,
  ring included). Not called from `predict_chunk`, which mutates nothing.

- *The rule, stated once.* Row-lagged state is a function of the learned
  rows of the group since the last reset, session change or capped gap, in
  order; it is chunk-invariant because all three are properties of the row
  sequence. EXTENDING.md gains the step: "if the model keeps row-lagged
  state, override `clear_lags` and add the three-event test"; the
  `model_contract.rs` probe gains a `clear_lags` call on every model,
  asserting `predict` unchanged for the models that do not override it.

*Task 47 as built, 2026-09-06.* Three decisions worth keeping.

- *`capped` is per row, not per accumulated gap.* Skipped rows fold their
  time into the next accepted row's delta, and that total can exceed
  `max_dclock` without any single jump doing so. It is the one row's jump
  that breaks adjacency, so that is what the flag reports.

- *`on_clock_reset = "max"` counts as capped.* The policy's answer to a
  backwards clock is "as far apart as they can be", which is the ceiling:
  the rows are not adjacent, and a ring built on them is wrong. `"zero"`,
  `"reset_state"` and `"error"` do not set it — the first says the rows *are*
  adjacent, the second rebuilds, the third refuses the chunk.

- *`predict_chunk` never clears.* Scoring hands each row a copy of the model
  in the state a learning stream would have reached; a copy that cleared its
  lags would answer for a stream that had already learned the row. `predict`
  moves nothing, the ring included.

The call site is in `run_instance`, before the `accept` test (a skipped
row's gap breaks adjacency too) and under the `else` of `plan.reset`. No
model keeps a ring yet, so the behavioural test is task 48's; what ships here
is the flag with its unit tests and the contract check that `clear_lags` is a
no-op for every model that has declared no ring (`KEEPS_LAGS`, empty today).

*Task 48 — lagged co-moments on `ew_cov` (E56).*

- *Parameters.* `lags: Option<Vec<usize>>` on `ModelKind::EwCov`, strictly
  increasing and `≥ 1`, refused otherwise (the list order is the output
  order, so it is not sorted in silence). `"lagcorr"` in `stats` requires
  `lags`; `lags` without `"lagcorr"` accumulates only.

- *Update.* `EwCovModel` owns an `EwLagCov { lags, ring: VecDeque<Vec<f64>>
  (capacity max lag), c: Vec<f64> (L × k × k) }`. Per learned row, before
  `cov.update`: `W = cov.n_eff()`, `W' = λW + w`, `a = λW/W'`, `b = w/W'` —
  the same expressions as `EwCov::update`, on the same operands, so the
  same bits — and `d_t = x − m_old` with `m_old = cov.means()`. For each
  `ℓ`: if the ring holds `x_{t−ℓ}`, `C_ℓ ← a·C_ℓ + a·b·d_t·(x_{t−ℓ} −
  m_old)'`; otherwise `C_ℓ ← a·C_ℓ` (`EwAutoCorr`'s rule for a lag not yet
  seen). Then `cov.update`, then push `x` onto the ring. A row enters the
  ring iff it entered `cov` (the zero-weight and skipped rows that only
  decay do not); when `cov` decays without a row, `c` decays by the same
  factor. `ℓ = 0` would be `comoments` exactly, and the test that says so
  is the guard on the shared arithmetic.

- *Clearing.* `clear_lags` empties the ring and nothing else: the ring is
  the row memory, `c` is a decayed statistic and keeps decaying.

- *Outputs.* `lagcorr_<a>_<b>_l<ℓ>` = `C_ℓ[a,b] / √(C₀[a,a]·C₀[b,b])`, NaN
  when undefined, emitted at `"lagcorr"`'s position in `stats`: for each
  lag, for each `a`, for each `b` (both orientations and the auto terms,
  `k²` per lag). `EwCovModel::labels` and the `ew_cov` arm of
  `output_index` walk it the same way (`columns = [a, b]`, a new
  `FieldMeta.lag: Option<usize>`).

- *State and export.* `ModelState::EwCovModel` gains `#[serde(default,
  skip_serializing_if = "Option::is_none")] lags: Option<EwLagState>`;
  legacy states read `None` and the spec's `lags` starts a fresh ring.
  `Gram` gains `lags: Option<Vec<usize>>`, `lag_comoments: Option<Vec<f64>>`;
  the Python dict gains the same keys (`None` without lags); `po.gram.merge`
  sets them `None` and says why (a ring boundary is not mergeable
  Chan-style); `subset` slices them. The E54 row's `lags`/`lag_comoments`
  columns come from here.

- *Tests.* A longhand `numpy` implementation of the recursion above (not of
  some other definition of an EW lagged covariance: this one centres both
  legs at the current pre-row mean, which is the choice that makes `ℓ = 0`
  coincide); with `lam = 1`, unit weights and `ℓ = 0`, `numpy.cov(ddof=0)`;
  the three clearing events each empty the ring and a fourth non-event (a
  gap just under `max_dclock`) does not; a spec with `lags` and one without
  give bit-identical `n_eff`, `n_kish`, `means`, `comoments`; chunk
  invariance; the sweeps.

*Task 48 as built, 2026-09-06.* The design stood. Four notes.

- *The `vech` stops at the contemporaneous matrix.* The closed row packs
  `comoments` as the upper triangle (task 45) because that matrix is
  symmetric to 1e-16; a **lagged** matrix is not symmetric at all, so
  `lag_comoments` is the full `L·k·k`, and `po.gram.from_row` reshapes
  rather than reflecting.

- *`ew_cov` is not in `KEEPS_LAGS`.* The contract probe builds it without
  lags, so `clear_lags` is a no-op there and the check still holds it to
  "the bytes did not move". The behavioural test lives in
  `tests/test_ew_cov.py`, where the ring is exercised through the stream --
  which is the only place the three clearing events exist.

- *The clearing test had to isolate the cap from the decay.* A longer gap
  decays the state more, so "gap of 4 versus no gap" differs in the 11th
  digit whatever the ring does. At `halflife = inf` the factor is exactly
  1, so a gap changes *nothing* but the ring, and the test runs there: a gap
  of 4 under a ceiling of 5 is bit-identical to no gap, a gap of 50 under
  the same ceiling is not, and the same gap of 50 under a ceiling of 100 is
  bit-identical again. That last case is what separates "capped" from
  "long".

- *pyo3 converts tuples up to twelve elements* and `GramRow` was already
  twelve, so the lag block rides as a nested `(lags, comoments)` pair beside
  it rather than as two more slots.

*Task 49 — `po.prep.refresh_time` (E58).*

- *A Rust operator, not a Python generator.* The recursion `τ_{j+1} = max_i
  (first tick of series i after τ_j)` is a sequential scan no window
  expression writes, and a per-row Python loop over ticks is the cost this
  library exists to avoid. So: `crates/online-polars/src/refresh.rs`,
  `RefreshTime::new(names, pairs)`, `feed(&DataFrame) -> PolarsResult<
  DataFrame>`, state per `by` key (a `HashMap<GroupKey, _>` as the bank
  keeps) of last time, last value, ticks since the last refresh and an
  updated flag per series — `O(m)`, or `O(m²)` pairwise; exposed through
  `online-py` as a class with `feed`, and wrapped in `prep.py` as a lazy
  IO-plugin source the way `_frame.py` wraps the bank, so the result is a
  `LazyFrame` that streams and honours the projection, predicate and slice
  polars pushes into a Python source.

- *Signature.* `refresh_time(lf, *, series, names, time, value, by=None,
  pairs=False, keep=()) -> pl.LazyFrame`. `names` is new against §10 and
  required: the output columns `<s>_value` are the schema a lazy plan must
  declare before a row is read, so the set of series cannot be discovered
  from the data. A row whose `series` is not in `names` is an error naming
  it (dropping it would hide a misspelling). Long input only: §10's "wide
  frame already on a common grid" is already synchronised, and the long
  form is one `unpivot` away — the docstring gives that line.

- *Semantics.* Rows must be in `time` order within `by`; a backwards time
  is an error naming the row. A null `value` is not an update. A grid point
  is emitted at the tick that completes the set — every series updated
  since the previous point — with `time_refresh` = that tick's time (=
  max over series of their last update, Definition 1), `<s>_value` = each
  series' last value, `n_obs_<s>` = ticks of `s` since the previous point,
  `retained_fraction` = `m / Σ_s n_obs_s` for the interval, and `keep`
  columns at their value on the completing tick. Then every flag clears
  and the counts restart. The docstring carries BNHLS §2.1's caveat
  (ANSWERS): the refresh vector is treated as observed at `time_refresh`,
  though each series' value is stale by up to one of its own inter-tick
  intervals — the price of a common grid, and why `n_obs_<s>` is emitted
  (a large count on one series is that series' staleness made visible).
  `pairs=True` runs an independent two-series state
  per pair and emits the long frame `(by?, pair, time_refresh, a_value,
  b_value, n_obs_a, n_obs_b)` with `pair = "a|b"` in `names` order.

- *Tests.* A longhand Python loop over the same rows on random Poisson
  streams (the oracle), with `by` and with `pairs`; §10's three-series
  example — `n = 8, 9, 10` ticks giving `N = 7` and `21/27` retained, which
  ANSWERS verified against BNHLS but whose tick times exist only as the
  paper's figure — so the test is a constructed three-series stream with
  those tick counts that the loop reduces to exactly `N = 7`, asserting
  `N ≤ min nᵢ` and `retained = N·m/Σnᵢ = 21/27`; a synchronous
  input (every series ticks at every time) returned with `n_obs = 1` and
  `retained_fraction = 1` everywhere; a volume clock as `time`; identical
  frames from 1 and 1000 batches; the unknown-series and backwards-time
  errors.

*Task 49 as built, 2026-09-06.* The design stood; three notes.

- *The `N = 7`, `21/27` example is constructed, and says so.* ANSWERS
  verified the numbers against BNHLS, but their tick *times* exist only as a
  figure, so the test builds a 27-tick stream with their counts (8, 9, 10)
  and asserts the reduction: seven points, three values kept each, the
  per-interval fractions `3/5, 3/4, 3/4, 3/4, 3/4, 1, 1`, and `N <= min nᵢ`.
  The first attempt at such a stream gave eight points -- seven clean cycles
  plus a tail that completed one more -- so the extras had to be placed
  *inside* an interval, before its completing tick. That is the property
  worth remembering: a repeat only drops a tick when it precedes the tick
  that closes the set.

- *`retained_fraction` is per interval, not cumulative.* `m / Σ n_obs` for
  that point. The paper's `21/27` is the aggregate, which the test computes
  as `3·N / total`.

- *The slice pushdown counts output rows.* Unlike the bank's source, where
  `head(n)` limits what the *bank is fed*, here the grid is what the query
  sliced: `head(5)` means five grid points, however many ticks that took.

*Task 50 — `rcov` (E57).*

- *Shape.* A group-scoped `OnlineModel` in `rcov.rs` with `n_outputs() = 0`
  (per row, `n_eff` alone — the count of returns in the block so far),
  `n_targets() = 0`, no decay: `halflife`/`lam` refused by name, `group` and
  `group_close` required, `weight` accepted only as `0` or `1` (any other
  value is an error naming the row); its value is the block computed at
  close, `Rcov::estimate(&self) -> RcovEstimate`, which `close_stream` calls
  for the `AnyModel::Rcov` arm and writes into the E54 row's `rcov` block.
  Input rows are **returns** (the caller differences upstream; `refresh_time`
  emits levels and `.diff().over(by)` is the line), which is what every
  formula below is written on. `ModelState::Rcov`, appended.

- *`kind = "plain"`.* `Σⱼ xⱼxⱼ'`, accumulated as it arrives, no ring. Equal
  at close to `n × EwCov::raw` under `lam = 1`, unit weights, to the bit —
  the cross-check §10 names, and the reference the other two kinds are
  measured against.

- *`kind = "kernel"` (BNHLS 2011).* `K = Σ_{h=−H}^{H} k(h/(H+1))·Γ̂_h`,
  `Γ̂_h = Σ_{j=|h|+1}^{n} xⱼx'_{j−h}`, `Γ̂_{−h} = Γ̂_h'`. Parzen: `k(x) = 1 −
  6x² + 6x³` on `[0, ½]`, `2(1−x)³` on `(½, 1]`, `0` beyond — the only
  kernel offered (`kernel="parzen"`; another name is refused). End-point
  jitter with `m = jitter`: `X̃₀ = mean(X₀..X_{m−1})`, `X̃ₙ = mean(X_{n−m+1}
  ..Xₙ)`, so the effective return series is `x̃₁ = X_m − X̃₀` (formed from
  the first `m` raw returns once row `m` is in), the raw `x_{m+1} ..
  x_{n−m}`, and `x̃_end = X̃ₙ − X_{n−m}` (formed at close from the last `m`
  raw returns); `n_eff_returns = n − 2m + 2`. BNHLS §2.2 (ANSWERS,
  verified): `m = 1` is mean-square optimal for their flat-top form and
  `m = 1 … 4` moves the estimate under 0.5 % — so `jitter` defaults to `2`
  as §10 asked and the docstring says the choice is immaterial at that
  scale; their worked `m = 2` is exactly `X̃₀ = ½(X_{τ₀} + X_{τ₁})`, `X̃ₙ =
  ½(X_{τ_{N−1}} + X_{τ_N})`, which the formula above reproduces. **Products
  enter `Γ̂_h` only once both legs are final**: at row `t`, add `x_{t−m}·x'_{t−m−h}` for `h ≤
  max_bandwidth` (the return `m` rows back can no longer be replaced by the end
  jitter); at close, add the products of `x̃_end` with the last `max_bandwidth` final
  returns. That is what makes the state a ring of `max_bandwidth + m` vectors and
  the close `O(max_bandwidth·k²)`, with no retraction. Bandwidth: `bandwidth` an
  integer `H` (then `max_bandwidth = H`, `block_rows` unneeded) or `"auto"`, BNHLS §4.1
  as ANSWERS read it: `c* = ((12)²/0.269)^{1/5} = 3.5134` for Parzen (the
  kernel constant `k''(0)²/∫k² = 12²/0.269`); per feature `ξ̂ᵢ² =
  ω̂ᵢ²/IV̂ᵢ`, with `IV̂ᵢ` the realised variance on a sparse grid
  (`iv_stride`, default 20 rows, averaged over the `iv_stride` offsets —
  their 20-minute RV) and `ω̂ᵢ²` the mean over `q` offsets of
  `RV_dense^{(i)}/(2n^{(i)})` taken on every `q`-th return (`noise_stride`,
  default `1`, i.e. the full dense grid; their `q ≈ 25` trades or `≈ 70`
  quotes for a ~2-minute grid — the estimate is deliberately upward biased,
  which their §4.1 accepts, and a stride above 1 is the caller's call); `Hᵢ
  = c*·ξ̂ᵢ^{4/5}·n^{3/5}`, `H = ⌈mean Hᵢ⌉` clipped to `max_bandwidth`, reported as
  `bandwidth_used`. `"auto"` requires `block_rows`, from which `max_bandwidth` defaults
  to `⌈c*·block_rows^{3/5}⌉` (`ξ̂ = 1`, a noise variance equal to the block's
  integrated variance, beyond which the estimator is not worth having);
  `max_bandwidth` may be given. `block_rows` is a sizing hint, not a limit: a longer
  block runs, clipped, and says so. Parzen is a positive-definite function,
  so `K` is PSD up to rounding (BNHLS: the Bartlett kernel is *not*
  consistent for this estimator, and Parzen's efficiency 0.97 beats the
  quadratic-spectral 0.93 — the reason it is the only kernel offered);
  `psd=True` clips negative eigenvalues at `0` and sets `psd_repaired` when
  it had to.

- *`kind = "preavg"` (CKP 2010; Hautsch–Podolskij 2013 for the constants).*
  `g(x) = min(x, 1−x)`, `Ȳᵢ = Σ_{j=1}^{kₙ−1} g(j/kₙ)·x_{i+j}`; `MRC = n/(n −
  kₙ + 2) · 1/(ψ₂^{kₙ}·kₙ) · Σᵢ ȲᵢȲᵢ' − ψ₁^{kₙ}/(θ²·ψ₂^{kₙ}) · 1/(2n) ·
  Σⱼ xⱼxⱼ'`, `kₙ = ⌊θ·√n⌋` (CKP's floor, ANSWERS — not the ceiling the
  first draft had), with the finite-sample constants as CKP write them:
  `ψ₁^{k} = k·Σ_{i=1}^{k}(g(i/k) − g((i−1)/k))²`, `ψ₂^{k} = (1/k)·
  Σ_{i=1}^{k−1} g(i/k)²` (limits `ψ₁ = 1`, `ψ₂ = 1/12`). The `Φ` constants
  appear only in the asymptotic variance, which the estimator does not
  report, so the estimator does not use them; the test still pins the
  limits `Φ₁₁ = 1/6`, `Φ₁₂ = 1/96`, `Φ₂₂ = 151/80640` against the
  finite-sample forms `φ₁^k(j) = Σ_{i=j+1}^{k−1}(g((i−1)/k) −
  g(i/k))·(g((i−j−1)/k) − g((i−j)/k))`, `φ₂^k(j) = Σ_{i=j+1}^{k−1}
  g(i/k)·g((i−j)/k)`, `Φ₁₁^k = k·(Σⱼ φ₁^k(j)² − ½φ₁^k(0)²)`, `Φ₁₂^k =
  (1/k)·(Σⱼ φ₁^k(j)φ₂^k(j) − ½φ₁^k(0)φ₂^k(0))` (and `Φ₂₂^k` by the same
  pattern), because they are the check that `g` and the `ψ` sums are coded
  as the paper has them. `psd=False` is that balanced, bias-corrected form
  (optimal rate, not guaranteed PSD; clipped and flagged as the kernel is);
  `psd=True` is CKP §3's longer window without the bias term — **the
  exponent and the dropped term are taken from CKP §3 during
  implementation; the `kₙ = ⌈θ·n^{1/2+δ}⌉`, `δ = 0.1` of the first draft is
  Hautsch–Podolskij's reading and stands only if §3 agrees** (the one open
  read in this task; ANSWERS did not cover §3). `kₙ` must be fixed before
  the block starts, so it is `⌊θ·√block_rows⌋` (or the §3 form on `block_rows`) from
  a required `block_rows`, or a given `window`. Streaming: a ring of `kₙ − 1`
  returns; each arriving return completes one `Ȳ`, whose outer product is
  added — `O(kₙ·k + k²)` a row.

- *The row's `rcov` block.* `rcov` list[f64] (vech, as `comoments`), `rcorr`
  list[f64] (vech, unit diagonal), `rcov_n` i64 (effective returns),
  `rcov_kind` str, `bandwidth_used` i64 (null for `plain`/`preavg`),
  `omega2`, `iv_sparse`, `iq` list[f64] per feature — `iq` the realised
  quarticity `(n_s/3)·Σ x_s⁴` on the sparse grid, a proxy and labelled one —
  `psd_repaired` bool. All null when the block is too short (`n < 2m`
  for the kernel, `n < kₙ` for pre-averaging; `n < 2` for `plain`'s
  `rcorr`) — nulls, never a panic.

- *Tests.* `plain` to the bit against `ew_cov(lam=1)`; the kernel and the
  MRC against longhand `numpy` on the same synchronised returns (from task
  49 over task 52's asynchronous rows) to `1e-10` relative; the jittered
  end formed from state alone (a test that feeds the block one row at a
  time and checks the close equals the offline computation on the full
  block — this is the acceptance for "nothing reads a future row"); the
  auto bandwidth against its formula on a block with known `ω²` and `IV`;
  PSD on adversarial streams (a constant series, a series that is one spike,
  `k = 1`); the short-block nulls; a zero-weight row absent from the ring
  (the estimate equals the one from the stream with that row removed);
  chunk invariance; the sweeps, with the residual flags refused.

*Task 50 as built, 2026-09-06.* The design stood; five notes, one of them
still open.

- **CKP §3 was not read, and the source says so** (read 2026-09-28, task
  114: §3.4 agrees, `δ = 0.1` and no bias term). `psd = true`'s longer
  window is `kₙ = ⌈θ·n^{0.6}⌉` with the bias term dropped -- Hautsch &
  Podolskij's reading, as ANSWERS flagged. `RcovCfg::window_for` carries the
  comment; if §3 disagrees, that line and the dropped term move together.
  Everything else in the model is from the papers directly.

- *The jitter bound is a claim about a block, not a handful of rows.* BNHLS
  say `m = 1..4` move the estimate by under 0.5 %; at 300 returns per block
  it is 0.8 %, because the jitter is an end effect. At 1000 it is 0.11 %,
  and the test runs there.

- *A `Point`'s legs must both be final before it enters `Γ̂_h`.* The ring
  keeps `m + 1` raw returns so `x_{t−m}` can be finalised at row `t`; the
  first `m` are kept separately for the leading jittered return, and the
  trailing one is formed at close from the *last* `m` of that ring -- not
  all `m + 1`, which was the first bug the longhand oracle caught.

- *The pre-averaged window starts one row later than the ring fills.* CKP's
  `Ȳᵢ = Σ_{j=1}^{kₙ−1} g(j/kₙ)·x_{i+j}` skips `x_i`, so the window is the
  `kₙ − 1` returns *after* the oldest in a ring of `kₙ`. Filling a ring of
  `kₙ − 1` and reading it whole is off by one window, which the longhand
  oracle also caught.

- *A session close restarts the stream, so a closing spec's live `summary()`
  is the current span's.* `test_summary.py`'s whole-stream oracle skips
  those specs, and `test_closed_groups.py` asserts the other half: the spans
  before the current one are in the closed rows.

*Task 51 — `po.corr` (E62).*

- *Shape.* `python/polars_online/corr.py`, `gram.py`'s pattern exactly: the
  `_np()` import guard with the install hint, pure functions over arrays,
  `gram()` dicts and E54 rows (a helper `_matrix(obj)` accepts an array, a
  dict with `comoments`/`columns`, or a row and returns the `k × k`
  correlation), one longhand check per function in `tests/test_corr.py`.
  Nothing here touches Rust or the models; it lands as `po.corr`, listed in
  `api_surface.txt` and in a Sphinx page beside `po.gram`.

- *`to_z` / `from_z`.* `atanh` / `tanh` with the input clipped at `|ρ| ≤ 1 −
  1e-6` (`z ≈ 7.25`) so a unit off-diagonal from a degenerate block is
  finite; the clip is in the docstring.

- *`nearest(A, W=None, *, tol=1e-8, max_iter=100)`.* Higham (2002),
  Algorithm 3.3 exactly: `ΔS₀ = 0`, `Y₀ = A`; for `k = 1, 2, …`: `R_k =
  Y_{k−1} − ΔS_{k−1}` (Dykstra's correction), `X_k = P_S(R_k)`, `ΔS_k = X_k
  − R_k`, `Y_k = P_U(X_k)`; with diagonal `W`, `P_U` sets the diagonal to 1
  and `P_S(A) = W^{−1/2}·(W^{1/2}AW^{1/2})₊·W^{−1/2}`, `(·)₊` the
  eigenvalue clip at zero. Stop (his test 4.1) when the largest of
  `‖X_k − X_{k−1}‖_∞/‖X_k‖_∞`, `‖Y_k − Y_{k−1}‖_∞/‖Y_k‖_∞`, `‖Y_k −
  X_k‖_∞/‖Y_k‖_∞` is below `tol`. Returns `(X, dist, iters)`, `dist` the
  weighted Frobenius distance. **The partial-eigensolve path §10 mentions
  is dropped**: `numpy.linalg.eigh` is the only eigensolver on the
  dependency list, and the matrices this library produces are small enough
  that a full decomposition per iteration is the cheaper code. Tests, with
  the values ANSWERS verified against the paper (§2 and §4): `A = [[1, 1,
  0], [1, 1, 1], [0, 1, 1]]` (eigenvalues `1 ± √2`, `1`) → `[[1, .7607,
  .1573], [.7607, 1, .7607], [.1573, .7607, 1]]`, `‖A − X‖_F = 0.5278`,
  `X` singular with null vector `[−.4814, .7324, −.4814]` (and `ee'`, the
  obvious guess, is at distance `√2`); `A = tridiag(−1, 2, −1)` of order 4
  → `[[1, −.8084, .1916, .1068], [−.8084, 1, −.6562, .1916], [.1916,
  −.6562, 1, −.8084], [.1068, .1916, −.8084, 1]]`, `‖A − X‖_F = 2.13`,
  rank 3, **19 iterations at `tol = 1e-8`** (linear convergence, a factor
  ≈ 3 per iteration — assert the count, it pins the algorithm and not just
  the fixed point); entries at `1e-4`, distances at `1e-3`. Bounds worth a
  test each: a diagonal `A` gives `I`; a PSD `A` with diagonal `≤ 1` gives
  `A` with its diagonal set to 1; a unit-diagonal `A` with `t` nonpositive
  eigenvalues gives an `X` with at least `t` zero eigenvalues (Theorem
  2.5 — the 4×4 example's rank 3). PSD (smallest eigenvalue `≥ −1e-10`) and
  unit diagonal on random symmetric inputs; a correlation matrix returned
  in one iteration.

- *`shrink(R, target="constant", alpha=None, X=None)`.* Ledoit & Wolf
  (2004), the constant-correlation target: `fᵢᵢ = sᵢᵢ`, `fᵢⱼ = r̄·√(sᵢᵢsⱼⱼ)`
  with `r̄ = 2/((N−1)N)·Σ_{i<j} rᵢⱼ` — on a correlation matrix `sᵢᵢ = 1`
  and `F` is the equicorrelation matrix at `r̄`. The intensity `δ̂* =
  max{0, min{κ̂/T, 1}}`, `κ̂ = (π̂ − ρ̂)/γ̂`, with (their Appendices A–B,
  verified in ANSWERS)

  `π̂ = Σᵢⱼ π̂ᵢⱼ`, `π̂ᵢⱼ = (1/T)·Σₜ ((yᵢₜ − ȳᵢ)(yⱼₜ − ȳⱼ) − sᵢⱼ)²`;
  `ρ̂ = Σᵢ π̂ᵢᵢ + Σ_{i≠j} (r̄/2)·(√(sⱼⱼ/sᵢᵢ)·ϑ̂ᵢᵢ,ᵢⱼ + √(sᵢᵢ/sⱼⱼ)·ϑ̂ⱼⱼ,ᵢⱼ)`,
  `ϑ̂ᵢᵢ,ᵢⱼ = (1/T)·Σₜ ((yᵢₜ − ȳᵢ)² − sᵢᵢ)·((yᵢₜ − ȳᵢ)(yⱼₜ − ȳⱼ) − sᵢⱼ)`;
  `γ̂ = Σᵢⱼ (fᵢⱼ − sᵢⱼ)²`.

  `π̂` and `ϑ̂` are fourth-moment sums over the rows, so the intensity needs
  the row data `X` (`T × k`, the standardised rows) and cannot be formed
  from `R` and `T` — §10's `T=None` becomes `X=None` and the function
  requires `alpha` or `X` (recorded so the next reader does not try to
  recover `π̂` from the matrix). `target="identity"` is the other target
  offered. Tests: a small `X` fixture with `δ̂*` worked longhand from the
  formulae above; `alpha=0` returns `R`, `alpha=1` returns `F`; the result
  is positive definite whenever `F` is.

- *`equicorr(R)` / `equicorr_row(r)` / `equicorr_loglik(r, rho)`.* The mean
  off-diagonal; task 46's `u` on one standardised row; task 46's closed-form
  log-density — all offline, and the pins for the model's own tests.

- *`absorption(R, k)`* = `Σ_{i≤k} λᵢ / Σᵢ λᵢ` (Kritzman et al.), and
  *`shift(ar_fast, ar_slow, *, scale=None)`* = `(ar_fast − ar_slow)/scale`
  elementwise over two aligned arrays, `scale` defaulting to
  `std(ar_slow)` over the sample — the standardised shift as published, with
  the caller choosing the two windows.

- *`spectral(R, r)` / `from_spectral(vals, vecs, *, unit_diag=True)`.* The
  top-`r` eigenpairs descending with `Pca`'s first-refresh sign rule (the
  largest-magnitude entry of each vector positive), and the completion
  `VΛV' + diag(1 − diag(VΛV'))` — the remainder is non-negative because the
  dropped components are PSD, so the result is a correlation matrix.

- *`block_means(R, labels)` / `from_blocks(B, labels)`.* `B[a, b]` the mean
  of `R[i, j]` over `i ∈ a`, `j ∈ b`, excluding `i = j`; returns `(B,
  counts)`; `from_blocks` rebuilds the block-equicorrelation matrix with a
  unit diagonal. `block_means(from_blocks(B)) == B` is the test.

- *`mp_edge(n, m, sigma2=1.0)`* = `σ²(1 + 1/Q ± 2√(1/Q))` = `σ²(1 ±
  1/√Q)²`, `Q = n/m ≥ 1` (Laloux, Cizeau, Bouchaud & Potters 1999,
  verified); the density `ρ(λ) = (Q/2πσ²)·√((λ₊ − λ)(λ − λ₋))/λ` is offered
  as `mp_density(lam, n, m, sigma2)` for a plot. Tests: `Q = 1 → (0, 4σ²)`;
  a random Wishart spectrum lying inside the edges up to a finite-`n`
  margin.

- *`signal_share(z_blocks, n_eff_blocks)`.* Per pair over `B` blocks: `clip(1
  − mean_b(1/(n_b − 3)) / var_b(z_b), 0, 1)` — the share of the between-block
  variance of Fisher-z that the sampling floor does not explain; `1-D`
  input returns a scalar.

- *`loss(fcst, real, kind, *, mu=None, n=None)`.* `"qlike"`: `tr(R̂⁻¹R) −
  log det(R̂⁻¹R) − k`, which is §10's `log|R̂| + tr(R̂⁻¹R)` shifted by the
  forecast-free constant `log|R| + k` so that it is `≥ 0` with equality iff
  `R̂ = R` (the test); `"z_mse"`: `Σ_{i<j} (ẑᵢⱼ − zᵢⱼ)² · (n − 3)` with `n`
  required; `"minvar"`: `w'Rw` with `w = R̂⁻¹μ / (μ'R̂⁻¹μ)`, `μ` defaulting
  to ones (Engle–Colacito, minimised over `R̂` at `R̂ = R` — a second test).

- *`epps_invert(gram_or_row, *, L)`.* Tóth–Kertész eq. 12 exactly
  (ANSWERS, verified): the correlation at scale `L` rows from the lagged
  co-moments at scale 1 carries **triangular** weights on the numerator
  *and* both denominators, and both orientations of the cross term —

  `ρ̂_L[a, b] = Σ_{x=−(L−1)}^{L−1} (L − |x|)·C_x[a, b] / √(Σ_x (L −
  |x|)·C_x[a, a] · Σ_x (L − |x|)·C_x[b, b])`, with `C_{−x}[a, b] = C_x[b, a]`,

  i.e. numerator `L·C₀[a, b] + Σ_{ℓ=1}^{L−1} (L − ℓ)·(C_ℓ[a, b] + C_ℓ[b,
  a])` and auto terms `L·C₀[a, a] + 2·Σ_{ℓ=1}^{L−1} (L − ℓ)·C_ℓ[a, a]`.
  The flat sum the first draft had is the `L → ∞` limit and is dropped;
  `L` is required, and the input must carry every lag `1 … L−1` (the
  docstring says `lags=list(range(1, L))`; a missing lag is an error naming
  it). Test: a simulated pair where `b` is `a` delayed by two rows — the
  scale-1 correlation near zero, `ρ̂_L` at `L = 8` near the truth — and the
  identity at `L = 1` (the plain correlation).

- *`fisher_se(n, rho=None, phi_a=None, phi_b=None)`.* `1/√(n − 3)` is the
  standard error of `z`; with `rho` the delta-method error of `ρ` itself,
  `(1 − ρ²)/√(n − 3)`; with both `phi` the inflation `√((1 + φₐφᵦ)/(1 −
  φₐφᵦ))` for an AR(1) pair — Bartlett's `Var(r) ≈ (1/n)·Σₖ ρₐ(k)ρᵦ(k)`
  under zero true cross-correlation, which for two AR(1)s sums to `(1 +
  φₐφᵦ)/(1 − φₐφᵦ)` (the geometric series `1 + 2·Σ_{k≥1}(φₐφᵦ)ᵏ`; a test
  asserts the closed form against the partial sum). The docstring says
  both assumptions — zero true correlation, linear dependence — and that
  the factor is invalid under ARCH-type innovations. One `phi` without the
  other is an error.

*Task 51 as built, 2026-09-06.* The design stood, and Higham's published
values came out exactly: the 3×3's `[.7607, .1573]` and distance 0.5278 with
a singular result whose null vector is `[−.4814, .7324, −.4814]`, the 4×4's
entries, distance 2.13 and rank 3. Four notes.

- *The iteration count is 20, not 19.* Same run, same fixed point; the
  difference is bookkeeping — this counts the iteration in which the
  convergence test passed. The count is still asserted, because it pins the
  algorithm and not just where it lands, and the test says which convention
  it is.

- *`nearest` returns `Y`, the algorithm's own answer.* `Y_k` has an
  **exactly** unit diagonal and is PSD only to `tol` — the iteration
  converges to the boundary of the cone, and `tol` is how close. Polishing
  it with one more eigenvalue clip would break the diagonal again, so the
  docstring states the bound and the test asserts `> −1e-8` at `tol =
  1e-10`. Measured over 200 random symmetric inputs, the worst smallest
  eigenvalue was `−9e-9` at `tol = 1e-8`.

- *A one-factor sample shrinks all the way.* The first `shrink` test asked
  for `0 < δ̂* < 1` on an equicorrelated sample and got exactly 1 — correctly,
  because such a sample *is* the constant-correlation target and there is
  nothing in the sample matrix worth keeping over it. The interior case
  needs a target that is wrong, so the test uses two blocks with different
  within-block correlations, and the saturating case became a test of its
  own.

- *`api_surface.txt` gained a `[helper modules]` section.* `po.corr`'s
  function names are API and nothing pinned them; the section lists
  `corr`, `eval`, `gram` and `prep`'s `__all__`, which pins all four.

*Task 52 — `po.sim.regimes` (E64).*

- *Shape.* `python/polars_online/sim.py`, numpy only, `np.random.default_
  rng(seed)` for every draw, returning `{"rows", "truth_rows",
  "truth_blocks"}` as `pl.DataFrame`s. Everything after `m` is keyword-only
  (§10's positional list puts required names after defaulted ones):
  `regimes(m, *, states, transition, n_blocks, rows_per_block,
  durations=None, design="step", smooth_rows=0, phi=0.0, scale_state=0.0,
  async_rates=None, noise=0.0, cycle_profile=None, cycle_rows=None,
  activity=None, seed=0)`.

- *The chain.* `states` is a list of `K` correlation matrices (`m × m`,
  checked for a unit diagonal and PSD) or `K` floats, each an
  equicorrelation; `transition` is `K × K` row-stochastic and drives one
  state per block; `durations` (per state, in blocks) makes the sojourn
  deterministic and uses `transition` with its diagonal removed to pick the
  next state — the recurring-state design. `design="step"` switches at the
  block boundary; `design="smooth"` interpolates the correlation matrix
  linearly over `smooth_rows` rows around it (a convex combination of two
  correlation matrices is one).

- *The series.* Latent returns `εₜ ~ N(0, Rₜ)` by Cholesky per distinct
  `Rₜ` (cached — one factorisation per state under `"step"`), `yᵢₜ = φᵢ
  yᵢ,ₜ₋₁ + εᵢₜ` with `phi` a scalar or per-series list (the documented
  truth is the innovation correlation; the AR filter moves the return
  correlation of unequal-`φ` pairs, which is what task 51's `fisher_se`
  inflation is for), scaled by `exp(scale_state · sₜ)` — a scalar `scale_state`
  makes volatility rise with the state index, a list gives it per state.
  `cycle_profile`, a list of `cycle_rows` multipliers in `(0, 1]`, scales the
  off-diagonal of `Rₜ` at bar `t mod cycle_rows` (a mix toward the
  identity, so PSD is kept); `cycle_rows` defaults to `rows_per_block`;
  `session = t // cycle_rows`. `volume`, `(mean, shape)`, draws a Gamma
  volume per bar with the mean scaled by the same `exp(scale_state · sₜ)`;
  `clock` is cumulative volume when `volume` is given and `t` otherwise.

- *Observation.* `xᵢ` are **levels** — the cumulative sum of the latent
  returns plus `noise · N(0, 1)` i.i.d. per observed bar (microstructure
  noise on the level, so the observed return is an MA(1) as the literature
  models it) — because levels are what `refresh_time` and then `.diff()`
  expect. Under `async_rates` (per-series expected ticks per bar) a bar with
  `Poisson(rateᵢ) = 0` ticks carries `null` for `xᵢ` — previous-tick
  sampling is the caller's `forward_fill`, so both the sparse and the
  filled forms are one line away, and `unpivot` on the non-null rows is
  `refresh_time`'s long input.

- *Frames.* `rows`: `entity` (a constant string `"sim"`, the join key a
  multi-entity caller would vary), `t`, `clock`, `session`, `x_1 … x_m`,
  `volume` (null when `activity=None`). `truth_rows`: `t`, `block`, `state`,
  `scale_mult`, `mix` (the `"smooth"` fraction, `0` otherwise). `truth_blocks`:
  `block`, `state`, `n_rows`, `corr` (vech of the block's mean true
  correlation — the state's matrix under `"step"`).

- *Tests.* Per-state matrices recovered from the block sample correlations,
  pooled over the blocks in each state, within `3·SE` of the Fisher-z noise
  floor; a `phi` recovered from the lag-1 autocorrelation; the Epps curve of
  an asynchronous simulation — the sample correlation of previous-tick
  returns at sampling intervals `1, 5, 20, 100` rows — increasing toward the
  truth; `noise > 0` giving a negative lag-1 return autocorrelation; every
  `Rₜ` under `"smooth"` PSD; `session` and `clock` consistent with
  `cycle_rows` and `volume`; two calls with the same seed byte-identical
  (`DataFrame.equals` on all three frames); schema and lengths fixed.

*Task 52 as built, 2026-09-06.* The design stood. Three notes.

- *The documented truth is the innovation correlation, and the tests say
  so.* An AR filter moves the *return* correlation of a pair with unequal
  `φ`; the per-state recovery test therefore runs at `phi = 0`, and `phi`
  gets its own test through the lag-1 autocorrelation. Task 51's
  `fisher_se` inflation is the tool for the other case.

- *Noise goes on the level, not the return.* That is what the literature
  models and what makes the observed return an MA(1) with a negative first
  autocorrelation -- measured at `−0.2` or below at `noise = 2` against
  `|ρ₁| < 0.03` clean, which is the microstructure effect `rcov` undoes.

- *The Cholesky factor is cached by `(state, previous state, cycle_profile bar,
  mix)`.* Under `"step"` with no cycle_profile that is one factorisation per
  state, which is what the row asked for; under `"smooth"` the mix is part
  of the key, so a ramp of `smooth_rows` costs that many factorisations and
  no more.

*Task 53 — `hmm` (E60).*

- *The filter.* Before the row: `p̃ₗ = Σₖ pₖ Πₖₗ` from the filtered `p`
  left by the previous row (uniform `1/K` before the first). Emitted from
  that alone: `p_<k>` = `p` (the filtered posterior before the row), `p1_<k>`
  = `p̃`, `state = argmax p̃` (first maximum wins, `i32` through
  `Source::Cluster`). Then, with the *pre-row* parameters, `fₗ = N(x | μₗ,
  Σₗ + rₗI)` through the `quad_forms_logdet` + softmax path `ew_class`
  already uses — the same `precision_prior` ridge with the same decaying
  scale, and `precision_prior` **required** as it is there (§10's `None`
  default is not an option: a state's centred co-moments start at zero) —
  `loglik = ln Σₗ p̃ₗ fₗ` (a row output computed from the row, as `kmeans`'s
  `dist` is), and `pₗ ← p̃ₗ fₗ / Σ`.

- *Learning.* Each state's `EwCov` (or `EwDiag` under `covariance="diag"`)
  takes the row at weight `w · pₗ` — the responsibilities sum to `w`, so the
  model-level `n_eff` is the shared recursion unchanged. **Π is not §10's
  `EW mean of pₖ(t−1)·pₗ(t)/pₖ(t−1)`**: that ratio is `pₗ(t)`, whose mean
  does not depend on `k` and cannot identify a transition matrix. The
  quantity that does is the filtered joint of consecutive states, `ξₖₗ =
  pₖ(t−1)·Πₖₗ·fₗ / Σ_{k'l'} p_{k'}(t−1)·Π_{k'l'}·f_{l'}`; the counts `Aₖₗ ←
  decay·Aₖₗ + w·ξₖₗ` decay on the clock like every accumulator, and `Πₖₗ =
  (Aₖₗ + τ)/Σₗ(Aₖₗ + τ)` with `transition_prior = τ` a Dirichlet
  pseudo-count per cell (default `1.0`; it is what keeps a never-visited
  row of Π a distribution). `transition` given seeds `A` at `τ·K·Π₀`, so the
  given matrix is the prior mean; `None` means uniform, which is what makes
  the reduction test below exact. `exog_tvtp=col` reads one more column
  (declared like `weight` and `clock`, not a feature) and sets `Πₖₗ(t) =
  softmaxₗ(Aₖₗ + Bₖₗ·xcol,t)` from the fixed `tvtp_coef = (A, B)`; the
  count-based learning of Π is off under it (`A`, `B` are not estimated
  here — a fitted-elsewhere form, like `transition` with `learn=False`).

- *Seeding.* `means`/`covs` given: no warm-up. Else `kmeans`'s recipe on a
  buffer of `warm_rows` learned rows — `seed_centres` (`pub(crate)`, same
  crate, no visibility change) under `seed_rule`, then the buffer replayed
  through the frozen seeds as hard assignments to initialise each state's
  moments — and, as in `kmeans`, every output is null until seeded; the
  buffer is state, and its rows carry their weights and decay as `kmeans`'s
  do. `learn=False` with nothing given is refused: there would be nothing
  to filter with.

- *Outputs and coefficients.* `p_<k>`, `p1_<k>`, `state`, `loglik`, `n_eff`;
  `coef` = the state means in `ew_class`'s layout so `coef()` and
  `last_row()` carry them; `predict` = the step without the step (`p`, `p̃`,
  `state`, `loglik` under the current parameters, nothing moved).
  `ModelState::Hmm {states: Vec<ModelState>, p, a, dynamics…, buffer}`,
  appended. Cost `O(K² + K·d²)` (`O(K² + K·d)` diagonal).

- *Tests.* The reduction: build an `ew_class` and an `hmm` whose states are
  that classifier's own fitted `EwCov` states (`means`/`covs` from
  `ew_class`'s `ModelState`), `transition=None`, `learn=False`, and feed
  both a stream whose `ew_class` labels are balanced — `p` equals the
  classifier's posteriors to the bit, because both run the same
  `quad_forms_logdet` and softmax on the same numbers; a longhand numpy
  Hamilton filter at fixed parameters (`learn=False`) to `1e-12`; recovery of
  Π and the state means on task 52's streams within stated tolerances, as
  the `docs/REGIMES.md` experiment (generated by `scripts/regime_experiments.
  py`, committed, not gate-regenerated — slow); predict parity; a zero-weight
  first row leaving `p` and every state untouched; the clock-rescaling test;
  the sweeps; `MINIMAL["hmm"] = {"k": 2, "features": ["x0", "x1"],
  "precision_prior": 0.1}`.

*Task 53 as built, 2026-09-06 (the model), 2026-09-06 (its experiment).*

- *Π is learned from the filtered joint, not §10's ratio.* Corrected in the
  plan block above before implementation; `EW mean of pₖ(t−1)·pₗ(t)/pₖ(t−1)`
  is `pₗ(t)`, whose mean does not depend on `k`.

- *A given `transition` seeds the Dirichlet prior, not the counts.* Seeding
  the counts at `τ·K·Π₀` gives `(K·Π₀ + 1)/(2K)`, not `Π₀`, and would need
  negative counts for a cell below `1/K`.

- *A normalisation bug the recovery test found.* The filtered joint was
  normalised about `logf[0]`; a state whose density is astronomically small
  overflows that to NaN. It is normalised about the largest density now.

- *The `docs/REGIMES.md` experiment this task's acceptance asked for landed
  with task 56, and it is the finding that matters most about this model:
  **the default seeding cannot find a regime that lives in the
  covariance**. `warm_rows` seeds with k-means over the rows, two zero-mean
  states differ in nothing k-means can see, and the filter splits them by
  direction — 56 % accuracy, which is chance, against 98 % for the same
  filter given `covs`. And a `precision_prior` of 1.0 on data of variance
  1.0 halves every correlation and collapses it to 58 %. Both are in the
  docstring, with the two ways out (pass `means`/`covs`, or give it a
  feature in which the regime is a shift in location).

*Task 54 — `corrchange` (E59).*

- *`kind="monitor"` is WKD's closed-sample test, span by span.* §10's
  sequential form centres on a training-span `ρ̂_train` and needs the
  boundary function and critical values of Wied & Galeano (2013), which
  `docs/ANSWERS-E54-E64.md` could not read (paywalled) and offers no
  constant for. What *can* be pinned against published tables is the
  closed-sample fluctuation test of Wied, Krämer & Dehling (2012), so that
  is what ships: the stream is cut into consecutive spans of `horizon`
  learned rows (`T = horizon`, required; `train` is gone — the test has no
  training span), and at the last row of a span, per pair,

  `Q = max_{2≤j≤T} (j/√T)·|ρ̂ⱼ − ρ̂_T| / D̂`,

  `ρ̂ⱼ` the sample correlation of the span's first `j` rows and `D̂` the
  delta-method long-run standard deviation of `ρ̂` over the span: the five
  raw moments `Uₜ = (x², y², x, y, xy)` centred at their span means, their
  Bartlett-kernel long-run covariance `Σ̂ = (1/T)·Σₜ Σᵤ k((t−u)/γ_T)·VₜVᵤ'`
  with `k(x) = 1 − |x|` and `γ_T = ⌊ln T⌋` — the paper writes `[log T]`
  without a base, and the natural log is the only reading that gives a
  usable bandwidth (`6` at `T = 500`; base 10 gives `2`); exposed as
  `bandwidth=None` meaning that default — mapped to `(σ_x², σ_y², σ_xy)` by
  `D₂` (rows `(1, 0, −2μ_x, 0, 0)`, `(0, 1, 0, −2μ_y, 0)`, `(0, 0, −μ_y,
  −μ_x, 1)`) and to `ρ` by `D₃ = (−½σ_xy σ_y σ_x⁻³, −½σ_xy σ_x σ_y⁻³,
  1/(σ_xσ_y))`: `D̂² = D₃D₂Σ̂D₂'D₃'`. (The paper's `D̂` is the reciprocal and
  multiplies the statistic — the same number.) Under `H₀`, `Q →_d
  sup_{0≤z≤1}|B(z)|`, so `crit` is the Kolmogorov quantile **computed** from
  `P(sup|B| ≤ x) = 1 − 2·Σ_{k≥1} (−1)^{k−1}·exp(−2k²x²)` at `alpha`, not a
  pinned constant — `1.3581` at 5%, `1.6276` at 1%, `1.2239` at 10% are what
  the function must reproduce (a test). Over the pairs `stat` is the
  maximum and `crit` is taken at `alpha/npairs` (Bonferroni; the paper
  leaves the correction open; offering `alpha_adjust="none"` is
  *implementer's call*). The statistic needs `ρ̂ⱼ` for every `j` *and*
  `ρ̂_T`, so the span's rows sit in a ring of `horizon` rows and the
  statistic is one `O(T·k² + T·γ_T·k²)` pass at the span's end — per row,
  only the push. Outputs are null except on a span's last row, where
  `stat`, `crit` and `flag` are written; the delay is at most `horizon`
  rows, which is the price of a known null. `scalar=True` runs the same
  CUSUM on task 46's `u`: `max_j (j/√T)·|ūⱼ − ū_T| / D̂ᵤ`, `D̂ᵤ` the Bartlett
  long-run standard deviation of `u` over the span, on rows an `EwDiag`
  standardises; `halflife`/`lam` are accepted under `scalar=True` only (they
  parametrise that standardiser) and refused by name otherwise, since
  neither kind decays anything else. The sequential Wied–Galeano detector
  stays in §10 as the step after this one, to be taken when its paper has
  been read; it does not block the task.

- *`kind="window"`.* Two adjacent rings of `window` rows; `stat =
  ‖vech(R̂_pre − R̂_post)‖` in ℓ₁ or ℓ∞ over the strict upper triangle, each
  `R̂` the sample correlation of its window (scale-free, so no
  standardiser). `crit` a number, or `"permute"`: **not §10's sign flips** —
  negating a whole row leaves every `xₜxₜ'` and so every correlation matrix
  unchanged, so a sign-flip null has zero spread (ANSWERS confirms). The
  exchangeable null for "the two windows share a distribution" is a
  permutation of the pooled `2·window` rows between the windows: `n_perm`
  draws, the `(1 − alpha)` quantile of the permuted statistics, recomputed
  every `permute_every` rows (each draw is `O(window·k²)` from scratch — the
  reason it is not per row), with `perm_block` (default `1`) permuting
  blocks of consecutive rows so that serially dependent rows do not make
  the null too liberal. A fixed-seed generator in state keeps the draws
  chunk-invariant.

- *Outputs and state.* `stat`, `crit` (null until a critical value exists),
  `flag` (`Source::Flag`), `since_flag` (`Source::Id`; learned rows since
  the last flag, null before the first), `n_eff`. `reset=True` empties both
  rings at a flag (`window`; `monitor`'s spans are disjoint by
  construction, so `reset` does not apply to it); `reset=False` keeps
  monitoring and the flag simply stays up while the statistic is over.
  Task 47's `clear_lags` empties the rings and, for `monitor`, abandons the
  current span and starts a new one — a break in the clock is a break in
  the data the statistic assumes contiguous. `ModelState::CorrChange`,
  appended. `n_outputs() = 5`.

- *Tests.* `kind="monitor"` against WKD's Tables 1–2 (5000 replications,
  5%, i.i.d. bivariate `t₅` innovations), which ANSWERS transcribes: size
  at `T = 500` is `.040 / .035 / .041` at `ρ = −0.5 / 0 / 0.5` and at `T =
  1000` `.038 / .034 / .039` — the gate's size test uses `|ρ| ≤ 0.5` (the
  test over-rejects at `|ρ| = 0.9` for `T ≤ 500`: `.142`–`.144`, which is
  the paper's finding, not a bug) and the paper's numbers as its
  expectation with a binomial band, over `≥ 300` seeds in the gate and `≥
  2000` in the `docs/REGIMES.md` experiment; size-adjusted power on `0.5 →
  0.7` at `T/2`, `.587` at `T = 500` and `.830` at `T = 1000`, within
  Monte-Carlo error; `D̂` and `Q` on one span against a longhand numpy
  computation to `1e-12`; the Kolmogorov quantiles above; `kind="window"`
  against a longhand numpy statistic to the bit and the permutation
  quantile against a numpy re-draw with the same seed; average run length
  to false alarm and mean detection delay of the `window` kind on task
  52's streams recorded in `docs/REGIMES.md` (the monitor's false-alarm
  rate per span *is* its size, already pinned); rings emptied across a
  session change and a
  `max_dclock` breach; `reset` both ways; the clock-rescaling test; the
  sweeps; `MINIMAL["corrchange"] = {"features": ["x0", "x1"], "window": 8,
  "crit": 0.5}`.

*Task 54 as built, 2026-09-06.* The re-scoping held. **The comparison with
WKD's tables did not, and task 56's `docs/REGIMES.md` is where that was
found**: their Tables 1 and 2 are for "i.i.d. bivariate `t₅` innovations",
the gate's tests draw *Gaussian* pairs, and the two are not interchangeable.
Measured at 2000 replications across three DGPs: at `ρ = 0` everything
agrees with the paper; at `|ρ| = 0.5` a shared-scale `t₅` makes this
implementation liberal (`.075` against their `.040`) and independent `t₅`
marginals conservative (`.025`), with their number between the two, while
Gaussian pairs sit at the nominal 5 %. Size-adjusted power is *below* their
table under either `t₅` (`.43` and `.47` against `.587`) and well above it
on Gaussian pairs. The gate's size test now pins **the nominal level on
Gaussian pairs**, which is a property this implementation has, instead of a
`t₅` table it does not reproduce.

**And the cause is one number, measured (§4 of that document).** `D̂` is the
delta-method sd of `√T·ρ̂`, and for an elliptical law the quantity it
estimates is `(1−ρ²)√(1+κ)` in closed form. Against that: on Gaussian pairs
`D̂` is exact and tight (0.750 against 0.750 at `T = 2000`, sd 0.03); on a
tail-dependent `t₅` it is **15 % low, no better at `T = 2000` than at
`T = 500`**, with a scatter 40 % of its own size. It is built from fourth
moments and a `t₅` has kurtosis only just (`ν > 4` by one), so there is no
rate at which it settles. Too small a denominator is the liberal size; the
scatter is the lost size-adjusted power. The estimator is not wrong — it is
right where the delta method promises it will be. Five notes.

- *`scalar = true` is a **mean** CUSUM, not a correlation one.* The first
  implementation pushed `[u, u]` into the pair machinery, whose correlation
  is exactly 1 on every prefix, so the statistic was identically zero. The
  equicorrelation is already one number: what there is to test is its
  *level*, `max_j (j/√T)·|ūⱼ − ū_T| / D̂ᵤ` with the Bartlett long-run sd of
  `u`. The ring is one column wide under `scalar`.

- *`predict` computes the span-closing statistic too.* The first version
  reported nothing from `predict`, which breaks the contract on exactly the
  rows that matter. `read(x, weight)` is now a pure function of the state
  and the row that both `step` and `predict` call; the permutation critical
  value is **refreshed after** the row is reported, so the two cannot see
  different ones.

- *The predict-parity helper needed a `SPARSE_OUTPUT` list.* It asks for 300
  of 400 rows to have every slot ready, and a span-based model has one row
  in `horizon`. Ten is enough there, and the list says which models it is
  for.

- *The window kind's flag rate per row is not `alpha`.* Two windows that
  slide by one row are almost the same windows, so a statistic above the
  quantile stays above it for a run of rows. Measured: 9 % of rows on a
  stationary stream against 27 % on a broken one, which is the separation
  the test asserts. Documented in the docstring and the README.

- *A capped clock gap abandons the span, and that is visible.* The golden
  pipeline's stream jumps 9 units every 17 rows; at `max_dclock = 6` every
  jump is capped, task 47's `clear_lags` empties the ring, and **no span
  ever closes**. Correct, and it pins nothing, so that spec uses a cap above
  the gaps. `tests/test_corrchange.py` asserts the abandonment directly.

*Task 55 — `bocpd` (E61).*

- *The recursion.* Adams & MacKay (2007) in log space: the growth branch
  `P(rₜ = r+1, x₁:ₜ) = P(rₜ₋₁ = r, x₁:ₜ₋₁)·πₜ^{(r)}·(1 − H)`, the changepoint
  branch `P(rₜ = 0, x₁:ₜ) = Σᵣ P(rₜ₋₁ = r, x₁:ₜ₋₁)·πₜ^{(r)}·H`, `H = 1/hazard`
  with `hazard` a number or a column read per row (declared like `weight`,
  not a feature); ANSWERS confirms the two branches against the paper's
  Algorithm 1. Truncation: runs whose normalised mass is below
  `truncate` are dropped and the vector renormalised; `max_run` caps its
  length by folding the tail into the last kept run. The vector and one
  sufficient-statistic block per kept run are the state; the cost is
  `O(runs · d²)` a row (`O(runs · d)` diagonal).

- *Emissions.* `"gaussian"`: normal-inverse-Wishart, `(μ₀, κ₀, ν₀, Ψ₀)` from
  `prior_mean` (default zeros), `prior_kappa` (`1.0`), `prior_nu` (default
  `d + 2`), `prior_scale` (a scalar `s` for `Ψ₀ = sI`, or a `d × d` matrix;
  default `1.0`, and the docstring says to set it from the data's scale);
  per run the statistics `(n, Σx, Σxx')` and the predictive `t_{ν−d+1}(μₙ,
  Ψₙ(κₙ+1)/(κₙ(ν−d+1)))`. `"diag"`: normal-inverse-gamma per feature, the
  predictive a product of Student-t densities. `"robust"`: the
  diffusion-score-matching conjugate posterior of Altamirano, Briol &
  Knoblauch (2023) with `robust_beta` its one hyperparameter — the plan
  does not restate its equations; the implementer takes them from the
  paper (its Gaussian case is closed form, quadratic in weighted
  sufficient statistics), and the acceptance test below is what pins it.
  `robust_beta` with another emission, or `prior_*` that do not fit the
  emission, are refused by name. Univariate use (one feature) is allowed and
  is the paper's own setting. ANSWERS did not verify the ABK 2023 posterior
  (PMLR 202, open access) — it is the second of the two reads left to the
  implementer, alongside CKP §3 in task 50.

- *Outputs.* `p_r0` — the posterior mass of `rₜ = 0`, i.e. that *this* row
  started a run, computed from the row (`kmeans`-`dist` style, and the
  docstring says so; the pre-row probability of a changepoint is `H` itself
  under a constant hazard and carries nothing); `run_mode` and `run_mean`
  from the run-length posterior **before** the row (`Source::Id` and `f64`);
  `pred_<f>` the pre-row predictive mean (mixture over runs); `logscore` the
  log predictive density of the row under the pre-row mixture; `n_eff`.
  `predict` returns the pre-row quantities without moving. `coef` empty.
  `ModelState::Bocpd`, appended.

- *Tests.* Against a longhand numpy Algorithm 1 on a synthetic univariate
  mean-shift stream, the full run-length posterior to `1e-10` at every row
  (`truncate = 0`); a variance step detected on Adams–MacKay's own finance
  fixture shape (ANSWERS, their §3): zero-mean Gaussian rows with a
  piecewise-constant variance, the `"diag"` emission with a gamma prior on
  the inverse variance (`a = 1`, `b = 1e-4`, their values — `prior_nu = 2a`,
  `prior_scale = 2b` in the inverse-gamma parametrisation the emission
  exposes), `hazard = 250` (their `λ_gap`), `p_r0` peaking within a stated
  delay of each step; `truncate = 1e-4` changing no reported output by more than
  `1e-4` against `truncate = 0`; the robust variant's `p_r0` unmoved by a
  single 20-σ row where the Gaussian one restarts; a zero-weight row leaving
  the posterior untouched; the hazard column; chunk invariance and save/load
  through the sweeps; `MINIMAL["bocpd"] = {"features": ["x0"]}`.

*Task 55 as built, 2026-09-06.*

- *`p_r0` is not a quantity.* Algorithm 1 puts the **same** predictive on
  the growth and the changepoint branch, so the normalised mass at `r = 0`
  is `H·Σ Jπ / Σ Jπ` = **exactly the hazard, on every row, whatever the
  data**. §11a asked for it and expected it to spike; it cannot. What ships
  is `p_change = P(rₜ ≤ 1)`, which adds the run that started one row ago —
  the one evaluated against the *prior* predictive on the break row.

- *And `run_mode` is the output to read.* It is the run length before the
  row, so **`t − run_mode` is the row the current run began on**, and that
  is what a caller wants. Measured on three fixtures: a tenfold variance
  step, a four-sigma mean shift and a correlation-only break are all dated
  to the right row within one to three rows of it, while `p_change` reaches
  0.83 on the first, barely lifts on the second and never moves on the
  third. The docstring, the README and the module docs all lead with the
  run length and call `p_change` the alarm.
