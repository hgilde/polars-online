# faer's small Cholesky under `-C target-cpu=x86-64-v3`: what changes, and every lever

Research only; nothing in the repository was edited or built. 2026-11-17, rustc 1.99.1,
faer 0.24.4, faer-traits 0.24.0, pulp 0.22.3. All codegen evidence is a cross-compile from
this arm64 Mac to `x86_64-apple-darwin` of the project in this directory (`src/main.rs` is
`solve.rs`'s `cholesky_seq` alone in a binary; `Cargo.toml` has the release profile, thin
LTO and one codegen unit, less `strip`), read with `nm -C` and `objdump -d`. Nothing was
timed: no x86 here, and no Rosetta. Registry paths below are relative to
`~/.cargo/registry/src/index.crates.io-*/`.

## 1. The mechanism

**The algorithm does not change. The same AVX2 kernel runs; under v3 it has no symbol of
its own.** "faer runs `cholesky_recursion_right_looking` instead of `simd_cholesky`" is how
a function list reads it, because the kernel's body is now inside that function.

The path, for an `f64` matrix of order <= 64:

1. `llt::factor::cholesky_in_place` (`faer-0.24.4/src/linalg/cholesky/llt/factor.rs:67`)
   calls `ldlt::factor::cholesky_block_left_looking`
   (`.../ldlt/factor.rs:499`), whose `if true || ...` (`:529`) always calls
   `cholesky_recursion_right_looking` (`:367`).
2. There `if n <= recursion_threshold` (`:380`, default 64, `:709`) calls `simd_cholesky`
   (`:218`). That is the only choice of algorithm, it depends on `n` alone, and no `cfg`,
   `Par` or SIMD width enters it. (The SIMD width sets only the kernel's inner batch,
   `4 * lanes` columns, `:190`: 16 for AVX2 `f64`, so one batch at order 2-12.)
3. `simd_cholesky` runs `dispatch!` (`:263`; the macro is `faer-0.24.4/src/lib.rs:201`),
   which is `pulp::Arch::default().dispatch(Impl {..})`.
4. `Arch::new` (`pulp-0.22.3/src/x86.rs:313`, `#[inline]`) is `V3::try_new()`; pulp's
   `x86-v4` feature is off (faer asks for `x86-v3` only, `faer-0.24.4/Cargo.toml:167`).
   `try_new` -> `is_available` (`pulp-0.22.3/src/core_arch/mod.rs:321`, `:341`) loads a
   static `AtomicU8` and, on first use, calls `__detect_is_available`
   (`:350`, `#[inline(never)]`). **This run-time check stays under v3**: the v3 binary
   still loads `V3::__static_available::AVAILABLE` and still carries the `Scalar` arm.
   Only the detection behind it folds to `true`, because
   `std::is_x86_feature_detected!` is `cfg!(target_feature = ..) || <cpuid>`
   (`library/std_detect/src/detect/macros.rs:9`). No static `Arch` is chosen.
5. `Arch::dispatch` (`x86.rs:327`, `#[inline(always)]`) -> `Simd::vectorize`
   (`x86/v3.rs:1215`, `#[inline(always)]`) -> the inherent `V3::vectorize`
   (`core_arch/mod.rs:363`, `#[inline(always)]`), which calls

   ```rust
   $(#[target_feature(enable = $feature)])*      // core_arch/mod.rs:382-386
   #[inline]
   unsafe fn imp<F: NullaryFnOnce>(f: F) -> F::Output { f.call() }
   ```

   (`imp_fastcall`, `:365-380`, is the same for closures of at most 8 words; the
   Cholesky's `Impl` is larger, so it takes `imp`.) Everything below `imp` is
   `#[inline(always)]` (`with_simd`, `simd_cholesky_matrix`, `simd_cholesky_row_batch`,
   `ldlt/factor.rs:6,178,239`), so `imp::<..simd_cholesky::Impl<f64>>` *is* the kernel.

What decides whether `imp` is a function of its own is one rule: a `#[target_feature]`
function is not inlined into a caller that lacks those features (the Reference,
`attributes/codegen.html`: "Functions marked with target_feature are not inlined into a
context that does not support the given features"; LLVM's test is callee features being a
subset of the caller's).

- **Baseline x86-64**: `simd_cholesky` has no AVX2, so `imp` cannot be inlined. It is the
  run-time-dispatched kernel the VM saw.
- **v3**: every function has all 14 of `V3`'s features (`x86/v3.rs:12-27`: sse, sse2, fxsr,
  sse3, ssse3, sse4.1, sse4.2, popcnt, avx, avx2, bmi1, bmi2, fma, lzcnt). The barrier is
  gone, `imp` is `#[inline]` with one caller, and it is inlined into `simd_cholesky`, which
  (one caller, one codegen unit) is inlined into `cholesky_recursion_right_looking`.

All of this code is generic over `T`, so it is instantiated **in the calling crate**
(the symbols end `..12fable_inline` here; `online-core` in the product). The inlining is
inside our own codegen unit and needs no LTO.

### Measured on x86 codegen

| build | `vectorize::imp` symbols | `imp::<..simd_cholesky::Impl<f64>>` |
|---|---|---|
| baseline | 9 (5 pulp `V3`, 4 `gemm_common` `V3`) | present, 547 lines of disassembly |
| `-C target-cpu=x86-64-v3` | 0 | gone |

- Baseline: `cholesky_recursion_right_looking` is 1096 lines, no `ymm`, frame 0x1f8;
  it holds the scalar arm and calls `imp` (frame 0x118, 10 `vfmadd231pd`, 14 indirect
  jumps).
- v3: `cholesky_recursion_right_looking` is 1880 lines, frame 0x320, with the kernel
  inside it: 10 `vfnmadd231pd` (LLVM folded the kernel's sign flip into the FMA, which
  is exact, so the bits agree, as the VM found), 16 indirect jumps, and the scalar arm
  re-encoded in AVX. Stack references: 509, against 118 (kernel) + 290 (caller).
- The indirect jumps are pulp's partial load and store for a column tail: inline asm that
  jumps through the `LD_ST` table and returns by `jmp rcx` (`x86.rs:259-292`,
  `x86/v3.rs:241-258`). Order 2-12 is all tail, so this kernel is indirect-branch-bound.

### What this does and does not show

It shows the kernel is the same and where it went. **It does not show that the inlining is
what costs 5-7%.** One fewer call and 6% fewer instructions should be faster; candidates
for the loss are the merged function's spills (the kernel now shares a 0x320 frame with
the scalar arm), the indirect jumps landing at new addresses, or code outside the
Cholesky: under v3 all of `hmm` is recompiled with AVX2, and the attribution came from
function lists. Lever A below separates these in one build.

### The same thing happens to every pulp dispatch

Under v3 all nine `imp`s vanished, not one: `rank_update`, `matvec_colmajor`,
`matvec_rowmajor` and `solve_lower_triangular_in_place` beside the Cholesky. faer has 37
`dispatch` sites in 20 files (householder, jacobi, QR, bidiag, LDLT/LLT update,
Hessenberg, tridiag, Bunch-Kaufman, full-pivot LU, the norms, sum, meanvar, matmul).

A second, separate change is a real compile-time selection, but a benign one: faer's
product picks `private-gemm-x86` by `is_x86_feature_detected!("avx2") && ..("fma")`
(`faer-0.24.4/src/linalg/matmul/mod.rs:1359-1366`, `matmul/internal/mod.rs:13,35,97`,
`matmul/triangular.rs:621`). Under v3 that is constant, and the `gemm` fallback is
deleted: no `gemm_common` symbol is left in the v3 binary. It selects what the run-time
check selects on any AVX2 machine.

## 2. The levers

| | lever | keeps the kernel out of line under v3? | evidence |
|---|---|---|---|
| A | v3 less one feature pulp's `V3` needs: `-C target-cpu=x86-64-v3 -C target-feature=-lzcnt` | **yes** | built: 5 pulp `imp`s back, kernel within register naming of baseline's (550 lines against 547) |
| B | `[patch.crates-io]` pulp with `#[inline(never)]` on `imp` and `imp_fastcall` | **yes** | built: same 5 `imp`s, kernel identical to A's, line for line |
| C | keep the baseline build; give our own hot loops AVX2 through `pulp::Arch::new().dispatch(..)` | yes, by construction (it is today's build) | not built |
| D | our own small-order Cholesky | removes this call only | already measured, 14-18% |
| E | faer parameters: `LltParams { recursion_threshold, block_size }`, `Par`, `Spec`/`auto!` | **no** | source: both branches end in `simd_cholesky`; it and `cholesky_recursion_right_looking` are private / `pub(crate)`; no public entry to the kernel |
| F | pulp features | **no** | pulp has `std`, `x86-v3`, `x86-v4`, `macro`, `nightly`, `relaxed-simd`; none forces dispatch, and the run-time check is already there, it is the inlining that changes |
| G | `[profile.release.package.{pulp,faer,faer-traits}]` `opt-level = 1`, `codegen-units = 16` | **no** | built: 0 `imp`s. The generic code is instantiated in our crate, so their profiles do not compile it |
| H | `lto = "off"` or `"fat"` | **no** | built both: 0 `imp`s. LTO is not involved |
| I | `-C llvm-args=-inline-threshold=0 -C llvm-args=-inlinehint-threshold=0` | **no** | built: 0 `imp`s (a single-caller local function is inlined at any threshold), and it would degrade the whole build |
| J | `#[inline(never)]` wrapper of ours | **no** | `cholesky_seq` in the test is `#[inline(never)]`; the inlining is below it, between two faer/pulp functions |
| K | `#[target_feature]` wrapper of ours | **no**, and it can cause it | see the caution under C |
| L | nightly: `-Z inline-llvm=no`, Cargo `profile-rustflags` | not usable | nightly only, not tried. Per-package rustflags would have to build `online-core` itself at baseline, which gives up v3 where it pays |

### A. v3 less `lzcnt`

```
RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=-lzcnt"
```

`rustc --print cfg` with these flags lists avx2, fma, bmi1, bmi2 and no lzcnt, with no
warning. The caller then lacks one feature `imp` enables, so the Reference's rule keeps
`imp` out of line, for all 37 sites at once, with no source change and no fork.

- Cost: `leading_zeros` becomes `bsr` and a fix-up outside pulp's kernels (which enable
  lzcnt themselves). The binary still needs an AVX2 machine, and every such machine has
  lzcnt, so compatibility is the v3 build's.
- `popcnt`, `bmi1` or `bmi2` would do as well; `lzcnt` is the cheapest to lose. Dropping
  `fma` or `avx2` would work but gives up what v3 is for (`f64::mul_add` becomes a libm
  call without `fma`).
- It rests on a rule the Reference states, not on an optimizer heuristic, but nothing
  in the build names its purpose: it needs a comment, and a check (below) that fails
  if a pulp release changes `V3`'s feature list.
- `gemm_common`'s `V3` (avx, avx2, fma, no lzcnt: `gemm-common-0.19.0/src/simd.rs:618`)
  would still be inlined, but that code is unreachable under v3 (above).
- The same idea with baseline tuning (not built): `-C target-feature=+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+movbe,+popcnt`
  and no `target-cpu`.

### B. A pulp fork

`pulp-inline-never.patch` in this directory: two lines in `src/core_arch/mod.rs`
(`:366` and `:383`), `#[inline]` -> `#[inline(never)]`. `simd_type!` is exported and
`gemm-common` expands it too, so the patch covers both.

```toml
[patch.crates-io]
pulp = { path = "vendor/pulp" }   # or a git rev
```

- Cost: a fork to carry across every pulp release; `[patch]` applies to this workspace's
  builds (wheel, CLI) but not to anyone depending on `online-core` from a registry; and it
  also forbids the inlining where it may help. A carries none of that. B's merit is that it
  is explicit, and that it is the change to offer upstream.

### C. Stay at baseline and vectorize our own loops

v3's gains were in our code (`ew_class`, `ewridge`, `kalman`); faer already has AVX2 at
baseline. `pulp::Arch::new().dispatch(op)` with a `WithSimd` whose `with_simd` is
`#[inline(always)]` is safe code (no `unsafe` in `online-core`, rule 6) and recompiles the
body with AVX2+FMA behind the same run-time check.

- Cost: code per hot loop, and each needs its bits checked: inside `imp` LLVM may not
  contract `a*b+c`, but any `mul_add` written there becomes a real FMA.
- **Caution**: do not call faer from inside such a body. The body has `V3`'s features, so
  faer's baseline functions can be inlined into it and `imp` after them: the same merge,
  made by hand. An `#[inline(never)]` function with no `target_feature` between the two
  stops it there, and only there.

## 3. Upstream

No report of this was found. Searched the issues and pull requests of
`sarah-quinones/pulp` and `sarah-quinones/faer-rs` for target-cpu, x86-64-v3, native,
target-feature, inline and dispatch, and the web.

- https://github.com/sarah-quinones/faer-rs/issues/108 (open): the only faer issue that
  touches build flags, about a dynamic library's speed; not this.
- https://github.com/sarah-quinones/faer-rs/issues/169 (closed): a slowdown across faer
  versions; not this.
- https://github.com/rust-lang/rust/issues/83027 and
  https://internals.rust-lang.org/t/slower-code-with-c-target-cpu-native/17315: the
  general case of `target-cpu` and function-level `target_feature` interacting badly.
- pulp ships no changelog in the crate; its README only says to mark `with_simd`
  `#[inline(always)]`.

Seen in passing, unrelated to this question but about the pinned version:
https://github.com/sarah-quinones/faer-rs/issues/273 (open) reports wrong results from
faer 0.24.4's parallel `f64` matmul on row-major views.

Any report upstream is the user's to file, after the measurements below.

## 4. Recommendation, and what the VM should measure

Measure A first: it costs one flag, and it answers whether inlining is the cause.

Builds, all from one sha, `hmm` at orders 2-12, interleaved, load logged, each under the
default and the aligned layout:

1. baseline;
2. `-C target-cpu=x86-64-v3`;
3. `-C target-cpu=x86-64-v3 -C target-feature=-lzcnt`.

For each: wall time (median and min), instructions, branch mispredicts **and indirect
branch mispredicts** (`perf stat -e branch-misses,br_misp_retired.indirect` or the
host's equivalent), the output digest, and evidence the lever took:
`nm -C <binary> | grep -c 'vectorize::imp'` must be non-zero in 1 and 3 and zero in 2
(build unstripped for this, or read the function list).

- 3 at or under 1 on `hmm`: inlining was the cause. Then run the whole bench under 3
  against 2: 3 must keep v3's gains (`ew_class`, `ewridge`, `kalman`, `rcov`), and the
  other faer paths (rank update, matvec, triangular solve, eigensolver) must be no slower
  than under 2, since A un-inlines all of them and some may have gained from it.
- 3 as slow as 2: the Cholesky's inlining is not the cause. Look at `hmm`'s own code under
  AVX2 (per-function time, 1 against 3), and A buys nothing.
- If A wins, hold it with a test on the built artifact (the `nm` count above, on an
  unstripped build of the same flags), so a pulp that changes `V3`'s features cannot undo
  it silently.

B is the fallback if A cannot be used, and C the route if v3 as a whole is turned down.
D stands either way for this one call.
