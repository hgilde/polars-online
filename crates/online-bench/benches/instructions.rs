//! Instruction counts of the hot paths, under Callgrind, for `budgets.toml`.
//!
//!     cargo bench -p online-bench --bench instructions -- --output-format=json
//!
//! Linux only, where Valgrind runs: CI's `budgets` workflow runs it on every
//! push and pull request and holds each count to its budget
//! (`scripts/budgets.py`). Each case is set up outside the count
//! (`online_bench`'s `*_setup`), and Callgrind's entry point is the case's
//! measured function, so a count is the block of rows and nothing else.
//! Every case but one runs on one thread of each pool: Callgrind collects
//! from an entry point on the thread that enters it, so a second thread's
//! share of the work would be missing from the count. The exception is
//! `bank_ew_cov_wide_t4`, a bank on a pool of four threads: it is counted
//! with Callgrind's instrumentation off until a client request turns it on
//! around the call, which counts every thread (Valgrind runs them one at a
//! time), and so the count is the work and what spreading it costs.
//!
//! A case added here needs an entry in `budgets.toml`; its first CI run
//! reports its count, which is then committed as its budget.

use std::hint::black_box;

use gungraun::prelude::*;
use gungraun::{Callgrind, EntryPoint};
use online_bench::{
    BankRun, ClockRun, ModelRun, bank_setup, clock_setup, model_setup,
    polars_online_budget_bank_rows, polars_online_budget_clock_rows,
    polars_online_budget_model_rows,
};

/// Count from `entry` on, with each pool held to one thread.
fn counted_from(entry: &str) -> LibraryBenchmarkConfig {
    let mut config = LibraryBenchmarkConfig::default();
    config
        .tool(Callgrind::default().entry_point(EntryPoint::Custom(entry.to_owned())))
        .env("POLARS_ONLINE_MAX_THREADS", "1")
        .env("POLARS_MAX_THREADS", "1");
    config
}

#[library_benchmark(config = counted_from("polars_online_budget_model_rows"))]
#[bench::ewridge_k5(args = ("ewridge_k5"), setup = model_setup)]
#[bench::ewridge_k20(args = ("ewridge_k20"), setup = model_setup)]
#[bench::sgd(args = ("sgd"), setup = model_setup)]
#[bench::pa(args = ("pa"), setup = model_setup)]
#[bench::seqtest(args = ("seqtest"), setup = model_setup)]
#[bench::bocpd(args = ("bocpd"), setup = model_setup)]
#[bench::kmeans(args = ("kmeans"), setup = model_setup)]
#[bench::micro(args = ("micro"), setup = model_setup)]
#[bench::kalman(args = ("kalman"), setup = model_setup)]
#[bench::ew_cov(args = ("ew_cov"), setup = model_setup)]
fn model(mut run: ModelRun) -> ModelRun {
    black_box(polars_online_budget_model_rows(black_box(&mut run)));
    run
}

#[library_benchmark(config = counted_from("polars_online_budget_clock_rows"))]
#[bench::clock_float(args = ("clock_float"), setup = clock_setup)]
#[bench::clock_temporal(args = ("clock_temporal"), setup = clock_setup)]
fn clock(mut run: ClockRun) -> ClockRun {
    black_box(polars_online_budget_clock_rows(black_box(&mut run)));
    run
}

// On the bank's own pool, where `fit_predict`'s hand-off to the pool runs
// inline: the entry point and the rows it learns are on one thread. (A
// plain comment: the harness's attribute refuses a doc comment.)
#[library_benchmark(config = counted_from("polars_online_budget_bank_rows"))]
#[bench::bank_seqtest(args = ("bank_seqtest"), setup = bank_setup)]
#[bench::bank_ewridge_k5(args = ("bank_ewridge_k5"), setup = bank_setup)]
#[bench::bank_ew_cov_wide_t1(args = ("bank_ew_cov_wide_t1"), setup = bank_setup)]
fn bank(mut run: BankRun) -> BankRun {
    let pool = online_polars::pool().expect("the bank's pool, built by the setup");
    black_box(pool.install(|| polars_online_budget_bank_rows(black_box(&mut run))));
    run
}

/// Callgrind's client requests to start and stop its instrumentation:
/// `CALLGRIND_START_INSTRUMENTATION` and `CALLGRIND_STOP_INSTRUMENTATION`
/// of Valgrind's `callgrind.h`, the tool base `('C' << 24) | ('T' << 16)`
/// plus 4 and 5. Instrumentation, unlike collection, is one state for the
/// whole process, so what every thread runs between the two is counted.
/// Issued as `valgrind.h`'s client-request sequence for the architecture:
/// rotations of a register by 128 bits in all, which leave it as it was,
/// then an instruction that does nothing; Valgrind recognises the sequence
/// and serves the request, and run natively it does nothing. Elsewhere, no
/// request at all: Callgrind runs this bench on Linux.
mod callgrind {
    const BASE: usize = ((b'C' as usize) << 24) | ((b'T' as usize) << 16);

    pub fn start_instrumentation() {
        request(BASE + 4);
    }

    pub fn stop_instrumentation() {
        request(BASE + 5);
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[inline(always)]
    fn request(code: usize) {
        let args: [usize; 6] = [code, 0, 0, 0, 0, 0];
        // SAFETY: the rotations of `rdi` add up to 128 bits and leave it
        // unchanged, `xchg rbx, rbx` changes nothing, and `rdx` (the
        // request's default result, 0) is declared as written. Under
        // Valgrind the request reads the six words `rax` points to, which
        // live until the block ends.
        unsafe {
            std::arch::asm!(
                "rol rdi, 3",
                "rol rdi, 13",
                "rol rdi, 61",
                "rol rdi, 51",
                "xchg rbx, rbx",
                inout("rdx") 0usize => _,
                in("rax") args.as_ptr(),
            );
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[inline(always)]
    fn request(code: usize) {
        let args: [usize; 6] = [code, 0, 0, 0, 0, 0];
        // SAFETY: as on x86_64: the rotations of `x12` add up to 128 bits,
        // `orr x10, x10, x10` changes nothing, and `x3` (the default
        // result) is declared as written; `x4` points to the six words.
        unsafe {
            std::arch::asm!(
                "ror x12, x12, 3",
                "ror x12, x12, 13",
                "ror x12, x12, 51",
                "ror x12, x12, 61",
                "orr x10, x10, x10",
                inout("x3") 0usize => _,
                in("x4") args.as_ptr(),
            );
        }
    }

    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    fn request(_code: usize) {}
}

/// Count every thread, from `callgrind::start_instrumentation` to its stop
/// and nothing outside, with the process default's pool and Polars' held to
/// one thread each, so no idle pool but the bank's own is there to spin.
fn counted_on_every_thread() -> LibraryBenchmarkConfig {
    let mut tool = Callgrind::with_args(["--instr-atstart=no"]);
    tool.entry_point(EntryPoint::None);
    let mut config = LibraryBenchmarkConfig::default();
    config
        .tool(tool)
        .env("POLARS_ONLINE_MAX_THREADS", "1")
        .env("POLARS_MAX_THREADS", "1");
    config
}

// A bank on a pool of four threads, called from this thread, which waits
// while the pool works: counted from the client request on, on every
// thread, rather than from an entry point, which would count this thread's
// waiting alone. The output is dropped after the count stops. (A plain
// comment: the harness's attribute refuses a doc comment.)
#[library_benchmark(config = counted_on_every_thread())]
#[bench::bank_ew_cov_wide_t4(args = ("bank_ew_cov_wide_t4"), setup = bank_setup)]
fn bank_threads(mut run: BankRun) -> BankRun {
    callgrind::start_instrumentation();
    let out = black_box(polars_online_budget_bank_rows(black_box(&mut run)));
    callgrind::stop_instrumentation();
    drop(out);
    run
}

library_benchmark_group!(
    name = hot_paths,
    benchmarks = [model, clock, bank, bank_threads]
);

main!(library_benchmark_groups = hot_paths);
