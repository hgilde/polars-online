//! Instruction counts of the hot paths, under Callgrind, for `budgets.toml`.
//!
//!     cargo bench -p online-bench --bench instructions -- --output-format=json
//!
//! Linux only, where Valgrind runs: CI's `budgets` workflow runs it on every
//! push and pull request and holds each count to its budget
//! (`scripts/budgets.py`). Each case is set up outside the count
//! (`online_bench`'s `*_setup`), and Callgrind's entry point is the case's
//! measured function, so a count is the block of rows and nothing else.
//! Every case runs on one thread of each pool: Callgrind counts the thread
//! an entry point runs on, and a second thread's share of the work would be
//! missing from the count, and its spinning would make the count vary.
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
fn bank(mut run: BankRun) -> BankRun {
    let pool = online_polars::pool().expect("the bank's pool, built by the setup");
    black_box(pool.install(|| polars_online_budget_bank_rows(black_box(&mut run))));
    run
}

library_benchmark_group!(name = hot_paths, benchmarks = [model, clock, bank]);

main!(library_benchmark_groups = hot_paths);
