//! Instruction counts of the hot paths, under Callgrind, for `budgets.toml`.
//!
//!     cargo bench -p online-bench --bench instructions -- --output-format=json
//!
//! Linux only, where Valgrind runs: CI's `budgets` workflow runs it on every
//! push and pull request and holds each count to its budget
//! (`scripts/budgets.py`). Each case is set up outside the count
//! (`online_bench`'s `*_setup`), and Callgrind's entry point is the case's
//! measured function, so a count is the block of rows and nothing else.
//! Every case runs on one thread of each pool: Callgrind collects from an
//! entry point on the thread that enters it, so a second thread's share of
//! the work would be missing from the count. A case on a pool of four
//! threads, counted on every thread between Callgrind's client requests,
//! was tried and taken out (2026-10-13): two runs of one binary differed
//! by 0.655%, past the quarter of the tolerance the check allows, since the
//! pool's idle workers spin and steal work for as long as the scheduler
//! lets them. Pool sizes are held instead by the bit-identity tests at 1,
//! 2, 4 and 8 threads and by the measured table in docs/PERFORMANCE.md.
//!
//! A case added here needs an entry in `budgets.toml`; its first CI run
//! reports its count, which is then committed as its budget.

use std::hint::black_box;

use gungraun::prelude::*;
use gungraun::{Callgrind, EntryPoint};
use online_bench::{
    BankRun, ClockRun, ModelRun, WindowsCase, bank_setup, clock_setup, model_setup,
    polars_online_budget_bank_rows, polars_online_budget_clock_rows,
    polars_online_budget_model_rows, polars_online_budget_windows_rows, windows_setup,
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
#[bench::ewridge_gram_t10(args = ("ewridge_gram_t10"), setup = model_setup)]
#[bench::huber_k20(args = ("huber_k20"), setup = model_setup)]
#[bench::sgd(args = ("sgd"), setup = model_setup)]
#[bench::sgd_huber(args = ("sgd_huber"), setup = model_setup)]
#[bench::pa(args = ("pa"), setup = model_setup)]
#[bench::seqtest(args = ("seqtest"), setup = model_setup)]
#[bench::bocpd(args = ("bocpd"), setup = model_setup)]
#[bench::kmeans(args = ("kmeans"), setup = model_setup)]
#[bench::micro(args = ("micro"), setup = model_setup)]
#[bench::kalman(args = ("kalman"), setup = model_setup)]
#[bench::ew_cov(args = ("ew_cov"), setup = model_setup)]
#[bench::lasso_k20(args = ("lasso_k20"), setup = model_setup)]
#[bench::rls(args = ("rls"), setup = model_setup)]
#[bench::quantile(args = ("quantile"), setup = model_setup)]
#[bench::ftrl(args = ("ftrl"), setup = model_setup)]
#[bench::holt(args = ("holt"), setup = model_setup)]
#[bench::marginal(args = ("marginal"), setup = model_setup)]
#[bench::marginal_bins_huber(args = ("marginal_bins_huber"), setup = model_setup)]
#[bench::ew_class(args = ("ew_class"), setup = model_setup)]
#[bench::deco(args = ("deco"), setup = model_setup)]
#[bench::hmm(args = ("hmm"), setup = model_setup)]
#[bench::rcov(args = ("rcov"), setup = model_setup)]
#[bench::audit_continuous(args = ("audit_continuous"), setup = model_setup)]
#[bench::bocpd_full(args = ("bocpd_full"), setup = model_setup)]
#[bench::ew_cov_lags(args = ("ew_cov_lags"), setup = model_setup)]
#[bench::corrchange_monitor_w2500(args = ("corrchange_monitor_w2500"), setup = model_setup)]
#[bench::corrchange_window_w100(args = ("corrchange_window_w100"), setup = model_setup)]
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
#[bench::bank_seqtest_groups(args = ("bank_seqtest_groups"), setup = bank_setup)]
#[bench::bank_ewridge_k5_groups(args = ("bank_ewridge_k5_groups"), setup = bank_setup)]
#[bench::formula_target_held1k(args = ("formula_target_held1k"), setup = bank_setup)]
#[bench::bank_ew_cov_wide_t1(args = ("bank_ew_cov_wide_t1"), setup = bank_setup)]
fn bank(mut run: BankRun) -> BankRun {
    let pool = online_polars::pool().expect("the bank's pool, built by the setup");
    black_box(pool.install(|| polars_online_budget_bank_rows(black_box(&mut run))));
    run
}

#[library_benchmark(config = counted_from("polars_online_budget_windows_rows"))]
#[bench::window_ewm_mean_w10k(args = ("window_ewm_mean_w10k"), setup = windows_setup)]
#[bench::window_ewm_sum_w10k(args = ("window_ewm_sum_w10k"), setup = windows_setup)]
fn windows(mut run: WindowsCase) -> WindowsCase {
    black_box(polars_online_budget_windows_rows(black_box(&mut run)));
    run
}

library_benchmark_group!(name = hot_paths, benchmarks = [model, clock, bank, windows]);

main!(library_benchmark_groups = hot_paths);
