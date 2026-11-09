//! What a row allocates on the paths review round 9 found allocating for
//! nothing (docs/PLAN.md task 270: D-5, D-6, D-7, F-3, C-2, M-4), counted by a
//! global allocator that keeps each thread's count of calls, so the tests
//! of this binary may run side by side.
//!
//! Each model case is built from its spec as the bank builds it, stepped
//! through a warm-up, and its allocations are counted over a block of rows
//! of `AnyModel::step`. The returned `Step` owns its `pred` (and `lasso`'s
//! `lam_selected`), which is one allocation a row the trait asks for and
//! these bounds include; the bank steps through `AnyModel::step_into`
//! instead, into one kept `Step`, which every model fills in
//! place (M-4, the last test). Each bound is the count the fix leaves; the
//! count before it is beside it, so a bound that loosens says what came
//! back.

use online_polars::online_core::{FeatureHealth, Specification};
use online_polars::{AnyModel, Spec, build_models};
use serde_json::{Value, json};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

fn count() {
    CALLS.with(|c| c.set(c.get() + 1));
}

fn calls() -> usize {
    CALLS.with(Cell::get)
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, and only counts besides.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// A seeded draw in `[-1, 1)`.
fn draw(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{i}")).collect()
}

fn model(value: Value) -> AnyModel {
    let mut spec: Spec = serde_json::from_value(value).expect("a spec");
    spec.check().expect("a valid spec");
    let mut built = build_models(&spec).expect("built");
    assert_eq!(built.len(), 1, "one instance");
    built.pop().expect("one").1
}

const WARM: usize = 2_000;
const ROWS: usize = 1_000;

/// Allocations a row of `AnyModel::step` over [`ROWS`] rows after [`WARM`],
/// of `k` features and `m` targets: the rows are drawn up front, outside
/// the count.
fn per_row(value: Value, k: usize, m: usize) -> f64 {
    per_row_with(value, k, m, false)
}

/// [`per_row`], through `AnyModel::step_into` into one kept `Step` where
/// `into`, as the bank steps a model.
fn per_row_with(value: Value, k: usize, m: usize, into: bool) -> f64 {
    let mut model = model(value);
    let mut s = 7u64;
    let rows: Vec<(Vec<f64>, Vec<Option<f64>>)> = (0..WARM + ROWS)
        .map(|_| {
            let x: Vec<f64> = (0..k).map(|_| draw(&mut s)).collect();
            let y = (0..m)
                .map(|j| Some(x.get(j % k.max(1)).map_or(0.0, |v| *v) + draw(&mut s)))
                .collect();
            (x, y)
        })
        .collect();
    for (i, (x, y)) in rows[..WARM].iter().enumerate() {
        std::hint::black_box(model.step(x, y, if i == 0 { 0.0 } else { 1.0 }, 1.0));
    }
    let mut kept = online_polars::online_core::Step::default();
    // One row into the buffer before the count, which sizes it.
    model.step_into(&rows[WARM - 1].0, &rows[WARM - 1].1, 1.0, 0.0, &mut kept);
    let before = calls();
    for (x, y) in &rows[WARM..] {
        if into {
            model.step_into(x, y, 1.0, 1.0, &mut kept);
            std::hint::black_box(&kept);
        } else {
            std::hint::black_box(model.step(x, y, 1.0, 1.0));
        }
    }
    (calls() - before) as f64 / ROWS as f64
}

fn check(what: &str, got: f64, bound: f64, before: f64) {
    eprintln!("{what}: {got:.3} allocations a row (bound {bound}, {before} before the fix)");
    assert!(
        got <= bound,
        "{what}: {got} allocations a row, past {bound}"
    );
}

/// `ew_cov` with lags: the row's deviations, a past row's, and the row
/// pushed into the ring were three allocations a row; the ring's oldest row
/// now holds the new one (D-5). The rest is the `Step`, which `ew_cov`
/// without lags allocates too.
#[test]
fn ew_cov_with_lags_allocates_no_more_than_without() {
    let spec = |lags: bool| {
        let mut model = json!({"type": "ew_cov"});
        model["stats"] = json!(["mean", "corr"]);
        if lags {
            model["lags"] = json!([1, 2, 3, 4, 5]);
        }
        json!({"name": "m", "model": model, "features": names("x", 20), "clock": "t",
               "gap_cap": 100.0, "half_life": 1000.0})
    };
    let without = per_row(spec(false), 20, 0);
    let with = per_row(spec(true), 20, 0);
    check("ew_cov k=20, no lags", without, without, without);
    check("ew_cov k=20, lags 1..5", with, without, without + 3.0);
}

/// `huber` under `emit_huber_scale` reports each target's scale a row, into
/// the vector the kept `Step` brings back from the row before (task 285),
/// so stepped into one buffer it allocates no more a row than without the
/// scales; a new vector of scales would be one more a row (the mutants of
/// 1b7e5178, the guard on reusing it).
#[test]
fn huber_reports_its_scales_into_the_kept_buffer() {
    let spec = |emit: bool| {
        json!({"name": "m", "model": {"type": "huber", "emit_huber_scale": emit},
               "targets": ["y0", "y1"], "features": names("x", 20), "clock": "t",
               "half_life": 1000.0, "gap_cap": 100.0, "min_weight": 20.0})
    };
    let without = per_row_with(spec(false), 20, 2, true);
    let with = per_row_with(spec(true), 20, 2, true);
    check(
        "huber k=20, two targets, scales emitted",
        with,
        without,
        without + 1.0,
    );
}

/// `deco` standardised the row and formed its block sums twice, once to
/// report and once to learn (D-6): 19 allocations a row, 13 after. Its
/// density's matrix, factor and faer's scratch are kept since (task 298):
/// three fewer. Its density's quadratic form solves in a kept column, and
/// the row's own vectors are kept (task 304): what is left is the `Step`.
#[test]
fn deco_standardises_a_row_once() {
    let got = per_row(
        json!({"name": "m", "model": {"type": "deco"}, "features": names("x", 20),
               "clock": "t", "gap_cap": 100.0, "half_life": 1000.0}),
        20,
        0,
    );
    check("deco m=20", got, 1.0, 19.0);
}

/// `hmm` built `Π` twice on a learning row, a vector a matrix row each time
/// (D-7): 29 allocations a row, 24 after. Its states' factors, their
/// matrices, faer's scratch and the list of factors are kept since (task
/// 298): seven fewer. Each state's quadratic form solves in a kept column,
/// and the row's own vectors are kept (task 304): what is left is the `Step`'s `pred`,
/// a vector and its one growth, which the bank's kept `Step` does not pay.
#[test]
fn hmm_builds_its_transition_matrix_once_a_row() {
    let got = per_row(
        json!({"name": "m", "model": {"type": "hmm", "n_components": 2, "precision_prior": 1.0},
               "features": names("x", 4), "clock": "t", "gap_cap": 100.0, "half_life": 1000.0}),
        4,
        0,
    );
    check("hmm K=2 d=4", got, 2.0, 29.0);
}

/// `lasso` copied each target's selection errors a row to take their
/// argmin (C-2). No solve falls in the counted block, so what is left is
/// the `Step`: `pred` and `lam_selected`. The row cap of 1,800 ends the
/// first period -- solved on the default schedule (task 278) -- on the
/// warm-up's 1,800th row, and the next solve is 1,800 rows on, past the
/// block's last.
#[test]
fn lasso_reads_its_selection_errors_in_place() {
    let got = per_row(
        json!({"name": "m", "model": {"type": "lasso", "penalties": [1.0, 0.1, 0.01, 0.001],
                                       "solve_every": 1e9, "max_rows_between_solves": 1800},
               "targets": names("y", 4), "features": names("x", 20), "clock": "t",
               "gap_cap": 100.0, "half_life": 100.0, "min_weight": 20.0}),
        20,
        4,
    );
    check("lasso k=20, 4 targets", got, 2.0, 6.0);
}

/// The bank's feature health and specification tests copied the row into
/// a new vector a row (F-3): kept buffers now, and nothing a row.
#[test]
fn feature_health_and_specification_allocate_nothing_a_row() {
    let k = 10;
    let mut h = FeatureHealth::new(k);
    let mut spec = Specification::new(k, 0, 4, 0);
    let mut s = 3u64;
    let rows: Vec<Vec<f64>> = (0..WARM + ROWS)
        .map(|_| (0..k).map(|_| draw(&mut s)).collect())
        .collect();
    let feed = |h: &mut FeatureHealth, spec: &mut Specification, rows: &[Vec<f64>]| {
        for (i, x) in rows.iter().enumerate() {
            // Every seventh row of weight 0, which the health reads as zeros.
            let w = if i % 7 == 3 { 0.0 } else { 1.0 };
            // Two decays in turn, so the slow factor is not one value.
            let lam = if i % 2 == 0 { 0.99 } else { 0.98 };
            h.update(x, lam, w);
            spec.update(x, x[0], x[1], lam, w);
        }
    };
    feed(&mut h, &mut spec, &rows[..WARM]);
    let before = calls();
    feed(&mut h, &mut spec, &rows[WARM..]);
    let got = (calls() - before) as f64 / ROWS as f64;
    check("feature health and specification, k=10", got, 0.0, 1.86);
}

/// A bench case's spec, `k` features and `m` targets, through
/// [`per_row_with`].
fn case_per_row(id: &str, into: bool) -> f64 {
    let spec = online_bench::spec(id);
    let (k, m) = (spec.k(), spec.m());
    let value = serde_json::to_value(&spec).expect("a spec serializes");
    per_row_with(value, k, m, into)
}

/// The models whose step writes into the caller's `Step` (M-4): `pred` was a
/// new vector a row, and `ew_cov`'s correlations' standard deviations a
/// second, which the model now keeps, so its `step` allocates one fewer too.
/// Stepped into one kept buffer, as the bank steps them, the five allocate
/// nothing a row but `ewridge`'s solves (a solve every 17 rows or so here,
/// at the default weight-share cadence), one allocation fewer than `step`;
/// two fewer a solve since its solves keep their factors' working space
/// (task 301).
/// `sgd` joined them once task 271's rewrite of its Huber scale had landed
/// (task 270 (h)), under its squared loss and under Huber's, whose scale is
/// a sketch of `|e|` aged and fed each row. Then every model's step was
/// written into the caller's `Step` (task 285): `holt`, `ftrl`, `rls` and
/// `kalman` allocate nothing a row, `micro` its rare new summary, `huber`
/// what its solves do -- nothing, since they keep their working space
/// (task 297) -- and `lasso` its solves without its `pred` and
/// `lam_selected`, two fewer. Each row: the bound through `step_into`, the
/// bound through `step`, and `step`'s count before the change.
#[test]
fn the_cheap_models_step_into_a_kept_buffer_without_allocating() {
    for (id, into_bound, step_bound, before) in [
        ("seqtest", 0.0, 1.0, 1.0),
        ("pa", 0.0, 1.0, 1.0),
        ("kmeans", 0.0, 1.0, 1.0),
        ("ew_cov", 0.0, 1.0, 2.0),
        ("ewridge_k5", 1.30, 2.30, 2.48),
        ("ewridge_k20", 1.30, 2.30, 2.48),
        ("sgd", 0.0, 1.0, 1.0),
        ("sgd_huber", 0.0, 1.0, 1.0),
        ("holt", 0.0, 1.0, 1.0),
        ("ftrl", 0.0, 1.0, 1.0),
        ("rls", 0.0, 1.0, 1.0),
        ("kalman", 0.0, 1.0, 1.0),
        ("micro", 0.05, 1.05, 1.05),
        ("huber_k20", 0.0, 1.0, 1.77),
        ("lasso_k20", 0.77, 2.77, 2.77),
    ] {
        let (step, into) = (case_per_row(id, false), case_per_row(id, true));
        check(&format!("{id} through step_into"), into, into_bound, before);
        check(&format!("{id} through step"), step, step_bound, before);
        // Counts over `ROWS`, compared within their rounding.
        assert!(
            into + 1.0 <= step + 1e-9,
            "{id}: step_into {into} a row, step {step}: not one fewer"
        );
    }
}

/// `bocpd` keeps its row's working space -- the prior read for the row,
/// each run's log predictive and weight, the posterior, the mixture, the
/// new log joint -- and the runs `prune` drops, each the next row's empty
/// run (task 292): 16 allocations a row under the diagonal emission, and
/// none now. Under the full covariance each run's predictive formed its
/// mean, its scale and the row's deviation, and solved for the quadratic
/// form in a matrix and a vector of its own; what was left, 21.24 a row,
/// was the factor of each run's scale, a matrix and faer's scratch made for
/// every run, which is written over the last run's now (task 298).
#[test]
fn bocpd_keeps_its_rows_working_space() {
    for (id, bound, before) in [("bocpd", 0.0, 16.0), ("bocpd_full", 0.0, 72.63)] {
        check(id, case_per_row(id, true), bound, before);
    }
}

/// `ew_class` formed its class weights, its class scores, the row's
/// deviations, the matrix a stale class factorizes and each quadratic
/// form's column anew on every row, and every solve made a list of its one
/// band of right-hand sides (task 293): kept buffers now, and what was
/// left, 2 a row, was the learned class's new factor, its matrix and
/// faer's scratch, which is written over the factor the class dropped now
/// (task 298).
/// `quantile` kept its nudge's column already; its solves' lists went the
/// same way, and then its solves' and its nudges' band systems and factors
/// (task 297): what is left is a system's first build after a refit or a
/// save drops one. Counted over each bench case's own measured block, as
/// the bank steps it, whose labels name classes; this file's rows name
/// none, and a class with no row scores nothing.
#[test]
fn ew_class_and_quantile_keep_their_buffers_between_rows() {
    for (id, bound, before) in [("ew_class", 0.0, 15.0), ("quantile", 0.08, 2.40)] {
        let mut run = online_bench::model_setup(id);
        let start = calls();
        std::hint::black_box(online_bench::polars_online_budget_model_rows(&mut run));
        let got = (calls() - start) as f64 / online_bench::ROWS as f64;
        check(id, got, bound, before);
    }
}

/// `huber`'s solves formed their band system -- its scales, kept columns
/// and matrix -- the right-hand side, the factor's matrix and faer's
/// scratch for it, the solution, the coefficients and the data shares'
/// lists anew at every solve, and the coefficient table at every solve
/// (task 297): about 12 allocations a solve, 16 a row under three feature
/// sets, whose cost moved with the heap's layout from one process to the
/// next. They are kept between solves now, a spare factor of each order
/// written over and handed to the data shares, whose factor comes back as
/// the next spare. Counted over each bench case's own measured block, as
/// the bank steps it; the refit's block holds its one refit, whose passes
/// still make their own row buffers. Each pass made its coefficient vector
/// and a list of the buffer's history to replay as well, until a pass took
/// over the vector of the fit it replaced and the replay walked the history
/// in place (task 300): 110 a refit block to 95, 253 to 244 for a quantile
/// fit's. A quantile fit's solves go the same way.
#[test]
fn huber_keeps_its_solves_working_space() {
    for (id, bound, before) in [
        ("huber_k20", 0.001, 0.83),
        ("huber_k20_sets", 0.001, 2.55),
        ("huber_k20_sum", 0.001, 0.90),
        ("huber_k20_refit", 0.095, 1.02),
        ("quantile", 0.08, 1.48),
        ("quantile_refit", 0.244, 1.77),
    ] {
        let mut run = online_bench::model_setup(id);
        let start = calls();
        std::hint::black_box(online_bench::polars_online_budget_model_rows(&mut run));
        let got = (calls() - start) as f64 / online_bench::ROWS as f64;
        check(id, got, bound, before);
    }
}

/// `hmm` factorized each state's matrix anew on every learned row, and
/// `deco` its block matrix `S` on every row: a new factor's matrix, faer's
/// scratch for it and the matrix it read, each made and dropped (task 298).
/// The dropped factors are kept and written over now, with the scratch and
/// the matrix: `hmm`'s two states' three allocations each, and the list
/// its factors are held in, and `deco`'s three. Then each quadratic form, a
/// matrix and a vector a form, went to a kept column, and each row's own
/// vectors with them, so neither allocates a row (task 304). Counted over
/// each bench case's own measured block, as the bank steps it.
#[test]
fn hmm_and_deco_keep_their_factors_working_space() {
    for (id, bound, before) in [("hmm", 0.0, 22.0), ("deco", 0.0, 12.0)] {
        let mut run = online_bench::model_setup(id);
        let start = calls();
        std::hint::black_box(online_bench::polars_online_budget_model_rows(&mut run));
        let got = (calls() - start) as f64 / online_bench::ROWS as f64;
        check(id, got, bound, before);
    }
}

/// `hmm` under the shared covariance (`covariance_type = "tied"`) made the
/// states' weights, the shared
/// matrix and the row's deviations from each state's mean afresh on every
/// row, and the factor of that matrix with faer's scratch, its forms'
/// matrix and their list (review 11's S5-1). Each is kept in the row's
/// working space now, the factor written over the last row's, so a row
/// stepped into a kept `Step` allocates nothing, as under the full and
/// diagonal covariances (task 316).
#[test]
fn hmm_with_a_shared_covariance_allocates_nothing_a_row() {
    for (covariance, before) in [("tied", 7.0), ("full", 0.0), ("diag", 0.0)] {
        let got = per_row_with(
            json!({"name": "m", "model": {"type": "hmm", "n_components": 2,
                                           "covariance_type": covariance, "precision_prior": 1.0},
                   "features": names("x", 4), "clock": "t", "gap_cap": 100.0, "half_life": 1000.0}),
            4,
            0,
            true,
        );
        check(&format!("hmm {covariance} K=2 d=4"), got, 0.0, before);
    }
}

/// `ewridge`'s solves made the factor's matrix and faer's scratch for it
/// afresh at every solve, and dropped the factor the last solve's data
/// shares held (task 301): two allocations a solve, 69 solves in the
/// `ewridge_k5` and `ewridge_k20` blocks at the default weight-share
/// cadence and 3 in `ewridge_gram_t10`'s. The factors the solves have done
/// with are kept and written over now, with the scratch; what is left is
/// the rest of each solve's working lists. Counted over each bench case's
/// own measured block, as the bank steps it.
#[test]
fn ewridge_keeps_its_solves_working_space() {
    for (id, bound, before) in [
        ("ewridge_k5", 1.518, 1.656),
        ("ewridge_k20", 1.518, 1.656),
        ("ewridge_gram_t10", 1.241, 1.247),
    ] {
        let mut run = online_bench::model_setup(id);
        let start = calls();
        std::hint::black_box(online_bench::polars_online_budget_model_rows(&mut run));
        let got = (calls() - start) as f64 / online_bench::ROWS as f64;
        check(id, got, bound, before);
    }
}

/// `ewridge`'s noise gate made a list of each Gram's Kish size on every read
/// (task 304): one allocation a row on the bank's gate, two under a formula
/// target. With no window the live accumulators are read where they stand,
/// and the gate allocates nothing. Past the first solve, so it reads a size.
#[test]
fn ewridge_reads_its_kish_sizes_in_place() {
    let run = online_bench::model_setup("ewridge_k5");
    let mut out = Vec::new();
    assert!(run.model.error_inflation_into(&mut out));
    assert!(
        out.iter().all(|v| v.is_finite()),
        "the gate reads a size: {out:?}"
    );
    let start = calls();
    for _ in 0..ROWS {
        run.model.error_inflation_into(&mut out);
    }
    let got = (calls() - start) as f64 / ROWS as f64;
    check("ewridge_k5's noise gate", got, 0.0, 1.0);
}

/// `corrchange` copied the row twice a row, one copy dropped, and its monitor
/// copied every ring row at the span's close, whose 2,500 rows close once in
/// its case's block (task 304): 4,508 allocations a block. The row is read
/// where it is, copied only to enter the ring, and the span's rows are read
/// where they are, through one list of them. Counted over each bench case's
/// own measured block, as the bank steps it.
#[test]
fn corrchange_reads_its_rows_where_they_are() {
    for (id, bound, before) in [
        ("corrchange_monitor_w2500", 1.008, 4.508),
        ("corrchange_window_w100", 0.0, 3.812),
    ] {
        let mut run = online_bench::model_setup(id);
        let start = calls();
        std::hint::black_box(online_bench::polars_online_budget_model_rows(&mut run));
        let got = (calls() - start) as f64 / online_bench::ROWS as f64;
        check(id, got, bound, before);
    }
}

/// The window operators dropped a group's list of pending rows each time they
/// were resolved, and the next row's push made it again (task 304): 1 of about
/// 1.4 allocations a row, on the thread that feeds. What is left is the feed's
/// lazy queries' and output's, per block. Counted over each bench case's own
/// measured block, on this thread: a query's pool threads count on their own.
#[test]
fn the_window_operators_keep_their_pending_list() {
    for id in online_bench::WINDOW_CASES {
        let mut run = online_bench::windows_setup(id);
        let start = calls();
        let out = std::hint::black_box(online_bench::polars_online_budget_windows_rows(&mut run));
        let got = (calls() - start) as f64 / online_bench::ROWS as f64;
        drop(out);
        check(id, got, 0.6, 1.42);
    }
}

/// While a refit waits, the MAD is read from a tree of the residuals its
/// buffer holds (task 323), each row inserted and the median walked: the
/// tree's nodes are reserved with the buffer, at its first learned row, so
/// a row allocates nothing for it. Counted over 1,000 rows of a refit
/// waiting past them (a count of 5,000) beside the same spec with the refit
/// off, for each model that keeps the read -- `huber`, `quantile`,
/// `sgd(loss="huber")` and `marginal`'s Huber bins on given edges -- the
/// waiting refit allocates no more a row, but for the bins' own hold of
/// each row (their refit holds the rows whole); the deviations beside it
/// are reserved with the hold.
#[test]
fn the_held_median_allocates_nothing_a_row() {
    let bins = json!({"type": "marginal", "bin_edges": [[-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]], "bin_stat": "huber"});
    for (kind, base, key) in [
        ("huber", json!({"type": "huber"}), "huber_refit_rows"),
        (
            "quantile",
            json!({"type": "quantile", "quantile": 0.5}),
            "quantile_refit_rows",
        ),
        (
            "sgd",
            json!({"type": "sgd", "loss": "huber", "learning_rate": 0.01}),
            "huber_refit_rows",
        ),
        ("marginal", bins, "bin_refit_rows"),
    ] {
        let spec = |rows: u32| {
            let mut model = base.clone();
            model[key] = json!(rows);
            json!({"name": "m", "model": model, "targets": ["y0"], "features": names("x", 3), "half_life": 500.0})
        };
        let waiting = per_row_with(spec(5_000), 3, 1, true);
        let off = per_row_with(spec(0), 3, 1, true);
        // The bins hold each row whole for their refit, as they did before
        // the tree: its features and targets, two vectors a row, and the
        // list of rows grown by doubling up to the count, at most
        // `log2(5000 / 4) + 1 = 11` times.
        let hold = if kind == "marginal" {
            2.0 + 11.0 / ROWS as f64
        } else {
            0.0
        };
        check(
            &format!("{kind}, its refit waiting (off: {off:.3})"),
            waiting,
            off + hold,
            off,
        );
    }
}

/// `corrchange(kind="window")` reads its statistic from merged block
/// summaries (task 328), and each report took a buffer of two windows'
/// summaries and its working space, each row's copy into the ring a vector
/// of its own, each new chunk's suffix sums a list, and each permutation
/// draw two windows' correlations (task 335, F7). Now kept between rows: a
/// row allocates nothing, at 4 and 20 columns, with a fixed critical value
/// or the permutation's redrawn every 100 rows (the default cadence, a
/// window's rows), and with windows read whole (5 columns, 20 rows: fewer
/// blocks than the merged form pays for).
#[test]
fn the_window_kind_allocates_nothing_a_row() {
    let mut counts = Vec::new();
    for (k, w, crit, before) in [
        (4, 100, true, 2.060),
        (20, 100, true, 2.052),
        (4, 100, false, 6.090),
        (20, 100, false, 6.082),
        (5, 20, false, 23.150),
    ] {
        let mut model = json!({"type": "corrchange", "kind": "window", "window_rows": w});
        if crit {
            model["crit"] = json!(0.5);
        }
        let spec = json!({"name": "m", "model": model, "features": names("x", k)});
        let got = per_row_with(spec, k, 0, true);
        counts.push((
            format!("corrchange window, {k} columns, w {w}, fixed crit {crit}"),
            got,
            before,
        ));
    }
    for (what, got, before) in counts {
        check(&what, got, 0.0, before);
    }
}

/// With learned edges and a refit count no larger than the warm-up, the
/// freeze replays the warm-up into a new Huber histogram, holding each
/// row's deviations, and the refit runs on them in the same row. The hold
/// was reserved for the refit's count and grew by doubling through the
/// rest of the replay: at a warm-up of 400 rows and a count of 50, three
/// doublings of its three lists and each target's tree (task 335, F3). Now
/// reserved for the rows replayed, the freeze row allocates as many times
/// at a count of 50 as at the warm-up's own 400.
#[test]
fn the_freeze_reserves_the_rows_it_replays() {
    let (k, m, warm) = (3, 2, 400);
    let freeze_row = |refit: u32| {
        let spec = json!({"name": "m", "model": {"type": "marginal", "bins": 4,
            "bin_warm_rows": warm, "bin_stat": "huber", "bin_refit_rows": refit},
            "targets": names("y", m), "features": names("x", k), "half_life": 500.0});
        let mut model = model(spec);
        let mut s = 5u64;
        let rows: Vec<(Vec<f64>, Vec<Option<f64>>)> = (0..warm)
            .map(|_| {
                let x: Vec<f64> = (0..k).map(|_| draw(&mut s)).collect();
                let y = (0..m).map(|j| Some(x[j] + draw(&mut s))).collect();
                (x, y)
            })
            .collect();
        let mut kept = online_polars::online_core::Step::default();
        for (x, y) in &rows[..warm - 1] {
            model.step_into(x, y, 1.0, 1.0, &mut kept);
        }
        let (x, y) = &rows[warm - 1];
        let before = calls();
        model.step_into(x, y, 1.0, 1.0, &mut kept);
        let got = calls() - before;
        let s = model.refit_status();
        assert!(s.iter().all(|r| r.done && r.passes > 0), "{s:?}");
        got
    };
    let (whole, short) = (freeze_row(warm as u32), freeze_row(50));
    eprintln!("the freeze row: {whole} allocations at a count of {warm}, {short} at 50");
    assert_eq!(short, whole, "the hold grew past its reservation");
}

/// Allocations a row of a windowed model of `k` features stepped into one
/// kept `Step`, as the bank steps it, over [`ROWS`] rows after [`WARM`],
/// which fill its window of 200 clock units (a row a unit) ten times: `y` is
/// the class of `x0`'s sign where `label`, and `x0` plus noise otherwise.
fn windowed_per_row(model: Value, k: usize, label: bool) -> f64 {
    let ew_cov = model["type"] == "ew_cov";
    let mut spec = json!({"name": "m", "model": model, "features": names("x", k),
        "clock": "t", "gap_cap": 10.0, "half_life": 1e6});
    if !ew_cov {
        spec["targets"] = json!(names(if label { "lab" } else { "y" }, 1));
    }
    let mut model = self::model(spec);
    let mut s = 11u64;
    let rows: Vec<(Vec<f64>, Vec<Option<f64>>)> = (0..WARM + ROWS)
        .map(|_| {
            let x: Vec<f64> = (0..k).map(|_| draw(&mut s)).collect();
            let y = if label {
                f64::from(u8::from(x[0] > 0.0))
            } else {
                x[0] + 0.1 * draw(&mut s)
            };
            (x, vec![Some(y)])
        })
        .collect();
    let mut kept = online_polars::online_core::Step::default();
    for (x, y) in &rows[..WARM] {
        model.step_into(x, y, 1.0, 1.0, &mut kept);
    }
    let before = calls();
    for (x, y) in &rows[WARM..] {
        model.step_into(x, y, 1.0, 1.0, &mut kept);
        std::hint::black_box(&kept);
    }
    (calls() - before) as f64 / ROWS as f64
}

/// A windowed model reads its window, `A(t) − f·A(u)`, on every row it
/// scores or solves, and formed it afresh each time (task 374): each
/// accumulator cloned, and its deviations, means, co-moments and
/// unresolved variances made and dropped, 11 allocations an accumulator;
/// `ewridge` and `lasso` their cross-moments and target moments besides,
/// on every solve. The view is kept between rows now and written over in
/// place; what is left is the ring's (a snapshot every 12 rows here, 0.083
/// a row, which `marginal`, whose window is no view, shows alone) and,
/// for `ewridge` and `lasso`, what a solve allocates of its own. `ew_class`
/// under `tied` formed its pooled matrix, its classes, their deviations and
/// a factor and its forms on every row, windowed or not: kept too. Each:
/// the bound in a release build and in a debug one, where the ring checks
/// every snapshot a view writes against the snapshot it forms, which
/// allocates that snapshot (`window/flat.rs`), and the count before, in a
/// release build.
#[test]
fn a_windowed_row_reads_its_window_in_kept_buffers() {
    let w = 200.0;
    let class = |shape: &str| {
        json!({"type": "ew_class", "classes": ["0", "1"], "precision_prior": 0.1,
               "covariance_type": shape, "window_size": w})
    };
    for (what, model, label, bound, before) in [
        (
            "ew_class full",
            class("full"),
            true,
            (0.084, 15.084),
            23.084,
        ),
        (
            "ew_class diag",
            class("diag"),
            true,
            (0.083, 14.083),
            19.083,
        ),
        (
            "ew_class tied",
            class("tied"),
            true,
            (0.084, 15.084),
            30.084,
        ),
        (
            "ew_cov",
            json!({"type": "ew_cov", "stats": ["mean", "corr"], "window_size": w}),
            false,
            (0.083, 10.083),
            11.083,
        ),
        (
            "ewridge",
            json!({"type": "ewridge", "window_size": w}),
            false,
            (9.432, 42.432),
            19.786,
        ),
        (
            "lasso",
            json!({"type": "lasso", "penalties": [0.1, 0.0], "window_size": w}),
            false,
            (9.092, 43.092),
            18.778,
        ),
        (
            "marginal",
            json!({"type": "marginal", "window_size": w}),
            false,
            (0.083, 17.083),
            0.083,
        ),
    ] {
        let got = windowed_per_row(model, 4, label);
        let bound = if cfg!(debug_assertions) {
            bound.1
        } else {
            bound.0
        };
        check(&format!("windowed {what}, 4 features"), got, bound, before);
    }
}
