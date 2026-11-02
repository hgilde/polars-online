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
