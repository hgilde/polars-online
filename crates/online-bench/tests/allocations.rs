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
//! instead, into one kept `Step`, which the models it matters for fill in
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
    eprintln!("{what}: {got:.2} allocations a row (bound {bound}, {before} before the fix)");
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

/// `deco` standardised the row and formed its block sums twice, once to
/// report and once to learn (D-6).
#[test]
fn deco_standardises_a_row_once() {
    let got = per_row(
        json!({"name": "m", "model": {"type": "deco"}, "features": names("x", 20),
               "clock": "t", "gap_cap": 100.0, "half_life": 1000.0}),
        20,
        0,
    );
    check("deco m=20", got, 14.0, 19.0);
}

/// `hmm` built `Π` twice on a learning row, a vector a matrix row each time
/// (D-7).
#[test]
fn hmm_builds_its_transition_matrix_once_a_row() {
    let got = per_row(
        json!({"name": "m", "model": {"type": "hmm", "n_components": 2, "precision_prior": 1.0},
               "features": names("x", 4), "clock": "t", "gap_cap": 100.0, "half_life": 1000.0}),
        4,
        0,
    );
    check("hmm K=2 d=4", got, 26.0, 29.0);
}

/// `lasso` copied each target's selection errors a row to take their
/// argmin (C-2). No solve falls in the counted block, so what is left is
/// the `Step`: `pred` and `lam_selected`.
#[test]
fn lasso_reads_its_selection_errors_in_place() {
    let got = per_row(
        json!({"name": "m", "model": {"type": "lasso", "penalties": [1.0, 0.1, 0.01, 0.001],
                                       "solve_every": 1e9, "max_rows_between_solves": 1000000},
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
/// at the default weight-share cadence), one allocation fewer than `step`.
/// `sgd` stays on the trait's default, which goes through `step`, until its
/// own `step_into`. Each row: the bound through `step_into`, the bound
/// through `step`, and `step`'s count before the change.
#[test]
fn the_cheap_models_step_into_a_kept_buffer_without_allocating() {
    for (id, into_bound, step_bound, before) in [
        ("seqtest", 0.0, 1.0, 1.0),
        ("pa", 0.0, 1.0, 1.0),
        ("kmeans", 0.0, 1.0, 1.0),
        ("ew_cov", 0.0, 1.0, 2.0),
        ("ewridge_k5", 1.48, 2.48, 2.48),
        ("ewridge_k20", 1.48, 2.48, 2.48),
        ("sgd", 1.0, 1.0, 1.0),
    ] {
        let (step, into) = (case_per_row(id, false), case_per_row(id, true));
        check(&format!("{id} through step_into"), into, into_bound, before);
        check(&format!("{id} through step"), step, step_bound, before);
        if id != "sgd" {
            assert!(
                into <= step - 1.0,
                "{id}: step_into {into} a row, step {step}: not one fewer"
            );
        }
    }
}
