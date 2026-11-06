//! The heap a window's snapshot holds against what `window_budget` counts of
//! it, for every model with a window, at a few features and at many.
//!
//! The budget counts the numbers a snapshot holds (`Footprint`). The heap is
//! the allocator's, as requested: what the model holds with its ring full,
//! less what it held with one snapshot, over the snapshots between. A
//! snapshot held as a struct of its own vectors took several times its
//! numbers: an `ewridge` of three features and one target held 1,207 B a
//! snapshot against 376 B counted, `marginal` of three features 701 B
//! against 120 B, before the allocator's own rounding of each vector. The
//! ring now holds its snapshots as words in blocks, so the heap is the count
//! and a small fixed overhead, and less than the count where a co-moment
//! matrix is written as its lower triangle.
//!
//! `WINDOW_MEMORY_REPORT=1` prints every case's figures;
//! `WINDOW_MEMORY_ROWS` sets the snapshots a case fills its ring with.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicIsize, Ordering};

use online_polars::online_core::*;
use online_polars::{Bank, Spec};
use polars::prelude::{Column, DataFrame};

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, and only counts besides.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The counts are process-wide, so the cases take turns.
static TURN: Mutex<()> = Mutex::new(());

/// Longer than any stream here: every row stays inside the window.
const WINDOW: f64 = 1e12;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn rows() -> usize {
    std::env::var("WINDOW_MEMORY_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000)
}

/// What one case measured: per snapshot, the heap and the budget's count.
struct Measured {
    heap: f64,
    counted: f64,
}

/// The model `make` builds, stepped over `rows` rows of `k` features and
/// `t` targets (each target an integer class below `classes` where that is
/// set), a snapshot a row.
fn measure<M: OnlineModel>(
    make: impl Fn() -> M,
    k: usize,
    t: usize,
    classes: Option<usize>,
    rows: usize,
) -> Measured {
    let _turn = TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut s = 349u64;
    let mut x = vec![0.0; k];
    let mut y = vec![None; t];
    let mut out = Step::default();
    let mut m = make();
    let mut step = |m: &mut M, i: usize| {
        for v in x.iter_mut() {
            *v = lcg(&mut s);
        }
        for (j, v) in y.iter_mut().enumerate() {
            *v = Some(match classes {
                Some(c) => ((i + j) % c) as f64,
                None => 0.5 * x[0] - 0.25 * x[k - 1] + 0.1 * lcg(&mut s),
            });
        }
        m.step_into(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0, &mut out);
    };
    // Warm every lazily sized buffer, then read the heap with one snapshot
    // held and with `rows`.
    step(&mut m, 0);
    let (live_1, shadow) = (LIVE.load(Ordering::Relaxed), m.window_shadow());
    let shadow = shadow.expect("a windowed model");
    let (bytes_1, held_1) = (shadow.bytes(), shadow.snapshots());
    drop(shadow);
    for i in 1..rows {
        step(&mut m, i);
    }
    let live_n = LIVE.load(Ordering::Relaxed);
    let shadow = m.window_shadow().expect("a windowed model");
    let (bytes_n, held_n) = (shadow.bytes(), shadow.snapshots());
    assert_eq!(
        held_n - held_1,
        rows - 1,
        "a snapshot a row, all inside the window"
    );
    let n = (held_n - held_1) as f64;
    Measured {
        heap: (live_n - live_1) as f64 / n,
        counted: (bytes_n - bytes_1) as f64 / n,
    }
}

fn ewridge(k: usize, t: usize) -> EwRidge {
    EwRidge::new(EwRidgeCfg {
        n_features: k,
        n_targets: t,
        fit_intercept: true,
        decay: Decay::Halflife(1e6),
        ridge: vec![1e-6],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: k as f64 + 1.0,
        solve_every: 1e18,
        max_rows_between_solves: u32::MAX,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: Some(WINDOW),
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap()
}

fn lasso(k: usize, t: usize) -> Lasso {
    Lasso::new(LassoCfg {
        min_weight_holds_learning: false,
        n_features: k,
        n_targets: t,
        fit_intercept: true,
        decay: Decay::Halflife(1e6),
        lasso_path: vec![0.1, 0.01, 0.0],
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: k as f64 + 1.0,
        target_min_weight: Vec::new(),
        solve_every: 1e18,
        max_rows_between_solves: u32::MAX,
        solve_share: None,
        window: Some(WINDOW),
        window_every: None,
        max_rows_between_snapshots: None,
        max_iter: 100,
        tol: 1e-10,
        target_gaps: TargetGaps::OwnRows,
    })
    .unwrap()
}

fn ew_cov(k: usize) -> EwCovModel {
    EwCovModel::new(EwCovCfg {
        bias: false,
        n_features: k,
        decay: Decay::Halflife(1e6),
        stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
        min_weight: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: vec![],
        window: Some(WINDOW),
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap()
}

fn ew_class(k: usize, covariance: Covariance) -> EwClass {
    EwClass::new(EwClassCfg {
        n_features: k,
        n_classes: 3,
        decay: Decay::Halflife(1e6),
        min_weight: 3.0,
        covariance,
        precision_prior: 0.1,
        window: Some(WINDOW),
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap()
}

fn marginal(k: usize, t: usize, lags: Vec<usize>) -> Marginal {
    Marginal::new(MarginalCfg {
        n_features: k,
        n_targets: t,
        decay: Decay::Halflife(1e6),
        min_weight: vec![3.0; t],
        lags,
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: FeatureMomentLayout::PerTarget,
        window: Some(WINDOW),
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap()
}

/// Every windowed model, at a few features and at many: its name and what
/// it measured.
fn cases(rows: usize) -> Vec<(String, Measured)> {
    let mut out = Vec::new();
    for (k, t) in [(3, 1), (30, 3)] {
        out.push((
            format!("ewridge k={k} t={t}"),
            measure(|| ewridge(k, t), k, t, None, rows),
        ));
        out.push((
            format!("lasso k={k} t={t}"),
            measure(|| lasso(k, t), k, t, None, rows),
        ));
        out.push((
            format!("marginal k={k} t={t}"),
            measure(|| marginal(k, t, vec![]), k, t, None, rows),
        ));
    }
    out.push((
        "marginal k=3 t=1 lags=[1, 2]".into(),
        measure(|| marginal(3, 1, vec![1, 2]), 3, 1, None, rows),
    ));
    for k in [4, 30] {
        out.push((
            format!("ew_cov k={k}"),
            measure(|| ew_cov(k), k, 0, None, rows),
        ));
    }
    for k in [3, 20] {
        for (name, c) in [("full", Covariance::Full), ("diag", Covariance::Diagonal)] {
            out.push((
                format!("ew_class k={k} {name}"),
                measure(|| ew_class(k, c), k, 1, Some(3), rows),
            ));
        }
    }
    out
}

/// A snapshot's heap is the budget's count and a small fixed overhead: at
/// most an eighth over the count, the blocks' spare room, and 96 bytes, the
/// place each snapshot is keyed by (its clock, its stamp and its running
/// product).
#[test]
fn a_snapshot_holds_about_what_the_budget_counts() {
    let report = std::env::var_os("WINDOW_MEMORY_REPORT").is_some();
    let mut over = Vec::new();
    for (name, m) in cases(rows()) {
        if report {
            println!(
                "{name:<30} heap {:>9.1} B  counted {:>9.1} B  ratio {:.2}",
                m.heap,
                m.counted,
                m.heap / m.counted
            );
        }
        if m.heap > 1.125 * m.counted + 96.0 {
            over.push(format!(
                "{name}: {:.0} B held, {:.0} B counted",
                m.heap, m.counted
            ));
        }
    }
    assert!(
        over.is_empty(),
        "snapshots held past their count: {over:#?}"
    );
}

/// A bank's windowed ridge, as a user runs one: its model's ring and the
/// ring of the residual spread beside it, through `Bank::fit_predict` in
/// chunks of 500 rows, with every row inside the window. Its heap a
/// snapshot, against the count of both rings' snapshots, stays within the
/// same bound.
#[test]
fn a_banks_windowed_ridge_holds_about_what_the_budget_counts() {
    let _turn = TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rows = rows();
    let spec: Spec = serde_json::from_value(serde_json::json!({
        "name": "m",
        "model": {"type": "ewridge", "window_size": WINDOW},
        "targets": ["y"],
        "features": ["x0", "x1", "x2"],
        "clock": "t",
        "gap_cap": 100.0,
        "half_life": 1e9,
        "min_weight": 4.0,
    }))
    .expect("a spec");
    let mut bank = Bank::new(vec![spec]).expect("a bank");
    let mut s = 349u64;
    let mut chunk = |from: usize, n: usize| {
        let mut cols: Vec<Column> = (0..3)
            .map(|j| {
                Column::new(
                    format!("x{j}").into(),
                    (0..n).map(|_| lcg(&mut s)).collect::<Vec<f64>>(),
                )
            })
            .collect();
        let y: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        cols.push(Column::new("y".into(), y));
        cols.push(Column::new(
            "t".into(),
            (from..from + n).map(|i| i as f64).collect::<Vec<f64>>(),
        ));
        DataFrame::new(n, cols).expect("a frame")
    };
    let first = chunk(0, 500);
    bank.fit_predict(&first).expect("a chunk");
    drop(first);
    let live_1 = LIVE.load(Ordering::Relaxed);
    let mut fed = 500;
    while fed < rows + 500 {
        let df = chunk(fed, 500);
        let out = bank.fit_predict(&df).expect("a chunk");
        drop((df, out));
        fed += 500;
    }
    let live_n = LIVE.load(Ordering::Relaxed);
    let per = (live_n - live_1) as f64 / (fed - 500) as f64;
    // The ridge's snapshot of three features and a target, 376 bytes,
    // and the residual spread's of its one slot, 24.
    let counted = 376.0 + 24.0;
    if std::env::var_os("WINDOW_MEMORY_REPORT").is_some() {
        println!("bank ewridge k=3 t=1           heap {per:>9.1} B  counted {counted:>9.1} B");
    }
    assert!(
        per <= 1.125 * counted + 2.0 * 96.0,
        "{per:.0} B a snapshot held"
    );
}
