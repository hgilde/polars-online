//! The workloads behind the instruction-count budgets (`budgets.toml`).
//!
//! Each case is a fixed block of [`ROWS`] rows through one hot path
//! ([`WIDE_ROWS`] for the wide bank cases), after [`WARM_ROWS`] rows that
//! bring the state to where a long stream keeps it: past `min_weight` and
//! the first solves, a full run vector, settled centres. A case is built by
//! a *setup* function and run by one of three measured functions,
//! [`polars_online_budget_model_rows`], [`polars_online_budget_clock_rows`]
//! and [`polars_online_budget_bank_rows`]. `benches/instructions.rs` hands
//! each to Callgrind with the measured function as its entry point, so the
//! count is those rows and nothing else: not the setup, not the data, not
//! the drop. The functions carry fixed, unmangled names because Callgrind
//! finds an entry point by its symbol. The one case on several threads is
//! counted from a client request around the call instead, on every thread
//! (`benches/instructions.rs`).
//!
//! The data are generated here, from a fixed seed, so a count depends on the
//! code and the toolchain alone. `cargo test` runs every case natively on
//! every platform and checks that it reaches the path it is named for.

use online_polars::online_core::{ClockCfg, ClockState, ClockValue, ExactCaps};
use online_polars::{AnyModel, Bank, Spec, WindowsConfig, WindowsRun, build_models};
use polars::prelude::*;
use serde_json::{Value, json};

/// Rows each case steps before the measured block, outside the count.
pub const WARM_ROWS: usize = 2_000;

/// Rows in the measured block. A case's budget is for the block; its count
/// per row is the budget over this.
pub const ROWS: usize = 1_000;

/// The model cases, by the id the bench and `budgets.toml` give them.
pub const MODEL_CASES: &[&str] = &[
    "ewridge_k5",
    "ewridge_k20",
    "ewridge_gram_t10",
    "huber_k20",
    "huber_k20_sets",
    "huber_k20_sum",
    // The refit of the first rows (docs/PLAN.md task 277) inside the
    // measured block: its buffer of 2,500 rows fills on the block's 500th.
    "huber_k20_refit",
    "sgd",
    "sgd_huber",
    "sgd_huber_refit",
    "pa",
    "seqtest",
    "bocpd",
    "kmeans",
    "micro",
    "kalman",
    "ew_cov",
    "lasso_k20",
    // One case per model kind that had none, and the paths the review of
    // 2026-10-23 found costly (task 270 (e): J-3, slice M's table, F-2).
    "rls",
    "quantile",
    // Its refit of the first rows (docs/PLAN.md task 278, D-Q2) inside the
    // measured block, as `huber_k20_refit`'s.
    "quantile_refit",
    "ftrl",
    "holt",
    "marginal",
    // Its bins under Huber's location (docs/PLAN.md task 272 (c)).
    "marginal_bins_huber",
    "ew_class",
    "deco",
    "hmm",
    "rcov",
    "audit_continuous",
    "bocpd_full",
    "ew_cov_lags",
    "corrchange_monitor_w2500",
    "corrchange_window_w100",
];

/// The one model case that reports on a single row of its block:
/// `corrchange`'s monitor tests its span when the span's last row arrives,
/// and its span of 2,500 rows closes once in the block, at its 500th row.
/// That close is the cost the case is for (docs/PLAN.md task 270, D-1).
pub const SPAN_CASE: &str = "corrchange_monitor_w2500";

/// The model cases whose model predicts nothing by design: `Step::pred` is
/// empty on every row, so the native test holds each to the path it is
/// named for by another reading (`tests/workloads.rs`).
pub const SILENT_CASES: &[&str] = &[
    "marginal",
    "marginal_bins_huber",
    "rcov",
    "audit_continuous",
];

/// The clock cases: a float clock, and a temporal one on the exact path.
pub const CLOCK_CASES: &[&str] = &["clock_float", "clock_temporal"];

/// The bank cases: a frame through `Bank::fit_predict`, every per-row step
/// of the bank around the model, for the cheapest model and a regression;
/// and the same two over [`GROUPS`] groups of one row each, on a pool of one
/// thread, where what a `(spec, group)` task costs whatever its rows is most
/// of the count (docs/PLAN.md task 270, G-1); and a regression whose target
/// is a window expression looking 1,000 rows ahead under an embargo of the
/// same span, so each row waits behind about a thousand held rows
/// (docs/PLAN.md task 270, M-3).
pub const BANK_CASES: &[&str] = &[
    "bank_seqtest",
    "bank_ewridge_k5",
    "bank_seqtest_groups",
    "bank_ewridge_k5_groups",
    "formula_target_held1k",
];

/// Groups in a grouped bank case: its block is one row of each.
pub const GROUPS: usize = 1_000;

/// Rows of a grouped case's warm-up: ten a group, past `min_weight` and the
/// first solve of each group's fit.
pub const GROUPED_WARM_ROWS: usize = 10 * GROUPS;

fn grouped(id: &str) -> bool {
    id.ends_with("_groups")
}

/// The window cases: a frame through the window operators' core
/// (`WindowsRun::feed`, as `with_windows` runs it), one operator over a
/// window of 12,500 clock units, about 10,000 rows at the cases' mean step
/// of 1.25. The warm-up and the block are inside the stream's first window,
/// where nothing has left it yet and a mean cuts its oldest row's held
/// interval at the window's edge on every row: the path that cost a mean
/// O(rows in the window) a row where the sum costs O(1) (docs/PLAN.md task
/// 270, M-2). The two are held within twice each other. And the quantile
/// (task 272 (d)): over the same window, its sketches walked on every row,
/// and with no window, the running sketch an EW-MAD reads.
pub const WINDOW_CASES: &[&str] = &[
    "window_ewm_mean_w10k",
    "window_ewm_sum_w10k",
    "window_ewm_quantile_w10k",
    "window_ewm_quantile_running",
];

/// The wide bank cases: a frame of [`WIDE_ROWS`] rows of the `ew_cov` case's
/// 230 statistics through `Bank::fit_predict`, on a pool of one thread.
/// The frame is several runs long (about a thousand rows a run at this
/// width), so each run is copied into the output as it ends while the
/// stream learns the next (`rayon::join`, which on one thread runs the two
/// in turn), and it is past `PAR_MIN_ROWS`, so the output columns are
/// allocated and finished in parallel. `bank_setup` also knows
/// `bank_ew_cov_wide_t4`, the same on a pool of four, which the native test
/// holds to the one-thread case's output; it has no count (benches/
/// instructions.rs says why).
pub const WIDE_CASES: &[&str] = &["bank_ew_cov_wide_t1"];

/// Rows in a wide case's measured block.
pub const WIDE_ROWS: usize = 8_192;

/// A clock step of a second on average, irregular: 0.5, 1, 1.5 or 2.
fn step_of(draw: f64) -> f64 {
    0.5 * (1.0 + (4.0 * draw).floor().min(3.0))
}

/// A seeded generator: Knuth's MMIX constants, the top 53 bits as a
/// uniform draw in `[0, 1)`. Integer arithmetic only, so the data are the
/// same bits on every platform.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A draw in `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
}

/// What a case's rows look like.
#[derive(Clone, Copy)]
enum Shape {
    /// `k` features in `[-1, 1)`, and a target linear in them plus noise.
    Linear,
    /// Four features around one of three centres six apart, as
    /// `scripts/benchmark.py`'s blobs: what a clustering model is for.
    Blobs,
    /// `k` features around one of three centres two apart, and the centre's
    /// number as the target: a label a classifier learns, not always right.
    Labelled,
}

/// Rows in the layout a model steps them: features row-major, targets
/// row-major, and each row's clock step and clock value. Over `groups`
/// groups (0: none), row `i` is group `i % groups`'s, and each group keeps
/// a clock of its own, a unit a row.
pub struct Rows {
    groups: usize,
    k: usize,
    m: usize,
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    d: Vec<f64>,
    t: Vec<f64>,
}

impl Rows {
    fn generate(shape: Shape, k: usize, m: usize, n: usize, groups: usize, seed: u64) -> Self {
        let mut g = Lcg(seed);
        let (mut x, mut y) = (Vec::with_capacity(n * k), Vec::with_capacity(n * m));
        let (mut d, mut t) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut clock = 1_000.0;
        for i in 0..n {
            let row = x.len();
            match shape {
                Shape::Linear => x.extend((0..k).map(|_| g.signed())),
                Shape::Blobs => {
                    let label = (3.0 * g.next()).floor() as usize;
                    x.extend((0..k).map(|j| g.signed() + if j % 3 == label { 6.0 } else { 0.0 }));
                }
                Shape::Labelled => {
                    let label = (3.0 * g.next()).floor() as usize;
                    x.extend((0..k).map(|j| g.signed() + if j % 3 == label { 2.0 } else { 0.0 }));
                    y.extend((0..m).map(|_| Some(label as f64)));
                }
            }
            let fit: f64 = x[row..]
                .iter()
                .enumerate()
                .map(|(j, v)| v / (j + 1) as f64)
                .sum();
            // With no features, a column whose sign a sequential test
            // watches: either sign, a small drift up.
            let (level, noise) = if k == 0 { (0.05, 1.0) } else { (0.5, 0.3) };
            if !matches!(shape, Shape::Labelled) {
                y.extend((0..m).map(|j| Some(level + fit + j as f64 + noise * g.signed())));
            }
            let step = if i == 0 { 0.0 } else { step_of(g.next()) };
            clock += step;
            d.push(step);
            t.push(match i.checked_div(groups) {
                None => clock,
                Some(row) => 1_000.0 + row as f64,
            });
        }
        Self {
            groups,
            k,
            m,
            x,
            y,
            d,
            t,
        }
    }

    fn len(&self) -> usize {
        self.d.len()
    }

    fn row(&self, i: usize) -> (&[f64], &[Option<f64>], f64) {
        (
            &self.x[i * self.k..(i + 1) * self.k],
            &self.y[i * self.m..(i + 1) * self.m],
            self.d[i],
        )
    }

    /// Rows `from..to` as a frame: `x0..`, `y0..` and the clock `t`.
    fn frame(&self, from: usize, to: usize) -> DataFrame {
        let mut cols: Vec<Column> = (0..self.k)
            .map(|j| {
                let v: Vec<f64> = (from..to).map(|i| self.x[i * self.k + j]).collect();
                Column::new(format!("x{j}").into(), v)
            })
            .collect();
        cols.extend((0..self.m).map(|j| {
            let v: Vec<Option<f64>> = (from..to).map(|i| self.y[i * self.m + j]).collect();
            Column::new(format!("y{j}").into(), v)
        }));
        cols.push(Column::new("t".into(), self.t[from..to].to_vec()));
        if self.groups > 0 {
            let g: Vec<i64> = (from..to).map(|i| (i % self.groups) as i64).collect();
            cols.push(Column::new("g".into(), g));
        }
        DataFrame::new_infer_height(cols).expect("columns of one length")
    }
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|j| format!("{prefix}{j}")).collect()
}

/// A case's spec, as `po.spec` writes it, with the shape of its rows. The
/// sizes and settings are `scripts/benchmark.py`'s rows of the same names,
/// so a count and a throughput figure describe one workload.
fn case_spec(id: &str) -> (Value, Shape) {
    let fit = |model: Value, k: usize| {
        json!({
            "name": "m",
            "model": model,
            "targets": ["y0"],
            "features": names("x", k),
            "clock": "t",
            "half_life": 1000.0,
            "gap_cap": 100.0,
            "min_weight": k as f64,
        })
    };
    let four = |model: Value, decays: bool| {
        let mut s = json!({
            "name": "m",
            "model": model,
            "features": names("x", 4),
            "clock": "t",
            "gap_cap": 100.0,
        });
        if decays {
            s["half_life"] = json!(1000.0);
        }
        s
    };
    match id {
        "ewridge_k5" | "bank_ewridge_k5" => (fit(json!({"type": "ewridge"}), 5), Shape::Linear),
        // A solve every 4 rows of a group's clock: at the default cadence a
        // fit ten rows old solves on nearly every row, and the case is for
        // the task's constant, not a young fit's solves. The cadence's
        // first period, which solves on the default schedule (task 278,
        // "Default cadence first"), ends on a group's 5th row, and its
        // solves fall on the 5th, 9th and 13th: the measured 11th solves
        // none, as the old cadence of 20 from the first solve did not.
        "bank_ewridge_k5_groups" => {
            let mut s = fit(json!({"type": "ewridge", "solve_every": 4.0}), 5);
            s["group"] = json!("g");
            (s, Shape::Linear)
        }
        "ewridge_k20" => (fit(json!({"type": "ewridge"}), 20), Shape::Linear),
        // The target the mean of `y0` over the next 1,250 clock units
        // (about 1,000 rows), resolved by the bank's own window core, each
        // row learned once its window has closed and an embargo of the
        // same span has passed: about a thousand rows held at every row.
        "formula_target_held1k" => {
            let mut s = fit(json!({"type": "ewridge"}), 5);
            s["targets"] = json!([{
                "name": "fwd",
                "formula": ["rewm_mean", ["col", "y0"], {"half_life": 1000.0, "window_size": 1250.0}],
            }]);
            s["embargo"] = json!(1250.0);
            (s, Shape::Linear)
        }
        // The blocked Gram with ten targets folded into its merge
        // (docs/PLAN.md task 257): 50 features, a 64-row block, one Gram
        // over every row, a solve every 400 clock units.
        "ewridge_gram_t10" => {
            let model = json!({
                "type": "ewridge",
                "gram_block_rows": 64,
                "target_gaps": "pairwise",
                "solve_every": 400.0,
            });
            let mut s = fit(model, 50);
            s["targets"] = json!(names("y", 10));
            (s, Shape::Linear)
        }
        // Huber's cut in the MAD scale, the default since task 271: a
        // sketch of `|e|` per target, aged and fed each row.
        "huber_k20" => (fit(json!({"type": "huber"}), 20), Shape::Linear),
        // Three nested feature sets of 5, 10 and 20 features from the one
        // target's sums, each solved on the cadence, the rows weighed by
        // the largest set's residual (docs/PLAN.md task 272 (a)).
        // RLS's sum-scale prior on the target's raw moments, the
        // intercept penalized (docs/PLAN.md task 272 (a)).
        "huber_k20_sum" => (
            fit(json!({"type": "huber", "ridge_scale": "sum"}), 20),
            Shape::Linear,
        ),
        // The default refit runs in the warm-up's first 500 rows, outside
        // every block; here it holds 2,500 rows and runs on the block's
        // 500th, so the block pays for it once (docs/PLAN.md task 277).
        "huber_k20_refit" => (
            fit(json!({"type": "huber", "huber_refit_rows": 2500}), 20),
            Shape::Linear,
        ),
        "huber_k20_sets" => {
            let sets = json!([
                ["x5", names("x", 5)],
                ["x10", names("x", 10)],
                ["all", names("x", 20)],
            ]);
            (
                fit(json!({"type": "huber", "feature_sets": sets}), 20),
                Shape::Linear,
            )
        }
        "sgd" => (
            fit(json!({"type": "sgd", "learning_rate": 0.01}), 20),
            Shape::Linear,
        ),
        "sgd_huber" => (
            fit(
                json!({"type": "sgd", "learning_rate": 0.01, "loss": "huber"}),
                20,
            ),
            Shape::Linear,
        ),
        "sgd_huber_refit" => (
            fit(
                json!({"type": "sgd", "learning_rate": 0.01, "loss": "huber", "huber_refit_rows": 2500}),
                20,
            ),
            Shape::Linear,
        ),
        "pa" => (fit(json!({"type": "pa"}), 20), Shape::Linear),
        // The penalty path's selection on every row and its solves on the
        // default cadence (docs/PLAN.md task 270, C-2); the path is
        // `scripts/wide_benchmark.py`'s.
        "lasso_k20" => (
            fit(json!({"type": "lasso", "penalties": [0.1, 0.01, 0.0]}), 20),
            Shape::Linear,
        ),
        "kalman" => (
            fit(json!({"type": "kalman", "coef_half_life": 2000.0}), 20),
            Shape::Linear,
        ),
        "bank_seqtest_groups" => {
            let (mut s, shape) = case_spec("seqtest");
            s["group"] = json!("g");
            (s, shape)
        }
        "seqtest" | "bank_seqtest" => (
            json!({
                "name": "m",
                "model": {"type": "seqtest"},
                "targets": ["y0"],
                "features": [],
                "clock": "t",
                "gap_cap": 100.0,
            }),
            Shape::Linear,
        ),
        "ew_cov" | "bank_ew_cov_wide_t1" | "bank_ew_cov_wide_t4" => {
            let mut s = fit(json!({"type": "ew_cov"}), 20);
            let s = s.as_object_mut().expect("an object");
            s.remove("targets");
            s.remove("min_weight");
            (Value::Object(s.clone()), Shape::Linear)
        }
        "bocpd" => (
            four(
                json!({"type": "bocpd", "prune_below": 1e-6, "max_run": 200}),
                false,
            ),
            Shape::Blobs,
        ),
        "kmeans" => (
            four(json!({"type": "kmeans", "n_clusters": 8}), true),
            Shape::Blobs,
        ),
        "micro" => (
            four(json!({"type": "micro", "eps": 1.0}), true),
            Shape::Blobs,
        ),
        // Four columns, six pairs, a span of 2,500 rows: the warm-up's 2,000
        // and the block's first 500 (docs/PLAN.md task 270, D-1).
        "corrchange_monitor_w2500" => (
            four(
                json!({"type": "corrchange", "kind": "monitor", "window_rows": 2500}),
                false,
            ),
            Shape::Linear,
        ),
        // Two adjacent windows of 100 rows over four columns, a permutation
        // critical value of 100 draws redrawn every 250 rows (docs/PLAN.md
        // task 270, D-2).
        "corrchange_window_w100" => (
            four(
                json!({"type": "corrchange", "kind": "window", "window_rows": 100,
                       "n_perm": 100, "permute_every_rows": 250}),
                false,
            ),
            Shape::Linear,
        ),
        // From here, the cases added with the review of 2026-10-23 (task
        // 270 (e)), each `scripts/benchmark.py`'s row of the same kind.
        "rls" => (fit(json!({"type": "rls", "delta": 1.0}), 20), Shape::Linear),
        "quantile" => (
            fit(json!({"type": "quantile", "quantile": 0.5}), 20),
            Shape::Linear,
        ),
        // The default refit runs in the warm-up's first 500 rows, outside
        // every block; here it holds 2,500 rows and runs on the block's
        // 500th, so the block pays for it once (task 278, D-Q2).
        "quantile_refit" => (
            fit(
                json!({"type": "quantile", "quantile": 0.5, "quantile_refit_rows": 2500}),
                20,
            ),
            Shape::Linear,
        ),
        "ftrl" => (fit(json!({"type": "ftrl"}), 20), Shape::Linear),
        // Level and trend of one column, no features.
        "holt" => {
            let mut s = fit(json!({"type": "holt"}), 0);
            s.as_object_mut().expect("an object").remove("min_weight");
            (s, Shape::Linear)
        }
        // Twenty features against one target, every pair's moments.
        "marginal" => (fit(json!({"type": "marginal"}), 20), Shape::Linear),
        // The same, with four given bins a feature and each bin's location
        // Huber's (docs/PLAN.md task 272 (c)): a cell's step, its judge
        // and each target's scale on every row.
        "marginal_bins_huber" => (
            fit(
                json!({
                    "type": "marginal",
                    "bin_edges": vec![vec![-0.5, 0.0, 0.5]; 20],
                    "bin_stat": "huber",
                }),
                20,
            ),
            Shape::Linear,
        ),
        "ew_class" => {
            let model = json!({
                "type": "ew_class",
                "classes": ["0", "1", "2"],
                "precision_prior": 1.0,
            });
            let mut s = fit(model, 20);
            s.as_object_mut().expect("an object").remove("min_weight");
            (s, Shape::Labelled)
        }
        "deco" | "ew_cov_lags" => {
            let model = if id == "deco" {
                json!({"type": "deco"})
            } else {
                json!({"type": "ew_cov", "stats": ["mean", "cov"], "lags": [1, 2, 3, 4, 5]})
            };
            let mut s = fit(model, 20);
            let o = s.as_object_mut().expect("an object");
            o.remove("targets");
            o.remove("min_weight");
            (s, Shape::Linear)
        }
        "hmm" => (
            four(
                json!({"type": "hmm", "n_components": 2, "precision_prior": 1.0}),
                true,
            ),
            Shape::Blobs,
        ),
        // The kernel estimator, as `scripts/benchmark.py`'s row.
        // A group that closes is required; the case counts the rows the
        // block takes in, not its close, which the bank calls.
        "rcov" => {
            let mut s = four(
                json!({"type": "rcov", "kind": "kernel", "block_rows": 2000}),
                false,
            );
            s["group"] = json!("g");
            s["group_close"] = json!("monotone");
            (s, Shape::Blobs)
        }
        // A continuous column at the default cap of 256 counters: nearly
        // every value new, so past the cap on nearly every row (F-2).
        "audit_continuous" => (
            json!({
                "name": "m",
                "model": {"type": "audit"},
                "features": ["x0"],
                "clock": "t",
                "gap_cap": 100.0,
            }),
            Shape::Linear,
        ),
        // The full covariance, whose runs each factor a 4 x 4 scale a row.
        "bocpd_full" => (
            four(
                json!({
                    "type": "bocpd",
                    "covariance_type": "full",
                    "degrees_of_freedom_prior": 6.0,
                    "prune_below": 1e-6,
                    "max_run": 200,
                }),
                false,
            ),
            Shape::Blobs,
        ),
        other => panic!("no case {other:?}"),
    }
}

/// A case's spec, filled and checked as a bank checks it.
pub fn spec(id: &str) -> Spec {
    let (value, _) = case_spec(id);
    let mut spec: Spec = serde_json::from_value(value).unwrap_or_else(|e| panic!("{id}: {e}"));
    spec.check().unwrap_or_else(|e| panic!("{id}: {e}"));
    spec
}

/// A case's rows: warm-up and measured block, from one seed.
fn rows(id: &str, spec: &Spec) -> Rows {
    let (_, shape) = case_spec(id);
    let block = if WIDE_CASES.contains(&id) || id == "bank_ew_cov_wide_t4" {
        WIDE_ROWS
    } else {
        ROWS
    };
    let groups = if grouped(id) { GROUPS } else { 0 };
    Rows::generate(shape, spec.k(), spec.m(), warm_rows(id) + block, groups, 7)
}

/// Rows a case steps before its measured block.
fn warm_rows(id: &str) -> usize {
    if grouped(id) {
        GROUPED_WARM_ROWS
    } else {
        WARM_ROWS
    }
}

/// A model case, warmed up: the model, the rows of its measured block, and
/// the `Step` it is stepped into, as the bank keeps one per model.
pub struct ModelRun {
    pub model: AnyModel,
    rows: Rows,
    step: online_polars::online_core::Step,
}

/// The setup of a model case: the model the spec builds, stepped through
/// the warm-up rows.
pub fn model_setup(id: &str) -> ModelRun {
    let spec = spec(id);
    let mut built = build_models(&spec).unwrap_or_else(|e| panic!("{id}: {e}"));
    assert_eq!(built.len(), 1, "{id}: one instance");
    let (_, mut model) = built.pop().expect("one instance");
    let rows = rows(id, &spec);
    let mut step = online_polars::online_core::Step::default();
    for i in 0..WARM_ROWS {
        let (x, y, d) = rows.row(i);
        model.step_into(x, y, d, 1.0, &mut step);
    }
    ModelRun { model, rows, step }
}

/// What a measured block reports: a checksum, so no row's work can be
/// optimized away, and how many rows predicted a number.
#[derive(Debug, Default, Clone, Copy)]
pub struct Block {
    pub checksum: f64,
    pub predicted: usize,
}

/// The measured block of a model case: [`ROWS`] rows of
/// `AnyModel::step_into`, each with weight 1, into the one `Step` the case
/// keeps, as the bank steps a model (docs/PLAN.md task 270, M-4).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn polars_online_budget_model_rows(run: &mut ModelRun) -> Block {
    let mut out = Block::default();
    for i in WARM_ROWS..run.rows.len() {
        let (x, y, d) = run.rows.row(i);
        run.model.step_into(x, y, d, 1.0, &mut run.step);
        let step = &run.step;
        let finite: f64 = step.pred.iter().filter(|p| p.is_finite()).sum();
        out.predicted += usize::from(step.pred.iter().any(|p| p.is_finite()));
        out.checksum += finite + step.n_eff;
    }
    out
}

/// A clock case, warmed up.
pub struct ClockRun {
    pub state: ClockState,
    cfg: ClockCfg,
    caps: ExactCaps,
    clocks: Vec<ClockValue>,
}

/// The setup of a clock case: `ClockState::advance_stamped`, as the bank
/// calls it on every row, over a float clock in seconds or a temporal one
/// in nanoseconds since 2024, each with a cap of 100 seconds.
pub fn clock_setup(id: &str) -> ClockRun {
    let mut g = Lcg(11);
    let mut seconds = 1_000.0_f64;
    let mut ns: i64 = 1_704_067_200_000_000_000;
    let mut clocks = Vec::with_capacity(WARM_ROWS + ROWS);
    for i in 0..WARM_ROWS + ROWS {
        let step = if i == 0 { 0.0 } else { step_of(g.next()) };
        seconds += step;
        ns += (step * 1e9) as i64;
        clocks.push(match id {
            "clock_float" => ClockValue::F64(seconds),
            "clock_temporal" => ClockValue::Ns(ns),
            other => panic!("no case {other:?}"),
        });
    }
    let cfg = ClockCfg {
        gap_cap: 100.0,
        ..ClockCfg::default()
    };
    let caps = ExactCaps {
        gap_cap_ns: (id == "clock_temporal").then_some(100_000_000_000),
        session_gap_ns: None,
    };
    let mut state = ClockState::new();
    for &c in &clocks[..WARM_ROWS] {
        state.advance_stamped(&cfg, &caps, Some(c), None, true);
    }
    ClockRun {
        state,
        cfg,
        caps,
        clocks,
    }
}

/// The measured block of a clock case: [`ROWS`] accepted rows.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn polars_online_budget_clock_rows(run: &mut ClockRun) -> Block {
    let mut out = Block::default();
    for &c in &run.clocks[WARM_ROWS..] {
        let adv = run
            .state
            .advance_stamped(&run.cfg, &run.caps, Some(c), None, true);
        out.checksum += adv.d_clock;
        out.predicted += usize::from(adv.stamp.is_some());
    }
    out
}

/// A bank case, warmed up: the bank and the frame of its measured block.
pub struct BankRun {
    pub bank: Bank,
    frame: DataFrame,
}

/// The setup of a bank case: a bank of the case's one spec, fed the
/// warm-up rows as one frame. A wide or grouped case's bank is given its
/// pool size:
/// one thread is the process default's pool when that has one thread, as
/// under the bench, so the rows are learned on the thread that calls.
pub fn bank_setup(id: &str) -> BankRun {
    let spec = spec(id);
    let rows = rows(id, &spec);
    let mut bank = Bank::new(vec![spec]).unwrap_or_else(|e| panic!("{id}: {e}"));
    let threads = match id {
        "bank_ew_cov_wide_t1" => Some(1),
        "bank_ew_cov_wide_t4" => Some(4),
        id if grouped(id) => Some(1),
        _ => None,
    };
    if threads.is_some() {
        online_polars::pool().unwrap_or_else(|e| panic!("{id}: {e}"));
        bank.set_threads(threads)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
    }
    let warm = warm_rows(id);
    bank.fit_predict(&rows.frame(0, warm))
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    let frame = rows.frame(warm, rows.len());
    BankRun { bank, frame }
}

/// The measured block of a bank case: one `Bank::fit_predict` of [`ROWS`]
/// rows, below the bank's parallel threshold, so every row is learned on
/// the thread that calls it. The bench calls this on the bank's own pool,
/// where `fit_predict`'s hand-off to the pool runs inline; Callgrind counts
/// the thread an entry point runs on, and nothing of the waiting caller.
/// The output is returned, so its drop falls outside the count. A wide
/// case's block is [`WIDE_ROWS`] rows, on its bank's pool of one thread or
/// four.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn polars_online_budget_bank_rows(run: &mut BankRun) -> Vec<Column> {
    run.bank
        .fit_predict(&run.frame)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// A window case, warmed up: the core and the frame of its measured block.
pub struct WindowsCase {
    pub run: WindowsRun,
    frame: DataFrame,
}

/// A window case's formula, as `po.ops` writes it.
fn window_tree(id: &str) -> Value {
    let op = match id {
        "window_ewm_mean_w10k" => "ewm_mean",
        "window_ewm_sum_w10k" => "ewm_sum",
        "window_ewm_quantile_w10k" => {
            return json!(["ewm_quantile", ["col", "x0"], {"half_life": 1000.0, "window_size": 12_500.0, "quantile": 0.5}]);
        }
        "window_ewm_quantile_running" => {
            return json!(["ewm_quantile", ["abs", ["col", "x0"]], {"half_life": 1000.0, "quantile": 0.5}]);
        }
        other => panic!("no case {other:?}"),
    };
    json!([op, ["col", "x0"], {"half_life": 1000.0, "window_size": 12_500.0}])
}

/// The setup of a window case: the core over one feature column and the
/// clock, fed the warm-up rows as one frame.
pub fn windows_setup(id: &str) -> WindowsCase {
    let config: WindowsConfig = serde_json::from_value(json!({
        "formulas": [{"name": "w", "tree": window_tree(id)}],
        "clock": "t",
        "gap_cap": 100.0,
    }))
    .unwrap_or_else(|e| panic!("{id}: {e}"));
    let rows = Rows::generate(Shape::Linear, 1, 0, WARM_ROWS + ROWS, 0, 7);
    let warm = rows.frame(0, WARM_ROWS);
    let mut run = WindowsRun::new(config, warm.schema()).unwrap_or_else(|e| panic!("{id}: {e}"));
    run.feed(&warm, None)
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    let frame = rows.frame(WARM_ROWS, rows.len());
    WindowsCase { run, frame }
}

/// The measured block of a window case: one `WindowsRun::feed` of [`ROWS`]
/// rows. A backward window emits each row as it is read, so the output is
/// the block's values, returned so its drop falls outside the count.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn polars_online_budget_windows_rows(run: &mut WindowsCase) -> DataFrame {
    run.run
        .feed(&run.frame, None)
        .unwrap_or_else(|e| panic!("{e}"))
}
