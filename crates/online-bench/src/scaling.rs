//! How each surface's cost grows with each size it has: the cells of the
//! scaling table (docs/PERFORMANCE.md, "How cost grows").
//!
//! A budget pins what one workload costs at one size, so it catches a change
//! that made the workload slower than before, and never a cost that grew the
//! wrong way from the start: a row whose work walks every group the stream
//! has seen costs nothing measurable at the handful of groups a budget case
//! has. A cell here instead holds the *ratio* of what one workload costs at
//! two sizes of one dimension, `n` and `step·n`, to the growth its algorithm
//! declares: `cost(step·n) / cost(n) <= g(step·n) / g(n)`, `g` the declared
//! growth. That needs no number from an earlier run, so it catches a defect
//! that was there on the first day, and the bound comes from the algorithm,
//! not from a measurement.
//!
//! Each cell measures two things at each size:
//!
//! - **the work of a row**, as the cost of a fixed block of rows (the same
//!   number at both sizes), after a warm-up that brings the state to where a
//!   long stream keeps it. Natively it is timed (the least of several
//!   blocks, small and large interleaved) and its allocations are counted;
//!   on CI's Linux runner Callgrind counts its instructions
//!   (`benches/instructions.rs`, the `scale_*` cases, held by
//!   `scripts/budgets.py`'s `scaling` check).
//! - **the state** the surface keeps between rows, as the bytes its saved
//!   state takes (msgpack) and the heap it holds, read from the allocator.
//!
//! The data are generated here from a fixed seed, as the budget cases' are.

use std::collections::HashMap;

use online_polars::{AnyModel, Bank, RefreshCols, RefreshTime, Spec, WindowsConfig, WindowsRun};
use polars::prelude::*;
use serde_json::{Value, json};

use crate::{Lcg, Rows, Shape, names};

/// How a cost grows with a dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Growth {
    /// Not at all: the dimension must not move it.
    Const,
    /// As the logarithm of the size.
    Log,
    /// As the size to this power.
    Pow(u32),
    /// As the size to the power `num / den`.
    Frac(u32, u32),
}

impl Growth {
    /// The most the cost may grow from size `small` to size `large`.
    pub fn bound(self, small: usize, large: usize) -> f64 {
        match self {
            Growth::Const => 1.0,
            Growth::Log => (large as f64).ln() / (small as f64).ln(),
            Growth::Pow(e) => (large as f64 / small as f64).powi(e as i32),
            Growth::Frac(a, b) => (large as f64 / small as f64).powf(f64::from(a) / f64::from(b)),
        }
    }

    /// As the table writes it, in the dimension `d`.
    pub fn show(self, d: &str) -> String {
        match self {
            Growth::Const => "O(1)".to_owned(),
            Growth::Log => format!("O(log {d})"),
            Growth::Pow(1) => format!("O({d})"),
            Growth::Pow(e) => format!("O({d}^{e})"),
            Growth::Frac(a, b) => format!("O({d}^({a}/{b}))"),
        }
    }
}

/// One cell of the table: a surface, a dimension, two sizes of it, and the
/// growth the algorithm declares for a row's work and for the state.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    /// `<surface>.<dimension>`, the id the tests and the bench use.
    pub id: &'static str,
    /// The smaller size; the larger is `small * step`.
    pub small: usize,
    pub step: usize,
    /// A row's work.
    pub work: Growth,
    /// The state kept between rows.
    pub state: Growth,
    /// The finding a cell is ignored for until it is fixed, if any.
    pub finding: Option<&'static str>,
}

impl Cell {
    pub fn large(&self) -> usize {
        self.small * self.step
    }
}

const fn cell(id: &'static str, small: usize, step: usize, work: Growth, state: Growth) -> Cell {
    Cell {
        id,
        small,
        step,
        work,
        state,
        finding: None,
    }
}

/// A cell whose measurement exceeds its declaration: held to the
/// declaration in the extended tier, where it fails until it is fixed. No
/// cell is one now (findings S1 to S3 are fixed).
#[allow(dead_code)]
const fn finding(
    id: &'static str,
    small: usize,
    step: usize,
    work: Growth,
    state: Growth,
    finding: &'static str,
) -> Cell {
    Cell {
        id,
        small,
        step,
        work,
        state,
        finding: Some(finding),
    }
}

use Growth::{Const, Frac, Log, Pow};

/// Every measured cell. docs/PERFORMANCE.md's table holds the reason for
/// each declaration; `tests/scaling.rs` holds every cell to it.
pub const CELLS: &[Cell] = &[
    // --- models, one row of `AnyModel::step_into` --------------------------
    cell("ewridge.k", 4, 8, Pow(2), Pow(2)),
    cell("ewridge.solve_k", 4, 8, Pow(3), Pow(2)),
    cell("ewridge.targets", 1, 8, Pow(1), Pow(1)),
    cell("ewridge.ridge_grid", 1, 8, Pow(1), Pow(1)),
    cell("ewridge.feature_sets", 1, 8, Pow(1), Pow(1)),
    cell("ewridge.window", 500, 8, Const, Pow(1)),
    cell("ewridge.gram_block", 8, 8, Const, Pow(1)),
    cell("ewridge.history", 1000, 8, Const, Const),
    cell("rls.k", 4, 8, Pow(2), Pow(2)),
    cell("rls.targets", 1, 8, Pow(1), Pow(1)),
    cell("rls.history", 1000, 8, Const, Const),
    cell("lasso.k", 4, 8, Pow(2), Pow(2)),
    cell("lasso.path", 2, 8, Pow(1), Pow(1)),
    cell("lasso.history", 1000, 8, Const, Const),
    cell("kalman.k", 4, 8, Pow(2), Pow(2)),
    cell("kalman.targets", 1, 8, Pow(1), Pow(1)),
    cell("kalman.history", 1000, 8, Const, Const),
    cell("huber.k", 4, 8, Pow(2), Pow(2)),
    cell("huber.targets", 1, 8, Pow(1), Pow(1)),
    cell("huber.feature_sets", 1, 8, Pow(1), Pow(1)),
    cell("huber.refit_rows", 1000, 8, Log, Pow(1)),
    cell("huber.history", 1000, 8, Const, Const),
    cell("quantile.k", 4, 8, Pow(2), Pow(2)),
    cell("quantile.refit_rows", 1000, 8, Log, Pow(1)),
    cell("quantile.history", 1000, 8, Const, Const),
    cell("sgd.k", 4, 8, Pow(1), Pow(1)),
    cell("sgd.targets", 1, 8, Pow(1), Pow(1)),
    cell("sgd.refit_rows", 1000, 8, Log, Pow(1)),
    cell("sgd.history", 1000, 8, Const, Const),
    cell("pa.k", 4, 8, Pow(1), Pow(1)),
    cell("pa.targets", 1, 8, Pow(1), Pow(1)),
    cell("pa.history", 1000, 8, Const, Const),
    cell("ftrl.k", 4, 8, Pow(1), Pow(1)),
    cell("ftrl.history", 1000, 8, Const, Const),
    cell("holt.targets", 1, 8, Pow(1), Pow(1)),
    cell("holt.history", 1000, 8, Const, Const),
    cell("ew_cov.k", 4, 8, Pow(2), Pow(2)),
    cell("ew_cov.lags", 1, 8, Pow(1), Pow(1)),
    cell("ew_cov.window", 500, 8, Const, Pow(1)),
    cell("ew_cov.history", 1000, 8, Const, Const),
    cell("marginal.k", 4, 8, Pow(1), Pow(1)),
    cell("marginal.targets", 1, 8, Pow(1), Pow(1)),
    cell("marginal.lags", 1, 8, Pow(1), Pow(1)),
    cell("marginal.bins", 4, 8, Log, Pow(1)),
    cell("marginal.history", 1000, 8, Const, Const),
    cell("deco.k", 4, 8, Pow(1), Pow(1)),
    cell("deco.blocks", 1, 8, Pow(2), Pow(2)),
    cell("deco.history", 1000, 8, Const, Const),
    cell("rcov.k", 2, 8, Pow(2), Pow(2)),
    cell("rcov.block_rows", 500, 8, Frac(3, 5), Frac(3, 5)),
    cell("hmm.states", 2, 8, Pow(2), Pow(2)),
    cell("hmm.k", 2, 8, Pow(1), Pow(1)),
    cell("hmm.history", 1000, 8, Const, Const),
    cell("kmeans.clusters", 2, 8, Pow(1), Pow(1)),
    cell("kmeans.k", 4, 8, Pow(1), Pow(1)),
    cell("kmeans.history", 1000, 8, Const, Const),
    cell("micro.max_clusters", 8, 8, Pow(1), Pow(1)),
    cell("micro.history", 1000, 8, Const, Const),
    cell("ew_class.classes", 2, 8, Pow(1), Pow(1)),
    cell("ew_class.k", 4, 4, Pow(3), Pow(2)),
    cell("ew_class.k_diag", 4, 8, Pow(1), Pow(1)),
    cell("ew_class.history", 1000, 8, Const, Const),
    cell("seqtest.targets", 1, 8, Pow(1), Pow(1)),
    cell("seqtest.history", 1000, 8, Const, Const),
    cell("corrchange.k", 2, 8, Pow(2), Pow(2)),
    cell("corrchange.window", 100, 8, Const, Pow(1)),
    cell("corrchange.window_crit", 100, 8, Const, Pow(1)),
    cell("corrchange.history", 1000, 8, Const, Const),
    cell("bocpd.max_run", 25, 8, Pow(1), Pow(1)),
    cell("bocpd.k", 2, 8, Pow(1), Pow(1)),
    cell("bocpd.k_full", 2, 4, Pow(3), Pow(2)),
    cell("bocpd.history", 1000, 8, Const, Const),
    cell("audit.k", 4, 8, Pow(1), Pow(1)),
    cell("audit.pairs", 4, 8, Pow(2), Pow(2)),
    cell("audit.distinct_cap", 32, 8, Const, Pow(1)),
    cell("audit.history", 1000, 8, Const, Log),
    // --- the bank, `Bank::fit_predict` on a pool of one thread --------------
    cell("bank.groups_seen", 250, 8, Const, Pow(1)),
    // Specs on one stream: over the same columns, each over columns of its
    // own, a mix of five kinds (the size is the mix's copies, five specs
    // each), and the same in calls of four rows, where what a call costs a
    // spec -- its columns looked up, its output assembled -- is the cost.
    cell("bank.specs", 1, 8, Pow(1), Pow(1)),
    cell("bank.specs_wide", 8, 8, Pow(1), Pow(1)),
    cell("bank.specs_distinct", 1, 8, Pow(1), Pow(1)),
    cell("bank.specs_distinct_wide", 8, 8, Pow(1), Pow(1)),
    cell("bank.specs_mixed", 1, 8, Pow(1), Pow(1)),
    cell("bank.specs_per_call", 1, 8, Pow(1), Pow(1)),
    cell("bank.specs_per_call_wide", 8, 8, Pow(1), Pow(1)),
    cell("bank.k", 4, 8, Pow(2), Pow(2)),
    cell("bank.targets", 1, 8, Pow(1), Pow(1)),
    cell("bank.chunk_rows", 64, 8, Const, Const),
    cell("bank.history", 1000, 8, Const, Const),
    cell("bank.embargo", 125, 8, Const, Pow(1)),
    cell("bank.formula_window", 125, 8, Const, Pow(1)),
    cell("bank.ljung_box_lags", 2, 8, Pow(1), Pow(1)),
    cell("bank.robust_se_lags", 1, 8, Pow(1), Pow(1)),
    cell("bank.resid_quantiles", 1, 8, Pow(1), Pow(1)),
    // --- the runner behind the command line, `online_polars::run` -----------
    // A block is a whole run: `n` rows in calls of 1,024 (the work is read a
    // row; the memory a run holds at its peak must not grow with its rows),
    // and specs on one stream, in calls of 1,024 rows and of 16 (16 specs
    // then 128, where an O(specs²) cost a call reads 13x).
    cell("runner.rows", 4_096, 8, Const, Const),
    cell("runner.specs", 1, 8, Pow(1), Pow(1)),
    cell("runner.specs_per_call", 16, 8, Pow(1), Pow(1)),
    // --- the window operators, `WindowsRun::feed` ---------------------------
    cell("windows.groups", 1000, 8, Const, Pow(1)),
    cell("windows.groups_seen", 1000, 8, Const, Pow(1)),
    cell("windows.kernels", 1, 8, Pow(1), Pow(1)),
    cell("windows.operators", 1, 8, Pow(1), Pow(1)),
    cell("windows.window_mean", 1250, 8, Const, Pow(1)),
    cell("windows.window_var", 1250, 8, Const, Pow(1)),
    cell("windows.window_quantile", 1250, 8, Log, Pow(1)),
    cell("windows.forward", 1250, 8, Const, Pow(1)),
    cell("windows.history", 1000, 8, Const, Const),
    // --- refresh_time, `RefreshTime::feed` ----------------------------------
    cell("refresh.series", 2, 8, Pow(1), Pow(1)),
    cell("refresh.pairs", 8, 8, Pow(1), Pow(2)),
    cell("refresh.groups", 100, 8, Const, Pow(1)),
    cell("refresh.history", 1000, 8, Const, Const),
    // --- po.gbdt's sample of the rows, `Sampler::push_frame` ----------------
    cell("gbdt.history", 10_000, 8, Const, Const),
    cell("gbdt.sample", 1000, 8, Log, Pow(1)),
];

/// The cells CI's Callgrind counts at both sizes (`benches/instructions.rs`,
/// the `scale_*` cases), each held by `scripts/budgets.py` to the growth
/// `budgets.toml`'s `[scaling.cells]` declares: the cells where a count
/// sees what a timing cannot -- a growth hidden under the cache's, or a
/// constant only a few percent off -- and that are cheap under Valgrind.
/// Each runs on the thread Callgrind counts: `po.gbdt`'s sample pushes its
/// rows on a pool of its own, whose thread the count would miss, so it has
/// none here.
pub const BENCH_CELLS: &[&str] = &[
    "ewridge.k",
    "ewridge.history",
    "huber.history",
    "quantile.history",
    "sgd.k",
    "ew_cov.k",
    "ew_cov.window",
    "marginal.k",
    "bank.groups_seen",
    "bank.history",
    "bank.embargo",
    "bank.formula_window",
    "bank.specs",
    "bank.specs_distinct",
    "bank.specs_mixed",
    "bank.specs_per_call",
    "windows.groups_seen",
    "windows.history",
    "windows.window_mean",
    "windows.kernels",
    "refresh.groups",
    "refresh.history",
];

/// The bench id of one size of a cell: `scale_<surface>_by_<dimension>_small`
/// or `..._large`, as `scripts/budgets.py` reads it back.
pub fn bench_id(cell: &str, size: &str) -> String {
    format!("scale_{}_{size}", cell.replacen('.', "_by_", 1))
}

/// The cell of an id.
pub fn cell_of(id: &str) -> &'static Cell {
    CELLS
        .iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("no scaling cell {id:?}"))
}

/// A cell's workload at one size, warmed up, with the blocks it is
/// measured on.
pub struct ScaleRun {
    pub id: &'static str,
    pub size: usize,
    work: Work,
    /// Rows a block steps: the same at both sizes of a cell.
    pub block_rows: usize,
    blocks: usize,
    next: usize,
}

enum Work {
    Model {
        model: Box<AnyModel>,
        step: online_polars::online_core::Step,
        rows: Rows,
        warm: usize,
    },
    Bank {
        bank: Box<Bank>,
        /// Each block's calls: one frame each, but for `bank.chunk_rows`.
        frames: Vec<Vec<DataFrame>>,
    },
    Windows {
        run: Box<WindowsRun>,
        frames: Vec<DataFrame>,
    },
    /// A block is a run of its calls through the runner, into a sink.
    Runner {
        bank: Box<Bank>,
        frames: Vec<Vec<DataFrame>>,
        schema: Schema,
    },
    Refresh {
        rt: Box<RefreshTime>,
        frames: Vec<DataFrame>,
        grouped: bool,
    },
    Gbdt {
        sampler: Box<online_polars::gbdt::Sampler>,
        layout: online_polars::gbdt::Layout,
        pool: rayon::ThreadPool,
        frames: Vec<DataFrame>,
    },
}

impl ScaleRun {
    /// Whether a block runs through the bank, on its pool.
    pub fn on_the_pool(&self) -> bool {
        matches!(self.work, Work::Bank { .. })
    }

    /// Whether a block is a whole run, whose memory is read at its peak.
    pub fn is_run(&self) -> bool {
        matches!(self.work, Work::Runner { .. })
    }

    /// Blocks left to run.
    pub fn blocks_left(&self) -> usize {
        self.blocks - self.next
    }

    /// The bytes of the saved state: msgpack, as a state file holds it.
    pub fn state_bytes(&self) -> usize {
        match &self.work {
            Work::Model { model, .. } => rmp_serde::to_vec(model.as_ref())
                .expect("a model's state serializes")
                .len(),
            Work::Bank { bank, .. } | Work::Runner { bank, .. } => {
                bank.save_bytes().expect("a bank saves").len()
            }
            Work::Windows { run, .. } => run.save_bytes().expect("a window core saves").len(),
            Work::Refresh { rt, .. } => rt.save_bytes().expect("a sampler saves").len(),
            // The sampler is never saved: its state is its kept rows,
            // `rows_kept` of them, each a float a feature.
            Work::Gbdt {
                sampler, layout, ..
            } => sampler.rows_kept() * layout.names.len() * 4,
        }
    }

    /// Whether the run is on the path its cell is for, read once its
    /// blocks have run: a refit cell's refit still waits, so its rows are
    /// held; `micro`'s summaries fill their cap; a window cell's window holds
    /// rows. `Err` names what is not so.
    pub fn on_its_path(&self) -> Result<(), String> {
        let id = self.id;
        match &self.work {
            Work::Model { model, .. } if id.ends_with(".refit_rows") => {
                let s = model.refit_status();
                if s.is_empty() || s.iter().any(|r| r.done) {
                    return Err(format!("{id}: the refit ran inside the blocks: {s:?}"));
                }
            }
            Work::Model { model, .. } if id == "micro.max_clusters" => {
                if let AnyModel::Micro(m) = model.as_ref()
                    && m.micro_clusters().len() + 1 < self.size
                {
                    let n = m.micro_clusters().len();
                    return Err(format!("{id}: {n} summaries under a cap of {}", self.size));
                }
            }
            Work::Windows { run, .. }
                if (id.starts_with("windows.window") || id == "windows.forward")
                    && run.queued() < self.size / 2 =>
            {
                return Err(format!("{id}: the window holds {} rows", run.queued()));
            }
            _ => {}
        }
        Ok(())
    }

    /// Drop the rows not yet stepped, so what the allocator holds after is
    /// the state alone.
    pub fn drop_data(&mut self) {
        match &mut self.work {
            Work::Model { rows, .. } => *rows = Rows::empty(),
            Work::Bank { frames, .. } | Work::Runner { frames, .. } => *frames = Vec::new(),
            Work::Windows { frames, .. }
            | Work::Refresh { frames, .. }
            | Work::Gbdt { frames, .. } => *frames = Vec::new(),
        }
    }
}

/// The measured block of a scaling case: the next block of its rows. Its
/// checksum keeps the work from being optimized away.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn polars_online_budget_scale_rows(run: &mut ScaleRun) -> f64 {
    assert!(run.next < run.blocks, "{}: no block left", run.id);
    let b = run.next;
    run.next += 1;
    let n = run.block_rows;
    match &mut run.work {
        Work::Model {
            model,
            step,
            rows,
            warm,
        } => {
            let mut sum = 0.0;
            for i in *warm + b * n..*warm + (b + 1) * n {
                let (x, y, d) = rows.row(i);
                model.step_into(x, y, d, 1.0, step);
                sum += step.pred.iter().filter(|p| p.is_finite()).sum::<f64>() + step.n_eff;
            }
            sum
        }
        Work::Bank { bank, frames } => {
            let mut sum = 0.0;
            for f in &frames[b] {
                let out = bank.fit_predict(f).unwrap_or_else(|e| panic!("{e}"));
                sum += out.len() as f64;
            }
            sum
        }
        Work::Runner {
            bank,
            frames,
            schema,
        } => {
            let calls = std::mem::take(&mut frames[b]);
            let mut rows = 0usize;
            let mut sink = |df: DataFrame| {
                rows += df.height();
                Ok(())
            };
            online_polars::run(
                bank,
                online_polars::Input::Batches {
                    frames: Box::new(calls.into_iter().map(Ok)),
                    schema: schema.clone(),
                },
                online_polars::Output::Batches(&mut sink),
                online_polars::RunOptions::default(),
                |_| Ok(()),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            rows as f64
        }
        Work::Windows { run: w, frames } => {
            let out = w.feed(&frames[b], None).unwrap_or_else(|e| panic!("{e}"));
            out.height() as f64
        }
        Work::Refresh {
            rt,
            frames,
            grouped,
        } => {
            let cols = refresh_cols(*grouped);
            let out = rt.feed(&frames[b], &cols).unwrap_or_else(|e| panic!("{e}"));
            out.height() as f64
        }
        Work::Gbdt {
            sampler,
            layout,
            pool,
            frames,
        } => {
            sampler
                .push_frame(&frames[b], layout, pool)
                .unwrap_or_else(|e| panic!("{e}"));
            sampler.rows_kept() as f64
        }
    }
}

// --- the workloads -------------------------------------------------------------

/// A model spec over `k` features and `t` targets, as the budget cases'
/// (`case_spec`'s `fit`).
fn fit(model: Value, k: usize, t: usize) -> Value {
    json!({
        "name": "m",
        "model": model,
        "targets": names("y", t),
        "features": names("x", k),
        "clock": "t",
        "half_life": 1000.0,
        "gap_cap": 100.0,
        "min_weight": k as f64,
    })
}

/// A spec with no targets: the moment and clustering families.
fn untargeted(model: Value, k: usize, decays: bool) -> Value {
    let mut s = json!({
        "name": "m",
        "model": model,
        "features": names("x", k),
        "clock": "t",
        "gap_cap": 100.0,
    });
    if decays {
        s["half_life"] = json!(1000.0);
    }
    s
}

/// What a model cell steps: its spec, the shape of its rows, the warm-up
/// and the block.
struct ModelCase {
    spec: Value,
    shape: Shape,
    warm: usize,
    block: usize,
    /// The row a refit runs on, which the last block must end before: the
    /// warm-up is then set from the blocks asked for, so every block steps
    /// with about this many rows held.
    refit_at: Option<usize>,
}

impl ModelCase {
    fn new(spec: Value, shape: Shape) -> Self {
        Self {
            spec,
            shape,
            warm: 2_000,
            block: 1_000,
            refit_at: None,
        }
    }

    fn warm(mut self, warm: usize) -> Self {
        self.warm = warm;
        self
    }

    fn block(mut self, block: usize) -> Self {
        self.block = block;
        self
    }
}

fn model_case(id: &str, n: usize) -> ModelCase {
    let lin = Shape::Linear;
    let (kind, dim) = id.split_once('.').expect("<kind>.<dimension>");
    match (kind, dim) {
        ("ewridge", "k") => ModelCase::new(
            fit(json!({"type": "ewridge", "solve_every": 1e18}), n, 1),
            lin,
        ),
        ("ewridge", "solve_k") => ModelCase::new(
            fit(
                json!({"type": "ewridge", "max_rows_between_solves": 1}),
                n,
                1,
            ),
            lin,
        )
        .block(200),
        ("ewridge", "targets") => ModelCase::new(fit(json!({"type": "ewridge"}), 8, n), lin),
        ("ewridge", "ridge_grid") => {
            let grid: Vec<f64> = (0..n).map(|i| 1e-3 * (i + 1) as f64).collect();
            ModelCase::new(
                fit(
                    json!({"type": "ewridge", "ridge": grid, "max_rows_between_solves": 1}),
                    8,
                    1,
                ),
                lin,
            )
            .block(200)
        }
        ("ewridge" | "huber", "feature_sets") => {
            let sets: Vec<Value> = (0..n)
                .map(|i| json!([format!("s{i}"), names("x", 8)]))
                .collect();
            let model = if kind == "ewridge" {
                json!({"type": "ewridge", "feature_sets": sets, "max_rows_between_solves": 1})
            } else {
                json!({"type": "huber", "feature_sets": sets, "max_rows_between_solves": 1})
            };
            ModelCase::new(fit(model, 8, 1), lin).block(200)
        }
        // A window of `n` clock units, about `n / 1.25` rows, a snapshot a
        // row (the default): the warm-up fills it twice over.
        ("ewridge" | "ew_cov", "window") => {
            let spec = if kind == "ewridge" {
                fit(json!({"type": "ewridge", "window_size": n as f64}), 4, 1)
            } else {
                untargeted(json!({"type": "ew_cov", "window_size": n as f64}), 4, true)
            };
            ModelCase::new(spec, lin).warm(2 * n + 1_000)
        }
        ("ewridge", "gram_block") => ModelCase::new(
            fit(
                json!({"type": "ewridge", "gram_block_rows": n, "solve_every": 400.0}),
                16,
                1,
            ),
            lin,
        ),
        ("ewridge", "history") => {
            ModelCase::new(fit(json!({"type": "ewridge"}), 8, 1), lin).warm(n)
        }
        ("rls", "k") => ModelCase::new(fit(json!({"type": "rls", "delta": 1.0}), n, 1), lin),
        ("rls", "targets") => ModelCase::new(fit(json!({"type": "rls", "delta": 1.0}), 8, n), lin),
        ("rls", "history") => {
            ModelCase::new(fit(json!({"type": "rls", "delta": 1.0}), 8, 1), lin).warm(n)
        }
        ("lasso", "k") => ModelCase::new(
            fit(
                json!({"type": "lasso", "penalties": [0.1, 0.01, 0.0], "solve_every": 1e18}),
                n,
                1,
            ),
            lin,
        ),
        ("lasso", "path") => {
            let path: Vec<f64> = (0..n).map(|i| 0.1 / (1 << i.min(40)) as f64).collect();
            ModelCase::new(fit(json!({"type": "lasso", "penalties": path}), 8, 1), lin)
        }
        ("lasso", "history") => ModelCase::new(
            fit(
                json!({"type": "lasso", "penalties": [0.1, 0.01, 0.0]}),
                8,
                1,
            ),
            lin,
        )
        .warm(n),
        ("kalman", "k") => ModelCase::new(
            fit(json!({"type": "kalman", "coef_half_life": 2000.0}), n, 1),
            lin,
        ),
        ("kalman", "targets") => ModelCase::new(
            fit(json!({"type": "kalman", "coef_half_life": 2000.0}), 8, n),
            lin,
        ),
        ("kalman", "history") => ModelCase::new(
            fit(json!({"type": "kalman", "coef_half_life": 2000.0}), 8, 1),
            lin,
        )
        .warm(n),
        ("huber", "k") => ModelCase::new(
            fit(json!({"type": "huber", "solve_every": 1e18}), n, 1),
            lin,
        ),
        ("huber", "targets") => ModelCase::new(fit(json!({"type": "huber"}), 8, n), lin),
        ("huber", "history") => ModelCase::new(fit(json!({"type": "huber"}), 8, 1), lin).warm(n),
        // The rows of the first refit held: the last block ends before the
        // refit runs, with about `n` rows held (the refit's buffer).
        ("huber" | "quantile" | "sgd", "refit_rows") => {
            let model = match kind {
                "huber" => json!({"type": "huber", "huber_refit_rows": n}),
                "quantile" => {
                    json!({"type": "quantile", "quantile": 0.5, "quantile_refit_rows": n})
                }
                _ => {
                    json!({"type": "sgd", "learning_rate": 0.01, "loss": "huber", "huber_refit_rows": n})
                }
            };
            let mut c = ModelCase::new(fit(model, 4, 1), lin).block(200);
            c.refit_at = Some(n);
            c
        }
        ("quantile", "k") => ModelCase::new(
            fit(
                json!({"type": "quantile", "quantile": 0.5, "solve_every": 1e18}),
                n,
                1,
            ),
            lin,
        )
        .block(200),
        ("quantile", "history") => {
            ModelCase::new(fit(json!({"type": "quantile", "quantile": 0.5}), 8, 1), lin).warm(n)
        }
        ("sgd", "k") => ModelCase::new(
            fit(json!({"type": "sgd", "learning_rate": 0.01}), n, 1),
            lin,
        ),
        ("sgd", "targets") => ModelCase::new(
            fit(json!({"type": "sgd", "learning_rate": 0.01}), 8, n),
            lin,
        ),
        ("sgd", "history") => ModelCase::new(
            fit(json!({"type": "sgd", "learning_rate": 0.01}), 8, 1),
            lin,
        )
        .warm(n),
        ("pa", "k") => ModelCase::new(fit(json!({"type": "pa"}), n, 1), lin),
        ("pa", "targets") => ModelCase::new(fit(json!({"type": "pa"}), 8, n), lin),
        ("pa", "history") => ModelCase::new(fit(json!({"type": "pa"}), 8, 1), lin).warm(n),
        ("ftrl", "k") => ModelCase::new(fit(json!({"type": "ftrl"}), n, 1), lin),
        ("ftrl", "history") => ModelCase::new(fit(json!({"type": "ftrl"}), 8, 1), lin).warm(n),
        ("holt", "targets" | "history") => {
            let t = if dim == "targets" { n } else { 1 };
            let mut s = fit(json!({"type": "holt"}), 0, t);
            s.as_object_mut().expect("an object").remove("min_weight");
            let c = ModelCase::new(s, lin);
            if dim == "history" { c.warm(n) } else { c }
        }
        ("ew_cov", "k") => ModelCase::new(untargeted(json!({"type": "ew_cov"}), n, true), lin),
        ("ew_cov", "lags") => {
            let lags: Vec<usize> = (1..=n).collect();
            ModelCase::new(
                untargeted(
                    json!({"type": "ew_cov", "stats": ["mean", "cov"], "lags": lags}),
                    4,
                    true,
                ),
                lin,
            )
        }
        ("ew_cov", "history") => {
            ModelCase::new(untargeted(json!({"type": "ew_cov"}), 8, true), lin).warm(n)
        }
        ("marginal", "k") => ModelCase::new(fit(json!({"type": "marginal"}), n, 1), lin),
        ("marginal", "targets") => ModelCase::new(fit(json!({"type": "marginal"}), 8, n), lin),
        ("marginal", "lags") => {
            let lags: Vec<usize> = (1..=n).collect();
            ModelCase::new(fit(json!({"type": "marginal", "lags": lags}), 4, 1), lin)
        }
        // `n` bins a feature, from `n - 1` edges spread over the features'
        // range.
        ("marginal", "bins") => {
            let edges: Vec<f64> = (1..n).map(|i| -1.0 + 2.0 * i as f64 / n as f64).collect();
            ModelCase::new(
                fit(
                    json!({"type": "marginal", "bin_edges": vec![edges; 4]}),
                    4,
                    1,
                ),
                lin,
            )
        }
        ("marginal", "history") => {
            ModelCase::new(fit(json!({"type": "marginal"}), 8, 1), lin).warm(n)
        }
        ("deco", "k") => ModelCase::new(untargeted(json!({"type": "deco"}), n, true), lin),
        // `n` blocks of four series each.
        ("deco", "blocks") => {
            let blocks: Vec<(String, Vec<String>)> = (0..n)
                .map(|b| {
                    (
                        format!("b{b}"),
                        (4 * b..4 * b + 4).map(|j| format!("x{j}")).collect(),
                    )
                })
                .collect();
            ModelCase::new(
                untargeted(json!({"type": "deco", "blocks": blocks}), 4 * n, true),
                lin,
            )
        }
        ("deco", "history") => {
            ModelCase::new(untargeted(json!({"type": "deco"}), 8, true), lin).warm(n)
        }
        // The kernel estimator's block, never closed here: the bank closes
        // it, so a model alone holds every row it is fed.
        ("rcov", "k") => {
            let mut s = untargeted(
                json!({"type": "rcov", "kind": "kernel", "block_rows": 2000}),
                n,
                false,
            );
            s["group"] = json!("g");
            s["group_close"] = json!("monotone");
            ModelCase::new(s, Shape::Blobs)
        }
        ("rcov", "block_rows") => {
            let mut s = untargeted(
                json!({"type": "rcov", "kind": "kernel", "block_rows": n}),
                4,
                false,
            );
            s["group"] = json!("g");
            s["group_close"] = json!("monotone");
            ModelCase::new(s, Shape::Blobs).warm(n).block(200)
        }
        ("hmm", "states") => ModelCase::new(
            untargeted(
                json!({"type": "hmm", "n_components": n, "precision_prior": 1.0}),
                4,
                true,
            ),
            Shape::Blobs,
        )
        .block(200),
        ("hmm", "k") => ModelCase::new(
            untargeted(
                json!({"type": "hmm", "n_components": 2, "precision_prior": 1.0, "covariance_type": "diag"}),
                n,
                true,
            ),
            Shape::Blobs,
        ),
        ("hmm", "history") => ModelCase::new(
            untargeted(
                json!({"type": "hmm", "n_components": 2, "precision_prior": 1.0}),
                4,
                true,
            ),
            Shape::Blobs,
        )
        .warm(n),
        ("kmeans", "clusters") => ModelCase::new(
            untargeted(json!({"type": "kmeans", "n_clusters": n}), 4, true),
            Shape::Blobs,
        ),
        ("kmeans", "k") => ModelCase::new(
            untargeted(json!({"type": "kmeans", "n_clusters": 4}), n, true),
            Shape::Blobs,
        ),
        ("kmeans", "history") => ModelCase::new(
            untargeted(json!({"type": "kmeans", "n_clusters": 8}), 4, true),
            Shape::Blobs,
        )
        .warm(n),
        // Uniform rows and a radius a sixth of their range a coordinate, so
        // the summaries fill to their cap (a smaller radius opens an outlier
        // a row, pruned before it grows).
        ("micro", "max_clusters") => ModelCase::new(
            untargeted(
                json!({"type": "micro", "eps": 0.3, "max_clusters": n}),
                4,
                true,
            ),
            Shape::Linear,
        ),
        ("micro", "history") => ModelCase::new(
            untargeted(json!({"type": "micro", "eps": 1.0}), 4, true),
            Shape::Blobs,
        )
        .warm(n),
        ("ew_class", "classes" | "k" | "k_diag" | "history") => {
            let (classes, k) = match dim {
                "classes" => (n, 8),
                "k" | "k_diag" => (3, n),
                _ => (3, 8),
            };
            let labels: Vec<String> = (0..classes).map(|c| c.to_string()).collect();
            let mut model = json!({"type": "ew_class", "classes": labels, "precision_prior": 1.0});
            if dim == "k_diag" {
                model["covariance_type"] = json!("diag");
            }
            let mut s = fit(model, k, 1);
            s.as_object_mut().expect("an object").remove("min_weight");
            let c = ModelCase::new(s, Shape::Classes(classes)).block(200);
            if dim == "history" { c.warm(n) } else { c }
        }
        ("seqtest", "targets" | "history") => {
            let t = if dim == "targets" { n } else { 1 };
            let s = json!({
                "name": "m",
                "model": {"type": "seqtest"},
                "targets": names("y", t),
                "features": [],
                "clock": "t",
                "gap_cap": 100.0,
            });
            let c = ModelCase::new(s, lin);
            if dim == "history" { c.warm(n) } else { c }
        }
        ("corrchange", "k") => ModelCase::new(
            untargeted(
                json!({"type": "corrchange", "kind": "monitor", "window_rows": 500}),
                n,
                false,
            ),
            lin,
        ),
        ("corrchange", "window") => ModelCase::new(
            untargeted(
                json!({"type": "corrchange", "kind": "window", "window_rows": n,
                       "n_perm": 100, "permute_every_rows": 250}),
                4,
                false,
            ),
            lin,
        )
        .warm(2 * n + 2_000)
        .block(200),
        // A fixed threshold: no permutation drawn, the report alone.
        ("corrchange", "window_crit") => ModelCase::new(
            untargeted(
                json!({"type": "corrchange", "kind": "window", "window_rows": n, "crit": 1.0}),
                4,
                false,
            ),
            lin,
        )
        .warm(2 * n + 2_000)
        .block(200),
        ("corrchange", "history") => ModelCase::new(
            untargeted(
                json!({"type": "corrchange", "kind": "window", "window_rows": 100,
                       "n_perm": 100, "permute_every_rows": 250}),
                4,
                false,
            ),
            lin,
        )
        .warm(n),
        ("bocpd", "max_run") => ModelCase::new(
            untargeted(
                json!({"type": "bocpd", "prune_below": 0.0, "max_run": n}),
                4,
                false,
            ),
            Shape::Blobs,
        ),
        ("bocpd", "k") => ModelCase::new(
            untargeted(
                json!({"type": "bocpd", "prune_below": 0.0, "max_run": 50}),
                n,
                false,
            ),
            Shape::Blobs,
        ),
        ("bocpd", "k_full") => ModelCase::new(
            untargeted(
                json!({"type": "bocpd", "covariance_type": "full",
                       "degrees_of_freedom_prior": n as f64 + 2.0,
                       "prune_below": 0.0, "max_run": 50}),
                n,
                false,
            ),
            Shape::Blobs,
        )
        .block(200),
        ("bocpd", "history") => ModelCase::new(
            untargeted(
                json!({"type": "bocpd", "prune_below": 1e-6, "max_run": 200}),
                4,
                false,
            ),
            Shape::Blobs,
        )
        .warm(n),
        ("audit", "k" | "pairs" | "distinct_cap" | "history") => {
            let (k, pairs, cap) = match dim {
                "k" => (n, false, 256),
                "pairs" => (n, true, 256),
                "distinct_cap" => (4, false, n),
                _ => (4, false, 256),
            };
            let s = json!({
                "name": "m",
                "model": {"type": "audit", "pairs": pairs, "distinct_cap": cap},
                "features": names("x", k),
                "clock": "t",
                "gap_cap": 100.0,
            });
            let c = ModelCase::new(s, lin);
            if dim == "history" { c.warm(n) } else { c }
        }
        _ => panic!("no model cell {id:?}"),
    }
}

fn built(id: &str, value: Value) -> Spec {
    let mut spec: Spec = serde_json::from_value(value).unwrap_or_else(|e| panic!("{id}: {e}"));
    spec.check().unwrap_or_else(|e| panic!("{id}: {e}"));
    spec
}

fn model_setup(id: &'static str, n: usize, blocks: usize) -> ScaleRun {
    let mut case = model_case(id, n);
    if let Some(at) = case.refit_at {
        case.warm = at
            .checked_sub(blocks * case.block + 100)
            .unwrap_or_else(|| {
                panic!("{id}: {n} rows too few for {blocks} blocks before the refit")
            });
    }
    let spec = built(id, case.spec);
    let mut models = online_polars::build_models(&spec).unwrap_or_else(|e| panic!("{id}: {e}"));
    assert_eq!(models.len(), 1, "{id}: one instance");
    let (_, mut model) = models.pop().expect("one instance");
    let rows = Rows::generate(
        case.shape,
        spec.k(),
        spec.m(),
        case.warm + blocks * case.block,
        0,
        7,
    );
    let mut step = online_polars::online_core::Step::default();
    for i in 0..case.warm {
        let (x, y, d) = rows.row(i);
        model.step_into(x, y, d, 1.0, &mut step);
    }
    rayon::current_num_threads();
    ScaleRun {
        id,
        size: n,
        work: Work::Model {
            model: Box::new(model),
            step,
            rows,
            warm: case.warm,
        },
        block_rows: case.block,
        blocks,
        next: 0,
    }
}

// --- frames for the bank, the window core and the samplers -----------------------

/// `n` rows of `k` features and `t` targets as a frame, with the clock `t`
/// and, when `group_of` is given, the group `g` of each row, each group
/// keeping a clock of its own, a unit a row.
fn frame(
    k: usize,
    t: usize,
    n: usize,
    seed: u64,
    group_of: Option<&dyn Fn(usize) -> usize>,
) -> DataFrame {
    let rows = Rows::generate(Shape::Linear, k, t, n, 0, seed);
    let mut df = rows.frame(0, n);
    if let Some(group_of) = group_of {
        let mut clocks: HashMap<usize, f64> = HashMap::new();
        let (mut g, mut c) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let gi = group_of(i);
            let clock = clocks.entry(gi).or_insert(1_000.0);
            *clock += 1.0;
            g.push(gi as i64);
            c.push(*clock);
        }
        df.with_column(Column::new("g".into(), g))
            .expect("a column");
        df.with_column(Column::new("t".into(), c))
            .expect("a column");
    }
    df
}

/// Rows `from..from + n` of `df` as a frame of their own buffers, as a
/// stream's chunks arrive: a slice shares its parent's, so a state that
/// keeps a row of a slice would keep every row of the parent.
fn owned(df: &DataFrame, from: usize, n: usize) -> DataFrame {
    let part = df.slice(from as i64, n);
    let cols: Vec<Column> = part
        .columns()
        .iter()
        .map(|c| {
            let s = c.as_materialized_series().rechunk();
            let fresh: Series = match s.dtype() {
                DataType::Float64 => {
                    let v = s
                        .f64()
                        .expect("floats")
                        .cont_slice()
                        .expect("no nulls")
                        .to_vec();
                    Series::new(c.name().clone(), v)
                }
                DataType::Int64 => {
                    let v = s
                        .i64()
                        .expect("integers")
                        .cont_slice()
                        .expect("no nulls")
                        .to_vec();
                    Series::new(c.name().clone(), v)
                }
                other => panic!("no copy for {other:?}"),
            };
            fresh.with_name(c.name().clone()).into_column()
        })
        .collect();
    DataFrame::new_infer_height(cols).expect("columns of one length")
}

/// `frame`'s rows `[warm, warm + blocks·block)` cut into blocks.
fn blocks_of(df: &DataFrame, warm: usize, block: usize, blocks: usize) -> Vec<DataFrame> {
    (0..blocks)
        .map(|b| owned(df, warm + b * block, block))
        .collect()
}

/// `frame`'s first `warm` rows, as chunks of `chunk` rows.
fn chunks_of(df: &DataFrame, warm: usize, chunk: usize) -> Vec<DataFrame> {
    (0..warm)
        .step_by(chunk)
        .map(|from| owned(df, from, chunk.min(warm - from)))
        .collect()
}

fn bank_setup(id: &'static str, n: usize, blocks: usize) -> ScaleRun {
    let ewridge = |k: usize, t: usize| fit(json!({"type": "ewridge"}), k, t);
    let (dim, block) = (id.split_once('.').expect("bank.<dimension>").1, 1_000);
    let mut warm = 2_000;
    let mut chunk = block;
    let mut specs: Vec<Value> = vec![ewridge(4, 1)];
    let mut group_of: Option<Box<dyn Fn(usize) -> usize>> = None;
    let (mut k, mut t) = (4, 1);
    match dim {
        // Every group seen in the warm-up, then a block that touches the
        // first 50 alone: a row's work must not grow with the groups behind
        // it.
        "groups_seen" => {
            specs[0]["group"] = json!("g");
            warm = 10 * n;
            group_of = Some(Box::new(move |i| if i < 10 * n { i % n } else { i % 50 }));
        }
        "specs" | "specs_wide" | "specs_per_call" | "specs_per_call_wide" => {
            specs = (0..n)
                .map(|s| {
                    let mut v = ewridge(4, 1);
                    v["name"] = json!(format!("m{s}"));
                    v
                })
                .collect();
            if dim.starts_with("specs_per_call") {
                chunk = 4;
            }
        }
        // Spec `s` reads features `x{4s}..x{4s+3}` and target `y{s}`.
        "specs_distinct" | "specs_distinct_wide" => {
            (k, t) = (4 * n, n);
            specs = (0..n)
                .map(|s| {
                    let mut v = ewridge(4, 1);
                    v["name"] = json!(format!("m{s}"));
                    v["features"] = json!(
                        (4 * s..4 * s + 4)
                            .map(|j| format!("x{j}"))
                            .collect::<Vec<_>>()
                    );
                    v["targets"] = json!([format!("y{s}")]);
                    v
                })
                .collect();
        }
        // `n` copies of a mix of five kinds over the same four features.
        "specs_mixed" => {
            let four = |model: Value, decays: bool| untargeted(model, 4, decays);
            specs = (0..n)
                .flat_map(|c| {
                    let mix = [
                        fit(json!({"type": "ewridge"}), 4, 1),
                        fit(json!({"type": "huber"}), 4, 1),
                        four(json!({"type": "ew_cov"}), true),
                        four(json!({"type": "kmeans", "n_clusters": 4}), true),
                        four(
                            json!({"type": "bocpd", "prune_below": 1e-6, "max_run": 50}),
                            false,
                        ),
                    ];
                    mix.into_iter().enumerate().map(move |(i, mut v)| {
                        v["name"] = json!(format!("m{c}_{i}"));
                        v
                    })
                })
                .collect();
        }
        "k" => {
            k = n;
            specs = vec![ewridge(n, 1)];
        }
        "targets" => {
            t = n;
            specs = vec![ewridge(4, n)];
        }
        // The same rows in calls of `n` rows each: a block is 4,096 rows at
        // either size.
        "chunk_rows" => chunk = n,
        "history" => warm = n,
        "embargo" => specs[0]["embargo"] = json!(n as f64),
        "formula_window" => {
            specs[0]["targets"] = json!([{
                "name": "fwd",
                "formula": ["rewm_mean", ["col", "y0"], {"half_life": 1000.0, "window_size": n as f64}],
            }]);
            specs[0]["embargo"] = json!(n as f64);
            warm = 2_000 + 2 * n;
        }
        "ljung_box_lags" => {
            specs[0]["emit_specification"] = json!(true);
            specs[0]["ljung_box_lags"] = json!(n);
        }
        "robust_se_lags" => {
            specs[0]["emit_robust_se"] = json!(true);
            specs[0]["robust_se_lags"] = json!(n);
        }
        "resid_quantiles" => {
            let q: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
            specs[0]["resid_quantiles"] = json!(q);
        }
        other => panic!("no bank cell {other:?}"),
    }
    let block = if dim == "chunk_rows" {
        4_096
    } else if chunk == 4 {
        1_024
    } else {
        block
    };
    let specs: Vec<Spec> = specs.into_iter().map(|v| built(id, v)).collect();
    let mut bank = Bank::new(specs).unwrap_or_else(|e| panic!("{id}: {e}"));
    online_polars::pool().unwrap_or_else(|e| panic!("{id}: {e}"));
    bank.set_threads(Some(1))
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    let df = frame(k, t, warm + blocks * block, 7, group_of.as_deref());
    for part in chunks_of(&df, warm, 4_096) {
        bank.fit_predict(&part)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
    }
    // Each block fed as calls of `chunk` rows (all of it in one call but
    // for `bank.chunk_rows`).
    let frames = blocks_of(&df, warm, block, blocks)
        .into_iter()
        .map(|f| {
            (0..block / chunk)
                .map(|c| owned(&f, c * chunk, chunk))
                .collect()
        })
        .collect();
    ScaleRun {
        id,
        size: n,
        work: Work::Bank {
            bank: Box::new(bank),
            frames,
        },
        block_rows: block,
        blocks,
        next: 0,
    }
}

fn windows_setup(id: &'static str, n: usize, blocks: usize) -> ScaleRun {
    let dim = id.split_once('.').expect("windows.<dimension>").1;
    let block = 1_000;
    let sum = |col: &str| json!(["ewm_sum", ["col", col], {"half_life": 1000.0}]);
    let windowed =
        |op: &str| json!([op, ["col", "x0"], {"half_life": 1000.0, "window_size": n as f64}]);
    let mut warm = 2_000;
    let mut k = 1;
    let mut group_of: Option<Box<dyn Fn(usize) -> usize>> = None;
    let formulas: Vec<Value> = match dim {
        "groups" => {
            warm = 2 * n;
            group_of = Some(Box::new(move |i| i % n));
            vec![sum("x0")]
        }
        "groups_seen" => {
            warm = 2 * n;
            group_of = Some(Box::new(move |i| if i < 2 * n { i % n } else { i % 50 }));
            vec![sum("x0")]
        }
        // `n` kernels: one operator each, at `n` half-lives.
        "kernels" => (0..n)
            .map(|i| json!(["ewm_sum", ["col", "x0"], {"half_life": 100.0 * (i + 1) as f64}]))
            .collect(),
        // `n` operators on one kernel, one a column.
        "operators" => {
            k = n;
            (0..n).map(|j| sum(&format!("x{j}"))).collect()
        }
        "window_mean" => {
            warm = 2 * n;
            vec![windowed("ewm_mean")]
        }
        "window_var" => {
            warm = 2 * n;
            vec![windowed("ewm_var")]
        }
        "window_quantile" => {
            warm = 2 * n;
            vec![
                json!(["ewm_quantile", ["col", "x0"], {"half_life": 1000.0, "window_size": n as f64, "quantile": 0.5}]),
            ]
        }
        "forward" => {
            warm = 2 * n;
            vec![windowed("rewm_mean")]
        }
        "history" => {
            warm = n;
            vec![sum("x0")]
        }
        other => panic!("no windows cell {other:?}"),
    };
    let mut config = json!({
        "formulas": formulas
            .into_iter()
            .enumerate()
            .map(|(i, tree)| json!({"name": format!("w{i}"), "tree": tree}))
            .collect::<Vec<_>>(),
        "clock": "t",
        "gap_cap": 100.0,
    });
    if group_of.is_some() {
        config["group"] = json!("g");
        config["gap_cap"] = json!(1e12);
    }
    let config: WindowsConfig =
        serde_json::from_value(config).unwrap_or_else(|e| panic!("{id}: {e}"));
    let df = frame(k, 0, warm + blocks * block, 7, group_of.as_deref());
    let mut run = WindowsRun::new(config, df.schema()).unwrap_or_else(|e| panic!("{id}: {e}"));
    // The warm-up in chunks of the block's size, as a stream arrives: a
    // chunk's scratch is sized to the chunk, whatever the stream's length.
    for part in chunks_of(&df, warm, block) {
        run.feed(&part, None)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
    }
    ScaleRun {
        id,
        size: n,
        work: Work::Windows {
            run: Box::new(run),
            frames: blocks_of(&df, warm, block, blocks),
        },
        block_rows: block,
        blocks,
        next: 0,
    }
}

fn refresh_cols(grouped: bool) -> RefreshCols<'static> {
    RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: grouped.then_some("g"),
        keep: &[],
    }
}

/// Ticks of `m` series in turn, a clock unit apart, each a random value;
/// over `groups` groups (0: none), tick `i` is group `(i / m) % groups`'s.
fn ticks(m: usize, n: usize, groups: usize, from: usize) -> DataFrame {
    let mut g = Lcg(5 + from as u64);
    let series: Vec<String> = (from..from + n).map(|i| format!("s{}", i % m)).collect();
    let t: Vec<f64> = (from..from + n)
        .map(|i| match groups {
            0 => i as f64,
            gs => ((i / m) / gs * m + i % m) as f64,
        })
        .collect();
    let v: Vec<f64> = (0..n).map(|_| g.signed()).collect();
    let mut cols = vec![
        Column::new("series".into(), series),
        Column::new("t".into(), t),
        Column::new("v".into(), v),
    ];
    if groups > 0 {
        let gc: Vec<i64> = (from..from + n)
            .map(|i| ((i / m) % groups) as i64)
            .collect();
        cols.push(Column::new("g".into(), gc));
    }
    DataFrame::new_infer_height(cols).expect("columns of one length")
}

fn refresh_setup(id: &'static str, n: usize, blocks: usize) -> ScaleRun {
    let dim = id.split_once('.').expect("refresh.<dimension>").1;
    let block = 1_000;
    let (m, groups, pairs, warm) = match dim {
        "series" => (n, 0, false, 2_000),
        "pairs" => (n, 0, true, 2_000),
        "groups" => (4, n, false, 8 * n),
        "history" => (4, 0, false, n),
        other => panic!("no refresh cell {other:?}"),
    };
    let names: Vec<String> = (0..m).map(|i| format!("s{i}")).collect();
    let mut rt = RefreshTime::new(names, pairs).unwrap_or_else(|e| panic!("{id}: {e}"));
    let cols = refresh_cols(groups > 0);
    rt.feed(&ticks(m, warm, groups, 0), &cols)
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    let frames = (0..blocks)
        .map(|b| ticks(m, block, groups, warm + b * block))
        .collect();
    ScaleRun {
        id,
        size: n,
        work: Work::Refresh {
            rt: Box::new(rt),
            frames,
            grouped: groups > 0,
        },
        block_rows: block,
        blocks,
        next: 0,
    }
}

fn gbdt_setup(id: &'static str, n: usize, blocks: usize) -> ScaleRun {
    use online_polars::gbdt::{Kind, Layout, Sampler};
    let dim = id.split_once('.').expect("gbdt.<dimension>").1;
    let block = 1_000;
    let (sample, warm) = match dim {
        "history" => (1_000, n),
        "sample" => (n, 2 * n),
        other => panic!("no gbdt cell {other:?}"),
    };
    let k = 4;
    let layout = Layout {
        names: names("x", k),
        kinds: vec![Kind::Float; k],
        label: "y0".to_owned(),
        weight: None,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool");
    let mut sampler = Sampler::new(vec![Kind::Float; k], sample, 7, false);
    let df = frame(k, 1, warm + blocks * block, 7, None)
        .select(names("x", k).into_iter().chain(["y0".to_owned()]))
        .expect("the columns");
    for part in chunks_of(&df, warm, block) {
        sampler
            .push_frame(&part, &layout, &pool)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
    }
    ScaleRun {
        id,
        size: n,
        work: Work::Gbdt {
            sampler: Box::new(sampler),
            layout,
            pool,
            frames: blocks_of(&df, warm, block, blocks),
        },
        block_rows: block,
        blocks,
        next: 0,
    }
}

fn runner_setup(id: &'static str, n: usize, blocks: usize) -> ScaleRun {
    let dim = id.split_once('.').expect("runner.<dimension>").1;
    let (specs, rows, call) = match dim {
        "rows" => (1, n, 1_024),
        "specs" => (n, 4_096, 1_024),
        "specs_per_call" => (n, 512, 16),
        other => panic!("no runner cell {other:?}"),
    };
    let specs: Vec<Spec> = (0..specs)
        .map(|s| {
            let mut v = fit(json!({"type": "ewridge"}), 4, 1);
            v["name"] = json!(format!("m{s}"));
            built(id, v)
        })
        .collect();
    let mut bank = Bank::new(specs).unwrap_or_else(|e| panic!("{id}: {e}"));
    online_polars::pool().unwrap_or_else(|e| panic!("{id}: {e}"));
    bank.set_threads(Some(1))
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    let warm = 2_048;
    let df = frame(4, 1, warm + blocks * rows, 7, None);
    for part in chunks_of(&df, warm, 1_024) {
        bank.fit_predict(&part)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
    }
    let frames = blocks_of(&df, warm, rows, blocks)
        .into_iter()
        .map(|f| {
            (0..rows.div_ceil(call))
                .map(|c| owned(&f, c * call, call.min(rows - c * call)))
                .collect()
        })
        .collect();
    ScaleRun {
        id,
        size: n,
        work: Work::Runner {
            bank: Box::new(bank),
            frames,
            schema: df.schema().as_ref().clone(),
        },
        block_rows: rows,
        blocks,
        next: 0,
    }
}

/// A cell's workload at size `n`, warmed up, with `blocks` blocks to run.
pub fn scale_setup_blocks(id: &'static str, n: usize, blocks: usize) -> ScaleRun {
    match id.split_once('.').expect("<surface>.<dimension>").0 {
        "bank" => bank_setup(id, n, blocks),
        "runner" => runner_setup(id, n, blocks),
        "windows" => windows_setup(id, n, blocks),
        "refresh" => refresh_setup(id, n, blocks),
        "gbdt" => gbdt_setup(id, n, blocks),
        _ => model_setup(id, n, blocks),
    }
}

/// The bench's setup: `"<cell>:small"` or `"<cell>:large"`, one block.
pub fn scale_setup(arg: &str) -> ScaleRun {
    let (id, which) = arg.split_once(':').expect("<cell>:small or <cell>:large");
    let c = cell_of(id);
    let n = match which {
        "small" => c.small,
        "large" => c.large(),
        other => panic!("{other:?}: small or large"),
    };
    scale_setup_blocks(c.id, n, 1)
}
