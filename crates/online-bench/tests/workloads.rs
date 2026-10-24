//! Every instruction-count case, run natively: it builds, and its measured
//! block reaches the path it is named for. A count of a block that predicts
//! nothing, or of a model still in its warm-up, would hold a budget on a
//! path no user runs, so each case proves it ran past the warm-up first.

use online_bench::{
    BANK_CASES, CLOCK_CASES, MODEL_CASES, ROWS, SPAN_CASE, WARM_ROWS, WIDE_CASES, WIDE_ROWS,
    WINDOW_CASES, bank_setup, clock_setup, model_setup, polars_online_budget_bank_rows,
    polars_online_budget_clock_rows, polars_online_budget_model_rows,
    polars_online_budget_windows_rows, spec, windows_setup,
};
use online_polars::{ChunkOut, PAR_MIN_ROWS, Stream};
use polars::prelude::*;

/// The bench names every case, and names nothing else: a case missing from
/// it has no count, and so no budget.
#[test]
fn the_bench_names_every_case_and_nothing_else() {
    let bench = include_str!("../benches/instructions.rs");
    let named: Vec<&str> = bench
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#[bench::"))
        .map(|l| &l[..l.find('(').expect("an argument list")])
        .collect();
    let mut cases: Vec<&str> = MODEL_CASES
        .iter()
        .chain(CLOCK_CASES)
        .chain(BANK_CASES)
        .chain(WIDE_CASES)
        .chain(WINDOW_CASES)
        .copied()
        .collect();
    let mut sorted = named.clone();
    sorted.sort_unstable();
    cases.sort_unstable();
    assert_eq!(sorted, cases);
}

#[test]
fn every_model_case_predicts_on_every_row_of_its_block() {
    for &id in MODEL_CASES {
        let mut run = model_setup(id);
        let block = polars_online_budget_model_rows(&mut run);
        assert!(block.checksum.is_finite(), "{id}: {block:?}");
        let want = if id == SPAN_CASE { 1 } else { ROWS };
        assert_eq!(block.predicted, want, "{id}: still warming up: {block:?}");
    }
}

#[test]
fn every_model_case_is_the_same_count_of_work_each_time() {
    // The same rows, the same state, the same numbers: what makes one run's
    // count stand for every run's.
    for &id in MODEL_CASES {
        let (mut a, mut b) = (model_setup(id), model_setup(id));
        let (x, y) = (
            polars_online_budget_model_rows(&mut a),
            polars_online_budget_model_rows(&mut b),
        );
        assert_eq!(x.checksum.to_bits(), y.checksum.to_bits(), "{id}");
    }
}

#[test]
fn every_clock_case_stamps_every_row_and_caps_none() {
    for &id in CLOCK_CASES {
        let mut run = clock_setup(id);
        let block = polars_online_budget_clock_rows(&mut run);
        assert_eq!(block.predicted, ROWS, "{id}: {block:?}");
        // Steps of 0.5 to 2 seconds, a second and a quarter on average:
        // under the cap of 100 every one.
        let mean = block.checksum / ROWS as f64;
        assert!((1.0..1.5).contains(&mean), "{id}: mean step {mean}");
    }
}

#[test]
fn every_bank_case_learns_its_block_below_the_parallel_threshold() {
    const { assert!(ROWS < online_polars::PAR_MIN_ROWS) };
    // The field each case's model fills on every row once it is warm.
    for (id, field) in [
        ("bank_seqtest", "log_e_pos_y0"),
        ("bank_ewridge_k5", "pred_y0"),
        ("formula_target_held1k", "pred_fwd"),
    ] {
        assert!(BANK_CASES.contains(&id), "{id}");
        let mut run = bank_setup(id);
        let before = run.bank.rows_fed();
        let out = polars_online_budget_bank_rows(&mut run);
        assert_eq!(run.bank.rows_fed() - before, ROWS as u64, "{id}");
        let fields = out[0]
            .as_materialized_series()
            .struct_()
            .expect("a struct")
            .fields_as_series();
        let p = fields
            .iter()
            .find(|f| f.name() == field)
            .unwrap_or_else(|| panic!("{id}: no {field} in {fields:?}"));
        assert_eq!(p.len(), ROWS, "{id}");
        assert_eq!(p.null_count(), 0, "{id}: {field} has nulls");
        let v = p.f64().expect("a float field");
        assert!(v.into_no_null_iter().all(f64::is_finite), "{id}: {field}");
        // Something moved: a sequential test whose evidence against the
        // drift stayed at zero, or a fit that predicted one constant, would
        // be a path no stream takes.
        assert!(
            v.into_no_null_iter().any(|x| x != v.get(0).unwrap()),
            "{id}: {field}"
        );
    }
    assert_eq!(BANK_CASES.len(), 3);
}

/// The formula-target case holds about a thousand rows behind each one: its
/// window of 1,250 clock units and its embargo of the same span, at a mean
/// step of 1.25.
#[test]
fn the_formula_target_case_holds_about_a_thousand_rows() {
    let mut run = bank_setup("formula_target_held1k");
    polars_online_budget_bank_rows(&mut run);
    // The rows the embargo holds, read from the saved state: the stream's
    // `pending` rows, beside which `notified.pending` holds only strings.
    fn held(v: &serde_json::Value) -> usize {
        match v {
            serde_json::Value::Object(m) => m
                .iter()
                .map(|(k, v)| match v {
                    serde_json::Value::Array(a)
                        if k == "pending" && a.iter().all(|r| r.is_object()) =>
                    {
                        a.len()
                    }
                    _ => held(v),
                })
                .sum(),
            serde_json::Value::Array(a) => a.iter().map(held).sum(),
            _ => 0,
        }
    }
    let state: serde_json::Value =
        serde_json::from_str(&run.bank.save_json_string(false).expect("a state")).expect("json");
    let held = held(&state);
    assert!((800..1_300).contains(&held), "{held} rows held");
}

/// Each window case emits its block, valued on every row, from inside the
/// stream's first window: no row has left it, so every row the case reads
/// is the case's path (a mean cuts its oldest row's held interval at the
/// edge on every one).
#[test]
fn every_window_case_reads_its_block_inside_the_first_window() {
    for &id in WINDOW_CASES {
        let mut run = windows_setup(id);
        let out = polars_online_budget_windows_rows(&mut run);
        assert_eq!(out.height(), ROWS, "{id}");
        let w = out
            .column("w")
            .expect("the operator")
            .f64()
            .expect("floats");
        assert_eq!(w.null_count(), 0, "{id}");
        assert!(w.into_no_null_iter().all(f64::is_finite), "{id}");
        assert_eq!(
            run.run.queued(),
            WARM_ROWS + ROWS,
            "{id}: a row left the window"
        );
    }
}

/// The wide cases learn their block on the pool they are named for, in
/// several runs and past the parallel threshold, so the count covers a
/// run's copy into the output overlapped with the next run's learning and
/// the parallel allocation and finish of the output columns; and the two
/// pool sizes return the same output to the bit.
#[test]
fn every_wide_case_learns_its_block_in_runs_on_its_pool() {
    let stream = Stream::new(&spec(WIDE_CASES[0])).expect("a stream");
    let run_rows = ChunkOut::run_rows(&spec(WIDE_CASES[0]), stream.n_models(), stream.n_slots());
    assert!(WIDE_ROWS >= 4 * run_rows, "{run_rows} rows a run");
    const { assert!(WIDE_ROWS >= PAR_MIN_ROWS) };
    let mut outs: Vec<Vec<Column>> = Vec::new();
    for (id, threads) in [(WIDE_CASES[0], 1), ("bank_ew_cov_wide_t4", 4)] {
        let mut run = bank_setup(id);
        assert_eq!(run.bank.threads().asked(), Some(threads), "{id}");
        let before = run.bank.rows_fed();
        let out = polars_online_budget_bank_rows(&mut run);
        assert_eq!(run.bank.rows_fed() - before, WIDE_ROWS as u64, "{id}");
        let fields = out[0]
            .as_materialized_series()
            .struct_()
            .expect("a struct")
            .fields_as_series();
        // Every statistic on every row: 20 means, 20 deviations and 190
        // correlations, beside the weight, the settled share and the reason.
        let stats: Vec<&Series> = fields
            .iter()
            .filter(|f| {
                ["mean_", "std_", "corr_"]
                    .iter()
                    .any(|p| f.name().starts_with(p))
            })
            .collect();
        assert_eq!((stats.len(), fields.len()), (230, 233), "{id}");
        for f in stats {
            assert_eq!(f.len(), WIDE_ROWS, "{id}");
            assert_eq!(f.null_count(), 0, "{id}: {} has nulls", f.name());
        }
        outs.push(out);
    }
    let one = outs[0][0].as_materialized_series();
    assert!(
        one.equals_missing(outs[1][0].as_materialized_series()),
        "one thread and four differ"
    );
}
