//! The two ways a spec's runs reach its output struct (task 255 (a)): one
//! run at a time as its stream runs, for a spec one stream serves, and every
//! run at once after the streams are done. A test-only switch sends the
//! specs named `sink-off-*` the second way, and a counter records the runs
//! the first way takes for the specs named `sink-on-*`, so the other tests
//! of this binary, running beside these, are untouched. The first way's
//! copy of a run overlaps the learning of the next through `rayon::join`,
//! which on a pool of one thread runs the two in turn, so its output is
//! held to the bit at every pool size as well.
use std::collections::HashMap;
use std::sync::Mutex;

use polars::prelude::*;
use polars_arrow::array::{Array, Float64Array, StructArray};

use crate::{Bank, Spec};

static ABSORBED: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

/// Whether `spec` is sent the way every run goes at once.
pub(in crate::bank) fn sink_off(spec: &str) -> bool {
    spec.starts_with("sink-off-")
}

/// Count a run a sink took as its stream ran, for the specs probed here.
pub(super) fn absorbed(spec: &str) {
    if spec.starts_with("sink-on-") {
        *ABSORBED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_insert_with(HashMap::new)
            .entry(spec.to_string())
            .or_default() += 1;
    }
}

fn runs_absorbed(spec: &str) -> usize {
    ABSORBED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(|m| m.get(spec).copied())
        .unwrap_or(0)
}

/// Every output a regression of a grid can emit, `coef` on every row:
/// `ewridge` the uncertainty fields, `lasso` the selected slot.
fn specs(name: &str) -> Vec<Spec> {
    let common = r#""targets": ["y", "z"], "features": ["x0", "x1", "x2"], "clock": "t",
        "gap_cap": 6.0, "weight": "w", "emit_clocks": true, "coef_every": 0,
        "emit_sigma": true, "emit_zscore": true, "emit_averaged": true, "emit_metrics": true,
        "conformal": 0.9, "resid_quantiles": [0.1, 0.9], "emit_autocorr": true,
        "emit_drift": true, "drift_threshold": 20.0, "emit_calibration": true,
        "emit_breaks": true, "emit_specification": true, "emit_tails": true,
        "emit_feature_health": true"#;
    let ewridge = format!(
        r#"{{"name": "{name}-ewridge", "model": {{"type": "ewridge", "ridge": [1e-6, 1.0]}},
            "half_life": [25.0, 100.0], "emit_error_inflation": true, "emit_se_coef": true,
            "emit_robust_se": true, "emit_influence": true, {common}}}"#
    );
    let lasso = format!(
        r#"{{"name": "{name}-lasso", "model": {{"type": "lasso", "penalties": [0.2, 0.0]}},
            "half_life": 25.0, "emit_selected": true, {common}}}"#
    );
    [ewridge, lasso]
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}

/// An irregular clock with a gap past the cap, null features (skipped
/// rows), null targets, zero weights: rows of every kind in every run.
fn frame(n: usize) -> DataFrame {
    let mut t = 0.0;
    let mut clock = Vec::with_capacity(n);
    for i in 0..n {
        t += [0.5, 1.0, 1.5][i % 3] + if i == n / 3 { 40.0 } else { 0.0 };
        clock.push(t);
    }
    let x = |a: usize, b: usize| -> Vec<Option<f64>> {
        (0..n)
            .map(|i| (i % 97 != a).then(|| ((i * b) % 23) as f64 / 7.0 - 1.5))
            .collect()
    };
    let y: Vec<Option<f64>> = (0..n)
        .map(|i| (i % 89 != 5).then(|| ((i * 7) % 19) as f64 / 5.0))
        .collect();
    let z: Vec<f64> = (0..n).map(|i| ((i * 11) % 17) as f64 / 3.0).collect();
    let w: Vec<f64> = (0..n)
        .map(|i| if i % 31 == 0 { 0.0 } else { 1.0 })
        .collect();
    df!("t" => clock, "x0" => x(3, 5), "x1" => x(50, 3), "x2" => x(98, 13), "y" => y,
        "z" => z, "w" => w)
    .unwrap()
}

/// Two struct arrays equal field for field: the same dtype and validity,
/// and for a float the same bits in every slot, a null's included.
fn assert_same(a: &StructArray, b: &StructArray, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    assert_eq!(a.dtype(), b.dtype(), "{what}");
    for (i, (x, y)) in a.values().iter().zip(b.values()).enumerate() {
        assert_eq!(x.validity(), y.validity(), "{what}: field {i}");
        match (
            x.as_any().downcast_ref::<Float64Array>(),
            y.as_any().downcast_ref::<Float64Array>(),
        ) {
            (Some(x), Some(y)) => {
                let bits =
                    |v: &Float64Array| v.values().iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(x), bits(y), "{what}: field {i}");
            }
            _ => assert!(x.as_ref() == y.as_ref(), "{what}: field {i}"),
        }
    }
}

fn structs(cols: &[Column]) -> Vec<StructArray> {
    cols.iter()
        .map(|c| {
            let s = c.as_materialized_series().rechunk();
            s.chunks()[0]
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone()
        })
        .collect()
}

/// A spec one stream serves takes its runs one at a time, and its output is
/// the output of the same runs taken at once, to the bit, over one chunk and
/// over three.
#[test]
fn a_run_at_a_time_is_every_run_at_once() {
    let df = frame(2400);
    for (k, parts) in [1usize, 3].into_iter().enumerate() {
        let on = format!("sink-on-{k}");
        let mut a = Bank::new(specs(&on)).unwrap();
        let mut b = Bank::new(specs(&format!("sink-off-{k}"))).unwrap();
        let step = df.height().div_ceil(parts);
        for p in 0..parts {
            let chunk = df.slice((p * step) as i64, step);
            let (sa, sb) = (
                structs(&a.fit_predict(&chunk).unwrap()),
                structs(&b.fit_predict(&chunk).unwrap()),
            );
            for (x, y) in sa.iter().zip(&sb) {
                assert_same(x, y, &format!("{parts} chunks, chunk {p}"));
            }
        }
        // The states differ only by the specs' names, which they carry;
        // the stream's last row is kept before a run goes to its sink.
        // Both specs took the first way, and `ewridge`, the wider, split
        // the one chunk into runs.
        let runs = |s: &str| runs_absorbed(&format!("{on}-{s}"));
        assert!(runs("ewridge") >= parts.max(2), "{} runs", runs("ewridge"));
        assert!(runs("lasso") >= parts, "{} runs", runs("lasso"));
        assert_eq!(runs_absorbed(&format!("sink-off-{k}-ewridge")), 0);
    }
}

/// The specs of the thread-count tests, named from `name`: a wide `ew_cov`
/// and a four-target `ewridge` over two half-lives that one stream each
/// serves (their runs go to a sink as they end), and a spec of five groups,
/// whose runs wait for the end of the chunk.
fn thread_specs(name: &str) -> Vec<Spec> {
    let wide: Vec<String> = (0..20).map(|j| format!("x{j}")).collect();
    let wide = serde_json::to_string(&wide).unwrap();
    let ewcov = format!(
        r#"{{"name": "{name}-ewcov", "model": {{"type": "ew_cov"}}, "features": {wide},
            "clock": "t", "gap_cap": 6.0, "weight": "w", "half_life": 50.0}}"#
    );
    let ewridge = format!(
        r#"{{"name": "{name}-ewridge", "model": {{"type": "ewridge", "ridge": 1e-3}},
            "half_life": [25.0, 100.0], "targets": ["y0", "y1", "y2", "y3"],
            "features": ["x0", "x1", "x2", "x3", "x4"], "clock": "t", "gap_cap": 6.0,
            "weight": "w", "emit_sigma": true, "emit_clocks": true}}"#
    );
    let grouped = format!(
        r#"{{"name": "{name}-grouped", "model": {{"type": "ewridge"}},
            "targets": ["y0", "y1"], "features": ["x0", "x1", "x2"], "group": "g",
            "clock": "t", "gap_cap": 6.0, "weight": "w", "half_life": 30.0,
            "emit_sigma": true, "coef_every": 0}}"#
    );
    [ewcov, ewridge, grouped]
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}

/// Twenty features with nulls in them, four targets with nulls, an
/// irregular clock with a gap past the cap, zero weights, and five groups.
fn thread_frame(n: usize) -> DataFrame {
    let mut t = 0.0;
    let clock: Vec<f64> = (0..n)
        .map(|i| {
            t += [0.5, 1.0, 1.5][i % 3] + if i == n / 3 { 40.0 } else { 0.0 };
            t
        })
        .collect();
    let mut cols: Vec<Column> = (0..20)
        .map(|j| {
            let v: Vec<Option<f64>> = (0..n)
                .map(|i| (i % (89 + j) != 3).then(|| ((i * (j + 5)) % 23) as f64 / 7.0 - 1.5))
                .collect();
            Column::new(format!("x{j}").into(), v)
        })
        .collect();
    cols.extend((0..4).map(|j| {
        let v: Vec<Option<f64>> = (0..n)
            .map(|i| (i % 83 != j).then(|| ((i * (7 + j)) % 19) as f64 / 5.0))
            .collect();
        Column::new(format!("y{j}").into(), v)
    }));
    let w: Vec<f64> = (0..n)
        .map(|i| if i % 31 == 0 { 0.0 } else { 1.0 })
        .collect();
    let g: Vec<i64> = (0..n).map(|i| ((i * 7) % 5) as i64).collect();
    cols.push(Column::new("t".into(), clock));
    cols.push(Column::new("w".into(), w));
    cols.push(Column::new("g".into(), g));
    DataFrame::new_infer_height(cols).unwrap()
}

/// The structs a bank of `specs` returns for `df` in `parts` chunks at
/// `threads`, chunk by chunk, its saved state, and the runs each spec's
/// sink took as its stream ran.
fn at_threads(
    specs: &[Spec],
    df: &DataFrame,
    parts: usize,
    threads: Option<usize>,
) -> (Vec<Vec<StructArray>>, Vec<u8>, Vec<usize>) {
    let mut bank = Bank::new(specs.to_vec()).unwrap();
    bank.set_threads(threads).unwrap();
    let before: Vec<usize> = specs.iter().map(|s| runs_absorbed(&s.name)).collect();
    let step = df.height().div_ceil(parts);
    let outs = (0..parts)
        .map(|p| {
            structs(
                &bank
                    .fit_predict(&df.slice((p * step) as i64, step))
                    .unwrap(),
            )
        })
        .collect();
    let runs = specs
        .iter()
        .zip(before)
        .map(|(s, b)| runs_absorbed(&s.name) - b)
        .collect();
    (outs, bank.save_bytes().unwrap(), runs)
}

/// Each of `banks` fed `df` in each of `parts` chunks gives the same
/// structs, chunk by chunk and bit by bit, and the same saved state, at
/// every one of `threads` as on one thread; and each spec that one stream
/// serves took the overlapped way, the wide `ew_cov` at least `min_runs`
/// runs to a chunk, while a grouped spec never did.
fn same_at_every_count(
    banks: &[Vec<Spec>],
    df: &DataFrame,
    parts: &[usize],
    threads: &[Option<usize>],
    min_runs: usize,
) {
    for specs in banks {
        for &parts in parts {
            let what = |t: Option<usize>| {
                let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
                format!("{names:?}, {parts} chunks, threads {t:?}")
            };
            let (want, state, runs) = at_threads(specs, df, parts, Some(1));
            for (s, &r) in specs.iter().zip(&runs) {
                if s.group.is_some() {
                    assert_eq!(r, 0, "{}: {r} runs", what(Some(1)));
                } else {
                    let least = if s.name.ends_with("ewcov") {
                        min_runs
                    } else {
                        1
                    };
                    assert!(
                        r >= least * parts,
                        "{}: {} took {r} runs",
                        what(Some(1)),
                        s.name
                    );
                }
            }
            for &t in threads {
                let (got, saved, got_runs) = at_threads(specs, df, parts, t);
                assert_eq!(got_runs, runs, "{}", what(t));
                for (p, (g, w)) in got.iter().zip(&want).enumerate() {
                    for (x, y) in g.iter().zip(w) {
                        assert_same(x, y, &format!("{}, chunk {p}", what(t)));
                    }
                }
                assert!(saved == state, "{}: the states differ", what(t));
            }
        }
    }
}

/// The rows of one run of the wide `ew_cov`, as the stream splits a chunk.
fn ewcov_run_rows() -> usize {
    let mut spec = thread_specs("sink-on-run-rows").remove(0);
    spec.check().unwrap();
    let stream = crate::Stream::new(&spec).unwrap();
    crate::ChunkOut::run_rows(&spec, stream.n_models(), stream.n_slots())
}

/// The wide `ew_cov` and the grouped spec in one bank, over one chunk of
/// three runs of the `ew_cov`, on one thread and on four: what the test
/// below holds at every pool size, chunking and bank, in its fastest form.
#[test]
fn a_run_at_a_time_is_the_same_on_one_thread_and_four() {
    let df = thread_frame(3 * ewcov_run_rows());
    let mut specs = thread_specs("sink-on-thr-one-four");
    specs.remove(1);
    same_at_every_count(&[specs], &df, &[1], &[Some(4)], 3);
}

/// Every output field and the saved state are the same to the bit at every
/// pool size, the process default's included: on one thread, where
/// `rayon::join` runs the copy of a run and the learning of the next one
/// after the other, and on two to eight, where they overlap. Each spec
/// alone and the three in one bank, over one chunk (past `PAR_MIN_ROWS`, so
/// the sinks allocate and finish their columns in parallel) and over two
/// (below it). The single-stream specs took the overlapped way, the wide
/// `ew_cov` three runs to a chunk; the grouped one never did.
#[test]
#[ignore = "extended: threads, a grid of pool sizes, chunkings and banks"]
fn a_run_at_a_time_is_the_same_at_every_thread_count() {
    let df = thread_frame(6 * ewcov_run_rows());
    assert!(df.height() >= crate::PAR_MIN_ROWS);
    assert!(df.height() / 2 < crate::PAR_MIN_ROWS);
    let all = thread_specs("sink-on-thr-every");
    let banks: Vec<Vec<Spec>> = all
        .iter()
        .map(|s| vec![s.clone()])
        .chain(std::iter::once(all.clone()))
        .collect();
    same_at_every_count(&banks, &df, &[1, 2], &[Some(2), Some(4), Some(8), None], 3);
}
