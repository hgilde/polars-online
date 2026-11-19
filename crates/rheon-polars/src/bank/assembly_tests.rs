//! The two ways a spec's runs reach its output struct (task 255 (a)): one
//! run at a time as its stream runs, for a spec one stream serves, and every
//! run at once after the streams are done. A test-only switch sends the
//! specs named `sink-off-*` the second way, and a counter records the runs
//! the first way takes for the specs named `sink-on-*`, so the other tests
//! of this binary, running beside these, are untouched. The first way's
//! copy of a run overlaps the learning of the next through `rayon::join`,
//! which on a pool of one thread runs the two in turn, so its output is
//! held to the bit at every pool size as well.
//!
//! The two ways read each field from the same place, so a field read from
//! the wrong place reads wrong both ways alike. A spec at its fullest is
//! therefore also held to references that know nothing of its other
//! instances and slots -- the same spec as one half-life, and as one
//! half-life, target and grid point -- and its selected and averaged
//! predictions to their definitions over its own fields (task 255 (a),
//! the mutants that survived it).
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

/// How a frame's rows fall into groups: no group column, each group's rows
/// in one block, or the groups' rows interleaved, so that no run of a group
/// is one unbroken range of the chunk.
#[derive(Clone, Copy, Debug)]
enum Layout {
    Single,
    Blocks,
    Interleaved,
}

/// The features and clock of [`frame`], a target `y` whose relation to `x0`
/// drifts (so the short half-life tracks it better than the long one) and
/// that is null over a block of rows (so its weight falls under its
/// `min_weight` there, for the short half-life only: a slot with an error
/// spread and no prediction), a target `z` with no nulls, and the groups of
/// `layout`.
fn full_frame(n: usize, layout: Layout) -> DataFrame {
    let mut t = 0.0;
    let clock: Vec<f64> = (0..n)
        .map(|i| {
            t += [0.5, 1.0, 1.5][i % 3] + if i == n / 3 { 40.0 } else { 0.0 };
            t
        })
        .collect();
    let xv = |i: usize, b: usize| ((i * b) % 23) as f64 / 7.0 - 1.5;
    let x = |a: usize, b: usize| -> Vec<Option<f64>> {
        (0..n).map(|i| (i % 97 != a).then(|| xv(i, b))).collect()
    };
    let gap = n / 2..n / 2 + 60;
    let y: Vec<Option<f64>> = (0..n)
        .map(|i| {
            let beta = 2.0 * (i as f64 / 120.0).sin();
            let noise = ((i * 7) % 19) as f64 / 19.0 - 0.5;
            (i % 89 != 5 && !gap.contains(&i)).then(|| beta * xv(i, 5) + noise)
        })
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| ((i * 11) % 17) as f64 / 3.0 - 2.5 + 0.5 * xv(i, 3))
        .collect();
    let w: Vec<f64> = (0..n)
        .map(|i| if i % 31 == 0 { 0.0 } else { 1.0 })
        .collect();
    let mut df = df!("t" => clock, "x0" => x(3, 5), "x1" => x(50, 3), "x2" => x(98, 13),
        "y" => y, "z" => z, "w" => w)
    .unwrap();
    let g: Option<Vec<i64>> = match layout {
        Layout::Single => None,
        Layout::Blocks => Some((0..n).map(|i| (i * 3 / n) as i64).collect()),
        Layout::Interleaved => Some((0..n).map(|i| ((i * 7) % 3) as i64).collect()),
    };
    if let Some(g) = g {
        df.with_column(Column::new("g".into(), g)).unwrap();
    }
    df
}

/// A spec at its fullest: every per-row output `model` can emit, over the
/// two targets of [`full_frame`] (a `min_weight` that `y` falls under in
/// its block of nulls), with `extra` over the top (the half-lives, the
/// grid, the targets, the group).
fn full_spec(name: &str, model: &serde_json::Value, extra: &serde_json::Value) -> Spec {
    let mut v = serde_json::json!({
        "name": name, "model": model, "targets": ["y", "z"],
        "features": ["x0", "x1", "x2"], "clock": "t", "gap_cap": 6.0, "weight": "w",
        "min_weight": 25.0, "emit_clocks": true, "coef_every": 0,
        "emit_sigma": true, "emit_zscore": true, "emit_metrics": true, "conformal": 0.9,
        "resid_quantiles": [0.1, 0.25, 0.75, 0.9], "emit_autocorr": true,
        "emit_drift": true, "drift_threshold": 20.0, "emit_calibration": true,
        "emit_breaks": true, "emit_specification": true, "emit_tails": true,
        "emit_feature_health": true,
    });
    if model["type"] == "ewridge" {
        for k in [
            "emit_error_inflation",
            "emit_se_coef",
            "emit_robust_se",
            "emit_influence",
        ] {
            v[k] = true.into();
        }
    }
    for (k, x) in extra.as_object().unwrap() {
        v[k] = x.clone();
    }
    serde_json::from_value(v).unwrap()
}

/// One field's values equal another's: the same validity, and for a float
/// the same bits in every slot, a null's included.
fn same_field(x: &dyn Array, y: &dyn Array) -> bool {
    if x.len() != y.len() || x.validity() != y.validity() {
        return false;
    }
    match (
        x.as_any().downcast_ref::<Float64Array>(),
        y.as_any().downcast_ref::<Float64Array>(),
    ) {
        (Some(x), Some(y)) => x
            .values()
            .iter()
            .zip(y.values().iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        _ => x == y,
    }
}

/// A float field's values, null as `None`.
fn floats(a: &dyn Array) -> Vec<Option<f64>> {
    let a = a.as_any().downcast_ref::<Float64Array>().unwrap();
    a.iter().map(|v| v.copied()).collect()
}

/// The spec a per-instance field of a full spec is held to: the same spec
/// run as that one half-life (`target` and `ridge` `None`), or as that one
/// half-life and target (and, for `ewridge`, ridge).
struct Reference {
    spec: Spec,
    half_life: f64,
    target: Option<String>,
    ridge: Option<f64>,
}

/// The references of a full spec of `model` over `grid` (a ridge grid for
/// `ewridge`, whose grid points are separate fits of one Gram; a penalty
/// path for `lasso`, whose points are warm-started one from the last, so
/// its references keep the whole path): each half-life alone, and each
/// half-life and target (and ridge) alone.
fn references(
    name: &str,
    model: &serde_json::Value,
    grid: &[f64],
    half_lives: &[f64],
    extra: &serde_json::Value,
) -> Vec<Reference> {
    use serde_json::json;
    let ewridge = model["type"] == "ewridge";
    let mut out = Vec::new();
    for &h in half_lives {
        let mut e = extra.clone();
        e["half_life"] = json!(h);
        out.push(Reference {
            spec: full_spec(&format!("{name}-h{h}"), model, &e),
            half_life: h,
            target: None,
            ridge: None,
        });
        for t in ["y", "z"] {
            let mut e = e.clone();
            e["targets"] = json!([t]);
            let ridges: Vec<Option<f64>> = if ewridge {
                grid.iter().copied().map(Some).collect()
            } else {
                vec![None]
            };
            for r in ridges {
                let mut m = model.clone();
                if let Some(r) = r {
                    m["ridge"] = json!(r);
                }
                out.push(Reference {
                    spec: full_spec(&format!("{name}-h{h}-{t}-r{r:?}"), &m, &e),
                    half_life: h,
                    target: Some(t.to_string()),
                    ridge: r,
                });
            }
        }
    }
    out
}

/// The output metadata of `spec`, field by field in struct order.
fn meta_of(spec: &Spec) -> Vec<crate::FieldMeta> {
    let mut s = spec.clone();
    s.check().unwrap();
    crate::output_index(&s)
}

/// What [`a_full_spec_is_its_parts`] counted: fields held to a reference,
/// and the rows where the averaged prediction's two rules part.
#[derive(Default, Debug)]
struct Held {
    fields: usize,
    /// A slot with an error spread and no prediction, below the best slot
    /// with both, beside at least two with both: the averaged prediction
    /// takes the best only over the slots that have both.
    spread_without_pred: usize,
    /// A slot of `z` whose prediction is there where `y`'s at the same
    /// instance and grid point is not, or the other way round.
    targets_part: usize,
}

/// A target's averaged prediction, selected prediction and selected slot,
/// row by row.
type Derived = (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<u32>>);

/// `emit_averaged` and `emit_selected` for target `t` of `full`, from its
/// own `pred` and `sigma` fields, by their definitions: each slot with both
/// weighs `exp(-eta * (sigma² / best - 1))`, `best` the least `sigma²` of
/// such slots; the selected slot is the first of least `sigma`, and gives
/// its prediction where it has one.
fn derived_of(
    meta: &[crate::FieldMeta],
    full: &StructArray,
    t: &str,
    eta: f64,
    held: &mut Held,
) -> Derived {
    let field = |kind: &str, t: &str| -> Vec<Vec<Option<f64>>> {
        meta.iter()
            .enumerate()
            .filter(|(_, f)| f.kind == kind && f.target.as_deref() == Some(t))
            .map(|(i, _)| floats(full.values()[i].as_ref()))
            .collect()
    };
    let (pred, sigma) = (field("pred", t), field("sigma", t));
    let other = field("pred", if t == "y" { "z" } else { "y" });
    assert!(pred.len() >= 4 && pred.len() == sigma.len());
    let n = full.len();
    let (mut avg, mut sel, mut name) = (vec![None; n], vec![None; n], vec![None; n]);
    for i in 0..n {
        let both: Vec<(f64, f64)> = (0..pred.len())
            .filter_map(|s| Some((sigma[s][i]?, pred[s][i]?)))
            .collect();
        let best = both
            .iter()
            .map(|(s, _)| s * s)
            .fold(f64::INFINITY, f64::min);
        if both.len() >= 2
            && (0..pred.len())
                .any(|s| pred[s][i].is_none() && sigma[s][i].is_some_and(|x| x * x < best))
        {
            held.spread_without_pred += 1;
        }
        if t == "z"
            && (0..pred.len())
                .any(|s| sigma[s][i].is_some() && pred[s][i].is_some() != other[s][i].is_some())
        {
            held.targets_part += 1;
        }
        if best.is_finite() {
            let (mut num, mut den) = (0.0, 0.0);
            for &(s, p) in &both {
                let excess = if best > 0.0 {
                    s * s / best - 1.0
                } else if s * s == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                };
                let wgt = if excess > 0.0 {
                    (-eta * excess).exp()
                } else {
                    1.0
                };
                num += wgt * p;
                den += wgt;
            }
            if den > 0.0 {
                avg[i] = Some(num / den);
            }
        }
        let mut pick: Option<(f64, usize)> = None;
        for (s, sig) in sigma.iter().enumerate() {
            if let Some(x) = sig[i]
                && pick.is_none_or(|(b, _)| x < b)
            {
                pick = Some((x, s));
            }
        }
        if let Some((_, s)) = pick
            && let Some(p) = pred[s][i]
        {
            sel[i] = Some(p);
            name[i] = Some(s as u32);
        }
    }
    (avg, sel, name)
}

/// Every field of `full` (a spec of several half-lives, two targets and a
/// grid) against its references, in one chunk's output: each per-instance
/// field is the same field of the spec run as that half-life alone, and
/// each per-slot one the same field of the spec run as that half-life,
/// target and grid point alone, to the bit; no two of its float fields are
/// the same column, so a field read from another's place cannot pass; and
/// the selected and averaged predictions are what their definitions make of
/// its own `pred` and `sigma` fields.
fn hold_to_references(
    full_spec: &Spec,
    full: &StructArray,
    refs: &[(&Reference, &StructArray)],
    what: &str,
    held: &mut Held,
) {
    let meta = meta_of(full_spec);
    assert_eq!(meta.len(), full.values().len(), "{what}");
    let ref_meta: Vec<Vec<crate::FieldMeta>> = refs.iter().map(|(r, _)| meta_of(&r.spec)).collect();
    for (i, f) in meta.iter().enumerate() {
        if f.instance.is_none() {
            continue;
        }
        let (ri, (r, out)) = refs
            .iter()
            .enumerate()
            .find(|(_, (r, _))| {
                f.half_life == Some(r.half_life)
                    && r.target == f.target
                    && (r.ridge.is_none() || r.ridge == f.ridge)
            })
            .unwrap_or_else(|| panic!("{what}: no reference for {}", f.field));
        let hit = ref_meta[ri]
            .iter()
            .position(|g| {
                (
                    &g.kind, &g.target, g.quantile, &g.columns, g.ridge, g.penalty,
                ) == (
                    &f.kind, &f.target, f.quantile, &f.columns, f.ridge, f.penalty,
                )
            })
            .unwrap_or_else(|| panic!("{what}: {} not in {}", f.field, r.spec.name));
        assert!(
            same_field(full.values()[i].as_ref(), out.values()[hit].as_ref()),
            "{what}: {} is not {}'s {}",
            f.field,
            r.spec.name,
            ref_meta[ri][hit].field
        );
        held.fields += 1;
    }
    // Every float field is a column of its own.
    let float: Vec<usize> = (0..meta.len())
        .filter(|&i| full.values()[i].as_any().is::<Float64Array>())
        .collect();
    // Except `break_wald` across a grid: the distance between two
    // least-squares fits of the target on the features, which no grid
    // point enters.
    let twins = |f: &crate::FieldMeta, g: &crate::FieldMeta| {
        f.kind == "break_wald"
            && g.kind == f.kind
            && (&f.target, &f.instance) == (&g.target, &g.instance)
    };
    // And two `coverage` columns whose slots' intervals differ: the share
    // of rows each interval held is the same where the two slots hit and
    // miss on the same rows, as two path points of nearly one fit do once
    // their radii start from the same spread (task 278, batch 17: `lasso`'s
    // 0.05 and 0 over 600 rows). Their `lo` columns tell a buffer read
    // twice from a coincidence.
    let lo_of = |f: &crate::FieldMeta| {
        let name = f.field.replacen("coverage_", "lo_", 1);
        meta.iter().position(|g| g.field == name)
    };
    let coincide = |i: usize, j: usize| {
        meta[i].kind == "coverage"
            && meta[j].kind == "coverage"
            && match (lo_of(&meta[i]), lo_of(&meta[j])) {
                (Some(a), Some(b)) => {
                    !same_field(full.values()[a].as_ref(), full.values()[b].as_ref())
                }
                _ => false,
            }
    };
    let mut dup = vec![];
    for (a, &i) in float.iter().enumerate() {
        for &j in &float[a + 1..] {
            if !twins(&meta[i], &meta[j])
                && !coincide(i, j)
                && same_field(full.values()[i].as_ref(), full.values()[j].as_ref())
            {
                dup.push(format!(
                    "{} == {} ({} nulls)",
                    meta[i].field,
                    meta[j].field,
                    full.values()[i].null_count()
                ));
            }
        }
    }
    assert!(dup.is_empty(), "{what}: the same column: {dup:#?}");
    // The selected and averaged predictions, by their definitions.
    let eta = full_spec.average_eta_or_default();
    for t in ["y", "z"] {
        let (avg, sel, name) = derived_of(&meta, full, t, eta, held);
        let at = |field: &str| {
            let i = meta.iter().position(|f| f.field == field).unwrap();
            full.values()[i].as_ref()
        };
        let bits = |v: &[Option<f64>]| v.iter().map(|x| x.map(f64::to_bits)).collect::<Vec<_>>();
        assert_eq!(
            bits(&floats(at(&format!("pred_{t}__averaged")))),
            bits(&avg),
            "{what}: {t}"
        );
        assert_eq!(
            bits(&floats(at(&format!("pred_{t}__selected")))),
            bits(&sel),
            "{what}: {t}"
        );
        // An `Enum` over the slots' labels, in slot order, as its codes.
        let keys = at(&format!("selected_{t}"))
            .as_any()
            .downcast_ref::<polars_arrow::array::PrimitiveArray<u8>>()
            .unwrap()
            .iter()
            .map(|k| k.map(|&k| u32::from(k)))
            .collect::<Vec<_>>();
        assert_eq!(keys, name, "{what}: selected_{t}");
    }
}

/// The full specs of the test below and their references, under `layout`:
/// `ewridge` over a ridge grid and `lasso` over a penalty path, each over two
/// half-lives, every per-row output on, the selected and averaged
/// predictions too; each full spec twice, under a `sink-on-` name (a spec one
/// stream serves takes its runs one at a time) and under a `sink-off-` one
/// (every run at once).
fn full_banks(tag: &str, layout: Layout) -> Vec<(Spec, Spec, Vec<Reference>)> {
    use serde_json::json;
    let half_lives = [25.0, 100.0];
    let mut extra = json!({"half_life": half_lives});
    if !matches!(layout, Layout::Single) {
        // A group's rows come a third as often, and its weight tops out at a
        // third: about 12 at the short half-life, interleaved.
        extra["group"] = json!("g");
        extra["min_weight"] = json!(8.0);
    }
    let mut full = extra.clone();
    full["emit_averaged"] = json!(true);
    full["emit_selected"] = json!(true);
    let ridge = [1e-6, 3.0, 30.0];
    [
        (
            json!({"type": "ewridge", "ridge": ridge}),
            ridge.to_vec(),
            "ewridge",
        ),
        (
            json!({"type": "lasso", "penalties": [0.2, 0.05, 0.0]}),
            vec![],
            "lasso",
        ),
    ]
    .into_iter()
    .map(|(model, grid, kind)| {
        let name = format!("{tag}-{kind}");
        (
            full_spec(&format!("sink-on-{name}"), &model, &full),
            full_spec(&format!("sink-off-{name}"), &model, &full),
            references(&format!("ref-{name}"), &model, &grid, &half_lives, &extra),
        )
    })
    .collect()
}

/// A spec at its fullest -- several half-lives, two targets, a grid, every
/// per-instance output on -- is its parts: each of its per-instance fields
/// is the field of the same spec run as that half-life alone, and each of
/// its per-slot fields the field of the spec run as that half-life, target
/// and grid point alone, to the bit, null slots included. The run-at-a-time
/// way and the every-run-at-once way give the same output; but a field read
/// from the wrong place reads wrong both ways alike, and only a reference
/// that knows nothing of the other instances and slots tells it.
fn a_full_spec_is_its_parts(n: usize, layout: Layout, parts: &[usize]) -> Held {
    let df = full_frame(n, layout);
    let mut held = Held::default();
    for &p in parts {
        let tag = format!("full-{layout:?}-{p}").to_lowercase();
        let sets = full_banks(&tag, layout);
        let mut specs = Vec::new();
        for (on, off, refs) in &sets {
            specs.push(on.clone());
            specs.push(off.clone());
            specs.extend(refs.iter().map(|r| r.spec.clone()));
        }
        let names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
        let mut bank = Bank::new(specs).unwrap();
        let step = df.height().div_ceil(p);
        for c in 0..p {
            let out = structs(
                &bank
                    .fit_predict(&df.slice((c * step) as i64, step))
                    .unwrap(),
            );
            let by = |name: &str| &out[names.iter().position(|x| x == name).unwrap()];
            for (on, off, refs) in &sets {
                let what = format!("{}, {p} chunks, chunk {c}", on.name);
                assert_same(by(&on.name), by(&off.name), &what);
                let refs: Vec<(&Reference, &StructArray)> =
                    refs.iter().map(|r| (r, by(&r.spec.name))).collect();
                hold_to_references(on, by(&on.name), &refs, &what, &mut held);
            }
        }
        for (on, off, _) in &sets {
            // One stream serves a chunk of a single group: in blocks, each of
            // three chunks is one group's block.
            let runs = runs_absorbed(&on.name);
            match (layout, p) {
                (Layout::Single, _) | (Layout::Blocks, 3) => {
                    assert!(runs >= p, "{}: {runs} runs", on.name);
                }
                _ => assert_eq!(runs, 0, "{}", on.name),
            }
            assert_eq!(runs_absorbed(&off.name), 0, "{}", off.name);
        }
    }
    held
}

/// [`a_full_spec_is_its_parts`] for a spec one stream serves, in two
/// chunks, its runs taken one at a time: every field is where its
/// references put it, and the averaged prediction's two rules -- the best
/// slot is of those with a prediction, the slots are each target's own --
/// each part somewhere in it.
#[test]
fn a_full_spec_one_stream_serves_is_its_parts() {
    let held = a_full_spec_is_its_parts(1500, Layout::Single, &[2]);
    assert!(held.fields > 1000, "{held:?}");
    assert!(held.spread_without_pred > 0, "{held:?}");
    assert!(held.targets_part > 0, "{held:?}");
}

/// [`a_full_spec_is_its_parts`] for a spec of three groups, whose runs wait
/// for the end of a chunk of several: each group's rows in a block, so each
/// run is one unbroken range of the chunk (and in three chunks each chunk is
/// one group's, whose runs go to the sink one at a time), and the groups
/// interleaved, so none is; over one chunk and three.
#[test]
#[ignore = "extended: grouped full specs over two layouts and two chunkings"]
fn a_full_spec_of_groups_is_its_parts() {
    for layout in [Layout::Blocks, Layout::Interleaved] {
        let held = a_full_spec_is_its_parts(1800, layout, &[1, 3]);
        assert!(held.fields > 1000, "{layout:?}: {held:?}");
    }
}

/// The selected and averaged predictions at their edges, held to their
/// definitions over the spec's own `pred` and `sigma` fields: a `lasso`
/// path whose two largest penalties both shrink every coefficient to 0, so
/// their slots predict the same and tie on `sigma` (the first of the tied
/// slots is the one selected), and `average_eta = "inf"`, the argmin, at
/// which the best slot still weighs 1 and every other none.
#[test]
fn the_selected_and_averaged_predictions_at_ties_and_an_infinite_eta() {
    use serde_json::json;
    let df = full_frame(900, Layout::Single);
    let model = json!({"type": "lasso", "penalties": [50.0, 20.0, 0.0]});
    let extra = json!({"half_life": [25.0, 100.0], "emit_averaged": true,
        "emit_selected": true, "average_eta": "inf"});
    let spec = full_spec("sink-on-ties", &model, &extra);
    let meta = meta_of(&spec);
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    let out = structs(&bank.fit_predict(&df).unwrap()).remove(0);
    let mut held = Held::default();
    let eta = spec.average_eta_or_default();
    assert_eq!(eta, f64::INFINITY);
    let (mut ties, mut averaged) = (0, 0);
    for t in ["y", "z"] {
        let (avg, sel, name) = derived_of(&meta, &out, t, eta, &mut held);
        let at = |field: &str| {
            let i = meta.iter().position(|f| f.field == field).unwrap();
            out.values()[i].as_ref()
        };
        let bits = |v: &[Option<f64>]| v.iter().map(|x| x.map(f64::to_bits)).collect::<Vec<_>>();
        assert_eq!(
            bits(&floats(at(&format!("pred_{t}__averaged")))),
            bits(&avg),
            "{t}"
        );
        assert_eq!(
            bits(&floats(at(&format!("pred_{t}__selected")))),
            bits(&sel),
            "{t}"
        );
        let keys = at(&format!("selected_{t}"))
            .as_any()
            .downcast_ref::<polars_arrow::array::PrimitiveArray<u8>>()
            .unwrap()
            .iter()
            .map(|k| k.map(|&k| u32::from(k)))
            .collect::<Vec<_>>();
        assert_eq!(keys, name, "selected_{t}");
        // The rows whose least `sigma` two slots share.
        let sigma: Vec<Vec<Option<f64>>> = meta
            .iter()
            .enumerate()
            .filter(|(_, f)| f.kind == "sigma" && f.target.as_deref() == Some(t))
            .map(|(i, _)| floats(out.values()[i].as_ref()))
            .collect();
        for i in 0..out.len() {
            let row: Vec<f64> = sigma.iter().filter_map(|s| s[i]).collect();
            let least = row.iter().copied().fold(f64::INFINITY, f64::min);
            ties += usize::from(row.iter().filter(|&&s| s == least).count() >= 2);
        }
        averaged += avg.iter().flatten().count();
    }
    assert!(ties > 100, "{ties} rows tie");
    assert!(averaged > 100, "{averaged} rows averaged");
}
