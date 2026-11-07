//! `label_clock` (docs/PLAN.md task 341; docs/records/TREE-FEATURES.md
//! §2.4): each row scored where it sits, held until the first row whose
//! clock reaches its label time, and learned at that label time with the
//! prediction it was served scored against its label.
//!
//! The oracles, each the definition rather than the code:
//!
//! - `embargo = d` is the case `label_clock = clock + d`: every output of
//!   every kind `label_clock` takes is the same to the bit, on an integer
//!   clock (whose steps add up exactly) with skipped rows, capped gaps and a
//!   clock reset.
//! - The doubled stream: each row fed twice, at its clock with weight 0 to be
//!   scored and at its label time to be learned, in time order with a label
//!   before a row at the same time. It learns the same labels at the same
//!   clocks, so its scored copies' predictions are the served ones to the
//!   bit, wherever its extra steps decay exactly: no decay, or a half-life
//!   of one clock unit on whole steps (each factor a power of two), with no
//!   gap capped.
//! - Rule 3: any chunking, and a save and load while rows are held, give the
//!   same output to the bit.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// A seeded generator in `[0, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn spec_of(text: &str) -> Spec {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// The kinds `label_clock` takes, each as `(name, model, the rest)`. The
/// spec adds its clock and the delay. `rcov` is not among them: it needs
/// `group_close`, which neither delay works beside.
const KINDS: &[(&str, &str, &str)] = &[
    (
        "bocpd",
        r#"{"type": "bocpd", "hazard": 250.0, "covariance_type": "diag"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"]"#,
    ),
    (
        "corrchange",
        r#"{"type": "corrchange", "kind": "monitor", "window_rows": 20, "alpha": 0.05}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"]"#,
    ),
    (
        "deco",
        r#"{"type": "deco", "dynamics": "ew"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 50.0"#,
    ),
    (
        "ew_class",
        r#"{"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0}"#,
        r#""targets": ["lab"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "ew_cov",
        r#"{"type": "ew_cov"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 50.0"#,
    ),
    (
        "ew_cov_windowed",
        r#"{"type": "ew_cov", "window_size": 30.0}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 20.0"#,
    ),
    (
        "ewridge",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": [20.0, 50.0], "weight": "w", "min_weight": 3.0, "conformal": 0.9, "emit_sigma": true, "emit_zscore": true, "emit_metrics": true, "emit_drift": true, "drift_threshold": 20.0, "emit_calibration": true, "emit_autocorr": true, "resid_quantiles": [0.5, 0.9], "emit_clocks": true, "emit_selected": true, "emit_breaks": true, "emit_tails": true, "emit_influence": true"#,
    ),
    (
        "ewridge_windowed",
        r#"{"type": "ewridge", "window_size": 25.0}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0, "min_weight": 2.0"#,
    ),
    (
        "ftrl",
        r#"{"type": "ftrl", "strict_binary": false, "loss": "logistic"}"#,
        r#""targets": ["yb"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "huber",
        r#"{"type": "huber", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "kmeans",
        r#"{"type": "kmeans", "n_clusters": 2}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 50.0"#,
    ),
    (
        "lasso",
        r#"{"type": "lasso", "penalties": [0.1, 0.0]}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "marginal",
        r#"{"type": "marginal"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "pa",
        r#"{"type": "pa", "mode": "pa1"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "quantile",
        r#"{"type": "quantile", "quantile": 0.5, "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "rls",
        r#"{"type": "rls"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "seqtest",
        r#"{"type": "seqtest"}"#,
        r#""targets": ["y"], "features": []"#,
    ),
    (
        "sgd",
        r#"{"type": "sgd", "loss": "squared", "learning_rate": 0.01, "schedule": "constant", "standardize": true}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "sgd_huber",
        r#"{"type": "sgd", "loss": "huber", "huber_delta": 1.5, "learning_rate": 0.01, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "sgd_logistic",
        r#"{"type": "sgd", "loss": "logistic", "learning_rate": 0.01, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["yb"], "features": ["x0"], "half_life": 50.0"#,
    ),
];

/// One kind's spec named `m`, on the integer clock `t` with a `gap_cap` of
/// 10 and a step back past 500 a restart, and `delay` (an `embargo` or a
/// `label_clock`).
fn kind_spec(model: &str, rest: &str, delay: &str) -> Spec {
    spec_of(&format!(
        r#"{{"name": "m", "model": {model}, {rest}, "clock": "t", "gap_cap": 10.0,
            "restart_after_step_back": 500.0, {delay}}}"#
    ))
}

/// `n` rows on an integer clock `t` of whole steps from 0 to 7, every
/// twenty-third a gap of 40 past the cap, and, where `reset_at` says, a
/// step back of 1,000 that restarts the models; `x1` null on every
/// eleventh row, `x0` on every seventeenth, so rows are skipped; a weight
/// in `[0.5, 1.5)`; and the label times `avail = t + delay`, the delay
/// `delays(i)`.
fn frame(
    n: usize,
    seed: u64,
    reset_at: Option<usize>,
    delays: &dyn Fn(usize) -> Option<i64>,
) -> DataFrame {
    let mut r = lcg(seed);
    let mut t = 0i64;
    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        if i > 0 {
            t += if Some(i) == reset_at {
                -1_000
            } else if i % 23 == 0 {
                40
            } else {
                [0, 1, 1, 2, 3, 7][(r() * 6.0) as usize]
            };
        }
        ts.push(t);
    }
    let x0: Vec<Option<f64>> = (0..n)
        .map(|i| (i % 17 != 5).then(|| r() * 2.0 - 1.0))
        .collect();
    let x1: Vec<Option<f64>> = (0..n)
        .map(|i| (i % 11 != 3).then(|| r() * 2.0 - 1.0))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.5 + 2.0 * x0[i].unwrap_or(0.0) + r() - 0.5)
        .collect();
    let yb: Vec<f64> = y.iter().map(|&v| f64::from(u8::from(v > 0.5))).collect();
    let lab: Vec<&str> = y.iter().map(|&v| if v > 0.5 { "a" } else { "b" }).collect();
    let w: Vec<f64> = (0..n).map(|_| 0.5 + r()).collect();
    let avail: Vec<Option<i64>> = (0..n).map(|i| delays(i).map(|d| ts[i] + d)).collect();
    df!("t" => ts, "x0" => x0, "x1" => x1, "y" => y, "yb" => yb, "lab" => lab, "w" => w,
        "avail" => avail)
    .unwrap()
}

/// `df` through `bank` in chunks of `size` rows (all at once for 0), as the
/// bank's output columns.
fn feed(bank: &mut Bank, df: &DataFrame, size: usize) -> DataFrame {
    let size = if size == 0 { df.height() } else { size };
    let mut out: Option<DataFrame> = None;
    let mut start = 0;
    while start < df.height() {
        let piece = df.slice(start as i64, size);
        let got = DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap()).unwrap();
        match out.as_mut() {
            None => out = Some(got),
            Some(o) => {
                o.vstack_mut(&got).unwrap();
            }
        }
        start += size;
    }
    out.unwrap()
}

fn run(spec: &Spec, df: &DataFrame, size: usize) -> DataFrame {
    feed(&mut Bank::new(vec![spec.clone()]).unwrap(), df, size)
}

/// The frame's bytes as IPC, after one rechunk: equal bytes are equal
/// values to the bit, NaN payloads and nulls included.
fn ipc(df: &DataFrame) -> Vec<u8> {
    let mut df = df.clone();
    df.rechunk_mut();
    let mut out = Vec::new();
    IpcWriter::new(&mut out).finish(&mut df).unwrap();
    out
}

/// The first field whose values differ, for the message.
fn first_difference(a: &DataFrame, b: &DataFrame) -> String {
    let (sa, sb) = (
        a.column("m").unwrap().struct_().unwrap().fields_as_series(),
        b.column("m").unwrap().struct_().unwrap().fields_as_series(),
    );
    for (x, y) in sa.iter().zip(&sb) {
        if !x.equals_missing(y) {
            return format!("{}: {:?} vs {:?}", x.name(), x, y);
        }
    }
    "none by value (bits differ)".into()
}

fn assert_same(a: &DataFrame, b: &DataFrame, what: &str) {
    assert!(ipc(a) == ipc(b), "{what}: {}", first_difference(a, b));
}

#[test]
fn an_embargo_is_a_label_time_of_the_clock_plus_the_delay_for_every_kind() {
    for reset in [None, Some(260)] {
        let df = frame(400, 7, reset, &|_| Some(6));
        for (name, model, rest) in KINDS {
            let embargo = run(&kind_spec(model, rest, r#""embargo": 6.0"#), &df, 0);
            let label = kind_spec(model, rest, r#""label_clock": "avail""#);
            assert_same(
                &embargo,
                &run(&label, &df, 0),
                &format!("{name}, reset {reset:?}"),
            );
            // And under a chunking whose boundaries fall between rows and
            // their releases (the same chunking for both: without a cadence,
            // `coef` is each chunk's last row's).
            let embargo = run(&kind_spec(model, rest, r#""embargo": 6.0"#), &df, 13);
            assert_same(
                &embargo,
                &run(&label, &df, 13),
                &format!("{name} in chunks of 13"),
            );
        }
    }
}

#[test]
fn a_list_of_one_column_per_target_reads_as_the_column() {
    let df = frame(200, 3, None, &|i| Some((i % 9) as i64));
    let one = kind_spec(
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y", "yb"], "features": ["x0"], "half_life": 30.0"#,
        r#""label_clock": "avail""#,
    );
    let list = kind_spec(
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y", "yb"], "features": ["x0"], "half_life": 30.0"#,
        r#""label_clock": ["avail", "avail"]"#,
    );
    assert_same(&run(&one, &df, 0), &run(&list, &df, 0), "the list");
}

/// The delays the label-time tests use: 0 to 24 clock units, so a short
/// delay often overtakes a long one, and every thirteenth label null.
fn varied(i: usize) -> Option<i64> {
    (i % 13 != 4).then(|| ((i * 7919) % 25) as i64)
}

/// `ewridge` on `x0` and `x1` with a label time per row, on the integer
/// clock `t`, a gap cap past every gap, a solve every row and no floor, and
/// the spec-level settings `rest`.
fn varied_spec(rest: &str) -> Spec {
    spec_of(&format!(
        r#"{{"name": "m", "model": {{"type": "ewridge", "max_rows_between_solves": 1}},
            "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 1e9,
            "min_weight": 0.0, "weight": "w", "restart_after_step_back": 500.0, {rest}}}"#
    ))
}

/// The doubled stream of `df`: a copy of each row at `t` with weight 0, to
/// be scored, and a copy at its label time `avail` with its own weight, to
/// be learned, for a row with a label time and a weight above 0; in order
/// of time, and at one time in order of arrival, a row's learned copy after
/// its own scored one: a row is scored before its label can be learned, and
/// a label at a row's clock is learned before that row if it arrived
/// earlier. Returns the frame and, for each scored copy, its row.
fn doubled(df: &DataFrame) -> (DataFrame, Vec<usize>) {
    let t: Vec<i64> = df
        .column("t")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let avail: Vec<Option<i64>> = df.column("avail").unwrap().i64().unwrap().iter().collect();
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    // (time, row, 1 for a scored copy and 2 for a learned one)
    let mut events: Vec<(i64, usize, u8)> = (0..df.height()).map(|i| (t[i], i, 1)).collect();
    for i in 0..df.height() {
        if let Some(a) = avail[i].filter(|_| w[i] > 0.0) {
            events.push((a, i, 2));
        }
    }
    events.sort();
    let idx: Vec<IdxSize> = events.iter().map(|e| e.1 as IdxSize).collect();
    let mut out = df.take(&IdxCa::from_vec("i".into(), idx)).unwrap();
    let times: Vec<i64> = events.iter().map(|e| e.0).collect();
    let weights: Vec<f64> = events
        .iter()
        .map(|&(_, i, role)| if role == 1 { 0.0 } else { w[i] })
        .collect();
    out.with_column(Series::new("t".into(), times).into_column())
        .unwrap();
    out.with_column(Series::new("w".into(), weights).into_column())
        .unwrap();
    let scored = events.iter().filter(|e| e.2 == 1).map(|e| e.1).collect();
    (out, scored)
}

fn preds(out: &DataFrame) -> Vec<u64> {
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("pred_y")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(|v| v.map_or(u64::MAX, f64::to_bits))
        .collect()
}

#[test]
fn the_served_predictions_are_the_doubled_streams_scored_copies() {
    for half_life in ["\"inf\"", "1.0"] {
        let df = frame(500, 11, None, &varied);
        // The noise gate off: on a half-life of one unit it withholds most
        // rows, and the oracle is for the predictions.
        let rest = format!(r#""half_life": {half_life}, "max_error_inflation": "inf""#);
        let native = run(
            &varied_spec(&format!(r#"{rest}, "label_clock": "avail""#)),
            &df,
            0,
        );
        let (twice, scored) = doubled(&df);
        let plain = varied_spec(&rest);
        let oracle = run(&plain, &twice, 0);
        let p = preds(&oracle);
        // The scored copies, back in row order.
        let mut want = vec![0u64; df.height()];
        let mut k = 0;
        for (pos, ev) in (0..twice.height()).zip(twice.column("w").unwrap().f64().unwrap().iter()) {
            if ev == Some(0.0) {
                want[scored[k]] = p[pos];
                k += 1;
            }
        }
        let got = preds(&native);
        let served = got.iter().filter(|&&b| b != u64::MAX).count();
        assert!(
            served > 400,
            "half_life {half_life}: {served} rows served a prediction"
        );
        let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        assert!(bad.is_empty(), "half_life {half_life}: rows {bad:?}");
    }
}

#[test]
fn a_short_delay_overtakes_a_long_one_and_is_learned_first() {
    // Row 0's label arrives at 10, row 1's at 2: at row 2 (clock 2) only row
    // 1 is learned, so the weight is row 1's alone.
    let df = df!("t" => [0i64, 1, 2, 3, 10, 11], "x0" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 "x1" => [0.5, 0.1, 0.3, 0.2, 0.9, 0.4], "y" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 "w" => [1.0, 2.0, 4.0, 8.0, 16.0, 32.0], "avail" => [10i64, 2, 20, 20, 20, 20])
    .unwrap();
    let out = run(
        &varied_spec(r#""half_life": "inf", "label_clock": "avail""#),
        &df,
        0,
    );
    let ws: Vec<Option<f64>> = out
        .column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("weight_sum")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(
        ws,
        [
            Some(0.0),
            Some(0.0),
            Some(2.0),
            Some(2.0),
            Some(3.0),
            Some(3.0)
        ]
    );
}

#[test]
fn any_chunking_and_a_save_while_rows_are_held_give_the_same_output() {
    let df = frame(400, 5, Some(300), &varied);
    let spec = varied_spec(
        r#""half_life": 30.0, "label_clock": "avail", "conformal": 0.9, "emit_metrics": true, "emit_sigma": true, "emit_drift": true, "drift_threshold": 20.0, "emit_calibration": true, "max_held_rows": 12, "held_overflow": "drop_oldest", "coef_every": 0"#,
    );
    let whole = run(&spec, &df, 0);
    let mut r = lcg(99);
    for size in [1, 2, 7, 50] {
        assert_same(&whole, &run(&spec, &df, size), &format!("chunks of {size}"));
    }
    // Random chunkings.
    for _ in 0..5 {
        let mut bank = Bank::new(vec![spec.clone()]).unwrap();
        let mut parts = Vec::new();
        let mut at = 0;
        while at < df.height() {
            let n = 1 + (r() * 40.0) as usize;
            let piece = df.slice(at as i64, n);
            parts.push(DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap()).unwrap());
            at += n;
        }
        let mut got = parts[0].clone();
        for p in &parts[1..] {
            got.vstack_mut(p).unwrap();
        }
        assert_same(&whole, &got, "a random chunking");
    }
    // A save and load at every 37th row, rows held across each.
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    let mut got: Option<DataFrame> = None;
    let mut at = 0;
    while at < df.height() {
        let piece = df.slice(at as i64, 37);
        let out = DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap()).unwrap();
        match got.as_mut() {
            None => got = Some(out),
            Some(g) => {
                g.vstack_mut(&out).unwrap();
            }
        }
        let bytes = bank.save_bytes().unwrap();
        bank = Bank::load_bytes(&bytes, Some(std::slice::from_ref(&spec))).unwrap();
        assert_eq!(
            bank.save_bytes().unwrap(),
            bytes,
            "a loaded bank re-saves its bytes"
        );
        at += 37;
    }
    assert_same(&whole, &got.unwrap(), "saved and loaded every 37 rows");
}

/// `summary()`'s count `name` for the one stream.
fn summary_count(bank: &Bank, name: &str) -> i64 {
    bank.summary(0, None)
        .unwrap()
        .column(name)
        .unwrap()
        .i64()
        .unwrap()
        .get(0)
        .unwrap()
}

#[test]
fn missing_and_dropped_labels_are_counted() {
    let df = frame(300, 2, Some(200), &varied);
    let accepted_null = (0..300)
        .filter(|&i| varied(i).is_none() && i % 17 != 5 && i % 11 != 3)
        .count() as i64;
    let mut bank = Bank::new(vec![varied_spec(
        r#""half_life": 30.0, "label_clock": "avail""#,
    )])
    .unwrap();
    feed(&mut bank, &df, 0);
    assert_eq!(summary_count(&bank, "labels_missing"), accepted_null);
    // Only the restart drops rows here: those held at row 200.
    let at_reset = summary_count(&bank, "labels_dropped");
    assert!(at_reset > 0, "rows were held at the restart");
    let mut capped = Bank::new(vec![varied_spec(r#""half_life": 30.0, "label_clock": "avail", "max_held_rows": 3, "held_overflow": "drop_oldest""#)]).unwrap();
    feed(&mut capped, &df, 0);
    assert!(summary_count(&capped, "labels_dropped") > at_reset + 20);
    // Under no delay nothing is counted.
    let mut plain = Bank::new(vec![varied_spec(r#""half_life": 30.0"#)]).unwrap();
    feed(&mut plain, &df, 0);
    assert_eq!(
        (
            summary_count(&plain, "labels_missing"),
            summary_count(&plain, "labels_dropped")
        ),
        (0, 0)
    );
}

#[test]
fn a_zero_weight_row_and_a_null_label_are_scored_and_never_held() {
    let df =
        df!("t" => [0i64, 1, 2, 3], "x0" => [1.0, 2.0, 3.0, 4.0], "x1" => [0.5, 0.1, 0.3, 0.2],
                 "y" => [1.0, 2.0, 3.0, 4.0], "w" => [0.0, 1.0, 1.0, 1.0],
                 "avail" => [Some(1i64), None, Some(2), Some(9)])
        .unwrap();
    let mut bank = Bank::new(vec![varied_spec(
        r#""half_life": "inf", "label_clock": "avail""#,
    )])
    .unwrap();
    let out = feed(&mut bank, &df, 0);
    let ws: Vec<Option<f64>> = out
        .column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("weight_sum")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect();
    // Row 0 weighs 0 and row 1 has no label time: neither is held. Row 2
    // is held at its own row, and released, and learned, at row 3.
    assert_eq!(ws, [Some(0.0), Some(0.0), Some(0.0), Some(1.0)]);
    assert_eq!(summary_count(&bank, "labels_missing"), 1);
}

#[test]
fn a_label_before_its_row_is_refused_by_name_and_the_bank_is_untouched() {
    let df = df!("t" => [0i64, 5, 9], "x0" => [1.0, 2.0, 3.0], "x1" => [0.5, 0.1, 0.3],
                 "y" => [1.0, 2.0, 3.0], "w" => [1.0, 1.0, 1.0], "avail" => [3i64, 4, 12])
    .unwrap();
    let mut bank = Bank::new(vec![varied_spec(
        r#""half_life": "inf", "label_clock": "avail""#,
    )])
    .unwrap();
    let before = bank.save_bytes().unwrap();
    let err = bank.fit_predict(&df).unwrap_err().to_string();
    assert!(
        err.contains(
            "spec \"m\": label_clock column \"avail\" is 1 before clock column \"t\" at row 1 \
             (counting from 0), but a row's target is never known before the row, so the bank \
             refused the chunk. Give each row a label time at or after its clock."
        ),
        "{err}"
    );
    assert_eq!(bank.save_bytes().unwrap(), before);
}

#[test]
fn a_full_buffer_is_refused_unless_it_drops_its_oldest() {
    let df = frame(60, 4, None, &|_| Some(30));
    let mut bank = Bank::new(vec![varied_spec(
        r#""half_life": 30.0, "label_clock": "avail", "max_held_rows": 5"#,
    )])
    .unwrap();
    let before = bank.save_bytes().unwrap();
    let err = bank.fit_predict(&df).unwrap_err().to_string();
    assert!(
        err.contains(
            "(counting from 0) would put 6 rows waiting for their label times, above \
             max_held_rows = 5, so the bank refused the chunk. Increase max_held_rows, or set \
             held_overflow=\"drop_oldest\" to drop the row that arrived first."
        ),
        "{err}"
    );
    assert_eq!(bank.save_bytes().unwrap(), before);
}

#[test]
fn a_temporal_clock_takes_a_temporal_label_column() {
    let df = frame(120, 6, None, &varied);
    let ns = |c: &str| {
        df.column(c)
            .unwrap()
            .as_materialized_series()
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap()
    };
    let mut timed = df.clone();
    timed.with_column(ns("t").into_column()).unwrap();
    timed.with_column(ns("avail").into_column()).unwrap();
    let spec = spec_of(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0"],
        "clock": "t", "gap_cap": "1s", "half_life": "20ms", "label_clock": "avail"}"#,
    );
    let embargo = spec_of(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0"],
        "clock": "t", "gap_cap": "1s", "half_life": "20ms", "embargo": "10ms"}"#,
    );
    let mut fixed = timed.clone();
    let plus = fixed.column("t").unwrap().as_materialized_series().clone()
        + Series::new("d".into(), [10i64])
            .cast(&DataType::Duration(TimeUnit::Milliseconds))
            .unwrap();
    fixed
        .with_column(plus.unwrap().with_name("avail".into()).into_column())
        .unwrap();
    assert_same(
        &run(&embargo, &fixed, 0),
        &run(&spec, &fixed, 0),
        "a Datetime clock",
    );
    run(&spec, &timed, 7);
    // A number column beside a temporal clock is refused, naming both.
    let mut mixed = timed.clone();
    mixed
        .with_column(df.column("avail").unwrap().clone())
        .unwrap();
    let err = Bank::new(vec![spec.clone()])
        .unwrap()
        .fit_predict(&mixed)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(
            "label_clock column \"avail\" is Int64, and clock column \"t\" is Datetime(\"ms\")"
        ),
        "{err}"
    );
}

#[test]
fn every_refusal_beside_it_names_the_setting_and_the_remedy() {
    let base = r#""name": "m", "targets": ["y"], "features": ["x0"], "clock": "t", "gap_cap": 10.0, "half_life": 5.0"#;
    let cases: &[(&str, &str)] = &[
        (
            r#""model": {"type": "ewridge"}, "label_clock": "avail", "embargo": 2.0"#,
            "label_clock does not work beside embargo",
        ),
        (
            r#""model": {"type": "ewridge"}, "label_clock": "avail", "session": "s""#,
            "label_clock does not work beside a session column",
        ),
        (
            r#""model": {"type": "ewridge"}, "label_clock": "avail", "emit_drift": true, "drift_threshold": 5.0, "drift_action": "reset""#,
            "label_clock does not work beside drift_action=\"reset\"",
        ),
        (
            r#""model": {"type": "ewridge"}, "label_clock": "avail", "group": "g", "group_close": "monotone""#,
            "label_clock does not work beside group_close",
        ),
        (
            r#""model": {"type": "kalman"}, "label_clock": "avail""#,
            "label_clock does not work with kalman",
        ),
        (
            r#""model": {"type": "ewridge"}, "label_clock": "t""#,
            "label_clock is the clock column \"t\"",
        ),
        (
            r#""model": {"type": "ewridge"}, "label_clock": ["a", "b"]"#,
            "label_clock lists 2 columns for 1 targets",
        ),
        (
            r#""model": {"type": "ewridge"}, "max_held_rows": 4"#,
            "max_held_rows is given without label_clock",
        ),
        (
            r#""model": {"type": "ewridge"}, "label_clock": "avail", "max_held_rows": 0"#,
            "max_held_rows is 0",
        ),
        (
            r#""model": {"type": "ewridge"}, "label_clock": "avail", "held_overflow": "oldest""#,
            "held_overflow must be one of \"refuse\", \"drop_oldest\"; got \"oldest\"",
        ),
    ];
    for (rest, want) in cases {
        let text = format!("{{{base}, {rest}}}");
        let err = match serde_json::from_str::<Spec>(&text) {
            Ok(s) => Bank::new(vec![s])
                .err()
                .unwrap_or_else(|| panic!("{text} was accepted")),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(want), "{text}: {err}");
    }
    let two = r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y", "z"], "features": ["x0"], "clock": "t", "gap_cap": 10.0, "half_life": 5.0, "label_clock": ["a", "b"]}"#;
    let err = Bank::new(vec![spec_of(two)]).err().unwrap();
    assert!(
        err.contains("label_clock lists \"a\" and \"b\", and a spec reads one label time a row"),
        "{err}"
    );
    let no_clock = r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0"], "half_life": 5.0, "label_clock": "avail"}"#;
    let err = Bank::new(vec![spec_of(no_clock)]).err().unwrap();
    assert!(
        err.contains("label_clock \"avail\" is given with no clock"),
        "{err}"
    );
}

fn field_bits(out: &DataFrame, name: &str) -> Vec<Option<u64>> {
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(|v| v.map(f64::to_bits))
        .collect()
}

#[test]
fn settled_frac_reads_the_clock_since_the_first_row_as_with_no_delay() {
    // Every row's step counts once it is scored, held or not -- a weight of
    // 0 and a null label time included -- and a released row's label step
    // moves it from the held clock to the decay time. So `settled_frac` is
    // the undelayed stream's, which decays by every row's step: to the bit,
    // on an integer clock whose steps add up exactly.
    let mut df = frame(400, 9, Some(250), &varied);
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .enumerate()
        .map(|(i, w)| if i % 7 == 2 { 0.0 } else { w })
        .collect();
    df.with_column(Series::new("w".into(), w).into_column())
        .unwrap();
    let plain = run(&varied_spec(r#""half_life": 30.0"#), &df, 0);
    let label = run(
        &varied_spec(r#""half_life": 30.0, "label_clock": "avail""#),
        &df,
        0,
    );
    let (a, b) = (
        field_bits(&plain, "settled_frac"),
        field_bits(&label, "settled_frac"),
    );
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    assert!(bad.is_empty(), "rows {bad:?}");
    assert!(a.iter().flatten().count() > 300);
}
