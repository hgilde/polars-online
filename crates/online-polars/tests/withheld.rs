//! Every row whose outputs a model holds back says why. `ew_cov`'s floor
//! of 2 -- a variance needs two rows, so a lower `min_weight` is raised --
//! held rows back with no `withheld_reason`, as did the rows `kmeans`,
//! `hmm` and `bocpd` hold to seed themselves from (`warm_rows`). The floor
//! is the gate's now, `below_min_weight`; the seeding warm-up is its own,
//! `below_warm_rows`, a count of rows that a decayed weight can sit above
//! for every row of it.

use online_polars::{Bank, Spec};
use polars::prelude::*;

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap()
}

/// Two features of no structure, `n` rows, with zero weights here and
/// there (rows 0 and 7 among them): a row of weight 0 is scored and
/// learns nothing.
fn frame(n: usize) -> DataFrame {
    let mut s: u64 = 7;
    let mut draw = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let x0: Vec<f64> = (0..n).map(|_| draw()).collect();
    let x1: Vec<f64> = (0..n).map(|_| draw()).collect();
    let w: Vec<f64> = (0..n)
        .map(|i| if i == 0 || i % 7 == 0 { 0.0 } else { 1.0 })
        .collect();
    df!("x0" => x0, "x1" => x1, "w" => w).unwrap()
}

/// The first field's nulls and each row's reason, `""` for none, over the
/// whole frame fed in chunks of `size`.
fn run(spec: &Spec, df: &DataFrame, size: usize) -> (Vec<bool>, Vec<String>) {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
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
    let out = out.unwrap();
    let fields = out
        .column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series();
    let first = &fields[0];
    let null: Vec<bool> = first.is_null().iter().map(|v| v.unwrap()).collect();
    let reason = fields
        .iter()
        .find(|f| f.name().as_str() == "withheld_reason")
        .unwrap()
        .cast(&DataType::String)
        .unwrap();
    let reason = reason
        .str()
        .unwrap()
        .iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect();
    (null, reason)
}

/// Every null row says why, every row that says why is null, and the
/// reasons in order are `want`: `(reason, rows)` runs from row 0, then
/// none.
fn check(json: &str, want: &[(&str, usize)]) {
    let s = spec(json);
    let df = frame(80);
    let (null, reason) = run(&s, &df, 80);
    for (i, (n, r)) in null.iter().zip(&reason).enumerate() {
        assert_eq!(
            *n,
            !r.is_empty(),
            "{}: row {i} null {n}, reason {r:?}",
            s.name
        );
    }
    let mut expected = Vec::new();
    for (r, rows) in want {
        expected.extend(std::iter::repeat_n(r.to_string(), *rows));
    }
    expected.resize(reason.len(), String::new());
    assert_eq!(reason, expected, "{json}");
    // The same in chunks of 3 rows.
    let (_, chunked) = run(&s, &df, 3);
    assert_eq!(chunked, reason, "{json}: in chunks");
}

#[test]
fn ew_cov_rows_its_floor_holds_back_name_min_weight() {
    // Row 0 weighs 0, so the weight before rows 0 and 1 is 0, before row 2
    // it is 1 and before row 3 1.99 at a half-life of 50: below 2. Before
    // row 4 it is 2.96.
    for mw in ["0", "1", "1.5"] {
        check(
            &format!(
                r#"{{"name": "m", "model": {{"type": "ew_cov", "stats": ["mean"]}},
                    "targets": ["x0"], "features": ["x0", "x1"], "half_life": 50,
                    "weight": "w", "min_weight": {mw}}}"#
            ),
            &[("below_min_weight", 4)],
        );
    }
    // Under a window of 5 rows the floor reads the weight inside it.
    check(
        r#"{"name": "m", "model": {"type": "ew_cov", "stats": ["mean"], "window_size": 5},
            "targets": ["x0"], "features": ["x0", "x1"], "half_life": 50,
            "weight": "w", "min_weight": 0}"#,
        &[("below_min_weight", 4)],
    );
}

#[test]
fn kmeans_rows_before_seeding_name_warm_rows() {
    // `min_weight` 3 (the features and one) holds rows 0 to 4, the weight
    // before row 4 being 2.96, and the warm-up the rest of the buffer of
    // 20 learned rows: rows 0, 7, 14 and 21 weigh 0, so the 20th learned
    // row is row 23, which reports nothing and seeds the centres.
    check(
        r#"{"name": "m", "model": {"type": "kmeans", "n_clusters": 3, "warm_rows": 20},
            "targets": ["x0"], "features": ["x0", "x1"], "half_life": 50, "weight": "w"}"#,
        &[("below_min_weight", 5), ("below_warm_rows", 19)],
    );
}

#[test]
fn hmm_rows_before_seeding_name_warm_rows() {
    check(
        r#"{"name": "m", "model": {"type": "hmm", "n_components": 2, "precision_prior": 1.0,
            "warm_rows": 10}, "targets": ["x0"], "features": ["x0", "x1"],
            "half_life": 50, "weight": "w"}"#,
        &[("below_warm_rows", 12)],
    );
}

#[test]
fn bocpd_rows_held_for_its_prior_name_warm_rows() {
    // `min_weight` 1 holds row 0, and the warm-up the rest of its four
    // learned rows.
    check(
        r#"{"name": "m", "model": {"type": "bocpd", "hazard": 50.0}, "targets": ["x0"],
            "features": ["x0", "x1"], "weight": "w"}"#,
        &[("below_min_weight", 2), ("below_warm_rows", 3)],
    );
}

/// `x1 = x0 + eps·b` and an independent `x2`, `n` rows, with `y` their
/// combination and some noise.
fn collinear(n: usize, eps: f64) -> DataFrame {
    let mut s: u64 = 278;
    let mut draw = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let (mut x0, mut x1, mut x2, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let (a, b, c, e) = (draw(), draw(), draw(), draw());
        x0.push(2.0 + a);
        x1.push(2.0 + a + eps * b);
        x2.push(-1.0 + c);
        y.push(0.7 + x0[x0.len() - 1] - 2.0 * x1[x1.len() - 1] + 0.5 * c + 0.1 * e);
    }
    df!("x0" => x0, "x1" => x1, "x2" => x2, "y" => y).unwrap()
}

/// A fit whose normal equations are past the condition at which a solve
/// keeps a digit is withheld and names it, `ill_conditioned` (task 278
/// (d4)): every row predicted from such a fit is null and says so, the
/// same in chunks of 3 rows, and its solves are counted. The noise gate,
/// on by default here, reads such a fit as not a number and leaves the row
/// to this reason. Inside the limit (`eps = 1e-4`) no row says it.
#[test]
fn ewridge_rows_of_a_fit_past_its_condition_name_ill_conditioned() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "ridge": 0.0, "max_rows_between_solves": 1},
            "targets": ["y"], "features": ["x0", "x1", "x2"], "lam": 1.0}"#,
    );
    let df = collinear(300, 1e-11);
    let (null, reason) = run(&s, &df, 300);
    for (i, (n, r)) in null.iter().zip(&reason).enumerate() {
        assert_eq!(*n, !r.is_empty(), "row {i} null {n}, reason {r:?}");
    }
    let ill = reason.iter().filter(|r| *r == "ill_conditioned").count();
    // Measured 2026-10-25 on aarch64: 116 of the 300 rows. Which solves
    // rounding lets through past `1/u` is the platform's arithmetic's, so
    // the count is held loosely.
    assert!(ill > 30, "{ill} rows named ill_conditioned");
    let (_, chunked) = run(&s, &df, 3);
    assert_eq!(chunked, reason, "in chunks");
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&df).unwrap();
    let failures: u64 = bank.solve_failures()[0].iter().map(|(_, n)| n).sum();
    assert!(
        failures >= ill as u64,
        "{failures} failures for {ill} withheld rows"
    );

    let (_, reason) = run(&s, &collinear(300, 1e-4), 300);
    assert!(
        reason[10..].iter().all(|r| r != "ill_conditioned"),
        "{reason:?}"
    );
}
