//! The diagnostics' own memories (docs/PLAN.md task 278, batch 5): each
//! `*_half_life` moves its own diagnostic's fields and nothing else.
//!
//! At the defaults every memory but the calibration's is the fit's own, so
//! two memories wired to each other's accumulators (`checks.rs`,
//! `CheckCfg::of`) would pass every test that leaves them out. Here every
//! diagnostic is on at once, each memory has a value of its own, and each in
//! turn is changed alone: the fields of its diagnostic move and every other
//! field of the struct keeps its bits. The Python twin
//! (`tests/test_online_diagnostics.py`) runs the same matrix on `rls`, `sgd`
//! and `holt`; this one is where `cargo mutants` sees a swap.
//!
//! A memory left out is a multiple of the fit's: four for the calibration,
//! one for the others (`Spec::memory_multiple`). That is held by the
//! default against the same memory written out.

use online_polars::{Bank, Spec};
use polars::prelude::*;

const N: usize = 1500;
const HALF_LIFE: f64 = 120.0;

/// A seeded generator in `[-1, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Two features; a residual that is serially correlated (AR(2)), wider with
/// the first feature and heavy-tailed now and then, so every diagnostic has
/// something to read; the intercept moves by 3 halfway; uneven weights,
/// with a zero every so often.
fn frame() -> DataFrame {
    let mut r = lcg(278);
    let (mut x0, mut x1, mut y, mut w) = (vec![], vec![], vec![], vec![]);
    let (mut u1, mut u2) = (0.0, 0.0);
    for i in 0..N {
        let (a, b) = (1.7 * r(), 1.7 * r());
        let shock = r() + if i % 41 == 7 { 4.0 * r() } else { 0.0 };
        let u = 0.1 * u1 + 0.5 * u2 + shock;
        (u2, u1) = (u1, u);
        let level = if i < N / 2 { 0.3 } else { 3.3 };
        x0.push(a);
        x1.push(b);
        y.push(level + 1.2 * a - 0.7 * b + u * (1.0 + 0.3 * a.abs()));
        w.push(if i % 23 == 5 { 0.0 } else { 1.0 + 0.5 * r() });
    }
    df!("x0" => x0, "x1" => x1, "y" => y, "w" => w).unwrap()
}

/// The seven memories, the switch each needs, and the fields its
/// diagnostic writes for target `y` and features `x0`, `x1`.
const MEMORIES: [(&str, &str, &[&str]); 7] = [
    (
        "calibration_half_life",
        "emit_calibration",
        &[
            "calibration_slope_y",
            "calibration_intercept_y",
            "calibration_wald_y",
        ],
    ),
    (
        "breaks_half_life",
        "emit_breaks",
        &["studentized_y", "cusum_y", "cusum_sq_y", "break_wald_y"],
    ),
    (
        "robust_se_half_life",
        "emit_robust_se",
        &["se_coef_hc0", "se_coef_hac"],
    ),
    (
        "specification_half_life",
        "emit_specification",
        &["ljung_box_y", "breusch_pagan_y", "ramsey_reset_y"],
    ),
    (
        "tails_half_life",
        "emit_tails",
        &["skew_y", "kurtosis_y", "jarque_bera_y"],
    ),
    ("influence_half_life", "emit_influence", &["influence_y"]),
    (
        "feature_health_half_life",
        "emit_feature_health",
        &[
            "spread_ratio_x0",
            "spread_ratio_x1",
            "mean_shift_x0",
            "mean_shift_x1",
        ],
    ),
];

/// Seven distinct memories, one a key, none a multiple of another's or of
/// the fit's.
const DISTINCT: [f64; 7] = [33.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0];

/// An `ewridge` over `y` with every diagnostic on, and each memory in
/// `memories` (key, value) written out.
fn spec(memories: &[(&str, f64)]) -> Spec {
    let mut text = format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
            "features": ["x0", "x1"], "half_life": {HALF_LIFE}, "weight": "w",
            "min_weight": 10.0, "coef_every": 0, "robust_se_lags": 2"#
    );
    for (_, switch, _) in MEMORIES {
        text.push_str(&format!(r#", "{switch}": true"#));
    }
    for (key, value) in memories {
        text.push_str(&format!(r#", "{key}": {value:?}"#));
    }
    text.push('}');
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// Each field's name, its numbers' bits row by row, and its other values as
/// text.
type Fields = Vec<(String, Vec<Vec<Option<u64>>>, Vec<String>)>;

/// Every field of the struct, by name, each row as the bits of its numbers:
/// a float's bits, a list's entries' bits, anything else as text.
fn run(spec: Spec, df: &DataFrame) -> Fields {
    let mut bank = Bank::new(vec![spec]).unwrap();
    let out = DataFrame::new(df.height(), bank.fit_predict(df).unwrap()).unwrap();
    let fields = out
        .column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series();
    fields
        .into_iter()
        .map(|s| {
            let name = s.name().to_string();
            match s.dtype() {
                DataType::Float64 => {
                    let bits = s.f64().unwrap().iter().map(|v| vec![v.map(f64::to_bits)]);
                    (name, bits.collect(), vec![])
                }
                DataType::List(inner) if **inner == DataType::Float64 => {
                    let rows = s.list().unwrap().amortized_iter().map(|row| match row {
                        None => vec![],
                        Some(v) => v
                            .as_ref()
                            .f64()
                            .unwrap()
                            .iter()
                            .map(|x| x.map(f64::to_bits))
                            .collect(),
                    });
                    (name, rows.collect(), vec![])
                }
                _ => {
                    let text = (0..s.len()).map(|i| format!("{:?}", s.get(i).unwrap()));
                    (name, vec![], text.collect())
                }
            }
        })
        .collect()
}

/// The names of the fields of `b` whose values differ from `a`'s.
fn moved(a: &Fields, b: &Fields) -> Vec<String> {
    assert_eq!(
        a.iter().map(|f| &f.0).collect::<Vec<_>>(),
        b.iter().map(|f| &f.0).collect::<Vec<_>>(),
        "the same fields"
    );
    a.iter()
        .zip(b)
        .filter(|(fa, fb)| fa.1 != fb.1 || fa.2 != fb.2)
        .map(|(fa, _)| fa.0.clone())
        .collect()
}

/// How many rows of field `name` hold a number: the evidence a diagnostic
/// was read at all.
fn values(f: &Fields, name: &str) -> usize {
    let (_, rows, _) = f.iter().find(|x| x.0 == name).expect(name);
    rows.iter()
        .filter(|r| {
            r.iter()
                .any(|v| v.is_some_and(|b| f64::from_bits(b).is_finite()))
        })
        .count()
}

/// Every diagnostic on, seven distinct memories; each memory changed alone
/// moves exactly its diagnostic's fields.
#[test]
fn each_memory_moves_its_own_fields_and_nothing_else() {
    let df = frame();
    let base_memories: Vec<(&str, f64)> = MEMORIES
        .iter()
        .zip(DISTINCT)
        .map(|((key, _, _), h)| (*key, h))
        .collect();
    let base = run(spec(&base_memories), &df);
    // Each diagnostic reads values on most rows, so a field that does not
    // move is not one that was empty both times.
    for (_, _, fields) in MEMORIES {
        for f in fields {
            assert!(values(&base, f) > 700, "{f}: {} values", values(&base, f));
        }
    }
    for (i, (key, _, fields)) in MEMORIES.iter().enumerate() {
        let mut changed = base_memories.clone();
        changed[i].1 = 71.0;
        let got = run(spec(&changed), &df);
        let mut want: Vec<String> = fields.iter().map(|f| f.to_string()).collect();
        let mut seen = moved(&base, &got);
        want.sort();
        seen.sort();
        assert_eq!(seen, want, "{key} from {} to 71", DISTINCT[i]);
    }
}

/// A memory left out is the fit's (four times it for the calibration): the
/// same struct, bit for bit, as the memory written out; and the
/// calibration's default is not the fit's own.
#[test]
fn a_memory_left_out_is_its_multiple_of_the_fits() {
    let df = frame();
    let defaults = run(spec(&[]), &df);
    let written: Vec<(&str, f64)> = MEMORIES
        .iter()
        .map(|(key, _, _)| {
            let multiple = if *key == "calibration_half_life" {
                4.0
            } else {
                1.0
            };
            (*key, multiple * HALF_LIFE)
        })
        .collect();
    assert_eq!(
        moved(&defaults, &run(spec(&written), &df)),
        Vec::<String>::new()
    );
    let once = run(spec(&[("calibration_half_life", HALF_LIFE)]), &df);
    let mut seen = moved(&defaults, &once);
    seen.sort();
    assert_eq!(
        seen,
        [
            "calibration_intercept_y",
            "calibration_slope_y",
            "calibration_wald_y"
        ]
    );
}

/// A spec from JSON, filled and validated, as every door does it.
fn check(json: &str) -> Result<(), String> {
    let mut s: Spec = serde_json::from_str(json).map_err(|e| format!("did not parse: {e}"))?;
    s.check()
}

fn refused(extra: &str, want: &str) {
    let json = format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
            "features": ["x0", "x1"], "half_life": 10.0{extra}}}"#
    );
    let e = check(&json).expect_err(&format!("accepted: {json}"));
    assert!(e.contains(want), "wanted {want:?} in: {e}");
    assert!(
        e.starts_with("spec \"m\": "),
        "the spec is not named first: {e}"
    );
}

/// Each memory is refused at 0 and below, and without its switch; the lags
/// without theirs and past the ring's ceiling.
#[test]
fn the_memories_and_lags_are_refused_out_of_range_or_without_their_switch() {
    for (key, switch, _) in MEMORIES {
        refused(
            &format!(r#", "{key}": 30.0"#),
            &format!("{key} needs {switch}"),
        );
        for bad in ["0.0", "-1.0"] {
            refused(
                &format!(r#", "{switch}": true, "{key}": {bad}"#),
                &format!("{key} must be > 0 (\"inf\" is the run-once form"),
            );
        }
    }
    refused(
        r#", "ljung_box_lags": 3"#,
        "ljung_box_lags needs emit_specification",
    );
    refused(
        r#", "emit_specification": true, "ljung_box_lags": 1048577"#,
        "ljung_box_lags must be at most 1048576 (2^20), got 1048577",
    );
    refused(
        r#", "emit_robust_se": true, "robust_se_lags": 1048577"#,
        "robust_se_lags must be at most 1048576 (2^20), got 1048577",
    );
    refused(
        r#", "robust_se_lags": 2"#,
        "robust_se_lags needs emit_robust_se",
    );
    // The largest of each is taken.
    for ok in [
        r#", "emit_robust_se": true, "robust_se_lags": 1048576"#,
        r#", "emit_specification": true, "ljung_box_lags": 1048575"#,
    ] {
        let json = format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x0", "x1"], "half_life": 10.0{ok}}}"#
        );
        check(&json).unwrap_or_else(|e| panic!("refused: {e}\n{json}"));
    }
}
