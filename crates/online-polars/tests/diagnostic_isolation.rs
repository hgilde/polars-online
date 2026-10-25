//! A diagnostic never moves the fit (docs/PLAN.md task 278, batch 6): the
//! Rust twin of `tests/test_diagnostic_isolation.py`, for `ewridge` and
//! `sgd`, where `cargo mutants` sees it.
//!
//! Each case switches one diagnostic on, or sets one of a diagnostic's knobs
//! off its default, and holds every field the plain spec writes -- `pred_*`,
//! `resid_*`, `coef`, `support_coef`, `weight_sum`, `settled_frac`,
//! `withheld_reason` -- to the plain spec's, to the bit. A diagnostic that
//! fed a row to the model, consumed one it then missed, or advanced a clock
//! the model shares would show as a moved bit.
//!
//! No case compares two identical runs: each asserts the switch wrote a
//! value of its own, and each knob that its diagnostic fields differ from
//! its switch's at the default. The drift detector fires (a level shift of 8
//! at row 450 against noise of about 1), and that is asserted too.
//!
//! Two streams: `rows`, with no clock, uneven weights with zeros and a
//! second target missing on one row in twenty; and `clocked`, the same rows
//! on an irregular clock with steps past `gap_cap`, two sessions and an
//! `embargo`.

use std::collections::HashMap;

use online_polars::{Bank, Spec};
use polars::prelude::*;

const N: usize = 600;
/// The row the level shifts on.
const SHIFT: usize = 450;

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

/// Two features; `y` with an AR(2) residual, heteroskedastic in `x0`, and
/// a level shift of 8 at row 450, away from the gaps and session changes; `y2` missing on one row in twenty; weights
/// on `[0.5, 1.5]` with one row in thirty at 0; a clock with steps on `[0.5,
/// 1.5]` and one of 12 every 200 rows; a session change at rows 200 and 400.
fn frame() -> DataFrame {
    let mut r = lcg(27806);
    let (mut t, mut s, mut x0, mut x1, mut y, mut y2, mut w) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut clock, mut u1, mut u2) = (0.0, 0.0, 0.0);
    for i in 0..N {
        clock += if i % 200 == 0 { 12.0 } else { 1.0 + 0.5 * r() };
        let (a, b) = (r(), r());
        let u = 0.1 * u1 + 0.5 * u2 + 1.7 * r();
        (u2, u1) = (u1, u);
        let level = if i < SHIFT { 0.3 } else { 8.3 };
        t.push(clock);
        s.push((i / 200) as i64);
        x0.push(a);
        x1.push(b);
        y.push(level + 1.2 * a - 0.7 * b + u * (1.0 + 0.3 * a.abs()));
        let miss = r() < -0.9;
        y2.push((!miss).then(|| -0.5 + 0.4 * a + 0.9 * b + r()));
        w.push(if r() < -14.0 / 15.0 {
            0.0
        } else {
            1.0 + 0.5 * r()
        });
    }
    df!("t" => t, "s" => s, "x0" => x0, "x1" => x1, "y" => y, "y2" => y2, "w" => w).unwrap()
}

const VARIANTS: [(&str, &str); 2] = [
    ("rows", ""),
    (
        "clocked",
        "clock = \"t\"\ngap_cap = 5.0\nsession = \"s\"\nsession_gap = 10.0\nembargo = 2.0\n",
    ),
];

const MODELS: [(&str, &str); 2] = [
    ("ewridge", "type = \"ewridge\""),
    ("sgd", "type = \"sgd\"\nlearning_rate = 0.02"),
];

/// One case: its TOML lines beside the base spec's, the case whose knob it
/// varies, whether it runs on a half-life grid (the switches that choose
/// across slots), and whether `sgd` refuses it (the switches that read a
/// row's leverage or a least-squares covariance).
struct Case {
    name: &'static str,
    top: &'static str,
    parent: Option<&'static str>,
    grid: bool,
    ewridge_only: bool,
}

const fn case(name: &'static str, top: &'static str, parent: Option<&'static str>) -> Case {
    Case {
        name,
        top,
        parent,
        grid: false,
        ewridge_only: false,
    }
}

const fn leverage(name: &'static str, top: &'static str, parent: Option<&'static str>) -> Case {
    Case {
        name,
        top,
        parent,
        grid: false,
        ewridge_only: true,
    }
}

const fn grid(name: &'static str, top: &'static str, parent: Option<&'static str>) -> Case {
    Case {
        name,
        top,
        parent,
        grid: true,
        ewridge_only: false,
    }
}

const CASES: &[Case] = &[
    case("sigma", "emit_sigma = true", None),
    case("zscore", "emit_zscore = true", None),
    case("metrics", "emit_metrics = true", None),
    case("clocks", "emit_clocks = true", None),
    case("conformal", "conformal = 0.9", None),
    case("conformal 0.5", "conformal = 0.5", Some("conformal")),
    case(
        "conformal_rate",
        "conformal = 0.9\nconformal_rate = 0.3",
        Some("conformal"),
    ),
    case("resid_quantiles", "resid_quantiles = [0.5]", None),
    case(
        "resid_quantiles levels",
        "resid_quantiles = [0.05, 0.5, 0.99]",
        Some("resid_quantiles"),
    ),
    case("autocorr", "emit_autocorr = true", None),
    case(
        "resid_autocorr_lag",
        "emit_autocorr = true\nresid_autocorr_lag = 2",
        Some("autocorr"),
    ),
    case("drift", "emit_drift = true", None),
    case(
        "drift_delta",
        "emit_drift = true\ndrift_delta = 2.0",
        Some("drift"),
    ),
    case(
        "drift_threshold",
        "emit_drift = true\ndrift_threshold = 5.0",
        Some("drift"),
    ),
    case("calibration", "emit_calibration = true", None),
    case(
        "calibration_half_life",
        "emit_calibration = true\ncalibration_half_life = 20.0",
        Some("calibration"),
    ),
    case(
        "calibration_half_life inf",
        "emit_calibration = true\ncalibration_half_life = inf",
        Some("calibration"),
    ),
    case(
        "calibration horizon_rows",
        "emit_calibration = true\nhorizon_rows = 3",
        Some("calibration"),
    ),
    case("breaks", "emit_breaks = true", None),
    case(
        "breaks_half_life",
        "emit_breaks = true\nbreaks_half_life = 20.0",
        Some("breaks"),
    ),
    case(
        "breaks_half_life inf",
        "emit_breaks = true\nbreaks_half_life = inf",
        Some("breaks"),
    ),
    case(
        "breaks horizon_rows",
        "emit_breaks = true\nhorizon_rows = 3",
        Some("breaks"),
    ),
    case("specification", "emit_specification = true", None),
    case(
        "specification_half_life",
        "emit_specification = true\nspecification_half_life = 20.0",
        Some("specification"),
    ),
    case(
        "specification_half_life inf",
        "emit_specification = true\nspecification_half_life = inf",
        Some("specification"),
    ),
    case(
        "ljung_box_lags",
        "emit_specification = true\nljung_box_lags = 3",
        Some("specification"),
    ),
    case(
        "specification horizon_rows",
        "emit_specification = true\nhorizon_rows = 3",
        Some("specification"),
    ),
    case("tails", "emit_tails = true", None),
    case(
        "tails_half_life",
        "emit_tails = true\ntails_half_life = 20.0",
        Some("tails"),
    ),
    case(
        "tails_half_life inf",
        "emit_tails = true\ntails_half_life = inf",
        Some("tails"),
    ),
    case("feature_health", "emit_feature_health = true", None),
    case(
        "feature_health_half_life",
        "emit_feature_health = true\nfeature_health_half_life = 20.0",
        Some("feature_health"),
    ),
    leverage("error_inflation", "emit_error_inflation = true", None),
    leverage("se_coef", "emit_se_coef = true", None),
    leverage("influence", "emit_influence = true", None),
    leverage(
        "influence_half_life",
        "emit_influence = true\ninfluence_half_life = 20.0",
        Some("influence"),
    ),
    leverage(
        "influence_half_life inf",
        "emit_influence = true\ninfluence_half_life = inf",
        Some("influence"),
    ),
    leverage("robust_se", "emit_robust_se = true", None),
    leverage(
        "robust_se_half_life",
        "emit_robust_se = true\nrobust_se_half_life = 20.0",
        Some("robust_se"),
    ),
    leverage(
        "robust_se_half_life inf",
        "emit_robust_se = true\nrobust_se_half_life = inf",
        Some("robust_se"),
    ),
    leverage(
        "robust_se_lags",
        "emit_robust_se = true\nrobust_se_lags = 3",
        Some("robust_se"),
    ),
    leverage(
        "robust_se horizon_rows",
        "emit_robust_se = true\nhorizon_rows = 3",
        Some("robust_se"),
    ),
    grid("selected", "emit_selected = true", None),
    grid("averaged", "emit_averaged = true", None),
    grid(
        "average_eta",
        "emit_averaged = true\naverage_eta = 10.0",
        Some("averaged"),
    ),
];

/// Every switch the model takes, at once, with knobs off their defaults.
fn everything(model: &str) -> String {
    let mut lines = vec![
        "emit_sigma = true",
        "emit_zscore = true",
        "emit_metrics = true",
        "emit_clocks = true",
        "conformal = 0.9",
        "conformal_rate = 0.3",
        "resid_quantiles = [0.05, 0.5, 0.99]",
        "emit_autocorr = true",
        "resid_autocorr_lag = 2",
        "emit_drift = true",
        "emit_calibration = true",
        "emit_breaks = true",
        "emit_specification = true",
        "horizon_rows = 3",
        "emit_tails = true",
        "emit_feature_health = true",
    ];
    if model == "ewridge" {
        lines.extend([
            "emit_error_inflation = true",
            "emit_se_coef = true",
            "emit_influence = true",
            "emit_robust_se = true",
        ]);
    }
    lines.join("\n")
}

fn spec(model: &str, variant: &str, top: &str, grid: bool) -> Spec {
    let (_, extra) = VARIANTS.iter().find(|(v, _)| *v == variant).unwrap();
    let (_, table) = MODELS.iter().find(|(m, _)| *m == model).unwrap();
    // At 60, the fit is 95% settled by row 260, where the diagnostics that
    // wait for it (calibration, breaks, specification, tails) start.
    let half_life = if grid { "[30.0, 120.0]" } else { "60.0" };
    // On a clock the drift threshold, in sigma x clock units, has no default.
    let threshold =
        if !extra.is_empty() && top.contains("emit_drift") && !top.contains("drift_threshold") {
            "drift_threshold = 20.0\n"
        } else {
            ""
        };
    let text = format!(
        "name = \"m\"\ntargets = [\"y\", \"y2\"]\nfeatures = [\"x0\", \"x1\"]\nweight = \"w\"\n\
         half_life = {half_life}\ncoef_every = 0.0\n{extra}{threshold}{top}\n[model]\n{table}\n"
    );
    toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"))
}

fn fields(spec: Spec, df: &DataFrame) -> Vec<Series> {
    let mut bank = Bank::new(vec![spec]).unwrap();
    let out = DataFrame::new(df.height(), bank.fit_predict(df).unwrap()).unwrap();
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series()
}

/// A series as its exact contents: a float by its bits (so `-0.0` and `0.0`
/// differ), a list by its lengths and its values' bits, anything else by its
/// `AnyValue`'s text.
fn bits(s: &Series) -> Vec<String> {
    if let Ok(f) = s.f64() {
        return f
            .iter()
            .map(|v| format!("{:?}", v.map(f64::to_bits)))
            .collect();
    }
    if let Ok(l) = s.list() {
        return (0..l.len())
            .map(|i| match l.get_as_series(i) {
                None => "null".to_string(),
                Some(inner) => format!("{:?}", bits(&inner)),
            })
            .collect();
    }
    (0..s.len())
        .map(|i| format!("{:?}", s.get(i).unwrap()))
        .collect()
}

fn named<'a>(out: &'a [Series], name: &str) -> Option<&'a Series> {
    out.iter().find(|s| s.name().as_str() == name)
}

/// Every field of `base` in `out`, bit for bit.
fn assert_fit_same(base: &[Series], out: &[Series], what: &str) {
    for b in base {
        let o = named(out, b.name()).unwrap_or_else(|| panic!("{what}: {} is gone", b.name()));
        let (x, y) = (bits(b), bits(o));
        if let Some(i) = (0..x.len()).find(|&i| x[i] != y[i]) {
            panic!(
                "{what}: {} moved, first at row {i}: {:?} vs {:?}",
                b.name(),
                b.get(i).unwrap(),
                o.get(i).unwrap()
            );
        }
    }
}

/// The fields `out` writes beyond `base`'s, by name and exact contents.
fn own(base: &[Series], out: &[Series]) -> Vec<(String, Vec<String>)> {
    out.iter()
        .filter(|s| named(base, s.name()).is_none())
        .map(|s| (s.name().to_string(), bits(s)))
        .collect()
}

#[test]
fn a_diagnostic_never_moves_the_fit() {
    let df = frame();
    for (model, _) in MODELS {
        for (variant, _) in VARIANTS {
            let plain = fields(spec(model, variant, "", false), &df);
            let plain_grid = fields(spec(model, variant, "", true), &df);
            let mut ran: HashMap<&str, Vec<(String, Vec<String>)>> = HashMap::new();
            for c in CASES {
                if c.ewridge_only && model != "ewridge" {
                    continue;
                }
                let what = format!("{model} on {variant}, {}", c.name);
                let base = if c.grid { &plain_grid } else { &plain };
                let out = fields(spec(model, variant, c.top, c.grid), &df);
                assert_fit_same(base, &out, &what);
                let mine = own(base, &out);
                let values: usize = out
                    .iter()
                    .filter(|s| named(base, s.name()).is_none())
                    .map(|s| s.len() - s.null_count())
                    .sum();
                assert!(
                    !mine.is_empty() && values > 0,
                    "{what}: wrote nothing of its own, so the comparison proves nothing"
                );
                if let Some(p) = c.parent {
                    // A parent comes before its knobs in `CASES`.
                    let theirs = &ran[p];
                    assert_ne!(
                        &mine, theirs,
                        "{what}: the knob changed nothing against {p}"
                    );
                }
                if c.name == "drift" {
                    let fired: usize = out
                        .iter()
                        .filter(|s| s.name().starts_with("drift_"))
                        .map(|s| s.bool().unwrap().num_trues())
                        .sum();
                    assert!(fired > 0, "{what}: the detector never fired");
                }
                ran.insert(c.name, mine);
            }
        }
    }
}

#[test]
fn every_switch_at_once_never_moves_the_fit() {
    let df = frame();
    for (model, _) in MODELS {
        for (variant, _) in VARIANTS {
            let what = format!("{model} on {variant}, every switch");
            let base = fields(spec(model, variant, "", false), &df);
            let out = fields(spec(model, variant, &everything(model), false), &df);
            assert_fit_same(&base, &out, &what);
            let empty: Vec<_> = out
                .iter()
                .filter(|s| named(&base, s.name()).is_none() && s.null_count() == s.len())
                .map(|s| s.name().to_string())
                .collect();
            assert!(empty.is_empty(), "{what}: fields with no value: {empty:?}");
        }
    }
}
