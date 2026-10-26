//! A readiness gate never moves the residual spread, and the residual
//! autocorrelation weighs each residual by its row (docs/PLAN.md task 278,
//! batch 17): the Rust side of `tests/test_readiness_sweep.py` and
//! `tests/test_autocorr_weights.py`, where `cargo mutants` sees it.
//!
//! The spread -- each slot's EW mean squared out-of-sample residual, which
//! `sigma`, `zscore`, the drift detector's scale and the ranking of
//! `emit_selected` and `emit_averaged` read -- folds each slot's own
//! prediction on the rows a gate withholds (`Persisted::ungated_spread`): a
//! gate withholds output only. Its definition, checked here: the EW mean
//! square of the residuals of the same spec with no gate, at each row's
//! weight. A state written before schema 64 has neither flag and goes on
//! folding only the residuals it reports, and weighing each residual 1 in
//! the autocorrelation: made here by leaving both flags out of a fresh
//! bank's bytes, as such a state leaves them, and checked against the same
//! definitions read over its own reported residuals.

use online_polars::online_core::EwAutoCorr;
use online_polars::{Bank, Spec};
use polars::prelude::*;

const N: usize = 400;

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

/// Two features; `y` with an AR(1) residual; weights on `[0.5, 1.5]` with
/// one row in twenty at 0; a clock with steps on `[0.5, 1.5]` and one of 12
/// every 150 rows, past the clocked specs' `gap_cap` of 5, so the weight
/// falls back under the floors after it.
fn frame() -> DataFrame {
    let mut r = lcg(27817);
    let (mut t, mut x0, mut x1, mut y, mut w) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut clock, mut u) = (0.0, 0.0);
    for i in 0..N {
        clock += if i % 150 == 0 { 12.0 } else { 1.0 + 0.5 * r() };
        let (a, b) = (r(), r());
        u = 0.6 * u + r();
        t.push(clock);
        x0.push(a);
        x1.push(b);
        y.push(0.3 + 1.2 * a - 0.7 * b + u);
        w.push(if r() < -0.9 { 0.0 } else { 1.0 + 0.5 * r() });
    }
    df!("t" => t, "x0" => x0, "x1" => x1, "y" => y, "w" => w).unwrap()
}

/// A spec of `kind`'s model table over the frame, with `extra` keys.
fn spec(model: &str, extra: &str) -> Spec {
    let text = format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": ["x0", "x1"],
            "weight": "w", "emit_sigma": true {extra}}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

/// The output's fields, by name.
fn run_bank(bank: &mut Bank, df: &DataFrame) -> Vec<Series> {
    let out = DataFrame::new(df.height(), bank.fit_predict(df).unwrap()).unwrap();
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series()
}

fn run(spec: &Spec, df: &DataFrame) -> (Vec<Series>, Vec<u8>) {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    let out = run_bank(&mut bank, df);
    (out, bank.save_bytes().unwrap())
}

/// The same spec from a state in the form one written before schema 64
/// holds: a fresh bank's streams with both flags left out.
fn run_kept(spec: &Spec, df: &DataFrame) -> Vec<Series> {
    let mut fresh = Bank::new(vec![spec.clone()]).unwrap();
    fresh.fit_predict(&df.slice(0, 0)).unwrap();
    let bytes = fresh.save_bytes().unwrap();
    let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let mut found = 0;
    strip(&mut v, &mut found);
    assert!(found > 0, "a spec's stream sets the flags");
    let mut kept = Vec::new();
    rmpv::encode::write_value(&mut kept, &v).unwrap();
    let mut bank = Bank::load_bytes(&kept, Some(std::slice::from_ref(spec))).unwrap();
    run_bank(&mut bank, df)
}

fn strip(v: &mut rmpv::Value, found: &mut usize) {
    match v {
        rmpv::Value::Map(m) => {
            let before = m.len();
            m.retain(|(k, _)| !matches!(k.as_str(), Some("ungated_spread" | "weighted_autocorr")));
            *found += before - m.len();
            m.iter_mut().for_each(|(_, v)| strip(v, found));
        }
        rmpv::Value::Array(a) => a.iter_mut().for_each(|v| strip(v, found)),
        _ => {}
    }
}

/// The values under every `resid_var` and `resid_w` key of a saved state.
fn spread(bytes: &[u8]) -> Vec<String> {
    fn walk(v: &rmpv::Value, out: &mut Vec<String>) {
        match v {
            rmpv::Value::Map(m) => {
                for (k, v) in m {
                    if matches!(k.as_str(), Some("resid_var" | "resid_w")) {
                        out.push(format!("{v}"));
                    }
                    walk(v, out);
                }
            }
            rmpv::Value::Array(a) => a.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(
        &rmpv::decode::read_value(&mut &bytes[..]).unwrap(),
        &mut out,
    );
    assert!(!out.is_empty(), "a state with a spread");
    out
}

fn named<'a>(out: &'a [Series], name: &str) -> &'a Series {
    out.iter()
        .find(|s| s.name().as_str() == name)
        .unwrap_or_else(|| panic!("no field {name}"))
}

fn with_prefix<'a>(out: &'a [Series], prefix: &str) -> Vec<&'a Series> {
    let found: Vec<&Series> = out
        .iter()
        .filter(|s| s.name().as_str() == prefix || s.name().starts_with(&format!("{prefix}@")))
        .collect();
    assert!(!found.is_empty(), "no field {prefix}");
    found
}

fn floats(s: &Series) -> Vec<Option<f64>> {
    s.f64().unwrap().iter().collect()
}

fn bits(s: &Series) -> Vec<Option<u64>> {
    floats(s).into_iter().map(|v| v.map(f64::to_bits)).collect()
}

fn withheld(out: &[Series]) -> usize {
    out.iter()
        .filter(|s| s.name().starts_with("withheld_reason"))
        .map(|s| s.len() - s.null_count())
        .sum()
}

/// The models: an `ewridge` and an `sgd` over a grid of two half-lives,
/// whose ranking the selection and the average read, and an `rls`.
const MODELS: [(&str, &str, &str); 3] = [
    (
        "ewridge",
        r#"{"type": "ewridge"}"#,
        r#", "half_life": [6.0, 40.0], "emit_selected": true, "emit_averaged": true"#,
    ),
    ("rls", r#"{"type": "rls"}"#, r#", "half_life": 30.0"#),
    (
        "sgd",
        r#"{"type": "sgd", "learning_rate": 0.02}"#,
        r#", "half_life": [6.0, 40.0], "emit_selected": true, "emit_averaged": true"#,
    ),
];

/// The gates, the noise gate on the two ridge systems only.
fn gates(kind: &str) -> Vec<&'static str> {
    // A weight of 6: the half-life of 6 tops out near 9 a row.
    let mut g = vec![r#", "min_weight": 6.0"#, r#", "min_settled_frac": 0.5"#];
    if kind != "sgd" {
        // Kish's size at a half-life of 6 tops out near 17, past the 7 a
        // ratio of 1.2 needs for three unknowns.
        g.push(r#", "max_error_inflation": 1.2"#);
    }
    g
}

const EMBARGO: &str = r#", "clock": "t", "gap_cap": 5.0, "embargo": 3.0"#;

/// With and without each gate, on the row clock and under an `embargo`,
/// every slot's `sigma` is the same on every row, and the spread the state
/// keeps is the same to the bit; on every row the gate withheld no slot of,
/// the selection and the average are the same.
#[test]
fn a_gate_never_moves_the_residual_spread_or_what_reads_it() {
    let df = frame();
    for (kind, model, top) in MODELS {
        for held in ["", EMBARGO] {
            let base = spec(model, &format!("{top}{held}"));
            let (b, b_bytes) = run(&base, &df);
            for gate in gates(kind) {
                let what = format!("{kind}{gate}{held}");
                let (g, g_bytes) = run(&spec(model, &format!("{top}{held}{gate}")), &df);
                assert!(
                    withheld(&g) > withheld(&b),
                    "{what}: the gate withheld nothing the base reports"
                );
                for s in with_prefix(&b, "sigma_y") {
                    let (x, y) = (bits(s), bits(named(&g, s.name())));
                    let first = x.iter().zip(&y).position(|(p, q)| p != q);
                    assert!(
                        first.is_none(),
                        "{what}: {} moved, first on row {first:?}",
                        s.name()
                    );
                }
                assert_eq!(
                    spread(&b_bytes),
                    spread(&g_bytes),
                    "{what}: the saved spread moved"
                );
                if !top.contains("emit_selected") {
                    continue;
                }
                let slots = with_prefix(&g, "pred_y");
                let every: Vec<bool> = (0..df.height())
                    .map(|i| slots.iter().all(|s| s.f64().unwrap().get(i).is_some()))
                    .collect();
                let rows = every.iter().filter(|&&e| e).count();
                assert!(rows > 150, "{what}: {rows} rows report every slot");
                for name in ["pred_y__selected", "pred_y__averaged"] {
                    let (x, y) = (bits(named(&b, name)), bits(named(&g, name)));
                    for i in (0..df.height()).filter(|&i| every[i]) {
                        assert_eq!(x[i], y[i], "{what}: {name} moved on row {i}");
                    }
                }
            }
        }
    }
}

/// `sigma` against its definition: before each row, the root of the EW
/// mean of the squared residuals at each row's weight, `W' = λW + w`,
/// `v' = (λWv + w r²)/W'` (a row with no residual only ages `W`), the
/// decay a half-life's on the row clock. Under a gate, over the residuals
/// the same spec reports with no gate -- each slot's own prediction; in a
/// state written before schema 64, over the residuals it reports itself.
/// The two differ on this stream, which is what the gate moved.
#[test]
fn sigma_is_the_ew_mean_square_of_each_slots_own_residual() {
    let df = frame();
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    // The ungated spec: `min_weight` 0, but on `ewridge` unset -- its own
    // floor, a row per unknown, below which its system has fewer rows than
    // unknowns -- and its noise gate off.
    for (model, ungated, gate, half_life) in [
        (
            r#"{"type": "ewridge"}"#,
            r#", "max_error_inflation": "inf""#,
            r#", "max_error_inflation": "inf", "min_weight": 25.0"#,
            20.0,
        ),
        (
            r#"{"type": "rls"}"#,
            r#", "min_weight": 0.0"#,
            r#", "min_weight": 25.0"#,
            30.0,
        ),
    ] {
        let top = format!(r#", "half_life": {half_life}"#);
        let (free, _) = run(&spec(model, &format!("{top}{ungated}")), &df);
        let gated = spec(model, &format!("{top}{gate}"));
        let (now, _) = run(&gated, &df);
        let kept = run_kept(&gated, &df);
        let lam = 0.5f64.powf(1.0 / half_life);
        for (out, resid_of, what) in [
            (&now, &free, "a gate's spread, from the ungated residuals"),
            (
                &kept,
                &kept,
                "a kept state's, from its own reported residuals",
            ),
        ] {
            let resid = floats(named(resid_of, "resid_y"));
            let sigma = floats(named(out, "sigma_y"));
            let (mut big_w, mut v) = (0.0f64, 0.0f64);
            let mut checked = 0;
            for i in 0..df.height() {
                let want = (big_w > 0.0).then(|| v.max(0.0).sqrt());
                match (sigma[i], want) {
                    (Some(g), Some(h)) => {
                        assert!(
                            (g - h).abs() <= 1e-12 * h.max(1e-300),
                            "{model} {what}: row {i}: {g} vs {h}"
                        );
                        checked += 1;
                    }
                    (None, None) => {}
                    (g, h) => panic!("{model} {what}: row {i}: {g:?} vs {h:?}"),
                }
                let step = if i == 0 { 1.0 } else { lam };
                match resid[i].filter(|r| r.is_finite()) {
                    Some(r) => {
                        let w_new = step * big_w + w[i];
                        if w_new > 0.0 {
                            v = (step * big_w * v + w[i] * r * r) / w_new;
                            big_w = w_new;
                        }
                    }
                    None => big_w *= step,
                }
            }
            assert!(checked > 300, "{model} {what}: {checked} rows");
        }
        // What moved: the rows the gate withheld feed the one and not the
        // other, until their weight has decayed away.
        let (a, b) = (
            floats(named(&now, "sigma_y")),
            floats(named(&kept, "sigma_y")),
        );
        let moved = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert!(moved > 50, "{model}: the gate moved {moved} rows of sigma");
    }
}

/// The residual autocorrelation folds each residual at its row's weight
/// (`EwAutoCorr::update_weighted`), and a state written before schema 64
/// at 1 (`EwAutoCorr::update`), over the residuals the output reports, a
/// row with none or of weight 0 only ageing it.
#[test]
fn the_residual_autocorrelation_weighs_each_residual_by_its_row() {
    let df = frame();
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let half_life = 30.0;
    let s = spec(
        r#"{"type": "rls"}"#,
        &format!(r#", "half_life": {half_life}, "emit_autocorr": true, "resid_autocorr_lag": 2"#),
    );
    let lam = 0.5f64.powf(1.0 / half_life);
    let (now, _) = run(&s, &df);
    let kept = run_kept(&s, &df);
    for (out, weighed) in [(&now, true), (&kept, false)] {
        let resid = floats(named(out, "resid_y"));
        let got = floats(named(out, "autocorr_y"));
        let mut est = EwAutoCorr::new(2).unwrap();
        let mut checked = 0;
        for i in 0..df.height() {
            match (got[i], est.get()) {
                (Some(g), Some(h)) => {
                    assert!(
                        (g - h).abs() <= 1e-12,
                        "weighed {weighed}: row {i}: {g} vs {h}"
                    );
                    checked += 1;
                }
                (None, None) => {}
                (g, h) => panic!("weighed {weighed}: row {i}: {g:?} vs {h:?}"),
            }
            let step = if i == 0 { 1.0 } else { lam };
            match resid[i].filter(|r| r.is_finite() && w[i] > 0.0) {
                Some(r) if weighed => est.update_weighted(r, w[i], step),
                Some(r) => est.update(r, step),
                None => est.age(step),
            }
        }
        assert!(checked > 300, "weighed {weighed}: {checked} rows");
    }
    let (a, b) = (
        floats(named(&now, "autocorr_y")),
        floats(named(&kept, "autocorr_y")),
    );
    let moved = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert!(
        moved > 300,
        "the weights moved {moved} rows of the autocorrelation"
    );
}
