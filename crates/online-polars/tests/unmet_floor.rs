//! The readiness warning that `min_weight` has not been met, for a target
//! whose weight is not at a steady level (task 350): the Rust side of
//! `tests/test_readiness.py`'s `TestTask116Readiness`, where `cargo mutants`
//! sees it.
//!
//! The warning projects where a target's weight tops out from the weight it
//! reads, taken for its steady level at the rows' spacing and weight so far:
//! for rows of weight 1 one row apart at a half-life of 5 rows, the ceiling
//! `1/(1 − 2^(−1/5)) = 7.7255`. A target the rows stopped carrying has no
//! such level: its weight halves every half-life, toward 0, and the warning
//! read it as a ceiling ("tops out near 2.21" five rows after the floor
//! first withheld a row, of a weight already decaying). So did a stream
//! whose rows weigh 0 from some row on. The model's shared weight, every
//! row it learned, stays at the ceiling the rows give, and tells such a
//! target from one every row carries; a row of weight 0 says no regular
//! stream of rows like it reads the weight there is.
//!
//! The rows: `y` present on rows `0..stop`, null after; half-life 5 rows,
//! no clock column, so the decay time before row `i` is `i − 1`, 95% settled
//! from row 23 (`1 − 2^(−22/5) = 0.953`), and the warning comes a half-life
//! after the first row the floor withholds from there.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// Where the warning sends its reader.
const READY: &str =
    "https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready";

const N: usize = 120;

/// `y` on rows `0..stop`, null after; every row weighs 1 before `zero_from`
/// and 0 from it.
fn frame(stop: usize, zero_from: usize) -> DataFrame {
    let x0: Vec<f64> = (0..N).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
    let y: Vec<Option<f64>> = (0..N)
        .map(|i| (i < stop).then(|| 2.0 * x0[i] + ((i * 3) % 7) as f64 * 0.1))
        .collect();
    let w: Vec<f64> = (0..N)
        .map(|i| if i < zero_from { 1.0 } else { 0.0 })
        .collect();
    df!("x0" => x0, "y" => y, "w" => w).unwrap()
}

fn spec(model: &str, floor: f64) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "{model}"}}, "targets": ["y"], "features": ["x0"],
            "half_life": 5.0, "min_weight": {floor}, "weight": "w"}}"#
    ))
    .unwrap()
}

/// The notices a bank of `spec` says over `df`, fed in chunks cut at `cuts`.
fn said(spec: Spec, df: &DataFrame, cuts: &[usize]) -> Vec<String> {
    let mut bank = Bank::new(vec![spec]).unwrap();
    let mut all = Vec::new();
    let mut from = 0;
    for &to in cuts.iter().chain([N].iter()) {
        bank.fit_predict(&df.slice(from as i64, to - from)).unwrap();
        all.extend(bank.take_notices());
        from = to;
    }
    all
}

/// The opening every form shares, for a floor of `floor`.
fn unmet(floor: &str) -> String {
    format!(
        "spec \"m\": min_weight = {floor} has not been met for target \"y\" on any row for the \
         last half-life, with the model within 5% of its steady state"
    )
}

/// The rows stopped carrying `y` at row 40, where its weight had reached
/// the ceiling of 7.73, past the floor of 5: it falls below 5 at row 43 and
/// halves every half-life after, so the floor can be met only once rows
/// carry the target again. The warning says so, with no figure but the
/// floor; it said "tops out near 2.21", and advised lowering the floor below
/// that. The shared weight stays at 7.73 throughout, for every model that
/// keeps a weight per target.
#[test]
fn a_target_the_rows_stopped_carrying_is_not_given_a_ceiling() {
    let df = frame(40, N);
    let want = format!(
        "{}. The model has not learned target \"y\" from every row, and its weight falls on each \
         row without it, so predictions for target \"y\" stay withheld until rows with a value of \
         the target bring that weight to 5. Check that rows reach the model with a value of target \
         \"y\", as {READY} explains.",
        unmet("5")
    );
    for model in ["rls", "ewridge", "sgd", "kalman"] {
        let s = if model == "kalman" {
            serde_json::from_str(
                r#"{"name": "m", "model": {"type": "kalman", "coef_half_life": 50.0},
                    "targets": ["y"], "features": ["x0"], "half_life": 5.0, "min_weight": 5.0,
                    "weight": "w"}"#,
            )
            .unwrap()
        } else {
            spec(model, 5.0)
        };
        for cuts in [&[][..], &[44, 47, 48][..]] {
            assert_eq!(
                said(s.clone(), &df, cuts),
                vec![want.clone()],
                "{model} {cuts:?}"
            );
        }
    }
}

/// The rows stopped carrying `y` at row 15, before the warning, under a
/// floor of 10 that rows carrying it on every row, topping out at 7.73,
/// would not meet either. The figure is that ceiling, said as the ceiling
/// of rows that each carry the target, with the half-life at which it
/// passes 10: `1/−log2(1 − 1/10) = 6.579`, rounded up. It said the
/// target's own decaying weight.
#[test]
fn a_target_short_of_the_rows_is_given_the_rows_ceiling() {
    let df = frame(15, N);
    let ceiling = 1.0 / (1.0 - (-1.0_f64 / 5.0).exp2());
    let figure = format!("{:.2}", (ceiling * 100.0).floor() / 100.0);
    assert_eq!(figure, "7.72");
    let want = format!(
        "{}. At the row weights seen so far, the weight the gate reads tops out near 7.72 if every \
         row carries a value of target \"y\", and the model has not learned the target from every \
         row, so predictions for target \"y\" stay withheld unless the rows carry more weight, and \
         each carries a value of the target. Lower min_weight below 7.72, or increase half_life \
         above 6.58, as {READY} explains.",
        unmet("10")
    );
    for cuts in [&[][..], &[20, 27, 28][..]] {
        assert_eq!(
            said(spec("rls", 10.0), &df, cuts),
            vec![want.clone()],
            "{cuts:?}"
        );
    }
}

/// Every row weighs 0 from row 40, so the shared weight decays with the
/// target's; the warning comes on a row of weight 0, which no regular stream
/// of rows like it reads a positive weight from, so it gives no figure. It
/// said "tops out near 2.21".
#[test]
fn a_row_of_weight_zero_is_given_no_ceiling() {
    let df = frame(N, 40);
    let want = format!(
        "{}, so predictions for target \"y\" stay withheld until the weight the gate reads \
         reaches 5. This warning comes on a row of weight 0, which adds nothing to the weight, so \
         it cannot say where the weight tops out. Check that rows reach the model with a positive \
         weight, lower min_weight, or increase half_life, as {READY} explains.",
        unmet("5")
    );
    for cuts in [&[][..], &[44, 48][..]] {
        assert_eq!(
            said(spec("rls", 5.0), &df, cuts),
            vec![want.clone()],
            "{cuts:?}"
        );
    }
}

/// A target every row carries keeps the projection it always had: at a floor
/// of 20, the ceiling of 7.72 and the half-life `1/−log2(1 − 1/20) = 13.51`,
/// rounded up.
#[test]
fn a_target_every_row_carries_keeps_its_ceiling() {
    let df = frame(N, N);
    let want = format!(
        "{}. At the row weights seen so far, the weight the gate reads tops out near 7.72, so \
         predictions for target \"y\" stay withheld unless the rows carry more weight. Lower \
         min_weight below 7.72, or increase half_life above 13.6, as {READY} explains.",
        unmet("20")
    );
    for model in ["rls", "ewridge", "sgd"] {
        assert_eq!(
            said(spec(model, 20.0), &df, &[]),
            vec![want.clone()],
            "{model}"
        );
    }
}
