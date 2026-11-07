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
//!
//! The noise gate's warning where the model has not fit at all, its
//! `weight_sum` short of a row per coefficient, projected `weight_sum` the
//! same way, and on rows of weight 0 read it falling as a ceiling too; its
//! tests are the last three (`unsolved`).

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

/// Ten features under `ewridge`'s default noise gate, at a half-life of 2
/// rows: the first fit needs a row per coefficient, 11 of weight, and
/// `weight_sum` tops out at `1/(1 − 2^(−1/2)) = 3.414`, so the gate's ratio
/// stays infinite. On a row clock the decay time before row `i` is `i − 1`,
/// 95% settled from row 10 (`1 − 2^(−9/2) = 0.956`), and the warning comes a
/// half-life later, on row 12. `w` is each row's weight.
fn unsolved(w: &[f64]) -> Vec<String> {
    let n = w.len();
    let mut cols: Vec<Column> = (0..10)
        .map(|j| {
            let x: Vec<f64> = (0..n)
                .map(|i| (((i + 3) * (2 * j + 7) + j * j) % 13) as f64 - 6.0)
                .collect();
            Column::new(format!("x{j}").into(), x)
        })
        .collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    cols.push(Column::new("y".into(), y));
    cols.push(Column::new("w".into(), w.to_vec()));
    let df = DataFrame::new(n, cols).unwrap();
    let features: Vec<String> = (0..10).map(|j| format!("\"x{j}\"")).collect();
    let spec: Spec = serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
            "features": [{}], "half_life": 2.0, "weight": "w"}}"#,
        features.join(", ")
    ))
    .unwrap();
    let mut bank = Bank::new(vec![spec]).unwrap();
    let mut all = Vec::new();
    for part in [df.slice(0, 12), df.slice(12, n - 12)] {
        bank.fit_predict(&part).unwrap();
        all.extend(bank.take_notices());
    }
    all
}

/// The noise gate's opening while the model has not fit.
const UNSOLVED: &str = "spec \"m\": max_error_inflation = 1.414 has not been met on any row for \
     the last half-life, with the model within 5% of its steady state";

/// The rows keep coming at weight 1: `weight_sum` tops out near 3.41,
/// rounded down, and the half-life that lets the model fit with a Kish size
/// that carries its 11 coefficients is 7.2725, rounded up. It said "tops
/// out near 3.4142" and "Raise the half_life above 7.272540897341715", and
/// cited the README, not the reference that explains the gates.
#[test]
fn the_unsolved_noise_gate_gives_where_weight_sum_tops_out() {
    let want = format!(
        "{UNSOLVED}. At the row weights seen so far, weight_sum tops out near 3.41, below the 11 \
         the model needs before it fits at all, one for each of its 11 coefficients, so \
         error_inflation stays infinite and predictions stay withheld unless the rows carry more \
         weight. Increase half_life above 7.28, as {READY} explains."
    );
    assert_eq!(unsolved(&[1.0; 40]), vec![want]);
}

/// Every row weighs 0 from row 3, so `weight_sum` falls toward 0 (0.0975
/// before row 12); the warning comes on a row of weight 0 and gives no
/// figure. It said "tops out near 0.0997", the falling weight over the
/// settled fraction.
#[test]
fn the_unsolved_noise_gate_on_a_row_of_weight_zero_gives_no_figure() {
    let w: Vec<f64> = (0..40).map(|i| if i < 3 { 1.0 } else { 0.0 }).collect();
    let want = format!(
        "{UNSOLVED}, so error_inflation stays infinite and predictions stay withheld until \
         weight_sum reaches the 11 the model needs before it fits at all, one for each of its 11 \
         coefficients. This warning comes on a row of weight 0, which adds nothing to the weight, \
         so it cannot say where weight_sum tops out. Check that rows reach the model with a \
         positive weight, or increase half_life, as {READY} explains."
    );
    assert_eq!(unsolved(&w), vec![want]);
}

/// A row of weight 1,000 where the warning comes: read as the weight every
/// row brings, it projects `weight_sum` at `(3.34 − 1000·2^(−11/2))/s`, near
/// −19, so there is no figure. Any weight past about 150 would do.
///
/// It was 1e90, and on Windows a second notice came beside this one, that
/// a coefficient was all ridge. A row that heavy is the whole of the
/// model's state to the bit (each later row's share of the mean-form
/// update, `w/(λW + w)`, is about 1e-90): the fit after it is one
/// observation, its Kish size 1 and its `edf` the intercept's 1, so the
/// gate's ratio `sqrt(1 + edf/n_kish)` is the threshold `sqrt(2)` but for
/// rounding, which the decay factors' last bits decide. Where it rounded
/// below, the gate let a row through and the notice on its all-ridge
/// slopes fired (on this platform every row after the heavy one passes a
/// threshold of 1.5). At 1,000 the heavy row's weight halves every two rows
/// and the others' rows are seen: no row passes a threshold of 2.1, half
/// again the gate's, so no platform's rounding lets one through, and this
/// notice is the only one.
#[test]
fn the_unsolved_noise_gate_on_uneven_rows_gives_no_figure() {
    let w: Vec<f64> = (0..40).map(|i| if i == 12 { 1e3 } else { 1.0 }).collect();
    let want = format!(
        "{UNSOLVED}, so error_inflation stays infinite and predictions stay withheld until \
         weight_sum reaches the 11 the model needs before it fits at all, one for each of its 11 \
         coefficients. The rows so far are too uneven (such as one row far heavier than the \
         rest) to project where weight_sum tops out. Increase half_life, as {READY} explains."
    );
    assert_eq!(unsolved(&w), vec![want]);
}
