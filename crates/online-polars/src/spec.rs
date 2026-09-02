//! Model bank specs (docs/PLAN.md §3, §5): serde-deserializable from JSON
//! (Python) and TOML (CLI), with common-parameter validation and the output
//! struct layout.

use online_core::{ClockCfg, Decay, OnClockReset, SessionGap};
use serde::{Deserialize, Serialize};
use std::fmt;

fn default_true() -> bool {
    true
}

/// A float that also accepts the JSON strings `"inf"` / `"-inf"`, since JSON
/// has no infinity literal and `halflife = inf` is meaningful (it pins a
/// coefficient, docs/PLAN.md §4.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num(pub f64);

/// A number rendered for a **field name** (`__r{ridge}`, `@h{halflife}`,
/// `absresid_q{level}`, `__l{lambda}`).
///
/// Rust's `Display` for `f64` never uses scientific notation, so a perfectly
/// legal `ridge = 1e-300` produced a **311-character field name** (three
/// hundred zeros). Values outside [1e-6, 1e7) render as compact scientific
/// (`1e-300`, `2.5e8`) instead; everything inside renders exactly as before,
/// so no existing name changes.
///
/// These strings are **public API** — users index the output struct by them —
/// and this function is deliberately the only place the rendering lives.
/// `tests/test_api_surface.py` pins the results, so a change here (or a change
/// in rustc's float formatting, which has happened historically) fails a test
/// instead of silently renaming users' columns.
#[cfg(test)]
mod num_label_tests {
    use super::num_label;

    #[test]
    fn ordinary_values_render_exactly_as_before() {
        // The compact form must not change any name the suite already pins.
        assert_eq!(num_label(1e-6), "0.000001");
        assert_eq!(num_label(1e-7 * 10.0), "0.000001");
        assert_eq!(num_label(0.1), "0.1");
        assert_eq!(num_label(0.5), "0.5");
        assert_eq!(num_label(100.0), "100");
        assert_eq!(num_label(250.5), "250.5");
        assert_eq!(num_label(3200.0), "3200");
        assert_eq!(num_label(0.0), "0");
    }

    #[test]
    fn extreme_values_render_compactly() {
        // The bug this exists for: 1e-300 was a 311-character field name.
        assert_eq!(num_label(1e-300), "1e-300");
        assert_eq!(num_label(2.5e8), "2.5e8");
        assert_eq!(num_label(1e7), "1e7");
        assert_eq!(num_label(1e9), "1e9");
        assert_eq!(num_label(-1e-300), "-1e-300");
        assert!(num_label(1e-300).len() < 10);
    }

    #[test]
    fn the_thresholds_are_where_the_doc_says() {
        // Just inside: plain. Just outside: scientific.
        assert_eq!(num_label(1e-6), "0.000001");
        assert_eq!(num_label(9.999e-7), "9.999e-7");
        assert_eq!(num_label(9_999_999.0), "9999999");
        assert_eq!(num_label(1e7), "1e7");
    }
}

pub fn num_label(v: f64) -> String {
    let a = v.abs();
    if v != 0.0 && v.is_finite() && !(1e-6..1e7).contains(&a) {
        // `{:e}` gives `1e-300` / `2.5e8`; normalize the `e0` exponent Rust
        // emits for values that just crossed the threshold.
        let s = format!("{v:e}");
        match s.strip_suffix("e0") {
            Some(t) => t.to_string(),
            None => s,
        }
    } else {
        format!("{v}")
    }
}

/// `v > 0` as a named predicate, so that `!positive(v)` refuses NaN too (a
/// plain `v <= 0.0` lets it through, and NaN in any of these parameters is
/// a state that never washes out).
fn positive(v: f64) -> bool {
    v > 0.0
}

fn non_negative(v: f64) -> bool {
    v >= 0.0
}

/// The first value that appears twice in a grid, if any. Grid entries become
/// field-name suffixes, so a repeated value is always a mistake.
fn first_duplicate(vals: &[f64]) -> Option<f64> {
    vals.iter()
        .enumerate()
        .find(|(i, v)| vals[..*i].iter().any(|u| u.to_bits() == v.to_bits()))
        .map(|(_, v)| *v)
}

/// `ridge` for the IRLS models: finite and non-negative (zero is plain least
/// squares).
fn check_ridge(name: &str, ridge: Option<f64>) -> Result<(), String> {
    if ridge.is_some_and(|r| !non_negative(r) || !r.is_finite()) {
        return Err(format!("spec {name:?}: ridge must be finite and >= 0"));
    }
    Ok(())
}

/// `solve_every`: clock units between solves; zero solves every row. A
/// negative value silently meant "every row" too, NaN meant "never".
fn check_solve_every(name: &str, v: Option<f64>) -> Result<(), String> {
    if v.is_some_and(|v| !non_negative(v) || !v.is_finite()) {
        return Err(format!(
            "spec {name:?}: solve_every must be finite and >= 0 (0 solves every row)"
        ));
    }
    Ok(())
}

impl Serialize for Num {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            s.serialize_f64(self.0)
        } else if self.0 == f64::INFINITY {
            s.serialize_str("inf")
        } else if self.0 == f64::NEG_INFINITY {
            s.serialize_str("-inf")
        } else {
            s.serialize_str("nan")
        }
    }
}

impl Num {
    /// The spellings of the non-finite values accepted in place of a number.
    fn from_word(w: &str) -> Option<Num> {
        match w.to_ascii_lowercase().as_str() {
            "inf" | "+inf" | "infinity" | "+infinity" => Some(Num(f64::INFINITY)),
            "-inf" | "-infinity" => Some(Num(f64::NEG_INFINITY)),
            "nan" => Some(Num(f64::NAN)),
            _ => None,
        }
    }
}

// Hand-written visitors rather than `#[serde(untagged)]`: an untagged enum
// that matches nothing reports "data did not match any variant of untagged
// enum FloatOrList", which names a Rust type and not what was expected. A
// visitor says `invalid type: string "10", expected a number or a list of
// numbers ("inf" allowed)`, and does so for JSON, TOML and the msgpack state
// file alike (all three are self-describing, so `deserialize_any` is exactly
// what the untagged form used underneath).

struct NumVisitor;

impl serde::de::Visitor<'_> for NumVisitor {
    type Value = Num;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a number or \"inf\"/\"-inf\"")
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Num, E> {
        Ok(Num(v))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Num, E> {
        Ok(Num(v as f64))
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Num, E> {
        Ok(Num(v as f64))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Num, E> {
        Num::from_word(v).ok_or_else(|| E::invalid_value(serde::de::Unexpected::Str(v), &self))
    }
}

impl<'de> Deserialize<'de> for Num {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(NumVisitor)
    }
}

/// A float or a list of floats (grids; `halflife` lists mean one accumulator
/// per value, docs/PLAN.md §4.1).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FloatOrList {
    Float(Num),
    List(Vec<Num>),
}

impl<'de> Deserialize<'de> for FloatOrList {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = FloatOrList;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a number or a list of numbers (\"inf\" allowed)")
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v)))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v as f64)))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v as f64)))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<FloatOrList, E> {
                Num::from_word(v)
                    .map(FloatOrList::Float)
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Str(v), &self))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                seq: A,
            ) -> Result<FloatOrList, A::Error> {
                Vec::<Num>::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
                    .map(FloatOrList::List)
            }
        }

        d.deserialize_any(V)
    }
}

impl FloatOrList {
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            FloatOrList::Float(f) => vec![f.0],
            FloatOrList::List(v) => v.iter().map(|n| n.0).collect(),
        }
    }
}

/// `session_gap`: clock units, or "reset". The gap is a [`Num`] so that an
/// infinite one ("never") survives JSON, which has no infinity literal.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SessionGapSpec {
    Gap(Num),
    Word(String),
}

impl<'de> Deserialize<'de> for SessionGapSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl serde::de::Visitor<'_> for V {
            type Value = SessionGapSpec;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a gap in clock units (\"inf\" for never) or the word \"reset\"")
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Num(v)))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Num(v as f64)))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Num(v as f64)))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<SessionGapSpec, E> {
                if v == "reset" {
                    Ok(SessionGapSpec::Word(v.to_string()))
                } else if let Some(n) = Num::from_word(v) {
                    Ok(SessionGapSpec::Gap(n))
                } else {
                    Err(E::invalid_value(serde::de::Unexpected::Str(v), &self))
                }
            }
        }

        d.deserialize_any(V)
    }
}

/// Model choice + model-specific params (docs/PLAN.md §4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelKind {
    EwRidge {
        #[serde(default)]
        ridge: Option<FloatOrList>,
        #[serde(default)]
        feature_sets: Option<Vec<(String, Vec<String>)>>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        ridge_decay: bool,
        /// Shrink toward these coefficients instead of toward zero, one vector
        /// per target of length `n_features + intercept`, in original units.
        #[serde(default)]
        coef0: Option<Vec<Vec<f64>>>,
        /// On a session change, mix the accumulators this far toward a
        /// slow-moving twin: 0 keeps today's fit, 1 reverts to the long run.
        /// Needs `long_halflife`.
        #[serde(default)]
        session_shrink: Option<f64>,
        /// Halflife of that twin.
        #[serde(default)]
        long_halflife: Option<f64>,
        #[serde(default)]
        solve_every: Option<f64>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
    },
    Lasso {
        /// Decreasing penalties on standardized stats; required.
        lasso_path: Vec<f64>,
        /// 1.0 = lasso, < 1.0 = elastic net.
        #[serde(default)]
        l1_ratio: Option<f64>,
        /// Halflife of the EW squared error used to select lambda.
        #[serde(default)]
        select_halflife: Option<f64>,
        #[serde(default)]
        solve_every: Option<f64>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        #[serde(default)]
        max_cd_iters: Option<u32>,
        #[serde(default)]
        cd_tol: Option<f64>,
    },
    Kalman {
        /// Per-factor coefficient halflife (scalar or one per slot, intercept
        /// first). `inf` pins a coefficient. Note this is the COEFFICIENT
        /// halflife; the spec-level `halflife` drives the standardization and
        /// residual-variance statistics.
        coef_halflife: FloatOrList,
        #[serde(default)]
        q: Option<Vec<Num>>,
        #[serde(default)]
        obs_var: Option<f64>,
        #[serde(default)]
        p0: Option<f64>,
        #[serde(default)]
        share_p: bool,
        /// Standardize features internally (default true). Off makes the filter
        /// a plain Bayesian linear regression on the features' own scale.
        #[serde(default = "default_true")]
        standardize: bool,
    },
    /// Huber regression (docs/PLAN.md §4.5).
    Huber {
        /// Cut point in units of the EW residual std. Default 1.5 [validate].
        #[serde(default)]
        huber_delta: Option<f64>,
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        solve_every: Option<f64>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
    },
    /// Quantile regression at level `quantile` (docs/PLAN.md §4.5).
    Quantile {
        quantile: f64,
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        solve_every: Option<f64>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        #[serde(default)]
        quantile_eps: Option<f64>,
    },
    /// Online logistic regression via FTRL-proximal (docs/PLAN.md §4.6).
    /// `pred` is a probability; `resid = y - p`.
    Ftrl {
        #[serde(default)]
        alpha: Option<f64>,
        #[serde(default)]
        beta: Option<f64>,
        #[serde(default)]
        l1: Option<f64>,
        #[serde(default)]
        l2: Option<f64>,
        /// Error on targets that are not 0/1 rather than clamping them.
        /// Logistic loss only.
        #[serde(default)]
        strict_binary: bool,
        /// "logistic" (default, binary targets, `pred` is a probability) or
        /// "squared" (continuous targets, sparse linear regression).
        #[serde(default)]
        loss: Option<String>,
    },
    /// EW moments of the feature columns, no regression (docs/PLAN.md §4.7).
    /// `targets` is ignored; every column of interest goes in `features`.
    EwCov {
        /// Any of "mean", "var", "std", "cov", "corr", "partial_corr".
        /// Default: mean + std + corr.
        #[serde(default)]
        stats: Option<Vec<String>>,
        /// Prior for the precision matrix, required by "partial_corr".
        #[serde(default)]
        precision_prior: Option<f64>,
    },
    /// Stochastic gradient descent with pluggable losses (ENHANCEMENTS E16).
    /// O(k) per row, no solves, and the only model here that takes count
    /// targets (via `loss = "poisson"`).
    Sgd {
        /// "squared" (default), "huber", "quantile", "epsilon_insensitive",
        /// "poisson" or "logistic".
        #[serde(default)]
        loss: Option<String>,
        /// Huber cut, in target units.
        #[serde(default)]
        huber_delta: Option<f64>,
        #[serde(default)]
        quantile: Option<f64>,
        /// Width of the insensitive tube.
        #[serde(default)]
        eps: Option<f64>,
        #[serde(default)]
        learning_rate: Option<f64>,
        /// "constant" (default), "inv_scaling" or "adagrad".
        #[serde(default)]
        schedule: Option<String>,
        /// Exponent for `inv_scaling`.
        #[serde(default)]
        power: Option<f64>,
        #[serde(default)]
        l2: Option<f64>,
        #[serde(default)]
        /// `Num`, not `f64`: the documented way to disable clipping is
        /// `inf`, and JSON cannot carry Infinity as a number -- `Num` accepts
        /// the string "inf", exactly as the halflife fields do.
        clip_gradient: Option<Num>,
        /// Standardize features against their running moments before the
        /// gradient step, unscaling the coefficients on the way out.
        #[serde(default)]
        scale_features: bool,
    },
    /// Passive-aggressive regression (ENHANCEMENTS E17). No learning rate:
    /// each row's update is the smallest change that satisfies it.
    Pa {
        /// "pa1" (default), "pa" (unbounded) or "pa2" (damped).
        #[serde(default)]
        mode: Option<String>,
        /// Aggressiveness cap. Ignored by "pa".
        #[serde(default)]
        c: Option<f64>,
        /// Insensitive tube: rows already this close leave the fit alone.
        #[serde(default)]
        eps: Option<f64>,
    },
    /// Holt's linear trend method (ENHANCEMENTS E25): level plus slope, no
    /// features. The baseline a feature-based model should have to beat.
    Holt {
        /// Halflife of the level, in clock units. Defaults to the spec's own
        /// `halflife`.
        #[serde(default)]
        level_halflife: Option<f64>,
        /// Halflife of the trend; `"inf"` pins it, giving a plain EW level.
        /// Defaults to four times the level halflife.
        #[serde(default)]
        trend_halflife: Option<Num>,
    },
    Rls {
        /// Prior strength: `A0 = ridge I`, i.e. `P0 = I / ridge`. Scalar only
        /// (baked into the state).
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        coef0: Option<Vec<Vec<f64>>>,
    },
}

impl ModelKind {
    pub fn kind_name(&self) -> &'static str {
        match self {
            ModelKind::EwRidge { .. } => "ew_ridge",
            ModelKind::Rls { .. } => "rls",
            ModelKind::Lasso { .. } => "lasso",
            ModelKind::Kalman { .. } => "kalman",
            ModelKind::Huber { .. } => "huber",
            ModelKind::Quantile { .. } => "quantile",
            ModelKind::Ftrl { .. } => "ftrl",
            ModelKind::EwCov { .. } => "ew_cov",
            ModelKind::Sgd { .. } => "sgd",
            ModelKind::Pa { .. } => "pa",
            ModelKind::Holt { .. } => "holt",
        }
    }
}

/// One model spec: common parameters (docs/PLAN.md §3) + the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spec {
    /// Output struct column name.
    pub name: String,
    pub model: ModelKind,
    pub targets: Vec<String>,
    pub features: Vec<String>,
    #[serde(default = "default_true")]
    pub add_intercept: bool,
    #[serde(default)]
    pub clock: Option<String>,
    #[serde(default)]
    pub halflife: Option<FloatOrList>,
    #[serde(default)]
    pub lam: Option<f64>,
    /// Ceiling on the clock delta, in clock units; `"inf"` for none.
    #[serde(default)]
    pub max_dclock: Option<Num>,
    #[serde(default)]
    pub on_clock_reset: OnClockReset,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub session_gap: Option<SessionGapSpec>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    /// Warmup in `n_eff` units. A scalar applies to every target; a list gives
    /// one threshold per target, in `targets` order (ENHANCEMENTS E7) — a
    /// 5-minute-ahead target and a 1-day-ahead target rarely deserve the same
    /// warmup. Warmup gates *output*, not learning: the model still updates
    /// from rows whose predictions are withheld.
    pub min_periods: Option<FloatOrList>,
    /// 0 = never; coefficients are also emitted on the last row of every chunk.
    #[serde(default)]
    pub coef_every: u32,
    /// Emit `sigma_<slot>`: the EW standard deviation of this slot's
    /// out-of-sample residuals, read from the state *before* each row. Off by
    /// default because it widens the output struct.
    #[serde(default)]
    pub emit_sigma: bool,
    /// Emit `resid_z_<slot>` = `resid / sigma`: how surprising this row was, in
    /// units of the model's own recent error. Off by default.
    #[serde(default)]
    pub emit_resid_z: bool,
    /// Emit `ic_<slot>`, `r2_<slot>` and `hit_rate_<slot>`: exponentially
    /// weighted evaluation metrics kept beside the model (ENHANCEMENTS E22).
    /// `polars_online.eval` computes the same things in Polars over collected
    /// output, which needs the whole frame; this is the O(state) version, so a
    /// long-running stream or the CLI can report how the fit is doing without
    /// keeping the rows.
    #[serde(default)]
    pub emit_metrics: bool,
    /// Emit `absresid_q<p>_<slot>` for each level in `resid_quantiles`: a P²
    /// estimate of that quantile of `|resid|` (ENHANCEMENTS E23). Five numbers
    /// per level, no window — a distribution-free interval where `sigma` only
    /// gives a Gaussian one.
    #[serde(default)]
    pub resid_quantiles: Option<Vec<f64>>,
    /// Emit `autocorr_<slot>`: EW lag-`resid_autocorr_lag` autocorrelation of
    /// the out-of-sample residuals. A residual stream should look like noise;
    /// autocorrelation is the classic sign that it does not.
    #[serde(default)]
    pub emit_autocorr: bool,
    /// Lag for `emit_autocorr`. Default 1.
    #[serde(default)]
    pub resid_autocorr_lag: Option<usize>,
    /// Emit `drift_<slot>`: a Page-Hinkley detector on each slot's absolute
    /// out-of-sample residual, true on the row where a break is detected.
    /// Complements the halflife: decay forgets smoothly and always, drift
    /// detection notices a break and says so.
    #[serde(default)]
    pub emit_drift: bool,
    /// Change magnitude the drift detector tolerates before accumulating,
    /// in units of the slot's own EW residual std. Default 0.5.
    #[serde(default)]
    pub drift_delta: Option<f64>,
    /// Accumulated excess that counts as drift. Default 20.
    #[serde(default)]
    pub drift_threshold: Option<f64>,
    /// What a detection does besides setting the flag: `"flag"` (default) or
    /// `"reset"`, which restarts this stream's models the way a clock reset
    /// does.
    #[serde(default)]
    pub drift_action: Option<String>,
    /// Emit `pred_<target>__averaged`: an exponentially weighted average of
    /// every slot's prediction, with weights `softmax(−eta · EW squared
    /// error)` (ENHANCEMENTS E14). The soft counterpart of `emit_selected`:
    /// averaging hedges where selection commits, which is usually the better
    /// trade when several slots are close.
    #[serde(default)]
    pub emit_averaged: bool,
    /// Sharpness of the averaging weights. Large values approach
    /// `emit_selected`'s argmin; small values approach an equal-weight mean.
    /// Default 1.
    #[serde(default)]
    pub average_eta: Option<f64>,
    /// Emit `selected_<target>` and `pred_<target>__selected`: online model
    /// selection across every grid slot for that target (ridge values, feature
    /// sets and halflives), by lowest EW out-of-sample error. Generalizes the
    /// lasso's `lam_selected`. Requires more than one slot per target.
    #[serde(default)]
    pub emit_selected: bool,
    /// ModelBank/CLI only; one state per key. The expression API uses `.over()`.
    #[serde(default)]
    pub group: Option<String>,
}

impl Spec {
    pub fn k(&self) -> usize {
        self.features.len()
    }

    pub fn m(&self) -> usize {
        self.targets.len()
    }

    /// Decay values, one per halflife grid entry (one model instance each).
    pub fn decays(&self) -> Result<Vec<(String, Decay)>, String> {
        match (&self.halflife, self.lam) {
            (Some(_), Some(_)) => Err(format!(
                "spec {:?}: halflife and lam are mutually exclusive",
                self.name
            )),
            (None, None) => Err(format!(
                "spec {:?}: one of halflife/lam is required",
                self.name
            )),
            (None, Some(l)) => {
                if !(0.0 < l && l <= 1.0) {
                    return Err(format!("spec {:?}: lam must be in (0, 1]", self.name));
                }
                Ok(vec![(String::new(), Decay::Lam(l))])
            }
            (Some(h), None) => {
                let hs = h.to_vec();
                // `!(h > 0)` rather than `h <= 0` so NaN is refused too: it
                // decays every accumulator to NaN and nothing washes it out.
                if hs.iter().any(|&h| !positive(h)) {
                    return Err(format!(
                        "spec {:?}: halflife must be > 0 (\"inf\" for no decay)",
                        self.name
                    ));
                }
                if let Some(dup) = first_duplicate(&hs) {
                    return Err(format!(
                        "spec {:?}: halflife lists {} more than once; each value is one \
                         model instance and the two would produce the same field names",
                        self.name,
                        num_label(dup)
                    ));
                }
                if hs.len() == 1 {
                    Ok(vec![(String::new(), Decay::Halflife(hs[0]))])
                } else {
                    Ok(hs
                        .iter()
                        .map(|&h| (format!("@h{}", num_label(h)), Decay::Halflife(h)))
                        .collect())
                }
            }
        }
    }

    pub fn clock_cfg(&self) -> Result<ClockCfg, String> {
        if self.clock.is_some() && self.max_dclock.is_none() {
            return Err(format!(
                "spec {:?}: max_dclock is required when clock is given",
                self.name
            ));
        }
        // A negative ceiling clips every delta to it, so the decay *grows*
        // (`n_eff` ran to 6e7 on 50 rows with `max_dclock = -5`); NaN poisons
        // the clock. Zero freezes it (a documented way to switch decay off)
        // and infinity removes the ceiling; both are legitimate.
        if self.max_dclock.is_some_and(|m| !non_negative(m.0)) {
            return Err(format!(
                "spec {:?}: max_dclock must be >= 0 (0 disables decay, \"inf\" removes the ceiling)",
                self.name
            ));
        }
        let session_gap = match &self.session_gap {
            None => None,
            Some(SessionGapSpec::Gap(g)) if !non_negative(g.0) => {
                return Err(format!(
                    "spec {:?}: session_gap must be >= 0 or \"reset\"",
                    self.name
                ));
            }
            Some(SessionGapSpec::Gap(g)) => Some(SessionGap::Gap(g.0)),
            Some(SessionGapSpec::Word(w)) if w == "reset" => Some(SessionGap::Reset),
            Some(SessionGapSpec::Word(w)) => {
                return Err(format!(
                    "spec {:?}: session_gap must be a number or \"reset\", got {w:?}",
                    self.name
                ));
            }
        };
        if self.session.is_some() && session_gap.is_none() {
            return Err(format!(
                "spec {:?}: session_gap is required when session is given",
                self.name
            ));
        }
        Ok(ClockCfg {
            max_dclock: self.max_dclock.map_or(f64::INFINITY, |m| m.0),
            on_clock_reset: self.on_clock_reset,
            session_gap,
        })
    }

    /// Threshold per target, in `targets` order.
    pub fn min_periods_per_target(&self) -> Vec<f64> {
        match &self.min_periods {
            None => vec![self.default_min_periods(); self.m()],
            Some(FloatOrList::Float(v)) => vec![v.0; self.m()],
            Some(FloatOrList::List(v)) => v.iter().map(|n| n.0).collect(),
        }
    }

    fn default_min_periods(&self) -> f64 {
        (self.k() + usize::from(self.add_intercept)) as f64
    }

    /// The threshold the *model* uses: the smallest across targets, so a model
    /// starts predicting as soon as any target is ready. Per-target gating of
    /// the reported values happens in the stream layer.
    pub fn min_periods_or_default(&self) -> f64 {
        self.min_periods_per_target()
            .into_iter()
            .fold(f64::INFINITY, f64::min)
    }

    /// Default solve cadence: halflife/50 (docs/PLAN.md §4.1, [validate]).
    pub fn solve_every_default(&self, decay: Decay) -> f64 {
        match decay {
            Decay::Halflife(h) if h.is_finite() => h / 50.0,
            _ => 0.0, // lam decay / infinite halflife: solve every row
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.targets.is_empty() {
            return Err(format!("spec {:?}: targets must be non-empty", self.name));
        }
        // Holt has no features by construction; every other model needs at
        // least one to regress on. Both directions are errors: silently
        // ignoring features passed to Holt would look like they were used.
        if matches!(self.model, ModelKind::Holt { .. }) {
            if !self.features.is_empty() {
                return Err(format!(
                    "spec {:?}: holt takes no features (got {}); it extrapolates the \
                     target's own level and trend. Use a regression model to use features.",
                    self.name,
                    self.features.len()
                ));
            }
        } else if self.features.is_empty() {
            return Err(format!("spec {:?}: features must be non-empty", self.name));
        }
        // A duplicated column silently splits its coefficient across identical
        // slots on an exactly singular system (the jitter fallback rescues the
        // solve, so nothing else complains), and a duplicated target collides
        // in the output struct. Both are always mistakes.
        for (label, cols) in [("features", &self.features), ("targets", &self.targets)] {
            let mut seen = std::collections::HashSet::new();
            if let Some(dup) = cols.iter().find(|c| !seen.insert(c.as_str())) {
                return Err(format!(
                    "spec {:?}: {label} lists {dup:?} more than once",
                    self.name
                ));
            }
        }
        // A target used as a feature reads the *current row's* target to
        // predict that same row: perfect leakage, measured as corr(pred, y)
        // = 1.0. Hard rule 2 (out-of-sample by construction) protects the
        // target as target; this is the other door, and it must be locked --
        // it is exactly the accident a long column list invites, and the
        // resulting backtest looks wonderful right up until deployment.
        //
        // `ew_cov` is exempt *by design*, not oversight: it predicts nothing.
        // Its "targets" mirror its columns for plumbing, and its statistics
        // are read from the state BEFORE each row, which is what makes an
        // ew_cov output safe to use as a same-row feature (E1).
        let is_ew_cov = matches!(self.model, ModelKind::EwCov { .. });
        if let Some(leak) = (!is_ew_cov)
            .then(|| self.features.iter().find(|f| self.targets.contains(f)))
            .flatten()
        {
            return Err(format!(
                "spec {:?}: {leak:?} is both a target and a feature; a feature is read \
                 from the current row, so this would predict the target with itself \
                 (use a lagged copy of the column if you mean its past values)",
                self.name
            ));
        }
        self.decays()?;
        self.clock_cfg()?;
        let mp = self.min_periods_per_target();
        if mp.len() != self.m() {
            return Err(format!(
                "spec {:?}: min_periods list has {} entries but there are {} targets",
                self.name,
                mp.len(),
                self.m()
            ));
        }
        if mp.iter().any(|v| *v < 0.0 || v.is_nan()) {
            return Err(format!("spec {:?}: min_periods must be >= 0", self.name));
        }
        if let Some(a) = &self.drift_action {
            if !["flag", "reset"].contains(&a.as_str()) {
                return Err(format!(
                    "spec {:?}: drift_action must be \"flag\" or \"reset\"",
                    self.name
                ));
            }
        }
        if self.drift_delta.is_some_and(|v| v < 0.0 || v.is_nan()) {
            return Err(format!("spec {:?}: drift_delta must be >= 0", self.name));
        }
        if self.drift_threshold.is_some_and(|v| v <= 0.0 || v.is_nan()) {
            return Err(format!("spec {:?}: drift_threshold must be > 0", self.name));
        }
        if let Some(qs) = &self.resid_quantiles {
            if qs.is_empty() {
                return Err(format!(
                    "spec {:?}: resid_quantiles must be non-empty",
                    self.name
                ));
            }
            if qs
                .iter()
                .any(|q| !(0.0..=1.0).contains(q) || *q == 0.0 || *q == 1.0)
            {
                return Err(format!(
                    "spec {:?}: resid_quantiles must be strictly between 0 and 1",
                    self.name
                ));
            }
        }
        if self.resid_autocorr_lag.is_some_and(|l| l == 0) {
            return Err(format!(
                "spec {:?}: resid_autocorr_lag must be >= 1",
                self.name
            ));
        }
        if self.average_eta.is_some_and(|v| v <= 0.0 || v.is_nan()) {
            return Err(format!("spec {:?}: average_eta must be > 0", self.name));
        }
        if self.emit_averaged && matches!(self.model, ModelKind::EwCov { .. }) {
            return Err(format!(
                "spec {:?}: emit_averaged does not apply to ew_cov (it has no predictions)",
                self.name
            ));
        }
        if self.emit_selected {
            if matches!(self.model, ModelKind::EwCov { .. }) {
                return Err(format!(
                    "spec {:?}: emit_selected does not apply to ew_cov (it has no predictions)",
                    self.name
                ));
            }
            let n_slots = self.decays()?.len() * crate::combo_labels(self).len();
            if n_slots < 2 {
                return Err(format!(
                    "spec {:?}: emit_selected needs more than one slot per target; add a \
                     ridge/feature_set/halflife grid or a lasso path",
                    self.name
                ));
            }
        }
        match &self.model {
            ModelKind::Holt {
                level_halflife,
                trend_halflife,
            } => {
                if level_halflife.is_some_and(|h| h <= 0.0 || h.is_nan()) {
                    return Err(format!("spec {:?}: level_halflife must be > 0", self.name));
                }
                if trend_halflife.is_some_and(|h| h.0 <= 0.0 || h.0.is_nan()) {
                    return Err(format!(
                        "spec {:?}: trend_halflife must be > 0 (\"inf\" pins the trend)",
                        self.name
                    ));
                }
            }
            ModelKind::Pa { mode, c, eps } => {
                if let Some(md) = mode {
                    if !["pa", "pa1", "pa2"].contains(&md.as_str()) {
                        return Err(format!(
                            "spec {:?}: unknown pa mode {md:?}; expected pa, pa1 or pa2",
                            self.name
                        ));
                    }
                }
                if c.is_some_and(|v| v <= 0.0 || v.is_nan()) {
                    return Err(format!("spec {:?}: pa c must be > 0", self.name));
                }
                if eps.is_some_and(|v| v < 0.0 || v.is_nan()) {
                    return Err(format!("spec {:?}: pa eps must be >= 0", self.name));
                }
            }
            ModelKind::Sgd {
                loss,
                quantile,
                learning_rate,
                schedule,
                ..
            } => {
                if let Some(l) = loss {
                    const OK: [&str; 6] = [
                        "squared",
                        "huber",
                        "quantile",
                        "epsilon_insensitive",
                        "poisson",
                        "logistic",
                    ];
                    if !OK.contains(&l.as_str()) {
                        return Err(format!(
                            "spec {:?}: unknown sgd loss {l:?}; expected one of {}",
                            self.name,
                            OK.join(", ")
                        ));
                    }
                    if l == "quantile" && quantile.is_none() {
                        return Err(format!(
                            "spec {:?}: sgd loss \"quantile\" needs a `quantile` level",
                            self.name
                        ));
                    }
                }
                if let Some(sc) = schedule {
                    if !["constant", "inv_scaling", "adagrad"].contains(&sc.as_str()) {
                        return Err(format!(
                            "spec {:?}: unknown sgd schedule {sc:?}; expected constant, \
                             inv_scaling or adagrad",
                            self.name
                        ));
                    }
                }
                if learning_rate.is_some_and(|v| v <= 0.0 || v.is_nan()) {
                    return Err(format!("spec {:?}: learning_rate must be > 0", self.name));
                }
            }
            ModelKind::EwCov {
                stats,
                precision_prior,
            } => {
                const OK: [&str; 6] = ["mean", "var", "std", "cov", "corr", "partial_corr"];
                if let Some(stats) = stats {
                    for st in stats {
                        if !OK.contains(&st.as_str()) {
                            return Err(format!(
                                "spec {:?}: unknown ew_cov statistic {st:?}; expected one of {}",
                                self.name,
                                OK.join(", ")
                            ));
                        }
                    }
                    let pairwise =
                        |st: &String| st == "cov" || st == "corr" || st == "partial_corr";
                    if self.k() < 2 && stats.iter().any(pairwise) {
                        return Err(format!(
                            "spec {:?}: ew_cov cov/corr/partial_corr need at least two features",
                            self.name
                        ));
                    }
                    if stats.iter().any(|st| st == "partial_corr") && precision_prior.is_none() {
                        return Err(format!(
                            "spec {:?}: ew_cov partial_corr needs `precision_prior`",
                            self.name
                        ));
                    }
                }
                if precision_prior.is_some_and(|p| p <= 0.0 || !p.is_finite()) {
                    return Err(format!(
                        "spec {:?}: precision_prior must be finite and > 0",
                        self.name
                    ));
                }
            }
            ModelKind::Ftrl {
                alpha,
                beta,
                l1,
                l2,
                ..
            } => {
                if alpha.is_some_and(|a| a <= 0.0 || a.is_nan()) {
                    return Err(format!("spec {:?}: ftrl alpha must be > 0", self.name));
                }
                for (name, v) in [("beta", beta), ("l1", l1), ("l2", l2)] {
                    if v.is_some_and(|v| v < 0.0 || v.is_nan()) {
                        return Err(format!("spec {:?}: ftrl {name} must be >= 0", self.name));
                    }
                }
            }
            ModelKind::Huber {
                huber_delta,
                ridge,
                solve_every,
                ..
            } => {
                if huber_delta.is_some_and(|d| !positive(d)) {
                    return Err(format!("spec {:?}: huber_delta must be > 0", self.name));
                }
                check_ridge(&self.name, *ridge)?;
                check_solve_every(&self.name, *solve_every)?;
            }
            ModelKind::Quantile {
                quantile,
                quantile_eps,
                ridge,
                solve_every,
                ..
            } => {
                if !(0.0 < *quantile && *quantile < 1.0) {
                    return Err(format!("spec {:?}: quantile must be in (0, 1)", self.name));
                }
                if quantile_eps.is_some_and(|e| !positive(e)) {
                    return Err(format!("spec {:?}: quantile_eps must be > 0", self.name));
                }
                check_ridge(&self.name, *ridge)?;
                check_solve_every(&self.name, *solve_every)?;
            }
            ModelKind::Kalman {
                coef_halflife,
                q,
                obs_var,
                p0,
                ..
            } => {
                let k_total = self.k() + usize::from(self.add_intercept);
                let hs = coef_halflife.to_vec();
                if hs.len() != 1 && hs.len() != k_total {
                    return Err(format!(
                        "spec {:?}: coef_halflife must be scalar or length {k_total}",
                        self.name
                    ));
                }
                if hs.iter().any(|&h| !positive(h)) {
                    return Err(format!(
                        "spec {:?}: coef_halflife must be > 0 (\"inf\" pins a coefficient)",
                        self.name
                    ));
                }
                if q.as_ref().is_some_and(|q| q.len() != k_total) {
                    return Err(format!(
                        "spec {:?}: q must have length {k_total}",
                        self.name
                    ));
                }
                if q.as_ref()
                    .is_some_and(|q| q.iter().any(|v| !non_negative(v.0) || !v.0.is_finite()))
                {
                    return Err(format!(
                        "spec {:?}: q values must be finite and >= 0 (0 pins a coefficient)",
                        self.name
                    ));
                }
                if obs_var.is_some_and(|v| !positive(v) || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: obs_var must be finite and > 0",
                        self.name
                    ));
                }
                if p0.is_some_and(|v| !positive(v) || !v.is_finite()) {
                    return Err(format!("spec {:?}: p0 must be finite and > 0", self.name));
                }
            }
            ModelKind::Lasso {
                lasso_path,
                l1_ratio,
                select_halflife,
                solve_every,
                cd_tol,
                ..
            } => {
                if lasso_path.is_empty() {
                    return Err(format!(
                        "spec {:?}: lasso_path must be non-empty",
                        self.name
                    ));
                }
                if lasso_path
                    .iter()
                    .any(|l| !non_negative(*l) || !l.is_finite())
                {
                    return Err(format!(
                        "spec {:?}: lasso_path values must be finite and >= 0",
                        self.name
                    ));
                }
                // Strictly: a repeated penalty is two identical slots with the
                // same field name.
                if !lasso_path.windows(2).all(|w| w[0] > w[1]) {
                    return Err(format!(
                        "spec {:?}: lasso_path must be strictly decreasing",
                        self.name
                    ));
                }
                if l1_ratio.is_some_and(|r| !(0.0..=1.0).contains(&r)) {
                    return Err(format!("spec {:?}: l1_ratio must be in [0, 1]", self.name));
                }
                if select_halflife.is_some_and(|h| !positive(h)) {
                    return Err(format!("spec {:?}: select_halflife must be > 0", self.name));
                }
                check_solve_every(&self.name, *solve_every)?;
                if cd_tol.is_some_and(|t| !positive(t) || !t.is_finite()) {
                    return Err(format!(
                        "spec {:?}: cd_tol must be finite and > 0",
                        self.name
                    ));
                }
            }
            ModelKind::Rls { ridge, coef0 } => {
                if ridge.is_some_and(|r| !positive(r) || !r.is_finite()) {
                    return Err(format!(
                        "spec {:?}: rls ridge must be finite and > 0",
                        self.name
                    ));
                }
                let k_total = self.k() + usize::from(self.add_intercept);
                if let Some(c) = coef0 {
                    if c.len() != self.m() || c.iter().any(|v| v.len() != k_total) {
                        return Err(format!(
                            "spec {:?}: coef0 must be n_targets x (n_features + intercept)",
                            self.name
                        ));
                    }
                }
            }
            ModelKind::EwRidge {
                ridge,
                feature_sets,
                coef0,
                session_shrink,
                long_halflife,
                solve_every,
                ..
            } => {
                if let Some(r) = ridge {
                    let rs = r.to_vec();
                    // A negative ridge makes the system indefinite and the
                    // coefficients garbage; NaN/inf make them zero. Zero is
                    // legal: plain least squares, rescued by the jitter
                    // fallback when singular.
                    if rs.iter().any(|r| !non_negative(*r) || !r.is_finite()) {
                        return Err(format!(
                            "spec {:?}: ridge must be finite and >= 0",
                            self.name
                        ));
                    }
                    if let Some(dup) = first_duplicate(&rs) {
                        return Err(format!(
                            "spec {:?}: ridge lists {} more than once; each value is one \
                             grid slot and the two would produce the same field names",
                            self.name,
                            num_label(dup)
                        ));
                    }
                }
                check_solve_every(&self.name, *solve_every)?;
                if long_halflife.is_some_and(|h| !positive(h)) {
                    return Err(format!("spec {:?}: long_halflife must be > 0", self.name));
                }
                if session_shrink.is_some_and(|f| !(0.0..=1.0).contains(&f)) {
                    return Err(format!(
                        "spec {:?}: session_shrink must be in [0, 1]",
                        self.name
                    ));
                }
                if session_shrink.is_some() && long_halflife.is_none() {
                    return Err(format!(
                        "spec {:?}: session_shrink needs long_halflife",
                        self.name
                    ));
                }
                if session_shrink.is_some() && self.session.is_none() {
                    return Err(format!(
                        "spec {:?}: session_shrink needs a `session` column to react to",
                        self.name
                    ));
                }
                if let Some(c) = coef0 {
                    let k_total = self.k() + usize::from(self.add_intercept);
                    if c.len() != self.m() || c.iter().any(|v| v.len() != k_total) {
                        return Err(format!(
                            "spec {:?}: coef0 must be {} vectors of length {k_total}",
                            self.name,
                            self.m()
                        ));
                    }
                }
                if let Some(fs) = feature_sets {
                    for (name, cols) in fs {
                        for c in cols {
                            if !self.features.contains(c) {
                                return Err(format!(
                                    "spec {:?}: feature set {name:?} references unknown feature {c:?}",
                                    self.name
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
