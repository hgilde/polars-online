//! The warm-up of a standardizing fit: `sgd` and `pa` under `standardize`
//! (docs/PLAN.md task 206; review round 5, G1). `kalman` took it too until
//! task 211, which replaced it: the filter's `P` carries its own
//! uncertainty through the first rows, so it sizes each coefficient's prior
//! once the feature's scale is usable and follows the moments from the
//! first row (`kalman.rs`'s module doc). The numbers below are task 206's.
//!
//! Each of the three takes its step in the coordinates of the features' EW
//! moments, the scaler: `z_i = (x_i − m_i) / s_i`. Holding the fit in those
//! coordinates and reading it out through the moments as they stand makes a
//! prediction move whenever the scaler does, though no step was taken: under
//! a finite half-life the scaler wanders for ever (the EW mean of a
//! unit-variance feature has standard deviation `sqrt((1 − λ) / (1 + λ))`,
//! 0.083 at a half-life of 50), and that wander alone cost `sgd` 248 noise
//! variances of out-of-sample error at R² 0.99998 and a half-life of 50,
//! where the unstandardized fit paid 0.010. Holding the fit so that the
//! scaler's moving moves no prediction -- in the caller's units (`sgd`,
//! `pa`), or re-mapped through every move (`kalman`) -- fixes that, but not
//! from the first row: over the first rows the moments rest on almost
//! nothing, and a fit in the caller's units keeps what a step taken against
//! them did. Read through the moments as they stand, a fit made against a
//! scale a few rows old is read again at the next scale, and a bad early
//! step is forgotten as the scale settles; held in the caller's units it is
//! kept, R² −206 at rows 25–50 of 200-row groups where today's read 0.43
//! (task 74's regime). So the model runs today's design while the scaler
//! warms up, and switches once.
//!
//! **The count** is the number of rows the scaler has learned, undecayed,
//! and with weights Kish's effective sample size, `n = (Σ w)² / Σ w²`: the
//! row count at unit weights, and less where the weights are uneven. A row
//! of weight 0 adds nothing. Undecayed because the question is how many
//! rows the moments have seen, not how many they hold now: an EW count
//! tops out at `(1 + λ) / (1 − λ)`, 14.5 at a half-life of 5, and never
//! switched a short half-life, which is where the wander costs most.
//!
//! **The switch** happens once, on the row after whose scaler update the
//! count first reaches [`WARMUP_ROWS`]: until then every number is the one
//! the model gave before task 206, to the bit. A restart or a drift reset
//! builds the model again, scaler and count with it, so a new warm-up
//! begins. The count stops at the switch: it has nothing more to decide.
//!
//! **Why 22.** The scaler's scale is a sample standard deviation, and its
//! square has a relative standard error of `sqrt(2 / (n − 1))` over `n`
//! normal rows, about `sqrt(2 / n)`: at a relative error `ρ`, `n = 2 / ρ²`,
//! and at `ρ = 0.3` that is 22.2. A step is a learning rate times `|z|²`,
//! so a variance read four times too small makes a step four times too
//! large. With the mean read from the same rows, `(n − 1) ŝ² / σ²` is
//! `χ²` with `n − 1` degrees of freedom, so at `n = 22`
//!
//! ```text
//! P(ŝ² < σ² / 4) = P(χ²₂₁ < 21 / 4 = 5.25) = 1.97e-4
//! ```
//!
//! per feature (scipy's `chi2.cdf(5.25, 21)`; with the mean known, 22
//! degrees of freedom, the closed form `1 − e^(−x/2) Σ_{j<11} (x/2)^j / j!`
//! at `x = 5.5` gives 1.40e-4). Measured (the research behind task 206):
//! a switch at 22 kept the short-history regime at R² 0.449 / 0.708 /
//! 0.689 (`sgd` / `pa` / `kalman`, rows 25–50) against today's 0.433 /
//! 0.712 / 0.675, where a switch at 5 took `pa` to 0.676 and one at 0
//! (no warm-up) took `sgd` to −206.

use serde::{Deserialize, Serialize};

/// How many rows the scaler learns before the fit is held so that its
/// moving moves no prediction (the module docs).
pub const WARMUP_ROWS: f64 = 22.0;

/// The warm-up count of a standardizing model and whether it has switched
/// (the module docs). Part of the model's state since schema 47.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "WarmupWire")]
pub struct Warmup {
    /// `Σ w` over the rows the scaler has learned, undecayed, until the
    /// switch.
    w: f64,
    /// `Σ w²` over the same rows.
    w2: f64,
    /// Whether the count has reached [`WARMUP_ROWS`].
    switched: bool,
}

/// The wire layout, checked on the way in: two sums no row can make
/// negative or other than a number.
#[derive(Deserialize)]
struct WarmupWire {
    w: f64,
    w2: f64,
    switched: bool,
}

impl TryFrom<WarmupWire> for Warmup {
    type Error = String;

    fn try_from(v: WarmupWire) -> Result<Self, String> {
        if !(v.w.is_finite() && v.w >= 0.0 && v.w2.is_finite() && v.w2 >= 0.0) {
            return Err(format!(
                "warm-up: the weight sums must be finite and >= 0, got {} and {}",
                v.w, v.w2
            ));
        }
        Ok(Self {
            w: v.w,
            w2: v.w2,
            switched: v.switched,
        })
    }
}

impl Warmup {
    /// Kish's count of the rows the scaler has learned, `(Σ w)² / Σ w²`:
    /// 0 before any row of positive weight, and where the weights are so
    /// small that their squares underflow.
    pub fn count(&self) -> f64 {
        if self.w2 > 0.0 {
            self.w * self.w / self.w2
        } else {
            0.0
        }
    }

    /// Whether the count has reached [`WARMUP_ROWS`]: from the row after
    /// that, the fit is held so that the scaler's moving moves no
    /// prediction.
    pub fn switched(&self) -> bool {
        self.switched
    }

    /// The scaler has learned a row at `weight`: counted, and `true` on the
    /// row that makes the switch. A row after the switch is not counted.
    pub(crate) fn learn(&mut self, weight: f64) -> bool {
        if self.switched {
            return false;
        }
        self.w += weight;
        self.w2 += weight * weight;
        self.switched = self.count() >= WARMUP_ROWS;
        self.switched
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At unit weights the count is the row count, exactly, and the switch
    /// comes on the 22nd row; a row of weight 0 adds nothing, and uneven
    /// weights count for less.
    #[test]
    fn the_count_is_kish_and_the_switch_comes_once() {
        let mut w = Warmup::default();
        for i in 1..=21 {
            assert!(!w.learn(1.0), "row {i}");
            assert_eq!(w.count(), f64::from(i));
        }
        assert!(!w.learn(0.0));
        assert_eq!(w.count(), 21.0);
        assert!(w.learn(1.0), "the 22nd row switches");
        assert!(w.switched());
        assert!(!w.learn(1.0), "once");
        assert_eq!(w.count(), 22.0, "and the count stops");

        // Weights 2.5 and 0.5 alternating: (3n)² / (6.5n) per pair.
        let mut w = Warmup::default();
        let mut rows = 0;
        while !w.learn(if rows % 2 == 0 { 2.5 } else { 0.5 }) {
            rows += 1;
        }
        let n = (rows + 1) as f64;
        assert!(rows > 22, "uneven weights count for less: {rows}");
        assert!(
            w.count() >= 22.0 && w.count() < 22.0 * (n / (n - 1.0)),
            "{}",
            w.count()
        );
    }

    /// Weights whose squares underflow count for nothing, and nothing is
    /// not 0/0.
    #[test]
    fn weights_too_small_to_square_count_nothing() {
        let mut w = Warmup::default();
        assert_eq!(w.count(), 0.0);
        for _ in 0..100 {
            assert!(!w.learn(1e-200));
        }
        assert_eq!(w.count(), 0.0);
    }

    #[test]
    fn a_damaged_count_is_refused() {
        #[derive(Serialize)]
        struct Raw {
            w: f64,
            w2: f64,
            switched: bool,
        }
        for (w, w2) in [(f64::NAN, 1.0), (1.0, -1.0), (f64::INFINITY, 1.0)] {
            let bytes = rmp_serde::to_vec_named(&Raw {
                w,
                w2,
                switched: false,
            })
            .unwrap();
            let got = rmp_serde::from_slice::<Warmup>(&bytes);
            assert!(
                got.as_ref()
                    .is_err_and(|e| e.to_string().contains("weight sums")),
                "{w} {w2}: {got:?}"
            );
        }
        let good = Warmup {
            w: 3.0,
            w2: 5.0,
            switched: false,
        };
        let back: Warmup = rmp_serde::from_slice(&rmp_serde::to_vec_named(&good).unwrap()).unwrap();
        assert_eq!(back, good);
    }
}
