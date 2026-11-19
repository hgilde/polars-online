//! A backward row's held mass: the time its value is held over, the partner
//! of its discount ([`mass_held_to`]).
use super::*;

/// The distance, in half-lives, from which a held mass is taken from the
/// discount's `exp2` instead of `exp_m1`: a hundredth of a half-life.
pub(super) const HELD_MASS_FROM: f64 = 1e-2;

impl KernelDef {
    /// The mass a row holds its value over, `d` units of time: its discount's
    /// partner, through `memo`'s slot for it (the cut's mass of the same
    /// distance is [`Self::mass_in`]'s, and keeps `exp_m1`).
    #[inline]
    pub(super) fn mass_held_in(&self, d: f64, memo: &mut DecayMemo) -> f64 {
        if d <= 0.0 {
            return 0.0;
        }
        if self.half_life.is_infinite() {
            return d;
        }
        let (bits, v) = memo.held;
        if bits == d.to_bits() {
            return v;
        }
        let v = self.mass_held_finite(d, memo);
        memo.held = (d.to_bits(), v);
        v
    }

    /// [`Self::mass_finite`] of the distance a row holds its value over. From
    /// `HELD_MASS_FROM` half-lives up it is `c (1 - d)` from the very
    /// `d = 2^(-x)` the row's discount is, taken through the discount's own
    /// slot, so a row's step and its mass cost one `exp2` between them
    /// where they ask for the same distance; the two forms agree to a few
    /// ulps there (`1 - d` loses at most `2^-52 / (1 - d)` of the mass, 1.6e-14
    /// at the cut). Below it `1 - d` is too coarse (doubles next to 1 are
    /// 1.1e-16 apart) and `exp_m1` is kept. The choice is on `x` (an IEEE
    /// division and a compare), never on a library result, so every platform
    /// takes the same form for the same row.
    #[inline]
    fn mass_held_finite(&self, d: f64, memo: &mut DecayMemo) -> f64 {
        let h = self.half_life;
        if d / h >= HELD_MASS_FROM {
            return (h / std::f64::consts::LN_2) * (1.0 - self.discount_in(d, memo));
        }
        self.mass_finite(d)
    }
}
