//! Running means carried to twice a double's precision (docs/PLAN.md task
//! 101).
//!
//! A weighted running mean steps toward each value it is given,
//!
//! ```text
//! m' = m + b·(x − m)          b = w / (λ·W + w)
//! ```
//!
//! and given one value row after row it converges on it geometrically,
//! until the step `b·(x − m)` is under half a rounding step of `m`. Then
//! `m + b·(x − m)` rounds back to `m`, and the mean stays `1/(2b)` rounding
//! steps short of the value for good. Everything the mean centres is off by
//! that gap from then on: a variance fed `(x − m)²` settles on the gap's
//! square instead of decaying with its history, a co-moment fed
//! `(x − m)·(y − ȳ)` on the gap times the other side's noise, and a model
//! that divides one by the other reads the ratio of two rounding artefacts.
//!
//! Here a mean is a pair, `hi + lo`: `hi` the double nearest it and `lo`
//! what `hi` leaves out. A row's step `s = b·d`, with `d = (x − hi) − lo`
//! the deviation from the pair, goes in by Kahan's compensated step: `y = s
//! + lo` is a plain rounded sum, and `hi + y` is split exactly into its
//! double and the part left out (Fast2Sum) while `|y| ≤ |hi|`. So the pair
//! is not a double-double: it carries each step's own rounding, about
//! `ε·|b·d|` a row, which is `ε·σ/L` of a mean at level `L` over values of
//! spread `σ`, not 2^-106 of it. Measured against the same recursion in
//! 80-digit decimals (`b = 0.034`, 20,000 rows): a value held at 1e8 is
//! approached to 5.7e-26 of the mean (the plain step stalls at 2.1e-15),
//! values of spread 1 moving at 1e8 are followed to 4.7e-25 (plain
//! 6.6e-16), and at level 1, where `|y| ≤ |hi|` often fails, to 5.9e-17,
//! a rounding of the mean (plain 8.4e-16; review 2026-10-23, A-4). What
//! matters is the held value: the deviation, a small number in its own
//! right, keeps its full precision as it shrinks, so a value held row
//! after row is approached as exact arithmetic approaches it, and what is
//! fed its deviation decays with its history. Nothing needs to know that
//! the value is held, how much the row weighs, or whether the mean decays.

/// The deviation of `x` from the mean `hi + lo`: `(x − hi) − lo`. Where `x`
/// is near `hi` the first difference is exact, so the deviation is good to
/// a rounding of itself however small it is.
#[inline]
pub fn dev(x: f64, hi: f64, lo: f64) -> f64 {
    (x - hi) - lo
}

/// `hi + lo += s`, Kahan's compensated step: the part of `s` that `hi`
/// cannot take is kept in `lo`. Exact while the step is smaller than the
/// mean -- where a plain step rounds away, a value held row after row
/// among them -- and no worse than the plain step where it is larger.
///
/// Kahan's step does not keep `lo` under half a rounding step of `hi`, so
/// adding zero can round the pair afresh: the same number, with `hi` a step
/// off; `lo` exceeds half a step only after a step larger than the mean
/// itself, where Fast2Sum's `|s + lo| <= |hi|` fails, which is common on a
/// feature centred near zero and rare at a level (review 2026-09-25). A
/// caller therefore does not take the step of a row of weight 0
/// (CLAUDE.md hard rule 9) -- `tests/test_label_delay.py` holds a stream
/// with a zero-weight copy of every row to the one without, to the bit --
/// and that is one test a row, where a test here, a step at a time, stopped
/// the loops that call it from vectorizing (`EwDiag::update` at 64 slots,
/// 23 ns a row without it and 36 with the cheapest form of it).
#[inline]
pub fn add(hi: &mut f64, lo: &mut f64, s: f64) {
    let y = s + *lo;
    let t = *hi + y;
    *lo = y - (t - *hi);
    *hi = t;
}

// A `lo` vector is sized with its means when they are made, and a restored
// one of another length is refused as every other vector of a state is. The
// repair that sized one at its first use, for a state written before the low
// parts existed, loaded nothing a file this build reads can hold, and mended
// a damaged vector with zeros (docs/PLAN.md task 198; review round 4, CC8).

#[cfg(test)]
mod tests {
    use super::*;

    /// The pair is exact where the sum is representable, and keeps what a
    /// double would drop.
    #[test]
    fn add_keeps_what_rounding_drops() {
        let (mut hi, mut lo) = (1e8, 0.0);
        add(&mut hi, &mut lo, 1e-10);
        assert_eq!(hi, 1e8, "1e-10 is under half a rounding step of 1e8");
        assert_eq!(lo, 1e-10);
        for _ in 0..1_000_000 {
            add(&mut hi, &mut lo, 1e-10);
        }
        // 1,000,001 steps of 1e-10: 1.000001e-4, which the plain sum drops
        // entirely.
        let want = 1.000001e-4;
        // Tolerance (task 252 (b)): against exact arithmetic, an oracle
        // computed another way, which the pair follows to about 2^-106 of
        // the mean. The pair this build computes is pinned to the bit as
        // well: IEEE sums and differences only, no libm, the same on every
        // platform.
        assert!(((hi - 1e8) + lo - want).abs() < 1e-18, "{hi} {lo}");
        assert_eq!(
            (hi.to_bits(), lo.to_bits()),
            (4_726_483_295_884_286_519, 13_698_645_039_560_920_696),
            "{hi} {lo:e}"
        );
        let mut plain = 1e8;
        for _ in 0..1_000_001 {
            plain += 1e-10;
        }
        assert_eq!(plain, 1e8, "the plain sum drops every step");
    }

    /// A far mean that the rows reach at a tiny weight -- an `hmm` state the
    /// data is nowhere near, whose posterior weights each row by 1e-100 --
    /// moves by what exact arithmetic moves it, which is nothing a double
    /// holds.
    #[test]
    fn a_tiny_weight_moves_the_pair_as_exact_arithmetic_does() {
        let (mut hi, mut lo) = (1.0, 0.0);
        for _ in 0..10_000 {
            let d = dev(5.0, hi, lo);
            add(&mut hi, &mut lo, 1e-100 * d);
        }
        assert_eq!(hi, 1.0);
        // Tolerance (task 252 (b)): against exact arithmetic's `4e-96`, an
        // oracle computed another way; the low part this build computes is
        // pinned to the bit beside it (no libm in the path).
        assert!((lo - 4e-96).abs() <= 1e-9 * 4e-96, "{lo:e}");
        assert_eq!(lo.to_bits(), 3_179_847_552_505_719_914, "{lo:e}");
    }

    /// A mean given one value row after row: the plain step stops `1/(2b)`
    /// rounding steps short of it; the pair's deviation keeps shrinking as
    /// exact arithmetic's does, `d' = (1 − b)·d`, and the mean reaches the
    /// value.
    #[test]
    fn a_mean_given_one_value_reaches_it_as_exact_arithmetic_does() {
        for (level, b) in [
            (0.5, 0.034),
            (1e3, 0.034),
            (1e8, 0.034),
            (-1e8, 1e-3),
            (1e12, 0.3),
            (-0.37, 0.3),
        ] {
            let x: f64 = level + 0.37;
            let (mut plain, mut hi, mut lo) = (level, level, 0.0);
            let mut exact = x - level;
            for n in 0..40_000 {
                plain += b * (x - plain);
                let d = dev(x, hi, lo);
                // The deviation is exact arithmetic's gap, stepped on its own.
                // Tolerance (task 252 (b)): `exact` is stepped by plain
                // products, so it is itself rounded each row; the two are
                // two computations of one quantity, neither the other's.
                assert!(
                    (d - exact).abs() <= 1e-9 * exact.abs() + 1e-300,
                    "level {level}, row {n}: {d:e} against {exact:e}"
                );
                add(&mut hi, &mut lo, b * d);
                exact -= b * exact;
            }
            assert_ne!(
                plain, x,
                "level {level}: the plain step reaches it; no case"
            );
            if x == 0.0 {
                // Nothing below the smallest subnormal holds a remainder, so
                // at zero the pair stops where the plain step does, and
                // harmlessly: the deviation it leaves squares to zero.
                assert_eq!(dev(x, hi, lo) * dev(x, hi, lo), 0.0);
            } else {
                assert_eq!(hi, x, "level {level}");
            }
        }
    }
}
