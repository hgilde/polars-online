//! Coefficient constraints for the gradient models (`sgd`, `pa`): a box
//! `lo_i <= c_i <= hi_i` per slope, a sum `sum(c_i) = s`, or both, imposed by
//! Euclidean projection after each update (docs/ENHANCEMENTS.md E40). The
//! intercept is never constrained.
//!
//! The projection of `v` onto `{lo <= b <= hi, sum(a_i b_i) = s}` is
//! `b_i = clamp(v_i - mu a_i, lo_i, hi_i)` for the one `mu` at which the sum
//! holds (the KKT conditions of `min ||b - v||^2`). `g(mu) = sum(a_i
//! clamp(v_i - mu a_i, lo_i, hi_i)) - s` is piecewise linear and
//! non-increasing in `mu`, with a breakpoint where a coordinate leaves its
//! upper bound, `(v_i - hi_i) / a_i`, and one where it reaches its lower bound,
//! `(v_i - lo_i) / a_i`; sorting the `2k` breakpoints and finding the first
//! one with `g <= 0` names the segment that holds the root, on which `g` is
//! linear and `mu` is exact. O(k) for a box alone, O(k log k) with a sum.
//!
//! The weights `a_i` carry a standardization: `sgd` with `scale_features`
//! steps in standardized coordinates `b_i = c_i * scale_i`, where the
//! caller's bound on `c_i` is a bound `lo_i * scale_i` on `b_i` and the sum
//! `sum(c_i) = sum(b_i / scale_i)`, i.e. `a_i = 1 / scale_i`. Without
//! standardization every scale is 1.

use serde::{Deserialize, Serialize};

/// Bounds on the slopes; `-inf` / `inf` for none. `sum` fixes the slopes'
/// total in the caller's units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub sum: Option<f64>,
}

impl Constraint {
    /// One bound per feature, no NaN, `lo <= hi`, and a finite `sum` the
    /// box can reach. `who` prefixes the messages (`"sgd"`, `"pa"`).
    pub fn validate(&self, n_features: usize, who: &str) -> Result<(), String> {
        for (name, bounds) in [("coef_min", &self.lo), ("coef_max", &self.hi)] {
            if bounds.len() != n_features {
                return Err(format!(
                    "{who}: {name} lists {} bounds for {n_features} features",
                    bounds.len()
                ));
            }
            if let Some(i) = bounds.iter().position(|b| b.is_nan()) {
                return Err(format!("{who}: {name}[{i}] is NaN"));
            }
        }
        // An infinite bound is the absence of one, so only -inf makes sense
        // below and only inf above; the other way round pins a slope at
        // infinity.
        if let Some(i) = self.lo.iter().position(|l| *l == f64::INFINITY) {
            return Err(format!(
                "{who}: coef_min[{i}] is inf (use -inf for no bound)"
            ));
        }
        if let Some(i) = self.hi.iter().position(|h| *h == f64::NEG_INFINITY) {
            return Err(format!(
                "{who}: coef_max[{i}] is -inf (use inf for no bound)"
            ));
        }
        for (i, (lo, hi)) in self.lo.iter().zip(&self.hi).enumerate() {
            if lo > hi {
                return Err(format!(
                    "{who}: coef_min[{i}] = {lo} is above coef_max[{i}] = {hi}"
                ));
            }
        }
        if let Some(s) = self.sum {
            if !s.is_finite() {
                return Err(format!("{who}: coef_sum must be finite, got {s}"));
            }
            let lo: f64 = self.lo.iter().sum();
            let hi: f64 = self.hi.iter().sum();
            // Rounding slack: `[0.1, 0.2, 0.3]` sums to 0.6000000000000001
            // and must not refuse a `coef_sum` of 0.6. A sum inside the
            // slack projects onto the nearer end of the interval.
            let finite = |v: f64| if v.is_finite() { v.abs() } else { 0.0 };
            let slack = 1e-12 * (1.0 + finite(lo).max(finite(hi)).max(s.abs()));
            if s < lo - slack || s > hi + slack {
                return Err(format!(
                    "{who}: coef_sum = {s} is outside what the bounds allow, [{lo}, {hi}]"
                ));
            }
        }
        Ok(())
    }

    /// No sum and every bound infinite: nothing to project onto.
    pub fn is_trivial(&self) -> bool {
        self.sum.is_none()
            && self.lo.iter().all(|&l| l == f64::NEG_INFINITY)
            && self.hi.iter().all(|&h| h == f64::INFINITY)
    }

    /// Project the slopes `b` in place, `b_i` living in the standardized
    /// coordinate `c_i * scales[i]`; `None` when the model works in the
    /// caller's units. `breaks` is scratch space the caller keeps between
    /// rows, so a projection allocates nothing.
    pub fn project(&self, b: &mut [f64], scales: Option<&[f64]>, breaks: &mut Vec<f64>) {
        let k = b.len();
        debug_assert_eq!(k, self.lo.len());
        debug_assert!(scales.is_none_or(|s| s.len() == k));
        // In b-space the sum reads sum(a_i b_i) = s with a_i = 1 / scale_i
        // and the bounds are lo_i scale_i, hi_i scale_i.
        let bound = |i: usize| {
            let sc = scales.map_or(1.0, |s| s[i]);
            (1.0 / sc, self.lo[i] * sc, self.hi[i] * sc)
        };
        let Some(s) = self.sum else {
            for (i, bi) in b.iter_mut().enumerate() {
                let (_, lo, hi) = bound(i);
                *bi = bi.clamp(lo, hi);
            }
            return;
        };
        // Coordinate i sits at hi for mu < t_hi = (b_i - hi_i') / a_i and at
        // lo for mu > t_lo = (b_i - lo_i') / a_i.
        let t_hi = |i: usize, bi: f64| {
            let (a, _, hi) = bound(i);
            (bi - hi) / a
        };
        let t_lo = |i: usize, bi: f64| {
            let (a, lo, _) = bound(i);
            (bi - lo) / a
        };
        breaks.clear();
        for (i, &bi) in b.iter().enumerate() {
            for t in [t_hi(i, bi), t_lo(i, bi)] {
                if t.is_finite() {
                    breaks.push(t);
                }
            }
        }
        breaks.sort_unstable_by(f64::total_cmp);
        let g = |mu: f64| -> f64 {
            let mut acc = 0.0;
            for (i, &bi) in b.iter().enumerate() {
                let (a, lo, hi) = bound(i);
                acc += a * (bi - mu * a).clamp(lo, hi);
            }
            acc - s
        };
        // The root lies in the segment ending at the first breakpoint with
        // g <= 0 (g is non-increasing), or past the last one. Each
        // coordinate's state on that segment's interior is read off the
        // breakpoint: just below `t`, i is at hi iff t <= t_hi and at lo iff
        // t > t_lo; just above the last one, at lo iff t >= t_lo.
        let j = breaks.partition_point(|&t| g(t) > 0.0);
        let (t, below) = match breaks.get(j) {
            Some(&t) => (t, true),
            None => (breaks.last().copied().unwrap_or(0.0), false),
        };
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &bi) in b.iter().enumerate() {
            let (a, lo, hi) = bound(i);
            let at_hi = below && t <= t_hi(i, bi);
            let at_lo = if below {
                t > t_lo(i, bi)
            } else {
                t >= t_lo(i, bi)
            };
            if at_hi {
                num += a * hi;
            } else if at_lo {
                num += a * lo;
            } else {
                num += a * bi;
                den += a * a;
            }
        }
        // sum_free a_i (b_i - mu a_i) + sum_fixed a_i bound_i = s. With no
        // free coordinate the sum does not depend on mu within the segment.
        let mu = if den > 0.0 { (num - s) / den } else { t };
        for (i, bi) in b.iter_mut().enumerate() {
            let (a, lo, hi) = bound(i);
            *bi = (*bi - mu * a).clamp(lo, hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn boxed(lo: &[f64], hi: &[f64], sum: Option<f64>) -> Constraint {
        Constraint {
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            sum,
        }
    }

    fn simplex(k: usize) -> Constraint {
        boxed(&vec![0.0; k], &vec![f64::INFINITY; k], Some(1.0))
    }

    /// Duchi, Shalev-Shwartz, Singer & Chandra (2008): sort, then the
    /// largest rho with u_rho - (sum_{r<=rho} u_r - 1) / rho > 0.
    fn simplex_by_sorting(v: &[f64]) -> Vec<f64> {
        let mut u = v.to_vec();
        u.sort_by(|p, q| q.total_cmp(p));
        let mut cum = 0.0;
        let mut theta = 0.0;
        for (r, &ur) in u.iter().enumerate() {
            cum += ur;
            let cand = (cum - 1.0) / (r + 1) as f64;
            if ur - cand > 0.0 {
                theta = cand;
            }
        }
        v.iter().map(|x| (x - theta).max(0.0)).collect()
    }

    fn feasible(c: &Constraint, b: &[f64], scales: &[f64], tol: f64) -> bool {
        let inside = (0..b.len())
            .all(|i| b[i] >= c.lo[i] * scales[i] - tol && b[i] <= c.hi[i] * scales[i] + tol);
        let sum_ok = c
            .sum
            .is_none_or(|s| ((0..b.len()).map(|i| b[i] / scales[i]).sum::<f64>() - s).abs() <= tol);
        inside && sum_ok
    }

    /// KKT for the Euclidean projection: for the direction d = v - b, every
    /// free coordinate has d_i = mu a_i (one mu), a coordinate at its lower
    /// bound has d_i <= mu a_i, at its upper bound d_i >= mu a_i.
    fn kkt(c: &Constraint, v: &[f64], b: &[f64], scales: &[f64]) -> bool {
        let k = v.len();
        let a: Vec<f64> = scales.iter().map(|s| 1.0 / s).collect();
        let d: Vec<f64> = (0..k).map(|i| v[i] - b[i]).collect();
        let mu = if c.sum.is_none() {
            0.0
        } else {
            match (0..k)
                .find(|&i| b[i] > c.lo[i] * scales[i] + 1e-9 && b[i] < c.hi[i] * scales[i] - 1e-9)
            {
                Some(i) => d[i] / a[i],
                None => {
                    // All pinned: any mu between the bounds' multipliers.
                    let lower = (0..k)
                        .filter(|&i| b[i] <= c.lo[i] * scales[i] + 1e-9)
                        .map(|i| d[i] / a[i])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let upper = (0..k)
                        .filter(|&i| b[i] >= c.hi[i] * scales[i] - 1e-9)
                        .map(|i| d[i] / a[i])
                        .fold(f64::INFINITY, f64::min);
                    return lower <= upper + 1e-9;
                }
            }
        };
        (0..k).all(|i| {
            let want = mu * a[i];
            let at_lo = b[i] <= c.lo[i] * scales[i] + 1e-9;
            let at_hi = b[i] >= c.hi[i] * scales[i] - 1e-9;
            (at_lo && d[i] <= want + 1e-9)
                || (at_hi && d[i] >= want - 1e-9)
                || (d[i] - want).abs() <= 1e-9
        })
    }

    #[test]
    fn a_box_clamps_coordinatewise() {
        let c = boxed(&[0.0, -1.0, f64::NEG_INFINITY], &[1.0, 1.0, 0.5], None);
        let mut b = [1.5, -3.0, 2.0];
        c.project(&mut b, None, &mut Vec::new());
        assert_eq!(b, [1.0, -1.0, 0.5]);
        let mut inside = [0.5, 0.0, -7.0];
        c.project(&mut inside, None, &mut Vec::new());
        assert_eq!(inside, [0.5, 0.0, -7.0]);
    }

    #[test]
    fn a_box_scales_with_the_standardization() {
        // Bound 0.5 on c_i is bound 0.5 * scale_i on b_i.
        let c = boxed(&[0.0, 0.0], &[0.5, 0.5], None);
        let mut b = [1.0, 1.0];
        c.project(&mut b, Some(&[4.0, 0.25]), &mut Vec::new());
        assert_eq!(b, [1.0, 0.125]);
    }

    #[test]
    fn the_simplex_matches_the_sorting_algorithm() {
        let mut s = 7u64;
        for k in [1usize, 2, 3, 5, 8, 13] {
            let c = simplex(k);
            for _ in 0..200 {
                let v: Vec<f64> = (0..k).map(|_| 3.0 * lcg(&mut s)).collect();
                let mut b = v.clone();
                c.project(&mut b, None, &mut Vec::new());
                let want = simplex_by_sorting(&v);
                for i in 0..k {
                    assert!(
                        (b[i] - want[i]).abs() <= 1e-12,
                        "k={k} v={v:?} got {b:?} want {want:?}"
                    );
                }
                assert!((b.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn a_hyperplane_alone_shifts_by_the_mean_deficit() {
        let c = boxed(&[f64::NEG_INFINITY; 3], &[f64::INFINITY; 3], Some(1.0));
        let mut b = [2.0, -1.0, 5.0];
        c.project(&mut b, None, &mut Vec::new());
        // sum was 6, deficit 5 spread over three coordinates.
        for (got, want) in b
            .iter()
            .zip([2.0 - 5.0 / 3.0, -1.0 - 5.0 / 3.0, 5.0 - 5.0 / 3.0])
        {
            assert!((got - want).abs() <= 1e-12);
        }
        // With scales, the shift per coordinate is mu * a_i, in b-space.
        let mut b = [2.0, -1.0, 5.0];
        let scales = [1.0, 2.0, 4.0];
        c.project(&mut b, Some(&scales), &mut Vec::new());
        let total: f64 = b.iter().zip(&scales).map(|(bi, s)| bi / s).sum();
        assert!((total - 1.0).abs() <= 1e-12);
        let mu = (2.0 / 1.0 + (-1.0) / 2.0 + 5.0 / 4.0 - 1.0) / (1.0 + 0.25 + 1.0 / 16.0);
        for i in 0..3 {
            assert!((b[i] - ([2.0, -1.0, 5.0][i] - mu / scales[i])).abs() <= 1e-12);
        }
    }

    #[test]
    fn random_boxes_with_a_sum_satisfy_the_kkt_conditions() {
        let mut s = 99u64;
        for trial in 0..500 {
            let k = 1 + trial % 7;
            let mut lo = Vec::new();
            let mut hi = Vec::new();
            for _ in 0..k {
                let l = lcg(&mut s);
                let h = l + lcg(&mut s).abs() * 2.0;
                // A third of the bounds are infinite, some coordinates pinned.
                lo.push(if lcg(&mut s) > 0.4 {
                    f64::NEG_INFINITY
                } else {
                    l
                });
                hi.push(if lcg(&mut s) > 0.4 {
                    f64::INFINITY
                } else if lcg(&mut s) > 0.7 {
                    l
                } else {
                    h
                });
            }
            let scales: Vec<f64> = (0..k).map(|_| 0.25 + 2.0 * lcg(&mut s).abs()).collect();
            // A feasible sum: a random point inside the box, in caller units.
            let inside: Vec<f64> = (0..k)
                .map(|i| {
                    let l = lo[i].max(-3.0);
                    let h = hi[i].min(3.0);
                    l + (h - l) * (0.5 + 0.5 * lcg(&mut s))
                })
                .collect();
            let sum = if lcg(&mut s) > 0.0 {
                Some(inside.iter().sum())
            } else {
                None
            };
            let c = boxed(&lo, &hi, sum);
            c.validate(k, "test").unwrap();
            let v: Vec<f64> = (0..k).map(|_| 4.0 * lcg(&mut s)).collect();
            let mut b = v.clone();
            c.project(&mut b, Some(&scales), &mut Vec::new());
            assert!(
                feasible(&c, &b, &scales, 1e-9),
                "trial {trial}: {c:?} v={v:?} b={b:?}"
            );
            assert!(
                kkt(&c, &v, &b, &scales),
                "trial {trial}: {c:?} v={v:?} b={b:?}"
            );
            // Projecting a projected point moves nothing beyond rounding.
            let again = {
                let mut t = b.clone();
                c.project(&mut t, Some(&scales), &mut Vec::new());
                t
            };
            for i in 0..k {
                assert!((again[i] - b[i]).abs() <= 1e-12 * (1.0 + b[i].abs()));
            }
        }
    }

    #[test]
    fn a_feasible_point_is_left_exactly_where_it_is_by_a_box() {
        let c = boxed(&[0.0, 0.0], &[1.0, 1.0], None);
        let mut b = [0.3, 0.7];
        c.project(&mut b, None, &mut Vec::new());
        assert_eq!(b, [0.3, 0.7]);
    }

    #[test]
    fn everything_pinned_lands_on_the_only_point() {
        let c = boxed(&[0.25, 0.75], &[0.25, 0.75], Some(1.0));
        c.validate(2, "test").unwrap();
        let mut b = [-5.0, 9.0];
        c.project(&mut b, None, &mut Vec::new());
        assert_eq!(b, [0.25, 0.75]);
        // A sum inside the rounding slack but below what the floors add up
        // to: the projection lands on the floors, from either side.
        let c = boxed(&[0.1, 0.2, 0.3], &[0.1, 0.2, 0.3], Some(0.6));
        for start in [[-5.0, 9.0, 0.0], [0.1, 0.2, 0.3], [1.0, 1.0, 1.0]] {
            let mut b = start;
            c.project(&mut b, None, &mut Vec::new());
            assert_eq!(b, [0.1, 0.2, 0.3], "from {start:?}");
        }
    }

    #[test]
    fn the_root_can_lie_past_the_last_breakpoint() {
        // One coordinate free below, everything else pinned high: the sum is
        // reached only for mu beyond every breakpoint.
        let c = boxed(
            &[f64::NEG_INFINITY, 0.0],
            &[f64::INFINITY, 1.0],
            Some(-10.0),
        );
        let mut b = [5.0, 5.0];
        c.project(&mut b, None, &mut Vec::new());
        assert!((b[0] + 10.0).abs() <= 1e-12 && b[1] == 0.0, "{b:?}");
    }

    #[test]
    fn validation_names_the_offence() {
        let k = 2;
        let err = |c: Constraint| c.validate(k, "sgd").unwrap_err();
        assert_eq!(
            err(boxed(&[0.0], &[1.0, 1.0], None)),
            "sgd: coef_min lists 1 bounds for 2 features"
        );
        assert_eq!(
            err(boxed(&[0.0, 0.0], &[1.0], None)),
            "sgd: coef_max lists 1 bounds for 2 features"
        );
        assert_eq!(
            err(boxed(&[0.0, f64::NAN], &[1.0, 1.0], None)),
            "sgd: coef_min[1] is NaN"
        );
        assert_eq!(
            err(boxed(&[f64::INFINITY, 0.0], &[f64::INFINITY, 1.0], None)),
            "sgd: coef_min[0] is inf (use -inf for no bound)"
        );
        assert_eq!(
            err(boxed(
                &[f64::NEG_INFINITY, 0.0],
                &[f64::NEG_INFINITY, 1.0],
                None
            )),
            "sgd: coef_max[0] is -inf (use inf for no bound)"
        );
        assert_eq!(
            err(boxed(&[0.0, 2.0], &[1.0, 1.0], None)),
            "sgd: coef_min[1] = 2 is above coef_max[1] = 1"
        );
        assert_eq!(
            err(boxed(&[0.0, 0.0], &[1.0, 1.0], Some(f64::INFINITY))),
            "sgd: coef_sum must be finite, got inf"
        );
        assert_eq!(
            err(boxed(&[0.0, 0.0], &[1.0, 1.0], Some(2.5))),
            "sgd: coef_sum = 2.5 is outside what the bounds allow, [0, 2]"
        );
        assert_eq!(
            err(boxed(
                &[0.0, 0.0],
                &[f64::INFINITY, f64::INFINITY],
                Some(-1.0)
            )),
            "sgd: coef_sum = -1 is outside what the bounds allow, [0, inf]"
        );
        // Floors that sum to 0.6000000000000001 accept a sum of 0.6.
        boxed(&[0.1, 0.2, 0.3], &[0.1, 0.2, 0.3], Some(0.6))
            .validate(3, "sgd")
            .unwrap();
        boxed(&[0.0, 0.0], &[1.0, 1.0], Some(2.0))
            .validate(k, "sgd")
            .unwrap();
        boxed(&[f64::NEG_INFINITY; 2], &[f64::INFINITY; 2], Some(-7.0))
            .validate(k, "sgd")
            .unwrap();
        assert!(boxed(&[f64::NEG_INFINITY; 2], &[f64::INFINITY; 2], None).is_trivial());
        assert!(!boxed(&[0.0; 2], &[f64::INFINITY; 2], None).is_trivial());
        assert!(!boxed(&[f64::NEG_INFINITY; 2], &[f64::INFINITY; 2], Some(0.0)).is_trivial());
    }
}
