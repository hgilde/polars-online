//! [`MergeBound::merged`] against its form at 46dcb86c (task 308), where
//! each of the two terms built on `η` was chosen whole between 0 and its
//! product, a choice that compiles to a select computing both arms (on
//! x86-64, `nu √η √η` was formed on every feature and masked away, a
//! subnormal). The choice now sits on a factor; these tests hold that it
//! gives the same bits -- the terms themselves, sign of zero included, and
//! `e'` -- on the edges of every input.

use super::*;

/// 46dcb86c's `MergeBound::merged`, verbatim, on `e` in place of `self.0`.
fn merged_before(e: &mut Vec<f64>, c: &LowerTriangle, k: usize, s: &MergeStep<'_>) {
    if e.len() != k || k == 0 {
        return e.clear();
    }
    let n = s.n as f64;
    let m = (4.0 * n + 32.0) * U;
    let eps = m / (1.0 - m);
    let usable = !s.shares
        && eps <= 1e-3
        && s.alpha >= ALPHA_MIN
        && s.alpha.is_finite()
        && s.within.is_normal()
        && s.u_sum.is_normal()
        && s.scale >= 0.0
        && s.scale.is_finite()
        && s.between >= 0.0
        && s.between.is_finite();
    if !usable {
        return e.clear();
    }
    let (sigma, alpha, beta, omega) = (s.scale, s.alpha, s.between, s.within);
    let ratio = over(n, s.u_sum);
    let spread = ratio + 1.0 + ratio.sqrt();
    // The residue's `7 ω τ²/ε = η² r`, `τ = η τ_n`: left to the second
    // `FLOOR` where `r` is finite, and so at most `2⁻¹¹²⁴`. Where it is
    // not, `τ` is formed, `2η` times `spread` in two scalings, the
    // first exact, and the term as before, which forgets `e` where it is
    // not finite either (the module docs, item 8).
    let tau_n = 2.0 * spread;
    let r = times(times(times(7.0 / eps, omega), tau_n), tau_n);
    let residue = if r.is_finite() {
        0.0
    } else {
        let tau = spread * ROOT_ETA * (2.0 * ROOT_ETA);
        // Every factor is nonnegative; the order keeps a large factor
        // from multiplying an underflowed product.
        7.0 / eps * omega * tau * tau
    };
    let rows = times(n, 4.0 + alpha);
    let grow = 1.0 + eps;
    let shrink = 1.0 / (1.0 - 10.0 * eps);
    let (ten, twenty_two) = (10.0 * eps, 22.0 * eps);
    let lower = c.lower();
    let mut finite = true;
    for (i, ei) in e.iter_mut().enumerate() {
        let (g, h) = (s.big[i].abs(), s.delta[i].abs());
        let cii = lower[i * k + i];
        let cii = if cii.is_nan() {
            f64::INFINITY
        } else {
            cii.max(0.0)
        };
        // The underflows' `η ν_i`: left to the second `FLOOR` while the
        // computed `ν_i` is at most `2⁷²`; past it, or where it is not a
        // number, formed in two exact scalings by `√η`, a normal number.
        let nu = (1.0 + g) * (1.0 + g) + (1.0 + h) * (1.0 + h) + 3.0 + rows;
        let under = if nu <= UNDER_MAX {
            0.0
        } else {
            times(times(nu, ROOT_ETA), ROOT_ETA)
        };
        // Every term is nonnegative; each product's order keeps a
        // large factor from multiplying an underflowed one.
        let sum = times(times(*ei, sigma), grow)
            + times(ten, cii)
            + times(twenty_two, times(omega, times(h, h)))
            + times(eps, times(beta, times(g, g)))
            + residue
            + under;
        *ei = times(times(sum, shrink), UP) + FLOOR + FLOOR;
        finite &= ei.is_finite();
    }
    if !finite {
        e.clear();
    }
}

/// Edge values: signed zeros, the subnormal and normal limits, the
/// overflows, the infinities and NaN, and ordinary numbers.
const EDGES: [f64; 17] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    1e-160,
    f64::MIN_POSITIVE,
    ETA,
    68_719_476_736.0, // 2³⁶
    1e12,
    1e154,
    1e300,
    1e-300,
    f64::MAX,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

/// `ν` near `2⁷²`, `n` ulps either side, and the reachable range's edges.
fn nus() -> Vec<f64> {
    let mut v = vec![5.0, 9.0, 1e10, 1e30, f64::MAX, f64::INFINITY, f64::NAN];
    let at = UNDER_MAX.to_bits();
    v.extend((at - 1000..=at + 1000).map(f64::from_bits));
    v
}

#[test]
fn the_underflows_term_is_the_select_it_replaced_to_the_bit() {
    let (mut below, mut at, mut above) = (0, 0, 0);
    for nu in nus() {
        let before = if nu <= UNDER_MAX {
            0.0
        } else {
            times(times(nu, ROOT_ETA), ROOT_ETA)
        };
        let now = underflows(nu);
        assert_eq!(now.to_bits(), before.to_bits(), "nu {nu:e}");
        match nu.partial_cmp(&UNDER_MAX) {
            Some(std::cmp::Ordering::Less) => below += 1,
            Some(std::cmp::Ordering::Equal) => at += 1,
            _ => above += 1,
        }
    }
    assert!(below > 1000 && at == 1 && above > 1000);
}

#[test]
fn the_residue_term_is_the_select_it_replaced_to_the_bit() {
    let mut kept = 0;
    for lead in EDGES {
        for r in EDGES {
            // Where `r` is finite, `merged` has `spread` finite and at least
            // 1 (the term's docs); elsewhere it can be anything.
            let spreads: &[f64] = if r.is_finite() {
                &[1.0, 1.5, 3.0, 1e300, f64::MAX]
            } else {
                &EDGES
            };
            for &spread in spreads {
                let before = if r.is_finite() {
                    0.0
                } else {
                    let tau = spread * ROOT_ETA * (2.0 * ROOT_ETA);
                    lead * tau * tau
                };
                let now = residue_term(lead, r, spread);
                assert_eq!(
                    now.to_bits(),
                    before.to_bits(),
                    "lead {lead:e} r {r:e} spread {spread:e}"
                );
                kept += usize::from(!r.is_finite() && now.is_normal());
            }
        }
    }
    // The term formed, and a normal number, where `r` is not finite.
    assert!(kept > 0);
}

/// One merge's inputs, at `k = 2`.
#[derive(Clone)]
struct Case {
    e: Vec<f64>,
    c: [f64; 4],
    n: usize,
    scale: f64,
    alpha: f64,
    between: f64,
    within: f64,
    u_sum: f64,
    big: [f64; 2],
    delta: [f64; 2],
}

impl Case {
    fn ordinary() -> Self {
        Self {
            e: vec![1e-3, 2e-3],
            c: [2.0, 0.0, 0.5, 3.0],
            n: 16,
            scale: 0.9,
            alpha: 0.01,
            between: 0.01,
            within: 0.16,
            u_sum: 16.0,
            big: [0.3, -0.2],
            delta: [1e-17, -2e-17],
        }
    }

    /// `e'` now and at 46dcb86c, as bits; empty where forgotten.
    fn both(&self) -> (Vec<u64>, Vec<u64>) {
        let c = LowerTriangle::from(self.c.to_vec());
        let s = MergeStep {
            n: self.n,
            scale: self.scale,
            alpha: self.alpha,
            between: self.between,
            within: self.within,
            u_sum: self.u_sum,
            big: &self.big,
            delta: &self.delta,
            shares: false,
        };
        let mut now = MergeBound(self.e.clone());
        now.merged(&c, 2, &s);
        let mut before = self.e.clone();
        merged_before(&mut before, &c, 2, &s);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect();
        (bits(&now.0), bits(&before))
    }
}

/// `e'` over edge inputs, one field or one pair at a time: the same bits as
/// 46dcb86c's, and kept or forgotten alike. Activation: merges kept with
/// `r` not finite, and with `ν_0` below, at and above `2⁷²`.
#[test]
fn a_merge_gives_the_bound_it_gave_before_on_every_edge() {
    let mut cases = vec![Case::ordinary()];
    let vary = |f: &dyn Fn(&mut Case, f64)| -> Vec<Case> {
        EDGES
            .iter()
            .map(|&v| {
                let mut c = Case::ordinary();
                f(&mut c, v);
                c
            })
            .collect()
    };
    cases.extend(vary(&|c, v| c.e[0] = v));
    cases.extend(vary(&|c, v| c.c[0] = v));
    cases.extend(vary(&|c, v| c.c[3] = v));
    cases.extend(vary(&|c, v| c.scale = v));
    cases.extend(vary(&|c, v| c.alpha = v));
    cases.extend(vary(&|c, v| c.between = v));
    for a in EDGES {
        for b in EDGES {
            let mut c = Case::ordinary();
            (c.big[0], c.delta[0]) = (a, b);
            cases.push(c);
            let mut c = Case::ordinary();
            (c.within, c.u_sum) = (a, b);
            cases.push(c);
        }
        for n in [0, 1, 16, 1000, 1 << 20, 1 << 40] {
            let mut c = Case::ordinary();
            (c.n, c.u_sum) = (n, a);
            cases.push(c);
        }
    }
    // `ν_0` across `2⁷²`: `g = 2³⁶ − 1` makes `(1 + g)² = 2⁷²`.
    let edge = (68_719_476_736.0f64 - 1.0).to_bits();
    for j in -64i64..=64 {
        let mut c = Case::ordinary();
        c.big[0] = f64::from_bits(edge.wrapping_add_signed(j * 1024));
        c.delta[0] = 0.0;
        cases.push(c);
    }

    let (mut overflowed, mut below, mut at, mut above) = (0, 0, 0, 0);
    for c in &cases {
        let (now, before) = c.both();
        assert_eq!(
            now, before,
            "e {:?} c {:?} n {} scale {:e} alpha {:e} between {:e} within {:e} \
             u_sum {:e} big {:?} delta {:?}",
            c.e, c.c, c.n, c.scale, c.alpha, c.between, c.within, c.u_sum, c.big, c.delta
        );
        if now.is_empty() {
            continue;
        }
        // What `merged` computed, for the activation counts.
        let n = c.n as f64;
        let m = (4.0 * n + 32.0) * U;
        let eps = m / (1.0 - m);
        let ratio = n / c.u_sum;
        let tau_n = 2.0 * (ratio + 1.0 + ratio.sqrt());
        overflowed += usize::from(!(7.0 / eps * c.within * tau_n * tau_n).is_finite());
        let (g, h) = (c.big[0].abs(), c.delta[0].abs());
        let nu = (1.0 + g) * (1.0 + g) + (1.0 + h) * (1.0 + h) + 3.0 + n * (4.0 + c.alpha);
        match nu.partial_cmp(&UNDER_MAX) {
            Some(std::cmp::Ordering::Less) => below += 1,
            Some(std::cmp::Ordering::Equal) => at += 1,
            _ => above += 1,
        }
    }
    eprintln!(
        "{} cases; kept with r not finite {overflowed}, nu below {below}, at {at}, above {above}",
        cases.len()
    );
    assert!(overflowed > 0 && below > 0 && at > 0 && above > 0);
}
