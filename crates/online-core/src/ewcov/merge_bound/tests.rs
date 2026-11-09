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
    let lower = c.packed();
    let mut finite = true;
    for (i, ei) in e.iter_mut().enumerate() {
        let (g, h) = (s.big[i].abs(), s.delta[i].abs());
        let cii = lower[crate::ewcov::row_at(i) + i];
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
    // The residue formed (`r` not finite: `n/Û = 2⁴⁴`, `7 ω/ε` near
    // `2¹⁰²⁴`) and summed with `10 ε c_ii` at a rounding edge of `e'`: the
    // residue, about `2⁻¹⁰¹⁶`, moves by a few subnormal steps where
    // `spread` loses or doubles its `1` (`n/Û · 1 + √(n/Û)`, or `n/Û − 1 +
    // √(n/Û)`), and `e'` by its last bit here (found by search over
    // `c_ii`; task 312).
    cases.push(Case {
        e: vec![0.0, 0.0],
        c: [
            f64::from_bits(0x0230_64bc_471e_c15a),
            0.0,
            0.0,
            f64::from_bits(0x0230_64bc_471e_c15a),
        ],
        n: 1,
        scale: 0.5,
        alpha: 1.0,
        between: 0.0,
        within: f64::from_bits(0x7cc3_319e_49d8_5b71),
        u_sum: f64::from_bits(0x3d30_0000_0000_0000),
        big: [0.0, 0.0],
        delta: [0.0, 0.0],
    });

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

// ---- `e'` against item 8's recursion, written from the condition module's
// docs rather than from `merged` (task 312: the mutants of 5f890340). The
// tests above hold `merged` to its earlier form, which shares its constants
// and, on their inputs, never let `η ν_i` or the residue's term past the
// floor.

/// `2⁻¹⁰⁷⁴`, the smallest subnormal.
const ETA_DEF: f64 = 5e-324;

/// `2⁷²`, written out.
const TWO_72: f64 = 4_722_366_482_869_645_213_696.0;

/// `2⁻⁴⁶`, written out.
const TWO_M46: f64 = 1.0 / 70_368_744_177_664.0;

/// Item 8's `e'` for `e`, the merge's `c'` (its diagonal is what is read)
/// and `s`, in plain arithmetic: `ε = γ_{4n+32}`; outside the argument --
/// rows weighted by their shares, `ε` past `1/1000`, `α` under `2⁻⁹⁰⁰` or
/// not finite, `ω` or `Û` not normal, `σ` or `β` negative or not finite --
/// `None`; otherwise, per feature,
///
/// ```text
/// e'_i = ( σ (1 + ε) e_i + 10 ε c'_ii⁺ + 22 ε ω h_i² + ε β g_i² + 7 ω τ²/ε
///          + η ν_i ) / (1 − 10 ε) · (1 + 2⁻⁴⁶) + 2⁻¹⁰⁰⁰ + 2⁻¹⁰⁰⁰,
/// ```
///
/// `ν_i = (1 + |g_i|)² + (1 + h_i)² + 3 + n (4 + α)`, `τ = 2η (n/Û + 1 +
/// √(n/Û))`, the residue's term left out where the computed `r = 7 ω
/// τ_n²/ε`, `τ_n = τ/η`, is finite and `η ν_i` where the computed `ν_i` is
/// at most `2⁷²`, and `None` where any `e'_i` is not finite.
fn by_definition(e: &[f64], c: &[f64], k: usize, s: &MergeStep<'_>) -> Option<Vec<f64>> {
    let n = s.n as f64;
    let m = (4.0 * n + 32.0) / 9_007_199_254_740_992.0;
    let eps = m / (1.0 - m);
    let normal = |v: f64| v.is_normal() && v > 0.0;
    if s.shares
        || eps > 1e-3
        || !(s.alpha >= 2f64.powi(-900) && s.alpha.is_finite())
        || !normal(s.within)
        || !normal(s.u_sum)
        || !(s.scale >= 0.0 && s.scale.is_finite())
        || !(s.between >= 0.0 && s.between.is_finite())
    {
        return None;
    }
    let (sigma, alpha, beta, omega) = (s.scale, s.alpha, s.between, s.within);
    let spread = n / s.u_sum + 1.0 + (n / s.u_sum).sqrt();
    let tau_n = 2.0 * spread;
    let r = 7.0 / eps * omega * tau_n * tau_n;
    let residue = if r.is_finite() {
        0.0
    } else {
        let tau = spread * 2.0 * ETA_DEF;
        7.0 * omega / eps * tau * tau
    };
    let floor = 2f64.powi(-1000);
    let out: Vec<f64> = (0..k)
        .map(|i| {
            let (g, h) = (s.big[i].abs(), s.delta[i].abs());
            let cii = c[i * k + i];
            let cii = if cii.is_nan() {
                f64::INFINITY
            } else {
                cii.max(0.0)
            };
            let nu = (1.0 + g).powi(2) + (1.0 + h).powi(2) + 3.0 + n * (4.0 + alpha);
            let under = if nu > TWO_72 { ETA_DEF * nu } else { 0.0 };
            let sum = sigma * (1.0 + eps) * e[i]
                + 10.0 * eps * cii
                + 22.0 * eps * omega * h * h
                + eps * beta * g * g
                + residue
                + under;
            sum / (1.0 - 10.0 * eps) * (1.0 + TWO_M46) + floor + floor
        })
        .collect();
    out.iter().all(|v| v.is_finite()).then_some(out)
}

impl Case {
    /// `e'` from `merged`, `None` where forgotten; and by the definition.
    fn against_definition(&self) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
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
        let now = now.get(2).map(<[f64]>::to_vec);
        (now, by_definition(&self.e, &self.c, 2, &s))
    }

    /// `merged`'s `e'`, which must be the definition's to within the
    /// roundings the two orders take (about `12 u` of a sum of nonnegative
    /// terms; `4e-15` is `36 u`).
    fn is_its_definition(&self, what: &str) -> Vec<f64> {
        let (now, def) = self.against_definition();
        let (now, def) = match (now, def) {
            (Some(a), Some(b)) => (a, b),
            (now, def) => panic!("{what}: kept {:?}, by definition {:?}", now, def),
        };
        for (a, b) in now.iter().zip(&def) {
            assert!(
                (a - b).abs() <= 4e-15 * a.abs().max(b.abs()),
                "{what}: {a:e} against {b:e} by definition"
            );
        }
        now
    }
}

/// The floor's `2⁻¹⁰⁰⁰`, written out.
fn floor() -> f64 {
    2f64.powi(-1000)
}

/// Every term of `e'` on, ordinary numbers: the factor `1 + 2⁻⁴⁶` that
/// takes in the roundings is `2⁻⁴⁵` of `e'` from `1 − 2⁻⁴⁶`.
#[test]
fn an_ordinary_merge_is_its_bound_by_definition() {
    let e = Case::ordinary().is_its_definition("ordinary");
    // Activation: the sum, not the floor, is `e'`.
    assert!(e.iter().all(|&v| v > 5e-4), "{e:?}");
}

/// `η ν_i` at the edge it is formed at: `n = 2²⁰` rows of `α = 2⁵² − 4`
/// make `n (4 + α) = 2⁷²`, where the doubles are `2²⁰` apart, so `ν_i`
/// rounds past `2⁷²` only where the rest of it is past `2¹⁹`. With `|g| = h
/// = 511`, `(1 + |g|)² + (1 + h)² + 3 = 2¹⁹ + 3` is past it and the term,
/// about `2⁻¹⁰⁰²`, is an eighth of `e'`; each of its parts is held, as `|g|
/// (1 + |g|)` or `h (1 + h)` (the doubles' `2⁹` less), `− 3`, or a row
/// count off by a factor, moves `ν_i` across the edge. At `510` it rounds
/// to `2⁷²` and the term is left to the floor.
#[test]
fn the_underflows_term_is_formed_by_definition_past_two_to_the_72() {
    let mut c = Case::ordinary();
    c.e = vec![0.0, 0.0];
    c.c = [0.0; 4];
    c.n = 1 << 20;
    c.alpha = 4_503_599_627_370_492.0; // 2⁵² − 4
    c.between = 0.0;
    c.within = f64::MIN_POSITIVE;
    c.u_sum = (1u64 << 20) as f64;
    c.big = [511.0, -510.0];
    c.delta = [-511.0, 510.0];
    let n = c.n as f64;
    assert_eq!(n * (4.0 + c.alpha), TWO_72);
    let e = c.is_its_definition("at 2⁷²");
    // Activation: formed on feature 0, about `2⁻¹⁰⁰²`, and left to the
    // floor on feature 1.
    let nu = |g: f64| (1.0 + g).powi(2) * 2.0 + 3.0 + n * (4.0 + c.alpha);
    assert!(nu(511.0) > TWO_72 && nu(510.0) == TWO_72);
    let above = e[0] - 2.0 * floor();
    assert!(
        (0.124..0.126).contains(&(above / (2.0 * floor()))),
        "{above:e}"
    );
    assert!(e[1] - 2.0 * floor() < 1e-6 * floor(), "{:e}", e[1]);
    // A row's `α` past `2¹⁰⁰`: the term on its own, far past the floor.
    c.alpha = 2f64.powi(100);
    c.n = 1;
    c.u_sum = 1.0;
    let e = c.is_its_definition("alpha 2¹⁰⁰");
    assert!(e.iter().all(|&v| v > 1e7 * floor()), "{e:?}");
}

/// The residue's `7 ω τ²/ε` formed and far past the floor: `n/Û = 2⁶⁶`,
/// so `τ = 2η (2⁶⁶ + 1 + 2³³)`, and `7 ω/ε = 2¹⁰²⁰`, so `r`, about
/// `2¹¹⁵⁴`, is not finite. `√(n/Û)` is `2⁻³³` of `spread`: the term moves
/// by `2⁻³²` with it subtracted, and by far more with it a factor; and
/// with `n/Û` read as 0, 1 or `n Û`, `spread` is at most 3 and the term,
/// formed or not, rounds to 0 beside the floor.
#[test]
fn the_residues_term_is_formed_by_definition_where_r_is_not_finite() {
    let mut c = Case::ordinary();
    c.e = vec![0.0, 0.0];
    c.c = [0.0; 4];
    c.n = 1;
    c.between = 0.0;
    c.big = [0.0, 0.0];
    c.delta = [0.0, 0.0];
    c.u_sum = 2f64.powi(-66);
    let m = 36.0 / 9_007_199_254_740_992.0;
    let eps = m / (1.0 - m);
    c.within = 2f64.powi(1020) * eps / 7.0;
    let e = c.is_its_definition("residue");
    // Activation: the term is `7 ω/ε · 4η² · 2¹³²`, about `2⁻⁹⁹⁴`: 64
    // floors, beside the two added last.
    assert!(
        e.iter().all(|&v| v > 65.0 * floor() && v < 67.0 * floor()),
        "{e:?}"
    );
    // And at `n/Û = 1`, `r` is finite and the term is left to the floor.
    c.u_sum = 1.0;
    let e = c.is_its_definition("residue at n/Û = 1");
    assert!(e.iter().all(|&v| v == 2.0 * floor()), "{e:?}");
}

/// A merge is outside the argument from `α` under `2⁻⁹⁰⁰`: kept at
/// `2⁻⁹⁰⁰`, forgotten at the double below it.
#[test]
fn alpha_under_two_to_the_minus_900_forgets_by_definition() {
    let mut c = Case::ordinary();
    c.alpha = 2f64.powi(-900);
    c.is_its_definition("alpha 2⁻⁹⁰⁰");
    c.alpha = f64::from_bits(c.alpha.to_bits() - 1);
    assert_eq!(c.against_definition(), (None, None));
}

/// `e` is known only for as many features as it was built for, and never
/// for none.
#[test]
fn the_bound_is_known_for_its_own_features_alone() {
    assert_eq!(MergeBound::default().get(0), None);
    assert_eq!(MergeBound(vec![]).get(0), None);
    assert_eq!(MergeBound(vec![1.0, 2.0]).get(1), None);
    assert_eq!(MergeBound(vec![1.0, 2.0]).get(2), Some(&[1.0, 2.0][..]));
}

/// Item 7's start, `e_i = maxⱼ |c_ij|`, infinite on every feature where an
/// entry is not finite -- not only on that entry's row and column, and not
/// passed over where it is NaN (`f64::max` passes over a NaN).
#[test]
fn a_bound_opened_on_a_matrix_with_an_entry_not_finite_is_infinite() {
    // `C` row-major: [[4, ., .], [x, 1, .], [-2, 0.5, 9]].
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let c = LowerTriangle::from(vec![4.0, 0.0, 0.0, x, 1.0, 0.0, -2.0, 0.5, 9.0]);
        let mut e = MergeBound::default();
        e.open(&c, 3);
        assert_eq!(e.get(3), Some(&[f64::INFINITY; 3][..]), "x {x}");
    }
    let c = LowerTriangle::from(vec![4.0, 0.0, 0.0, -3.0, 1.0, 0.0, -2.0, 0.5, 9.0]);
    let mut e = MergeBound::default();
    e.open(&c, 3);
    assert_eq!(e.get(3), Some(&[4.0, 3.0, 9.0][..]));
}

/// A merge of `k` features over a bound kept for another number of them
/// forgets it rather than reading one past its end or leaving it short.
#[test]
fn a_merge_over_a_bound_of_another_width_forgets_it() {
    let c = LowerTriangle::from(vec![2.0, 0.0, 0.5, 3.0]);
    let case = Case::ordinary();
    let s = MergeStep {
        n: case.n,
        scale: case.scale,
        alpha: case.alpha,
        between: case.between,
        within: case.within,
        u_sum: case.u_sum,
        big: &case.big,
        delta: &case.delta,
        shares: false,
    };
    for e in [vec![1e-3], vec![1e-3, 2e-3, 3e-3]] {
        let mut bound = MergeBound(e.clone());
        bound.merged(&c, 2, &s);
        assert!(bound.0.is_empty(), "{e:?}: {:?}", bound.0);
    }
    // At the right width it is kept.
    let mut bound = MergeBound(vec![1e-3, 2e-3]);
    bound.merged(&c, 2, &s);
    assert_eq!(bound.0.len(), 2);
}
