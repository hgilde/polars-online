//! Test-only: a running bound on the rounding [`super::EwCov`]'s rank-one
//! updates leave in its moments, against the exact weighted moments of the
//! same rows (docs/PLAN.md task 274).
//!
//! The update is weighted Welford's: `W' = λW + w`, `a = λW/W'`, `b =
//! w/W'`, `d = x − m` against the old mean (a compensated pair, `(x − hi)
//! − lo`), `C' = aC + (ab·dᵢ)·dⱼ`, `m' = m + b·d`, `Q' = (λλ)Q + w²`. With
//! starred values exact and `ε` relative errors, to first order:
//!
//! - `ε_W' ≤ a(ε_W + ε_λ + u) + u`, the product and the sum;
//! - with `η = ε_W + ε_λ + u` the error of `fl(λW)`, `a = a*(1 + bη)` and
//!   `b = b*(1 − aη)` to first order, so `ε_a ≤ bη + 2u`, `ε_b ≤ aη + 2u`;
//! - `|Δd| ≤ |Δm| + 2u|d|`;
//! - `m' − m*' = (1 − b)(m − m*) + b·Δd_r + Δb·d` plus the step's own
//!   rounding, so `|Δm'| ≤ a|Δm| + b|d|(ε_b + 2u) + 3u|bd|`, the pair
//!   taking the sum itself to within `u` of the step;
//! - the term `T = ab·dᵢdⱼ` within `|T|(ε_a + ε_b + 3u) + ab(|Δdᵢ||dⱼ| +
//!   |dᵢ||Δdⱼ|)`, and `|ΔC'| ≤ a|ΔC| + a|C|(ε_a + u) + |ΔT| + u|C'|`;
//! - `ε_Q' ≤ (λ²Q(ε_Q + 2ε_λ + 2u) + u w² + u Q')/Q'`.
//!
//! `ε_λ` is how far each decay factor the model is handed may be off the
//! one the definition decays by: 0 where both read the same `λ`. The
//! magnitudes the bounds are read off are the recursion replayed here in
//! plain arithmetic; no expected value is taken from it. The blocked path
//! ([`super::EwCov::set_block_rows`]) and windows are other algorithms
//! and are not tracked.

use crate::oracle::U;

#[derive(Clone, Debug)]
pub(crate) struct Rounding {
    k: usize,
    lam_err: f64,
    /// The replayed weight, mean and co-moments, row-major.
    pub(crate) w: f64,
    pub(crate) q: f64,
    pub(crate) m: Vec<f64>,
    pub(crate) c: Vec<f64>,
    /// `ε_W` and `ε_Q`, relative; `|Δm|` and `|ΔC|`, absolute.
    pub(crate) ew: f64,
    pub(crate) eq: f64,
    pub(crate) em: Vec<f64>,
    pub(crate) ec: Vec<f64>,
}

impl Rounding {
    /// An empty accumulator of `k` columns, each decay factor it is handed
    /// within `lam_err` of the definition's, relative.
    pub(crate) fn new(k: usize, lam_err: f64) -> Self {
        Self {
            k,
            lam_err,
            w: 0.0,
            q: 0.0,
            m: vec![0.0; k],
            c: vec![0.0; k * k],
            ew: 0.0,
            eq: 0.0,
            em: vec![0.0; k],
            ec: vec![0.0; k * k],
        }
    }

    /// [`super::EwCov::update`] of `x` at decay `lam` and weight `w`.
    pub(crate) fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        let lw = lam * self.w;
        if w == 0.0 && lw == 0.0 {
            return self.decay(lam);
        }
        let w_new = lw + w;
        let (a, b) = (lw / w_new, w / w_new);
        let ew_new = if lw > 0.0 {
            a * (self.ew + self.lam_err + U) + U
        } else {
            0.0
        };
        let q_new = lam * lam * self.q + w * w;
        self.eq = if q_new > 0.0 {
            (lam * lam * self.q * (self.eq + 2.0 * self.lam_err + 2.0 * U) + U * w * w + U * q_new)
                / q_new
        } else {
            0.0
        };
        if w == 0.0 {
            // `a` is 1 and `b` 0 exactly: the moments stand.
            self.w = w_new;
            self.ew = ew_new;
            return;
        }
        let k = self.k;
        // `fl(λŴ) = λW(1 + η)`, `η = ε_W + ε_λ + u`, so `a = fl(λŴ)/(fl(λŴ)
        // + w)` is `a*(1 + bη)` and `b` is `b*(1 − aη)`, each with two
        // roundings of its own: the weight's error is shared by the two
        // ratios, not added to each.
        let eta = self.ew + self.lam_err + U;
        let ea = if lw > 0.0 { b * eta + 2.0 * U } else { 0.0 };
        let eb = a * eta + 2.0 * U;
        let d: Vec<f64> = x.iter().zip(&self.m).map(|(xi, mi)| xi - mi).collect();
        let dd: Vec<f64> = d
            .iter()
            .zip(&self.em)
            .map(|(di, e)| e + 2.0 * U * di.abs())
            .collect();
        for i in 0..k {
            for j in 0..k {
                let v = i * k + j;
                let t = a * b * d[i] * d[j];
                let dt = t.abs() * (ea + eb + 3.0 * U)
                    + a * b * (dd[i] * d[j].abs() + d[i].abs() * dd[j]);
                let c_new = a * self.c[v] + t;
                self.ec[v] = a * self.ec[v] + a * self.c[v].abs() * (ea + U) + dt + U * c_new.abs();
                self.c[v] = c_new;
            }
        }
        for ((m, em), di) in self.m.iter_mut().zip(self.em.iter_mut()).zip(&d) {
            let step = b * di;
            *em = a * *em + b * di.abs() * (eb + 2.0 * U) + 3.0 * U * step.abs();
            *m += step;
        }
        self.w = w_new;
        self.q = q_new;
        self.ew = ew_new;
    }

    /// [`super::EwCov::decay`]: the weights age, the moments stand.
    pub(crate) fn decay(&mut self, lam: f64) {
        self.w *= lam;
        self.ew += self.lam_err + U;
        self.q *= lam * lam;
        self.eq += 2.0 * self.lam_err + 2.0 * U;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMix64;

    /// The bounds hold an `EwCov`'s moments to their exact values, on a
    /// stream whose exact moments are rationals held in integers: whole
    /// features at a level of 1,000, whole weights (0 among them), and
    /// decays of 1 or ½, so every weight is a dyadic number. The worst
    /// share of each bound the error took is printed; a bound that the
    /// error never came within a thousandth of would not bite.
    #[test]
    fn the_bounds_hold_the_moments_to_their_exact_values() {
        let k = 3;
        let mut worst = [0.0f64; 3];
        for seed in 0..20u64 {
            let mut rng = SplitMix64::new(seed);
            let mut ew = crate::EwCov::new(k);
            let mut tr = Rounding::new(k, 0.0);
            // Exact sums at weights scaled by `2^SCALE`, the weight of the
            // newest row; each halving shifts every older one down.
            const SCALE: u32 = 24;
            let mut rows: Vec<(Vec<i128>, i128)> = Vec::new();
            let mut halvings = 0u32;
            for i in 0..60 {
                let lam = if i > 0 && rng.uniform() < 0.3 {
                    0.5
                } else {
                    1.0
                };
                if lam == 0.5 {
                    halvings += 1;
                    for r in rows.iter_mut() {
                        r.1 /= 2;
                    }
                }
                assert!(halvings < SCALE, "the exact weights stay whole");
                let x: Vec<i128> = (0..k)
                    .map(|_| 1000 + (rng.uniform() * 41.0) as i128 - 20)
                    .collect();
                let w = if i > 0 && rng.uniform() < 0.15 {
                    0
                } else {
                    1 + (rng.uniform() * 4.0) as i128
                };
                let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                ew.update(&xf, lam, w as f64);
                tr.update(&xf, lam, w as f64);
                rows.push((x, w << SCALE));
                let s0: i128 = rows.iter().map(|r| r.1).sum();
                if s0 == 0 {
                    continue;
                }
                let s1: Vec<i128> = (0..k)
                    .map(|a| rows.iter().map(|r| r.1 * r.0[a]).sum())
                    .collect();
                for a in 0..k {
                    let m = s1[a] as f64 / s0 as f64;
                    let err = (ew.mean(a) - m).abs();
                    let bound = tr.em[a] + 2.0 * U * m.abs();
                    assert!(
                        err <= bound,
                        "seed {seed} row {i} mean {a}: {err:e} > {bound:e}"
                    );
                    worst[0] = worst[0].max(err / bound);
                    for b in 0..k {
                        let s2: i128 = rows.iter().map(|r| r.1 * r.0[a] * r.0[b]).sum();
                        let num = s2 * s0 - s1[a] * s1[b];
                        let c = num as f64 / (s0 as f64 * s0 as f64);
                        let err = (ew.cov(a, b) - c).abs();
                        let bound = tr.ec[a * k + b] + 4.0 * U * c.abs();
                        assert!(
                            err <= bound,
                            "seed {seed} row {i} cov {a},{b}: {err:e} > {bound:e}"
                        );
                        worst[1] = worst[1].max(err / bound);
                    }
                }
                let w_exact = s0 as f64 / (1u64 << SCALE) as f64;
                let err = (ew.n_eff() - w_exact).abs();
                assert!(
                    err <= tr.ew * w_exact + U * w_exact,
                    "seed {seed} row {i} weight"
                );
                worst[2] = worst[2].max(err / (tr.ew * w_exact + U * w_exact));
            }
        }
        eprintln!(
            "worst error / bound: means {:.2e}, co-moments {:.2e}, weight {:.2e}",
            worst[0], worst[1], worst[2]
        );
        assert!(
            worst[1] > 1e-3,
            "the co-moments' bound bites: {:.2e}",
            worst[1]
        );
    }
}
