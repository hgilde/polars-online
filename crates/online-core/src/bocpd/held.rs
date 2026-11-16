//! What [`Bocpd`](super::Bocpd) keeps between rows that is not state (the
//! gamma terms by a run's `nu`, the chance of a break, the joint's total),
//! and the prune's decision without the call where the call cannot change it.

use super::*;

/// Whether [`Bocpd::prune`] keeps a run whose log mass is `gap` below the
/// joint's total: `exp(gap) >= prune_below`, with the exponential taken only
/// where it can decide: `band` is the gaps within a thousandth of
/// `ln(prune_below)`, and outside it the answer is the same on every libm,
/// whose `exp` is within a few last bits. A band of NaNs (a `prune_below`
/// too near the subnormals for that margin to hold) decides nothing. The
/// test is one branch on two compares taken together, which is almost
/// always the same way, where a branch on each compare followed the data.
#[inline]
pub(super) fn prune_keeps(gap: f64, prune_below: f64, band: (f64, f64)) -> bool {
    let (lo, hi) = band;
    if (gap < lo) | (gap > hi) {
        gap > hi
    } else {
        gap.exp() >= prune_below
    }
}

/// The band of [`prune_keeps`] for a `prune_below`.
pub(super) fn prune_band(prune_below: f64) -> (f64, f64) {
    if prune_below >= 1e-290 {
        let at = prune_below.ln();
        (at - 1e-3, at + 1e-3)
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// The `log_sum_exp` of the model's log joint, while it is known to be of
/// the joint as it stands: [`Bocpd::prune`] sets it when it left the joint
/// as the row's recursion made it, and every other write to the joint
/// empties it. Not state, a clone or restored model starts without it, and
/// every model is equal to every other as to it.
#[derive(Debug, Default)]
pub(super) struct HeldZ(pub(super) Option<f64>);

impl Clone for HeldZ {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for HeldZ {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// What the predictive takes of a run's `ν` alone -- [`gamma_diff`], and
/// under `gaussian` `ln(ν·π)` beside it -- held by the bits of `ν` and `a`
/// (tasks 250 and 389). A run's `ν` is the prior's plus its weight, so
/// under rows of one weight the run of length `r` asks this row for the
/// values the run of length `r` asked the last row for, and two `ln_gamma`
/// a run a row -- four logarithms and a Lanczos sum -- become a lookup. An
/// entry keeps the value the function gave for exactly those bits, so a hit
/// is the value a miss computes: two `ν` that differ in any bit, `0.0` and
/// `-0.0` included, never share an entry, and a different `a` empties the
/// table. Open addressing over a power of two at most a quarter full of the
/// runs one row asks for, so every `ν` of a row has its own entry: a
/// direct-mapped table put the 200 keys of a stream of 200 runs in 111
/// slots and hit 30% of the time. A table that fills up is emptied, which
/// a stream whose `ν` never repeat (fractional weights) does every few
/// rows, at the cost of the lookups it could not have answered. Not state:
/// a state file carries none of it, a restored or cloned model starts
/// empty, and two models are the same model whatever each has held, so
/// [`PartialEq`] ignores it.
#[derive(Debug, Default)]
pub(super) struct LengthTerms {
    /// The bits of the `a` the entries were made for, and whether they hold
    /// `ln(ν·π)` (the `gaussian` form's) or only the gamma difference.
    pub(super) a: u64,
    pub(super) with_pi: bool,
    /// A power of two, or empty before the first lookup. A key that is a
    /// NaN's bits marks an entry never filled.
    pub(super) slots: Vec<LengthSlot>,
    /// Entries filled.
    pub(super) filled: usize,
    /// Lookups answered from an entry, and computed: the tests read them to
    /// show the table was used.
    #[cfg(test)]
    pub(super) hits: u64,
    #[cfg(test)]
    pub(super) misses: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LengthSlot {
    pub(super) key: u64,
    pub(super) gamma: f64,
    pub(super) ln_pi: f64,
}

impl Clone for LengthTerms {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for LengthTerms {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl LengthTerms {
    /// The least entries a table has: a row's `runs` runs get four each, so
    /// the table is at most a quarter full of one row's keys and a half full
    /// of two rows'.
    pub(super) const MIN_SLOTS: usize = 64;

    /// No `ν` asked for here has a NaN's bits, so they mark an entry never
    /// filled.
    pub(super) const EMPTY: u64 = f64::NAN.to_bits();

    /// Ready for a row of `runs` runs asking for the terms of dimension
    /// `a`, with `ln(ν·π)` beside the gamma difference where `with_pi`. A
    /// table built for another `a` or form is emptied, and one without room
    /// for the row's keys grows and starts again.
    #[inline(never)]
    pub(super) fn begin(&mut self, runs: usize, a: f64, with_pi: bool) {
        if self.a != a.to_bits() || self.with_pi != with_pi {
            self.a = a.to_bits();
            self.with_pi = with_pi;
            self.empty();
        }
        let need = 4 * (runs + 1);
        if self.slots.len() < need.max(Self::MIN_SLOTS) {
            self.grow(need);
        }
    }

    /// A table of room for `need` entries or more, empty.
    #[cold]
    pub(super) fn grow(&mut self, need: usize) {
        self.slots.clear();
        self.slots.resize(
            need.next_power_of_two().max(Self::MIN_SLOTS),
            LengthSlot {
                key: Self::EMPTY,
                gamma: 0.0,
                ln_pi: 0.0,
            },
        );
        self.filled = 0;
    }

    /// Empty every entry, keeping the room.
    pub(super) fn empty(&mut self) {
        for slot in self.slots.iter_mut() {
            slot.key = Self::EMPTY;
        }
        self.filled = 0;
    }

    /// The entry for `ν`'s bits: where they hash to, or the next one along
    /// that holds them or was never filled. The hash is Fibonacci's, of the
    /// bits folded onto themselves: the `ν` of a stream's runs differ in the
    /// exponent and the mantissa's top, which a fold of the halves brings
    /// down to where the product's top bits, the index, read them.
    #[inline]
    pub(super) fn find(&self, bits: u64) -> usize {
        let mask = self.slots.len() - 1;
        let folded = bits ^ (bits >> 32);
        let mut at = (folded.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask;
        loop {
            let key = self.slots[at].key;
            if key == bits || key == Self::EMPTY {
                return at;
            }
            at = (at + 1) & mask;
        }
    }

    /// The entry for `nu`, made if it is not held; the table is emptied
    /// first where it would hold more than half its entries. After
    /// [`Self::begin`] for `a`.
    #[inline]
    pub(super) fn entry(&mut self, nu: f64, a: f64) -> LengthSlot {
        debug_assert!(a.to_bits() == self.a && !self.slots.is_empty());
        let bits = nu.to_bits();
        let mut at = self.find(bits);
        if self.slots[at].key == bits {
            #[cfg(test)]
            {
                self.hits += 1;
            }
            return self.slots[at];
        }
        #[cfg(test)]
        {
            self.misses += 1;
        }
        if 2 * (self.filled + 1) > self.slots.len() {
            self.empty();
            at = self.find(bits);
        }
        let made = LengthSlot {
            key: bits,
            gamma: gamma_diff(nu, a),
            // The `gaussian` form's `ln(ν·π)`, as its expression takes it.
            ln_pi: if self.with_pi {
                (nu * std::f64::consts::PI).ln()
            } else {
                0.0
            },
        };
        self.slots[at] = made;
        self.filled += 1;
        made
    }

    /// The diagonal form's per-length constants with the product split: the
    /// gamma difference and `ln(pi (kn+1)/kn)`, held by `nu`'s bits (a table
    /// begun with `with_pi`). Scratch variant 426 (c).
    #[inline]
    pub(super) fn gamma_and_ln_c(&mut self, nu: f64, kn: f64) -> (f64, f64) {
        let ln_c = |kn: f64| (std::f64::consts::PI * (kn + 1.0) / kn).ln();
        if nu.is_nan() {
            return (gamma_diff(nu, 1.0), ln_c(kn));
        }
        let bits = nu.to_bits();
        let mut at = self.find(bits);
        if self.slots[at].key == bits {
            let e = self.slots[at];
            return (e.gamma, e.ln_pi);
        }
        if 2 * (self.filled + 1) > self.slots.len() {
            self.empty();
            at = self.find(bits);
        }
        let made = LengthSlot { key: bits, gamma: gamma_diff(nu, 1.0), ln_pi: ln_c(kn) };
        self.slots[at] = made;
        self.filled += 1;
        (made.gamma, made.ln_pi)
    }

    /// [`gamma_diff`]`(nu, a)`, for the `a` and form [`Self::begin`] was
    /// given.
    #[inline]
    pub(super) fn gamma(&mut self, nu: f64, a: f64) -> f64 {
        if nu.is_nan() {
            return gamma_diff(nu, a);
        }
        self.entry(nu, a).gamma
    }

    /// [`gamma_diff`]`(nu, a)` and `ln(nu·π)`.
    #[inline]
    pub(super) fn gamma_and_ln_pi(&mut self, nu: f64, a: f64) -> (f64, f64) {
        if nu.is_nan() {
            return (gamma_diff(nu, a), (nu * std::f64::consts::PI).ln());
        }
        let e = self.entry(nu, a);
        (e.gamma, e.ln_pi)
    }
}

/// The chance of a break the last row took, by the bits of what it came
/// from -- the row's hazard under a per-row hazard, the step under one on
/// the clock -- so that a stream of one hazard or of regular steps takes the
/// two logarithms of [`Branches`] once (task 389). Not state, and every
/// model is equal to every other as to it.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct HeldBranches(pub(super) Option<(u64, Branches)>);

impl PartialEq for HeldBranches {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// What the full covariance's predictive keeps between runs: the factor
/// working space, and the empty run's factor. One box, so the diagonal
/// emission's working space is no larger for either.
#[derive(Default)]
pub(super) struct GaussWork {
    pub(super) work: FactorWork,
    pub(super) held: Option<Box<HeldPrior>>,
}

/// The empty run's scale `Σ` -- the prior's `Ψ₀` scaled by `(κ₀+1)/(κ₀ν₀')`,
/// the same matrix every row until the prior changes -- as its lower
/// triangle's bits, and the factor of it. A factor is a function of those
/// bits alone, so a row whose empty run's `Σ` has them reads this one where
/// it would factorize afresh: the same factor, to the bit. Keyed by the bits
/// themselves, it is emptied by nothing: a prior that changes (the first rows
/// setting it) makes a different key, and a clone or a restored model
/// starts without it, as it does the rest of [`Work`].
pub(super) struct HeldPrior {
    key: Vec<u64>,
    pub(super) factor: SpdFactor,
}

impl HeldPrior {
    /// Whether this is the factor of the lower triangle of `sigma` (`d×d`).
    pub(super) fn is_of(&self, sigma: &[f64], d: usize) -> bool {
        let mut at = self.key.iter();
        (0..d).all(|i| (0..=i).all(|j| at.next() == Some(&sigma[i * d + j].to_bits())))
            && at.next().is_none()
    }

    pub(super) fn new(sigma: &[f64], d: usize, factor: SpdFactor) -> Box<Self> {
        let key = (0..d)
            .flat_map(|i| (0..=i).map(move |j| (i, j)))
            .map(|(i, j)| sigma[i * d + j].to_bits())
            .collect();
        Box::new(Self { key, factor })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OnlineModel;
    use crate::bocpd::tests::{Normals, cfg};

    /// `prune_keeps` is `exp(gap) >= prune_below` for every gap, including
    /// the ones within the band, and for a `prune_below` near the subnormals,
    /// where the band is off (task 419).
    #[test]
    fn the_prune_decision_is_the_exponentials_to_the_bit() {
        let mut g = Normals::new(420);
        let mut pbs = vec![0.5, 1e-6, 1e-12, 1e-289, 1e-290, 1e-300, 1e-310, 5e-324];
        for _ in 0..300 {
            pbs.push((-(g.unit() * 700.0)).exp());
        }
        let mut decided = 0u64;
        let mut called = 0u64;
        for pb in pbs {
            let band = prune_band(pb);
            let at = pb.ln();
            let mut gaps = vec![
                f64::NEG_INFINITY,
                f64::NAN,
                0.0,
                -0.0,
                at,
                -1e3,
                -800.0,
                -740.0,
            ];
            for _ in 0..2_000 {
                gaps.push(at + (g.unit() - 0.5) * 6e-3);
                gaps.push(at + (g.unit() - 0.5) * 2.0);
                gaps.push(at + (g.unit() - 0.5) * 2e-12);
                gaps.push(-g.unit() * 50.0);
            }
            for gap in gaps {
                assert_eq!(
                    prune_keeps(gap, pb, band),
                    gap.exp() >= pb,
                    "pb {pb:e} gap {gap:e}"
                );
                if (gap < band.0) | (gap > band.1) {
                    decided += 1;
                } else {
                    called += 1;
                }
            }
        }
        assert!(decided > 0 && called > 0, "both paths were taken");
        assert!(prune_band(1e-310).0.is_nan() && !prune_band(1e-6).0.is_nan());
    }

    /// What a model holds across rows -- the joint's `log_sum_exp` -- is
    /// the value the row would compute: a model that forgets it before
    /// every row is the same to the bit in
    /// every output and in the whole state, over diagonal and full emissions,
    /// default pruning and none, a small cap, fractional weights, rows of no
    /// weight, refused rows, a per-row and a clock hazard, and a save and a
    /// restore at the middle (task 419).
    #[test]
    fn the_held_joint_total_changes_no_bit() {
        use crate::OnlineModel;
        let mut used = 0usize;
        for (emission, d) in [(BocpdEmission::Diag, 3), (BocpdEmission::Gaussian, 2)] {
            for (prune_below, max_run) in [(1e-6, 200), (0.0, 25), (1e-2, 20), (1e-6, 5)] {
                for (weights, on_clock, edges) in [
                    (false, false, false),
                    (true, false, false),
                    (false, true, false),
                    (false, false, true),
                    (true, true, true),
                ] {
                    let c = BocpdCfg {
                        emission,
                        prior_nu: Some(d as f64 + 2.0),
                        prune_below,
                        max_run,
                        hazard_on_clock: on_clock,
                        hazard: if on_clock { 50.0 } else { 80.0 },
                        ..cfg(d)
                    };
                    let mut keeps = Bocpd::new(c.clone()).unwrap();
                    let mut forgets = Bocpd::new(c).unwrap();
                    let mut g = Normals::new(7 + max_run as u64);
                    let (mut a, mut b) = (crate::Step::default(), crate::Step::default());
                    for t in 0..700usize {
                        let level = if (t / 150) % 2 == 0 { 0.0 } else { 4.0 };
                        let mut x: Vec<f64> = (0..d).map(|_| g.normal() + level).collect();
                        let mut w = if weights { 0.25 + 1.5 * g.unit() } else { 1.0 };
                        let mut gap = if on_clock { 0.5 + 2.0 * g.unit() } else { 1.0 };
                        if edges && t % 41 == 5 {
                            x[0] = f64::NAN;
                        }
                        if edges && t % 29 == 3 {
                            w = 0.0;
                        }
                        if edges && t % 53 == 7 {
                            gap = 30.0;
                        }
                        forgets.joint_z = HeldZ::default();
                        keeps.step_into(&x, &[], gap, w, &mut a);
                        forgets.step_into(&x, &[], gap, w, &mut b);
                        used += keeps.joint_z.0.is_some() as usize;
                        assert_eq!(a.pred.len(), b.pred.len());
                        for (p, q) in a.pred.iter().zip(&b.pred) {
                            assert_eq!(
                                p.to_bits(),
                                q.to_bits(),
                                "{emission:?} {prune_below} row {t}"
                            );
                        }
                        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
                        if t == 350 {
                            // A restored model has held nothing, and goes on.
                            let st = keeps.state();
                            keeps = Bocpd::restore(&st).unwrap();
                        }
                    }
                    assert_eq!(keeps, forgets);
                    let (sa, sb) = (
                        rmp_serde::to_vec_named(&keeps.state()).unwrap(),
                        rmp_serde::to_vec_named(&forgets.state()).unwrap(),
                    );
                    assert_eq!(sa, sb, "the saved bytes");
                }
            }
        }
        // The held total was used: the cases above left it set on rows.
        assert!(used > 1000, "the held total was used {used} times");
    }

    /// The empty run's held factor is the factor of its scale made afresh,
    /// to the bit, and every output with it: over several orders, priors
    /// given whole (a scalar, a full matrix) and set from the first rows,
    /// `κ₀`, `ν₀`, a shift of level and a change in the middle. A model that
    /// forgets it before every row is the oracle; the factor it holds is
    /// compared with the one `SpdFactor::of` makes of the scale written
    /// longhand; and a restored model starts without it and makes it again.
    #[test]
    fn the_held_empty_run_factor_changes_no_bit() {
        let mut held_rows = 0;
        for d in [1usize, 2, 3, 5, 8] {
            for scale in 0..4 {
                for (kappa, extra) in [(0.5, 1.5), (1.0, 2.0), (7.0, 9.0)] {
                    let full: Vec<f64> = (0..d * d)
                        .map(|i| {
                            let (r, c) = (i / d, i % d);
                            if r == c {
                                2.0 + r as f64
                            } else {
                                0.3 / (1.0 + (r + c) as f64)
                            }
                        })
                        .collect();
                    let (mean, psi) = match scale {
                        0 => (None, None),
                        1 => (Some(vec![0.5; d]), Some(vec![1.7])),
                        2 => (Some(vec![-1.0; d]), Some(full)),
                        _ => (None, Some(vec![0.25])),
                    };
                    let c = BocpdCfg {
                        emission: BocpdEmission::Gaussian,
                        prior_mean: mean,
                        prior_scale: psi,
                        prior_kappa: kappa,
                        prior_nu: Some(d as f64 + extra),
                        prune_below: 1e-6,
                        warm_rows: (scale == 0 || scale == 3).then_some(d + 3),
                        ..cfg(d)
                    };
                    let mut kept = Bocpd::new(c).unwrap();
                    let mut fresh = kept.clone();
                    let mut n = Normals::new(7 + d as u64);
                    let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
                    for t in 0..260 {
                        let shift = if t < 130 { 0.0 } else { 5.0 };
                        let x: Vec<f64> = (0..d).map(|_| shift + n.normal()).collect();
                        fresh.work = Work::default();
                        held_rows += usize::from(
                            kept.work
                                .run
                                .factor
                                .as_ref()
                                .is_some_and(|g| g.held.is_some()),
                        );
                        if t == 100 {
                            // A load starts without it, and makes it again.
                            let bytes = format!("{:?}", kept.state());
                            kept = Bocpd::restore(&kept.state()).unwrap();
                            assert!(kept.work.run.factor.is_none());
                            assert_eq!(bytes, format!("{:?}", kept.state()));
                        }
                        let a = kept.step(&x, &[], 1.0, 1.0);
                        let b = fresh.step(&x, &[], 1.0, 1.0);
                        assert_eq!(bits(&a.pred), bits(&b.pred), "d {d} scale {scale} row {t}");
                        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
                    }
                    if let Some(h) = kept.work.run.factor.as_ref().and_then(|g| g.held.as_ref()) {
                        let (k0, nu0) = (kept.cfg.prior_kappa, kept.cfg.nu0());
                        let s = (k0 + 1.0) / (k0 * (nu0 - d as f64 + 1.0));
                        let psi0 = kept.psi0();
                        let mut sigma = vec![0.0; d * d];
                        for i in 0..d {
                            for j in 0..=i {
                                sigma[i * d + j] = (psi0[i * d + j] + (0.0 + 0.0)) * s;
                            }
                        }
                        assert!(h.factor == SpdFactor::of(&sigma, d).unwrap(), "d {d}");
                        assert!(h.is_of(&sigma, d));
                        // One bit of any entry of the triangle is another key.
                        for i in 0..d {
                            for j in 0..=i {
                                let mut other = sigma.clone();
                                other[i * d + j] = f64::from_bits(other[i * d + j].to_bits() ^ 1);
                                assert!(!h.is_of(&other, d), "d {d} at {i},{j}");
                            }
                        }
                    }
                    assert_eq!(
                        format!("{:?}", kept.state()),
                        format!("{:?}", fresh.state())
                    );
                }
            }
        }
        assert!(held_rows > 40 * 200, "{held_rows}");
    }
}
