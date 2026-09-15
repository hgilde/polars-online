//! Targets with gaps (docs/PLAN.md task 81; the code review of 2026-09-12,
//! N3): what a regression on several targets over one feature row keeps, and
//! which rows each target's fit is read from, when a target is null on some
//! of them. Shared by `ewridge` and `lasso`.
//!
//! Each target `j` keeps, over the rows it is present on, its weight `W_j`,
//! the target's mean `ȳ_j`, the mean `m_j` of the feature row `z`, and the
//! centred cross-moment `c_j = E[(z − m_j)(y_j − ȳ_j)]` ([`Cross`]). With an
//! intercept its slopes solve the centred system
//!
//! ```text
//! (C + ridge·I)·β = c_j        β_0 = ȳ_j − m_j·β
//! ```
//!
//! and [`TargetGaps`] says which rows the centred Gram `C` is over: the
//! target's own (`own_rows`, the fit of the rows with its nulls dropped), or
//! every row (`pairwise`). `m_j` is then the Gram's own mean under
//! `own_rows`, and the Gram's mean plus the target's offset under
//! `pairwise`. They used to be read over every row against the right-hand
//! side `c_j + (m_j − m)·ȳ_j` -- what the raw normal equations give for a
//! Gram and a cross-moment taken over different rows -- which added
//! `(m_j − m)·ȳ_j / Var(x)` to a slope: a fit that moved with the target's
//! level, `m` a feature's mean over every row. `pairwise` costs the same
//! without the term, so the old reading is gone.
//!
//! Under `own_rows` targets present on the same rows share a Gram
//! ([`Grams`]). A Gram learns the rows any of its targets has and ages over
//! the rows none has; where its targets part -- some present on a row, some
//! not -- the absent ones take a copy of it as it stood before the row and
//! keep their own from then on. A target present on every row reads the Gram
//! `pairwise` would give it, and a single target never copies.

use serde::{Deserialize, Serialize};

use crate::window::EMPTY_FRACTION;
use crate::{EwCov, Moments, TargetMoments};

/// Which rows a target's Gram is taken over, where the target is null on
/// some (docs/PLAN.md task 81; the module docs have the math).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetGaps {
    /// Each target's Gram is over exactly the rows it is present on, so its
    /// fit is the fit of the rows with its nulls dropped. Targets present on
    /// the same rows share one Gram; a bank of targets costs one `k×k` Gram
    /// per pattern of missing rows.
    #[default]
    OwnRows,
    /// One Gram over every row, and each target's cross-moments over its own
    /// rows, centred at its own means: pairwise-complete moments, as pandas'
    /// `DataFrame.cov` takes them. One Gram whatever the gaps. Exact when a
    /// target's gaps have nothing to do with the features; where they do,
    /// each slope is scaled by the ratio of the feature's variance over the
    /// target's rows to its variance over every row.
    Pairwise,
}

/// A row's share `w / W'` of an accumulated weight `W' = lam·W + w`, as
/// [`EwCov::update`] forms it -- `decayed` is `lam·W` -- and 0 where `W'` is:
/// nothing has been carried, and nothing moves (hard rule 9).
pub(crate) fn row_share(decayed: f64, w: f64) -> f64 {
    let w_new = decayed + w;
    if w_new > 0.0 { w / w_new } else { 0.0 }
}

/// Each target's cross-moments with the feature row `z`, centred (review
/// 2026-09-12, N1), and the weight and mean of `z` over every row. Over the
/// rows target `j` was present on: its EW mean `ȳ_j` (`my`), the centred
/// cross-moment `c_j = E[(z − m_j)(y_j − ȳ_j)]` (`c`), and the mean `m_j` of
/// `z` there, kept as its offset `δ_j = m_j − m` (`d`) from the mean `m` of
/// `z` over every row (`m`), whose weight is `w`.
///
/// They were kept raw, `r_j = E[z·y_j]`, and the solves with an intercept
/// formed `E[z·y] − m·ȳ` from them, or read the raw normal equations: two
/// numbers the size of `level²` subtracted to leave one the size of a
/// covariance, which at `1e8` left nothing of the fit. The raw moment,
/// `r_j = c_j + m_j·ȳ_j`, is one step away for what still reads it.
///
/// `δ_j` is a number of its own, updated from deviations, rather than a
/// difference formed from two means: it is exactly 0 while the target has
/// been present on every row, where two level-sized means would differ by
/// `level·ε`. `m` and `w` are over every row whatever the Grams learn: `w` is
/// the model's `n_eff` (hard rule 8), and `m` is what the offsets are kept
/// against. A Gram over every row takes the same steps, so unblocked the two
/// agree with it to the bit; a blocked Gram brings its own mean up to date
/// only at a flush, and the offsets need one on every row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Cross {
    pub(crate) w: f64,
    pub(crate) m: Vec<f64>,
    pub(crate) d: Vec<Vec<f64>>,
    pub(crate) my: Vec<f64>,
    pub(crate) c: Vec<Vec<f64>>,
}

impl Cross {
    pub(crate) fn new(n_targets: usize, k: usize) -> Self {
        Self {
            w: 0.0,
            m: vec![0.0; k],
            d: vec![vec![0.0; k]; n_targets],
            my: vec![0.0; n_targets],
            c: vec![vec![0.0; k]; n_targets],
        }
    }

    /// A row's share of the weight over every row, at decay `lam` and
    /// weight `w`: the `b` of [`Cross::learn`], [`Cross::miss`] and
    /// [`Cross::advance`].
    pub(crate) fn share(&self, lam: f64, w: f64) -> f64 {
        row_share(lam * self.w, w)
    }

    /// Target `j` present on a row, with the `a_j`/`b_j` of its own weight's
    /// update and the row's share `b` of the all-row weight. With `u = z − m`
    /// against the old all-row mean, `m_j` takes `b_j` of `z − m_j = u − δ_j`
    /// and `m` takes `b` of `u`, so
    ///
    /// ```text
    /// δ_j' = (1 − b_j)·δ_j + (b_j − b)·u
    /// ```
    ///
    /// written so that a target's first row (`b_j = 1`) drops the old offset
    /// outright and a target present on every row (`b_j = b`) leaves 0 at 0,
    /// both exactly. The deviations are from the means before the row, so
    /// this comes before [`Cross::advance`].
    pub(crate) fn learn(&mut self, j: usize, z: &[f64], y: f64, aj: f64, bj: f64, b: f64) {
        let dy = y - self.my[j];
        let ab_dy = aj * bj * dy;
        let (d, c) = (&mut self.d[j], &mut self.c[j]);
        for (((dji, cji), &zi), &mi) in d.iter_mut().zip(c.iter_mut()).zip(z).zip(&self.m) {
            let u = zi - mi;
            *cji = aj * *cji + ab_dy * (u - *dji);
            *dji = (1.0 - bj) * *dji + (bj - b) * u;
        }
        self.my[j] += bj * dy;
    }

    /// Target `j` absent from a row -- null, or nothing to learn from -- that
    /// moves the all-row mean by `b·u`: its own mean stays, so its offset
    /// takes the step back.
    pub(crate) fn miss(&mut self, j: usize, z: &[f64], b: f64) {
        if b > 0.0 {
            for ((dji, &zi), &mi) in self.d[j].iter_mut().zip(z).zip(&self.m) {
                *dji -= b * (zi - mi);
            }
        }
    }

    /// The all-row mean and weight take the row, after every target has read
    /// it. The weight by `EwCov::update`'s recursion, including its refusal
    /// of a row that would leave no weight at all.
    pub(crate) fn advance(&mut self, z: &[f64], lam: f64, w: f64, b: f64) {
        if b > 0.0 {
            for (mi, &zi) in self.m.iter_mut().zip(z) {
                *mi += b * (zi - *mi);
            }
        }
        let w_new = lam * self.w + w;
        if w_new > 0.0 {
            self.w = w_new;
        }
    }

    /// Target `j`'s uncentred `E[z·y_j] = c_j + (m + δ_j)·ȳ_j`.
    pub(crate) fn raw(&self, j: usize) -> Vec<f64> {
        let my = self.my[j];
        self.c[j]
            .iter()
            .zip(&self.d[j])
            .zip(&self.m)
            .map(|((c, d), m)| c + (m + d) * my)
            .collect()
    }

    /// Mix toward the twin's as [`Grams::blend`] mixes the Grams: `(a, b)`
    /// over every row, with `w_new` the mixed all-row weight, and `per[j] =
    /// (a_j, b_j)` over target `j`'s own rows (`None` where neither side has
    /// any). The centred mixture, `c = a_j·c + b_j·c' + a_j·b_j·(m_j −
    /// m_j')(ȳ − ȳ')`, as the co-moments' (C16); and since `m_j = m + δ_j` on
    /// both sides, the offsets mix as `δ = a_j·δ + b_j·δ' + (b_j − b)·(m' −
    /// m)`.
    pub(crate) fn blend(
        &mut self,
        other: &Self,
        a: f64,
        b: f64,
        w_new: f64,
        per: &[Option<(f64, f64)>],
    ) {
        // `m' − m`, the twin's all-row mean from this one's.
        let dm: Vec<f64> = other.m.iter().zip(&self.m).map(|(o, s)| o - s).collect();
        for (j, p) in per.iter().enumerate() {
            let Some((aj, bj)) = *p else { continue };
            let dy = self.my[j] - other.my[j];
            let mine = self.c[j].iter_mut().zip(self.d[j].iter_mut());
            let theirs = other.c[j].iter().zip(&other.d[j]);
            for (((c, d), (oc, od)), &dmi) in mine.zip(theirs).zip(&dm) {
                // `m_j − m_j' = (δ_j − δ_j') − (m' − m)`.
                let dz = (*d - od) - dmi;
                *c = aj * *c + bj * oc + aj * bj * dz * dy;
                *d = aj * *d + bj * od + (bj - b) * dmi;
            }
            self.my[j] = aj * self.my[j] + bj * other.my[j];
        }
        for (m, om) in self.m.iter_mut().zip(&other.m) {
            *m = a * *m + b * om;
        }
        self.w = w_new;
    }

    /// The rows after a snapshot `old`, `f` the decay since it, as
    /// `crate::truncated` takes them from a Gram: `None` when no weight is
    /// left inside the window. With `ratio = W_u/W_R` over every row and
    /// `per[j] = (ratio_j, g_j)` -- `W_u,j/W_R,j` and `W_j/W_R,j` -- over
    /// target `j`'s own (`None` where nothing of it is left), the pooling
    /// identity again, `c_R = g_j·c − ratio_j·c_u − ratio_j·g_j·(m_j,u −
    /// m_j)(ȳ_u − ȳ)`; the means by `x_R = x − ratio·(x_u − x)`; and the
    /// offset from offsets, `δ_R = δ − ratio_j·(δ_u − δ) + (ratio −
    /// ratio_j)·(m_u − m)`.
    pub(crate) fn truncated(&self, old: &Self, f: f64, per: &[Option<(f64, f64)>]) -> Option<Self> {
        let w_old = f * old.w;
        let w = self.w - w_old;
        if w <= EMPTY_FRACTION * self.w || !w.is_finite() {
            return None;
        }
        let ratio = w_old / w;
        let mut out = Self::new(self.my.len(), self.m.len());
        out.w = w;
        // `m_u − m`.
        let du: Vec<f64> = old.m.iter().zip(&self.m).map(|(u, m)| u - m).collect();
        for ((o, m), dui) in out.m.iter_mut().zip(&self.m).zip(&du) {
            *o = m - ratio * dui;
        }
        for (j, p) in per.iter().enumerate() {
            let Some((rj, gj)) = *p else { continue };
            let dy = old.my[j] - self.my[j];
            let into = out.c[j].iter_mut().zip(out.d[j].iter_mut());
            let now = self.c[j].iter().zip(&self.d[j]);
            let then = old.c[j].iter().zip(&old.d[j]);
            for ((((oc, od), (c, d)), (cu, du_j)), &dui) in into.zip(now).zip(then).zip(&du) {
                // `δ_u − δ`, and `m_j,u − m_j` from it.
                let dd = du_j - d;
                let dz = dui + dd;
                *oc = gj * c - rj * cu - rj * gj * dz * dy;
                *od = d - rj * dd + (ratio - rj) * dui;
            }
            out.my[j] = self.my[j] - rj * dy;
        }
        Some(out)
    }

    /// Whether the moments are those of `n_targets` targets over `k` slots.
    fn has_shape(&self, n_targets: usize, k: usize) -> bool {
        self.m.len() == k
            && self.my.len() == n_targets
            && self.d.len() == n_targets
            && self.c.len() == n_targets
            && self.d.iter().chain(&self.c).all(|v| v.len() == k)
    }
}

/// Per-Gram scratch for one row: whether any of its targets is present
/// ([`PRESENT`]) and whether any is absent ([`ABSENT`]). Not state: equal
/// whatever it holds, and never written.
#[derive(Debug, Clone, Default)]
struct Flags(Vec<u8>);

impl PartialEq for Flags {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

const PRESENT: u8 = 1;
const ABSENT: u8 = 2;

/// The feature Grams a regression's targets are fitted from, and which one
/// each reads (see the module docs). One under `pairwise`, over every row;
/// under `own_rows` one per set of targets present on the same rows. A
/// Gram's weight is each of its targets' weight to the bit: the two take the
/// same steps from the same start, the Gram's by [`EwCov::update`] and
/// [`EwCov::skip`], the target's by [`Acc::learn`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Grams {
    pub(crate) grams: Vec<EwCov>,
    /// The Gram each target reads, an index into `grams`.
    pub(crate) of: Vec<usize>,
    #[serde(skip)]
    flags: Flags,
}

impl Grams {
    pub(crate) fn new(n_targets: usize, k: usize, block_rows: usize) -> Self {
        let mut g = EwCov::new(k);
        g.set_block_rows(block_rows);
        Self {
            grams: vec![g],
            of: vec![0; n_targets],
            flags: Flags::default(),
        }
    }

    /// The row `z`, at decay `lam` and weight `w`, with the targets `y`.
    /// Under `pairwise` every Gram learns it -- there is one. Under
    /// `own_rows` a Gram learns it when one of its targets is present, ages
    /// over it when none is, and when some are and some are not, the absent
    /// ones move to a copy made before the row, which then ages over it: so
    /// a target's Gram is always over exactly its rows.
    ///
    /// A row with no weight learns nothing, so it parts no targets while
    /// there is weight to age: taking it with weight 0 ages the Gram by the
    /// same steps as skipping it, to the bit, so the copy would be the Gram
    /// itself, kept twice from then on. Only a zero-weight row that also
    /// takes all of the weight (`lam = 0`, an infinite clock gap) splits
    /// them, since there [`EwCov::update`] refuses the row and
    /// [`EwCov::skip`] empties the Gram.
    pub(crate) fn update(
        &mut self,
        z: &[f64],
        y: &[Option<f64>],
        lam: f64,
        w: f64,
        gaps: TargetGaps,
    ) {
        let Self { grams, of, flags } = self;
        if gaps == TargetGaps::Pairwise {
            for g in grams.iter_mut() {
                g.update(z, lam, w);
            }
            return;
        }
        debug_assert_eq!(of.len(), y.len());
        let n = grams.len();
        flags.0.clear();
        flags.0.resize(n, 0);
        for (&o, yj) in of.iter().zip(y) {
            flags.0[o] |= if yj.is_some() { PRESENT } else { ABSENT };
        }
        for g in 0..n {
            match flags.0[g] {
                PRESENT => grams[g].update(z, lam, w),
                ABSENT => grams[g].skip(z, lam),
                // A Gram every target has left; there is none.
                0 => {}
                // A row with no weight, and weight to age: no split.
                _ if w == 0.0 && lam * grams[g].n_eff() > 0.0 => grams[g].update(z, lam, w),
                // Its targets part on this row.
                _ => {
                    let id = grams.len();
                    let mut own = grams[g].clone();
                    own.skip(z, lam);
                    grams.push(own);
                    for (o, yj) in of.iter_mut().zip(y) {
                        if *o == g && yj.is_none() {
                            *o = id;
                        }
                    }
                    grams[g].update(z, lam, w);
                }
            }
        }
    }

    /// The targets that read Gram `g`, in order.
    pub(crate) fn readers(&self, g: usize) -> Vec<usize> {
        (0..self.of.len()).filter(|&j| self.of[j] == g).collect()
    }

    /// Bring every held block into its matrix.
    pub(crate) fn flush(&mut self) {
        for g in &mut self.grams {
            g.flush();
        }
    }

    /// Every Gram as it stands, decayed by `lam`, for a window's snapshot.
    fn snapshot(&self, lam: f64) -> GramsSnap {
        GramsSnap {
            grams: self.grams.iter().map(|g| Moments::of(g, lam)).collect(),
            of: self.of.clone(),
        }
    }

    /// The Gram that `g`'s targets read in `old`. They have only ever been
    /// split apart, never joined, so all of them read one Gram at any
    /// earlier time, and `g`'s rows up to its split are that Gram's.
    fn ancestor<'a>(&self, g: usize, old_of: &[usize], old: &'a [Moments]) -> Option<&'a Moments> {
        let j = self.of.iter().position(|&o| o == g)?;
        old.get(*old_of.get(j)?)
    }

    /// Each Gram with everything before the row the snapshot `old` precedes
    /// removed
    /// (`crate::truncated`; `f` the decay since it), and an empty one where
    /// nothing of it is left. A Gram made after the snapshot is truncated
    /// against its ancestor's there ([`Grams::ancestor`]).
    fn truncated(&self, old: &GramsSnap, f: f64) -> Vec<EwCov> {
        (0..self.grams.len())
            .map(|g| {
                let live = &self.grams[g];
                self.ancestor(g, &old.of, &old.grams)
                    .and_then(|then| crate::truncated(live, then, f))
                    .unwrap_or_else(|| empty(live))
            })
            .collect()
    }

    /// Mix each Gram toward the twin's that its targets read, `(1 − f)` of
    /// this one's weight to `f` of the twin's: a weight-respecting mixture of
    /// two sets of weighted moments, with the centred co-moments by `C =
    /// a·C_f + b·C_s + a·b·ΔΔᵀ`, `Δ = m_f − m_s`, where nothing is the size
    /// of `m²` (review 2026-09-12, C16). `Q` mixes by the same coefficients
    /// (`TargetMoments::blend` says why a union would be wrong). Both sides'
    /// blocks must be flushed. Returns whether any Gram had weight to mix.
    /// The twin sees the same rows and targets, so its Grams split where
    /// these do.
    fn blend(&mut self, other: &Grams, f: f64) -> bool {
        let mut moved = false;
        for g in 0..self.grams.len() {
            let slow = self
                .of
                .iter()
                .position(|&o| o == g)
                .and_then(|j| other.grams.get(other.of[j]));
            let Some(slow) = slow else { continue };
            let fast = &self.grams[g];
            let k = fast.k();
            let (wf, ws) = (fast.n_eff(), slow.n_eff());
            let w_new = (1.0 - f) * wf + f * ws;
            if w_new <= 0.0 || w_new.is_nan() {
                continue;
            }
            moved = true;
            let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);
            let mean: Vec<f64> = (0..k)
                .map(|i| af * fast.mean(i) + as_ * slow.mean(i))
                .collect();
            let delta: Vec<f64> = (0..k).map(|i| fast.mean(i) - slow.mean(i)).collect();
            let (cf, cs) = (fast.comoments(), slow.comoments());
            let mut c = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    let ij = i * k + j;
                    c[ij] = af * cf[ij] + as_ * cs[ij] + af * as_ * delta[i] * delta[j];
                }
            }
            let q = match (fast.q_sum(), slow.q_sum()) {
                (Some(qf), Some(qs)) => Some(af * qf + as_ * qs),
                _ => None,
            };
            // The decaying prior under `ridge_decay` is a pseudo-observation
            // on the sum scale, `prior_scale · ridge · I`, so it mixes as the
            // sum-scale weights do, by `1 − f` and `f`: at `f = 1` the Gram is
            // the twin's, prior and all. Built from `EwCov::new`, the blend
            // put the prior back at full strength on every session boundary
            // (review 2026-09-12, C6).
            let prior = (1.0 - f) * fast.prior_scale() + f * slow.prior_scale();
            let mut blended = fast.clone();
            blended.set_moments(&mean, &c, w_new, q);
            blended.set_prior_scale(prior);
            self.grams[g] = blended;
        }
        moved
    }
}

/// `cov` with nothing in it: no weight, zero moments. What a Gram with no
/// row inside a window reads as.
fn empty(cov: &EwCov) -> EwCov {
    let k = cov.k();
    let mut out = cov.clone();
    out.set_moments(
        &vec![0.0; k],
        &vec![0.0; k * k],
        0.0,
        cov.q_sum().map(|_| 0.0),
    );
    out
}

/// [`Grams`] as a window's snapshot holds them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct GramsSnap {
    grams: Vec<Moments>,
    of: Vec<usize>,
}

impl crate::Footprint for GramsSnap {
    fn footprint(&self) -> usize {
        self.grams
            .iter()
            .map(crate::Footprint::footprint)
            .sum::<usize>()
            + std::mem::size_of_val(self.of.as_slice())
    }
}

impl crate::Footprint for Cross {
    fn footprint(&self) -> usize {
        std::mem::size_of::<f64>()
            + crate::window::floats(&self.m)
            + crate::window::floats(&self.my)
            + self
                .d
                .iter()
                .chain(&self.c)
                .map(|v| crate::window::floats(v))
                .sum::<usize>()
    }
}

impl crate::Footprint for AccSnap {
    fn footprint(&self) -> usize {
        crate::Footprint::footprint(&self.grams)
            + crate::window::floats(&self.wj)
            + crate::Footprint::footprint(&self.cross)
    }
}

/// A regression's accumulators over its rows: its Grams, and per target its
/// weight, cross-moments and moments. The live ones, and a session twin's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Acc {
    pub(crate) grams: Grams,
    /// Per target, the weight of the rows it was present on.
    pub(crate) wj: Vec<f64>,
    pub(crate) cross: Cross,
    /// Per target: mean, variance and `Sum w^2` of the target itself, the
    /// other half of what the Gram export hands back (docs/ENHANCEMENTS.md
    /// E45).
    pub(crate) tm: TargetMoments,
}

impl Acc {
    pub(crate) fn new(n_targets: usize, k: usize, block_rows: usize) -> Self {
        Self {
            grams: Grams::new(n_targets, k, block_rows),
            wj: vec![0.0; n_targets],
            cross: Cross::new(n_targets, k),
            tm: TargetMoments::new(n_targets),
        }
    }

    /// One row, at decay `lam` and weight `w`: the Grams by `gaps`, each
    /// target's weight `W_j' = lam·W_j + w` and its moments when it is
    /// present, its weight aged when it is not, and the all-row weight and
    /// mean whatever the targets.
    pub(crate) fn learn(
        &mut self,
        z: &[f64],
        y: &[Option<f64>],
        lam: f64,
        w: f64,
        gaps: TargetGaps,
    ) {
        let b_all = self.cross.share(lam, w);
        self.grams.update(z, y, lam, w, gaps);
        for (j, yj) in y.iter().enumerate() {
            let wj = &mut self.wj[j];
            match *yj {
                // `W_j' == 0` means this row carries no weight and none has
                // ever been carried -- a zero-weight row at the head of a
                // stream. `a` and `b` would both be 0/0, and the NaN would
                // never wash out: `W_j` stays NaN, `NaN > 0.0` is false, and
                // the target silently stops predicting forever (hard rule 9).
                Some(yj) if lam * *wj + w > 0.0 => {
                    let wj_new = lam * *wj + w;
                    let (a, b) = (lam * *wj / wj_new, w / wj_new);
                    self.cross.learn(j, z, yj, a, b, b_all);
                    // The other half of the sufficient statistic, on the same
                    // `a`/`b` as the cross-moments (E45).
                    self.tm.learn(j, yj, a, b, lam, w);
                    *wj = wj_new;
                }
                Some(_) => {}
                None => {
                    self.cross.miss(j, z, b_all);
                    self.tm.age(j, lam);
                    *wj *= lam;
                }
            }
        }
        self.cross.advance(z, lam, w, b_all);
    }

    /// The accumulators as they stand, decayed by `lam`: the snapshot before
    /// a row whose clock is `lam` on, so that subtracting it later retains
    /// that row and everything after.
    pub(crate) fn snapshot(&self, lam: f64) -> AccSnap {
        let mut cross = self.cross.clone();
        cross.w *= lam;
        AccSnap {
            grams: self.grams.snapshot(lam),
            wj: self.wj.iter().map(|w| w * lam).collect(),
            cross,
        }
    }

    /// The accumulators with everything before the row the snapshot `old`
    /// precedes removed, `f` the decay since it: `None` when nothing has aged out and
    /// the live accumulators are the answer, and an empty view, all weights
    /// 0, when nothing is left inside the window -- a clock gap longer than
    /// it (review 2026-09-12, C2). A target with no row left keeps weight 0
    /// while the others stay windowed.
    pub(crate) fn window(&self, old: &AccSnap, f: f64) -> Option<AccView> {
        if old.cross.w == 0.0 {
            return None;
        }
        let (m, k) = (self.wj.len(), self.cross.m.len());
        let mut wj = vec![0.0; m];
        let mut per = vec![None; m];
        for j in 0..m {
            // The truncated weight is what makes a target's moments a mean
            // again; the test is `crate::truncated_mean`'s.
            let wj_old = f * old.wj[j];
            let w = self.wj[j] - wj_old;
            if w > EMPTY_FRACTION * self.wj[j] && w.is_finite() {
                wj[j] = w;
                per[j] = Some((wj_old / w, self.wj[j] / w));
            }
        }
        Some(match self.cross.truncated(&old.cross, f, &per) {
            Some(cross) => AccView {
                grams: self.grams.truncated(&old.grams, f),
                wj,
                cross,
            },
            None => AccView {
                grams: self.grams.grams.iter().map(empty).collect(),
                wj: vec![0.0; m],
                cross: Cross::new(m, k),
            },
        })
    }

    /// The window's weights alone, as [`Self::window`] computes them -- over
    /// every row, and per target, with the same rules for an empty one --
    /// without truncating a Gram or a cross-moment: the two numbers a row's
    /// `predict` and `n_eff` read, which built the O(k²) view every row
    /// (review 2026-09-12, P1). `None` where `window` is.
    pub(crate) fn window_weights(&self, old: &AccSnap, f: f64) -> Option<(f64, Vec<f64>)> {
        if old.cross.w == 0.0 {
            return None;
        }
        let m = self.wj.len();
        let w = self.cross.w - f * old.cross.w;
        if w <= EMPTY_FRACTION * self.cross.w || !w.is_finite() {
            return Some((0.0, vec![0.0; m]));
        }
        let wj = (0..m)
            .map(|j| {
                let wj = self.wj[j] - f * old.wj[j];
                if wj > EMPTY_FRACTION * self.wj[j] && wj.is_finite() {
                    wj
                } else {
                    0.0
                }
            })
            .collect();
        Some((w, wj))
    }

    /// Mix toward a twin's accumulators, `(1 − f)` of this side's weight to
    /// `f` of the twin's, each weighted mean by its own weights: the Grams
    /// by theirs ([`Grams::blend`]), each target's moments by its own, and
    /// the cross-moments by both ([`Cross::blend`]). Returns whether anything
    /// had weight to mix.
    pub(crate) fn blend(&mut self, other: &Acc, f: f64) -> bool {
        let mut moved = self.grams.blend(&other.grams, f);
        let mut per = vec![None; self.wj.len()];
        for (j, p) in per.iter_mut().enumerate() {
            let (wf, ws) = (self.wj[j], other.wj[j]);
            let w_new = (1.0 - f) * wf + f * ws;
            if w_new > 0.0 {
                moved = true;
                let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);
                *p = Some((af, as_));
                self.wj[j] = w_new;
                self.tm.blend(&other.tm, j, af, as_);
            }
        }
        let (wf, ws) = (self.cross.w, other.cross.w);
        let w_new = (1.0 - f) * wf + f * ws;
        if w_new > 0.0 {
            moved = true;
            let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);
            self.cross.blend(&other.cross, af, as_, w_new, &per);
        }
        moved
    }

    /// Whether the accumulators are those of `n_targets` targets over `k`
    /// slots, with every target reading a Gram that exists and every Gram
    /// read: what a restored state must hold to be stepped.
    pub(crate) fn has_shape(&self, n_targets: usize, k: usize) -> bool {
        let g = &self.grams;
        !g.grams.is_empty()
            && g.grams.iter().all(|c| c.k() == k)
            && g.of.len() == n_targets
            && g.of.iter().all(|&o| o < g.grams.len())
            && (0..g.grams.len()).all(|i| g.of.contains(&i))
            && self.wj.len() == n_targets
            && self.tm.means().len() == n_targets
            && self.cross.has_shape(n_targets, k)
    }
}

/// [`Acc`] as a window's snapshot holds it: decayed to the row it precedes.
/// The target moments are not in it, so a windowed Gram export reports none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct AccSnap {
    grams: GramsSnap,
    wj: Vec<f64>,
    cross: Cross,
}

/// The accumulators inside a window ([`Acc::window`]): one Gram per live
/// Gram, in the same order, so the live `Grams::of` indexes them.
pub(crate) struct AccView {
    pub(crate) grams: Vec<EwCov>,
    pub(crate) wj: Vec<f64>,
    pub(crate) cross: Cross,
}

/// One Gram of a regression's accumulators, as the Gram export hands it back
/// (docs/PLAN.md task 81): the feature moments, the targets fitted from them
/// (indices into the model's targets), and for each of those its raw
/// cross-moments `E[z·y]`, its weight, and its column means over the rows it
/// was present on. Under `pairwise` there is one, with every target; under
/// `own_rows` one per set of targets present on the same rows.
#[derive(Debug, Clone, PartialEq)]
pub struct GramPart {
    pub cov: EwCov,
    pub targets: Vec<usize>,
    pub cross_moments: Vec<Vec<f64>>,
    pub target_weights: Vec<f64>,
    pub means_by_target: Vec<Vec<f64>>,
}

/// The [`GramPart`]s of Grams read through `of`, from cross-moments `cross`
/// and target weights `wj`: the live accumulators or a window's. A held
/// block is merged into a copy, leaving the model's block where it was. Each
/// target's column means are the ones its solve reads: its Gram's under
/// `own_rows`, and the Gram's plus its offset under `pairwise`.
pub(crate) fn gram_parts(
    grams: &[EwCov],
    of: &[usize],
    cross: &Cross,
    wj: &[f64],
    gaps: TargetGaps,
) -> Vec<GramPart> {
    grams
        .iter()
        .enumerate()
        .map(|(g, cov)| {
            let cov = cov.flushed().into_owned();
            let targets: Vec<usize> = (0..of.len()).filter(|&j| of[j] == g).collect();
            let means_by_target = targets
                .iter()
                .map(|&j| match gaps {
                    TargetGaps::OwnRows => cov.means().to_vec(),
                    TargetGaps::Pairwise => cov
                        .means()
                        .iter()
                        .zip(&cross.d[j])
                        .map(|(m, d)| m + d)
                        .collect(),
                })
                .collect();
            GramPart {
                cross_moments: targets.iter().map(|&j| cross.raw(j)).collect(),
                target_weights: targets.iter().map(|&j| wj[j]).collect(),
                means_by_target,
                targets,
                cov,
            }
        })
        .collect()
}
