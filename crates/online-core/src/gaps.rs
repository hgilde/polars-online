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

/// The decay a history of weight `history` takes into a row of weight `w`:
/// `lam`, or 0 where the history would be in the subnormal range of the row
/// -- its share of the weight after the row, `lam·W / (lam·W + w)`, below
/// the smallest normal double. Forgotten there, as a decay that underflows
/// to 0 forgets it, from 1075 half-lives on (docs/PLAN.md task 215). A
/// history kept at that size gave the row's one-row Gram a spread of its
/// own, which a standardized solve divided by, and the solve on the row read
/// the old fit back. A per-row decay never takes a history there to 0: a
/// subnormal times a decay near 1 rounds back to itself, so a target absent
/// for some 1,060 half-lives sticks a few steps above 0 (task 217). A
/// history of no weight has nothing to forget, and a row of none learns
/// nothing, so both keep `lam`: the decay the prior's scale and the
/// weights then age by.
pub(crate) fn decay_into(lam: f64, history: f64, w: f64) -> f64 {
    let carried = lam * history;
    if w > 0.0 && carried > 0.0 && carried < w * f64::MIN_POSITIVE {
        0.0
    } else {
        lam
    }
}

/// Each target's cross-moments with the feature row `z`, centred (review
/// 2026-09-12, N1), and the weight and mean of `z` over every row. Over the
/// rows target `j` was present on: its EW mean `ȳ_j` (`my`), the centred
/// cross-moment `c_j = E[(z − m_j)(y_j − ȳ_j)]` (`c`), and the mean `m_j` of
/// `z` there (`mj`). `m` and `w` are over every row whatever the Grams
/// learn: `w` is the model's `n_eff` (hard rule 8), and `m` is what a
/// `pairwise` Gram, which is over every row, is centred on.
///
/// They were kept raw, `r_j = E[z·y_j]`, and the solves with an intercept
/// formed `E[z·y] − m·ȳ` from them, or read the raw normal equations: two
/// numbers the size of `level²` subtracted to leave one the size of a
/// covariance, which at `1e8` left nothing of the fit. The raw moment,
/// `r_j = c_j + m_j·ȳ_j`, is one step away for what still reads it.
///
/// `m_j` was kept as an offset `δ_j = m_j − m` until schema 17, a number of
/// its own that was exactly 0 while the target had been present on every
/// row. Reconstructed as `m + δ_j`, it lost `level·ε` of whatever level `m`
/// sat at: a target absent on a row whose features stood at the input
/// bound left `m` and `δ_j` at `1e99` each, their sum resolved nothing
/// below `1e83`, and the next present row's deviation of `0.4` read as
/// `1e83` (review 2026-09-26, G3, from the contract's proptest). Each own
/// mean is now its own pair, updated from the target's own rows by the
/// steps a Gram over those rows takes, so under `own_rows` it is the Gram's
/// mean to the bit, and a row the target misses leaves it exactly where it
/// was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Cross {
    pub(crate) w: f64,
    pub(crate) m: Vec<f64>,
    /// Each target's mean of `z` over its own rows.
    pub(crate) mj: Vec<Vec<f64>>,
    pub(crate) my: Vec<f64>,
    pub(crate) c: Vec<Vec<f64>>,
    /// What `m` and `my` leave out: each mean is a pair no step is rounded
    /// off ([`crate::comp`]; docs/PLAN.md task 101), as a Gram's means are.
    /// Sized with the means in the live accumulators, and empty in a
    /// window's snapshot, whose subtraction reads the doubles alone
    /// ([`Acc::snapshot`]).
    pub(crate) m_lo: Vec<f64>,
    pub(crate) my_lo: Vec<f64>,
    /// What each `mj` leaves out, as `m_lo`.
    pub(crate) mj_lo: Vec<Vec<f64>>,
    /// Per target, the rows of positive weight it was present on; and the
    /// rows of positive weight over every row. A window's snapshot holds
    /// them as they stood before its row, so the rows a window holds of a
    /// target are the live count less the snapshot's, exactly, where the
    /// weights' difference is a remainder of rounding: a target absent for
    /// some 1,060 half-lives keeps a weight stuck a few subnormal steps above
    /// 0, and its window weight read `5e-324` and predicted (docs/PLAN.md
    /// task 217). A row of weight 0 advances the clock only and is not
    /// counted (hard rule 9). Schema 52.
    #[serde(default)]
    pub(crate) nj: Vec<u64>,
    #[serde(default)]
    pub(crate) n: u64,
}

impl Cross {
    pub(crate) fn new(n_targets: usize, k: usize) -> Self {
        Self {
            w: 0.0,
            m: vec![0.0; k],
            mj: vec![vec![0.0; k]; n_targets],
            my: vec![0.0; n_targets],
            c: vec![vec![0.0; k]; n_targets],
            m_lo: vec![0.0; k],
            my_lo: vec![0.0; n_targets],
            mj_lo: vec![vec![0.0; k]; n_targets],
            nj: vec![0; n_targets],
            n: 0,
        }
    }

    /// A row's share of the weight over every row, at decay `lam` and
    /// weight `w`: the `b` of [`Cross::learn`], [`Cross::miss`] and
    /// [`Cross::advance`].
    pub(crate) fn share(&self, lam: f64, w: f64) -> f64 {
        row_share(lam * self.w, w)
    }

    /// Target `j` present on a row, with the `a_j`/`b_j` of its own weight's
    /// update: the cross-moment from the deviations `z − m_j` and `y − ȳ_j`
    /// against the means before the row, then each mean takes `b_j` of its
    /// deviation -- the steps `EwCov::update` takes for a Gram over the same
    /// rows. A row the target is absent on touches none of this; the
    /// all-row mean takes every row in [`Cross::advance`].
    pub(crate) fn learn(&mut self, j: usize, z: &[f64], y: f64, aj: f64, bj: f64) {
        use crate::comp::{add, dev};
        let k = self.m.len();
        let dy = dev(y, self.my[j], self.my_lo[j]);
        let ab_dy = aj * bj * dy;
        let (mj, c, mj_lo) = (&mut self.mj[j], &mut self.c[j], &mut self.mj_lo[j]);
        debug_assert_eq!(z.len(), k);
        // The low parts are sized with the means, so the loop is straight
        // arithmetic (task 132's own means cost a 10-target `ewridge` a
        // quarter of its rows a second with checks inside; PERFORMANCE §26).
        // A row of weight 0 takes no step (`crate::comp::add` says why).
        if bj > 0.0 {
            for (((m, lo), ci), &zi) in mj.iter_mut().zip(mj_lo.iter_mut()).zip(c.iter_mut()).zip(z)
            {
                let u = dev(zi, *m, *lo);
                *ci = aj * *ci + ab_dy * u;
                add(m, lo, bj * u);
            }
        } else {
            for (((m, lo), ci), &zi) in mj.iter().zip(mj_lo.iter()).zip(c.iter_mut()).zip(z) {
                let u = dev(zi, *m, *lo);
                *ci = aj * *ci + ab_dy * u; // `ab_dy` is a zero here: `bj` is 0
            }
        }
        if bj > 0.0 {
            add(&mut self.my[j], &mut self.my_lo[j], bj * dy);
        }
    }

    /// The all-row mean and weight take the row, after every target has read
    /// it. The weight by `EwCov::update`'s recursion.
    pub(crate) fn advance(&mut self, z: &[f64], lam: f64, w: f64, b: f64) {
        if b > 0.0 {
            use crate::comp::{add, dev};
            for ((mi, lo), &zi) in self.m.iter_mut().zip(self.m_lo.iter_mut()).zip(z) {
                let u = dev(zi, *mi, *lo);
                add(mi, lo, b * u);
            }
        }
        // `EwCov::update`'s recursion, whose one refusal -- no weight at all,
        // carried or added -- is a no-op here: that `W'` is 0 either way, at
        // the head of a stream and after a decay that took everything (task
        // 115 (c)).
        self.w = lam * self.w + w;
        if w > 0.0 {
            self.n += 1;
        }
    }

    /// Target `j`'s uncentred `E[z·y_j] = c_j + m_j·ȳ_j`.
    pub(crate) fn raw(&self, j: usize) -> Vec<f64> {
        let my = self.my[j];
        self.c[j]
            .iter()
            .zip(&self.mj[j])
            .map(|(c, m)| c + m * my)
            .collect()
    }

    /// Mix toward the twin's as [`Grams::blend`] mixes the Grams: `(a, b)`
    /// over every row, with `w_new` the mixed all-row weight, and `per[j] =
    /// (a_j, b_j)` over target `j`'s own rows (`None` where neither side has
    /// any). The centred mixture, `c = a_j·c + b_j·c' + a_j·b_j·(m_j −
    /// m_j')(ȳ − ȳ')`, as the co-moments' (C16), and each mean mixes as a
    /// mean does.
    pub(crate) fn blend(
        &mut self,
        other: &Self,
        a: f64,
        b: f64,
        w_new: f64,
        per: &[Option<(f64, f64)>],
    ) {
        let (n, k) = (self.my.len(), self.m.len());
        for (j, p) in per.iter().enumerate() {
            let Some((aj, bj)) = *p else { continue };
            let dy = self.my[j] - other.my[j];
            let mine = self.c[j].iter_mut().zip(self.mj[j].iter_mut());
            let theirs = other.c[j].iter().zip(&other.mj[j]);
            for ((c, m), (oc, om)) in mine.zip(theirs) {
                let dz = *m - om;
                *c = aj * *c + bj * oc + aj * bj * dz * dy;
                *m = aj * *m + bj * om;
            }
            self.my[j] = aj * self.my[j] + bj * other.my[j];
            self.my_lo[j] = 0.0;
        }
        for (m, om) in self.m.iter_mut().zip(&other.m) {
            *m = a * *m + b * om;
        }
        // The mixed means are doubles, with nothing left out of them.
        self.m_lo = vec![0.0; k];
        self.mj_lo = vec![vec![0.0; k]; n];
        self.w = w_new;
    }

    /// The rows after a snapshot `old`, `f` the decay since it, as
    /// `crate::truncated` takes them from a Gram: `None` when no weight is
    /// left inside the window. With `ratio = W_u/W_R` over every row and
    /// `per[j] = (ratio_j, g_j)` -- `W_u,j/W_R,j` and `W_j/W_R,j` -- over
    /// target `j`'s own (`None` where nothing of it is left), the pooling
    /// identity again, `c_R = g_j·c − ratio_j·c_u − ratio_j·g_j·(m_j,u −
    /// m_j)(ȳ_u − ȳ)`, and every mean by `x_R = x − ratio·(x_u − x)` at its
    /// own ratio. The snapshot's means are read as the doubles it holds.
    pub(crate) fn truncated(&self, old: &Self, f: f64, per: &[Option<(f64, f64)>]) -> Option<Self> {
        let w_old = f * old.w;
        let w = self.w - w_old;
        if w <= EMPTY_FRACTION * self.w || !w.is_finite() {
            return None;
        }
        let ratio = w_old / w;
        let mut out = Self::new(self.my.len(), self.m.len());
        out.w = w;
        for ((o, m), mu) in out.m.iter_mut().zip(&self.m).zip(&old.m) {
            *o = m - ratio * (mu - m);
        }
        for (j, p) in per.iter().enumerate() {
            let Some((rj, gj)) = *p else { continue };
            let dy = old.my[j] - self.my[j];
            let into = out.c[j].iter_mut().zip(out.mj[j].iter_mut());
            let now = self.c[j].iter().zip(&self.mj[j]);
            let then = old.c[j].iter().zip(&old.mj[j]);
            for (((oc, om), (c, m)), (cu, mu)) in into.zip(now).zip(then) {
                let dz = mu - m;
                *oc = gj * c - rj * cu - rj * gj * dz * dy;
                *om = m - rj * dz;
            }
            out.my[j] = self.my[j] - rj * dy;
        }
        Some(out)
    }

    /// Whether the moments are those of `n_targets` targets over `k` slots,
    /// with the low parts shaped as the means where `low_parts` -- the live
    /// accumulators -- and absent where not, as a window's snapshot holds
    /// them ([`Acc::snapshot`]).
    fn has_shape(&self, n_targets: usize, k: usize, low_parts: bool) -> bool {
        let lows = if low_parts {
            self.m_lo.len() == k
                && self.my_lo.len() == n_targets
                && self.mj_lo.len() == n_targets
                && self.mj_lo.iter().all(|v| v.len() == k)
        } else {
            self.m_lo.is_empty() && self.my_lo.is_empty() && self.mj_lo.is_empty()
        };
        self.m.len() == k
            && self.my.len() == n_targets
            && self.mj.len() == n_targets
            && self.c.len() == n_targets
            && self.mj.iter().chain(&self.c).all(|v| v.len() == k)
            && self.nj.len() == n_targets
            && lows
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
    /// `windowed` says whether the owner has a window, the one reader of
    /// the Grams' runs (docs/PLAN.md task 128). A Gram split off under
    /// `own_rows` is a clone, so it keeps the choice.
    pub(crate) fn new(n_targets: usize, k: usize, block_rows: usize, windowed: bool) -> Self {
        let mut g = if windowed {
            EwCov::new(k)
        } else {
            EwCov::new(k).without_runs()
        };
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
        // A Gram that learns the row forgets a history the row leaves in
        // the subnormal range ([`decay_into`]), by the rule each of its
        // targets' weights does in `Acc::learn`: the two are equal to the
        // bit, so they decide alike and stay equal (task 217).
        fn learn(g: &mut EwCov, z: &[f64], lam: f64, w: f64) {
            let lam = decay_into(lam, g.n_eff(), w);
            g.update(z, lam, w);
        }
        let Self { grams, of, flags } = self;
        if gaps == TargetGaps::Pairwise {
            for g in grams.iter_mut() {
                learn(g, z, lam, w);
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
                PRESENT => learn(&mut grams[g], z, lam, w),
                ABSENT => grams[g].skip(z, lam),
                // A Gram every target has left; there is none.
                0 => {}
                // A row with no weight: no split. Its present and its absent
                // targets take the same step, the decay alone, with weight
                // to age or without (task 159, R1: with none, the present
                // ones' Gram refused the row and kept the prior's scale at 1
                // where the absent ones' aged it, so the two parted).
                _ if w == 0.0 => grams[g].update(z, lam, w),
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
                    learn(&mut grams[g], z, lam, w);
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

    /// Mix each Gram toward the twin's that its targets read, as a mixture
    /// of the two data sets: `a = 1 − f` of this one's and `b = f` of the
    /// twin's, the means by those shares and the centred co-moments by `C =
    /// a·C_f + b·C_s + a·b·ΔΔᵀ`, `Δ = m_f − m_s`, where nothing is the size of
    /// `m²` (review 2026-09-12, C16). The weight, `Q` and the prior scale stay
    /// this Gram's: the blend moves what the fit is read from, not how much
    /// stands behind it, so `f` is the long run's share whatever the twin's
    /// weight (docs/PLAN.md task 145; mixed by weight, the twin's far larger
    /// weight took nearly all of any `f` above 0). A side with no weight has
    /// nothing to mix. Both sides' blocks must be flushed. Returns whether
    /// any Gram was mixed. The twin sees the same rows and targets, so its
    /// Grams split where these do.
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
            if !(wf > 0.0 && ws > 0.0) {
                continue;
            }
            moved = true;
            let (af, as_) = (1.0 - f, f);
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
            // The clone keeps this Gram's prior scale under `ridge_scale`, a
            // pseudo-observation on its own sum scale: the blend no more
            // strengthens it than it strengthens the data. (Built from
            // `EwCov::new`, the blend once put the prior back at full strength
            // on every session boundary; review 2026-09-12, C6.)
            let mut blended = fast.clone();
            blended.set_moments(&mean, &c, wf, fast.q_sum());
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
    /// Every vector a snapshot's copy holds, the means' low parts included:
    /// `Acc::snapshot` clones the whole of it (docs/PLAN.md task 130).
    fn footprint(&self) -> usize {
        // The weight and the row count over every row, and the counts per
        // target (task 217).
        std::mem::size_of::<f64>()
            + std::mem::size_of::<u64>()
            + std::mem::size_of_val(self.nj.as_slice())
            + crate::window::floats(&self.m)
            + crate::window::floats(&self.my)
            + crate::window::floats(&self.m_lo)
            + crate::window::floats(&self.my_lo)
            + self
                .mj
                .iter()
                .chain(&self.c)
                .chain(&self.mj_lo)
                .map(|v| crate::window::floats(v))
                .sum::<usize>()
    }
}

impl crate::Footprint for AccSnap {
    fn footprint(&self) -> usize {
        crate::Footprint::footprint(&self.grams)
            + crate::window::floats(&self.wj)
            + crate::Footprint::footprint(&self.cross)
            + self.tm.as_ref().map_or(0, |t| {
                crate::window::floats(t.means())
                    + crate::window::floats(t.vars())
                    + crate::window::floats(t.q())
                    + crate::window::floats(t.means_lo())
            })
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
    pub(crate) fn new(n_targets: usize, k: usize, block_rows: usize, windowed: bool) -> Self {
        Self {
            grams: Grams::new(n_targets, k, block_rows, windowed),
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
                    // A history of the target's own that the row leaves in
                    // the subnormal range is forgotten, as its Gram forgets
                    // it ([`decay_into`]): a target absent for some 1,060
                    // half-lives while the others go on kept one, its
                    // per-row decay stuck a few steps above 0, and the
                    // solve on its return read the old fit back from it
                    // (task 217). The all-row weight and mean need no rule:
                    // what the decay leaves of a history that small is
                    // under a rounding step of the row's weight.
                    let lam = decay_into(lam, *wj, w);
                    if w > 0.0 {
                        self.cross.nj[j] += 1;
                    }
                    let wj_new = lam * *wj + w;
                    let (a, b) = (lam * *wj / wj_new, w / wj_new);
                    self.cross.learn(j, z, yj, a, b);
                    // The other half of the sufficient statistic, on the same
                    // `a`/`b` as the cross-moments (E45).
                    self.tm.learn(j, yj, a, b, lam, w);
                    *wj = wj_new;
                }
                // Nothing carried and nothing added. At the head of a stream
                // that leaves everything at 0; after a decay that took the
                // whole history -- `lam·W_j` is 0 from 1075 half-lives on --
                // it is the decay alone, as for an absent target, where the
                // history used to be kept whole (task 115 (c), PLAN §12).
                Some(_) | None => {
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
        // The window's subtraction reads the cross accumulator's means as
        // doubles alone (`Cross::truncated`), so its snapshot carries no low
        // parts: `k + T` doubles a snapshot the ring would otherwise hold for
        // nothing (review 2026-09-26, C9). The target moments' do come, `T`
        // doubles, since their truncation reads the pairs (`TargetMoments::
        // decayed`; review 2026-10-05, CB2).
        cross.m_lo = Vec::new();
        cross.my_lo = Vec::new();
        cross.mj_lo = Vec::new();
        AccSnap {
            grams: self.grams.snapshot(lam),
            wj: self.wj.iter().map(|w| w * lam).collect(),
            cross,
            tm: Some(self.tm.decayed(lam)),
        }
    }

    /// Whether every Gram keeps runs for a window to read.
    pub(crate) fn keeps_runs(&self) -> bool {
        self.grams.grams.iter().all(EwCov::keeps_runs)
    }

    /// Keep no runs in any Gram from here (review 2026-09-26, C4).
    pub(crate) fn set_runs_off(&mut self) {
        self.grams.grams.iter_mut().for_each(EwCov::set_runs_off);
    }

    /// Whether the window since the snapshot `old` holds a row of positive
    /// weight of target `j`: the live count less the snapshot's, exact,
    /// where the weights' difference is a remainder that rounding leaves
    /// above 0 (docs/PLAN.md task 217). A window that holds none is empty
    /// for the target: it predicts and gates as a target never seen does.
    pub(crate) fn holds_rows_of(&self, old: &AccSnap, j: usize) -> bool {
        self.cross.nj[j] > old.cross.nj[j]
    }

    /// Whether the window since the snapshot `old` holds a row of positive
    /// weight at all ([`Self::holds_rows_of`] over every row).
    pub(crate) fn holds_rows(&self, old: &AccSnap) -> bool {
        self.cross.n > old.cross.n
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
            // again; the test is `crate::truncated_mean`'s, and a window that
            // holds none of the target's rows is empty for it, whatever the
            // weights' difference leaves ([`Self::holds_rows_of`]).
            let wj_old = f * old.wj[j];
            let w = self.wj[j] - wj_old;
            if w > EMPTY_FRACTION * self.wj[j] && w.is_finite() && self.holds_rows_of(old, j) {
                wj[j] = w;
                per[j] = Some((wj_old / w, self.wj[j] / w));
            }
        }
        // The target moments where the snapshot carries them (docs/PLAN.md
        // task 136); a snapshot from before has none, and the Gram export
        // then says `None` until the ring has rolled over.
        let tm = old.tm.as_ref().map(|t| self.tm.truncated(t, f, &per));
        let cross = if self.holds_rows(old) {
            self.cross.truncated(&old.cross, f, &per)
        } else {
            None
        };
        Some(match cross {
            Some(mut cross) => {
                let grams = self.grams.truncated(&old.grams, f);
                // A slot with no spread in the window of the Gram a target
                // reads has none over the target's rows inside it either, so
                // no covariance with the target there (Cauchy-Schwarz): zero
                // exactly, where the subtraction left a remainder that a
                // ridge divides by its penalty. The Gram's zero is exact for
                // a feature that held one value over the window (PLAN task
                // 94, `crate::truncated`).
                for (j, p) in per.iter().enumerate() {
                    let (Some(_), Some(gram)) = (p, grams.get(self.grams.of[j])) else {
                        continue;
                    };
                    for (i, c) in cross.c[j].iter_mut().enumerate() {
                        if gram.var(i) == 0.0 {
                            *c = 0.0;
                        }
                    }
                }
                // And a target with no spread in the window has no
                // covariance with any slot there, by the same inequality.
                // Its variance is 0 where the subtraction cannot resolve it
                // (`TargetMoments::truncated`, `crate::truncated`'s bound),
                // and the cross-moments' remainder then is the target's mean
                // rounded at its level: with two rows of weight 1e100 at one
                // target value inside the window, slopes of 6.6e-4 at a
                // level of 1e8 where the rows give 1e-95 (docs/PLAN.md task
                // 217). The rows' slopes are at most the target's spread
                // over the feature's, so 0 moves a prediction by less than
                // the spread the window cannot resolve.
                if let Some(tm) = tm.as_ref() {
                    for (j, p) in per.iter().enumerate() {
                        if p.is_some() && tm.vars()[j] == 0.0 {
                            cross.c[j].iter_mut().for_each(|c| *c = 0.0);
                        }
                    }
                }
                AccView {
                    grams,
                    wj,
                    cross,
                    tm,
                }
            }
            None => AccView {
                grams: self.grams.grams.iter().map(empty).collect(),
                wj: vec![0.0; m],
                cross: Cross::new(m, k),
                tm: tm.map(|t| t.truncated(&t, 1.0, &vec![None; m])),
            },
        })
    }

    /// The window's weights alone, as [`Self::window`] computes them -- over
    /// every row, and per target, with the same rules for an empty one --
    /// without truncating a Gram or a cross-moment: the two numbers a row's
    /// `predict` and `n_eff` read, which built the O(k²) view every row
    /// (review 2026-09-12, P1). `None` where `window` is.
    /// Kish's effective sample size inside the window, per Gram, as
    /// [`crate::truncated`] would leave it -- `W_R = W − f·W_u` and
    /// `Q_R = Q − f²·Q_u` against the ancestor snapshot -- without
    /// truncating the moments themselves, for the readiness statistics a
    /// row reads (docs/WARMUP-AND-CONVERGENCE.md §2.1). `None` where
    /// [`Self::window`] is; an entry is `None` for a Gram with nothing left
    /// in the window, no Kish sum, or a Kish sum the subtraction leaves no
    /// digit of (`Q_R` within `64 ε` of `Q`; review 2026-10-05, CE6b).
    pub(crate) fn window_kish(&self, old: &AccSnap, f: f64) -> Option<Vec<Option<f64>>> {
        if old.cross.w == 0.0 {
            return None;
        }
        Some(
            (0..self.grams.grams.len())
                .map(|g| {
                    let live = &self.grams.grams[g];
                    let then = self.grams.ancestor(g, &old.grams.of, &old.grams.grams)?;
                    let w = live.n_eff() - f * then.w;
                    let q_now = live.q_sum()?;
                    let q = q_now - f * f * then.q?;
                    // A Kish sum the subtraction leaves no digit of is no
                    // size (`crate::truncated`; review 2026-10-05, CE6b).
                    let digits = q > 64.0 * f64::EPSILON * q_now;
                    // And a Gram that has learned no row since holds none
                    // (`Moments::holds_rows`; task 217).
                    (w > EMPTY_FRACTION * live.n_eff()
                        && w.is_finite()
                        && digits
                        && then.holds_rows(live))
                    .then(|| w * w / q)
                })
                .collect(),
        )
    }

    pub(crate) fn window_weights(&self, old: &AccSnap, f: f64) -> Option<(f64, Vec<f64>)> {
        if old.cross.w == 0.0 {
            return None;
        }
        let m = self.wj.len();
        let w = self.cross.w - f * old.cross.w;
        if w <= EMPTY_FRACTION * self.cross.w || !w.is_finite() || !self.holds_rows(old) {
            return Some((0.0, vec![0.0; m]));
        }
        let wj = (0..m)
            .map(|j| {
                let wj = self.wj[j] - f * old.wj[j];
                if wj > EMPTY_FRACTION * self.wj[j] && wj.is_finite() && self.holds_rows_of(old, j)
                {
                    wj
                } else {
                    0.0
                }
            })
            .collect();
        Some((w, wj))
    }

    /// Mix toward a twin's accumulators as a mixture of the two data sets,
    /// `1 − f` of this side's and `f` of the twin's: the Grams
    /// ([`Grams::blend`]), each target's moments and the cross-moments
    /// ([`Cross::blend`]), every weight kept, so `n_eff`, each target's
    /// weight and the Kish sums are this side's across the blend
    /// (docs/PLAN.md task 145). Returns whether anything was mixed.
    pub(crate) fn blend(&mut self, other: &Acc, f: f64) -> bool {
        let mut moved = self.grams.blend(&other.grams, f);
        let mut per = vec![None; self.wj.len()];
        for (j, p) in per.iter_mut().enumerate() {
            if self.wj[j] > 0.0 && other.wj[j] > 0.0 {
                moved = true;
                *p = Some((1.0 - f, f));
                self.tm.blend(&other.tm, j, 1.0 - f, f);
            }
        }
        let w = self.cross.w;
        if w > 0.0 && other.cross.w > 0.0 {
            moved = true;
            self.cross.blend(&other.cross, 1.0 - f, f, w, &per);
        }
        moved
    }

    /// Whether the accumulators are those of `n_targets` targets over `k`
    /// slots, with every target reading a Gram that exists and every Gram
    /// read: what a restored state must hold to be stepped. Each Gram's
    /// means and co-moments at its width, and the target moments at theirs
    /// in every part a row updates: a Gram was held to its `k` and the
    /// target moments to their means alone, so a short vector loaded and
    /// the next row indexed past it (review 2026-10-06, beside CA2).
    pub(crate) fn has_shape(&self, n_targets: usize, k: usize) -> bool {
        let g = &self.grams;
        !g.grams.is_empty()
            && g.grams.iter().all(|c| c.has_shape(k))
            && g.of.len() == n_targets
            && g.of.iter().all(|&o| o < g.grams.len())
            && (0..g.grams.len()).all(|i| g.of.contains(&i))
            && self.wj.len() == n_targets
            && tm_has_shape(&self.tm, n_targets)
            && self.cross.has_shape(n_targets, k, true)
    }
}

/// Whether target moments are `n_targets` wide in each part a row or a
/// window reads: the means, their low parts, the variances and the Kish
/// sums.
fn tm_has_shape(t: &TargetMoments, n_targets: usize) -> bool {
    t.means().len() == n_targets
        && t.vars().len() == n_targets
        && t.q().len() == n_targets
        && t.means_lo().len() == n_targets
}

/// [`Acc`] as a window's snapshot holds it: decayed to the row it precedes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct AccSnap {
    grams: GramsSnap,
    wj: Vec<f64>,
    cross: Cross,
    /// The target moments, `3·T` doubles beside a Gram's `k²` (docs/PLAN.md
    /// task 136), so the Gram export under a window reports the window's
    /// target means, variances and Kish counts. `None` in a snapshot written
    /// before them, and last, so the positional encoding reads one.
    #[serde(default)]
    pub(crate) tm: Option<TargetMoments>,
}

impl AccSnap {
    /// Whether a snapshot is of `n_targets` targets over `k` slots, as the
    /// live accumulators must be ([`Acc::has_shape`]): every Gram's moments
    /// at the width with a weight a subtraction can read, each target
    /// reading a Gram the snapshot holds, and every per-target vector
    /// `n_targets` long -- what [`Acc::window`] and [`Acc::window_weights`]
    /// index. A restored window held none of it, so a snapshot a target
    /// short loaded and the next row's window read past it (review
    /// 2026-10-06, CA2).
    pub(crate) fn has_shape(&self, n_targets: usize, k: usize) -> bool {
        let g = &self.grams;
        g.grams.iter().all(|m| m.has_shape(k))
            && g.of.len() == n_targets
            && g.of.iter().all(|&o| o < g.grams.len())
            && self.wj.len() == n_targets
            && self.cross.has_shape(n_targets, k, false)
            && self.tm.as_ref().is_none_or(|t| tm_has_shape(t, n_targets))
    }
}

/// The accumulators inside a window ([`Acc::window`]): one Gram per live
/// Gram, in the same order, so the live `Grams::of` indexes them.
pub(crate) struct AccView {
    pub(crate) grams: Vec<EwCov>,
    pub(crate) wj: Vec<f64>,
    pub(crate) cross: Cross,
    /// The target moments inside the window, where the snapshot carries
    /// them.
    pub(crate) tm: Option<TargetMoments>,
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
    /// Per target, `E[(z − m_j)(y − ȳ_j)]` over its rows: what the fit is
    /// solved from, where `cross_moments` is it plus `m_j·ȳ_j` (review
    /// 2026-09-12, N4).
    pub cross_centred: Vec<Vec<f64>>,
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
                    TargetGaps::Pairwise => cross.mj[j].clone(),
                })
                .collect();
            GramPart {
                cross_moments: targets.iter().map(|&j| cross.raw(j)).collect(),
                cross_centred: targets.iter().map(|&j| cross.c[j].clone()).collect(),
                target_weights: targets.iter().map(|&j| wj[j]).collect(),
                means_by_target,
                targets,
                cov,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Footprint;

    /// One row of a test stream: the feature row, each target, the row's
    /// weight.
    type Row = (Vec<f64>, Vec<Option<f64>>, f64);

    /// What a shape test breaks, and how.
    type Corruption<'a, T> = (&'a str, &'a dyn Fn(&mut T));

    /// The definition of what the accumulators hold, straight from the rows:
    /// with each row's weight as given, over every row the weight `W` and
    /// the mean `m` of `z`; over the rows target `j` is present on, its
    /// weight `W_j`, the means `m_j` of `z` and `ȳ_j` of `y_j`, and `c_j =
    /// Σ w·(z − m_j)(y_j − ȳ_j) / W_j`.
    struct Pooled {
        w: f64,
        m: Vec<f64>,
        wj: Vec<f64>,
        mj: Vec<Vec<f64>>,
        my: Vec<f64>,
        c: Vec<Vec<f64>>,
    }

    fn pooled(rows: &[Row]) -> Pooled {
        let (k, n) = (rows[0].0.len(), rows[0].1.len());
        let mean = |on: &dyn Fn(&Row) -> bool, of: &dyn Fn(&Row) -> f64| {
            let (mut s, mut w) = (0.0, 0.0);
            for r in rows.iter().filter(|r| on(r)) {
                s += r.2 * of(r);
                w += r.2;
            }
            s / w
        };
        let all = |_: &Row| true;
        let w: f64 = rows.iter().map(|r| r.2).sum();
        let m = (0..k).map(|i| mean(&all, &|r| r.0[i])).collect();
        let (mut wj, mut mj, mut my, mut c) = (vec![], vec![], vec![], vec![]);
        for j in 0..n {
            let on = |r: &Row| r.1[j].is_some();
            let yj = |r: &Row| r.1[j].unwrap_or(f64::NAN);
            wj.push(rows.iter().filter(|r| on(r)).map(|r| r.2).sum());
            let y = mean(&on, &yj);
            let z: Vec<f64> = (0..k).map(|i| mean(&on, &|r| r.0[i])).collect();
            c.push(
                (0..k)
                    .map(|i| mean(&on, &|r| (r.0[i] - z[i]) * (yj(r) - y)))
                    .collect(),
            );
            my.push(y);
            mj.push(z);
        }
        Pooled {
            w,
            m,
            wj,
            mj,
            my,
            c,
        }
    }

    fn close(got: f64, want: f64, tol: f64) -> bool {
        (got - want).abs() <= tol * want.abs().max(1.0)
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Two features and two targets, the second absent on every third row,
    /// at a level so a slip in a mean shows.
    fn stream(n: usize, seed: u64) -> Vec<Row> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let z = vec![5.0 + lcg(&mut s), -3.0 + 2.0 * lcg(&mut s)];
                let y0 = 1.0 + 2.0 * z[0] - z[1] + 0.3 * lcg(&mut s);
                let y1 = (i % 3 != 0).then(|| 4.0 - z[0] + 0.5 * lcg(&mut s));
                (z, vec![Some(y0), y1], 0.5 + 0.25 * (i % 4) as f64)
            })
            .collect()
    }

    /// The rows into fresh accumulators, `lam` a row after the first.
    fn learned(rows: &[Row], lam: f64, gaps: TargetGaps) -> Acc {
        let mut acc = Acc::new(rows[0].1.len(), rows[0].0.len(), 0, true);
        for (i, (z, y, w)) in rows.iter().enumerate() {
            acc.learn(z, y, if i == 0 { 1.0 } else { lam }, *w, gaps);
        }
        acc
    }

    /// Each row's weight decayed to the last row, `lam` a row.
    fn decayed(rows: &[Row], lam: f64) -> Vec<Row> {
        let n = rows.len();
        rows.iter()
            .enumerate()
            .map(|(i, (z, y, w))| (z.clone(), y.clone(), w * lam.powi((n - 1 - i) as i32)))
            .collect()
    }

    #[test]
    fn a_rows_share_of_nothing_is_nothing() {
        assert_eq!(row_share(3.0, 1.0), 0.25);
        assert_eq!(row_share(0.0, 2.0), 1.0);
        // No weight carried and none added: 0, not 0/0 (hard rule 9).
        assert_eq!(row_share(0.0, 0.0), 0.0);
    }

    /// PLAN §13 for the cross-moments: the accumulators with a snapshot
    /// subtracted are those of the rows after it, every one of them -- the
    /// weight and mean over every row, and each target's weight, means and
    /// centred cross-moment over its own rows -- as the definition computes
    /// them from those rows alone.
    #[test]
    fn a_window_is_the_moments_of_the_rows_inside_it() {
        let (lam, n, s) = (0.9_f64, 40, 25);
        let rows = stream(n, 3);
        for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
            let mut acc = learned(&rows[..s], lam, gaps);
            let snap = acc.snapshot(lam);
            for (z, y, w) in &rows[s..] {
                acc.learn(z, y, lam, *w, gaps);
            }
            let f = lam.powi((n - 1 - s) as i32);
            let view = acc.window(&snap, f).expect("rows have aged out");
            let want = pooled(&decayed(&rows[s..], lam));
            let tol = 1e-9;
            assert!(close(view.cross.w, want.w, tol), "{gaps:?} w");
            for i in 0..2 {
                let (got, w) = (view.cross.m[i], want.m[i]);
                assert!(close(got, w, tol), "{gaps:?} m[{i}]: {got} against {w}");
            }
            for j in 0..2 {
                assert!(close(view.wj[j], want.wj[j], tol), "{gaps:?} wj[{j}]");
                assert!(close(view.cross.my[j], want.my[j], tol), "{gaps:?} my[{j}]");
                for i in 0..2 {
                    let (got, w) = (view.cross.mj[j][i], want.mj[j][i]);
                    assert!(
                        close(got, w, tol),
                        "{gaps:?} mj[{j}][{i}]: {got} against {w}"
                    );
                    let (got, w) = (view.cross.c[j][i], want.c[j][i]);
                    assert!(
                        close(got, w, tol),
                        "{gaps:?} c[{j}][{i}]: {got} against {w}"
                    );
                }
            }
            // The second target's own rows are not every row: the test reads
            // two different means of each feature.
            assert!((want.mj[1][0] - want.m[0]).abs() > 1e-3);
        }
    }

    /// A window is empty below `EMPTY_FRACTION` of the live weight, at any
    /// scale of the weight: half the fraction left is nothing, twice it is
    /// a window. A window with nothing at all left is empty too.
    #[test]
    fn an_empty_window_is_a_fraction_of_the_weight_at_any_scale() {
        for scale in [1e-3, 1.0, 1e3] {
            let mut live = Cross::new(1, 1);
            live.w = scale;
            assert!(live.truncated(&live, 1.0, &[None]).is_none(), "{scale}");
            for (left, empty) in [(0.5, true), (2.0, false)] {
                let mut old = live.clone();
                old.w = scale * (1.0 - left * EMPTY_FRACTION);
                let got = live.truncated(&old, 1.0, &[None]);
                assert_eq!(got.is_none(), empty, "{left} of the fraction at {scale}");
            }
        }
    }

    /// Mixing toward a twin is the mixture of the two data sets, `1 − f` of
    /// this side's and `f` of the twin's, each normalised to its own weight
    /// (docs/PLAN.md task 145): computed from the pooled rows, the Gram's
    /// mean and co-moments over every row, the all-row mean, and each
    /// target's means and centred cross-moment over its own rows. The
    /// weights stay this side's.
    #[test]
    fn a_blend_is_the_mixture_of_the_two_data_sets() {
        let (lam, f) = (0.95_f64, 0.3_f64);
        let (mine, theirs) = (stream(30, 5), stream(30, 6));
        let mut fast = learned(&mine, lam, TargetGaps::Pairwise);
        let slow = learned(&theirs, lam, TargetGaps::Pairwise);
        let (w_before, wj_before) = (fast.cross.w, fast.wj.clone());
        assert!(fast.blend(&slow, f));
        // The pool: each side's rows at their decayed weights, scaled so the
        // side sums to its share -- per target, of that target's weight.
        let (a, b) = (decayed(&mine, lam), decayed(&theirs, lam));
        let (pa, pb) = (pooled(&a), pooled(&b));
        let scaled = |rows: &[Row], share: f64, total: f64, j: Option<usize>| -> Vec<Row> {
            rows.iter()
                .filter(|r| j.is_none_or(|j| r.1[j].is_some()))
                .map(|(z, y, w)| (z.clone(), y.clone(), share * w / total))
                .collect()
        };
        let all: Vec<Row> = [scaled(&a, 1.0 - f, pa.w, None), scaled(&b, f, pb.w, None)].concat();
        let want = pooled(&all);
        let tol = 1e-10;
        let gram = &fast.grams.grams[0];
        for i in 0..2 {
            assert!(close(fast.cross.m[i], want.m[i], tol), "m[{i}]");
            assert!(close(gram.mean(i), want.m[i], tol), "the Gram's mean {i}");
            for k in 0..2 {
                let pool: f64 = all
                    .iter()
                    .map(|r| r.2 * (r.0[i] - want.m[i]) * (r.0[k] - want.m[k]))
                    .sum();
                let got = gram.comoments()[i * 2 + k];
                assert!(
                    close(got, pool, tol),
                    "the Gram's C[{i}][{k}]: {got} against {pool}"
                );
            }
        }
        for j in 0..2 {
            let own = pooled(
                &[
                    scaled(&a, 1.0 - f, pa.wj[j], Some(j)),
                    scaled(&b, f, pb.wj[j], Some(j)),
                ]
                .concat(),
            );
            assert!(close(fast.cross.my[j], own.my[j], tol), "my[{j}]");
            for i in 0..2 {
                assert!(
                    close(fast.cross.mj[j][i], own.mj[j][i], tol),
                    "mj[{j}][{i}]"
                );
                let (got, w) = (fast.cross.c[j][i], own.c[j][i]);
                assert!(close(got, w, tol), "c[{j}][{i}]: {got} against {w}");
            }
        }
        assert_eq!((fast.cross.w, &fast.wj), (w_before, &wj_before));
        assert_eq!(gram.n_eff(), w_before);
    }

    /// A blend mixes only what both sides have weight in: nothing at all
    /// when either side has none, and per target only a target both sides
    /// have seen -- the other keeps its moments as they were.
    #[test]
    fn a_blend_moves_only_what_both_sides_have() {
        let gaps = TargetGaps::Pairwise;
        let fresh = Acc::new(2, 2, 0, true);
        let full = learned(&stream(20, 8), 0.9, gaps);
        for (what, mine, theirs) in [
            ("a twin with nothing", &full, &fresh),
            ("nothing toward a twin", &fresh, &full),
        ] {
            let mut acc = mine.clone();
            assert!(!acc.blend(theirs, 0.3), "{what}");
            assert_eq!(&acc, mine, "{what}");
        }
        // The second target on one side's rows only.
        let without: Vec<Row> = stream(20, 9)
            .into_iter()
            .map(|(z, y, w)| (z, vec![y[0], None], w))
            .collect();
        let partial = learned(&without, 0.9, gaps);
        for (what, mine, theirs) in [
            ("a target the twin never saw", &full, &partial),
            ("a target this side never saw", &partial, &full),
        ] {
            let mut acc = mine.clone();
            assert!(acc.blend(theirs, 0.3), "{what}: the first target mixes");
            assert_ne!(acc.cross.my[0], mine.cross.my[0], "{what}");
            assert_eq!(acc.wj[1], mine.wj[1], "{what}");
            assert_eq!(acc.cross.my[1], mine.cross.my[1], "{what}");
            assert_eq!(acc.cross.mj[1], mine.cross.mj[1], "{what}");
            assert_eq!(acc.cross.c[1], mine.cross.c[1], "{what}");
            assert_eq!(acc.tm.means()[1], mine.tm.means()[1], "{what}");
            assert_eq!(acc.tm.vars()[1], mine.tm.vars()[1], "{what}");
        }
    }

    /// A row of no weight parts no targets, at the head of a stream or
    /// after it: the present ones' Gram and the absent ones' take the same
    /// step, the decay alone, so the prior's scale ages by `lam` in both.
    /// At the head the present ones' Gram once refused the row (`W' = 0`)
    /// and kept its scale at 1 where the absent ones' aged it, which parted
    /// the two for the stream's life (task 159, R1).
    #[test]
    fn a_row_of_no_weight_parts_no_targets() {
        for first in [None, Some([Some(3.0), Some(1.0)])] {
            let mut g = Grams::new(2, 2, 0, false);
            if let Some(both) = first {
                g.update(&[1.0, 2.0], &both, 1.0, 1.0, TargetGaps::OwnRows);
            }
            g.update(
                &[1.0, 2.0],
                &[Some(3.0), None],
                0.5,
                0.0,
                TargetGaps::OwnRows,
            );
            assert_eq!((g.grams.len(), g.of.clone()), (1, vec![0, 0]));
            assert_eq!(
                g.grams[0].prior_scale(),
                0.5,
                "aged, head row or not: {first:?}"
            );
        }
    }

    /// One corruption per condition [`Cross::has_shape`] checks, each refused
    /// alone.
    #[test]
    fn each_part_of_the_cross_moments_shape_is_checked_alone() {
        let good = Cross::new(2, 3);
        assert!(good.has_shape(2, 3, true));
        let parts: [Corruption<Cross>; 9] = [
            ("a slot short in m", &|c: &mut Cross| {
                c.m.pop();
            }),
            ("a target short in my", &|c: &mut Cross| {
                c.my.pop();
            }),
            ("a target too many in mj", &|c: &mut Cross| {
                c.mj.push(vec![0.0; 3]);
            }),
            ("a target short in c", &|c: &mut Cross| {
                c.c.pop();
            }),
            ("a slot short in one mj", &|c: &mut Cross| {
                c.mj[1].pop();
            }),
            ("a target short in mj_lo", &|c: &mut Cross| {
                c.mj_lo.pop();
            }),
            ("a slot short in one mj_lo", &|c: &mut Cross| {
                c.mj_lo[0].pop();
            }),
            ("a slot short in m_lo", &|c: &mut Cross| {
                c.m_lo.pop();
            }),
            ("a target short in my_lo", &|c: &mut Cross| {
                c.my_lo.pop();
            }),
        ];
        for (what, corrupt) in parts {
            let mut c = good.clone();
            corrupt(&mut c);
            assert!(!c.has_shape(2, 3, true), "{what}");
        }
        // A state's low parts are required, where a state written before
        // them loaded with each at zero (docs/PLAN.md task 198); a window's
        // snapshot carries none, and one that does is not a snapshot.
        let mut old = good.clone();
        old.mj_lo.clear();
        assert!(!old.has_shape(2, 3, true), "a state without the low parts");
        let mut snap = good.clone();
        (snap.m_lo, snap.my_lo, snap.mj_lo) = (Vec::new(), Vec::new(), Vec::new());
        assert!(snap.has_shape(2, 3, false) && !good.has_shape(2, 3, false));
    }

    /// One corruption per condition [`Acc::has_shape`] checks, each refused
    /// alone: a Gram of another width, a reader too many, a reader of a Gram
    /// that does not exist, a Gram nobody reads, a target short in the
    /// weights, and one too many in the target moments.
    #[test]
    fn each_part_of_the_accumulators_shape_is_checked_alone() {
        let good = learned(&stream(6, 2), 0.9, TargetGaps::OwnRows);
        assert!(good.has_shape(2, 2));
        assert_eq!(good.grams.grams.len(), 2, "the second target split off");
        let parts: [Corruption<Acc>; 6] = [
            ("a Gram of three slots", &|a: &mut Acc| {
                a.grams.grams[0] = EwCov::new(3);
            }),
            ("a third reader", &|a: &mut Acc| a.grams.of.push(0)),
            ("a reader past the Grams", &|a: &mut Acc| {
                a.grams.grams.truncate(1);
            }),
            ("a Gram nobody reads", &|a: &mut Acc| {
                a.grams.of = vec![0, 0];
            }),
            ("a target short in the weights", &|a: &mut Acc| {
                a.wj.pop();
            }),
            ("a target too many in the moments", &|a: &mut Acc| {
                a.tm = TargetMoments::new(3);
            }),
        ];
        for (what, corrupt) in parts {
            let mut a = good.clone();
            corrupt(&mut a);
            assert!(!a.has_shape(2, 2), "{what}");
        }
    }

    /// A snapshot's copy of the cross-moments counts every vector it holds:
    /// the weight, `k + T` means, `T·k` own means and as many cross-moments,
    /// the low parts where they are kept -- a snapshot keeps none -- and the
    /// row counts, one per target and one over every row (task 217).
    #[test]
    fn the_cross_moments_footprint_counts_every_vector() {
        let (t, k) = (2, 3);
        let live = Cross::new(t, k);
        let floats = 1 + k + t + (k + t) + 3 * t * k;
        let counts = 1 + t;
        assert_eq!(live.footprint(), 8 * (floats + counts));
        let snap = Acc::new(t, k, 0, true).snapshot(1.0);
        assert_eq!(snap.cross.footprint(), 8 * (1 + k + t + 2 * t * k + counts));
    }

    /// The window's weights are each zero below `EMPTY_FRACTION` of the live
    /// one, judged at any scale, and so is a target's in the window's
    /// moments: over every row (1e-10 left of 1e3), and per target while
    /// the window holds other rows (the second target's 1e-10 of 1e3 beside
    /// the first's whole row). At the fraction exactly it is nothing too: 1
    /// of 1e12, whose fraction is the double 1.
    #[test]
    fn a_window_weight_at_or_below_the_fraction_is_nothing() {
        let gaps = TargetGaps::OwnRows;
        let both = |w: f64| (vec![0.5, 1.0], vec![Some(1.0), Some(2.0)], w);
        let first = |w: f64| (vec![1.5, -1.0], vec![Some(3.0), None], w);
        let second = |w: f64| (vec![-0.5, 2.0], vec![None, Some(-1.0)], w);
        let windowed = |before: &[Row], after: &[Row]| {
            let mut acc = learned(before, 1.0, gaps);
            let snap = acc.snapshot(1.0);
            for (z, y, w) in after {
                acc.learn(z, y, 1.0, *w, gaps);
            }
            (acc, snap)
        };
        // Over every row.
        let (acc, snap) = windowed(&[both(1e3)], &[both(1e-10)]);
        assert_eq!(acc.window_weights(&snap, 1.0), Some((0.0, vec![0.0, 0.0])));
        // The second target alone.
        let (acc, snap) = windowed(&[both(1e3)], &[first(1.0), second(1e-10)]);
        let (w, wj) = acc.window_weights(&snap, 1.0).unwrap();
        assert!(close(w, 1.0, 1e-9) && close(wj[0], 1.0, 1e-9), "{w} {wj:?}");
        assert_eq!(wj[1], 0.0);
        let view = acc.window(&snap, 1.0).unwrap();
        assert_eq!(view.wj[1], 0.0);
        assert!(close(view.wj[0], 1.0, 1e-9));
        // At the fraction exactly.
        let (acc, snap) = windowed(&[both(999_999_999_999.0)], &[both(1.0), first(1e6)]);
        assert_eq!(acc.wj[1] - snap.wj[1], EMPTY_FRACTION * acc.wj[1]);
        let (_, wj) = acc.window_weights(&snap, 1.0).unwrap();
        assert_eq!(wj[1], 0.0);
        let view = acc.window(&snap, 1.0).unwrap();
        assert_eq!(view.wj[1], 0.0);
    }

    /// A target never seen has nothing in the window: weight 0 and every
    /// moment 0, not the 0/0 of a share of no weight.
    #[test]
    fn a_target_never_seen_reads_nothing_in_the_window() {
        let rows: Vec<Row> = stream(12, 4)
            .into_iter()
            .map(|(z, y, w)| (z, vec![y[0], None], w))
            .collect();
        let mut acc = learned(&rows[..6], 0.9, TargetGaps::Pairwise);
        let snap = acc.snapshot(0.9);
        for (z, y, w) in &rows[6..] {
            acc.learn(z, y, 0.9, *w, TargetGaps::Pairwise);
        }
        let view = acc.window(&snap, 0.9_f64.powi(5)).unwrap();
        assert!(view.cross.w > 0.0);
        assert_eq!(view.wj[1], 0.0);
        assert_eq!(view.cross.my[1], 0.0);
        assert_eq!(view.cross.mj[1], [0.0, 0.0]);
        assert_eq!(view.cross.c[1], [0.0, 0.0]);
    }

    /// Kish's count inside the window is no count where the window holds no
    /// weight -- at or below `EMPTY_FRACTION` of the Gram's, at any scale --
    /// whatever its Kish sum's remainder says, and none where that sum
    /// rounds to nothing beside a weight that does not: a row of 1e-9 after
    /// one of 1 leaves `Q = 1 + 1e-18 = 1`. The snapshots' Kish sums are
    /// set by hand to leave the remainder a rounding could.
    #[test]
    fn the_windowed_kish_count_needs_weight_and_a_kish_sum() {
        let gaps = TargetGaps::Pairwise;
        let row = |w: f64| (vec![0.5, 1.0], vec![Some(1.0)], w);
        let windowed = |before: f64, after: Option<f64>| {
            let mut acc = learned(&[row(before)], 1.0, gaps);
            let snap = acc.snapshot(1.0);
            if let Some(w) = after {
                let (z, y, _) = row(w);
                acc.learn(&z, &y, 1.0, w, gaps);
            }
            (acc, snap)
        };
        // A Kish sum that rounds to nothing.
        let (acc, snap) = windowed(1.0, Some(1e-9));
        assert_eq!(acc.grams.grams[0].q_sum(), Some(1.0));
        assert_eq!(acc.window_kish(&snap, 1.0), Some(vec![None]));
        // A window with no weight left, beside a Kish sum's remainder.
        for (before, after) in [
            (1.0, None),
            (1e3, Some(1e-10)),
            (999_999_999_999.0, Some(1.0)),
        ] {
            let (acc, mut snap) = windowed(before, after);
            let q = acc.grams.grams[0].q_sum().unwrap();
            snap.grams.grams[0].q = Some(0.5 * q);
            let got = acc.window_kish(&snap, 1.0);
            assert_eq!(got, Some(vec![None]), "{before} {after:?}");
        }
        // And a window with weight and a Kish sum has its count.
        let (acc, snap) = windowed(1.0, Some(3.0));
        let got = acc.window_kish(&snap, 1.0).unwrap()[0].unwrap();
        assert!(close(got, 1.0, 1e-12), "{got}");
        // The floor is `64 ε` of the live sum, and exclusive: a remainder of
        // exactly that is no count, one of twice that the count, `1² / Q_R`.
        // Two unit rows put the live sum at 2 and the floor at `2^-45`, and
        // both subtractions are exact.
        for (left, count) in [
            (2f64.powi(-45), None),
            (2f64.powi(-44), Some(2f64.powi(44))),
        ] {
            let (acc, mut snap) = windowed(1.0, Some(1.0));
            assert_eq!(acc.grams.grams[0].q_sum(), Some(2.0), "the fixture");
            snap.grams.grams[0].q = Some(2.0 - left);
            assert_eq!(acc.window_kish(&snap, 1.0), Some(vec![count]), "{left:e}");
        }
    }

    /// Kish's sizes inside a window, `W_R² / Q_R` for a Gram and for each
    /// target, are read from remainders of subtractions, and `Q_R` loses its
    /// digits first: a window of light rows holds the squares of their
    /// weights. Where it keeps some, each size is the rows inside the
    /// window's, to the digits kept; where the subtraction leaves none --
    /// `Q_R` within `64 ε` of the sum it came from -- there is no size, not
    /// one made of rounding: the Gram's read 10.75 against 11 with the
    /// window's rows at 1e-6 (review 2026-10-05, CE6b; CE6's rule in
    /// `marginal`). A thousand unit rows, then eleven light ones after the
    /// snapshot, no decay.
    #[test]
    fn kish_sizes_inside_a_window_are_the_rows_inside_or_nothing() {
        let gaps = TargetGaps::Pairwise;
        let row = |i: usize, w: f64| -> Row {
            let z = vec![1.0, 0.5 + 0.01 * (i % 7) as f64];
            (z, vec![Some(1.0 + 0.1 * (i % 5) as f64)], w)
        };
        for (light, digits) in [
            (1e-3, true),
            (1e-5, true),
            (1e-6, false),
            (1e-7, false),
            (1e-9, false),
        ] {
            let heavy: Vec<Row> = (0..1000).map(|i| row(i, 1.0)).collect();
            let mut acc = learned(&heavy, 1.0, gaps);
            let snap = acc.snapshot(1.0);
            for i in 0..11 {
                let (z, y, w) = row(i, light);
                acc.learn(&z, &y, 1.0, w, gaps);
            }
            let gram = acc.window_kish(&snap, 1.0).unwrap()[0];
            let view = acc.window(&snap, 1.0).unwrap();
            let target = view.tm.as_ref().unwrap().n_kish(&view.wj)[0];
            for (what, got) in [("the Gram's", gram), ("the target's", target)] {
                if digits {
                    // Eleven rows of one weight: `(11 w)² / (11 w²) = 11`.
                    let got = got.unwrap_or_else(|| panic!("{light}: {what}: no size"));
                    assert!(close(got, 11.0, 1e-3), "{light}: {what}: {got}");
                } else {
                    assert_eq!(got, None, "{light}: {what}");
                }
            }
        }
    }

    /// A target held at one value over every row inside a window has no
    /// spread there, as a feature held there has none
    /// (`crate::window`'s `a_feature_constant_inside_the_window_has_no_spread_there`):
    /// the window's target moments, which the Gram export reports, read a
    /// variance of exactly 0 and the value held as the mean. The target
    /// side's subtraction had neither of `crate::truncated`'s guards and
    /// read the snapshot's means without their low parts, so a target held
    /// at 1e8 over a window of 30 exported a variance of 2.5e-9 (review
    /// 2026-10-05, CB2). A target that moves inside the window keeps its
    /// spread, against the definition.
    #[test]
    fn a_target_held_over_the_window_has_no_spread_there() {
        use crate::{EwRidge, EwRidgeCfg, OnlineModel};
        for level in [0.5, 1e3, 1e8, -1e8] {
            for (window, h, held) in [
                (30.0, 50.0, 100usize),
                (9.0, 10.0, 12),
                (0.5, 40.0, 3),
                (199.0, 70.0, 230),
                (1999.0, 2.0, 2100),
            ] {
                let mut m = EwRidge::new(EwRidgeCfg {
                    n_features: 1,
                    n_targets: 2,
                    fit_intercept: true,
                    decay: crate::Decay::Halflife(h),
                    ridge: vec![1e-6],
                    feature_sets: vec![],
                    standardize: false,
                    ridge_scale: false,
                    coef_prior: None,
                    session_shrink: None,
                    long_half_life: None,
                    min_weight: 3.0,
                    solve_every: 0.0,
                    max_rows_between_solves: 1,
                    solve_share: None,
                    gram_block_rows: 0,
                    target_gaps: TargetGaps::OwnRows,
                    window: Some(window),
                    window_every: None,
                    max_rows_between_snapshots: None,
                })
                .unwrap();
                // The inclusive edge, which `inside` below is gathered by:
                // what this tests is the guard, on either edge.
                m.set_window_closed(crate::WindowClosed::Both);
                let n = 300 + held;
                let mut s = 2u64;
                let lam = crate::Decay::Halflife(h).factor(1.0);
                let mut inside = Vec::new();
                for i in 0..n {
                    let x = lcg(&mut s);
                    let y0 = if i >= n - held {
                        level + 0.37
                    } else {
                        level + lcg(&mut s)
                    };
                    let y1 = level + 0.5 * x + lcg(&mut s);
                    let w = 0.75 + 0.5 * lcg(&mut s).abs();
                    m.step(
                        &[x],
                        &[Some(y0), Some(y1)],
                        if i == 0 { 0.0 } else { 1.0 },
                        w,
                    );
                    if (i as f64) >= (n - 1) as f64 - window {
                        inside.push((y1, w * lam.powi((n - 1 - i) as i32)));
                    }
                }
                let case = format!("level {level}, window {window}, h {h}");
                let tm = m.gram_parts().1.expect("the window's target moments");
                assert_eq!(tm.vars()[0], 0.0, "{case}: the held target");
                let mean = tm.means()[0];
                assert!(
                    (mean - (level + 0.37)).abs() <= 1e-12 * level.abs().max(1.0),
                    "{case}: the value held is the mean, {mean}"
                );
                let total: f64 = inside.iter().map(|r| r.1).sum();
                let m1 = inside.iter().map(|r| r.0 * r.1).sum::<f64>() / total;
                let v1 = inside.iter().map(|r| r.1 * (r.0 - m1).powi(2)).sum::<f64>() / total;
                if window >= 1.0 {
                    let got = tm.vars()[1];
                    assert!(
                        (got - v1).abs() <= 1e-6 * v1,
                        "{case}: the moving target {got} against {v1}"
                    );
                }
                let _ = OnlineModel::n_outputs(&m);
            }
        }
    }

    /// A target absent long enough that its own weight's per-row decay
    /// reaches the subnormal range keeps that history -- a subnormal times
    /// a decay near 1 rounds back to itself, so it sticks a few steps above
    /// 0 and never reaches it -- and the row it returns on met it: its one
    /// row's Gram had a spread of the history's, of a subnormal size, which a
    /// standardized solve divided by and read the old fit back from
    /// (docs/PLAN.md task 215, raised; task 217). A history whose share of
    /// the weight after the row is below the smallest normal double is
    /// forgotten, as a decay that underflows forgets it, per target and per
    /// Gram: two returns, one from deep in the subnormal range and one from
    /// where it sticks, read what a target never seen before reads. The
    /// first rows carry a weight of `1e-300`, so the history reaches the
    /// subnormal range within a hundred rows of `lam = 0.75` rather than
    /// 2,500 (`0.75` at a clock of 1, so no libm: the subnormals a decay
    /// leaves are the same bits everywhere).
    #[test]
    fn a_target_history_aged_below_the_normal_range_is_forgotten_on_its_return() {
        use crate::{EwRidge, EwRidgeCfg, Lasso, LassoCfg, OnlineModel};
        type R = ([f64; 2], [Option<f64>; 2], f64);
        let row = |seed: u64, second: bool, w: f64| -> R {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5;
            let x = [3.0 * lcg(&mut s), 3.0 * lcg(&mut s)];
            let y0 = 1.0 + x[0] - 0.5 * x[1] + 0.1 * lcg(&mut s);
            let y1 = -1.0 + 0.5 * x[0] + 2.0 * x[1] + 0.1 * lcg(&mut s);
            (x, [Some(y0), second.then_some(y1)], w)
        };
        // Sixty rows of both targets (none of the second where `fresh`), at
        // weight 1e-300; `gap` rows without the second, seeded from the
        // return back, so every gap ends on the same rows; forty of both.
        let stream = |gap: usize, fresh: bool| -> Vec<R> {
            let mut v: Vec<R> = (0..60).map(|i| row(1000 + i, !fresh, 1e-300)).collect();
            v.extend((0..gap).rev().map(|k| row(100_000 + k as u64, false, 1.0)));
            v.extend((0..40).map(|i| row(10 + i, true, 1.0)));
            v
        };
        // The second target's own weight before the return, and its
        // predictions on the rows after it.
        fn after<M: OnlineModel>(mut m: M, rows: &[R], slots: usize) -> (f64, Vec<f64>) {
            let back = rows.len() - 40;
            let (mut own, mut out, mut wj) = (Vec::new(), Vec::new(), f64::NAN);
            for (i, (x, y, w)) in rows.iter().enumerate() {
                if i == back {
                    m.target_n_eff_into(&mut own);
                    wj = own[1];
                }
                let s = m.step(x, y, if i == 0 { 0.0 } else { 1.0 }, *w);
                if i > back {
                    out.extend_from_slice(&s.pred[slots..2 * slots]);
                }
            }
            (wj, out)
        }
        let ridge = |gaps: TargetGaps| {
            EwRidge::new(EwRidgeCfg {
                n_features: 2,
                n_targets: 2,
                fit_intercept: true,
                decay: crate::Decay::Lam(0.75),
                ridge: vec![1e-6],
                feature_sets: vec![],
                standardize: true,
                ridge_scale: false,
                session_shrink: None,
                long_half_life: None,
                coef_prior: None,
                min_weight: 0.0,
                solve_every: 0.0,
                max_rows_between_solves: 1,
                solve_share: None,
                gram_block_rows: 0,
                target_gaps: gaps,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
            })
            .unwrap()
        };
        let lasso = |gaps: TargetGaps| {
            Lasso::new(LassoCfg {
                n_features: 2,
                n_targets: 2,
                fit_intercept: true,
                decay: crate::Decay::Lam(0.75),
                lasso_path: vec![0.1, 0.0],
                l1_ratio: 1.0,
                select_half_life: None,
                min_weight: 0.0,
                target_min_weight: Vec::new(),
                solve_every: 0.0,
                max_rows_between_solves: 1,
                solve_share: None,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
                max_iter: 100,
                tol: 1e-10,
                target_gaps: gaps,
            })
            .unwrap()
        };
        for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
            type Run<'a> = &'a dyn Fn(&[R]) -> (f64, Vec<f64>);
            let runs: [(&str, Run); 2] = [
                ("ewridge", &|r| after(ridge(gaps), r, 1)),
                ("lasso", &|r| after(lasso(gaps), r, 2)),
            ];
            for (name, run) in runs {
                let (deep, a) = run(&stream(100, false));
                let (stuck, b) = run(&stream(300, false));
                let (never, fresh) = run(&stream(300, true));
                // The case: one history deep in the subnormal range, one
                // stuck a few steps above 0, and one never there.
                assert!(deep > 1e-318 && deep < f64::MIN_POSITIVE, "{deep:e}");
                assert!(stuck > 0.0 && stuck < 1e-320, "{stuck:e}");
                assert_eq!(never, 0.0);
                for (i, ((u, v), r)) in a.iter().zip(&b).zip(&fresh).enumerate() {
                    assert!(
                        r.is_finite()
                            && (u - r).abs() <= 1e-9 * (1.0 + r.abs())
                            && (v - r).abs() <= 1e-9 * (1.0 + r.abs()),
                        "{name} {gaps:?}: slot {i} after the return, {u} from a history at \
                         {deep:e} of its weight, {v} from one stuck at {stuck:e}, {r} where the \
                         target was never seen"
                    );
                }
            }
        }
    }

    /// A row of [`window_stream`]'s, from its seed: two features, the first
    /// target on every row, the second where `second`, at weight `w`.
    type CountRow = ([f64; 2], [Option<f64>; 2], f64);

    fn count_row(seed: u64, second: bool, w: f64) -> CountRow {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5;
        let x = [3.0 * lcg(&mut s), 3.0 * lcg(&mut s)];
        let y0 = 1.0 + x[0] - 0.5 * x[1] + 0.1 * lcg(&mut s);
        let y1 = -1.0 + 0.5 * x[0] + 2.0 * x[1] + 0.1 * lcg(&mut s);
        (x, [Some(y0), second.then_some(y1)], w)
    }

    /// Sixty rows of both targets at weight `head` (none of the second where
    /// `fresh`), `gap` rows without the second, seeded from the return back
    /// so every gap ends on the same rows, and forty of both, the first of
    /// them, the return, at weight `back`.
    fn window_stream(gap: usize, fresh: bool, head: f64, back: f64) -> Vec<CountRow> {
        let mut v: Vec<CountRow> = (0..60).map(|i| count_row(1000 + i, !fresh, head)).collect();
        v.extend(
            (0..gap)
                .rev()
                .map(|k| count_row(100_000 + k as u64, false, 1.0)),
        );
        v.extend((0..40).map(|i| count_row(10 + i, true, if i == 0 { back } else { 1.0 })));
        v
    }

    /// `ewridge` (standardized) and `lasso` under a window of 5, at `lam`
    /// 0.75 and a clock of 1, so no libm, with no `min_weight`: what the
    /// window holds of a target is all that decides whether it predicts.
    fn windowed_models(gaps: TargetGaps) -> (crate::EwRidge, crate::Lasso) {
        let ridge = crate::EwRidge::new(crate::EwRidgeCfg {
            n_features: 2,
            n_targets: 2,
            fit_intercept: true,
            decay: crate::Decay::Lam(0.75),
            ridge: vec![1e-6],
            feature_sets: vec![],
            standardize: true,
            ridge_scale: false,
            session_shrink: None,
            long_half_life: None,
            coef_prior: None,
            min_weight: 0.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            gram_block_rows: 0,
            target_gaps: gaps,
            window: Some(5.0),
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let lasso = crate::Lasso::new(crate::LassoCfg {
            n_features: 2,
            n_targets: 2,
            fit_intercept: true,
            decay: crate::Decay::Lam(0.75),
            lasso_path: vec![0.1, 0.0],
            l1_ratio: 1.0,
            select_half_life: None,
            min_weight: 0.0,
            target_min_weight: Vec::new(),
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            window: Some(5.0),
            window_every: None,
            max_rows_between_snapshots: None,
            max_iter: 100,
            tol: 1e-10,
            target_gaps: gaps,
        })
        .unwrap();
        (ridge, lasso)
    }

    /// The second target's window weight before the return row, and every
    /// output of it from the return row on: its predictions, and its window
    /// weight before each row.
    fn from_the_return<M: crate::OnlineModel>(mut m: M, rows: &[CountRow]) -> (f64, Vec<f64>) {
        let back = rows.len() - 40;
        let slots = m.n_outputs() / 2;
        let (mut own, mut out, mut before) = (Vec::new(), Vec::new(), f64::NAN);
        for (i, (x, y, w)) in rows.iter().enumerate() {
            m.target_n_eff_into(&mut own);
            if i == back {
                before = own[1];
            }
            let s = m.step(x, y, if i == 0 { 0.0 } else { 1.0 }, *w);
            if i >= back {
                out.push(own[1]);
                out.extend_from_slice(&s.pred[slots..]);
            }
        }
        (before, out)
    }

    fn same(u: f64, v: f64) -> bool {
        (u.is_nan() && v.is_nan()) || (u - v).abs() <= 1e-9 * (1.0 + v.abs())
    }

    /// Two returns after the window has dropped every row of the target,
    /// at `gaps` rows each, against a target never seen: the window weight
    /// before the return, and each output from the return on.
    fn returns_agree(gap_a: usize, gap_b: usize, head: f64, back: f64) {
        for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
            for which in ["ewridge", "lasso"] {
                let run = |rows: &[CountRow]| {
                    let (ridge, lasso) = windowed_models(gaps);
                    if which == "ewridge" {
                        from_the_return(ridge, rows)
                    } else {
                        from_the_return(lasso, rows)
                    }
                };
                let (wa, a) = run(&window_stream(gap_a, false, head, back));
                let (wb, b) = run(&window_stream(gap_b, false, head, back));
                let (wr, never) = run(&window_stream(gap_b, true, head, back));
                for (what, w) in [("one return", wa), ("the other", wb), ("never seen", wr)] {
                    assert_eq!(
                        w.to_bits(),
                        0f64.to_bits(),
                        "{which} {gaps:?}, the return at weight {back}: {what}'s window weight \
                         before the return is {w:e}, from a window holding none of its rows"
                    );
                }
                for (i, ((u, v), r)) in a.iter().zip(&b).zip(&never).enumerate() {
                    assert!(
                        same(*u, *r) && same(*v, *r),
                        "{which} {gaps:?}, the return at weight {back}: output {i} from the \
                         return, {u} and {v} after gaps of {gap_a} and {gap_b} rows, {r} where \
                         the target was never seen"
                    );
                }
            }
        }
    }

    /// **A window that holds none of a target's rows is empty for it**
    /// (docs/PLAN.md task 217). A target absent longer than the window, the
    /// others going on, kept a live weight stuck a few subnormal steps above
    /// 0 (a subnormal times a decay near 1 rounds back to itself), and the
    /// window's weight, that less the snapshot's, a remainder of rounding:
    /// `5e-324`, and the return row predicted from it where a target never
    /// seen predicts nothing. The window counts the target's rows of positive
    /// weight now, live and in each snapshot, so it holds none exactly. The
    /// first rows carry a weight of `1e-300`, so the history is subnormal
    /// within 100 rows and stuck within 300; the extended test runs the
    /// 2,500 and 3,000 rows of weight 1 it was found on. Both layouts, both
    /// models. Measured on the base, the return row: -3.563 after either gap
    /// under `own_rows` (NaN never seen), 1.607277 and 1.607168 under
    /// `pairwise`.
    #[test]
    fn a_window_holding_none_of_a_targets_rows_is_empty_for_it() {
        returns_agree(100, 300, 1e-300, 1.0);
    }

    /// [`a_window_holding_none_of_a_targets_rows_is_empty_for_it`] on the
    /// stream it was found on: 2,500 and 3,000 rows absent, every weight 1.
    #[test]
    #[ignore = "extended: the stream as found, twelve runs of 3,100 rows (0.7 s in a debug build); the short form keeps the rule in the essentials"]
    fn a_window_holding_none_of_a_targets_rows_is_empty_for_it_as_found() {
        returns_agree(2_500, 3_000, 1.0, 1.0);
    }

    /// **A row of weight 0 is no row of the window's** (hard rule 9; task
    /// 217): a target that returns on a row of weight 0 has still no row in
    /// the window, on that row and the next, and reads as one never seen.
    /// And over every row: a window whose rows all have weight 0 holds none,
    /// and its weight is 0, not the stuck remainder of the history's.
    #[test]
    fn a_zero_weight_row_is_no_row_of_the_window() {
        use crate::OnlineModel;
        returns_agree(100, 300, 1e-300, 0.0);
        // Every row after the first sixty at weight 0, 300 of them: the
        // all-row weight ages into the subnormal range and sticks there.
        for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
            let (mut ridge, mut lasso) = windowed_models(gaps);
            let mut s = 7u64;
            for i in 0..360 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = [Some(x[0] + 0.1 * lcg(&mut s)), Some(x[1])];
                let w = if i < 60 { 1e-300 } else { 0.0 };
                let d = if i == 0 { 0.0 } else { 1.0 };
                let (a, b) = (ridge.step(&x, &y, d, w), lasso.step(&x, &y, d, w));
                if i > 300 {
                    for (name, out) in [("ewridge", &a), ("lasso", &b)] {
                        assert!(
                            out.n_eff == 0.0 && out.pred.iter().all(|p| p.is_nan()),
                            "{name} {gaps:?}, row {i}: weight {:e} and {:?} from a window of \
                             rows of weight 0",
                            out.n_eff,
                            out.pred
                        );
                    }
                }
            }
        }
    }

    /// **Rows of subnormal weight inside a window are rows** (task 217):
    /// the count is of rows of positive weight, whatever their size, where a
    /// floor on the weight's remainder at the smallest normal double would
    /// have emptied the window. Every row at `1e-310` fits as every row at
    /// 1 does: the mean-form moments do not see a common scale.
    #[test]
    fn rows_of_subnormal_weight_inside_a_window_are_rows() {
        use crate::OnlineModel;
        for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
            let fit = |w: f64| {
                let (mut ridge, mut lasso) = windowed_models(gaps);
                let mut s = 9u64;
                let mut out: Vec<f64> = Vec::new();
                for i in 0..60 {
                    let x = [lcg(&mut s), lcg(&mut s)];
                    let y = [
                        Some(0.5 + x[0] - x[1] + 0.1 * lcg(&mut s)),
                        Some(2.0 * x[1] + 0.1 * lcg(&mut s)),
                    ];
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    let (a, b) = (ridge.step(&x, &y, d, w), lasso.step(&x, &y, d, w));
                    if i > 10 {
                        out.extend(a.pred.iter().chain(&b.pred));
                    }
                }
                out
            };
            let (tiny, unit) = (fit(1e-310), fit(1.0));
            assert!(tiny.iter().all(|p| p.is_finite()), "{gaps:?}: {tiny:?}");
            for (i, (u, v)) in tiny.iter().zip(&unit).enumerate() {
                assert!(
                    (u - v).abs() <= 1e-6 * (1.0 + v.abs()),
                    "{gaps:?}: output {i}, {u} at weight 1e-310 and {v} at 1"
                );
            }
        }
    }
}
