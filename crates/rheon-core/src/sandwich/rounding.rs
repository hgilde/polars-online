//! Test-only: bounds on how far a [`super::Sandwich`]'s covariance and a
//! definition of it summed afresh from the rows may part through rounding
//! (docs/PLAN.md task 274). Shared by the sandwich's own tests and by
//! `calibration`'s Newey–West ones, which run a sandwich over `(1, pred)`.
//!
//! The model sums its bread, meat and lag sums about an origin, its first
//! folded row, each decayed by `λ` on every step (the meat and lags by
//! `λ²`, itself a rounding) and added to on each folded row: [`Track`]
//! replays that run with a running bound on each sum's error (Higham §3.3).
//! The definition sums the rows at their present weights about a level the
//! test chooses, then maps to the features' own units. Both errors are
//! carried to `V = B⁻¹MB⁻¹` to first order -- `2|B⁻¹|ΔB|V| +
//! |B⁻¹|ΔM|B⁻¹|` -- in the frame each side sums in, where its sums are
//! well conditioned, and mapped to the features' own units by `V = S V_c
//! Sᵀ`, `S = [[1, −c'], [0, I]]` for the frame's origin `c`, as `|S|·
//! |ΔV_c|·|S|ᵀ`.

use std::collections::VecDeque;

use crate::oracle::{self, U, gamma};

/// `|A|·|B|` for row-major `n × n` matrices.
pub(crate) fn abs_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n).map(|m| (a[i * n + m] * b[m * n + j]).abs()).sum()
        })
        .collect()
}

/// `A·B` for row-major `n × n` matrices.
pub(crate) fn mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n).map(|m| a[i * n + m] * b[m * n + j]).sum()
        })
        .collect()
}

fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    (0..n * n).map(|ij| a[(ij % n) * n + ij / n]).collect()
}

/// `S = [[1, −c'], [0, I]]` for an origin `c` (its first entry unused).
fn frame(c: &[f64]) -> Vec<f64> {
    let n = c.len();
    (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            if i == j {
                1.0
            } else if i == 0 {
                -c[j]
            } else {
                0.0
            }
        })
        .collect()
}

/// A folded row of the definition: its features, its residual, its
/// present weight, and the step it was folded on, whose distance from the
/// present is the number of times the weight has been decayed.
#[derive(Clone, Debug)]
pub(crate) struct DefRow {
    pub(crate) x: Vec<f64>,
    pub(crate) e: f64,
    pub(crate) w: f64,
    pub(crate) born: usize,
}

/// The model's sums' error bounds and their terms' absolute sums, as the
/// model forms them: bread, meat, each lag, and the ring of scores with
/// their own errors, about the model's origin. An intercept is assumed.
pub(crate) struct Track {
    d: usize,
    lags: usize,
    origin: Option<Vec<f64>>,
    ab: Vec<f64>,
    eb: Vec<f64>,
    am: Vec<f64>,
    em: Vec<f64>,
    ag: Vec<Vec<f64>>,
    eg: Vec<Vec<f64>>,
    ring: VecDeque<(Vec<f64>, Vec<f64>)>,
}

impl Track {
    pub(crate) fn new(k: usize, lags: usize) -> Self {
        let d = k + 1;
        Self {
            d,
            lags,
            origin: None,
            ab: vec![0.0; d * d],
            eb: vec![0.0; d * d],
            am: vec![0.0; d * d],
            em: vec![0.0; d * d],
            ag: vec![vec![0.0; d * d]; lags],
            eg: vec![vec![0.0; d * d]; lags],
            ring: VecDeque::new(),
        }
    }

    /// The model's origin, `(0, c)`.
    pub(crate) fn origin(&self) -> Vec<f64> {
        self.origin.clone().expect("a row folded")
    }

    /// [`super::Sandwich::update`]'s run: every sum decayed (one rounding
    /// of the partial sum, two for the meat and lags, `λ²` rounded first),
    /// and on a folded row its terms as the model forms them -- `z` about
    /// the origin (one rounding), the score `u = w e z` (two more), each
    /// product one more, each addition `u` of its sum.
    pub(crate) fn update(&mut self, x: &[f64], e: f64, lam: f64, w: f64) {
        let d = self.d;
        let l2 = lam * lam;
        for v in 0..d * d {
            self.ab[v] *= lam;
            self.eb[v] = lam * self.eb[v] + U * self.ab[v];
            self.am[v] *= l2;
            self.em[v] = l2 * self.em[v] + 2.0 * U * self.am[v];
            for l in 0..self.lags {
                self.ag[l][v] *= l2;
                self.eg[l][v] = l2 * self.eg[l][v] + 2.0 * U * self.ag[l][v];
            }
        }
        for (u, eu) in self.ring.iter_mut() {
            for c in 0..d {
                u[c] *= lam;
                eu[c] = lam * eu[c] + U * u[c].abs();
            }
        }
        if !(e.is_finite() && w > 0.0) {
            return;
        }
        let o = self
            .origin
            .get_or_insert_with(|| std::iter::once(0.0).chain(x.iter().copied()).collect())
            .clone();
        let z: Vec<f64> = std::iter::once(1.0)
            .chain(x.iter().zip(&o[1..]).map(|(v, c)| v - c))
            .collect();
        let u: Vec<f64> = z.iter().map(|zc| w * e * zc).collect();
        let eu: Vec<f64> = u.iter().map(|v| 3.0 * U * v.abs()).collect();
        for a in 0..d {
            for c in 0..d {
                let v = a * d + c;
                let tb = (w * z[a] * z[c]).abs();
                self.ab[v] += tb;
                self.eb[v] += 4.0 * U * tb + U * self.ab[v];
                let tm = (u[a] * u[c]).abs();
                self.am[v] += tm;
                self.em[v] += eu[a] * u[c].abs() + u[a].abs() * eu[c] + U * tm + U * self.am[v];
                for (l, (back, eback)) in self.ring.iter().enumerate() {
                    let tg = (u[a] * back[c]).abs();
                    self.ag[l][v] += tg;
                    self.eg[l][v] +=
                        eu[a] * back[c].abs() + u[a].abs() * eback[c] + U * tg + U * self.ag[l][v];
                }
            }
        }
        self.ring.push_front((u, eu));
        self.ring.truncate(self.lags);
    }
}

/// The bread and the meat (Newey and West's with `reach` lags of `lags`,
/// or the diagonal's) over `rows` at their present weights, `z` about
/// `origin`, in the definition's order -- each row's bread term, then for
/// each row each lag's meat term, Bartlett's weight times the product of
/// the two rows' `w·e·z` -- and bounds on how far its own rounding has
/// moved each entry: a term's products, `2u` of it in the bread and `6u`
/// in the meat, and each addition `u` of the partial sum it makes. The
/// weights' decays are no rounding of the sums' entries but of the rows'
/// weights, and [`weight_error`] carries them.
pub(crate) fn definition(
    rows: &[DefRow],
    origin: &[f64],
    reach: usize,
    lags: usize,
) -> [Vec<f64>; 4] {
    let d = origin.len();
    let n = rows.len();
    let z = |r: &DefRow| -> Vec<f64> {
        std::iter::once(1.0)
            .chain(r.x.iter().zip(&origin[1..]).map(|(v, c)| v - c))
            .collect()
    };
    let (mut b, mut m, mut eb, mut em) = (
        vec![0.0; d * d],
        vec![0.0; d * d],
        vec![0.0; d * d],
        vec![0.0; d * d],
    );
    for r in rows {
        let zr = z(r);
        for a in 0..d {
            for c in 0..d {
                let t = r.w * zr[a] * zr[c];
                b[a * d + c] += t;
                eb[a * d + c] += 2.0 * U * t.abs() + U * b[a * d + c].abs();
            }
        }
    }
    for t in 0..n {
        for l in 0..=reach.min(t) {
            let (rt, rs) = (&rows[t], &rows[t - l]);
            let (zt, zs) = (z(rt), z(rs));
            let k = if l == 0 {
                1.0
            } else {
                1.0 - l as f64 / (lags as f64 + 1.0)
            };
            for a in 0..d {
                for c in 0..d {
                    let p = rt.w * rt.e * zt[a] * (rs.w * rs.e * zs[c]);
                    let e = 6.0 * U * (k * p).abs();
                    m[a * d + c] += k * p;
                    em[a * d + c] += e + U * m[a * d + c].abs();
                    if l > 0 {
                        m[c * d + a] += k * p;
                        em[c * d + a] += e + U * m[c * d + a].abs();
                    }
                }
            }
        }
    }
    [b, m, eb, em]
}

/// How far the definition's decays of each row's weight move `V =
/// B⁻¹MB⁻¹`, `z` about `origin`: weight `w_r`, decayed `age_r` times by
/// repeated products, is within `age_r·u` of itself, and `V` moves by
/// `∂V/∂(ln w_r)` times that, `∂V = B⁻¹·∂M·B⁻¹ − B⁻¹·∂B·V − V·∂B·B⁻¹` with
/// `∂B = w_r z_r z_rᵀ` and `∂M` every meat term holding row `r`, twice its
/// own square. A row's weight scales its terms together, so this is no
/// rounding of the sums' entries one by one.
fn weight_error(
    rows: &[DefRow],
    now: usize,
    (origin, reach, lags): (&[f64], usize, usize),
    bi: &[f64],
    v: &[f64],
) -> Vec<f64> {
    let d = origin.len();
    let n = rows.len();
    let z: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            std::iter::once(1.0)
                .chain(r.x.iter().zip(&origin[1..]).map(|(x, c)| x - c))
                .collect()
        })
        .collect();
    let u: Vec<Vec<f64>> = rows
        .iter()
        .zip(&z)
        .map(|(r, zr)| zr.iter().map(|zc| r.w * r.e * zc).collect())
        .collect();
    let k = |l: usize| {
        if l == 0 {
            1.0
        } else {
            1.0 - l as f64 / (lags as f64 + 1.0)
        }
    };
    let mut out = vec![0.0; d * d];
    for r in 0..n {
        let mut dm = vec![0.0; d * d];
        for l in 0..=reach {
            for (t, s) in [(r, r.wrapping_sub(l)), (r + l, r)] {
                if s >= n || t >= n || (l == 0 && t != r) {
                    continue;
                }
                let mult = if l == 0 { 2.0 } else { 1.0 };
                for a in 0..d {
                    for c in 0..d {
                        let p = k(l) * u[t][a] * u[s][c];
                        dm[a * d + c] += mult * p;
                        if l > 0 {
                            dm[c * d + a] += p;
                        }
                    }
                }
            }
        }
        let db: Vec<f64> = (0..d * d)
            .map(|ac| rows[r].w * z[r][ac / d] * z[r][ac % d])
            .collect();
        let bdmb = mul(&mul(bi, &dm, d), bi, d);
        let bdbv = mul(&mul(bi, &db, d), v, d);
        let age = (now - rows[r].born) as f64 * U;
        for ac in 0..d * d {
            let ca = (ac % d) * d + ac / d;
            out[ac] += (bdmb[ac] - bdbv[ac] - bdbv[ca]).abs() * age;
        }
    }
    out
}

/// The covariance by its definition at step `now`, summed about `level`
/// and mapped to the features' own units, the intercept first; and, entry
/// by entry, a bound on how far the model's `covariance` may be off it:
/// the definition's own rounding (its sums, its weights' decays, the
/// oracle's inverse by its residual, the triple product's `γ₁₁` --
/// `γ_{4d−1}` -- and the map's) beside the model's, which [`Track`]
/// carried along its run: its sums' errors and two Cholesky solves of its
/// bread (`|ΔX| ≤ |B⁻¹|E|X|` for `X = B⁻¹M`, `|ΔV| ≤ |B⁻¹|E|V| +
/// |B⁻¹||ΔX|ᵀ`), the symmetrization, and the map from its origin, with
/// each of the map's sums rounding `γ_d` of its terms.
pub(crate) fn covariance_bound(
    track: &Track,
    rows: &[DefRow],
    now: usize,
    level: &[f64],
    (reach, lags): (usize, usize),
) -> Bound {
    let d = level.len();
    let [b, m, eb, em] = definition(rows, level, reach, lags);
    let bi = oracle::inverse(&b);
    let mut vt = vec![0.0; d * d];
    for a in 0..d {
        for c in 0..d {
            for p in 0..d {
                for q in 0..d {
                    vt[a * d + c] += bi[a * d + p] * m[p * d + q] * bi[q * d + c];
                }
            }
        }
    }
    let sm = frame(level);
    let st = transpose(&sm, d);
    let v = mul(&mul(&sm, &vt, d), &st, d);
    let abs_bi: Vec<f64> = bi.iter().map(|x| x.abs()).collect();
    let dbi = oracle::inverse_error(&b);
    let weights = weight_error(rows, now, (level, reach, lags), &bi, &vt);
    let bmb = abs_mul(&abs_mul(&bi, &m, d), &bi, d);
    let t1 = abs_mul(&abs_mul(&abs_bi, &eb, d), &vt, d);
    let t2 = abs_mul(&abs_mul(&abs_bi, &em, d), &abs_bi, d);
    let t3 = abs_mul(&abs_mul(&dbi, &m, d), &bi, d);
    let dvt: Vec<f64> = (0..d * d)
        .map(|ij| {
            let ji = (ij % d) * d + ij / d;
            t1[ij] + t1[ji] + t2[ij] + t3[ij] + t3[ji] + gamma(4 * d - 1) * bmb[ij] + weights[ij]
        })
        .collect();
    let svs = abs_mul(&abs_mul(&sm, &vt, d), &st, d);
    let back = abs_mul(&abs_mul(&sm, &dvt, d), &st, d);
    let test: Vec<f64> = (0..d * d)
        .map(|ij| back[ij] + gamma(2 * d) * svs[ij])
        .collect();

    // The model's side, about its origin.
    let o = track.origin();
    let [bc, mc, _, _] = definition(rows, &o, reach, lags);
    let bci = oracle::inverse(&bc);
    let abs_bci: Vec<f64> = bci.iter().map(|x| x.abs()).collect();
    let xc = mul(&bci, &mc, d);
    let vc = mul(&xc, &bci, d);
    let mut emc = track.em.clone();
    if reach > 0 {
        for (l, (eg, ag)) in track.eg.iter().zip(&track.ag).enumerate() {
            let bl = 1.0 - (l + 1) as f64 / (lags as f64 + 1.0);
            for ij in 0..d * d {
                let ji = (ij % d) * d + ij / d;
                emc[ij] += bl * (eg[ij] + eg[ji]) + 3.0 * U * bl * (ag[ij] + ag[ji]);
            }
        }
    }
    let e = oracle::cholesky_backward(&bc);
    let t1 = abs_mul(&abs_mul(&abs_bci, &track.eb, d), &vc, d);
    let t2 = abs_mul(&abs_mul(&abs_bci, &emc, d), &abs_bci, d);
    let s1 = abs_mul(&abs_mul(&abs_bci, &e, d), &vc, d);
    let dx = abs_mul(&abs_mul(&abs_bci, &e, d), &xc, d);
    let s2 = abs_mul(&abs_bci, &transpose(&dx, d), d);
    let dvc: Vec<f64> = (0..d * d)
        .map(|ij| {
            let ji = (ij % d) * d + ij / d;
            t1[ij] + t1[ji] + t2[ij] + s1[ij] + s2[ij] + U * vc[ij].abs()
        })
        .collect();
    let so = frame(&o);
    let sot = transpose(&so, d);
    let lib_back = abs_mul(&abs_mul(&so, &dvc, d), &sot, d);
    let lib_map = abs_mul(&abs_mul(&so, &vc, d), &sot, d);
    let dv = (0..d * d)
        .map(|ij| test[ij] + lib_back[ij] + gamma(2 * d) * lib_map[ij])
        .collect();
    // The same in the level's frame: the definition's before its map, and
    // the model's mapped from its origin to the level, `V_l = R V_o Rᵀ`
    // with `R = [[1, −(c_o − c_l)'], [0, I]]`, whose entries are the two
    // origins' small difference. The model's own map to the features'
    // units rounds in that frame as `γ_{2d}` of `|S||V||S|ᵀ` there, which
    // `S⁻¹` carries back to the level's.
    let rel: Vec<f64> = o.iter().zip(level).map(|(a, b)| a - b).collect();
    let r = frame(&rel);
    let rt = transpose(&r, d);
    let lib_level = abs_mul(&abs_mul(&r, &dvc, d), &rt, d);
    let sl_inv = frame(&level.iter().map(|c| -c).collect::<Vec<_>>());
    let sl_inv_t = transpose(&sl_inv, d);
    let map_level = abs_mul(&abs_mul(&sl_inv, &lib_map, d), &sl_inv_t, d);
    let dv_level = (0..d * d)
        .map(|ij| dvt[ij] + gamma(2 * d) * map_level[ij] + lib_level[ij])
        .collect();
    Bound {
        v,
        dv,
        v_level: vt,
        dv_level,
    }
}

/// [`covariance_bound`]'s answer: the covariance in the features' own
/// units and its bound entry by entry, and the same in the level's frame,
/// `V = S·V_l·Sᵀ`, where a quadratic form in `V⁻¹` is better bounded:
/// `d'V⁻¹d = d_l'V_l⁻¹d_l` with `d_l = S⁻¹d`.
pub(crate) struct Bound {
    pub(crate) v: Vec<f64>,
    pub(crate) dv: Vec<f64>,
    pub(crate) v_level: Vec<f64>,
    pub(crate) dv_level: Vec<f64>,
}
