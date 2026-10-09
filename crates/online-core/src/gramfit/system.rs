//! A regression on a set of a Gram's columns, as `polars_online.gram.solve`
//! and `lasso_path` form it: the matrix, shared by every target of the
//! Gram, and each target's right-hand side.
//!
//! With the intercept at column `c` and the slots `s`:
//!
//! ```text
//! A = C[s, s]                     r_t = cross_centred[t][s]
//! m_t = means_by_target[t][s]     ȳ_t = cross_moments[t][c]
//! ```
//!
//! centred at each target's own means, and through the origin
//!
//! ```text
//! A = C[s, s] + μ_s μ_sᵀ          r_t = cross_moments[t][s]       ȳ_t = 0
//! ```
//!
//! the raw second moments, with `C` the co-moments and `μ` the column
//! means. The slopes `b` solve `(A + penalty) b = r_t`, and the intercept
//! is `ȳ_t − m_t · b`. `A` does not depend on the target: the co-moments
//! are the Gram's, and only the cross-moments are each target's.

/// The arrays of one Gram, as `ModelBank.gram()` exports them, borrowed:
/// `k` columns and `m` targets, every matrix row-major.
#[derive(Clone, Copy, Debug)]
pub struct GramArrays<'a> {
    pub k: usize,
    /// The column means, `k`.
    pub means: &'a [f64],
    /// The centred co-moments, `k × k`.
    pub comoments: &'a [f64],
    /// Per target, `E[z y]` over its rows, `m × k`.
    pub cross_moments: &'a [f64],
    /// Per target, the column means over its rows, `m × k`.
    pub means_by_target: &'a [f64],
    /// Per target, `E[(z − m_t)(y − ȳ_t)]`, `m × k`.
    pub cross_centred: &'a [f64],
}

impl GramArrays<'_> {
    /// The number of targets.
    pub fn targets(&self) -> usize {
        self.cross_moments.len().checked_div(self.k).unwrap_or(0)
    }
}

/// `A` over the slots, `n × n` row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Design {
    pub a: Vec<f64>,
    pub n: usize,
}

/// One target's side of the system: `r_t`, `m_t` over the slots, and `ȳ_t`.
#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    pub rhs: Vec<f64>,
    pub means: Vec<f64>,
    pub ybar: f64,
}

/// The scale of each slot in the correlation form, `S = diag(√A_ii)`, and
/// whether it is live. A column with `A_ii = 0` (constant on the Gram's
/// rows) is dead: its scale is 1, its `d` 0, its row and column of `R` 0
/// but for a 1 on the diagonal, and its coefficient is held at 0 -- the
/// `lasso` model's rule.
#[derive(Clone, Debug, PartialEq)]
pub struct Scaling {
    pub scale: Vec<f64>,
    pub live: Vec<bool>,
}

/// [`Design`]'s correlation form, `R = S⁻¹ A S⁻¹`, held whole.
#[derive(Clone, Debug, PartialEq)]
pub struct Correlation {
    pub r: Vec<f64>,
    pub scaling: Scaling,
}

/// [`Correlation`] read a row at a time straight from a Gram's
/// co-moments, for a path that reads only its active rows
/// ([`super::CorrRows`]): the same numbers, row for row, without forming
/// `A` or `R` whole.
#[derive(Clone, Debug)]
pub struct GramRows<'a> {
    g: GramArrays<'a>,
    slots: &'a [usize],
    icept: Option<usize>,
    pub scaling: Scaling,
}

impl Design {
    /// `A` on `slots`, with the intercept at `icept` or none.
    pub fn of(g: &GramArrays<'_>, slots: &[usize], icept: Option<usize>) -> Self {
        let k = g.k;
        let n = slots.len();
        let mut a = vec![0.0; n * n];
        for (p, &i) in slots.iter().enumerate() {
            for (q, &j) in slots.iter().enumerate() {
                a[p * n + q] = g.comoments[i * k + j];
            }
        }
        if icept.is_none() {
            for (p, &i) in slots.iter().enumerate() {
                for (q, &j) in slots.iter().enumerate() {
                    a[p * n + q] += g.means[i] * g.means[j];
                }
            }
        }
        Self { a, n }
    }
}

impl Response {
    /// Target `t`'s side on `slots`, with the intercept at `icept` or none.
    pub fn of(g: &GramArrays<'_>, t: usize, slots: &[usize], icept: Option<usize>) -> Self {
        let k = g.k;
        let row = |v: &[f64]| -> Vec<f64> { slots.iter().map(|&i| v[t * k + i]).collect() };
        match icept {
            Some(c) => Self {
                rhs: row(g.cross_centred),
                means: row(g.means_by_target),
                ybar: g.cross_moments[t * k + c],
            },
            None => Self {
                rhs: row(g.cross_moments),
                means: slots.iter().map(|&i| g.means[i]).collect(),
                ybar: 0.0,
            },
        }
    }

    /// The coefficients over a Gram's `k` columns from slopes `b` over the
    /// slots in the correlation form's basis: `b_p / s_p` (0 for a dead
    /// column) at each slot, the intercept `ȳ − m · b` at `icept`, 0
    /// elsewhere.
    pub fn coef(
        &self,
        corr: &Scaling,
        b: &[f64],
        k: usize,
        slots: &[usize],
        icept: Option<usize>,
    ) -> Vec<f64> {
        let mut out = vec![0.0; k];
        let mut fit = 0.0;
        for (p, &i) in slots.iter().enumerate() {
            let v = if corr.live[p] {
                b[p] / corr.scale[p]
            } else {
                0.0
            };
            out[i] = v;
            fit += v * self.means[p];
        }
        if let Some(c) = icept {
            out[c] = self.ybar - fit;
        }
        out
    }
}

impl Scaling {
    /// The scaling of a matrix whose diagonal is `diag`.
    pub fn of(diag: impl Iterator<Item = f64>) -> Self {
        let s: Vec<f64> = diag.map(|v| v.max(0.0).sqrt()).collect();
        Self {
            live: s.iter().map(|&v| v > 0.0).collect(),
            scale: s.iter().map(|&v| if v > 0.0 { v } else { 1.0 }).collect(),
        }
    }

    /// `R_pq` from `A_pq`.
    fn entry(&self, a: f64, p: usize, q: usize) -> f64 {
        if self.live[p] && self.live[q] {
            a / (self.scale[p] * self.scale[q])
        } else if p == q {
            1.0
        } else {
            0.0
        }
    }

    /// A target's `d = S⁻¹ r_t`, 0 at a dead column.
    pub fn d(&self, resp: &Response) -> Vec<f64> {
        resp.rhs
            .iter()
            .zip(&self.scale)
            .zip(&self.live)
            .map(|((&r, &s), &live)| if live { r / s } else { 0.0 })
            .collect()
    }
}

impl Correlation {
    /// The correlation form of `A`.
    pub fn of(design: &Design) -> Self {
        let n = design.n;
        let a = &design.a;
        let scaling = Scaling::of((0..n).map(|i| a[i * n + i]));
        let mut r = vec![0.0; n * n];
        for p in 0..n {
            for q in 0..n {
                r[p * n + q] = scaling.entry(a[p * n + q], p, q);
            }
        }
        Self { r, scaling }
    }
}

impl<'a> GramRows<'a> {
    /// The correlation form of `A` on `slots`, with the intercept at
    /// `icept` or none, read from `g` as it is asked for.
    pub fn new(g: GramArrays<'a>, slots: &'a [usize], icept: Option<usize>) -> Self {
        let mut rows = Self {
            g,
            slots,
            icept,
            scaling: Scaling {
                scale: Vec::new(),
                live: Vec::new(),
            },
        };
        rows.scaling = Scaling::of((0..slots.len()).map(|p| rows.a(p, p)));
        rows
    }

    /// `A_pq`, as [`Design::of`] forms it.
    fn a(&self, p: usize, q: usize) -> f64 {
        let (i, j) = (self.slots[p], self.slots[q]);
        let c = self.g.comoments[i * self.g.k + j];
        match self.icept {
            Some(_) => c,
            None => c + self.g.means[i] * self.g.means[j],
        }
    }
}

impl super::CorrRows for GramRows<'_> {
    fn order(&self) -> usize {
        self.slots.len()
    }

    fn row(&self, p: usize, out: &mut [f64]) {
        for (q, o) in out.iter_mut().enumerate() {
            *o = self.scaling.entry(self.a(p, q), p, q);
        }
    }
}
