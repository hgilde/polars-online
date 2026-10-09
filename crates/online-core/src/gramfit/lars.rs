//! The lasso path by least angle regression (Efron, Hastie, Johnstone and
//! Tibshirani 2004, "Least angle regression", Annals of Statistics 32(2),
//! with their lasso modification of section 3.1), on a Gram in correlation
//! form: no rows, only `R` and `d`.
//!
//! The lasso minimizes, over `b`,
//!
//! ```text
//! f(b) = ½ b'Rb − d'b + λ Σ_j |b_j|
//! ```
//!
//! and its solution is piecewise linear in `λ`. LARS walks the pieces from
//! `λ_max = max_j |d_j|`, where every coefficient is 0, down: the knots are
//! where a column enters the active set `A` or, under the lasso
//! modification, leaves it. Between knots the correlations `c = d − Rb` of
//! the active columns stay equal in size, `|c_j| = λ`, with the signs
//! `s_j = sign(c_j)` they entered with, and
//!
//! ```text
//! δ_A = (R_AA)⁻¹ s_A                  the direction, in the active coefficients
//! a   = R_{·A} δ_A                    how fast each correlation falls (a_j = s_j on A)
//! b_A ← b_A + γ δ_A,  λ ← λ − γ,  c ← c − γ a
//! ```
//!
//! with `γ` the first of: a column `j` outside `A` reaching the active
//! correlation, `γ⁺ = min⁺{(λ − c_j)/(1 − a_j), (λ + c_j)/(1 + a_j)}`; an
//! active coefficient reaching 0, `γ⁻ = min⁺{−b_j/δ_j}`, where that column
//! leaves (the lasso modification); or `λ` reaching 0, where the path ends
//! at the least-squares fit on `A`. `min⁺` is the least strictly positive
//! value. `c` is formed afresh from `b` at each knot rather than stepped by
//! `γ a`, so rounding does not build up along the path.
//!
//! Only the active columns' rows of `R` are read -- `a` and `c` are sums
//! over them -- so a row is formed when its column first enters
//! ([`CorrRows`]) and a path stopped after `s` steps costs `O(k s²)`, never
//! the `O(k²)` of the whole matrix. `R_AA`'s Cholesky factor gains a row as
//! a column enters, `O(|A|²)`, and is formed afresh when one leaves.
//!
//! A column that would enter collinear with the active set (its pivot in
//! the factor below `1e-12` of its diagonal) is set aside instead: it adds
//! no direction, and its correlation, equal to the active one, would make
//! it the next to enter again at a step of 0. A column that has just left
//! sits at the active correlation with the sign it had, so on the next step
//! it may enter only with the other sign: the root with its own sign is a
//! step of 0, which rounding can make `1e-17`.

/// Rows of a `k × k` symmetric matrix `R`, formed on request: a path reads
/// only its active columns' rows, so `R` need never be formed whole.
pub trait CorrRows {
    /// `k`.
    fn order(&self) -> usize;
    /// Row `i` of `R` into `out`, of length `k`.
    fn row(&self, i: usize, out: &mut [f64]);
}

/// A matrix held whole, row-major.
#[derive(Clone, Copy, Debug)]
pub struct DenseRows<'a> {
    pub r: &'a [f64],
    pub k: usize,
}

impl CorrRows for DenseRows<'_> {
    fn order(&self) -> usize {
        self.k
    }

    fn row(&self, i: usize, out: &mut [f64]) {
        out.copy_from_slice(&self.r[i * self.k..(i + 1) * self.k]);
    }
}

/// Where a path stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LarsStop {
    /// After `max_steps` knots past the first.
    MaxSteps,
    /// At the knot where `max_active` columns were active.
    MaxActive,
    /// At `λ = 0`, the least-squares fit on the active set, or with every
    /// column that can enter active.
    End,
}

/// When to stop a path early; `None` is no limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LarsLimits {
    /// Knots past the first, each one column entering or leaving.
    pub max_steps: Option<usize>,
    /// Columns active at once.
    pub max_active: Option<usize>,
}

/// A lasso path: one entry per knot, in the order walked (`λ` falling).
#[derive(Clone, Debug, PartialEq)]
pub struct LarsPath {
    /// `λ` at each knot. The first is `λ_max`, with every coefficient 0.
    pub penalties: Vec<f64>,
    /// The coefficients at each knot, `n_knots × k` row-major, in the basis
    /// of `R` and `d`.
    pub coefs: Vec<f64>,
    /// The active set on the stretch of the path below each knot, in the
    /// order its columns entered. A column entering at a knot is in its set
    /// with a coefficient still 0 there, which grows from it; one leaving is
    /// out of it, at 0. So a knot's coefficients are nonzero on the set
    /// before it, less the one leaving, and the last knot's on its own.
    pub active: Vec<Vec<usize>>,
    pub stop: LarsStop,
}

/// The lasso path of `f(b)` above, for `R` row-major `k × k` (symmetric)
/// and `d` of length `k`. A column with `live[j]` false never enters. Ties
/// between columns go to the lower index.
pub fn lars_lasso(r: &[f64], d: &[f64], live: &[bool], limits: LarsLimits) -> LarsPath {
    lars_lasso_rows(&DenseRows { r, k: d.len() }, d, live, limits)
}

/// [`lars_lasso`] reading `R` a row at a time.
pub fn lars_lasso_rows(
    rows: &impl CorrRows,
    d: &[f64],
    live: &[bool],
    limits: LarsLimits,
) -> LarsPath {
    let k = d.len();
    debug_assert_eq!(rows.order(), k);
    debug_assert_eq!(live.len(), k);
    let mut path = LarsPath {
        penalties: Vec::new(),
        coefs: Vec::new(),
        active: Vec::new(),
        stop: LarsStop::End,
    };
    let mut b = vec![0.0; k];
    let mut c = d.to_vec();
    let mut usable: Vec<bool> = live.to_vec();
    // The first knot: λ_max, every coefficient 0, and the column with the
    // largest correlation about to enter.
    let Some(first) = argmax_abs(&c, &usable) else {
        path.record(0.0, &b, &[]);
        return path;
    };
    let mut lambda = c[first].abs();
    if lambda <= 0.0 {
        path.record(0.0, &b, &[]);
        return path;
    }
    let mut state = Active::new(k);
    // A column alone is collinear with nothing: its pivot is its diagonal,
    // and a live column's is above 0.
    if !state.enter(rows, first, c[first].signum()) {
        usable[first] = false;
    }
    path.record(lambda, &b, &state.cols);
    if limits.max_active.is_some_and(|m| state.cols.len() >= m) {
        path.stop = LarsStop::MaxActive;
        return path;
    }

    let mut steps = 0usize;
    let mut a = vec![0.0; k];
    let mut just_left: Option<(usize, f64)> = None;
    loop {
        if limits.max_steps.is_some_and(|m| steps >= m) {
            path.stop = LarsStop::MaxSteps;
            return path;
        }
        if state.cols.is_empty() {
            // Nothing could enter: the path is 0 to the end.
            path.record(0.0, &b, &state.cols);
            return path;
        }
        let delta = state.factor.solve(&state.signs);
        // How fast each column's correlation falls along the direction.
        state.combine(&delta, &mut a);
        // The first column outside A to reach the active correlation.
        let mut gamma = lambda;
        let mut enter: Option<(usize, f64)> = None;
        for j in 0..k {
            if !usable[j] || state.is_in[j] {
                continue;
            }
            for (g, s) in [
                ((lambda - c[j]) / (1.0 - a[j]), 1.0),
                ((lambda + c[j]) / (1.0 + a[j]), -1.0),
            ] {
                if just_left == Some((j, s)) {
                    continue;
                }
                if g > 0.0 && g < gamma {
                    gamma = g;
                    enter = Some((j, s));
                }
            }
        }
        // The first active coefficient to reach 0 (the lasso modification).
        let mut leave: Option<usize> = None;
        for (pos, (&i, &di)) in state.cols.iter().zip(&delta).enumerate() {
            let g = -b[i] / di;
            if g > 0.0 && g < gamma {
                gamma = g;
                leave = Some(pos);
                enter = None;
            }
        }
        for (&i, &di) in state.cols.iter().zip(&delta) {
            b[i] += gamma * di;
        }
        lambda -= gamma;
        steps += 1;
        just_left = None;
        if let Some(pos) = leave {
            let (i, s) = state.leave(pos);
            b[i] = 0.0;
            just_left = Some((i, s));
        } else if let Some((j, s)) = enter {
            if !state.enter(rows, j, s) {
                usable[j] = false;
            }
        } else {
            // λ reached 0: the least-squares fit on A, the path's end.
            path.record(0.0, &b, &state.cols);
            return path;
        }
        state.correlations(d, &b, &mut c);
        path.record(lambda, &b, &state.cols);
        if limits.max_active.is_some_and(|m| state.cols.len() >= m) {
            path.stop = LarsStop::MaxActive;
            return path;
        }
    }
}

/// [`lars_lasso`] with a penalty weight per column, `λ Σ_j w_j |b_j|` in
/// place of `λ Σ_j |b_j|`: the plain path in `b̃_j = w_j b_j`, on
/// `R̃_ij = R_ij / (w_i w_j)` and `d̃_j = d_j / w_j`, read back as
/// `b_j = b̃_j / w_j`. Every live column's weight must be finite and above
/// 0: a weight of 0 leaves a column unpenalized, active from the start
/// whatever `λ`, which is no point a path walked from `λ_max` can start
/// at (`Err` naming the column).
pub fn lars_lasso_weighted(
    rows: &impl CorrRows,
    d: &[f64],
    live: &[bool],
    weights: &[f64],
    limits: LarsLimits,
) -> Result<LarsPath, String> {
    let k = d.len();
    debug_assert_eq!(weights.len(), k);
    if let Some(j) = (0..k).find(|&j| live[j] && !(weights[j].is_finite() && weights[j] > 0.0)) {
        return Err(format!(
            "penalty weight {} for column {j}: a path from λ_max takes weights finite and above 0",
            weights[j]
        ));
    }
    if weights.iter().all(|&w| w == 1.0) {
        return Ok(lars_lasso_rows(rows, d, live, limits));
    }
    let w: Vec<f64> = (0..k)
        .map(|j| if live[j] { weights[j] } else { 1.0 })
        .collect();
    let dt: Vec<f64> = (0..k).map(|j| d[j] / w[j]).collect();
    let mut path = lars_lasso_rows(&Weighted { rows, w: &w }, &dt, live, limits);
    for row in path.coefs.chunks_mut(k.max(1)) {
        for (v, wj) in row.iter_mut().zip(&w) {
            *v /= wj;
        }
    }
    Ok(path)
}

/// `R̃_ij = R_ij / (w_i w_j)`, a row at a time.
struct Weighted<'a, R> {
    rows: &'a R,
    w: &'a [f64],
}

impl<R: CorrRows> CorrRows for Weighted<'_, R> {
    fn order(&self) -> usize {
        self.rows.order()
    }

    fn row(&self, i: usize, out: &mut [f64]) {
        self.rows.row(i, out);
        for (v, wj) in out.iter_mut().zip(self.w) {
            *v /= self.w[i] * wj;
        }
    }
}

impl LarsPath {
    fn record(&mut self, lambda: f64, b: &[f64], active: &[usize]) {
        self.penalties.push(lambda.max(0.0));
        self.coefs.extend_from_slice(b);
        self.active.push(active.to_vec());
    }

    /// The number of knots.
    pub fn len(&self) -> usize {
        self.penalties.len()
    }

    /// Whether the path has no knot (never: a path has at least its first).
    pub fn is_empty(&self) -> bool {
        self.penalties.is_empty()
    }
}

/// The usable column with the largest `|c_j|`, the lowest index on a tie.
fn argmax_abs(c: &[f64], usable: &[bool]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for j in 0..c.len() {
        if usable[j] && best.is_none_or(|i| c[j].abs() > c[i].abs()) {
            best = Some(j);
        }
    }
    best
}

/// The active set: its columns in the order they entered, their signs, the
/// factor of `R_AA`, and their rows of `R`.
struct Active {
    cols: Vec<usize>,
    signs: Vec<f64>,
    rows: Vec<Vec<f64>>,
    is_in: Vec<bool>,
    factor: Factor,
}

impl Active {
    fn new(k: usize) -> Self {
        Self {
            cols: Vec::new(),
            signs: Vec::new(),
            rows: Vec::new(),
            is_in: vec![false; k],
            factor: Factor::default(),
        }
    }

    /// Column `j` in, with sign `s`; false (and nothing changed) if it is
    /// collinear with the active set.
    fn enter(&mut self, rows: &impl CorrRows, j: usize, s: f64) -> bool {
        let mut row = vec![0.0; rows.order()];
        rows.row(j, &mut row);
        let cross: Vec<f64> = self.cols.iter().map(|&i| row[i]).collect();
        if !self.factor.push(&cross, row[j], false) {
            return false;
        }
        self.cols.push(j);
        self.signs.push(s);
        self.rows.push(row);
        self.is_in[j] = true;
        true
    }

    /// The column at `pos` out, returning it and its sign; the factor is
    /// formed afresh over the rest.
    fn leave(&mut self, pos: usize) -> (usize, f64) {
        let i = self.cols.remove(pos);
        let s = self.signs.remove(pos);
        self.rows.remove(pos);
        self.is_in[i] = false;
        self.factor = Factor::default();
        for p in 0..self.cols.len() {
            let cross: Vec<f64> = self.cols[..p].iter().map(|&q| self.rows[p][q]).collect();
            // Every column here entered once, so none is collinear with the
            // ones before it, and one leaving can only make the rest less
            // so: the floor is for rounding, never reached in a test.
            self.factor.push(&cross, self.rows[p][self.cols[p]], true);
        }
        (i, s)
    }

    /// `out = R_{·A} v`.
    fn combine(&self, v: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (row, &vi) in self.rows.iter().zip(v) {
            for (o, &r) in out.iter_mut().zip(row) {
                *o += r * vi;
            }
        }
    }

    /// `c = d − R_{·A} b_A`.
    fn correlations(&self, d: &[f64], b: &[f64], c: &mut [f64]) {
        let ba: Vec<f64> = self.cols.iter().map(|&i| b[i]).collect();
        self.combine(&ba, c);
        for (cj, &dj) in c.iter_mut().zip(d) {
            *cj = dj - *cj;
        }
    }
}

/// The lower Cholesky factor of `R_AA`, by rows: row `i` holds `i + 1`
/// entries.
#[derive(Default)]
struct Factor {
    rows: Vec<Vec<f64>>,
}

impl Factor {
    /// A pivot below this share of its diagonal is collinear with the rows
    /// before it.
    const COLLINEAR: f64 = 1e-12;

    /// Append a column whose entries against the factored ones are `cross`
    /// and whose diagonal is `diag`; false (and the factor unchanged) if it
    /// is collinear with them, or with `force` its pivot floored at the
    /// threshold instead.
    fn push(&mut self, cross: &[f64], diag: f64, force: bool) -> bool {
        let n = self.rows.len();
        debug_assert_eq!(cross.len(), n);
        let mut l: Vec<f64> = Vec::with_capacity(n + 1);
        for (row, &x) in self.rows.iter().zip(cross) {
            let s: f64 = row.iter().zip(&l).map(|(r, v)| r * v).sum();
            l.push((x - s) / row[l.len()]);
        }
        let pivot = diag - l.iter().map(|v| v * v).sum::<f64>();
        let floor = Self::COLLINEAR * diag;
        // NaN is collinear too: it is no pivot to divide by.
        if pivot > floor {
            l.push(pivot.sqrt());
        } else if force {
            l.push(floor.max(f64::MIN_POSITIVE).sqrt());
        } else {
            return false;
        }
        self.rows.push(l);
        true
    }

    /// `x` with `L Lᵀ x = s`.
    fn solve(&self, s: &[f64]) -> Vec<f64> {
        let n = self.rows.len();
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t: f64 = (0..i).map(|m| self.rows[i][m] * y[m]).sum();
            y[i] = (s[i] - t) / self.rows[i][i];
        }
        for i in (0..n).rev() {
            let t: f64 = (i + 1..n).map(|m| self.rows[m][i] * y[m]).sum();
            y[i] = (y[i] - t) / self.rows[i][i];
        }
        y
    }
}
