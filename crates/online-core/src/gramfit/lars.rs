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
//! the penalties where the active set `A` changes, as columns enter it or,
//! under the lasso modification, leave it. Between knots the correlations
//! `c = d − Rb` of the active columns stay equal in size, `|c_j| = λ`, with
//! the signs `s_j = sign(c_j)` they entered with, and
//!
//! ```text
//! δ_A = (R_AA)⁻¹ s_A                  the direction, in the active coefficients
//! a   = R_{·A} δ_A                    how fast each correlation falls (a_j = s_j on A)
//! b_A ← b_A + γ δ_A,  λ ← λ − γ,  c ← c − γ a
//! ```
//!
//! with `γ` the smallest of: a column `j` outside `A` reaching the active
//! correlation with a sign `s`, `γ⁺ = (λ − s c_j)/(1 − s a_j)` where
//! `1 − s a_j > 0`; an active coefficient reaching 0, `γ⁻ = −b_j/δ_j` where
//! `s_j δ_j < 0`, at which that column leaves (the lasso modification); or
//! `λ` reaching 0, where the path ends at the least-squares fit on `A`. `c`
//! is formed afresh from `b` at each knot rather than stepped by `γ a`, so
//! rounding does not build up along the path.
//!
//! **Ties are read to within rounding.** Every column whose step is within
//! `TIE · λ_max` of the smallest, with [`TIE`] 64 ulps, changes at that one
//! knot. `λ_max` bounds every penalty and correlation on the path, and `λ`
//! is `λ_max` less the steps taken, so it carries an error of a few ulps of
//! `λ_max`: two steps closer than that are one penalty. The first knot takes
//! every column whose `|d_j|` is that close to `λ_max`. A step within the
//! tolerance of 0 belongs to the knot just made, which its column joins.
//! A step below 0 counts as 0: a symmetry that ties two columns exactly
//! once left the second's step at `−2.8e-17`, since `λ` is carried by
//! subtraction and `c` formed afresh, and a test for a positive step passed
//! the column over for good. On designs without ties the closest two
//! knots were 3.9e5 ulps apart, and the path is the one it was before
//! ties were read, to the bit.
//!
//! **The lasso decides which tied columns are active below a knot.** A
//! column whose coefficient is 0 at the knot is in if the direction moves
//! it with its sign, `s_j δ_j > 0`, and out if its correlation then falls
//! at least as fast as `λ`, `s_j a_j ≥ 1`. With one column changing, the
//! block inverse gives `δ_j = (s_j − a_j)/p` for a pivot `p > 0`, where
//! `a_j` is read without it: a column whose step was positive moves with
//! its sign, and one whose coefficient crossed 0 stays out. So it enters,
//! or leaves. With several the choice is a linear complementarity problem,
//! solved by Murty's least-index principal pivoting (1974, "Note on a
//! Bard-type scheme for solving the complementarity problem"), which is
//! finite on a positive definite `R_AA`. All columns tied in correlation
//! do not always enter: on `R = 0.6 I + (−0.4) 11ᵀ` and `d = [1, 1, −1]`
//! all three tie at `λ = 1`, and the direction with all three moves column
//! 2 against its sign. A tied column left out sits at the active
//! correlation, so it may enter on the next step only with the other sign:
//! its step with its own sign is 0, which rounding can make positive.
//!
//! **At a knot where several columns change, the columns that stay are
//! solved afresh.** Steps up to `TIE · λ_max` apart are taken as one, so a
//! column leaving there is set to 0 from up to `TIE · λ_max · |δ_j|` short
//! of it, and `|δ_j|` is large where the active set is nearly collinear
//! (1.6e7 in a test). Solving `b_S = (R_SS)⁻¹ (d_S − λ s_S)` for the columns
//! `S` that stay leaves the conditions off by `TIE · λ_max · |1 − s_j a_j|`
//! at most, since the pivot times `δ_j` is `1 − s_j a_j`. A column that
//! enters a hair short of the active correlation stays that far off it.
//!
//! Only the active columns' rows of `R` are read -- `a` and `c` are sums
//! over them -- so a row is formed when its column first enters
//! ([`CorrRows`]) and a path stopped after `s` steps costs `O(k s²)`, never
//! the `O(k²)` of the whole matrix. `R_AA`'s Cholesky factor gains a row as
//! a column enters, `O(|A|²)`, and is formed afresh when one leaves.
//!
//! A column that would enter collinear with the active set (its pivot in
//! the factor below `1e-12` of its diagonal) is set aside instead: it adds
//! no direction, and its correlation, equal to the active one, would tie
//! every knot after. A duplicate of an active column is one.

/// Steps within `TIE · λ_max` of each other are one knot, and a step within
/// it of 0 is the knot just made (the module doc). 64 ulps, measured in
/// units of `ε λ_max` (docs/PLAN.md task 236): on Grams whose ties are
/// exact as stored, the path's own arithmetic parted tied steps by 31 at
/// most, and the closest two knots of designs without ties were 3.9e5
/// apart. Ties exact only before a Gram's own rounding, as in columns
/// standardized from rows in other orders, came out 1,925 apart at most,
/// 95% within 64. A tie past the tolerance is two knots a hair apart, which
/// is the path of the Gram as it is stored.
pub(super) const TIE: f64 = 64.0 * f64::EPSILON;

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
    /// At the first knot where `max_active` columns or more were active.
    MaxActive,
    /// At `λ = 0`, the least-squares fit on the active set, or with every
    /// column that can enter active.
    End,
}

/// When to stop a path early; `None` is no limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LarsLimits {
    /// Knots past the first, each a penalty of its own, at which one column
    /// or several (at a tie) enter or leave.
    pub max_steps: Option<usize>,
    /// Columns active at once: the path stops at the first knot with this
    /// many or more, more where columns tie there.
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
    /// order its columns entered, those entering at one knot in column
    /// order. A column entering at a knot is in its set with a coefficient
    /// still 0 there, which grows from it; one leaving is out of it, at 0.
    /// So a knot's coefficients are nonzero on the set before it, less the
    /// columns leaving, and the last knot's on its own. The one exception
    /// is a tie that turns a coefficient back as it reaches 0: it stays in
    /// the set, at 0 at the knot.
    pub active: Vec<Vec<usize>>,
    pub stop: LarsStop,
}

/// The lasso path of `f(b)` above, for `R` row-major `k × k` (symmetric)
/// and `d` of length `k`. A column with `live[j]` false never enters, nor
/// one whose `d_j` is not finite. Columns that tie enter or leave at one
/// knot (the module doc).
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
    // A correlation that is not finite has no step to compare: an infinite
    // one would make λ_max infinite and every step's tolerance NaN.
    let mut usable: Vec<bool> = (0..k).map(|j| live[j] && d[j].is_finite()).collect();
    // The first knot: λ_max, every coefficient 0.
    let lambda_max = (0..k)
        .filter(|&j| usable[j])
        .fold(0.0f64, |m, j| m.max(c[j].abs()));
    if lambda_max <= 0.0 {
        path.record(0.0, &b, &[]);
        return path;
    }
    let tie = TIE * lambda_max;
    let mut lambda = lambda_max;
    let mut state = Active::new(k);
    let mut a = vec![0.0; k];
    // Every column within `tie` of λ_max enters there. A column alone is
    // collinear with nothing (its pivot is its diagonal, above 0 for a live
    // column), so the first of them does.
    let tied: Vec<(usize, f64)> = (0..k)
        .filter(|&j| usable[j] && lambda - c[j].abs() <= tie)
        .map(|j| (j, c[j].signum()))
        .collect();
    // The columns at 0 and at the active correlation at the current knot,
    // each with its sign: those that entered or left there, and those the
    // lasso left out.
    let mut knot = state.admit(rows, &tied, &mut usable);
    state.settle(rows, &knot, &mut usable, &mut a);
    path.record(lambda, &b, &state.cols);
    if limits.max_active.is_some_and(|m| state.cols.len() >= m) {
        path.stop = LarsStop::MaxActive;
        return path;
    }

    let mut steps = 0usize;
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
        // A column outside A: its step to the active correlation, with the
        // sign it would enter with. A step below 0 is a tie that rounding
        // put past the knot, and counts as 0. One the knot left out sits at
        // the active correlation with its sign, a step of 0 that rounding
        // could make positive, so with that sign it is passed over.
        let entry = |j: usize| -> Option<(f64, f64)> {
            if !usable[j] || state.is_in[j] {
                return None;
            }
            let mut best: Option<(f64, f64)> = None;
            for s in [1.0, -1.0] {
                let den = 1.0 - s * a[j];
                if den > 0.0 && !knot.contains(&(j, s)) {
                    let g = (lambda - s * c[j]).max(0.0) / den;
                    if best.is_none_or(|(least, _)| g < least) {
                        best = Some((g, s));
                    }
                }
            }
            best
        };
        // An active coefficient: its step to 0, if it moves toward 0. One at
        // 0 was settled in at this knot; one past 0 by rounding is at 0
        // already.
        let leave = |pos: usize| -> Option<f64> {
            let s = state.signs[pos];
            let (size, toward) = (s * b[state.cols[pos]], s * delta[pos]);
            (toward < 0.0 && size != 0.0).then(|| size.max(0.0) / -toward)
        };
        let mut gamma = f64::INFINITY;
        for j in 0..k {
            if let Some((g, _)) = entry(j) {
                gamma = gamma.min(g);
            }
        }
        for pos in 0..state.cols.len() {
            if let Some(g) = leave(pos) {
                gamma = gamma.min(g);
            }
        }
        if gamma >= lambda - tie {
            // λ reaches 0 first: the least-squares fit on A, the path's end.
            for (&i, &di) in state.cols.iter().zip(&delta) {
                b[i] += lambda * di;
            }
            path.record(0.0, &b, &state.cols);
            return path;
        }
        // Every step within `tie` of the smallest is at one knot.
        let mut entering = Vec::new();
        for j in 0..k {
            if let Some((g, s)) = entry(j)
                && g <= gamma + tie
            {
                entering.push((j, s));
            }
        }
        let leaving: Vec<usize> = (0..state.cols.len())
            .filter(|&pos| leave(pos).is_some_and(|g| g <= gamma + tie))
            .collect();
        // A step within `tie` of 0 is at the knot just made: its columns
        // join that knot, which is recorded again.
        let same_knot = gamma <= tie;
        let moving = state.cols.len();
        let mut changed = Vec::new();
        if leaving.is_empty() {
            // A column collinear with the active set adds no direction: it
            // is set aside before the path moves, and the step is found
            // again without it. Its pivot reads the active set alone, not
            // `b`, so the test needs no step; taking the step first made a
            // knot where nothing entered or left (review 6, E-5).
            changed = state.admit(rows, &entering, &mut usable);
            if changed.is_empty() {
                continue;
            }
        }
        if same_knot {
            path.unrecord(k);
        } else {
            for (&i, &di) in state.cols[..moving].iter().zip(&delta) {
                b[i] += gamma * di;
            }
            lambda -= gamma;
            steps += 1;
            knot.clear();
        }
        if !leaving.is_empty() {
            for (i, s) in state.leave_all(&leaving) {
                b[i] = 0.0;
                changed.push((i, s));
            }
            changed.extend(state.admit(rows, &entering, &mut usable));
        }
        knot.retain(|&(j, _)| changed.iter().all(|&(i, _)| i != j));
        knot.extend(changed);
        knot.sort_by_key(|&(j, _)| j);
        state.settle(rows, &knot, &mut usable, &mut a);
        if knot.len() > 1 {
            // Steps up to `tie` apart were taken as one, so a column leaving
            // here was set to 0 from up to `tie |δ_j|`, and `|δ_j|` is large
            // where the active set is nearly collinear. The columns staying
            // are solved afresh at this penalty, which leaves the conditions
            // off by `tie |1 − s_j a_j|` at most (the module doc).
            state.anchor(d, lambda, &knot, &mut b);
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

    /// The last knot, of `k` coefficients, taken back to be recorded again.
    fn unrecord(&mut self, k: usize) {
        self.penalties.pop();
        self.coefs.truncate(self.coefs.len() - k);
        self.active.pop();
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

    /// Each of `cols` in, in turn, with its sign; one collinear with the
    /// active set is set aside for good (`usable` false). The ones that
    /// entered.
    fn admit(
        &mut self,
        rows: &impl CorrRows,
        cols: &[(usize, f64)],
        usable: &mut [bool],
    ) -> Vec<(usize, f64)> {
        let mut entered = Vec::with_capacity(cols.len());
        for &(j, s) in cols {
            if self.enter(rows, j, s) {
                entered.push((j, s));
            } else {
                usable[j] = false;
            }
        }
        entered
    }

    /// The columns at `positions` (ascending) out, returning each with its
    /// sign; the factor is formed afresh over the rest.
    fn leave_all(&mut self, positions: &[usize]) -> Vec<(usize, f64)> {
        let mut gone = Vec::with_capacity(positions.len());
        for &pos in positions.iter().rev() {
            let i = self.cols.remove(pos);
            gone.push((i, self.signs.remove(pos)));
            self.rows.remove(pos);
            self.is_in[i] = false;
        }
        self.factor = Factor::default();
        for p in 0..self.cols.len() {
            let cross: Vec<f64> = self.cols[..p].iter().map(|&q| self.rows[p][q]).collect();
            // Every column here entered once, so none is collinear with the
            // ones before it, and one leaving can only make the rest less
            // so: the floor is for rounding, never reached in a test.
            self.factor.push(&cross, self.rows[p][self.cols[p]], true);
        }
        gone
    }

    /// Where column `j` sits in the active set, if it is in it.
    fn position(&self, j: usize) -> Option<usize> {
        if !self.is_in[j] {
            return None;
        }
        self.cols.iter().position(|&i| i == j)
    }

    /// Which columns of `knot` -- each at 0 at a knot and at the active
    /// correlation, with its sign, in column order -- are active below it
    /// (the module doc). In: the direction moves it with its sign,
    /// `s_j δ_j ≥ 0`. Out: its correlation falls at least as fast as `λ`,
    /// `s_j a_j ≥ 1`. Both within [`TIE`]. Murty's least-index rule flips
    /// the first column that fails its test until none does, which Murty
    /// proved finite on a positive definite `R_AA`. The loop stops at `2ⁿ`
    /// flips for `n` columns, the number of sets to choose between, and at
    /// 65,536 past 16 columns. The most measured was 10, among six columns
    /// all tied at `λ_max` (docs/PLAN.md task 236). A single column needs no
    /// flip: it enters if its step was positive, and leaves if its
    /// coefficient crossed 0.
    fn settle(
        &mut self,
        rows: &impl CorrRows,
        knot: &[(usize, f64)],
        usable: &mut [bool],
        a: &mut [f64],
    ) {
        if knot.len() < 2 {
            return;
        }
        for _ in 0..1usize << knot.len().min(16) {
            let delta = self.factor.solve(&self.signs);
            self.combine(&delta, a);
            let size = delta.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let fails = knot.iter().copied().find(|&(j, s)| match self.position(j) {
                Some(pos) => s * delta[pos] < -TIE * size,
                None => usable[j] && s * a[j] < 1.0 - TIE,
            });
            let Some((j, s)) = fails else {
                return;
            };
            match self.position(j) {
                Some(pos) => {
                    self.leave_all(&[pos]);
                }
                None => {
                    if !self.enter(rows, j, s) {
                        usable[j] = false;
                    }
                }
            }
        }
    }

    /// The coefficients of the active columns outside `knot` solved afresh
    /// at `λ`, `b_S = (R_SS)⁻¹ (d_S − λ s_S)`, where their correlations are
    /// `λ` in size. Those columns lead the active set, since the ones of a
    /// knot entered there, so `R_SS`'s factor is the leading block of the
    /// active set's.
    fn anchor(&self, d: &[f64], lambda: f64, knot: &[(usize, f64)], b: &mut [f64]) {
        let staying = self
            .cols
            .iter()
            .take_while(|&&j| knot.iter().all(|&(i, _)| i != j))
            .count();
        debug_assert!(
            self.cols[staying..]
                .iter()
                .all(|&j| knot.iter().any(|&(i, _)| i == j))
        );
        let rhs: Vec<f64> = (0..staying)
            .map(|p| d[self.cols[p]] - lambda * self.signs[p])
            .collect();
        for (&j, v) in self.cols.iter().zip(self.factor.solve(&rhs)) {
            b[j] = v;
        }
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

    /// `x` with `L Lᵀ x = s`, on the leading `s.len()` rows of `L`: the
    /// factor of the matrix's leading block.
    fn solve(&self, s: &[f64]) -> Vec<f64> {
        let n = s.len();
        debug_assert!(n <= self.rows.len());
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
