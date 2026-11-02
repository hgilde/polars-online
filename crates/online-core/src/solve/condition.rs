//! Whether a factorized system can carry a digit (docs/PLAN.md task 278,
//! D4): an estimate of its condition from the factor, in `O(k²)`, and the
//! limit past which the solve's forward-error bound says nothing.
//!
//! A Cholesky factorization of a matrix past `1/u` in condition does not
//! always fail: rounding can leave the computed matrix positive definite, and
//! the factor then solves a matrix as far from the one meant as its smallest
//! eigenvalue, so the coefficients are off by more than their own size. The
//! jitter ladder catches the matrices whose factorization fails; this catches
//! the ones whose factorization succeeds.
//!
//! **Which condition.** A Cholesky solve is invariant to a diagonal scaling
//! of the system (van der Sluis's theorem; Higham, *Accuracy and Stability
//! of Numerical Algorithms*, 2nd ed., §7.3 and §10.1): its accuracy, in the
//! coordinates `D x` with `D = diag(√a_ii)`,
//! is governed by the equilibrated matrix `H = D⁻¹ A D⁻¹`, whose diagonal is
//! 1, not by `A` itself. So a design whose features run from `1e-10` to
//! `1e10` is no harder than one of unit scales, and it must not be refused.
//! `H` is the correlation matrix where the system is a centred Gram, and its
//! factor is `D⁻¹ L`, `L` the factor of `A`.
//!
//! **The estimate.** `‖H⁻¹‖₁` by Hager's method as Higham refined it
//! (LAPACK's `dlacn2`; Higham, Algorithm 15.4), which needs only products
//! with `H⁻¹`: two triangular solves each, `O(k²)`, about four of them in
//! all. Each iterate it reads is `‖H⁻¹ v‖₁` for a `v` of unit 1-norm, so the
//! estimate never exceeds `‖H⁻¹‖₁`, and in practice it is within a factor of
//! 3 of it and usually exact (Higham, §15.3). The ratio of the factor's
//! largest and smallest pivots squared is cheaper (`k` reads) but bounds the
//! condition only from below, and can miss it by orders of magnitude: a
//! nearly dependent column whose dependence is spread over many others keeps
//! every pivot moderate (Kahan's matrix, Higham §8.3, is the classic case).
//! Missing the condition here means a garbage fit is reported, so the
//! estimate that is usually exact is the one used.
//!
//! **The screen.** Most systems are far inside the limit, and an upper bound
//! that is cheaper than the estimate says so: with `R = D⁻¹ L` and `M(R)`
//! its comparison matrix (`|r_ii|` on the diagonal, `−|r_ij|` off it),
//! `|R⁻¹| ≤ M(R)⁻¹` entrywise (Higham, §8.3), so `‖H⁻¹‖₁ ≤ ‖R⁻ᵀ‖₁ ‖R⁻¹‖₁
//! = ‖R⁻¹‖∞ ‖R⁻¹‖₁ ≤ max(M(R)⁻¹ 1) · max(M(R)⁻ᵀ 1)`, two substitutions
//! whose terms are all nonnegative, a quarter of one of the estimate's
//! products. Where that bound is inside the limit the system is, and the
//! estimate is not run; where it is not -- it can exceed `‖H⁻¹‖₁` by a
//! factor that grows with `k` when the columns are correlated -- the
//! estimate decides. Either way the answer is the estimate's wherever the
//! estimate could reach the limit.
//!
//! **The limit.** The factorization and its two triangular solves are
//! backward stable: the computed `x̂` solves `(A + ΔA) x̂ = b` with `|ΔA| ≤
//! γ_{3k+1} |Lᵀ||L|` (Higham, Theorem 10.4), `γ_m = m u / (1 − m u)`. By
//! Cauchy–Schwarz each entry of `|L||Lᵀ|` is at most `√(a_ii a_jj)`, the
//! row norms of `L` being the diagonal of `A`, so in `H`'s coordinates every
//! entry of `ΔH = D⁻¹ ΔA D⁻¹` is at most `γ_{3k+1}`, and `‖ΔH‖₁ ≤ k
//! γ_{3k+1}`. Relative to `‖H‖₁` that is a backward error `η = k γ_{3k+1} /
//! ‖H‖₁`, and with the right-hand side formed in the same arithmetic, the
//! normwise forward error of the scaled solution is at most `2 κ₁ η / (1 −
//! κ₁ η)` while `κ₁ η < 1` (Higham, Theorem 7.2). With `κ₁ = ‖H‖₁ ‖H⁻¹‖₁`
//! the norms of `H` cancel: `κ₁ η = k γ_{3k+1} ‖H⁻¹‖₁`. The bound reaches 1
//! -- no digit of the solution guaranteed -- at `κ₁ η = 1/3`, so a system is
//! refused where
//!
//! ```text
//! 3 k γ_{3k+1} ‖H⁻¹‖₁ ≥ 1.
//! ```
//!
//! At `k = 3` that is `‖H⁻¹‖₁ ≥ 1.0e14`, at `k = 10` `9.7e12`, at `k = 50`
//! `4.0e11`. The bound is a worst case, which the data's own rounding (the
//! moments' accumulation) only adds to, so the limit is where no digit is
//! promised, not where one is. Measured 2026-10-25 (task 278 (d4)): NIST's
//! hardest designs the normal equations can solve read `‖H⁻¹‖₁ u` at most
//! `2.4e-10` (Wampler's, `k = 5`, against the limit's `4.2e-3`) and
//! Longley's `9.6e-13` (`k = 6`), over every row of the stream; NIST's
//! Filip, past `1/u`, read from `0.47` to `45` (`k = 10`, limit `1.1e-3`);
//! and every factorization that succeeded on `x₁ = x₀ + eps·b` at `eps` of
//! `1e-11` and `1e-13` read at least `0.028` (`k = 3`, limit `0.011`; 7,545
//! solves over 20 streams of 800 rows, their first ten rows left out).
//!
//! **A factor that needed jitter** factors `A + δI`, a ridge the solver
//! chose with `δ` at least `1e-12` of the mean diagonal, and is judged as
//! the system it solves: `H = D⁻¹ (A + δI) D⁻¹` with `D = diag(√(a_ii + δ))`.
//! The ridge bounds that condition, but not inside the limit at every
//! order. On a matrix whose factorization fails as given -- one with a
//! direction of no variance, its smallest eigenvalue 0 to rounding -- the
//! first rung's `δ`, `1e-12` of the mean diagonal, leaves `‖H⁻¹‖₂` near
//! `1e12`, and `‖H⁻¹‖₁` at least that and up to `k` times it. The limit is
//! `1.7e12` at `k = 24`, below `1e12` from `k = 32` and `4.0e11` at `k = 50`,
//! so a first-rung solve can be past it from the twenties, and is from the
//! thirties. Such a solve was served unjudged, its jitter counted as a
//! failure; it is refused now, as a factor past the limit that needed none
//! is.
//!
//! **The ridge's shortcut.** A solve whose matrix is known to be a ridge
//! over a semidefinite matrix -- `A = C + ρI`, `C` the centred co-moments,
//! `ρ > 0` -- is decided in `k` reads where the ridge alone holds it inside
//! the limit ([`SpdFactor::carries_no_digit_over`]). The comparison bound
//! does not settle wide systems: at a thousand features it read `1e14` to
//! `1e59` against a limit of `1e9` on every solve of a default fit, so each
//! paid for the bound and the estimate, about five serial `k²` passes. The
//! shortcut answers "not refused" only where the estimate itself, as
//! computed, is below the limit, so it decides as the long way does,
//! always; anywhere else the long way runs. The argument, with `u = 2⁻⁵³`,
//! `γ_m = m u / (1 − m u)`, `n` the rows of positive weight behind `C`,
//! `T = Σ a_ii` and `t = max a_ii` (the system's diagonal, ridge included,
//! before any jitter), and `δ` the factor's jitter:
//!
//! 1. *`C` is semidefinite to within its rounding, by an amount that grows
//!    with `n`.* An [`crate::EwCov`] row takes `c_ij ← a c_ij + (a b δ_i)
//!    δ_j`, `a, ab ≥ 0`. In exact arithmetic on the same computed `a`, `ab`
//!    and `δ` that recursion sums rank-one terms with nonnegative weights,
//!    `C* = Σ_t α_t δ_t δ_tᵀ`, semidefinite. In floating point a term takes
//!    three roundings at its own row and two (`a·c`, the sum) at each later
//!    one, so `c_ij = Σ_t α_t δ_ti δ_tj (1 + θ_t)` with `|θ_t| ≤ γ_{2n+1}`
//!    (Higham, Lemma 3.1), and `|C − C*| ≤ γ_{2n+1} M` entrywise, `M = Σ_t
//!    α_t |δ_t||δ_t|ᵀ`. `M` is semidefinite with `C*`'s diagonal, so `‖C −
//!    C*‖₂ ≤ ‖ |C − C*| ‖₂ ≤ γ_{2n+1} ‖M‖₂ ≤ γ_{2n+1} tr C*`: `λ_min(C) ≥
//!    −γ_{2n+1} tr C*`. A row of weight 0 rounds nothing (`a = 1` and `b =
//!    0` exactly), so `n` counts the rows of positive weight,
//!    [`crate::EwCov::rows_learned`]. The count is the stream's, not the
//!    weight `W`'s: rows of small weight at the mean round `a·c_ij` on
//!    every row while `W` stands still, and the entries' relative errors
//!    walk apart for as long as the stream runs (a test drives a rank-one
//!    `C` past any slack proportional to `W`).
//! 2. *The system as formed.* `A`'s off-diagonal entries are `C`'s, or with
//!    `standardize` `c_ij / (s_i s_j)` with `s_i = √c_ii` (four roundings),
//!    and its diagonal takes `ρ` with one more. So `A = S⁻¹ C* S⁻¹ + ρI +
//!    G`, `S = diag(√c_ii)` or `I`, and `‖G‖₂ ≤ γ_{2n+5} tr(S⁻¹ M S⁻¹) + u
//!    t/(1 − u)` (Lemma 3.3 joins the two `γ`). Either way `tr(S⁻¹ M S⁻¹)
//!    ≤ T / ((1 − γ_5)(1 − γ_{2n+1}))`: unscaled, `tr M = tr C* ≤ tr C / (1
//!    − γ_{2n+1})` (every term of a diagonal entry is nonnegative) and `tr C
//!    ≤ T/(1 − u)`; standardized, each `M_ii / c_ii ≤ 1/(1 − γ_{2n+1})` and
//!    `k ≤ T/(1 − γ_5)`. A factor that needed jitter holds `A + δI`, its
//!    diagonal rounded once more: `λ_min(A + δI) ≥ ρ + δ − g₁`, `g₁ ≤ 1.05
//!    ((2n + 5) u T + u (2t + δ))` once `(2n + 6) u` and `(3k + 1) u` are at
//!    most `1/100`, which the shortcut requires, and which puts every `γ_m`
//!    here within `1.0102 m u` and every `1/(1 − γ)` within `1.0103`.
//! 3. *The factor and its solves.* Whatever order faer sums in, with or
//!    without fused multiply-adds, the computed factor `L̂` and the two
//!    triangular solves give each product with `H⁻¹` as the exact solution
//!    of `(A + δI + ΔA) z = b`, `|ΔA| ≤ γ_{3k+1} |L̂||L̂ᵀ|` (Higham, Lemma
//!    8.4 and Theorem 10.4), and `‖ |L̂||L̂ᵀ| ‖₂ ≤ ‖L̂‖_F² = Σ_i (L̂L̂ᵀ)_ii ≤
//!    (T + kδ)(1 + u)/(1 − γ_{k+1})`. So `σ_min(A + δI + ΔA) ≥ μ = ρ + δ −
//!    g`, `g = g₁ + g₂`, `g₂ ≤ 1.05 (3k + 1) u (T + kδ)`.
//! 4. *The estimate.* In exact arithmetic each candidate Hager's method
//!    keeps is `‖H⁻¹ v‖₁ / ‖v‖₁ ≤ ‖H⁻¹‖₁` (the estimate's docs). As
//!    computed, each is a product of item 3 -- scaled by `D` before and
//!    after, `D`'s entries `√(a_ii + δ)` rounded -- whose 1-norm, summed,
//!    is divided by its vector's: `1`, `1`, or `3k/2` for the alternating
//!    vector, whose product's norm is taken times `2/(3k)`. With `‖D X D‖₁
//!    ≤ √k ‖D X D‖₂ ≤ √k (t + δ) ‖X‖₂`, every candidate, and so the
//!    estimate, is at most `(1 + γ_{k+8}) √k (t + δ) / μ`: nine roundings
//!    of the scaling, the vectors and the division, `k − 1` of the sum.
//! 5. *The test.* With `ĝ` the slack `(2n + 5) u T + (3k + 1) u (T + kδ) +
//!    u (2t + δ)` computed from the summed diagonal -- at least `g / 1.05`,
//!    its few roundings and the sum's `γ_{k−1}` taken in -- the shortcut
//!    requires `1.25 ĝ ≤ (ρ + δ)/2`. Then `g < 0.43 (ρ + δ)`,
//!    `μ > 0.57 (ρ + δ)`, and every candidate is below
//!    `1.78 √k (t + δ)/(ρ + δ)`. So where `2 √k (t + δ)/(ρ + δ)`, computed
//!    in five roundings, is below the limit, the estimate is below it too,
//!    the long way would not refuse the system, and the shortcut does not.
//!
//! Every step assumes Higham's model, no overflow and no underflow
//! (§2.2). Overflow cannot have happened: `A`'s entries are finite where
//! its factorization succeeded (a non-finite entry makes a pivot NaN or
//! `−∞`, which faer refuses), and an overflow in `C`'s history stays
//! (an infinity times an `a` in `(0, 1]` stays one, times 0 is NaN). The
//! shortcut requires `2⁻⁴⁰⁰ ≤ t ≤ 2⁴⁰⁰`; with the test passed, every
//! number the estimate forms then lies within `2^±900`, and an underflow's
//! absolute error, `2⁻¹⁰⁷⁴`, falls hundreds of binades under a rounding of
//! the result. Two things are not covered. An underflowing product in
//! `C`'s history, `ab·δ_i` below `2⁻¹⁰²²`, then multiplied by a `δ_j` past
//! about `1e150`, leaves an absolute error the current matrix cannot
//! bound. And a state file edited by hand can hold any matrix at all.
//!
//! Which systems take it: [`crate::EwRidge`]'s centred solve, where `C` is
//! a live Gram built row by row since it was empty, or a blocked Gram
//! (`gram_block_rows`) whose merges carried the bound of items 6 to 9. Not
//! a window's truncated Gram, a difference of two accumulators whose
//! rounding is relative to each of them rather than to their difference;
//! and not beside `session_shrink`'s twin, whose blend rewrites `C`
//! outside the row count. The raw normal equations -- `ridge_scale =
//! "sum"`, and fits without an intercept -- are formed another way (the
//! means' outer product added back, the intercept's column in the
//! system, under `ridge_scale` a penalty on the sum scale), and this
//! argument is not made for them: they take the long way.
//!
//! **A blocked Gram's bound.** A blocked Gram changes `C` only by merging
//! a block of `n` held rows ([`crate::EwCov::flush`]), and that rounding is
//! not relative to the result: the merge subtracts `ω δ̂δ̂ᵀ` for the residue
//! `δ̂` of the block's rounded mean, and the residue is relative to the
//! rows' distance from the row the mean was summed about, which a row of
//! small weight and large value, taken as that row, makes as large as it
//! likes against the block's own scatter. So the bound is not read from
//! `C`; it is kept beside it, `k` numbers that each merge updates in `O(k)`
//! and that no state file holds. Items 1 to 5 then hold with item 1's
//! premise replaced by item 6, and item 2's `(2n + 5) u T` by item 9's.
//! With gradual underflow -- a product or quotient that underflows is off
//! by at most `η = 2⁻¹⁰⁷⁴`, a sum by nothing (Higham, §2.2) -- and `ε =
//! γ_{4n+32}`, at most `1/1000`:
//!
//! 6. *What is kept.* `C = C* + E`, `C*` semidefinite, and `|E| ≤ M`
//!    entrywise, `M` a sum of terms `v vᵀ` with `v ≥ 0` -- nonnegative and
//!    semidefinite -- whose diagonal `e` is kept. Then `|c*_ij| ≤ √(c*_ii
//!    c*_jj)`, `c*_ii ≤ c_ii + e_i`, and for any positive diagonal `S`,
//!    `‖S⁻¹ E S⁻¹‖₂ ≤ ‖S⁻¹ M S⁻¹‖₂ ≤ tr(S⁻¹ M S⁻¹) = Σ_i e_i / s_i²` (a
//!    nonnegative matrix's norm bounds that of any it majorizes, and a
//!    semidefinite one's trace bounds its norm), on any principal
//!    submatrix too.
//! 7. *Where it starts.* Where `e` is not known -- an accumulator new, read
//!    from a file, its moments set, or stepped another way -- the next merge
//!    starts from `C* = 0` and `E = C`: `|c_ij| ≤ min(r_i, r_j) ≤ √(r_i
//!    r_j)`, `r_i = maxⱼ |c_ij|`, so `e_i = r_i`. That is exactly 0 for a
//!    new accumulator. For a loaded one it is the whole matrix, which the
//!    merges' `σ` then decay (item 8): the shortcut returns once the ridge
//!    outweighs what is left of it, after about `log₂(k t/ρ)` half-lives
//!    under a decay, and in practice not at all without one. It holds of any
//!    matrix, a state file edited by hand too.
//! 8. *A merge.* An entry is `c'_ij = fl(fl(σ c_ij) + α Σ_r d_ri d_rj) +
//!    fl(fl(b_i g_j) − fl(w_i δ̂_j))`, the sum in whatever order faer takes
//!    it: `σ = W_A/W'`, `α = 1/W'`, `β = W_A U/W'²` and `ω = U/W'` as
//!    computed, `d_r = p_r a_r` the block's rows (`a_r = x_r − m_B` and
//!    `p_r = √u_r`, each as computed), `g` the means' difference, `δ̂` the
//!    residue, `b_i = β g_i` and `w_i = ω δ̂_i`. The new target is
//!
//!    ```text
//!    C*' = σ C* + α Σ_r (d_r − p_r c)(d_r − p_r c)ᵀ + β g gᵀ
//!        = σ C* + α DᵀD − α P c cᵀ + β g gᵀ,    c = Σ_r p_r d_r / P,  P = Σ_r p_r²,
//!    ```
//!
//!    exact in the computed `σ, α, β, d_r, p_r, g`: semidefinite, every term
//!    of its diagonal nonnegative. A product of the sum takes at most `n + 3`
//!    roundings, `σ c_ij` `n + 2`, each correction four (Lemma 3.1, fused
//!    multiply-adds or not), so `E' = σ E + σ C∘Θ₁ + α Σ_r (d_r d_rᵀ)∘Θ_r +
//!    β ggᵀ∘Θ₂ − ω δ̂δ̂ᵀ∘Θ₃ + (α P c cᵀ − ω δ̂δ̂ᵀ) + N`, every `|Θ| ≤ ε` and
//!    `N` the underflows. With `h = |δ̂|` and `Q_i = α Σ_r d_ri²`, the parts'
//!    majorants have diagonals:
//!    - `σ e_i`, and `ε σ (c*_ii + e_i) ≤ ε c*'_ii + ε σ e_i` (item 6);
//!    - `ε Q_i`, `ε β g_i²` and `ε ω h_i²`;
//!    - the residue's. With `f = δ̂ − c`, `φ = |f|` and `κ = α P/ω`,
//!      `|κ − 1| ≤ γ_{n+3}`: `α P c cᵀ − ω δ̂δ̂ᵀ = ω ((κ − 1) c cᵀ − c fᵀ − f
//!      cᵀ − f fᵀ)`, at most `ω (ε (h + φ)(h + φ)ᵀ + (√ε h + φ/√ε)(√ε h +
//!      φ/√ε)ᵀ + 3 φφᵀ)` entrywise, a diagonal of at most `ω (4 ε h_i² + 3
//!      φ_i²/ε)`. Both `δ̂_i` and `c_i` are the mean `Σ_r u_r a_ri / U` to
//!      within `γ_{2n}` and `γ_8` of `ā_i = Σ_r u_r |a_ri| / U` (a sum and a
//!      quotient; `p_r d_ri = u_r a_ri (1 + θ₃)` and `P = Σ_r u_r (1 +
//!      θ₂)`), Cauchy–Schwarz gives `ā_i ≤ (1 + γ₅) √(Σ_r d_ri²/U)`, and
//!      `Σ_r d_ri²/U ≤ (1 + γ_{n+1}) Q_i/ω`; the underflows added, `φ_i ≤ ε
//!      √(Q_i/ω) + τ` with `τ = 2η (n/Û + 1 + √(n/Û))`, `Û` the computed
//!      `U`. So `3 ω φ_i²/ε ≤ 6 ε Q_i + 6 ω τ²/ε`;
//!    - the underflows'. `η` for `σ c_ij`; for the sum, the products and
//!      any scaling by `α` of one of their factors or of a partial sum, at
//!      most `n η (2 + α + R_i + R_j)`, `R_i = max_r |d_ri|`, `R_i² ≤
//!      Q_i/α`, under `n η ((2 + α) 1 1ᵀ + (1 + R)(1 + R)ᵀ)`; for the
//!      corrections, `η (1 + |g|)` and `η (1 + h)` against their partners,
//!      and `η` each. A diagonal of `η ((1 + |g_i|)² + (1 + h_i)² + 3 + n (4
//!      + α)) + 2 n η Q_i/α`, the first part's `1 + u` aside.
//!
//!    `Q_i` is not kept; the target bounds it: `Q_i = (α Σ_r (d_r − p_r
//!    c)(d_r − p_r c)ᵀ)_ii + κ ω c_i² ≤ c*'_ii + κ ω (h_i + φ_i)²`, so
//!    `Q_i ≤ 1.001 c*'_ii + 2.03 ω h_i² + 4.05 ω τ²`. With `α ≥ 2⁻⁹⁰⁰` the
//!    underflows' `2 n η/α` is under `ε`, so `Q_i`'s coefficient is at most
//!    `8 ε`; and `c*'_ii ≤ c'_ii + e'_i` (item 6). Then
//!
//!    ```text
//!    e'_i ≤ ( σ (1 + ε) e_i + 10 ε c'_ii⁺ + 22 ε ω h_i² + ε β g_i² + 7 ω τ²/ε
//!             + η ((1 + |g_i|)² + (1 + h_i)² + 3 + n (4 + α)) ) / (1 − 10 ε),
//!    ```
//!
//!    `c'_ii⁺ = max(c'_ii, 0)`. It is computed in about thirty roundings of
//!    nonnegative terms, taken in, with the `1 + u` above, by a factor `1 +
//!    2⁻⁴⁶`, and `2⁻¹⁰⁰⁰` is added for any of them that underflowed (each
//!    product is ordered so that nothing past `ω ≤ 1 + γ_{2n}` and the
//!    final factors multiplies it after). A merge outside the argument --
//!    rows weighted by their shares (`W'` below `2⁻¹⁰²⁴`), `α` under
//!    `2⁻⁹⁰⁰`, `ω` or `Û` not a normal double, `ε` past `1/1000`, or a bound
//!    that is not finite -- leaves `e` unknown, for the next merge to start
//!    again (item 7). A merge adds about `10 ε c'_ii` and decays the rest by
//!    `σ`: under a half-life of `L` rows `e_i` settles near `10 ε c_ii L/(n
//!    ln 2)`, and with no decay it grows with the merges, as item 1's slack
//!    grows with the rows.
//! 9. *The system.* Unscaled, `A`'s off-diagonal entries are `C`'s, so `G =
//!    E` and the diagonal's rounding: `‖G‖₂ ≤ Ê + u t/(1 − u)`, `Ê = Σ_i e_i`
//!    over the system's features. Standardized, `a_ij = c_ij (1 + θ)/(s_i
//!    s_j)`, `|θ| ≤ γ₅` with `ρ`'s rounding on the diagonal, `S =
//!    diag(√c_ii)`, so `‖G‖₂ ≤ Ê + γ₅ ‖S⁻¹ |C| S⁻¹‖₂ + u ρ ≤ Ê + γ₅ (k + 2
//!    Ê) + u ρ` (item 6), `Ê = Σ_i e_i / c_ii`, and `k ≤ T/(1 − γ₅)`. Either
//!    way, with `Ê` as computed (a sum, and a quotient each), `g₁ ≤ 1.05 (Ê +
//!    5 u T + u (2t + δ))`, and items 3 to 5 go through with `ĝ = Ê + 5 u T
//!    + (3k + 1) u (T + kδ) + u (2t + δ)`.

use super::SpdFactor;

/// The unit roundoff of a double, `2⁻⁵³`.
const U: f64 = f64::EPSILON / 2.0;

/// The iterations Hager's method takes at most (LAPACK's `ITMAX`).
const ITMAX: usize = 5;

/// The orders whose working vectors fit on the stack: a solve of up to 32
/// columns judges its condition without allocating, as `ewridge`'s per-row
/// cadence needs (`online-bench`'s allocation counts).
const STACK: usize = 32;

impl SpdFactor {
    /// `‖H⁻¹‖₁`, estimated, for `H = D⁻¹ (A + δI) D⁻¹` the equilibrated
    /// matrix of the system this factor holds, `A` row-major `k*k` as given
    /// to the factorization and `δ` its jitter ([`Self::jitter`], 0 for a
    /// factor that needed none), with `D = diag(√(a_ii + δ))`. A lower bound,
    /// usually exact (the module docs). Infinite where a product overflows
    /// or is not a number, which only a matrix as good as singular gives:
    /// `H`'s entries are at most 1 in size.
    pub fn equilibrated_inverse_norm(&self, a: &[f64]) -> f64 {
        self.with_work(a, |f, d, work| {
            norm1_estimate(d.len(), work, |x| f.equilibrated_solve(d, x))
        })
    }

    /// Run `f` with `D`'s diagonal, `√(a_ii + δ)` (`δ` the jitter, 0 where
    /// none was needed: the same bits as `√a_ii`), and a working space of `2k`,
    /// on the stack up to [`STACK`] columns; infinite where the factor's
    /// columns cannot be read as slices.
    fn with_work(&self, a: &[f64], f: impl FnOnce(&Self, &[f64], &mut [f64]) -> f64) -> f64 {
        let k = self.order();
        debug_assert_eq!(a.len(), k * k);
        if (0..k).any(|j| self.l.col(j).try_as_col_major().is_none()) {
            return f64::INFINITY;
        }
        let mut stack = [0.0f64; 3 * STACK];
        let mut heap = Vec::new();
        let buf: &mut [f64] = if k <= STACK {
            &mut stack[..3 * k]
        } else {
            heap.resize(3 * k, 0.0);
            &mut heap
        };
        let (d, work) = buf.split_at_mut(k);
        for (i, di) in d.iter_mut().enumerate() {
            *di = (a[i * k + i] + self.jitter).sqrt();
        }
        f(self, d, work)
    }

    /// Column `j` of `L` from its diagonal down.
    fn below(&self, j: usize) -> &[f64] {
        &self
            .l
            .col(j)
            .try_as_col_major()
            .expect("checked in `with_work`")
            .as_slice()[j..]
    }

    /// `x ← H⁻¹ x = D A⁻¹ D x`, from `L`: `D x`, then `L y = D x` forward by
    /// columns and `Lᵀ z = y` backward by dot products down each column,
    /// then `D z`. Nothing overflows on the way for any `A` a double holds:
    /// `L⁻¹ D = (D⁻¹ L)⁻¹` is the inverse of `H`'s own factor, so `y` is in
    /// `H`'s units, and `z = D⁻¹ H⁻¹ x` is at most `1.6e162` times them
    /// (`√a_ii ≥ 2⁻⁵³⁷`, the smallest subnormal's root).
    fn equilibrated_solve(&self, d: &[f64], x: &mut [f64]) {
        let k = x.len();
        for (v, di) in x.iter_mut().zip(d) {
            *v *= di;
        }
        for j in 0..k {
            let c = self.below(j);
            x[j] /= c[0];
            let xj = x[j];
            for (xi, &lij) in x[j + 1..].iter_mut().zip(&c[1..]) {
                *xi -= lij * xj;
            }
        }
        for j in (0..k).rev() {
            let c = self.below(j);
            let dot: f64 = x[j + 1..]
                .iter()
                .zip(&c[1..])
                .map(|(xi, lij)| xi * lij)
                .sum();
            x[j] = (x[j] - dot) / c[0];
        }
        for (v, di) in x.iter_mut().zip(d) {
            *v *= di;
        }
    }

    /// An upper bound on `‖H⁻¹‖₁` from the comparison matrix of `H`'s
    /// factor `R = D⁻¹ L` (the module docs, *The screen*). With `M(R) =
    /// D⁻¹ M(L)`, `M(R)⁻¹ 1 = M(L)⁻¹ d`, one forward substitution; and
    /// `M(R)⁻ᵀ 1 = D w` with `M(L)ᵀ w = 1`, one backward. Every term is
    /// nonnegative, so nothing cancels; infinite or NaN where a term
    /// overflows, which the caller reads as no bound. `z` and `w` are `k`
    /// long each.
    fn comparison_bound(&self, d: &[f64], z: &mut [f64], w: &mut [f64]) -> f64 {
        let k = d.len();
        z.copy_from_slice(d);
        for j in 0..k {
            let c = self.below(j);
            z[j] /= c[0];
            let zj = z[j];
            for (zi, &lij) in z[j + 1..].iter_mut().zip(&c[1..]) {
                *zi += lij.abs() * zj;
            }
        }
        for j in (0..k).rev() {
            let c = self.below(j);
            let dot: f64 = w[j + 1..]
                .iter()
                .zip(&c[1..])
                .map(|(wi, lij)| wi * lij.abs())
                .sum();
            w[j] = (1.0 + dot) / c[0];
        }
        // The largest entry, a NaN kept: `f64::max` would drop it, and a
        // NaN in `A`'s diagonal would read as a bound of 0.
        let largest = |m: f64, v: f64| if v > m || v.is_nan() { v } else { m };
        let rows = z.iter().fold(0.0f64, |m, &v| largest(m, v));
        let cols = w
            .iter()
            .zip(d)
            .fold(0.0f64, |m, (&v, &di)| largest(m, v * di));
        rows * cols
    }

    /// Whether this factor's system is past the condition at which its
    /// solve's forward-error bound reaches 1, so that no digit of the
    /// solution is guaranteed: `3 k γ_{3k+1} ‖H⁻¹‖₁ ≥ 1` (the module docs
    /// derive it). `a` is the matrix given to the factorization, row-major;
    /// a factor that needed jitter is judged as the `A + δI` it holds (the
    /// module docs). The comparison bound decides where it is inside the
    /// limit; the estimate decides the rest.
    pub fn carries_no_digit(&self, a: &[f64]) -> bool {
        let limit = digit_limit(self.order());
        let norm = self.judged_norm(a, limit);
        // An estimate that is NaN is refused, as one past the limit is.
        norm.is_nan() || norm >= limit
    }

    /// The `‖H⁻¹‖₁` a decision at `limit` is read from: the comparison
    /// bound where it is below `limit`, which then settles it, else the
    /// estimate.
    fn judged_norm(&self, a: &[f64], limit: f64) -> f64 {
        let k = self.order();
        self.with_work(a, |f, d, work| {
            let (z, w) = work.split_at_mut(k);
            let bound = f.comparison_bound(d, z, w);
            if bound < limit {
                return bound;
            }
            norm1_estimate(k, work, |x| f.equilibrated_solve(d, x))
        })
    }
}

/// What a solve knows of its matrix besides the entries: that it is
/// `C + ridge·I`, or its standardized form, with `C` the centred co-moments
/// and their rounding bounded by what is `behind` them (the module docs,
/// *The ridge's shortcut*, say which systems qualify and why).
#[derive(Debug, Clone, Copy)]
pub(crate) struct OverMoments {
    /// The ridge added to the diagonal.
    pub(crate) ridge: f64,
    /// What bounds `C`'s rounding.
    pub(crate) behind: Behind,
}

/// What bounds the co-moments' rounding in a ridge's system.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Behind {
    /// A live Gram that this many rows of positive weight built one by one
    /// since it was empty ([`crate::EwCov::rows_learned`]; the module docs,
    /// item 1).
    Rows(u64),
    /// A blocked Gram's kept bound, read over the system's features: `Ê =
    /// Σ e_i`, or `Σ e_i / c_ii` standardized (the module docs, items 6 to
    /// 9).
    Merged(f64),
}

impl OverMoments {
    /// A ridge over a live Gram of `rows` rows of positive weight.
    pub(crate) fn live(ridge: f64, rows: u64) -> Self {
        Self {
            ridge,
            behind: Behind::Rows(rows),
        }
    }

    /// A ridge over a blocked Gram whose kept bound reads `e_hat` over the
    /// system.
    pub(crate) fn merged(ridge: f64, e_hat: f64) -> Self {
        Self {
            ridge,
            behind: Behind::Merged(e_hat),
        }
    }
}

/// The largest `t = max a_ii` the shortcut reads, `2⁴⁰⁰`, and the smallest,
/// `2⁻⁴⁰⁰` (the module docs: inside them nothing the estimate forms
/// overflows, and an underflow's error is far under a rounding).
const TOP_MAX: f64 = f64::from_bits((1023 + 400) << 52);
const TOP_MIN: f64 = f64::from_bits((1023 - 400) << 52);

impl SpdFactor {
    /// [`Self::carries_no_digit`], taking the shortcut where `over` says the
    /// system is a ridge over semidefinite co-moments and the ridge alone
    /// holds it inside the limit: `k` reads of the diagonal, against the
    /// bound's and the estimate's `k²` passes. The answer is always the
    /// long way's (the module docs prove it).
    pub(crate) fn carries_no_digit_over(&self, a: &[f64], over: Option<OverMoments>) -> bool {
        if over.is_some_and(|o| self.ridge_settles(a, o)) {
            count_shortcut();
            return false;
        }
        self.carries_no_digit(a)
    }

    /// Whether the ridge alone puts this factor's system inside the limit
    /// (the module docs, *The ridge's shortcut*, items 1 to 5, and 9 for a
    /// blocked Gram): `(2n + 6) u` (for a live Gram) and `(3k + 1) u` at
    /// most `1/100`, `t = max a_ii` within `2^±400`, the slack `ĝ` at most
    /// `(ρ + δ)/2.5`, and `2 √k (t + δ)/(ρ + δ)` below the limit. `false`
    /// says only that the long way decides.
    pub(crate) fn ridge_settles(&self, a: &[f64], over: OverMoments) -> bool {
        let k = self.order();
        debug_assert_eq!(a.len(), k * k);
        // The long way reads an unreadable factor as past the limit.
        if (0..k).any(|j| self.l.col(j).try_as_col_major().is_none()) {
            return false;
        }
        ridge_holds((0..k).map(|i| a[i * k + i]), k, over, self.jitter)
    }
}

/// [`SpdFactor::ridge_settles`] from the system's diagonal `diag` (its `k`
/// entries as factorized, before the jitter `delta`): the module docs,
/// *The ridge's shortcut*, item 5, and the range it needs.
pub(crate) fn ridge_holds(
    diag: impl Iterator<Item = f64>,
    k: usize,
    over: OverMoments,
    delta: f64,
) -> bool {
    #[cfg(test)]
    if SHORTCUT_OFF.with(std::cell::Cell::get) {
        return false;
    }
    let order = k as f64;
    if k == 0 || (3.0 * order + 1.0) * U > 0.01 {
        return false;
    }
    match over.behind {
        Behind::Rows(rows) if (2.0 * rows as f64 + 6.0) * U > 0.01 => return false,
        // A bound that is not a finite nonnegative number is none.
        Behind::Merged(e_hat) if !(e_hat >= 0.0 && e_hat.is_finite()) => return false,
        _ => {}
    }
    let (mut trace, mut top) = (0.0f64, 0.0f64);
    for v in diag {
        trace += v;
        top = top.max(v);
    }
    let rho = over.ridge;
    // A NaN or an infinity in the diagonal leaves the trace one.
    if !(rho > 0.0 && trace.is_finite() && (TOP_MIN..=TOP_MAX).contains(&top)) {
        return false;
    }
    let shift = rho + delta;
    // Item 2's bound on the co-moments' rounding, or item 9's.
    let history = match over.behind {
        Behind::Rows(rows) => (2.0 * rows as f64 + 5.0) * U * trace,
        Behind::Merged(e_hat) => e_hat + 5.0 * U * trace,
    };
    let slack =
        history + (3.0 * order + 1.0) * U * (trace + order * delta) + U * (2.0 * top + delta);
    // Every term is finite here, so `>` refuses what `≤` would not pass.
    if 1.25 * slack > 0.5 * shift {
        return false;
    }
    2.0 * order.sqrt() * (top + delta) / shift < digit_limit(k)
}

#[cfg(test)]
pub(crate) use switches::{SHORTCUT_OFF, SHORTCUTS};

#[cfg(test)]
mod switches {
    thread_local! {
        /// The solves on this thread the ridge's shortcut decided: what a test
        /// reads to see that it ran, since its answer is the long way's.
        pub(crate) static SHORTCUTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        /// Set, the shortcut never fires on this thread: what a test sets to
        /// run the long way on the same stream and compare.
        pub(crate) static SHORTCUT_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
}

/// Count a solve the shortcut decided in [`SHORTCUTS`]; nothing outside
/// tests.
#[inline]
fn count_shortcut() {
    #[cfg(test)]
    SHORTCUTS.with(|c| c.set(c.get() + 1));
}

/// The `‖H⁻¹‖₁` at which an order-`k` solve keeps no digit: `1 / (3 k
/// γ_{3k+1})`, `γ_m = m u / (1 − m u)` (the module docs derive it). 0, so
/// that every system is refused, at an order whose `γ` is no bound, which no
/// matrix a machine holds reaches.
fn digit_limit(k: usize) -> f64 {
    let m = (3 * k + 1) as f64 * U;
    if m >= 1.0 {
        return 0.0;
    }
    (1.0 - m) / (3.0 * k as f64 * m)
}

/// `‖B‖₁` of a symmetric `k × k` matrix `B`, estimated from products
/// `x ← B x` (`apply`), by Hager's method as Higham refined it: LAPACK's
/// `dlacn2` with `Bᵀ = B`. The first product is with the vector of `1/k`;
/// then, in turn, with the signs of the last product, and with the unit
/// vector at the largest entry of that, until the signs repeat, the
/// estimate stops growing, the largest entry stays where it was, or five
/// iterations have run. Last, a vector of alternating signs and growing
/// size, `(−1)ⁱ (1 + i/(k − 1))`, whose product's 1-norm times `2/(3k)` is
/// taken where it is larger: it catches the matrices that fool the
/// iteration (Higham, §15.3). Every candidate is `‖B v‖₁ / ‖v‖₁` for some
/// `v`, so the estimate is at most `‖B‖₁`; it is the largest of them, where
/// `dlacn2` keeps the last. Infinite where a product holds an infinity or a
/// NaN. `work` is `2k` long: the product and the signs.
fn norm1_estimate(k: usize, work: &mut [f64], mut apply: impl FnMut(&mut [f64])) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let norm1 = |x: &[f64]| -> f64 {
        let s: f64 = x.iter().map(|v| v.abs()).sum();
        if s.is_finite() { s } else { f64::INFINITY }
    };
    let (x, signs) = work.split_at_mut(k);
    x.fill(1.0 / k as f64);
    apply(x);
    let mut est = norm1(x);
    if k == 1 || est.is_infinite() {
        return est;
    }
    for (s, &v) in signs.iter_mut().zip(x.iter()) {
        *s = sign(v);
    }
    x.copy_from_slice(signs);
    apply(x);
    let mut j = first_largest(x);
    for _ in 2..=ITMAX {
        x.fill(0.0);
        x[j] = 1.0;
        apply(x);
        let last = est;
        let now = norm1(x);
        est = est.max(now);
        if est.is_infinite() {
            return est;
        }
        let repeated = x.iter().zip(signs.iter()).all(|(&v, &s)| sign(v) == s);
        if repeated || now <= last {
            break;
        }
        for (s, &v) in signs.iter_mut().zip(x.iter()) {
            *s = sign(v);
        }
        x.copy_from_slice(signs);
        apply(x);
        let previous = j;
        j = first_largest(x);
        if x[previous].abs() == x[j].abs() {
            break;
        }
    }
    alternating(x);
    apply(x);
    est.max(2.0 * norm1(x) / (3 * k) as f64)
}

/// `+1` for a number at or above 0, `−1` below it and for a NaN: the sign
/// vector's entries, as `dlacn2` takes them (0 counts as positive).
fn sign(v: f64) -> f64 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// The index of the entry largest in absolute value, the first of several
/// equal ones, as `idamax` gives it.
fn first_largest(x: &[f64]) -> usize {
    let mut j = 0;
    for (i, v) in x.iter().enumerate() {
        if v.abs() > x[j].abs() {
            j = i;
        }
    }
    j
}

/// `x_i = (−1)ⁱ (1 + i/(k − 1))`, from `+1` at the first entry to `±2` at
/// the last, for `k = x.len() ≥ 2`: the vector `dlacn2` tries last.
fn alternating(x: &mut [f64]) {
    let step = 1.0 / (x.len() - 1) as f64;
    for (i, v) in x.iter_mut().enumerate() {
        let size = 1.0 + i as f64 * step;
        *v = if i % 2 == 0 { size } else { -size };
    }
}

#[cfg(test)]
#[path = "condition_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ridge_tests.rs"]
mod ridge_tests;

#[cfg(test)]
#[path = "blocked_tests.rs"]
mod blocked_tests;
