# Answers to the open points in `docs/PLAN.md` §11a, *Preparing E54–E64 for implementation*

Written 2026-09-05 against the papers themselves. Each answer says whether
it comes from a full reading of the source (**verified**) or is a
convention / memory item the implementer should still check (**check**).
Nothing below changes a decision already taken in §11a; it supplies the
numbers and formulae the block left to be confirmed.

---

## Task 54 — `corrchange`

### The Bartlett bandwidth `⌊log T⌋`: which log?

**Verified (Wied, Krämer & Dehling 2012, Appendix A.1).** The paper writes
the bandwidth as `γ_T = [log T]` with no base stated, using the Bartlett
kernel `k(x) = 1 − |x|` for `|x| ≤ 1`, in the de Jong–Davidson (2000) kernel
estimator of the long-run covariance of the five-moment vector
`U_t = (X_t², Y_t², X_t, Y_t, X_tY_t)`. **Check:** read it as the natural
logarithm — at `T = 500` that is `γ_T = 6`, at `T = 2000` it is `7`; a
base-10 reading would give `2` and `3`, too short to be a long-run variance
bandwidth. The implementation should expose the bandwidth as a parameter
with `⌊ln T⌋` as its default so the choice is visible.

### `D̂` exactly

**Verified.** `D̂ = (F̂₁D̂₃,₁ + F̂₂D̂₃,₂ + F̂₃D̂₃,₃)^{−1/2}` where
`D̂₃ = (−½ σ̂_xy σ̂_y σ̂_x^{−3}, −½ σ̂_xy σ̂_x σ̂_y^{−3}, 1/(σ̂_xσ̂_y))` is
the gradient of `ρ = σ_xy/(σ_xσ_y)` in `(σ_x², σ_y², σ_xy)`,
`Ê = D₂ D̂₁ D₂′` maps the five raw moments to those three
(`D₂` rows: `(1, 0, −2μ_x, 0, 0)`, `(0, 1, 0, −2μ_y, 0)`,
`(0, 0, −μ_y, −μ_x, 1)`), `F̂ = D̂₃ Ê`, and `D̂₁ = Σ_t Σ_u k((t−u)/γ_T) V_t V_u′`
with `V_t = U_t^{***}/√T` the centred moment vector (each entry minus its
sample mean over the span). So `D̂` is the delta-method long-run standard
deviation of `ρ̂` and the statistic is `(j/√T)|ρ̂_j − ρ̂_T| / D̂`.

### The closed-sample null and its quantiles

**Verified.** Under `H₀` and assumptions (A1)–(A4) (finite `(4+δ)`-th
moments; `L₂`-near-epoch dependence, which admits GARCH; means and
variances constant, or (A5) changing proportionally so that the correlation
stays constant), `sup_{0≤z≤1}|K_T(z)| →_d sup_{0≤z≤1}|B(z)|`, a Brownian
bridge. The quantiles of `sup|B|` are the Kolmogorov distribution,
`P(sup|B| ≤ x) = 1 − 2 Σ_{k≥1} (−1)^{k−1} exp(−2k²x²)`; the implementer
should compute the critical value from this series rather than pin a
constant — at 5% it is 1.3581, at 1% 1.6276, at 10% 1.2239 (**check:** these
three are the standard tabulated values; the series above reproduces them).

### Size and power to pin against (their Tables 1–2, 5000 replications, 5%)

**Verified.** Size (empirical rejection under `H₀`), i.i.d. bivariate `t₅`
innovations, rows `T = 200 / 500 / 1000 / 2000`:

| ρ | −0.9 | −0.5 | 0 | 0.5 | 0.9 |
|---|---|---|---|---|---|
| 200 | .144 | .054 | .039 | .053 | .142 |
| 500 | .064 | .040 | .035 | .041 | .064 |
| 1000 | .048 | .038 | .034 | .039 | .049 |
| 2000 | .043 | .038 | .036 | .038 | .043 |

With AR(1) serial dependence `φ = 0.1` the rows are .150/.052/.038/.053/.144,
.070/.049/.037/.044/.064, .055/.046/.039/.044/.053, .046/.043/.043/.045/.051.
So the test **over-rejects at |ρ| = 0.9 for T ≤ 500**; a size test in the
gate should use |ρ| ≤ 0.5.

Size-adjusted power (unit variances):

| alternative | T=200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| (1) 0.5 → 0.7 at T/2 | .309 | .587 | .830 | .967 |
| (2) 0.5 → 0.7 at T/4 | .255 | .488 | .733 | .928 |
| (3) 0.5 → −0.5 at T/2 | .953 | .996 | .998 | 1 |
| (4) 0.5 → −0.5 at T/4 | .880 | .989 | .998 | .999 |
| (5) 0.5 → 0.7 on (T/4, 3T/4] → 0.5 | .101 | .207 | .422 | .750 |

Alternative (1) at `T = 500` and `1000` is the pair §11a names.

### The sequential (monitoring) form

**Not verified here.** The closed-sample statistic centres on `ρ̂_T`, the
full-sample correlation, and the Brownian-bridge quantiles are exact for
that. The *monitoring* version — training span, then a detector against a
boundary function — is Wied & Galeano, *Monitoring correlation change in a
sequence of random variables*, Journal of Statistical Planning and
Inference 143(1), 2013, 186–196 (paywalled; only its abstract was read). Its
boundary and critical values must be taken from that paper; the §11a
decision to do so is right, and no constant is offered here in its place.
Two things are certain from the 2012 paper alone: the multivariate
extension is "pairwise comparisons, rejecting if the maximum of the
statistics is too large", with the multiple-testing correction left open;
and the finite-fourth-moment requirement is necessary, not technical (the
limit changes without it).

### The permutation null for `kind = "window"`

**Confirmed.** A sign flip of a whole row `x_t → −x_t` leaves `x_t x_t′`
unchanged, so every window correlation matrix and the statistic are
invariant; that null has zero spread. Permuting rows between the two
pooled windows is the exchangeable null for "same distribution", and
permuting *blocks* of consecutive rows is the standard way to keep serial
dependence from making it too liberal. The §11a correction stands.

---

## Task 51 — `po.corr`

### Higham's published examples

**Verified (Higham 2002, §2 and §4).**

- `A = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]`, eigenvalues `1 + √2, 1, 1 − √2`.
  Nearest correlation matrix in the unweighted Frobenius norm:
  `X = [[1.0000, 0.7607, 0.1573], [0.7607, 1.0000, 0.7607], [0.1573, 0.7607,
  1.0000]]`, `‖A − X‖_F = 0.5278`; `X` is singular with null vector
  `q = [−0.4814, 0.7324, −0.4814]′`. (The obvious candidate `ee′` is *not*
  the answer: `‖A − ee′‖_F = √2`.)
- `A = tridiag(−1, 2, −1)` of order 4: `X = [[1.0000, −0.8084, 0.1916,
  0.1068], [−0.8084, 1.0000, −0.6562, 0.1916], [0.1916, −0.6562, 1.0000,
  −0.8084], [0.1068, 0.1916, −0.8084, 1.0000]]`, `‖A − X‖_F = 2.13`, rank 3;
  Algorithm 3.3 converges in **19 iterations at `tol = 10⁻⁸`**, the relative
  differences shrinking by a factor ≈ 3 per iteration (linear convergence,
  as expected of alternating projections).
- The algorithm exactly: `ΔS₀ = 0, Y₀ = A`; for `k = 1, 2, …`: `R_k = Y_{k−1}
  − ΔS_{k−1}` (Dykstra's correction), `X_k = P_S(R_k)`, `ΔS_k = X_k − R_k`,
  `Y_k = P_U(X_k)`. With diagonal `W`, `P_U` sets the diagonal to 1 and
  `P_S(A) = W^{−1/2}((W^{1/2}AW^{1/2})_+)W^{−1/2}` with `(·)_+` the
  eigenvalue clip at zero. Convergence test (4.1): the maximum of
  `‖X_k − X_{k−1}‖_∞/‖X_k‖_∞`, `‖Y_k − Y_{k−1}‖_∞/‖Y_k‖_∞`,
  `‖Y_k − X_k‖_∞/‖Y_k‖_∞` below `tol`.
- Two bounds worth testing: if `A` is diagonal the answer is `I`; if `A` is
  PSD with diagonal ≤ 1 the answer is `A` with its diagonal set to 1. And
  Theorem 2.5: if `A` has unit diagonal and `t` nonpositive eigenvalues, the
  answer has at least `t` zero eigenvalues (the 4×4 example: one negative
  eigenvalue, rank 3).

### `shrink`: the Ledoit–Wolf intensity needs the rows

**Verified (Ledoit & Wolf 2004, Appendices A–B).** Target `F`: `f_ii = s_ii`,
`f_ij = r̄√(s_ii s_jj)` with `r̄ = 2/((N−1)N) Σ_{i<j} r_ij`. Intensity
`δ̂* = max{0, min{κ̂/T, 1}}`, `κ̂ = (π̂ − ρ̂)/γ̂`, with

```
π̂   = Σ_ij π̂_ij,      π̂_ij = (1/T) Σ_t ((y_it − ȳ_i)(y_jt − ȳ_j) − s_ij)²
ρ̂   = Σ_i π̂_ii + Σ_{i≠j} (r̄/2) ( √(s_jj/s_ii) ϑ̂_ii,ij + √(s_ii/s_jj) ϑ̂_jj,ij )
ϑ̂_ii,ij = (1/T) Σ_t ((y_it − ȳ_i)² − s_ii)((y_it − ȳ_i)(y_jt − ȳ_j) − s_ij)
γ̂   = Σ_ij (f_ij − s_ij)²
```

`π̂` and `ϑ̂` are fourth-moment sums over the rows, so §11a's change from
`T=` to `X=` is necessary: the intensity cannot be formed from the
correlation matrix and `T` alone. `alpha = 0` returns `S`, `alpha = 1`
returns `F`; the shrunk matrix is positive definite whenever `F` is.

### `mp_edge`

**Verified (Laloux, Cizeau, Bouchaud & Potters 1999).** For `Q = T/N ≥ 1`
and unit variance, `λ_max/min = σ²(1 + 1/Q ± 2√(1/Q))`, i.e. `σ²(1 ±
1/√Q)²`; density `ρ(λ) = (Q/2πσ²) √((λ_max − λ)(λ − λ_min))/λ` on
`[λ_min, λ_max]`. Their S&P example: `N = 406, T = 1309, Q = 3.22`; the
top eigenvalue ≈ 25× `λ_max`; 94% of the spectrum fits the bulk with
`σ² = 0.74`. A test can check `Q = 1 → (0, 4σ²)` and that a random Wishart
spectrum lies inside the edges up to a finite-`n` margin.

### `epps_invert`: the exact form of Tóth–Kertész eq. 12

**Verified.** With `f^{A/B}(x)`, `f^{A/A}(x)`, `f^{B/B}(x)` the lagged
cross- and auto-correlation decay functions at the short scale `Δt₀`
(normalised so `f(0) = 1`) and `L = Δt/Δt₀`,

```
ρ_Δt = [ Σ_{x=−(L−1)}^{L−1} (L − |x|) f^{A/B}(x) ]
       · [ Σ_x (L − |x|) f^{A/A}(x) ]^{−1/2} · [ Σ_x (L − |x|) f^{B/B}(x) ]^{−1/2} · ρ_Δt₀
```

— *triangular* weights `(L − |x|)` on both the numerator and the two
denominators, not a flat sum, and both orientations of the cross term
(`x < 0` and `x > 0`). In the co-moment form of task 48, with `C_ℓ[a,b]` the
lag-`ℓ` co-moment: numerator `Σ_{ℓ=0}^{L−1} (L − ℓ)·(C_ℓ[a,b] + [ℓ>0] C_ℓ[b,a])`,
denominators `Σ_{ℓ} (L − ℓ)·(C_ℓ[a,a] + [ℓ>0] C_ℓ[a,a])` and likewise for
`b`, the ratio being the correlation at scale `L`. §11a's flat sum is the
`L → ∞` limit and should carry `L` (the target scale, in rows) as a
parameter with the triangular weights. Their empirical setting: `Δt₀ =
120 s`, decays truncated at the first zero crossing, mean fit error ≈ 2%.

### `fisher_se`

`1/√(n − 3)` for `z = atanh(ρ)` and the delta-method `(1 − ρ²)/√(n − 3)` for
`ρ` are standard (**verified** as stated in the Fisher-transformation
references). The AR(1) inflation `√((1 + φ_aφ_b)/(1 − φ_aφ_b))` is Bartlett's
formula specialised to two independent AR(1) series — **check:** it is
quoted from memory of the general form `Var(r) ≈ (1/n) Σ_k ρ_a(k)ρ_b(k)`
under zero true cross-correlation and is invalid under ARCH-type
innovations; the docstring should say both.

---

## Task 50 — `rcov`

### The pre-averaging constants and the finite-sample forms

**Verified (Christensen, Kinnebrock & Podolskij 2010, §2.2–2.3).** For
`g(x) = min(x, 1 − x)`: `ψ₁ = 1`, `ψ₂ = 1/12`, `Φ₁₁ = 1/6`, `Φ₁₂ = 1/96`,
`Φ₂₂ = 151/80640`. The finite-sample replacements the paper prescribes
("to avoid biases in small samples"):

```
ψ₁^{k} = k · Σ_{i=1}^{k} (g(i/k) − g((i−1)/k))²
ψ₂^{k} = (1/k) · Σ_{i=1}^{k−1} g(i/k)²
φ₁^{k}(j) = Σ_{i=j+1}^{k−1} (g((i−1)/k) − g(i/k)) (g((i−j−1)/k) − g((i−j)/k))
φ₂^{k}(j) = Σ_{i=j+1}^{k−1} g(i/k) g((i−j)/k)
Φ₁₁^{k} = k ( Σ_{j=0}^{k−1} φ₁^{k}(j)² − ½ φ₁^{k}(0)² ),   Φ₁₂^{k} = (1/k)( Σ_j φ₁^{k}(j)φ₂^{k}(j) − ½ φ₁^{k}(0)φ₂^{k}(0) )
```

Pre-averaged returns `Ȳ_i = Σ_{j=1}^{k_n−1} g(j/k_n) Δ_{i+j}Y`, `k_n = ⌊θ√n⌋`
("balanced", the `n^{−1/4}` rate); `MRC = n/(n − k_n + 2) · 1/(ψ₂ k_n)
Σ_{i=0}^{n−k_n+1} Ȳ_i Ȳ_i′`. The balanced form needs a bias correction
that can make it non-PSD in finite samples; "increasing the pre-averaging
window length slightly" gives a PSD estimator at a slower rate. **Check:**
the exact exponent of that longer window and the bias-correction term are
in the paper's §3, which was not read for this note — take both from
there rather than from §11a's `δ = 0.1`.

### The kernel's bandwidth rule and end jitter

**Verified (Barndorff-Nielsen, Hansen, Lunde & Shephard 2009, §2).**
`H* = c* ξ^{4/5} n^{3/5}` with `c* = (k″(0)²/k^{0,0})^{1/5} =
((12)²/0.269)^{1/5} = 3.5134` for Parzen, `ξ² = ω²/√(T∫σ⁴)`; in practice
`ξ̂² = ω̂²/IV̂` with `IV̂ = RV_sparse` (a subsampled 20-minute realised
variance, averaged over start offsets) and `ω̂² = mean over q offsets of
RV_dense^{(i)}/(2n^{(i)})`, computed on every `q`-th observation with `q`
chosen so successive observations are ≈ 2 minutes apart (their data:
`q ≈ 25` for trades, `≈ 70` for mid-quotes), because `E(U_jU_{j+q}) = 0`
fails at `q = 1` for quote data. The estimate is deliberately upward-biased
(a conservative, larger `H`). End jitter with `m = 2`: `X₀ = ½(X_{τ₀} +
X_{τ₁})`, `X_j = X_{τ_{j+1}}` for `j = 1..n−1`, `X_n = ½(X_{τ_{N−1}} +
X_{τ_N})`; `m = 1` is mean-square optimal in practice and `m = 1..4` moved
their estimates by < 0.5%. Parzen `k(x) = 1 − 6x² + 6x³ (0 ≤ x ≤ ½), 2(1 −
x)³ (½ ≤ x ≤ 1), 0 (x > 1)`; the kernel needs only `H` autocovariances;
Bartlett's kernel is *not* consistent here. (BNHLS 2011, Table 1: Parzen's
efficiency constant 0.97 against QS 0.93; QS would need all `n`
autocovariances.)

### The refresh-time worked example

**Verified (BNHLS 2011, Figure 1 and §2.1).** Three assets with `n = 8, 9,
10` observations give `N = 7` refresh times and a retained fraction `p =
dN/Σn = 21/27 ≈ 0.78`. The figure gives the tick times only graphically, so
§11a's plan to construct an example with those counts and assert `N ≤ min
nᵢ` and `p = N·m/Σnᵢ` against a longhand loop is the right test. Under
independent Poisson arrivals the refresh sample size falls like `log d`
(their footnote 3).

---

## Task 46 — `deco`

**Verified (Engle & Kelly 2012).** `R_t = (1 − ρ_t)I + ρ_t J`; `R_t^{−1} =
(1/(1−ρ_t))[I − ρ_t/(1 + (n−1)ρ_t) J]`; `det R_t = (1 − ρ_t)^{n−1}(1 +
(n−1)ρ_t)`; PD iff `ρ_t ∈ (−1/(n−1), 1)`. LDECO (their eq. 20–21):
`u_t = Σ_{i≠j} r_{i,t}r_{j,t} / ((n−1) Σ_i r²_{i,t})`, `ρ_{t+1} = ω + α u_t +
β ρ_t`, `u_t ∈ (−1/(n−1), 1)` a.s. (Lemma 2.3). Two things §11a should
know: (i) in the paper `ω` is a free parameter; writing it as `(1 − α −
β)·ρ̄` (targeting) is a reparameterisation the paper applies to the
DECO-DCC `Q` process (eq. 5), not to LDECO — harmless, but it is a choice,
not the paper's; (ii) the paper notes `u_t` is a ratio and *downward
biased*, so the LDECO recursion "can be stationary for α + β slightly in
excess of 1" and they enforce the `(−1/(n−1), 1)` bounds numerically rather
than by constraining `α + β` — the `α + β < 1` refusal is stricter than the
paper and is fine as a library rule. The cross-sectional-variance
alternative `u^{var}_t = 1 − (1/(n−1)) Σ_i (r_{i,t} − r̄_t)²` is unbiased and
lower-variance under normality but can violate the lower bound and is less
robust to fat tails; below ρ ≈ 0.4 under `t₄`, `u_t` is the more efficient.
The log-likelihood (eq. 10) is
`−½ Σ_t [ n log 2π + log((1−ρ_t)^{n−1}(1+(n−1)ρ_t)) + (1/(1−ρ_t))( Σ_i r²_{i,t}
− ρ_t/(1+(n−1)ρ_t) (Σ_i r_{i,t})² ) ]`, matching §11a.

---

## Task 53 — `hmm`

**Confirmed.** The ratio `p_k(t−1)·p_l(t)/p_k(t−1)` is `p_l(t)` and carries
no information about `Π`; the filtered joint of consecutive states
`ξ_{kl} ∝ p_k(t−1)·Π_{kl}·f_l(x_t)` is the Baum–Welch quantity whose
(decayed) accumulation identifies `Π`, and a Dirichlet pseudo-count per
row keeps every row a distribution. The Hamilton prediction/update pair as
written (`p̃_l = Σ_k p_k Π_kl` before the row; `p_l ∝ p̃_l f_l` after) is the
standard filter; with `Π` uniform it reduces to the Gaussian classifier's
posterior, which is the reduction test.

---

## Task 55 — `bocpd`

**Verified (Adams & MacKay 2007).** The recursion, the hazard `H(τ) =
P_gap(g = τ)/Σ_{t≥τ} P_gap(g = t)` (constant `1/λ` for a geometric gap
prior), the conjugate-exponential sufficient statistics `ν^{(r+1)}_{t+1} =
ν^{(r)}_t + 1`, `χ^{(r+1)}_{t+1} = χ^{(r)}_t + u(x_t)`, the predictive
`P(x_{t+1}|x_{1:t}) = Σ_r P(x_{t+1}|x^{(r)}_t, r_t) P(r_t|x_{1:t})`, and the
truncation of run lengths whose tail mass is below `10⁻⁴` giving average
cost `O(E[r])` are all as §11a states. Their finance example is a
zero-mean Gaussian with piecewise-constant variance on Dow Jones daily
returns 1972–75 with a gamma prior on the inverse variance (`a = 1, b =
10⁻⁴`) and `λ_gap = 250` — a ready-made fixture shape for the "variance
step detected" test. **Not verified here:** the robust variant's
closed-form posterior (Altamirano, Briol & Knoblauch, ICML 2023, PMLR 202);
only its abstract was read — take the equations from the paper as §11a
says.

---

## Task 49 — `refresh_time`

Nothing open beyond the example above. One semantic point worth a line in
the docstring, **verified** from BNHLS 2011 §2.1: the refresh-time vector at
`τ_j` is *assumed* to be a fully fresh price vector although only one series
actually updated at `τ_j` and the others are stale by at most one refresh
interval; the kernel's asymptotics are shown to be unaffected by that
staleness. An implementation that emits each series' last value at the
completing tick is exactly this.
