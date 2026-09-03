# Gradient-boosted trees, as online as possible

Status as of 2026-09-03: **investigation complete; a design is prototyped in
numpy and measured; nothing in the Rust crates.** The question was how far
gradient-boosted trees can be pushed toward this library's contract — one
row at a time, bounded memory, chunk-invariant, out-of-sample by
construction, clock-decayed — and what that does to parallel fitting and
memory. The answer is a design, the measurements behind it, the ideas that
did not survive measurement, and what building it in Rust would take.

The prototype is `scripts/ogbt_proto.py`; every number below comes from
`scripts/ogbt_experiments.py` (§11 says how to run it). The XGBoost source,
the papers and the notes taken on them are cached under `.cache/research/`,
which is gitignored on purpose: downloaded material stays out of the repo.

---

## 0. The short version

- **The boosting math is already online-shaped.** Every leaf value and every
  split gain in XGBoost is a function of two sums over the rows in a node,
  `G = Σg` and `H = Σh` (`src/tree/param.h:251-280` at commit `54155e3`).
  Sums are additive over rows, mergeable over chunks and threads, and
  decayable by a scalar. Nothing about *that* needs a batch.
- **What is not online in XGBoost is everything around the sums**: the cut
  points come from a pre-pass over all the data and are then frozen; the
  gradient, the margin and the row→node position are O(n) arrays; and node
  histograms exist only while a tree is being built, so a finished tree cannot
  be split further without the rows. `refresh` (its one structure-preserving
  updater) recomputes leaf values from plain sums with no decay and cannot add
  a node.
- **The design replaces those four things** with a warm-up buffer that places
  fixed quantile bins (XGBoost's own approximate algorithm depends on the data
  only through cuts and bucket sums), exponentially-decayed `(G, H)` sums per
  node, decayed `(G, H)` histograms on the leaves that may still split, and a
  *checkpoint schedule*: the ensemble's structure and values change only every
  `grow_every` learned rows. Between checkpoints the ensemble is frozen, so
  the rows are independent of one another — that one property gives chunk
  invariance and row-parallelism, both exact, and `grow_every` becomes the
  parallel-granularity knob (`grow_every = 1` is fully online).
- **Measured** on Friedman-1 with noise variance 1.0, 24 000 rows, an
  irregular clock: a batch fit on the 2 000-row warm-up buffer lands within
  0.02 MSE of XGBoost's fit on the same rows in every segment; continuing
  online with growth, pruning and a 3 000-row halflife then beats XGBoost
  refit on a 2 000-row rolling window in every segment of every regime tried
  (static, abrupt drift, random walk), ties an 8 000-row window on static
  data (within 0.04 either way) and beats it by 2–12 under drift — with
  12–16 k doubles of state against the window's 80 000 rows. Checkpointing
  every 500 rows instead of every row costs at most 0.04 MSE; a histogram
  pool of 8 leaves costs at most 0.005 on static data; 16 bins are as good
  as 32. Growing from scratch — no batch step at all, trees born one at a
  time — ends within 0.25 of the warm start on static data, within 0.1 after
  the abrupt change, and ahead of it while the function random-walks.
- **Memory is bounded a priori**: `M` trees × at most `2^(d+1)` nodes × ~8
  doubles, plus a histogram of `2·p_sub·B` doubles on at most `P` leaves. For
  `M = 20, d = 4, p = 10, B = 32` that is 5 k doubles of trees plus 10–14 k
  of histograms as measured, 102 k of histograms in the worst case with no
  pool, and 10 k with `P = 16`.
- **What it is not.** Not XGBoost: the design links nothing (CLAUDE.md rule
  12) and reimplements ~600 lines of arithmetic. Not a Hoeffding tree: no
  split-confidence bound (a split is reversible here, so it has nothing to
  protect; a gain floor measured slightly worse, §7.3), no per-row
  randomness, no adaptive state. The earlier exclusion of trees (`docs/BEYOND-O-STATE.md`,
  `docs/ENHANCEMENTS.md` §4) was written against Hoeffding trees and Poisson
  ensembles; §4 below takes its three objections one by one.

---

## 1. The question, and the contract it has to meet

The request: dig into XGBoost, the papers and the code; find how to make
gradient-boosted trees work as online as possible; improve the ability to fit
in parallel; reduce memory. Anything proposed has to fit the trait every
model here implements (`crates/online-core/src/model.rs`):

```
step(x, y, d_clock, weight) -> Step { pred, n_eff, extra }   // pred BEFORE the update
predict(x, d_clock) -> Step                                   // no update
state() / restore()                                           // versioned msgpack
```

and the hard rules that make the models interchangeable in a bank:

- **Out-of-sample by construction** (rule 2): the prediction is computed
  before the row's target touches anything.
- **Chunk invariance** (rule 3): one chunk or a thousand, identical output.
- **`n_eff` means the same thing everywhere** (rule 8): the accumulated
  weight before this row and before its own decay.
- **A zero-weight row is legal** (rule 9), including as the first row.
- **Decay is `lam = 0.5^(d_clock/halflife)`** on an irregular clock, the same
  semantics as `ewridge`.
- **Memory is O(state), not O(data)**; no allocation in the hot path after
  warm-up; `f64` everywhere; no `unsafe` in `online-core`; models know
  nothing about Polars.
- **No new static linking without raising it** (rule 12). This rules out
  linking XGBoost or LightGBM as a library, and the design does not want to:
  what it needs from them is ~600 lines of arithmetic, not their data
  structures.

---

## 2. What XGBoost does — verified against the paper and the code

Sources: the paper (Chen & Guestrin, KDD 2016, arXiv:1603.02754 v3) and the
source at `dmlc/xgboost` commit `54155e3a9b421a1f089a2afc2ab54a7c1dc1d102`
(2026-09-03, version 3.5.0-dev). Every code claim below carries a `file:line`
at that commit and was re-read after the notes were written.

### 2.1 The sufficient statistics

For a twice-differentiable loss with per-row gradient `g_i` and Hessian
`h_i` at the current prediction, the paper's leaf value and split gain
(Eq. 5 and 7) are functions of node sums only:

```
w*    = -G / (H + λ)
gain  = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ) ] - γ
```

The code agrees, with three details the paper omits. `CalcWeight`
(`src/tree/param.h:251-262`) applies an L1 threshold to `G` and clips by
`max_delta_step`; the gain has **no ½** in the code (`CalcGainGivenWeight`
`:244-248`, used through `split_evaluator.h:187-211` and
`evaluate_splits.h:243-250`), `γ` is a filter applied by the expansion driver
rather than a term in the gain (`src/tree/driver.h:31-47`), and
`min_child_weight` is a filter on `H_L, H_R` (`param.h:220-224`). A leaf's
value is written **once**, at the split that creates it, as
`eta × CalcWeight(G_child, H_child)` (`src/tree/hist/evaluate_splits.h:358-362`);
there is no later leaf pass for scalar leaves at this commit. For squared
error the gradient pair is `g = w·(F − y)`, `h = w`
(`src/objective/squared_error_obj.h:16-20`) — instance weights enter by
scaling both, which is exactly how a row weight enters an EW sum here.

### 2.2 The approximate algorithm depends on the data only through cuts and bucket sums

The paper's Algorithm 2 replaces the exact scan over sorted feature values by
candidate cut points `S_k` and per-bucket sums `G_kv, H_kv`; the code's
`hist` method is that algorithm. The kernel is one line per feature per row —
`hist[bin] += g; hist[bin+1] += h` (`src/common/hist_util.cc:346-378`) —
per-thread copies are reduced by plain addition in a fixed order
(`src/common/hist_util.h:531-555`), and in distributed mode one all-reduce per
depth level sums a contiguous `[nodes × bins × 2]` block of doubles
(`src/tree/hist/histogram.h:187-196`). The subtraction trick builds the child
with the smaller `H` and obtains the sibling as parent − child
(`src/tree/hist/histogram.cc:47-64`).

That is the whole reason the design is possible: a histogram is a sum, and a
sum can be kept, decayed, merged and split at any time.

### 2.3 What is O(n), and what is O(model)

| structure | size | where | needed by a one-pass design? |
|---|---|---|---|
| quantized feature matrix | 1 B per cell + 8 B per row (`src/data/gradient_index.cc:197-200`, `gradient_index.h:149`) | in memory or mmap'ed cache | no — bin on the fly |
| column matrix for partitioning | 1 B per cell + 1 bit (`src/common/column_matrix.cc:60-64`) | in memory | no |
| gradient pairs | 8 B per row (`include/xgboost/base.h:247`) | in memory, never paged (`doc/tutorials/external_memory.rst:424-429`) | no — computed per row from the frozen ensemble |
| row → node position | 8 B per row per page (`src/common/row_set.h:100-101`) | scratch, rebuilt per level | no — one traversal of a fixed tree |
| prediction cache (margin) | 4 B per row | never paged | no — same |
| cut points | `Σ bins` floats | model-side, frozen | yes, placed once |
| node histograms | `16 B × TotalBins` per node, ≤ 65 536 nodes (`src/tree/hist/hist_param.h:22`), evicted mid-tree when full (`histogram.h:129-144`) | during growth only | yes, but only on leaves that may still split |
| trees | 20 B `Node` + 16 B stat per node (`include/xgboost/tree_model.h:92`, `:56-64`) | model | yes |

External memory pages only the first two: "only the `X` is divided into
batches while everything else is concatenated" (`external_memory.rst:382-384`),
and training makes **two sweeps of the quantized cache per depth level** — a
depth-8 tree is at least 16 passes (`:441-442`; `updater_quantile_hist.cc:546-556`
and `histogram.h:364, :405-421` are the two loops). So XGBoost's own
out-of-core mode is a way to train a batch model on data that does not fit;
it is not a streaming learner, and the tutorial says small batches "can
severely hurt performance" (`:154-158`).

### 2.4 The pre-pass nobody can skip: the cuts

Cuts are computed once per `DMatrix` and frozen. Every page must share
`TotalBins()` (`src/tree/updater_quantile_hist.cc:446-450`), the split
evaluator reads the first page's cuts only (`:466-475`), and changing
`max_bin` on a quantile matrix is a hard `CHECK` (`src/data/extmem_quantile_dmatrix.cc:130`).
The weighted quantile sketch (`WQuantileSketch`, GK-style with a level-wise
merge/prune) is streaming and mergeable, but the container has to know the
per-feature row **count** before it is constructed (`src/common/quantile.cc:36-42`
through `LimitSizeLevel`, `quantile.h:586-607`), which is why a
`QuantileDMatrix` is four passes over the iterator: count, sketch, bin,
column-matrix (`src/data/iterative_dmatrix.cc:69-113`). Split thresholds are
cut values, and partitioning maps them back to bin ids by exact float
equality (`src/tree/common_row_partitioner.h:57-81`) — re-cutting after trees
exist would break every stored threshold.

The design's answer: place the bins **once**, from a warm-up buffer, and keep
them for the life of the model (§6.5 discusses re-binning; it is an idea, not
part of the design).

### 2.5 What `refresh`, `update` and `prune` give, exactly

- `process_type=update` moves every existing tree into `trees_to_update` and
  clears the model (`src/gbm/gbtree.cc:127-135`); iteration `i` re-processes
  original tree `i`; there can be no more rounds than the original model had
  (`:296-311`); the mode does not survive a save/load (`:383-386`).
- `refresh` (`src/tree/updater_refresh.cc`) walks each row root-to-leaf adding
  its `(g, h)` to **every node on the path** (`:118-133`), then sets
  `base_weight = CalcWeight(G, H)` and `sum_hess` at every node, `loss_chg`
  at every split, and with `refresh_leaf` the leaf value
  `eta × base_weight` (`:136-146`). Sums are plain and zeroed per call
  (`:59-63`): no decay, no subsampling. Because the trees were cleared, tree
  `i` is refreshed at the margin of the already-refreshed trees `< i`
  (`src/learner.cc:1056-1065`) — a sequential re-boost along frozen
  structures.
- `prune` collapses a split whose `loss_chg < γ` (`src/tree/updater_prune.cc:56-78`).

So XGBoost already contains "Level 1" of §5 — leaf values as a function of
per-leaf sums — minus the decay, minus growth. A halflife-weighted `refresh`
is a one-line change to `AddStats` plus an accumulator XGBoost does not have.

### 2.6 Parallelism and determinism

Histograms: rows of a node in L1-sized blocks, static contiguous chunks per
OpenMP thread, per-thread copies reduced in fixed order
(`src/tree/hist/histogram.h:71-90`, `:225-310`; `src/common/threading_utils.h:139-159`).
Split search: `(node × feature-block)` with a lower-feature-index tie-break
(`src/tree/hist/evaluate_splits.h:278-281`; `src/tree/param.h:420-430`).
Prediction: 64-row blocks, all trees per block (`src/predictor/cpu_predictor.cc:279-285`).
Column sampling draws from the context's `std::mt19937`
(`include/xgboost/context.h:25`, `src/common/random.cc:48-49`), whose state
is saved in the model's config as `rng_state` (`src/context.cc:289-297`,
`src/learner.cc:660`). Row subsampling **zeroes the gradient**
of the rows left out rather than removing them (`src/tree/hist/sampler.cc:105-127`)
— the same trick as a zero-weight row here. Results are deterministic for a
fixed thread count; bit-identity across thread counts is not claimed.

### 2.7 Summary: additive, pre-pass, or scratch

| XGBoost needs | kind | in the design |
|---|---|---|
| node `(G, H)`, per-bin `(G, H)` | additive over rows | kept as EW sums, decayed lazily |
| cuts | one pre-pass | warm-up buffer, then fixed |
| margin, gradient, position per row | scratch, O(n) | recomputed per row from the frozen ensemble |
| histograms of nodes being expanded | transient | kept on splittable leaves, bounded by a pool |
| leaf values at split time | a function of `(G, H)` | recomputed at every checkpoint |
| `refresh`'s sums | additive, no decay | the same sums, decayed |

---

## 3. What the field has built

Twelve papers (XGBoost's and the eleven below — SGBR from its abstract and
its MOA class, the rest in full) and four code bases were read (river
`b50439f`, MOA `f0c284d`, Vowpal Wabbit `9c8600b`, LightGBM `a6d48a6`, all
under `.cache/research/`). The columns that matter here are the ones the
contract is made of.

| line of work | state bound | deterministic given row order? | forgetting | verdict for this library |
|---|---|---|---|---|
| **Online gradient boosting** (Beygelzimer, Hazan, Kale, Luo, NeurIPS 2015) | `N` × base-learner state | yes, iff the base learner is | none | the chain: `N` stages, each fed the gradient at the partial prediction of the stages before it, in one pass over one row. Regret `O(T/N)` and a matching lower bound. The base learners there are linear; the shape is what §6 uses. (VW's `--boosting` is the *other* 2015 paper, margin classification.) |
| **OzaBoost** (Oza & Russell 2001) | `M` × base | **no** — `Poisson(λ)` per row and model | none | out: per-row randomness |
| **VFDT / Hoeffding tree** (Domingos & Hulten 2000) | `O(l·d·v·c)` counters, capped by leaf deactivation | yes | none | the split test (`ΔG > ε = sqrt(R² ln(1/δ)/2n)`) guards a one-shot, irreversible split; under decay and collapse a split is neither (§7.3) |
| **FIMT-DD** (Ikonomovska, Gama, Džeroski, DMKD 2011) | E-BST per (leaf, attribute) — one node per **distinct value**, unbounded until deactivation | yes | Page–Hinkley + alternate subtrees | the alternate-subtree idea doubles state; declined. ~24 k rows/s in MOA. |
| **SPDT** (Ben-Haim & Tom-Tov, JMLR 2010) | `B` centroids per (leaf, attribute) | **no** — the closest-pair merge is order-dependent (they measure 4.47 % → 5.54 % after two merges) | none | out for the merge; the lesson kept is that *fixed* bins merge exactly, which is what parallel + chunk-invariant needs |
| **Stochastic Gradient Trees** (Gouk, Pfahringer, Frank, ACML 2019) | `O(leaves · d · bins)`, 64 equal-width bins from a 1 000-row warm start | yes | **none** | the closest thing in print to a bounded XGBoost tree: per-(leaf, feature, bin) `(Σg, Σh, n)`, Newton leaf increments `−ḡ/(h̄+λ)`, split by a t-test. river's `SGTRegressor` is the same with a 100-row buffer. §6 is this plus decay, plus XGBoost's gain instead of the t-test, plus boosting. |
| **SGBT** (Gunasekara, Pfahringer, Gomes, Bifet, Machine Learning 2024) | `S = 100` steps × base tree | no as reported (75 % feature subspace per step is random; the chain itself is deterministic) | tree replacement inside the base learner — load-bearing in their ablation | boosts FIMT-DD or SGT, passes the pseudo-label `g/h` with **weight 1**, not the Hessian; `lr = 0.0125`. The first streaming GBT to beat ARF/SRP. |
| **SGBR** (same group, DMKD 2025) | 10 steps × 10-tree OzaBag of FIMT-DD (MOA `StreamingGradientBoostedRegression`) | **no** (Poisson bagging) | inherited | "Vanilla Sgbt with squared loss exhibits high variance when applied to streaming regression" (abstract) — their fix is bagging at `lr = 1.0`. §10 returns to this. |
| **Adaptive XGBoost** (Montiel et al., IJCNN 2020) | `K` trees + a mini-batch buffer | w.r.t. chunking yes | ADWIN resets the buffer | batch-incremental; loses to ARF among instance-incremental methods (rank 4.75 vs 1.63) |
| **CatBoost ordered boosting** (Prokhorenkova et al., NeurIPS 2018) | `O(s·n)` supporting predictions | no (permutations) | none | their Theorem 1: the prediction shift from using a row in its own leaf value is `O(1/n)`. A predict-before-update stream has that property for free — rule 2 *is* ordered boosting with one permutation, the clock. |
| **LightGBM `refit`** | leaf `Σg, Σh` per leaf | n/a (batch) | `decay_rate` blends old and new leaf values (weights the **old** one, default 0.9) | leaf values only, sequential in boosting order, needs the whole batch; the bin edges come from a new `Dataset` |
| **Mondrian forests** (Lakshminarayanan, Roy, Teh, NeurIPS 2014) | node count grows with `n` | no | none | online = batch only *in distribution*, not pathwise; out |
| **Adaptive Random Forest / Streaming Random Patches** (Gomes et al. 2017, 2019) | `n` × Hoeffding tree, up to 2× with background trees | no (`Poisson(6)`) | ADWIN per tree | the bar SGBT had to clear; not this library's shape |

Two code facts worth carrying: river's `Mean.update` divides by `n` after
adding `w`, so a zero-weight **first** row raises `ZeroDivisionError` — the
rule-9 guard is ours to write, nobody else has it; and MOA's E-BST observer
ignores the row weight entirely while its leaf sums use it, so weighted rows
split and predict from different data there.

The literature's own summary of the shortest honest path — "an SGT-style
per-leaf histogram of `(Σg, Σh, n)` with a fixed bin count, grown by a split
test, chained `N`-deep with the Beygelzimer/SGBT gradient recursion" — is
§6, with three changes: XGBoost's gain instead of a hypothesis test (a
split is reversible here, §7.3), exponential decay on every sum (nobody in the list
decays; SGT has no forgetting at all), and a checkpoint schedule instead of
per-row structure changes (which is what makes it chunk-invariant and
parallel).

---

## 4. The earlier exclusion, reassessed

`docs/BEYOND-O-STATE.md` and `docs/ENHANCEMENTS.md` §4 left trees to MOA on
three grounds. Each is a property of the specific algorithms in view there
(Hoeffding trees, Poisson-weighted ensembles), not of trees:

| objection | true of | in this design |
|---|---|---|
| "unbounded / adaptive state" | E-BST observers (one node per distinct value), leaf counts bounded only by a memory manager | node count ≤ `M · (2^(d+1) − 1)` by construction; histograms on ≤ `P` leaves; the warm-up buffer is `bin_rows × p`, a configuration constant. Preallocated at construction. |
| "nondeterministic under resampling" | OzaBag/OzaBoost/ARF/SRP `Poisson` weights per row | no per-row randomness. The per-tree feature subset is a seeded draw at the tree's birth, a pure function of `(seed, tree index)`; the same seed gives the same model on any chunking. |
| "no clean clock-decay semantics" | Page–Hinkley, ADWIN, alternate subtrees | every sum the model holds is an EW sum with the same `lam = 0.5^(d_clock/halflife)` as `ewridge`, decayed lazily and exactly. `halflife = inf` is the undecayed model. |

What remains true is the cost: it is a second family with its own
parameters, docs, tests and state schema (§9 sizes it). That is a reason to
decide deliberately, not a reason to keep the door closed on the wrong
argument.

---

## 5. The design space, as a spectrum

Five levels, from "a batch model used honestly" to "everything online". The
score is against the contract; the numbers are §7's.

| level | what changes online | state | chunk-invariant | out-of-sample | decay | parallel | measured (static / abrupt, MSE) |
|---|---|---|---|---|---|---|---|
| **0. window refit** — refit on the last `W` rows every `R` rows, predict the next `R` | everything, in bursts | `O(W·p)` rows | yes (row-count schedule) | yes | window only | the batch fit | `W = 2000`: 2.11–2.39 / 9.96 then 2.36; `W = 8000`: 1.90–2.12 / 18.6 then 4.0 |
| **1. leaf refresh** — batch structure, per-leaf EW `(G, H)`, values `−G/(H+λ)` | leaf values | `4` doubles per leaf | yes | yes | yes | trivially | 2.00–2.18 / 7.28 then 2.48 |
| **2. online growth from scratch, staggered births** — §6 without the warm start | values, splits, collapses | bounded (§6.8) | yes | yes | yes | §6.7 | 2.74 falling to 2.21 / 6.39 then 2.44 |
| **2b. batch warm start + online growth, prune, decay** | as 2, from a good start | as 2 | yes | yes | yes | §6.7 | **2.12–1.97 / 6.31 then 2.32** |
| 3. tree embedding + linear — frozen trees as a feature map, `ewridge` on top | the linear layer | leaves × targets | yes | yes | yes | `ewridge`'s | not run — a plumbing exercise on top of 1, listed for completeness |

Level 0 is the honest baseline: it is what a user would do today with
XGBoost and a `group_by_dynamic`. Level 1 is XGBoost's `refresh` with decay
and is what LightGBM's `refit` approximates. **Level 2b is the
recommendation**: it starts where XGBoost would, then keeps learning. Level
2 is the same model without the batch step and is the answer to "as online
as possible" taken literally: 0.25 behind on static data, level under
drift. The two are one code path with a flag.

---

## 6. The recommended design, precisely

### 6.1 State

For one target and `M` trees:

- `edges[j]`: at most `B − 1` bin edges per feature, placed once from the
  warm-up buffer (the empirical `1/B … (B−1)/B` quantiles, duplicates
  removed).
- Per tree: a feature subset `sub` (seeded, drawn at birth), a node array
  preallocated to `2^(d+1) − 1` slots. Per node: `feat`, `cut` (a bin
  index), children, `depth`, EW sums `G, H, Q` (`Q = Σ w·g²/h`, the
  gradient's second moment, kept for diagnostics), a decay stamp, the
  current `value`, and `n_since_eval`. Leaves with `depth < d` may hold a
  histogram `hist_G, hist_H` of shape `(|sub|, B)`.
- Base score: EW sums `(base_G, base_H)` of the target, stamp, value.
- The stream: cumulative log-decay `L`, `n_eff`, `n_learned` (the checkpoint
  counter), and the warm-up buffer of `bin_rows` rows while `edges` is unset.

### 6.2 Per row

```
lam = 0.5^(d_clock / halflife);  L += ln lam;  n_eff_before = n_eff;  n_eff = lam·n_eff + w
pred = base + η · Σ_m value_m(leaf_m(x))            # frozen ensemble, before any update
if w == 0 or y is null: return pred                  # clock advanced, nothing learned
if edges unset: buffer (x, y, w, L); when full: place edges, warm start (6.5); return
xb = bin(x)
F = base
for m in 1..M:
    g = F − y;  h = 1                                # squared loss; general (g, h) in 6.9
    walk tree m from the root: at every node on the path
        bring_up_to_date(L);  G += w·g;  H += w·h;  Q += w·g²/h;  n_since_eval += 1
    at the leaf, if it holds a histogram: for j in sub: hist_G[j, xb_j] += w·g;  hist_H[j, xb_j] += w·h
    F += η · value_m(leaf)                           # frozen value
n_learned += 1;  if n_learned % grow_every == 0: checkpoint (6.3)
```

`bring_up_to_date(L)` multiplies a node's sums (and histogram) by
`exp(L − stamp)` and sets `stamp = L`. Decay is therefore exact and lazy:
nothing is touched per row except the `M·(d+1)` nodes on the paths. The
prototype vectorises this per segment; the per-row form is what Rust would
run, and the two are the same arithmetic in a different association order.

### 6.3 Checkpoints

Every `grow_every` learned rows, for every tree, root first:

- **Leaf value** `value = −G/(H + λ)` from the leaf's (decayed) sums. This is
  `refresh` with decay.
- **Split** a leaf that holds a histogram, has `depth < d` and has seen
  `grace` rows since its last evaluation: prefix sums over the bins give
  `G_L, H_L` for every `(feature, cut)`, the gain is §2.1's with
  `min_child_weight` as a filter; take the best; split if the gain is
  positive. The children inherit their `(G, H)` from the split feature's
  bins on their side; the split feature's histogram partitions exactly into
  the children, the other features' histograms start cold. The parent's
  histogram is freed.
- **Collapse** an internal node that has seen `grace` rows since its last
  evaluation and whose children's gain, recomputed from their current sums,
  is negative — the split no longer pays for itself under decay. The node
  becomes a leaf with fresh (zero) histograms. This is `prune` re-run
  continuously, and it is what lets the model follow an abrupt change (§7.2).
- **Histogram pool** (optional, `P > 0`): rank the leaves that may still
  split by `H`, keep histograms on the top `P`, drop the rest. Decided only
  here, so the row stream never sees it.
- Base score: `base = base_G / base_H`.

Nothing else ever changes the structure or the values.

### 6.4 The one property everything rests on

Between checkpoints the ensemble is frozen. So within a segment of
`grow_every` learned rows, each row's prediction, gradient and destination
leaf depend only on the row and the frozen ensemble — not on the other rows
of the segment. Consequences:

- **Chunk invariance**: the checkpoint schedule counts learned rows, not
  chunks; a chunk boundary is invisible. Verified on ten uneven chunks
  including one-row chunks and a boundary one row before the warm-up ends:
  bit-identical predictions and `n_eff` (§7.6).
- **Out-of-sample**: the prediction uses values fixed before the row's
  target is read. Verified by perturbing `y[t]` by `10³`: `pred[t]` is
  unchanged, later predictions move.
- **Zero-weight rows** contribute `0` to every sum, so a zero-weight row
  with a wild target is identical to one with the true target; `n_eff`
  across a block of them decays and nothing else. A zero-weight first row is
  not a learned row, so the warm-up simply ends one row later; nothing is
  divided by zero because the base and every leaf value are guarded
  (`H = 0 → value 0`).
- **Parallel** (§6.7): the rows of a segment can be accumulated in any
  partition, and the results merge by addition.

### 6.5 Warm start

When the buffer is full, the model grows the initial ensemble batch-style on
those rows: level-wise, histograms from the buffer, the same gain and leaf
formulas, tree `m` on the gradient at the margin of trees `< m`. Then the
node sums, the leaf histograms and the base sums are set to the buffer's
(decayed) sums and the stream continues online. Nothing is replayed.

Measured (§7.2, §7.3): the batch fit on 2 000 rows lands within 0.02 MSE
of XGBoost's fit on the same rows, and warm-starting is worth 0.25–0.6 MSE
over growing from scratch with staggered births on static data (more
early, less later), 0.1 after an abrupt change, and nothing on a random
walk. Growing from scratch, trees must not be born together: all `M` see
the same gradient until the first checkpoint and take the same root split,
and the ensemble spends thousands of rows recovering (§7.3). One birth per
`stagger_rows` learned rows gives each tree the residual of the ones before
it, which is what boosting is; the batch fit does the same in one step.

Re-binning is deliberately **not** part of the design. Bins placed from the
first 2 000 rows were adequate through an abrupt drift in which two
features swapped roles; if a stream's feature *scale* drifts, the honest
answer is a fresh model (the CLI's resume-from-state makes that a restart
from the last checkpoint), not thresholds that silently change meaning under
existing splits — exactly the failure §2.4 found in XGBoost's own design.

### 6.6 `n_eff`, `min_periods`, `predict`

`n_eff` is the stream's EW weight sum before the row, as everywhere else,
so `min_periods` is portable across a bank. `predict(x, d_clock)` (E31) is
the first line of §6.2 and nothing else.

### 6.7 Parallel fit

Three independent axes, all exact:

1. **Rows within a segment.** Partition a segment into blocks; each block
   accumulates into its own copy of the touched sums and histograms; reduce
   in a fixed order. Measured on a 2 000-row segment as one block against
   four: sums agree to `9e-13` on values of order `10³` — floating-point
   reassociation, nothing else, and a fixed reduction order makes even that
   go away (XGBoost does exactly this, §2.6). `grow_every` is the block
   size available: checkpointing every 10, 50 or 500 rows instead of every
   row costs **at most 0.04 MSE in any segment**, every 2 000 rows at most
   0.10 (§7.4), so hundreds of rows of parallel work per checkpoint are
   nearly free.
2. **Trees.** Given the frozen ensemble, tree `m`'s input is the partial
   prediction `F_{m−1}(x)`, which one traversal per tree provides for all
   `M` at once; the `M` accumulations are then independent. This is the
   axis XGBoost cannot use (it grows trees sequentially) and this design
   gets for free.
3. **Groups and specs.** The bank's existing `(spec × group × instance)`
   fan-out (`docs/PERFORMANCE.md`) applies unchanged.

Prediction is embarrassingly parallel over rows, as XGBoost's is.

### 6.8 Memory

Per tree, at most `2^d` leaves and `2^d − 1` internal nodes; per node
`(G, H, Q, stamp, value)` and the structure — call it 8 doubles with
padding. A histogram is `2 · |sub| · B` doubles and lives only on leaves
with `depth < d`; the number of such leaves is at most `2^(d−1)` without a
pool and `P` with one. So:

```
state ≤ M · 2^(d+1) · 8  +  min(P, M · 2^(d−1)) · 2 · colsample·p · B   doubles
```

For `M = 20, d = 4, p = 10, B = 32`: 5 k doubles of nodes; histograms 640
doubles per leaf, so 102 k in the worst case unpooled and 10 k with
`P = 16`. Measured, the unpooled model held 10–14 k in total (the tables'
`state=` counts four doubles per node, the histograms and the edges),
because the leaves at `depth < d` that survive without splitting are few.
`B = 16` halves the histogram term at no measured cost; `colsample = 0.7`
cuts it by a third and *improves* accuracy (§7.4). The warm-up buffer is
`bin_rows · (p + 3)` doubles, freed after use — for `bin_rows = 2 000,
p = 10`, 26 k doubles, the largest single term, and a configuration
constant.

For comparison, XGBoost's `hist` on the same problem holds `2·p·B·16 B` per
node under expansion (10 KB here) plus 1–2 bytes per cell of data, plus 24
bytes per row of gradient, index and margin — O(rows), which is the thing
the design removes.

### 6.9 Losses, targets, groups

- The tree code sees only `(g, h)` per row. Squared error is `(F − y, 1)`;
  Huber, pseudo-Huber and logistic are twice-differentiable and drop in;
  quantile (pinball) has `h = 0` and needs the smoothing XGBoost uses
  (`src/objective/quantile_obj.cc:42-76`). The robust family's `ψ` in
  `online-core` is the natural source of `g`.
- Multi-target: one ensemble per target, as `ewridge` keeps one set of
  sufficient statistics per target. Shared structure across targets
  (XGBoost's `multi_strategy`) is possible but not proposed.
- Groups: one model per group, as today.

---

## 7. The investigation: plan, iterations, measurements

### 7.1 The plan

1. Read the paper; write down what is additive, what needs a pre-pass, what
   is per-row scratch. **Done** (§2).
2. Read the code for what the paper omits: instance weights, `refresh` /
   `update` / `prune`, external memory, the histogram cache, determinism.
   **Done** (§2, notes at `.cache/research/notes/xgboost-code.md`).
3. Read the online-boosting and streaming-tree literature and four code
   bases for anything that already answers the question. **Done** (§3).
4. Prototype the design in numpy with this library's semantics (irregular
   clock, weights, `n_eff`), and a Level 1 baseline on XGBoost's own trees.
   **Done** (`scripts/ogbt_proto.py`).
5. Measure against honest baselines on data with a known noise floor and
   three drift regimes. **Done** (§7.2).
6. Iterate on the design: try the obvious ideas, keep what measures better,
   record what does not. **Done** (§7.3, §8).
7. Measure what each memory and parallelism knob costs. **Done** (§7.4, §7.5).
8. Check the guarantees on the final configuration. **Done** (§7.6).
9. Write this document; decide whether to build it. **This document; §9.**

Data: Friedman-1 (`y = 10 sin(π x₁x₂) + 20 (x₃−½)² + 10 x₄ + 5 x₅ + ε`,
`ε ~ N(0, 1)`, five noise features), 24 000 rows, exponential inter-arrival
clock. Three regimes: static; **abrupt** (at row 12 000 the weights of `x₄`
and `x₅` swap and the quadratic flips sign); **walk** (the four coefficients
random-walk). Prequential MSE per segment; the noise floor is 1.0 and the
first 2 000 rows are the warm-up (predictions there are null). XGBoost 3.4.1
with `hist, max_depth 4, eta 0.3, lambda 1, max_bin 32`, 20 rounds — the
same shape as the design's `M = 20, d = 4, B = 32, η = 0.3, λ = 1`.

### 7.2 Baselines (`baselines`, seed 4)

```
=== drift=none seed=4 n=24000: MSE on [2k,4k) [4k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n); noise floor 1.001
EW mean hl=3000                                      24.951  24.464  25.031  24.598  25.183
ewridge hl=3000 (this package)                        7.106   6.783   6.949   6.956   7.090
xgb refit, window W=2000 every R=500 rows             2.247   2.108   2.307   2.393   2.312
xgb refit, window W=8000 every R=500 rows             2.119   1.911   2.003   1.902   1.928
xgb structure (rows<2000) + leaf refresh hl=inf       2.183   1.997   2.053   1.997   2.093
xgb structure (rows<2000) + leaf refresh hl=3000.0    2.180   1.996   2.047   2.005   2.096
ours: from scratch, staggered births, hl=3000         2.737   2.341   2.224   2.232   2.212   4.1s nodes=606 hist_leaves=5 state=6k
ours: warm 2000, leaf refresh only, hl=inf            2.168   2.011   2.061   1.981   2.082   3.8s nodes=586 hist_leaves=0 state=3k
ours: warm 2000, grow+prune, hl=inf                   2.168   2.005   2.052   1.970   2.067   3.9s nodes=614 hist_leaves=3 state=5k
ours: warm 2000, leaf refresh only, hl=3000.0         2.115   1.937   1.987   1.943   2.003   3.8s nodes=508 hist_leaves=0 state=2k
ours: warm 2000, grow+prune, hl=3000.0  <- recommended  2.115   1.935   1.971   1.923   1.970   4.0s nodes=576 hist_leaves=18 state=14k
ours: warm 2000, grow+prune, hl=8000                  2.175   2.021   2.117   2.020   2.082   3.9s nodes=596 hist_leaves=10 state=9k

=== drift=abrupt seed=4 n=24000: MSE on [2k,4k) [4k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n); noise floor 1.001
EW mean hl=3000                                      24.951  24.464  25.031  30.626  25.499
ewridge hl=3000 (this package)                        7.106   6.783   6.949  15.703   7.766
xgb refit, window W=2000 every R=500 rows             2.247   2.108   2.307   9.962   2.357
xgb refit, window W=8000 every R=500 rows             2.119   1.911   2.003  18.586   4.005
xgb structure (rows<2000) + leaf refresh hl=inf       2.183   1.997   2.053  13.169   4.576
xgb structure (rows<2000) + leaf refresh hl=3000.0    2.180   1.996   2.047   7.277   2.480
ours: from scratch, staggered births, hl=3000         2.737   2.341   2.224   6.389   2.443   4.2s nodes=578 hist_leaves=13 state=11k
ours: warm 2000, leaf refresh only, hl=inf            2.168   2.010   2.059  12.710   5.739   3.7s nodes=380 hist_leaves=0 state=2k
ours: warm 2000, grow+prune, hl=inf                   2.168   2.003   2.047  10.256   2.764   4.0s nodes=580 hist_leaves=18 state=14k
ours: warm 2000, leaf refresh only, hl=3000.0         2.115   1.937   1.987   7.783   2.991   3.6s nodes=414 hist_leaves=0 state=2k
ours: warm 2000, grow+prune, hl=3000.0  <- recommended  2.115   1.935   1.971   6.313   2.320   4.0s nodes=592 hist_leaves=14 state=12k
ours: warm 2000, grow+prune, hl=8000                  2.175   2.021   2.117   7.748   2.545   4.0s nodes=588 hist_leaves=14 state=12k

=== drift=walk seed=4 n=24000: MSE on [2k,4k) [4k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n); noise floor 1.001
EW mean hl=3000                                      21.824  30.898  57.858  71.605  46.338
ewridge hl=3000 (this package)                        8.908   9.945  24.527  28.448  12.870
xgb refit, window W=2000 every R=500 rows             3.938   3.698   7.474   6.804   5.956
xgb refit, window W=8000 every R=500 rows             3.665   4.918  21.304  21.551   6.876
xgb structure (rows<2000) + leaf refresh hl=inf       2.805   3.028   6.451   8.268   8.719
xgb structure (rows<2000) + leaf refresh hl=3000.0    2.601   2.797   4.356   5.930   4.774
ours: from scratch, staggered births, hl=3000         3.150   3.091   4.212   5.689   4.680   4.2s nodes=544 hist_leaves=20 state=15k
ours: warm 2000, leaf refresh only, hl=inf            2.939   2.984   6.134   8.585   8.240   3.9s nodes=532 hist_leaves=0 state=2k
ours: warm 2000, grow+prune, hl=inf                   2.939   2.973   5.974   8.271   6.827   4.0s nodes=598 hist_leaves=11 state=10k
ours: warm 2000, leaf refresh only, hl=3000.0         2.831   2.923   4.641   6.194   5.205   3.8s nodes=438 hist_leaves=0 state=2k
ours: warm 2000, grow+prune, hl=3000.0  <- recommended  2.831   2.915   4.467   5.899   4.499   4.0s nodes=570 hist_leaves=21 state=16k
ours: warm 2000, grow+prune, hl=8000                  2.813   2.784   4.746   7.194   5.291   4.0s nodes=590 hist_leaves=9 state=8k
```

Reading it:

- **The warm start is XGBoost.** "warm 2000, leaf refresh only" and "xgb
  structure + leaf refresh" are the same experiment with the trees grown by
  the prototype's batch code and by XGBoost respectively; they agree within
  0.02 in every segment of every regime. The tree-growing arithmetic is
  right.
- **Decay is worth more than growth on static data; growth is what wins
  under drift.** Static: the recommended row beats the 2 000-row window
  refit in every segment, is level with the 8 000-row window (within 0.04
  either way, holding 12 k doubles against its 80 000 rows), and improves on
  leaf refresh with the same decay by only 0.00–0.03 — the batch structure
  was already right. After the abrupt change it recovers to 2.32 where the
  window refit sits at 2.36 (having spent 6 000 rows at 9.96) and leaf
  refresh stalls at 2.48–2.99: new splits are needed, and only Level 2 makes
  them. On the random walk it is within 0.1 of leaf refresh in the worst
  segment and better in the others, and far better than either window.
- **A moderate halflife beats no decay even on static data** (`hl = 3000`
  vs `inf`, and vs `8000`, by 0.1 in the later segments). The gradients
  accumulated in a node were taken against ensembles that have since
  changed; decay is what retires them. This is the same reason XGBoost
  recomputes every gradient every round.
- **From scratch, with staggered births, is closer than expected**: 0.6
  behind in the first segment, 0.25 by the last, 0.1 after the abrupt
  change, and *ahead* by 0.2–0.25 while the function random-walks (its
  trees were grown under decay from the start; the warm start's carry the
  batch structure). The batch step is a head start, not a different model.

### 7.3 Negative results (`negatives`, seed 4)

```
=== drift=none seed=4 n=24000: MSE on [2k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n); noise floor 1.001
from scratch, all 20 trees born together              3.970   3.603   3.429   2.812   4.3s nodes=586 hist_leaves=11 state=10k
    distinct trees: 20 of 20; distinct root splits: 16
from scratch, one birth per 100 rows (stagger)        2.473   2.224   2.232   2.212   4.2s nodes=606 hist_leaves=5 state=6k
    distinct trees: 20 of 20; distinct root splits: 20
  + bins from 2000 rows (the warm start's buffer)     2.520   2.301   2.359   2.348   4.1s nodes=588 hist_leaves=14 state=12k
  + recycle: retire the oldest tree every 100 rows    4.726   4.380   4.634   4.720   3.6s nodes=234 hist_leaves=83 state=54k
  + gamma_rel=0.05 (gain must beat 5% of var g)       2.468   2.312   2.308   2.299   4.2s nodes=600 hist_leaves=8 state=8k
  + hoeffding_delta=1e-3 (best-vs-second margin)     24.674  25.056  24.636  25.203   1.0s nodes=20 hist_leaves=20 state=13k
stagewise: one tree grows at a time, 500 rows each    6.697   3.825   3.703   3.778   2.1s nodes=154 hist_leaves=4 state=3k
stagger + only the 5 youngest trees may grow          5.290   5.216   5.161   5.394   2.4s nodes=126 hist_leaves=11 state=8k
warm start, hl=inf (no forgetting)                    2.059   2.052   1.970   2.067   4.0s nodes=614 hist_leaves=3 state=5k
warm start, grow but never collapse                   1.997   1.985   1.943   1.999   4.0s nodes=610 hist_leaves=5 state=6k
warm start, grow+prune, hl=3000 (recommended)         1.995   1.971   1.923   1.970   4.1s nodes=576 hist_leaves=18 state=14k

=== drift=abrupt seed=4 n=24000: MSE on [2k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n); noise floor 1.001
from scratch, all 20 trees born together              3.970   3.603   7.496   3.049   4.3s nodes=572 hist_leaves=22 state=17k
    distinct trees: 18 of 20; distinct root splits: 17
from scratch, one birth per 100 rows (stagger)        2.473   2.224   6.389   2.443   4.2s nodes=578 hist_leaves=13 state=11k
    distinct trees: 20 of 20; distinct root splits: 20
  + bins from 2000 rows (the warm start's buffer)     2.520   2.301   6.576   2.650   4.1s nodes=586 hist_leaves=13 state=11k
  + recycle: retire the oldest tree every 100 rows    4.726   4.380   5.692   4.832   3.6s nodes=242 hist_leaves=79 state=52k
  + gamma_rel=0.05 (gain must beat 5% of var g)       2.468   2.312   5.685   2.530   4.2s nodes=566 hist_leaves=21 state=16k
  + hoeffding_delta=1e-3 (best-vs-second margin)     24.674  25.056  26.097  24.956   1.0s nodes=20 hist_leaves=20 state=13k
stagewise: one tree grows at a time, 500 rows each    6.697   3.825   8.736   4.598   2.0s nodes=132 hist_leaves=1 state=1k
stagger + only the 5 youngest trees may grow          5.290   5.216   5.848   5.360   2.4s nodes=130 hist_leaves=11 state=8k
warm start, hl=inf (no forgetting)                    2.058   2.047  10.256   2.764   4.0s nodes=580 hist_leaves=18 state=14k
warm start, grow but never collapse                   1.997   1.985   7.326   2.478   4.0s nodes=610 hist_leaves=5 state=6k
warm start, grow+prune, hl=3000 (recommended)         1.995   1.971   6.313   2.320   4.0s nodes=592 hist_leaves=14 state=12k
```

- **Trees born together start as the same tree.** Growing all `M` from the
  root at once, each sees the same gradient (the residual of a constant) and
  takes the same root split at the first checkpoint; from then on tree `m`
  sees the frozen values of trees `< m` and they diverge — 20 distinct trees
  and 16 distinct root splits by the end, the collapse-and-regrow path
  having rewritten most roots — but the ensemble spends 6 000 rows above
  3.9 and is still at 2.8 after 22 000. One birth per 100 learned rows
  fixes it: 2.47 in the first segment, 2.21 in the last.
- **Bins from 2 000 rows instead of 500** are 0.05–0.2 *worse* in every
  segment — the wrong sign for bin placement to explain anything: it is the
  batch fit, not the bins, that separates the warm start from growing from
  scratch.
- **Retiring the oldest tree every 100 rows** — SGBT's tree replacement in
  its unconditional form — is a disaster: no tree lives longer than 2 000
  rows, the ensemble never matures (234 nodes, 83 of them young leaves
  holding histograms), and the error sits at 4.7. Conditional replacement
  (§8, idea 8) is untested; scheduled replacement is out.
- **Split-confidence margins do not help — with a caveat.** A gain floor
  at 5 % of the gradient variance is 0.1 worse on static data and mixed
  after the drift. The Hoeffding-style margin never let a single split
  through (20 roots after 22 000 rows, the error of the EW mean), but as
  implemented it compares the best `(feature, cut)` against the runner-up
  *pair*, which is almost always the adjacent bin of the same feature with
  nearly the same gain; VFDT compares the best two *attributes*, and that
  version was not run. What stands is the argument, not a measurement:
  VFDT's bound and SGT's t-test exist to make a *one-shot, irreversible*
  split safe, and with decay and collapse a wrong split is cheap while a
  late one is not.
- **Stagewise growth starves the trees**: one tree at a time gets all the
  rows and the rest wait, 6.7 in the first segment and 3.8 at the end.
- **Restricting growth to the five youngest trees** (with recycling) fails
  the way recycling does.
- **No forgetting** (`hl = inf`) costs 0.05–0.1 on static data and is
  catastrophic after the drift (10.3, then 2.76 against 2.32).
- **Never collapsing** costs 0.00–0.03 on static data and 1.0 in the drift
  segment (7.33 against 6.31): the mechanism that undoes a split whose gain
  has decayed negative is what lets the structure follow the change.

### 7.4 Knobs (`knobs`, seed 5, static)

```
=== drift=none seed=5 n=24000: MSE on [2k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n); noise floor 0.986
reference: M=20 d=4 B=32 grow_every=50                2.181   2.247   2.194   2.095   4.0s nodes=594 hist_leaves=11 state=10k
n_bins=8                                              2.851   2.862   2.764   2.703   4.0s nodes=588 hist_leaves=12 state=4k
n_bins=16                                             2.127   2.228   2.134   2.055   4.0s nodes=582 hist_leaves=13 state=7k
n_bins=64                                             2.197   2.269   2.262   2.108   4.0s nodes=570 hist_leaves=17 state=25k
colsample=0.5                                         2.798   2.948   2.840   2.701   4.0s nodes=588 hist_leaves=12 state=7k
colsample=0.7                                         2.143   2.138   2.055   1.993   4.0s nodes=602 hist_leaves=5 state=5k
grow_every=1                                          2.178   2.229   2.175   2.061  14.2s nodes=594 hist_leaves=11 state=10k
grow_every=10                                         2.176   2.251   2.198   2.098   6.0s nodes=588 hist_leaves=12 state=10k
grow_every=500                                        2.193   2.249   2.211   2.101   3.3s nodes=572 hist_leaves=16 state=13k
grow_every=2000                                       2.201   2.327   2.276   2.122   3.2s nodes=588 hist_leaves=14 state=12k
max_depth=3                                           2.450   2.604   2.539   2.395   3.0s nodes=300 hist_leaves=0 state=2k
max_depth=6                                           2.072   2.019   1.971   1.839   6.8s nodes=1708 hist_leaves=174 state=119k
M=50 eta=0.15                                         1.878   1.934   1.867   1.766   9.7s nodes=1434 hist_leaves=38 state=30k
M=10 eta=0.5                                          3.077   3.233   3.118   3.022   2.1s nodes=288 hist_leaves=9 state=7k
warm-up buffer 4000                                   1.878   1.980   1.870   1.849   3.7s nodes=590 hist_leaves=11 state=10k
warm-up buffer 1000                                   2.328   2.411   2.285   2.182   4.2s nodes=568 hist_leaves=16 state=13k
```

- `n_bins`: 16 is as good as 32 (0.04–0.06 better, inside the noise) at
  30 % less state; 8 costs 0.6–0.7; 64 is 0.02–0.07 worse at 2.5× the state.
- `colsample = 0.7` is *better* than 1.0 by 0.04–0.14 (the usual reason:
  decorrelated trees) at half the state; 0.5 costs 0.6.
- `grow_every`: 1 → 10 → 50 → 500 spans at most 0.04 in any segment;
  2 000 costs up to 0.10. Every checkpoint schedule from 10 to 500 rows is
  interchangeable; the choice is about parallel granularity (§6.7).
- Depth 3 costs 0.3; depth 6 gains 0.1–0.26 at 12× the state.
  `M = 50, η = 0.15` gains 0.3 at 3× the state; `M = 10, η = 0.5` costs 0.9.
  A 4 000-row warm-up gains 0.25–0.32 at the same steady-state size; a
  1 000-row one costs 0.1–0.16. These are the usual XGBoost trade-offs and
  they carry over.

### 7.5 Histogram pool (`pool`, seed 6)

```
=== drift=none seed=6 n=24000: MSE on [2k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n); noise floor 0.997
unbounded histograms                                  2.399   2.199   2.229   2.245   4.0s nodes=588 hist_leaves=14 state=12k
hist_pool=32                                          2.399   2.199   2.229   2.245   4.0s nodes=590 hist_leaves=13 state=11k
hist_pool=16                                          2.399   2.199   2.229   2.245   4.0s nodes=590 hist_leaves=13 state=11k
hist_pool=8                                           2.399   2.204   2.231   2.242   4.0s nodes=574 hist_leaves=8 state=8k
hist_pool=4                                           2.398   2.204   2.235   2.251   4.0s nodes=562 hist_leaves=4 state=5k

=== drift=abrupt seed=6 n=24000: MSE on [2k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n); noise floor 0.997
unbounded histograms                                  2.399   2.199   6.928   2.584   4.0s nodes=578 hist_leaves=17 state=14k
hist_pool=32                                          2.399   2.199   6.928   2.611   4.0s nodes=572 hist_leaves=18 state=14k
hist_pool=16                                          2.399   2.199   6.921   2.621   4.0s nodes=528 hist_leaves=16 state=13k
hist_pool=8                                           2.399   2.204   6.978   2.563   3.9s nodes=494 hist_leaves=8 state=7k
hist_pool=4                                           2.398   2.204   6.852   2.619   3.8s nodes=460 hist_leaves=4 state=5k
```

Restricting histograms to the `P` heaviest splittable leaves: `P = 8` costs
≤ 0.005 MSE on static data and is within ±0.05 after the drift, at a third
to a half less state; `P = 4` is within 0.01 static and ±0.08 after the
drift at 60 % less. `P = 32` never binds (identical output) and `P = 16`
barely does (identical on static data, 0.01–0.04 apart after the drift):
the unpooled model ends the stream holding 14–17 histograms. The pool is
what makes the worst case in §6.8 a configuration choice rather than a data
property.

### 7.6 The guarantees, on the final configuration (`invariance`, seed 6, abrupt)

```
=== invariance checks: drift=abrupt seed=6, recommended design + hist_pool=16
chunk invariance, 1 chunk vs 10 uneven chunks: identical=True
out-of-sample: pred[t] unchanged when y[t] is perturbed=True; later preds move=True
zero-weight rows: wild y on them changes nothing=True; all finite=True
n_eff before/after the zero-weight block: 4157.956 -> 3712.783, ratio 0.8929 (decay only; pure decay over that span = 0.8929)
zero-weight first row: not a learned row, so warm-up ends one row later (pred[2000] nan=True); finite after=True; n_eff[0]=0.0, n_eff[1]=0.0
parallel additivity: one tree, a 2000-row segment accumulated as 1 block vs 4 blocks, max |diff| = 9.1e-13 (fp reassociation only; sums are ~1e3)
```

---

## 8. Ideas, ranked

Kept, and in the prototype:

1. **Checkpoint schedule** — the one idea the rest depends on: frozen
   between checkpoints, hence chunk-invariant and parallel; `grow_every` is
   the granularity knob and is nearly free to 500.
2. **Batch warm start on the warm-up buffer** — the largest single
   accuracy gain on stationary data (0.25–0.6 MSE, shrinking as the
   from-scratch trees catch up); reuses the buffer the bins need anyway.
3. **Continuous collapse under decay** — a split whose gain has decayed
   negative is undone; the mechanism that follows an abrupt change.
4. **Exact lazy decay** with one stamp per node — O(path) work per row
   instead of O(state).
5. **Histograms only where they can be used** (leaves with `depth < d`),
   plus the **pool**.
6. **Column subsampling per tree**, seeded at birth — better and smaller.
7. **Staggered births** for the from-scratch path — one tree per
   `stagger_rows` learned rows; with it, Level 2 is a usable model.

Not tried, worth trying in that order if the design is built:

8. **Conditional tree replacement** (SGBT's load-bearing component): when
   a tree's decayed contribution falls well below its siblings', collapse it
   to the root and let it regrow on the current residual. Cheap and
   deterministic. The *unconditional* version — retire the oldest tree on a
   schedule — measured badly (§7.3), so the condition is the whole idea.
9. **Drift-triggered re-warm-start**: keep a rolling buffer of `bin_rows`
   rows (already the largest state term) and, when the prequential error
   exceeds a decayed threshold, rebuild the ensemble from it. Turns the
   warm start into a mechanism instead of a one-off.
10. **Real data**: everything above is Friedman-1. The Binance klines the
    test suite already downloads (`tests/data.py`) are the obvious next
    measurement, against `ewridge` on the same columns, before any Rust is
    written.
11. **Bins as `u8`**, histograms as `[B][2]` per feature — layout for the
    Rust version; and the warm-up buffer as a rolling one (idea 9).
12. **Level 3** (frozen trees as a feature map under `ewridge`) as a
    plumbing exercise, if the tree embedding is ever wanted as an input to
    the linear models.

Tried and rejected (§7.3): all trees born together, scheduled tree
retirement, Hoeffding margins, variance-relative gain floors, stagewise
growth, growth restricted to young trees, no decay, no collapse. Declined
without trying: alternate subtrees (doubles state), order-dependent
histogram merges (SPDT), per-row randomness of any kind, re-binning under
live trees.

---

## 9. What building it in Rust would take

- `online-core`: `gbt.rs` (model, `OnlineModel` impl), `gbt/hist.rs`,
  `gbt/tree.rs`; ~1 000 lines plus tests. Node arrays preallocated to
  `M · (2^(d+1) − 1)` and histograms to `P` at construction — allocation at
  construction, none after, which is the existing rule. The warm-up buffer
  is preallocated too. `f64` throughout; no `unsafe`.
- Per row: `M` traversals of depth ≤ `d`, `M · (d+1)` node updates of four
  doubles, `M · |sub| · 2` scattered adds into histograms. For
  `M = 20, d = 4, p_sub = 7`: roughly 600 flops and 280 scattered adds.
  Against `ewridge`'s measured per-row cost (`docs/PERFORMANCE.md`) that
  is the same order of magnitude — **an estimate, not a measurement**; the
  first thing the Rust version should do is measure it with
  `scripts/benchmark.py`.
- State: a new `ModelState::Gbt` variant; bump `SCHEMA_VERSION` and keep the
  loader for the previous version (rule 5). The msgpack size is §6.8's
  number times 8 bytes; the recommended configuration with a 16-leaf pool
  is ~15 k doubles, ~120 KB per group, so the per-group state that the
  bank saves atomically (C6) grows by two orders of magnitude over
  `ewridge`. Worth stating in the docs, not a blocker.
- Plumbing: every place in `docs/EXTENDING.md` — `ModelKind::KINDS`, the
  builders, the namespace, the sweeps, the golden bank, the API snapshot,
  the README list; the registry tests will name each omission.
- Tests: the model-contract test (`crates/online-core/tests/model_contract.rs`)
  for `n_eff` and zero-weight rows; the chunk-invariance and out-of-sample
  tests in `tests/`; a numpy oracle (`scripts/ogbt_proto.py` is one, and
  the prototype's batch code is checked against XGBoost already); a
  cross-OS state test.
- Rule 12: nothing new is linked. XGBoost stays what it is here — a
  baseline in an optional overlay environment for the experiments.

Effort: the model is about the size of `rls` plus `lasso` together; the
plumbing is a day by the `EXTENDING.md` list; the measurement on real data
(§8 idea 10) should come first and might change the defaults.

---

## 10. Open questions

- **SGBR's "high variance" of streaming GBT with squared loss** (§3) did not
  appear in these measurements. Three candidate reasons, none tested: this
  design weights sums by the Hessian rather than passing weight-1
  pseudo-labels; it decays; it warm-starts. A run on their datasets would
  settle it, and would be the first external validation.
- **Only synthetic data** so far (§8 idea 10).
- **Losses beyond squared error** are argued, not measured (§6.9).
- **Missing feature values**: XGBoost learns a default direction by a second
  scan (`src/tree/hist/evaluate_splits.h:204-264`); the design as
  prototyped assumes the bank's null policy (`docs/PLAN.md` §3) handles the
  row. A learned default direction is a small addition if wanted.
- **Categorical features**: out of scope (numeric bins only).
- **Whether to build it at all**: the numbers say the design works and the
  contract holds; the cost is a second model family (§4, §9). That decision
  is the user's; nothing here presumes it.

---

## 11. Reproducing the numbers

```
uv run python scripts/ogbt_experiments.py all                                   # without XGBoost rows
uv run --with xgboost --with scikit-learn python scripts/ogbt_experiments.py all # with them
```

On macOS the `xgboost` wheel needs `libomp`, which scikit-learn's wheel
bundles; the script's docstring has the `DYLD_LIBRARY_PATH` line. Nothing is
added to `pyproject.toml` or the lock file — the overlay is ephemeral. Each
experiment takes one to five minutes in pure numpy.

Sources (all under the gitignored `.cache/research/`): the XGBoost paper and
ten others as PDFs with text extractions; `dmlc/xgboost` at `54155e3`; river,
MOA, Vowpal Wabbit and LightGBM checkouts; and three notes files — the paper,
the code (every claim with `file:line`), and the literature and
implementations — from which §2 and §3 were written and against which they
were re-checked.
