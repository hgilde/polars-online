# Online clustering, every family that can be made to fit

Status as of 2026-09-04: **investigation in progress on the branch
`online-clustering`; designs prototyped in numpy and measured; nothing in the
Rust crates.** The question the user asked was to do "all the clustering types
that may be possible online" — so this document surveys every family the field
has produced, decides each against this library's contract, prototypes the ones
that pass, and measures them. It reopens the one-line exclusion in
`docs/ENHANCEMENTS.md` §4.

The prototypes are `scripts/clustering_proto.py`; every number below comes from
`scripts/clustering_experiments.py` (§11 says how to run it). The river, MOA,
scikit-learn and Spark sources, the papers and the notes taken on them are
cached under `.cache/research/`, which is gitignored on purpose: downloaded
material stays out of the repo.

---

## 0. The short version

- **Clustering fits the contract more easily than boosting did.** Every
  workable family reduces to the same object the library already has: a
  **weighted mean with an exponentially decayed count**, `EwCov`'s update with
  a soft or hard assignment in front of it. `n_j' = lam·n_j + w`,
  `c_j += (w / n_j')·(x − c_j)`. That is MacQueen's `1/n_k` step (Bottou &
  Bengio's Newton learning rate, `bottou1995:129-141`) with the count decayed,
  it is Spark's `StreamingKMeans` update at chunk size one
  (`StreamingKMeans.scala:117-126`), and it is scikit-learn's exact
  mini-batch mean at chunk size one (`_k_means_minibatch.pyx:87-104`).
- **What does *not* fit is anything that needs the rows back**: k-medoids and
  every pairwise-distance method (spectral, affinity propagation, HDBSCAN,
  agglomerative on rows), sliding windows, and the coreset constructions that
  carry the streaming k-means approximation guarantees. §2 rules on each family
  and says which of the three reasons it fails for.
- **The approximation guarantees are not available at all in bounded state.**
  Guha's constant factor costs `O(n^ε)` memory, Ailon's `O(log k)` costs
  `O(log(k)·√(nk))`, and Liberty's opens `O(k log n log(W*/w*))` centres —
  something grows with `n` in every one of them (§3). A bounded-state online
  clusterer is a heuristic by necessity, not by choice, and the same applies to
  the GMM's step size: Cappé & Moulines' convergence theorem requires
  `Σγᵢ² < ∞`, which a damped window violates by construction.
- **Seven model classes pass and are prototyped**, ten designs between them:
  fixed-`k` EW k-means with spherical, Huber and fuzzy variants; online EM for a
  Gaussian mixture (spherical, diagonal, full); DP-means and its frozen-centre
  leader variant; DenStream-style micro-clusters with a checkpointed macro step;
  a self-organising map; growing neural gas; and ODAC, which clusters the
  *columns* rather than the rows. Every one of them holds every guarantee: chunk
  invariance is bit-exact at 1, 37, 1 000 and one chunk; a zero-weight row and a
  null feature leave the state bit-identical; leading zero-weight rows never
  poison it (§7.1).
- **The one real defect of sequential k-means is duplicate centres.** Under
  drift, two centres end up describing one component while a third component is
  unowned, and no k-means step can undo it. It hit **5 of 20** drifting streams,
  costing ~0.25 ARI on each. A **split–merge move on a slower schedule** — merge
  the pair whose separation is under half their combined radius, re-place the
  freed centre at the farthest row seen since the last checkpoint — cuts that to
  **1 of 20** and lifts the mean ARI from 0.925 to 0.982, at 0.6 moves per
  stream (§7.5). On the individual stream in §7.3 it is 0.767 → 1.000. It is
  O(k²) per checkpoint, deterministic, and never fires on static data. A looser
  threshold is worse than no move at all: at 0.8 it thrashes.
- **Seeding matters more than the algorithm, and the right rule depends on the
  outliers you expect.** Over 10 seeded streams: farthest-first is perfect on
  clean data (ARI 1.000, 0/10 misses) and catastrophic with 5% outliers
  (0.125, 10/10) because Gonzalez's rule picks the outliers by construction;
  the first-`k`-rows rule is the reverse (0.861 clean, 0.805 with outliers);
  Lloyd on a 500-row warm-up buffer is the best clean rule (1.000, 0/10) and
  middling with outliers (0.678). §7.2 has the table; §6.2 recommends
  buffer + Lloyd with the split–merge move as the recovery mechanism.
- **`n_eff`, `min_periods`, zero weight, null rows, the irregular clock and the
  damped window all carry over unchanged.** The one place clustering needs a new
  contract rule is the **output schema**: the plugin needs a static schema known
  from the spec alone, so `k` must be a spec parameter for the fixed-`k` models,
  and the variable-`k` models emit a monotone integer id plus a count, never a
  column per cluster (§8).
- **Being online costs almost nothing; choosing the wrong family costs
  everything.** On seven deliberately hard geometries (§7.8), one-pass
  sequential k-means matches a converged batch Lloyd's fit to within 0.04 ARI on
  six of seven, and the seventh gap closes with the split–merge move. But every
  k-means and every Gaussian mixture — batch ones included — scores **0.000** on
  concentric rings, where batch DBSCAN scores 1.000. The streaming constraint is
  not what limits these models; the k-means assumption is.
- **For a non-convex shape the answer is micro-clusters with the linkage macro
  step**, and it reaches the batch ceiling: 0.998 on two moons, 0.999 on three
  concentric rings and 0.998 on parallel bars, against DBSCAN's 1.000. It needs
  `macro_link` set to chain along the shape — above the largest spacing between
  neighbouring micro-clusters and below the gap between clusters, a window
  that `eps` widens at the price of more micro-clusters (§7.8) — not §6.5's
  default of 2.0, and a setting that chains costs it on blob-shaped data —
  0.537 on sheared Gaussians against k-means' 0.933. **No configuration wins everywhere**, which
  is Sesame's "no silver bullet" reproduced on our own models.
- **Labels are the hard part of the API, not the maths.** A batch learner refit
  on a rolling window scores ARI 0.07–0.26 over a segment and 0.99 *within* one
  refit block: the drop is entirely label churn between refits. Online models
  keep their ids stable by construction; the variable-`k` ones must allocate ids
  monotonically and never reuse them (§6.5).
- **The bar is `O(1)` memory in `n` and `O(n · parameters)` processing**, which
  admits a constant number of passes (§2). Everything here is single-pass
  anyway; §2.1 records what a second pass would buy — chiefly seeding from the
  whole stream instead of a prefix — and the two things standing in its way,
  neither of which is complexity: it would be lookahead under hard rule 2, and
  the expression plugin cannot express it.
- **The design worth the build decision is `micro`.** Seven reasons, each
  measured or cited above: it alone reaches the batch ceiling on the shapes
  that define the problem, where every k-means and GMM scores 0.000 (§7.8);
  being online is not what limits any model here — the family is (§7.8); it is
  DBSCAN over a weighted quantisation with a measured resolution rule, and the
  rule derives the threshold at the checkpoint instead of shipping a constant
  (§6.5, §7.8); it meets the contract as it stands — bounded, deterministic,
  chunk-invariant, out-of-sample labels, monotone ids (§7.1, §6.5); the field
  agrees it is the building block and its three recorded reservations —
  thresholds, fragmentation, cost — are answered by the derived window and by
  `O(M·p)` per row (§3, §7.7); its limits against DBSCAN are structural and
  stated, not measured away — no revision of an emitted label, no `eps` sweep
  on rows not kept, a resolution floor of `eps`, and a silent all-null output
  when the threshold is wrong at high `p` (§7.8, §12); and nobody has measured
  a per-row, predict-before-update label from this family (§3). It is not a
  universal clusterer — a setting that chains along a shape loses to `kmeans`
  on convex data — so the honest exposure is `kmeans` for convex data and
  `micro` for shapes, with the trade stated. The one DBSCAN-faithful design
  inside the bar, batch DBSCAN over a retained sample, is unmeasured (§12).
- **Nothing is in the crates.** §9 costs a Rust build; the decision is the
  user's. The narrowest useful build is one model, `kmeans`, with the
  split–merge move and the seeding buffer; `micro` is about the same again, on
  the shared summary `kmeans` needs. Merged to `main` 2026-09-04 as
  documentation and numpy prototypes.

---

## 1. The question, and the contract it has to meet

Can an online clusterer be a `polars-online` model? Concretely, every rule the
bank already enforces (`docs/PLAN.md` §2–§3, `CLAUDE.md`):

1. **`step(x, y, d_clock, weight)` with the output computed before the update.**
   Clustering has no `y`; the assignment of a row must be made with the state as
   it stood before that row is learned. That is the natural order anyway.
2. **Memory is O(state), never O(rows).** The state may be bounded by
   parameters (`k`, `max_clusters`, a warm-up buffer of fixed size) and by
   nothing else.
3. **Chunk invariance.** Feeding the stream as one chunk or a thousand must give
   bit-identical output. This is the rule that kills most of the published
   algorithms as written: they process a *batch* at a time.
4. **Deterministic**, including across thread counts. Any randomness must be a
   pure function of a spec seed.
5. **A damped window on an irregular clock**: `lam = 0.5^(d_clock/halflife)`
   applied per row, `halflife = inf` being no decay.
6. **`n_eff` means the same thing in every model** (rule 8): the accumulated
   weight before this row's update and before its own decay. `min_periods`
   gates the output on it.
7. **A zero-weight row is legal** (rule 9), including as the first row: advance
   the clock, learn nothing, still emit. Every division by an accumulated weight
   must be guarded.
8. **A null feature** skips the row (`docs/PLAN.md` §3's null policy).
9. **State is versioned msgpack**, loadable on macOS and Windows; `f64`
   throughout; no `unsafe` in `online-core`; the model knows nothing about
   Polars.
10. **A static output schema** derivable from the spec alone — the expression
    plugin has no other way to declare its output type.

Two of these are unusual for clustering and shape everything below: **3**
(published stream clusterers are batch-at-a-time almost without exception) and
**10** (the number of clusters cannot decide the schema, so a variable-`k`
model must emit an id, not a column per cluster).

---

## 2. Every family, and the verdict

The families are the union of river 0.26.1's `cluster` module, MOA's
`moa.clusterers`, scikit-learn's `cluster` and `mixture`, Spark MLlib, and the
algorithms tabulated by the two surveys read for §3.

**The bar is a complexity bar, and it is worth stating separately from §1's
contract**, because the two are often conflated: *memory bounded by parameters
(constant in `n`), and total processing linear in `n` times those parameters* —
`O(n·k·p)` is fine, `O(n²)` is not, and neither is any state that grows with the
row count. A **constant number of passes is admissible**, since `2n` is still
`O(n)`; §2.1 says what a second pass would buy and what it would cost. Each
rejection below carries a code for which part of the bar it fails:

| code | meaning |
|---|---|
| **R** | needs the rows back — the raw points, retained, so memory is `O(n)` |
| **Q** | quadratic — `O(n²)` processing, or an `n×n` matrix |
| **G** | state grows with `n`, even if sub-linearly |
| **X** | randomness on the per-row output path (breaks determinism, §1.4) |
| **P** | bounded in principle, but by a constant exponential in `p` |
| **S** | the output schema is not derivable from the spec (§1.10) |
| **T** | the state is not `f64` (§1.9) |
| **C** | *passes the complexity bar* — excluded by convention or measurement instead |

Verdicts:

| family | representative | verdict | why |
|---|---|---|---|
| sequential / mini-batch k-means | MacQueen'67, Bottou–Bengio'95, Sculley'10, Spark | **in** | EW mean per centre; O(k·p); the library's own accumulator |
| spherical k-means | Dhillon–Modha'01 | **in** | same, on unit vectors with cosine distance |
| fuzzy c-means, single pass | Bezdek'81 + Hore'07 | **in** | soft weights `u_j^m`, still an EW mean per centre |
| k-medians / k-medoids by coordinate | P² quantiles per centre | **medians in, not prototyped** | a per-coordinate streaming quantile per cluster is O(k·p) state and the library already has the machinery (`robust.rs`); **medoids out** — a medoid must be an actual row |
| online EM / Gaussian mixture | Cappé–Moulines'09, Neal–Hinton'98, Titterington'84 | **in** | expected sufficient statistics are additive; `EwCov` per component weighted by `w·r_k` |
| DP-means / leader / threshold | Kulis–Jordan'12, Hartigan'75 | **in, capped** | one distance test per row; needs `max_clusters` and an eviction rule to be O(state) |
| micro-clusters, damped | DenStream'06, DBSTREAM'16, CluStream'03 | **in, capped** | (weight, centre, radius) is `EwCov`'s triple; the fading function *is* our decay |
| grid / density grid | D-Stream'07 | **out (P)** | the occupied-cell count is bounded by `bins^p`, so it *is* a parameter bound and constant in `n` — but that constant is astronomical past `p ≈ 6`, and the cells actually held grow with `n` until saturation. Bounded in principle, useless in practice |
| hierarchical CF-tree | BIRCH'96, ClusTree'11 | **partly** | the CF triple is exactly our summary and the insertion is per-row and chunk-invariant, but the tree is **not memory-bounded** as implemented (`_birch.py`, no rebuild) — a capped flat set of micro-clusters is the same idea with a bound |
| coreset / streaming k-means with guarantees | Guha'03, Ailon'09, StreamKM++'12, BICO'13 | **out (G, X)** | the guarantee is bought with memory that grows in `n` — that *is* the complexity bar, not a preference (§3) — the construction is randomized, and the clustering happens at the end |
| online facility location | Liberty'16 (`liberty2016:206-214`) | **out (X, G)** | opens a centre with probability `min(D²/f, 1)` — randomness on the output path — and `O(k log n log(W*/w*))` centres, which grows with `n` |
| self-organising map | Kohonen'82 | **in** | fixed grid, each neuron an EW mean with a neighbourhood weight |
| growing neural gas | Fritzke'95 | **in, capped** | the paper's own stopping criterion is "net size or some performance measure" (`fritzke1995:153`), so a `max_nodes` cap is its option, not a violation — but growth then simply stops (§6.6); constant learning rates also make it a constant-gain model |
| ODAC — clustering the *variables* | Rodrigues–Gama'08 | **in** | one EW correlation matrix; a static per-column output |
| DBSCAN / OPTICS / HDBSCAN **on rows** | Ester'96, Ankerst'99, Campello'13 | **out (R, Q)** | a point's label depends on which other points lie within `ε`, so either the points are retained (`O(n)`) or every pair is evaluated (`O(n²)`). A second problem is independent of memory: density-connectivity is global, so a later arrival can promote noise to core or bridge two clusters into one, and correct output means *revising labels already emitted*. **Density clustering over bounded summaries is a different question and is in** — that is DenStream's whole design and §6.5's macro step, and §7.8 measures it reaching batch DBSCAN's accuracy on non-convex shapes (0.998 on two moons and 0.999 on three concentric rings, against 1.000) |
| spectral / affinity propagation / kernel k-means | Ng'01, Frey–Dueck'07 | **out (R, Q)** | an `n×n` affinity matrix, and an eigendecomposition or message passing over it |
| agglomerative **on rows** | Ward'63 and the linkage family | **out (R, Q)** | `O(n²)` distances over retained rows. Linkage over *summaries* is bounded and is in (§6.5's macro step is single linkage over `M ≤ max_micro` micro-clusters, `O(M²)` at a checkpoint) |
| sliding-window clustering | SL-KMeans'20 | **out (C)** | a fixed window `W` is `W·p` doubles — a *parameter* bound, constant in `n`, so it **passes the complexity bar**. It is excluded on two other grounds: the library's convention reads a retained window as `O(data)` (`BEYOND-O-STATE.md` excludes kNN for exactly this), and the damped window buys the same recency in `O(1)` without storing a row. Sesame's O5 also measures the sliding window as the least accurate of the three window models |
| projected / subspace, high-dimensional | HPStream'04, PreDeCon | **out for now (S)** | the retained dimension set is per-cluster and evolves, so neither the state layout nor the output schema is fixed by the spec. Not a complexity failure — a much bigger design |
| sequence / time-series clustering | DTW-based | **out (R, Q)** | needs the series retained, and a DTW alignment is `O(L²)` per pair — it is not expressible as a decayed mean of anything |
| categorical / text | k-modes, TextClust | **out (T)** | the state is modes, token tables or tries, not `f64` |
| co-clustering / consensus | Dhillon'01, Strehl–Ghosh'02 | **out (R, Q)** | co-clustering needs the full `n×p` matrix resident; consensus needs every base model's label vector over all rows |
| clusterwise regression | gated `ewridge` instances | **later** | a natural follow-on once a clusterer exists: the gate is a clusterer, the experts are models the bank already has |

**Two reasons cover almost every "out"**: it needs the rows back (R), or it is
quadratic (Q) — and the two travel together, because what you would do with the
retained rows is compare them pairwise. The remaining rejections are one each of
state that grows with `n` (coresets), randomness on the output path (Liberty),
a bound exponential in `p` (grids), a schema that is not fixed by the spec
(subspace), a non-`f64` state (categorical), and one — sliding windows — that
passes the complexity bar outright and is excluded on other grounds.

**The R rejections are not a statement about density clustering.** DBSCAN *on
rows* is out; DBSCAN *over bounded summaries* is in, is what DenStream does, and
is what §6.5 implements as single linkage over capped micro-clusters. Same for
linkage: `O(n²)` over rows, `O(M²)` over `M ≤ max_micro` summaries at a
checkpoint. The published literature reached this conclusion first and said so
plainly — a naive approach "would be to maintain all the points in memory…
clustered by the DBSCAN algorithm", but "it is unrealistic to provide such a
precise result, because in a streaming environment the memory is limited", so
DenStream "resort[s] to an approximate result" over summaries
(`cao2006:199-217`).

### 2.1 What a second pass would change

A constant number of passes is inside the bar. It is not currently used, and the
prototypes are all single-pass, but four things would come within reach and one
of them addresses the largest weakness measured anywhere in this document:

- **Seeding from the whole stream, not a prefix.** §7.2 shows seeding dominating
  every other choice, and §6.3 shows a *bigger* warm-up buffer scoring worse
  (0.944 against 1.000) because its extra rows are older. That is an artefact of
  having only a prefix to look at. Two passes removes it: pass 1 builds capped
  micro-clusters over the whole stream, weighted Lloyd runs over those `M`
  summaries in memory (`O(M)`, not `O(n)`), and the fit pass starts from seeds
  that have seen everything. This is BIRCH's two-phase shape with a hard cap
  instead of a growing tree.
- **Approximate k-medoids.** Pass 1 gives centres; pass 2 keeps, per centre, the
  nearest actual row seen — `O(M·p)`. Exact medoids stay out.
- **Davies–Bouldin exactly** (§8 notes it needs a pass with the final centroids).
- **A fit-then-label mode**: the macro step runs once after pass 1 and pass 2
  labels every row from a frozen model, so labels are never revised and `k` is
  known before any output is emitted.

**Two things stand in the way, and neither is about complexity.**

*Hard rule 2, out-of-sample by construction.* A second pass over the same rows
means the model labelling row `i` has seen rows `i+1…n`. In a backtest that is
lookahead — the failure this library exists to prevent. Note that a two-phase
mode over *different* data already exists and leaks nothing: fit, `save_state`,
then load and score. Passing twice over the *same* data is the new thing, and
initialisation quality is its only real justification. If it is ever built, the
leak must be named in the output, not buried in a parameter.

*The expression plugin cannot do it at all.* It receives its column once. The
CLI and `po.run` over a file can re-scan cheaply; the IO plugin
(`python/polars_online/_frame.py`) would have to re-execute its input plan
inside the source, which doubles any upstream compute and is not obviously sound
under polars' semantics for a plan used twice in one query. So a two-pass model
cannot be an ordinary `ModelKind` in the bank — it needs its own entry point.
That is an architectural decision, not a modelling one, and it is why nothing
here assumes a second pass.

---

## 3. What the field has built

Read directly, not from memory. Sources under `.cache/research/`.

**Sesame** (Wang et al., SIGMOD 2023 — an empirical evaluation of stream
clustering) decomposes the design space into four aspects: the summarizing
structure (hierarchical: CF-tree, coreset tree, dependency tree; partitional:
micro-clusters, grids, adaptive slots), the window model (landmark, sliding,
damped), outlier detection, and offline refinement. Its findings that bear on a
design here: there is no silver bullet across data sets (O1); hierarchical
structures are faster and partitional ones more accurate (O2); the **sliding
window is the least accurate** window model (O5); outlier handling is worth
≥ 8% accuracy (O11); a timer-driven maintenance schedule is preferable (O14);
and — the one that saves the most work — **offline refinement is often
unnecessary and sometimes harmful** (O15). Too aggressive a decay hurts (O10).

**DenStream** (Cao et al., SDM 2006) is the closest published algorithm to what
this library's decay already is. Its fading function is `f(t) = 2^{−λt}`
(`cao2006:180-190`) — ours with `λ = 1/halflife`. A micro-cluster is
`{CF1, CF2, w}` with centre `CF1/w` and radius `sqrt(|CF2|/w − |CF1/w|²)`
(Definition 3.4, `cao2006:255-270`); potential ones have `w ≥ βµ` and outlier
ones `w < βµ` (Definition 3.5). Its merging rule (Algorithm 1,
`cao2006:297-313`) is: try the nearest potential micro-cluster, accept if the
merged radius stays ≤ `ε`; else the nearest outlier micro-cluster, promoting it
when its weight crosses `βµ`; else open a new outlier micro-cluster. Pruning
runs every `Tp = ⌈(1/λ)·log(βµ/(βµ−1))⌉` (eq. 4.1, `cao2006:355-358`) and drops
an outlier micro-cluster whose weight is below
`ξ(t_c,t_o) = (2^{−λ(t_c−t_o+Tp)} − 1)/(2^{−λTp} − 1)` (eq. 4.2,
`cao2006:389-390`). Its offline part is DBSCAN over the potential
micro-clusters, on demand.

**What the field makes of the micro-cluster family.** It is the consensus
building block, not a contender: every toolkit ships it (MOA's `WithDBSCAN`,
river's `DenStream` and `DBSTREAM`, R `stream`'s `DSC_DenStream`,
`DSC_DBSTREAM` and `DSC_DStream`); its fading function is what Hahsler credits
as the origin of the damped window — "introduced first for DenStream"
(`hahsler2017:169-174`); and in Sesame's decomposition micro-clusters are the
most accurate summarizing structure (O2, O4 — `wang2023:639-651, 705-711`),
with DenStream's buffered outlier handling worth ≥ 8 % accuracy (O11,
`:928-936`). Hahsler's own worked comparison is the clearest statement of the
utility: on a noisy four-cluster benchmark the two density-based methods score
cRand 0.782 and 0.795 against 0.581 and 0.550 for sampling and sliding-window
k-means (`hahsler2017:2523-2525`), and on two drifting clusters that cross they
are equal-best, "easily explained by the fact that these two algorithms cannot
detect noise" (`:2595-2598`). The reservations the field states are the ones
§7.8 measures, and none of them is about being online:

- **The thresholds.** Zubaroğlu's open problems put "density threshold,
  distance threshold" among parameters that are "very sensitive to the input
  data" and need "expert knowledge" (`zubaroglu2021:1183-1187, 1221-1226`),
  and call multi-density clusters — "different density thresholds ...
  different distance thresholds" — "another open problem by itself"
  (`:1213-1215`). DenStream's own recipe for `ε` is to **run batch DBSCAN on
  the initial points** and take `ε = α·d_min` from the nearest pair of points
  in different clusters (`cao2006:788-798`): the threshold is derived from a
  warm-up buffer, not chosen. §7.8 reaches the same conclusion for
  `macro_link`.
- **Fragmentation.** In Hahsler's example the density-based methods "identify
  the two denser clusters correctly, but split the lower density clusters into
  multiple pieces" — 7 and 6 macro-clusters for 4 true — and "do not assign
  some points which are not noise points" (`hahsler2017:2180-2183, 2523-2525`).
  That is `varied` at 0.690 and the `macro_link` trade in §7.8.
- **The macro step is contested — under a metric that cannot see what it
  does.** Sesame's O15, offline refinement is "unnecessary" and on two
  workloads harmful, verified by switching DenStream's refine off with no
  change (`wang2023:1035-1046`), is measured in purity with every algorithm
  tuned so its cluster count is "close to the ground truth" (`:558-561`) — a
  setup in which merging has nothing left to do and can only lower the score.
  Under ARI the macro step is the whole difference between 0.343 and 0.998 on
  `moons` (§7.8). O15 is a finding about purity; it does not transfer.
- **Cost.** Partitional structures run ~70 % slower than hierarchical ones
  (O2), the outlier buffer costs throughput (O12, `wang2023:937-949`), the
  damped window is the slowest window model (O7, `:759-762`), and every
  structure slows with `p` (O16, `:1197-1202`). Zubaroğlu relays DenStream
  reaching 800 micro-clusters and 650× CEDAS's time at 3 000 dimensions
  (`zubaroglu2021:864-872`). Here that is `O(max_micro · p)` per row, and the
  cap is the cost control.
- **Nobody has measured it the way it would ship here.** The published
  numbers are checkpoint or horizon evaluations; river's `DenStream.predict_one`
  runs the full DBSCAN per call with an inverted expansion condition, MOA's
  `getVotesForInstance` returns `null`, and no implementation reads a clock
  (see "Implementations" below). Per-row predict-before-update labels from
  this family are unmeasured in the literature.
- **Shapes are asserted more than benchmarked.** Sesame's workloads (FCT,
  KDD99, Sensor, Insects) contain no non-convex geometry; the arbitrary-shape
  claim rests on the DenStream paper's own figures (`cao2006:822-830`) and the
  batch-DBSCAN intuition. §7.8 is the only ARI measurement of it in this
  investigation.

Not read: Carnein, Assenmacher & Trautmann 2017 (Computing Frontiers) and
Carnein & Trautmann 2019 (BISE) — the other large empirical comparison and
survey — and Hahsler & Bolaños 2016 (DBSTREAM, TKDE); all three were paywalled
or bot-blocked when fetched. Their verdicts are not represented here.

**CluStream** (Aggarwal et al., VLDB 2003) defines the micro-cluster as
`(CF2ˣ, CF1ˣ, CF2ᵗ, CF1ᵗ, n)` (Definition 1, `aggarwal2003:220-245`) — BIRCH's
CF triple plus timestamp moments. It is a landmark-window algorithm: recency
comes from the timestamp moments (a micro-cluster is deleted when the
"relevance stamp", the `m/(2n)`-th percentile of its arrival times under a
normal assumption, is too old, `aggarwal2003:520-545`), and history from
*snapshots on disk* at a pyramidal time frame. Two parts of it are
incompatible here: the snapshots are O(data) on disk, and the initial `q`
micro-clusters come from k-means over the first `InitNumber` points buffered on
disk (`aggarwal2003:455-465`) — our warm-up buffer is the same idea with a
fixed size.

**BIRCH** (Zhang et al., SIGMOD 1996) contributes the object everything else
reuses: `CF = (N, LS, SS)` (Definition 4.1, `zhang1996:276-285`) and the
additivity theorem (4.1, `zhang1996:285-292`), which is why any of this can be
merged over chunks at all. Its CF-tree is height-balanced with a branching
factor and a radius threshold `T` (`zhang1996:305-340`); the tree is rebuilt at
a larger `T` when memory runs out, which scikit-learn's port does not implement
(so `Birch.partial_fit` is unbounded, `_birch.py:568-585`).

**DP-means** (Kulis & Jordan, ICML 2012) is the small-variance limit of a
Dirichlet-process mixture: a point farther than `λ` from every centre starts a
cluster of its own (Algorithm 1, `kulis2012:296-310`). The paper is explicit
that, unlike k-means, "the DP-means algorithm depends on the order in which
data points are processed" (`kulis2012:288-291`) — for us that order dependence
is the semantics, not a defect, because the stream has an order. Their
heuristic for `λ` is farthest-first: add the point maximising the distance to
the chosen set `k` times and take the last maximum (`kulis2012:653-660`).

**Online EM** (Cappé & Moulines, JRSS-B 2009) gives the update the GMM here
uses: replace the E-step by a stochastic approximation of the expected
sufficient statistic and keep the M-step,
`ŝ_{n+1} = ŝ_n + γ_{n+1}(s̄(Y_{n+1}; θ̂_n) − ŝ_n)`, `θ̂_{n+1} = θ̄(ŝ_{n+1})`
(eq. 15, `cappe2009:290-296`). They note the fixed-step version goes back to
Nowlan (1991) (`cappe2009:296-300`) and that the construction of the feasible
set reflects that "the M-step is unambiguous only when a sufficient number of
observations have been gathered" (`cappe2009:268-275`) — which is exactly why
the prototype runs a few EM iterations on the warm-up buffer before going
online (§6.3). Their asymptotic analysis assumes `γ_n = γ_0 n^{−α}` with
`α ∈ (1/2, 1]` (`cappe2009:593`); a fixed exponential window is the constant-γ
regime, which trades asymptotic efficiency for tracking — the trade this
library makes everywhere.

**Bottou & Bengio** (NIPS 1995) is the justification for the `1/n_k` step:
online k-means is online gradient descent with a prototype-dependent learning
rate `1/n_k` (`bottou1995:128-141`), and that rate is the Newton rate for this
objective (`bottou1995:153-190`) — so it needs no tuning, and there is no step
size to schedule.

**The approximation guarantees, and what they cost.** Three papers give
streaming k-means/k-median a provable factor, and each pays for it in a
different currency — read from their own statements:

| result | approximation | what grows with `n` |
|---|---|---|
| Guha et al. 2003 | constant factor, one pass | **memory `O(n^ε)`**, time `O(n^{1+ε})` (`guha2003:79-86`) |
| Ailon et al. 2009 | `O(log k)` | **memory `O(log(k)·√(nk))`** times the log of the input size (`ailon2009:303-310`) |
| Liberty et al. 2016 | `O(1)` expected cost, semi-online | **`O(k log n log(W*/w*))` clusters** in expectation; fully online, the factor itself "degrades by a `log n`-factor" (`liberty2016:84-99`) |

**Not one of them holds fixed `O(k)` state and a constant factor as `n → ∞`.**
Something always grows: the memory, the number of centres, or the factor. Rule 2
therefore does not merely make coresets inconvenient here — it is incompatible
with the guarantees, and any bounded-state clusterer this library ships is a
heuristic by necessity, not by choice. That is worth saying plainly in the
user-facing docs if one ever ships.

**The step size is the honest caveat on the GMM.** Cappé & Moulines' convergence
result for the online EM recursion (Theorem 5, `cappe2009:533-538`) assumes
`0 < γᵢ < 1`, `Σγᵢ = ∞` **and `Σγᵢ² < ∞`** — satisfied by `γᵢ = γ₀i^{−α}` with
`α ∈ (1/2, 1]` (`cappe2009:541-543`). A damped window is the constant-`γ` regime:
it satisfies the first two and **violates the third**, so the theorem does not
cover it. That is not a defect peculiar to clustering — it is the same trade
`ewridge` and every other model here makes, a stationary tracking estimator
instead of a convergent one — but it means the GMM's decayed step has no
convergence proof behind it, only the measurements in §7.

**Implementations.** In river, DenStream, DBSTREAM, CluStream and STREAMKMeans
are per-row; ODAC (`river/cluster/odac.py`) keeps a Pearson accumulator per
variable pair per leaf and tests the structure every `n_min` rows with a
Hoeffding bound `e = sqrt(ln(1/confidence)/(2n))` (`odac.py:497-521`), splitting
on the widest pair when `(d1 − d2 > e or tau > e)` and
`(d1 − d0)·|d1 + d0 − 2·avg| > e` (`odac.py:566-580`), and folding a leaf back
into its parent when `d1 − parent.d1 > max(parent.e, e)` (`odac.py:582-590`).
In scikit-learn, `MiniBatchKMeans.partial_fit` labels the whole chunk against
the chunk-start centres (`_kmeans.py:1633`) and then applies the exact
cumulative weighted mean (`_k_means_minibatch.pyx:87-104`); its low-count
reassignment can never fire at chunk size one because the cap is
`int(0.5·n_chunk)` (`_kmeans.py:1655-1658`); there is no forgetting anywhere.
Spark's `StreamingKMeansModel.update` labels against the pre-batch centres
(`StreamingKMeans.scala:83`), discounts every weight by `a`
(`StreamingKMeans.scala:114`) and applies `(a·n·c + S)/(a·n + m)`
(`StreamingKMeans.scala:117-126`) — the same arithmetic as here, but once per
micro-batch, so neither of its two decay modes is a per-row exponential window.
**None of the four libraries is chunk-invariant except scikit-learn's `Birch`,
and that one is unbounded.**

Two more things the implementations reveal, and both are load-bearing here.

**Nobody's fading function reads a clock.** In river, DenStream's `timestamp`
advances once per `stream_speed` calls to `learn_one`
(`river/cluster/denstream.py:313-317`) and CluStream's once per call
(`river/cluster/clustream.py:209-210`); in MOA, DenStream's advances once per
`processingSpeed` rows (`clusterers/denstream/WithDBSCAN.java:145-153`). So
`2^{−λt}` everywhere in the published implementations means "λ per *row*", never
"λ per unit of a time column". The library's `d_clock` — a real, irregular
clock supplied as a column — is not something to port from any of them; it is
the thing this library already has and they do not.

**A per-row label is barely a first-class idea in these libraries.** MOA has no
prediction path at all for its clusterers: `getVotesForInstance` throws
`UnsupportedOperationException` in CluStream
(`clusterers/clustream/Clustream.java:352-354`) and returns `null` in DenStream
(`clusterers/denstream/WithDBSCAN.java:320-322`). river does expose
`predict_one`, but for DenStream it means "run the offline DBSCAN now and tell
me which cluster this point lands in" — the comment says so outright, "this
function handles the case when a clustering request arrives"
(`river/cluster/denstream.py:351-353`) — so it is a full macro step per call,
not a cheap per-row output. A stream clusterer in this literature produces a
*clustering on demand*, not a label per row; rule 1's out-of-sample output is
something to design, not to copy, which is why §6.2 states the per-row order
explicitly and why §6.5's macro step runs on a checkpoint schedule rather than
on every read.

---

## 4. The earlier exclusion, reassessed

`docs/ENHANCEMENTS.md` §4 excluded clustering in one line: "not regression on
ordered streams (PLAN §4.6 scopes classification to binary)". That is a
statement about the *task*, not about the contract, and it bundles three
unrelated things — clustering, naive Bayes and multiclass softmax — under one
reason.

| the objection | true of | in these designs |
|---|---|---|
| "not regression" | all of it | true, and the bank already ships one unsupervised model: `ew_cov` (`spec.rs:872-877`, "targets mirror its columns for plumbing"). Clustering is the same shape: no target, a per-row output, a state that summarises the feature stream. |
| implied: unbounded state | grids, CF-trees, coresets | bounded by `k` or `max_clusters` and a fixed warm-up buffer, preallocated (§6.7) |
| implied: nondeterministic | k-means++ seeding, Liberty's rule, coresets | seeding is a pure function of a spec seed over a fixed buffer, and the recommended rule (Lloyd on the buffer) is deterministic outright; nothing random is on the per-row path |
| implied: no clock semantics | landmark and sliding-window algorithms | the damped window *is* DenStream's fading function; every summary is an EW mean with the library's `lam` |

What survives is the cost, which is the same cost boosting had (§9): a second
family with its own parameters, docs, tests and state schema — plus one thing
boosting did not have, an output that is a **label**, whose stability across
time is a user-visible API property (§6.5).

---

## 5. The design space, as a spectrum

Six levels, from "a batch clusterer used honestly" to "everything online",
scored against the contract. Numbers are §7's, on the drifting mixture.

| level | what changes online | state | chunk-invariant | out-of-sample | decay | labels stable | measured ARI (static / drifting) |
|---|---|---|---|---|---|---|---|
| **0. window refit** — Lloyd on the last `W` rows every `R` rows, label the next `R` | everything, in bursts | `O(W·p)` rows | yes (row-count schedule) | yes | window only | **no** — a fresh labelling per refit | 0.07–0.26 across a segment; 0.993 within one block |
| **1. mini-batch, no decay** — scikit-learn `MiniBatchKMeans.partial_fit` | centres | `O(k·p)` | **no** (`_kmeans.py:1633`) | yes | none | yes | 1.000 / 1.000 (but it cannot track a moving component; see §7.3's tracking column) |
| **2. sequential EW k-means** — §6.2 without the structural move | centres | `O(k·p)` | yes | yes | yes | yes | 1.000 / 0.767 |
| **2b. + split–merge on a slower clock** | centres and their assignment to components | `O(k·p)` | yes | yes | yes | yes (an id may change meaning at a move) | **1.000 / 1.000** (0.982 mean over 20 drifting seeds against 0.925) |
| **3. soft: online EM** — a Gaussian mixture | means, covariances, weights | `O(k·p)` diag, `O(k·p²)` full | yes | yes | yes | yes | 1.000 / 0.991 (diag) |
| **4. variable `k`: micro-clusters or DP-means** | the number of clusters too | `O(M·p)`, `M` capped | yes | yes | yes | ids monotone, meaning drifts | 0.997 / 0.984 (micro), 0.81 / 0.66 (DP-means) |

Level 0 is the honest baseline: it is what a user would do today with
scikit-learn and a `group_by_dynamic`, and its problem is not accuracy but
**label churn** — 0.99 within a refit block, 0.07–0.26 across a segment that
spans several. **Level 2b is the recommendation**, with level 3 as the variant
for elliptical components and level 4 for "I do not know `k`".

---

## 6. The recommended designs, precisely

### 6.1 The shared object

Every model here is built from one accumulator, which is `EwCov`'s update with
an assignment in front of it. For a cluster holding weight `n_j`, centre `c_j`
and mean squared radius `R_j`, a row `x` of effective weight `we` under decay
`lam`:

```text
n_j'  = lam · n_j + we
a     = lam · n_j / n_j'      b = we / n_j'          (a + b = 1)
delta = x − c_j
c_j' = c_j + b · delta
R_j' = a · R_j + a · b · ‖delta‖²
```

with the whole update skipped when `n_j' = 0` (rule 9: a zero-weight first row
gives `0/0` in both `a` and `b`). Clusters not receiving the row still decay:
`n_i' = lam · n_i`. `R` is the trace form of `EwCov`'s centred co-moments, so
`sqrt(R_j)` is the RMS distance of the cluster's members to its centre — the
same quantity DenStream calls the radius and BIRCH the CF radius.

`we` is where the variants differ: `w` for hard k-means; `w · u_j^m` for fuzzy
memberships; `w · r_j` (the responsibility) for the GMM; `w · h(j, bmu)` for the
SOM's neighbourhood; and `w · min(1, c/(d/sqrt(R_j)))` when Huber-weighting
against outliers.

**Standardisation goes in the metric, never in the coordinates.** Distances are
`d² = Σ_i (x_i − c_i)²/v_i` with `v` the EW feature variances *before the row*
(E24's rule); the state stays in raw units. Rescaling the coordinates instead —
storing `z = (x − m)/s` — slides every stored centre out from under itself as
the moments move, and measures **worse than not standardising at all** (§10).

### 6.2 `kmeans`: fixed `k`, the recommendation

**Scope first: this is the right model when the clusters are blob-shaped**, and
it is measurably the wrong one when they are not — §7.8 puts it at 0.000 ARI on
concentric rings, the same score batch Lloyd's gets. What it buys over batch
Lloyd's is not accuracy but bounded state, a stable label, and the ability to
follow a moving population.

State: `k` centres, `k` weights, `k` radii, plus the shared feature moments and
a warm-up buffer of `warm_rows × p`.

Per row, in order:
1. `lam = 0.5^(d_clock/halflife)`; decay every cluster weight.
2. Read `n_eff = Σ_j n_j` **before** the update and before this row's decay.
3. If the model is seeded and `n_eff ≥ min_periods`, emit: the nearest centre's
   index, the distance to it, and the distance to the second nearest (from
   which a simplified silhouette costs O(k) and nothing more).
4. If `w = 0` or the row has a null: stop. Nothing is learned; the state is
   bit-identical (§7.1).
5. If not yet seeded, append the row to the buffer; when it is full, seed (§6.3)
   and stop.
6. Otherwise apply §6.1 to the nearest centre.
7. Every `update_every` learned rows, a checkpoint (a no-op at 1); every
   `sm_every` learned rows, the split–merge move.

**The split–merge move** is the one non-obvious piece and the one that earns
its place. At a checkpoint, compute the pairwise centre distances (O(k²)) and
the ratio `d_ij / (sqrt(R_i) + sqrt(R_j))`. If the smallest such ratio is below
`split_merge` (0.5 measured best), the two centres describe one component: merge
`j` into `i` by the mean form, and re-place `j` at the farthest row seen since
the last checkpoint, giving it half the weight of that row's cluster. That is
the split half of ISODATA on EW summaries. It must run on **its own, slower
clock**: at every row it thrashes (19 413 moves, ARI 0.49), at every 100 rows it
fires twice and reaches ARI 1.000 (§7.5).

A separate, weaker rule handles a genuinely dead cluster (`n_j` below
`dead_frac · n_eff/k`): move it to the farthest row seen. It is kept because it
is cheap, but the split–merge move subsumes it in every case measured.

Variants, all the same code path: `spherical` (cosine distance on unit vectors,
centres renormalised at the checkpoint), `fuzzifier > 1` (fuzzy c-means
memberships `u_j ∝ d_j^{−2/(m−1)}`, every cluster updated with weight `u_j`),
`huber_c` (a row farther than `c` radii is down-weighted).

### 6.3 Seeding

The cold start is the single largest source of variance in the results, and no
library surveyed supplies a rule that works from row one (scikit-learn requires
`≥ k` rows in the first chunk, `_kmeans.py:876-879`; Spark has no data-driven
seeding at all, `StreamingKMeans.scala:234-263`).

Four rules were measured (§7.2):

| rule | deterministic | clean data | 5% outliers | overlapping (3σ) |
|---|---|---|---|---|
| first `k` distinct rows (MacQueen) | yes | 0.861, 5/10 miss | **0.805**, 8/10 | **0.897**, 2/10 |
| farthest-first over a 500-row buffer (Gonzalez) | yes | **1.000**, 0/10 | 0.125, 10/10 | 0.904, 2/10 |
| k-means++ over the buffer, seeded | yes, given the seed | 0.972, 1/10 | 0.652, 9/10 | 0.848, 4/10 |
| Lloyd (k-means++ then 10 iterations) over the buffer | yes, given the seed | **1.000**, 0/10 | 0.678, 9/10 | 0.874, 3/10 |

**Recommendation: buffer 500 rows, seed with Lloyd, and rely on the split–merge
move for recovery.** Farthest-first is not the default despite being perfect on
clean data: it picks the extremes by construction, so a single outlier in the
buffer destroys it. A bigger buffer is not better (2000 rows scored *worse* than
500 on clean data, 0.944 vs 1.000, because the extra rows arrive after the
components have drifted).

Seeding replays the buffer once, frozen: each buffered row is assigned to its
seed and the centres are recomputed from those assignments, with each row
carrying the weight it has decayed to by the time the buffer fills. A seed that
attracts nothing keeps weight zero and is replaced outright by the first row it
wins — MacQueen's `1/n_k` with `n_k = 0`, and the same behaviour scikit-learn
gets from initialising `_counts` to zero (`_kmeans.py:2297`).

### 6.4 `gmm`: online EM

State: `k` means, `k` weights, and per component either a scalar, a diagonal or
a full `p×p` centred second moment — `EwCov` per component, weighted by
`w · r_j` with `r_j` the responsibility computed from the parameters **before**
the row. That is Cappé–Moulines eq. 15 with `γ` replaced by the library's
`b = we/n_j'`, which is the decayed `1/n` step.

A variance floor of `var_floor × (the global EW feature variance)` is added to
every component's covariance at evaluation time. It is not optional: a component
that collapses onto one row has zero covariance and infinite likelihood, and the
floor is the standard remedy (scikit-learn's `reg_covar`).

The M-step needs enough rows to be well defined (`cappe2009:268-275`), so
seeding runs `warm_iters` frozen EM iterations over the warm-up buffer:
responsibilities from the current parameters, moments re-accumulated from zero.
Without it every component sits on the buffer's global mean and the model never
separates (measured: ARI 0.000 with a single online pass over the buffer, 1.000
with three frozen iterations).

Cost: `O(k·p)` per row for diagonal, `O(k·p²)` for full (a solve per component
per row) — measured 31 µs/row versus 60 µs/row in the numpy prototype. Diagonal
was as accurate as full in every experiment here and beat it under drift
(0.991 vs 0.789), because a full covariance is `p²` parameters per component
estimated from a decayed window.

### 6.5 Variable `k`, and what a label means

Two designs, both capped:

**`dpmeans`**: a row farther than `radius` from every centre opens a cluster at
itself (Kulis–Jordan Algorithm 1); otherwise it joins the nearest and moves it.
Ids are allocated monotonically. At `max_clusters` the lightest cluster is
evicted. Optionally a prune checkpoint drops clusters below a weight.

**`micro`**: DenStream's online part on EW summaries — try the nearest potential
micro-cluster, accept if the merged radius stays ≤ `eps`; else the nearest
outlier micro-cluster, promoting it at `beta_mu`; else open a new one. Pruning
uses their `ξ` (eq. 4.2) with `λ = 1/halflife`, on a **learned-row schedule**
rather than their clock schedule, so that a chunking of the stream cannot move
it. An optional macro step at the same checkpoints does single linkage over the
potential micro-clusters (centres within `macro_link · eps`), O(M²), and gives
each a macro label = the smallest id in its component. Sesame's O15 says the
offline step is often unnecessary; here it is what makes the output usable as a
*cluster* label rather than a micro-cluster id, and it is worth its cost: under
drift, 94 micro-clusters collapse to 28 distinct labels.

**`macro_link` is the most consequential parameter in this document and the
default of 2.0 is wrong.** It must be read against the *nearest-neighbour
spacing between potential micro-clusters*, which `eps` sets: §7.8 measures that
spacing at 1.84·eps median and 2.21·eps at p90, so a threshold of 2.0 links only
half the adjacent pairs and fragments every shape it is given. A threshold must
clear the p90 spacing — 2.5 or 3.0 here — to chain reliably, and one much above
it bridges genuinely separate clusters instead. **The fix is not a better
constant**: the macro checkpoint already computes the pairwise distances, so the
threshold should be derived from the observed spacing (say 1.5× its p90) and
`macro_link` retained only as an override.

**Label semantics is the API decision.** Three rules make a variable-`k` output
honest:
1. Ids are monotone and never reused. An evicted id never comes back.
2. A row's label is the id it *would* be absorbed by, computed before the
   update — so the first row of a new cluster is labelled with the new id.
3. The count of live clusters is part of the output, so a downstream consumer
   can see churn without diffing labels.

Even so, a variable-`k` model scores badly on ARI over a long segment
(DP-means 0.81 static, 0.40 under drift) *while scoring purity 1.000*: it is
not mixing components, it is splitting one component across several ids over
time. That is inherent, and the fixed-`k` models are the ones to use when a
stable label matters.

### 6.6 `som`, `gng`, `odac`

**`som`** is a fixed `rows × cols` grid; each neuron is an EW mean and a row of
weight `w` reaches neuron `j` with weight `w · exp(−g²/2σ²)` for grid distance
`g`. Kohonen's shrinking neighbourhood is replaced by a fixed `σ`, which makes
it a stationary estimator under decay rather than an annealing one. It is
`EWKMeans` with `K = rows·cols` centres and a coupling; it measured within
0.01 ARI of k-means on every stream and recovered from a regime change in 1 000
rows without any structural move, because the neighbourhood drags neurons along.

**`gng`** (Fritzke 1995) is bounded by `max_nodes` — which is the paper's own
suggested stopping criterion, "net size or some performance measure"
(`fritzke1995:153`), though its stated advantage is precisely *not* having to
pre-specify one (`fritzke1995:196-197`). On an endless stream that cap does not
bound a converged model; it stops growth wherever the stream happens to be. It
inserts a node every
`insert_every` learned rows between the highest-error node and its
highest-error neighbour; edges age out; connected components are relabelled at
the insertion checkpoints. Its constant steps `eps_b`, `eps_n` make it a
constant-gain model like `sgd`, not an averaging one, and it shows: it is the
only model here that fails badly with outliers (ARI 0.18–0.34), because a
constant step chases them.

**`odac`** clusters the **columns**, not the rows, which makes it the only
model here whose output is a fixed-length vector known from the spec: one label
per feature. Dissimilarity is `rnomc(a,b) = sqrt((1 − corr(a,b))/2)` from the EW
correlation matrix the library already computes in `EwCov`; the split and
aggregate tests are river's (`odac.py:566-590`) on a Hoeffding bound. It found
three correlated blocks in eight variables and split off a variable that
switched blocks mid-stream (§7.6). The damped window replaces river's
reset-on-aggregate, which is a simplification, not a compromise.

### 6.7 Memory, exactly

| model | doubles of state | for `k=5, p=4` |
|---|---|---|
| `kmeans` | `k(p + 2) + 2p + 4` (+ `k(p+2)` when `update_every > 1`) | 42 |
| `gmm` | `k(p + 1) + k·{1, p, p²} + 2p + 4` | 57 diag, 117 full |
| `dpmeans` | `M(p + 3) + 2p + 4` | 362 at `M = 50` |
| `micro` | `M(p + 5) + 2p + 4` | 1 812 at `M = 200` |
| `som` | `K(p + 1) + 2p + 4` | 57 at 3×3 |
| `gng` | `M(p + 5) + 2p + 4` | 282 at `M = 30` |
| `odac` | `p² + 2p + 4 + 8p` | 60 |

Plus the warm-up buffer, `warm_rows × p` doubles, which is freed at seeding.
All of it is preallocated at construction, which is the existing rule.

### 6.8 Parallelism

The same trick that made the boosted trees parallel applies, and for the same
reason: with `update_every > 1` the centres are frozen between checkpoints, so
the rows of a segment are independent and their contributions are plain weighted
sums (`S_j`, `W_j`, `Q_j` per cluster) that add across threads in any order.
The checkpoint schedule counts learned rows, never chunks, so it is invariant.
Measured cost of the coarser grain: **none worth reporting** — ARI 0.998 at
`update_every` 1, 10, 100 and 1 000 (§7.5). Unlike boosting, though, the per-row
work here is `O(k·p)` and tiny, so the parallel form is a convenience rather
than a necessity.

---

## 7. The measurements

Data: `k = 5` Gaussian components in `p = 4` dimensions, pairwise centre
distance ≥ 6σ, 20 000 rows, an irregular clock (exponential gaps, mean 1),
halflife 3 000. Variants: a random walk of 0.02σ per row on every centre
("drifting"); 5% uniform outliers over a box three times the mixture's extent;
unequal mixing weights 1:2:3:4:5 with spreads 0.5–1.5σ; a regime change at
`n/2` where one component dies and another is born elsewhere. ARI is computed
per segment over the clean rows; "track" is the mean distance from each live
true centre to the nearest model centre at the last row.

### 7.1 The guarantees hold (`guarantees`)

Every model, every check:

```
model                   chunks  rerun   w=0  null      drop  relabel  lead  finite
kmeans first              PASS   PASS  PASS  PASS  4.26e-14     0.0%  PASS    PASS
kmeans k++ warm           PASS   PASS  PASS  PASS  1.28e-13     0.0%  PASS    PASS
kmeans lloyd warm         PASS   PASS  PASS  PASS  1.28e-13     0.0%  PASS    PASS
kmeans reseed             PASS   PASS  PASS  PASS  2.66e-14     0.0%  PASS    PASS
kmeans split-merge        PASS   PASS  PASS  PASS  2.66e-14     0.0%  PASS    PASS
kmeans huber c=2          PASS   PASS  PASS  PASS  1.28e-13     0.0%  PASS    PASS
fuzzy m=2                 PASS   PASS  PASS  PASS  2.58e-14     0.0%  PASS    PASS
gmm diag                  PASS   PASS  PASS  PASS  1.42e-14     0.0%  PASS    PASS
gmm full                  PASS   PASS  PASS  PASS  1.85e-13     0.0%  PASS    PASS
dpmeans r=4s              PASS   PASS  PASS  PASS  3.55e-15     0.0%  PASS    PASS
micro eps=1.5s            PASS   PASS  PASS  PASS  7.11e-15     0.0%  PASS    PASS
som 3x3                   PASS   PASS  PASS  PASS  3.55e-15     0.0%  PASS    PASS
gng 30                    PASS   PASS  PASS  PASS  0.00e+00     0.0%  PASS    PASS
odac                      PASS   PASS  PASS  PASS  0.00e+00     0.0%  PASS    PASS
kmeans batch 50           PASS   PASS  PASS  PASS  1.07e-14     0.0%  PASS    PASS
kmeans std                PASS   PASS  PASS  PASS  1.95e-14     0.0%  PASS    PASS
```

- **chunks**: outputs bit-identical at 1, 37, 1 000 and one chunk.
- **rerun**: bit-identical on a second pass with a fresh model.
- **w=0 / null**: 100 such rows at `d_clock = 0` leave the canonical state
  bit-identical; the zero-weight rows still emit, the null rows do not.
- **drop**: zero-weight rows versus deleting them and folding the clock gap into
  the next row — max absolute difference over the float outputs, and the share
  of rows given a different label. Equality is up to the floating-point
  associativity of `0.5^(dt/halflife)` only. `n_eff` is excluded and is
  *expected* to differ: it is read before the row's own decay (rule 8), so
  folding a gap into a row moves that row's `n_eff` by the folded decay.
- **lead**: 100 leading zero-weight rows and 100 leading null rows leave no NaN
  anywhere in the state.
- **finite**: state finite after a stream with outliers, drift, a regime change
  and zero-weight rows.

### 7.2 Seeding (`seeding`)

Ten data seeds, ARI on the last quarter, "miss" = runs with purity below 0.9.
The table is in §6.3. The two findings worth repeating: **farthest-first is the
best clean rule and the worst outlier rule** (1.000 → 0.125), and **a bigger
warm-up buffer is not better** (Lloyd on 2 000 rows scores 0.944 against 1.000
on 500, because the buffer's rows are older).

### 7.3 Against the baselines (`baselines`)

Static mixture, then drifting, ARI per segment:

```
model                      [2k,5k)    [5k,10k)   [10k,15k)   [15k,20k)  purity  track live  seen
--- static
kmeans first                 0.724       0.724       0.712       0.723   0.801  2.732    5     5
kmeans lloyd warm            1.000       1.000       1.000       1.000   1.000  0.042    5     5
kmeans split-merge           1.000       1.000       1.000       1.000   1.000  0.042    5     5
gmm diag                     1.000       1.000       1.000       1.000   1.000  0.042    5     5
dpmeans r=4s                 0.808       0.796       0.801       0.736   1.000  0.511    9    14
micro eps=1.5s               0.997       1.000       1.000       1.000   1.000  1.300   34     5
som 3x3                      0.999       0.999       0.999       0.999   1.000  1.372    9     7
gng 30                       1.000       1.000       1.000       1.000   1.000  0.754   30     5
batch lloyd (in-sample)      1.000       1.000       1.000       1.000   1.000    nan    5     5
batch lloyd rolling          0.262       0.162       0.100       0.070   0.322    nan    5     5
  ...within a 500-row block  ARI 0.993 (the same labelling only holds between refits)
sklearn minibatch 256        1.000       1.000       1.000       1.000   1.000    nan    5     5
--- drifting (0.02 sigma / row)
kmeans lloyd warm            0.767       0.724       0.728       0.782   0.804  3.115    5     5
kmeans split-merge           1.000       1.000       1.000       1.000   1.000  1.662    5     5
kmeans huber c=2             0.767       0.724       0.727       0.781   0.804  3.115    5     5
fuzzy m=2                    0.791       0.788       0.797       0.786   0.804  6.324    5     5
gmm diag                     0.991       1.000       1.000       1.000   1.000  1.667    5     5
gmm full                     0.789       0.790       0.816       0.864   0.932  2.131    5     5
dpmeans r=4s                 0.659       0.487       0.392       0.405   1.000  0.768   31    35
micro eps=1.5s               0.984       0.987       0.991       0.842   1.000  1.032   94    28
som 3x3                      0.988       1.000       0.999       0.999   1.000  2.206    9     6
gng 30                       0.649       1.000       0.999       1.000   1.000  0.711   30     6
batch lloyd rolling          0.172       0.114       0.143       0.107   0.388    nan    5     5
  ...within a 500-row block  ARI 0.994
sklearn minibatch 256        1.000       1.000       0.999       1.000   1.000    nan    5     5
```

Readings:

- **The rolling batch refit is the baseline to beat, and it loses on labels, not
  on clustering.** 0.99 within a block, 0.07–0.26 across a segment. This is the
  strongest argument for an online clusterer in this library and it has nothing
  to do with accuracy.
- **`MiniBatchKMeans` scores 1.000 everywhere** — and it should, because these
  streams never require forgetting to *label* correctly. What it cannot do is
  track: with no decay its centres freeze (`_kmeans.py`, `_counts` only grows),
  and its 1.000 comes from components that stay separable, not from following
  them. The tracking column is where decay shows up (§7.4).
- **Sequential k-means loses ~0.24 ARI under drift, and the split–merge move
  recovers all of it** (0.767 → 1.000, tracking 3.12 → 1.66).
- **A full covariance is worse than a diagonal one under drift** (0.789 vs
  0.991) at double the cost.
- **Fuzzy memberships help with outliers and hurt tracking** (§7.4): 1.000 with
  5% outliers where hard k-means gets 0.782, but tracking 6.32 — a soft
  assignment pulls every centre toward every row, so the centres sit inside the
  convex hull of the components rather than on them.

### 7.4 Decay (`decay`)

Mean over five data seeds, ARI on the last quarter and tracking at the last row:

| halflife | kmeans | kmeans split-merge | gmm diag |
|---|---|---|---|
| ∞ | 0.943 / track 3.57 | 0.997 / 3.32 | 0.938 / 3.59 |
| 20 000 | 0.950 / 3.31 | 0.941 / 3.31 | 0.952 / 3.31 |
| 5 000 | 0.999 / 2.40 | 0.999 / 2.40 | 0.998 / 2.40 |
| 3 000 | 0.942 / 2.14 | 1.000 / 1.97 | 0.952 / 2.17 |
| 1 000 | 0.948 / 1.52 | 1.000 / 1.13 | 0.951 / 1.49 |
| 300 | 1.000 / 0.61 | 1.000 / 0.61 | 1.000 / 0.61 |
| 100 | 0.943 / 0.85 | 1.000 / 0.42 | 0.956 / 0.84 |

(drifting stream; the static and regime-change tables are in the log.) Two
things to read here. **Tracking improves monotonically as the halflife shortens
— down to 0.42 at halflife 100 — while ARI does not**, because ARI on
well-separated components is insensitive to where exactly the centre sits.
Sesame's O10 ("too fast a decay hurts") did not reproduce down to halflife 100
on these streams; it would at a halflife short enough to starve a component.
And **the plain model's ARI is non-monotone in the halflife** (0.999 at 5 000,
0.942 at 3 000, 0.948 at 1 000) — that is the duplicate-centre failure firing on
some seeds and not others. The split–merge column is flat at 1.000 from 3 000
down. **A fragile failure that a structural move removes is worth removing, even
though the average looks tolerable.**

On static data the split–merge model is at ARI 1.000 for every finite halflife,
and its tracking degrades gently as the window shortens (0.04 at halflife 5 000,
0.23 at 100) — the estimator gets noisier with a shorter window, exactly as
`ewridge` does. The plain model is at 0.886–0.944 with 1–2 misses in 5 at *every*
halflife including infinity, which is the duplicate-centre failure again and has
nothing to do with decay.

On the regime-change stream nothing recovers at a long halflife (5 misses in 5
from ∞ down to 3 000, for both models): a centre stuck on a dead component keeps
its weight forever, so there is nothing for a structural move to reclaim. Only
at halflife 300–100 does the dead cluster fade enough to be reused (0.911 with
the move, 0.854–0.897 without). **Forgetting is what makes a structural move
possible; the move is what makes forgetting sufficient.**

### 7.5 Knobs (`knobs`)

**The split–merge threshold and cadence**, over 20 drifting streams. The plain
model collapses two centres onto one component on 5 of them; "miss" counts those
(purity below 0.9 on the last quarter), "moves" is the mean per stream:

| threshold | cadence | ARI | track | miss | moves |
|---|---|---|---|---|---|
| off | — | 0.925 | 2.14 | 5 | 0.0 |
| 0.3 | any | 0.925 | 2.14 | 5 | 0.0 |
| **0.5** | every row | 0.971 | 1.85 | 2 | 1.0 |
| **0.5** | **every 100** | **0.982** | **1.76** | **1** | **0.6** |
| 0.5 | every 500 | 0.963 | 1.89 | 2 | 0.9 |
| 0.5 | every 2 000 | 0.973 | 1.90 | 2 | 0.5 |
| 0.8 | every row | 0.858 | 1.83 | 4 | 3 901 |
| 0.8 | every 100 | 0.950 | 1.82 | 3 | 29.8 |
| 1.0 | every row | 0.803 | 1.96 | 6 | 5 850 |
| 1.0 | every 2 000 | 0.918 | 2.18 | 5 | 2.7 |

Three readings. **0.3 is too tight to ever fire** — two centres one third of
their combined radius apart is already a pathology past saving. **0.5 is the
threshold**, and it is not sensitive to the cadence between 1 and 2 000 rows;
100 is the best of them. **0.8 and above thrash**: the move becomes a
perpetual-motion machine (3 901 moves in 20 000 rows) and ends up worse than
leaving the duplicate alone. The failure the move exists to fix is not fully
eliminated — 1 stream in 20 still misses — so it is a mitigation, not a proof.

**`update_every`** (the data-parallel form): ARI 0.998–1.000 at 1, 10, 100 and
1 000. The coarser grain costs nothing measurable.

**DP-means radius** (in σ), on the drifting stream — clusters found against a
true `k` of 5:

| radius | clusters | evicted | ARI | purity |
|---|---|---|---|---|
| 1.5σ | 50 (capped) | 11 674 | 0.299 | 0.999 |
| 2σ | 50 (capped) | 5 764 | 0.388 | 0.999 |
| 3σ | 50 (capped) | 316 | 0.329 | 0.999 |
| 4σ | 33 | 0 | 0.373 | 0.999 |
| 6σ | 6 | 0 | 0.960 | 0.999 |

The radius must be set to the *cluster* scale, not the noise scale: at 6σ it
finds 6 clusters for 5 components and scores 0.960; anywhere below that it
shatters them. Kulis & Jordan's farthest-first heuristic for `λ`
(`kulis2012:653-660`) is the principled way to pick it and needs a buffer, which
the seeding path already has.

**Micro-cluster `eps` and `beta_mu`**:

| eps | beta_mu | potential MCs | macro clusters | ARI | evicted | pruned |
|---|---|---|---|---|---|---|
| 1.0σ | 3 | 195 | 5 | 0.687 | 4 124 | 1 040 |
| 1.0σ | 10 | 112 | 5 | 0.681 | 0 | 3 767 |
| 1.5σ | 3 | 115 | 4 | 0.969 | 0 | 621 |
| 1.5σ | 10 | 67 | 5 | 0.794 | 0 | 966 |
| 2.0σ | 3 | 35 | 5 | 0.993 | 0 | 95 |
| 2.0σ | 10 | 30 | 5 | 0.994 | 0 | 151 |

The macro step recovers the true count from every setting; ARI tracks how
coarsely the micro-clusters tile the components. At `eps = 1σ` the cap binds and
4 124 evictions follow — a bounded state with a badly chosen `eps` degrades by
thrashing, which is the failure mode to document.

**Standardisation**: with feature 0 scaled by 100, ARI 0.566 unstandardised
and **0.926** with the metric scaled. Scaling the coordinates instead gives
0.370 (§10).

### 7.6 Outliers, regime change, ODAC

With 5% uniform outliers on a drifting stream, ARI on clean rows and the
precision/recall of each model's own outlier signal over the second half:

| model | ARI (last segment) | flag precision | flag recall |
|---|---|---|---|
| kmeans lloyd | 0.788 | 1.00 | 0.64 |
| kmeans split-merge | 0.787 | 1.00 | 0.83 |
| kmeans huber c=2 | 0.788 | 1.00 | 0.72 |
| fuzzy m=2 | **1.000** | 1.00 | 0.54 |
| gmm diag | 0.792 | 0.06 | 0.00 |
| dpmeans | 0.564 | 0.91 | **1.00** |
| micro | 0.977 | 0.40 | **1.00** |
| gng | 0.342 | — | — |

The k-means "flag" is `distance > 3·sqrt(R_j)` — an exact analogue of E21's
`resid_z`, free from state the model already holds, and it is precise at 1.00
with recall 0.64–0.83. The variable-`k` models detect every outlier (each
becomes its own cluster) at the price of precision. The GMM's Mahalanobis flag
is useless here because its covariances inflate to swallow the outliers — the
same reason its ARI does not degrade.

Regime change at `n/2`, rows to recover ARI > 0.9 in a 500-row block:

| model | ARI segments | recovered after |
|---|---|---|
| kmeans lloyd | 0.738 0.719 0.718 0.733 | never |
| **kmeans split-merge** | 1.000 1.000 0.925 1.000 | **1 500 rows** |
| gmm diag | 0.774 0.758 0.745 0.772 | never |
| micro | 0.998 0.998 0.985 0.997 | **0 rows** |
| som 3×3 | 0.936 0.996 0.958 0.997 | 1 000 rows |
| gng 30 | 1.000 1.000 0.884 1.000 | 1 500 rows |

Fixed-`k` models need a structural move to survive a component dying and
another being born; a variable-`k` model handles it for free, which is its one
decisive advantage.

**Correction (task 23, 2026-09-05).** The `kmeans split-merge` row above is a
seeding artefact: the prototype's single k-means++ start had put two seeds in
one blob, and the merge freed one of them. Seeded with `lloyd`'s restarts the
plain model scores 1.000 1.000 0.926 1.000 on this stream with no move at all.
The move's real case is a *stranded* centre, and what it does there is
measured in PLAN §11a (2026-09-05): a freed centre lands on the new blob in
one move, the freeing itself waits `log2(1/dead_frac)` halflives.

**ODAC** on eight variables in three correlated blocks, with variable 7
switching to block 0 at `n/2`:

```
row  1500 leaves 3  labels [3 3 3 2 2 4 4 4]
row  3000 leaves 3  labels [3 3 3 2 2 4 4 4]
row  4500 leaves 4  labels [3 3 3 2 2 6 6 5]
row  5999 leaves 4  labels [3 3 3 2 2 6 6 5]
splits 3 merges 0
```

It finds the three blocks and then splits variable 7 off within 1 500 rows of
the switch. The damped window is doing the forgetting that river does by
resetting a leaf's accumulators.

### 7.7 Cost

Prototype timings are numpy, one row at a time, and are only meaningful
relative to one another:

| model | state (f64) | µs/row | per-row work |
|---|---|---|---|
| kmeans | 42 | 23 | `O(k·p)` distances |
| fuzzy | 42 | 27 | `O(k·p)` |
| gmm diag | 57 | 31 | `O(k·p)` |
| gmm full | 117 | 60 | `O(k·p²)`, a solve per component |
| dpmeans | 362 | 7 | `O(M·p)` |
| micro | 1 812 | 16 | `O(M·p)`; macro `O(M²)` per checkpoint |
| som 3×3 | 57 | 13 | `O(K·p + K)` |
| gng 30 | 282 | 17 | `O(M·p + E)` |
| odac | 60 | 7 | `O(p²)`; tests `O(p²)` per checkpoint |

Against `ewridge`'s `O(k²)` solve per row, `kmeans` at `O(k·p)` is **cheaper
than the models the library already ships**. That is the one performance
statement worth making before a Rust build measures it.

---

### 7.8 Hard geometries: what these models are actually worth (`hard`)

Everything above runs on isotropic Gaussian blobs, which is precisely the shape
k-means is optimal for — so §7.3 can say the models are not broken and nothing
more. These seven streams are chosen to break them: two non-convex (`moons`,
`rings`), two that defeat a spherical distance (`aniso` — sheared Gaussians;
`elongated` — three long parallel bars), one with densities an order of
magnitude apart (`varied`), and two asking what `p` does (`highdim20`,
`highdim50`). 6 000 rows, i.i.d. and shuffled, regular clock, no drift, so the
only question asked is clustering quality. ARI over the second half, mean ± sd
over three seeds. Every model standardizes in the metric and is handed the true
`k` where it takes one. **The batch rows are in-sample and best-of-a-sweep — a
ceiling, not a competitor.**

**The headline is that streaming costs almost nothing.** Online sequential
k-means against batch Lloyd's, on the same standardized data:

| geometry | online `kmeans` | batch Lloyd's | difference |
|---|---|---|---|
| moons | 0.493 | 0.495 | −0.002 |
| rings | 0.000 | 0.000 | 0 |
| aniso | 0.933 | 0.934 | −0.001 |
| elongated | 0.448 | 0.488 | −0.040 |
| varied | 0.822 | 0.822 | 0 |
| highdim20 | 0.903 ± 0.137 | 1.000 | −0.097, and **split–merge closes it** (1.000) |
| highdim50 | 1.000 | 1.000 | 0 |

One pass with bounded state, seeded from a 500-row prefix, matches a converged
batch fit to within noise on six of seven — and the seventh is seeding variance
that the split–merge move removes. **The penalty for being online is not the
problem. The penalty for choosing the wrong family is.**

**And the k-means family is the wrong family for a non-convex shape** — batch or
online, it makes no difference. On `rings` every k-means and every Gaussian
mixture scores **0.000**, including batch Lloyd's and batch full-covariance EM,
while batch DBSCAN and single linkage score 1.000. That is a model-class
failure, not a streaming one.

**The one online model that handles a shape is micro-clusters with the linkage
macro step, and it reaches the batch ceiling** — but only when `macro_link` is
set to chain along the shape rather than to sit inside it:

| config | moons | rings | aniso | elongated | varied | hd20 | hd50 |
|---|---|---|---|---|---|---|---|
| `kmeans` split–merge | 0.493 | 0.000 | 0.933 | 0.448 | 0.822 | **1.000** | **1.000** |
| `gmm full` | 0.507 | 0.000 | **0.938** | 0.492 | **0.989** | 0.910 | **1.000** |
| `micro` eps=0.4 link=2.0 | 0.343 | 0.069 | 0.905 | 0.537 | 0.919 | **1.000** | **1.000** |
| `micro` eps=0.1 link=3.0 | **0.998** | 0.734 | 0.537 | **0.998** | 0.690 | *none* | *none* |
| `micro` eps=0.07 link=4.0 | — | **0.999** | — | — | — | — | — |
| batch DBSCAN (ceiling) | 1.000 | 1.000 | 0.572 | 1.000 | 0.978 | 1.000 | 1.000 |
| batch single linkage | 1.000 | 1.000 | 0.572 | 0.858 | 0.000 | 1.000 | 1.000 |

Five things to take from this.

**1. `macro_link` is the knob that decides whether the macro step follows a
shape or ignores it**, and §6.5's default of 2.0 sits exactly on the wrong side
of it — measured, not inferred. On `moons`, the nearest-neighbour spacing
between potential micro-clusters is **1.84·eps at the median and 2.21·eps at the
90th percentile** (`eps = 0.1·√p`, 45 micro-clusters); at `eps = 0.2·√p` it is
1.96 and 2.44. A threshold of 2.0 therefore sits *at the median spacing* and
links 25 of the adjacent pairs, severing the chain about every other step — the
model reports 24 fragments, ARI 0.139. At 2.5 it links 56 pairs and at 3.0, 62;
the chain holds and the model reports 3 clusters, ARI 0.998.

The rule that falls out is precise: **the threshold must clear the *p90* spacing,
not the median.** Since both quantities are available at the macro checkpoint —
the pairwise distance matrix is already being computed there — the principled
default is to derive `macro_link` from the observed spacing rather than to ship
a constant.

`rings` — the canonical DBSCAN shape, and the one number in the table that
could have meant a resolution limit — confirms it is the same threshold miss.
Labelling the potential micro-clusters by their nearest ring radius: at
`eps = 0.07·√p` the within-ring spacing has p90 2.47·eps and maximum 2.71·eps,
while the nearest pair *across* rings is 6.47·eps apart. Any `macro_link` in
(2.71, 6.47) therefore resolves the three rings, and it does — **0.999 ± 0.001
at 4.0 and 6.0, against batch DBSCAN's 1.000**; 0.720 with 18 fragments at
3.0, and 0.000 at 8.0 when the rings bridge. `eps` sets the width of the
window: (2.99, 4.72) at `0.1·√p`, still hit by 4.0, and (2.96, 3.26) at
`0.15·√p`, too narrow for any value tried. Smaller `eps` buys a wider window
with more micro-clusters — 109, 71, 43 — which is the family's memory–
resolution trade stated as a number: **a shape resolves when the gap between
clusters exceeds about three micro-cluster spacings, and `eps` decides how many
summaries that costs.** 109 two-dimensional summaries were enough here.

**2. It is a genuine trade, not a free win.** The same `link=3.0` that scores
0.998 on `moons` scores 0.537 on `aniso` and 0.690 on `varied`, because a
threshold loose enough to chain along a shape is loose enough to bridge two
nearby blobs. No configuration in the table wins everywhere — Sesame's O1 ("no
silver bullet", §3) reproduced with these models on these streams.

**3. The batch shape-aware methods are not a universal ceiling either.** DBSCAN
scores 0.572 on `aniso`; single linkage scores 0.000 on `varied`, chaining
straight through the sparse cluster, and 0.858 ± 0.200 on `elongated`. Each
family has a shape it cannot see.

**4. A bad threshold produces silence, not a bad answer.** `micro` at
`eps = 0.1·√p` emits **zero clusters** at `p = 20` and `p = 50` — every row
opens an outlier micro-cluster, none reaches `beta_mu`, the cap thrashes, and
the output is all-null rather than wrong. At `p = 50`, `eps = 0.2·√p` gives ARI
0.023 while the same setting at `p = 20` gives 0.882. This is the worst failure
mode in the whole investigation because it is invisible: a fixed threshold does
not degrade gracefully with `p`, it falls off a cliff. Any shipped model needs
either a threshold expressed in units of `√p` (which is how this table is
parameterised) or a hard diagnostic when the potential-micro-cluster set stays
empty.

**5. High dimensions are otherwise a non-event.** At `p = 20` and `p = 50` with
five well-separated components, every fixed-`k` model scores 1.000. There is no
distance-concentration problem at this separation; the only casualties are the
threshold models, and only through the threshold.

Two smaller observations. **`som` has high purity and low ARI everywhere** (0.986
purity, 0.266 ARI on `moons`): a 3×3 grid quantises 2 clusters into 9 cells,
each pure. It is a quantiser, and it needs a macro step over the grid before it
is a clusterer. **`gng` is the worst model on every hard geometry** (0.000
moons, 0.029 rings, 0.002 varied) — the constant-gain result of §7.6 again, now
confirmed on shape as well as on outliers. It should not be in a recommended
set. And **`gmm full` emits `slogdet` divide-by-zero warnings at `p = 50`**,
where a full covariance is 2 500 parameters per component: the variance floor
keeps the answer correct (ARI 1.000) but not the conditioning.

---

## 8. The spec and the output shape

This is where clustering needs decisions the existing models did not.

**No targets.** `Spec::validate` requires `targets` to be non-empty
(`spec.rs:833-834`), and separately rejects a feature that is also a target as
a leak — with `ew_cov` exempted *by design*, "its 'targets' mirror its columns
for plumbing" and "it predicts nothing" (`spec.rs:871-875`). A clustering spec
takes exactly that exemption. `spec.rs`'s `coef_fields` (`:1403`) and
`output_index` (`:1455`) already special-case models whose output is not
`pred`/`resid` per target, so the mechanism exists; a clustering model adds one
more arm to each.

**A static schema.** For a fixed-`k` model the output struct is:

| field | type | meaning |
|---|---|---|
| `cluster` | `i32` | the assigned cluster, `null` before seeding or under `min_periods` |
| `dist` | `f64` | distance to that centre, in the model's metric |
| `dist2` | `f64` | distance to the second-nearest centre (a simplified silhouette costs O(k) from these two) |
| `n_eff` | `f64` | rule 8, unchanged |
| `coef` | `list[list[f64]]` | the centres, `k × p`, null except on `coef_every` rows and the last row of a chunk — the existing `coef` plumbing, unchanged |
| `membership` | `list[f64]` | soft models only, length `k` |

For a variable-`k` model, `cluster` is a monotone id, and `n_clusters`
(`i32`) replaces the fixed shape; `coef` carries the live centres and is
therefore a ragged list — which the `coef` field already is.

**`k` is a spec parameter** for the fixed-`k` models, which is what makes the
schema static. There is no "choose `k` from the data" mode, and there cannot be
one behind the plugin.

**Parameters**, mapped onto names the library already uses:
`halflife`, `min_periods`, the per-model `standardize` flag (`spec.rs:358` and
its siblings — but read as a metric here, §6.1), `coef_every`, plus per model
`k`, `warm_rows`, `seed_rule`, `seed`, `update_every`, `split_merge`,
`sm_every`, and for the variable-`k` models `radius` / `eps`, `beta_mu`,
`max_clusters`, `prune_every`.

**Metrics.** ARI needs only the contingency table and is streamable
(`_supervised.py:262-269`); Calinski–Harabasz needs `(n, Σx, Σ‖x‖²)` per
cluster, which is exactly the state; Davies–Bouldin needs a second pass with the
final centroids; the true silhouette needs all pairwise distances and is out.
`po.eval` would gain an EW SSQ and the simplified silhouette, both O(k).

---

## 9. What building it in Rust would take

- `online-core`: `cluster/kmeans.rs` (+ `gmm.rs`, `micro.rs` if wanted) and a
  shared `cluster/summary.rs` holding §6.1's accumulator; ~400 lines for
  `kmeans` plus tests, and it reuses `EwCov` rather than reimplementing it.
  Arrays preallocated at construction; `f64`; no `unsafe`.
- Per row: `k` distance evaluations (`k·p` multiply-adds), one mean-form update
  (`p` adds), and a `k`-way decay. Cheaper than `ewridge`'s solve.
- State: a new `ModelState::KMeans` variant; bump `SCHEMA_VERSION` and keep the
  previous loader (rule 5). Sizes in §6.7 — 42 doubles for `k=5, p=4`, an order
  of magnitude *smaller* than `ewridge` with the same `p`.
- Plumbing: every place in `docs/EXTENDING.md` — `ModelKind::KINDS`, `KINDS`,
  `AnyModel` / `dispatch!` / `build_one` / `combos`, `bank.rs`'s output index
  and coef fields, `_spec.py`, `_kwargs.py`, `_expr.py`, `api_surface.txt`, the
  sweep lists, `test_model_registry`, the golden bank, the README list and the
  CHANGELOG. The registry tests name each omission.
- Tests: `model_contract.rs` for `n_eff` and zero-weight rows (the contract
  recursion is model-agnostic and applies unchanged); the chunk-invariance and
  determinism tests; a numpy oracle (`scripts/clustering_proto.py` is one);
  a cross-OS state test. Two new invariants worth their own tests: **ids are
  never reused**, and **a split–merge move never changes the number of live
  clusters**.
- Rule 12: nothing new is linked.

Effort: `kmeans` alone is about the size of `holt`; the plumbing is a day by the
`EXTENDING.md` list. The GMM and micro-clusters are each roughly the same again.

---

## 10. Ideas that did not survive measurement

1. **Standardising the coordinates.** Storing `z = (x − m)/s` and clustering in
   that space is the obvious reading of E24, and it is wrong: `m` and `s` move
   every row, so every stored centre silently refers to a different space than
   the one the next row is measured in. Measured on a stream whose first feature
   is scaled by 100: 0.370 ARI against 0.566 unstandardised. Putting the same
   scaling in the *metric* and keeping the state in raw units gives 0.926.
2. **The split–merge move on the per-row clock.** Correct in principle, useless
   in practice: 19 413 moves and ARI 0.49. A structural move needs a slower
   clock than the parameter update (§7.5) — the same lesson `grow_every` taught
   the boosting prototype.
3. **A dead-cluster rule alone.** Moving the lightest cluster when it falls
   below a fraction of the mean weight never fired on the failure it was meant
   to fix, because the failing centre is not light — it is a *duplicate*, and
   both copies carry real weight. Redundancy, not weight, is the right test.
4. **Seeding a GMM with one online pass over the buffer.** ARI 0.000: every
   component stays at the buffer's global mean because the first responsibilities
   are uniform and the mean-form update pulls all of them the same way. Three
   frozen EM iterations over the buffer fix it (1.000). Cappé & Moulines say why
   (`cappe2009:268-275`): the M-step is not well defined until enough rows are in.
5. **Farthest-first seeding as the default.** Perfect on clean data, 0.125 with
   5% outliers — Gonzalez's rule selects the extremes by construction.
6. **A full covariance per component.** Double the cost, worse under drift
   (0.789 vs 0.991) — `p²` parameters per component estimated inside a decayed
   window is too many.
7. **Growing neural gas as a general clusterer.** Elegant and bounded, but its
   constant learning rates make it a constant-gain model: with 5% outliers it
   scores 0.18–0.34 ARI while every mean-form model stays above 0.78.
8. **A bigger warm-up buffer.** 2 000 rows is worse than 500 (0.944 vs 1.000):
   the extra rows are older, and the components have moved.
9. **`macro_link = 2.0` as the micro-cluster default.** It sits *at* the median
   nearest-neighbour spacing between potential micro-clusters (measured at
   1.84·eps), so it links half the adjacent pairs and fragments every shape it
   is given: 0.139 on two moons where 3.0 gives 0.998 (§7.8). Picked because it
   looked safe; it is the worst of both worlds, and the replacement should be
   derived from the spacing rather than guessed again.
10. **A distance threshold in absolute units.** `eps` and `radius` fixed in
    standardized units mean something different at every `p`, and the failure at
    `p = 20` is not a bad clustering but **no clustering at all** — every row
    opens an outlier micro-cluster, none is ever promoted, and the output is
    all-null. Quoting every threshold as `c·√p` fixes it (§7.8).
11. **Growing neural gas, finally.** Bad with outliers (§7.6) *and* worst on
    every hard geometry (§7.8: 0.000 moons, 0.029 rings, 0.002 varied). Two
    independent failures from the same cause — constant gain — is enough. It
    stays prototyped for the record and out of any recommendation.
12. **The SOM as a clusterer.** Purity 0.986 with ARI 0.266 on two moons: a 3×3
    grid quantises two clusters into nine pure cells. It is a quantiser, and
    would need a macro step over the grid to be a clusterer.

---

## 11. Reproducing the numbers

```
uv run python scripts/clustering_experiments.py all                      # without the sklearn rows
uv run --with scikit-learn python scripts/clustering_experiments.py all  # with them
```

Experiments: `guarantees`, `baselines`, `seeding`, `decay`, `outliers`,
`regime`, `knobs`, `cost`, `hard`. `hard` needs scikit-learn for its batch
DBSCAN, single-linkage and full-covariance-EM ceilings; without it those rows
are skipped and the online rows still run. Nothing is added to `pyproject.toml` or the lock file
— the overlay is ephemeral. The whole suite takes about five minutes in pure
numpy; `decay`, `seeding` and `hard` are the slow ones (they sweep over data
seeds).

Sources (all under the gitignored `.cache/research/`): fifteen papers as PDFs
with text extractions; river 0.26.1 at `b50439f2`, MOA at `f0c284da`,
scikit-learn 1.9.0 and Spark's `StreamingKMeans.scala`; and the notes files with
every claim carrying a `file:line`, from which §3 was written and against which
it was re-checked.

---

## 12. Open questions

- **Whether to build any of it.** Nothing here presumes it. §9 costs the
  narrowest build (`kmeans` with the split–merge move and the seeding buffer).
- **Which family to expose, given that none dominates.** §7.8 shows the choice
  is the user's shape assumption, not ours. Either ship one model and document
  what it cannot see, or ship two (`kmeans` for blobs, `micro` for shapes) and
  make the trade explicit. Shipping only `kmeans` and calling it "clustering"
  would be the misleading option.
- **DBSCAN over a retained sample is the one DBSCAN-faithful design inside
  the bar, and it is unmeasured.** A fixed sample of `m` rows (`m·p` doubles,
  constant in `n`, so it passes §2's bar the way a sliding window does) with
  batch DBSCAN run over it at each checkpoint and rows labelled by their
  nearest core point is *exactly* DBSCAN at sample resolution — density
  semantics, border points and all — where micro-clusters are DBSCAN over a
  weighted quantisation (§7.8 shows the two agree to 0.001 on `rings` once
  the threshold sits in its window). It is Hahsler's `DSC_Sample` plus a
  macro step. Three things keep it out of the prototypes: the library's
  convention reads retained rows as `O(data)` (§2, sliding-window row), the
  sampling must be deterministic in the row counter to stay chunk-invariant
  (§1 rule 3 — a seeded reservoir is not), and every checkpoint costs
  `O(m²)` or an index. Whether the convention bends for a parameter-bounded
  sample is the user's call; if it does, this is the first thing to measure
  against `micro`.
- **Only synthetic data so far.** The next measurement should be the Binance
  intraday data `tests/data.py` already downloads — clusters of minutes by
  their return/volume/spread profile, scored by stability rather than by a
  ground truth that does not exist.
- **`k`-selection is out of scope behind the plugin** (a static schema needs a
  fixed `k`), but the bank could run several `k` at once and expose an EW SSQ
  per model — an elbow computed by the user rather than by us.
- **Whether to spend a second pass** (§2.1), and if so on initialisation only
  (keeping per-row outputs out-of-sample w.r.t. learning, with the seeds
  carrying a named leak) or on an explicitly in-sample fit-then-label mode. Left
  open deliberately on 2026-09-04; the prototypes stay single-pass until it is
  settled.
- **No convergence theory covers the constant-step regime** these models run in
  (§3). The measurements say they track; nothing says they converge, and a
  counter-example (a stream where a damped-window GMM oscillates instead of
  settling) has not been looked for. It would be worth constructing one.
- **Clusterwise regression** — a gate over `ewridge` instances — is the
  interesting thing a clusterer unlocks in a *regression* library, and is not
  investigated here.
- **The variable-`k` label contract** (ids monotone, never reused, meaning
  drifting) is the part most likely to surprise a user, and deserves a README
  paragraph before any such model ships.
