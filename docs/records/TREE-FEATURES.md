# What this library adds to XGBoost and LightGBM: designs, measured

**Status as of 2026-11-05: design and measurement for PLAN task 324, v2
(v1, then the self-review of §9 and the measurements it asked for);
nothing is built.** The question (the user, 2026-11-05): what can this
library add, at high impact, to the batch tree libraries without rewriting
them, choosing ideas that will not be rethought later. Each candidate was
prototyped in Python (numpy, Polars, the bank as it is, XGBoost 3.4.1,
LightGBM 4.7.0) on public data and on data that plants the effect, and is
ranked here by the gain it measured. The prototypes are kept as
`scripts/tree_features/` (§11 runs them). [BOOSTED-TREES.md](../BOOSTED-TREES.md)
is the earlier question, an online tree model inside the library, and
[GBDT-ADAPTER.md](../GBDT-ADAPTER.md) the one that built `po.gbdt`.

---

## 0. The short version

| rank | item | measured gain | cost to build | surface | risk |
|---:|---|---|---|---|---|
| 1 | **Target statistics of a categorical**, decayed, read before the row's label, labels folded in only once known | real data: arrival delay MAE 24.25 → 22.47 (−7%) with a 30-day half-life; → 20.92 (−14%) with short half-lives beside it; delayed-flight AUC 0.640 → 0.746; a cap of 1,024 levels costs nothing. Planted: RMSE 1.50 → 1.17, where every batch encoder stays at 1.45 to 1.55 | first, fix the `with_windows` defect of §6.1, which makes it an exact recipe; then a model kind, about 1,600 lines with tests | `po.spec.target_encoder` | low: the arithmetic is a mean |
| 2 | **`offset=`, and a model of the intercept alone**: an external model's prediction as the offset of an online one | 90% intervals around a frozen tree model: coverage 0.900 online against 0.59–0.77 static on planted drift, 0.880 against 0.917 on flights; online bias alone: flights RMSE 37.39 → 36.74 (−1.7%), planted 2.44–2.55 → 1.47–1.50 | about 600 lines with tests, no state change | `offset=` on the regression kinds | low |
| 3 | **Frozen trees as features of an online model**: each row's leaves as inputs to `sgd`/`ftrl`/`ewridge` | planted drift: RMSE 2.44–2.55 frozen → 1.02–1.03 (noise floor 1.00); real: flights 37.46 → 36.86 (−1.6%), BTCUSDT none | a recipe today in Python (§3.3, measured); a reader of the model files for the command line and for serving without the libraries, about 1,650 lines with tests | `trees=po.trees(...)`, appended by the bank to any kind's features | medium for the reader: a fifth unstable interface, the model file formats |
| 4 | **A base margin through `po.gbdt`** | planted level outside the training range: RMSE 5.83 trees, 3.13 with the online prediction as a feature, 1.02 as the margin; in range, margin = feature; real: BTCUSDT Poisson 0.4682 → 0.4603 log-RMSE, flights worse than as a feature | about 400 lines with tests | `base_margin=` / `init_score=` | low |
| 5 | **Ranking groups through `po.gbdt`** | a capability, not an accuracy: unmeasured. Found that XGBoost splits a group that straddles two batches, silently | about 300 lines with tests | `qid=` / `group_column=` | low, once the batching rule is kept |
| — | pairwise interaction screen | planted: linear model 1.337 → 1.148, XGBoost constraints 1.034 → 1.025; real: none on either dataset | deferred (§7) | | |
| — | decay-weighted batch retrain, forward validation | planted drift only; on real data uniform weights were best | a recipe (§7) | | |

Dropped (§8): monotone constraints from `marginal`'s slopes, online
quantile cuts, missing-value indicators, target transforms, a count
encoding of its own, LightGBM's `refit` / XGBoost's `refresh` as a
library feature.

Two findings outside the candidates (§6): `with_windows(group=...)` costs
time per row in proportion to the number of groups (182 ns a row at 3
groups, 80 µs at 10,000); and XGBoost's callback route splits a ranking
group at a batch boundary and accepts query ids that step back across
batches, both without a word.

---

## 1. How it was measured

**Data.** Every file is downloaded by `tests/data.py`'s helpers into
`.cache/` and read with `python -I` from scripts outside it (hard rule 1).

| data | rows | what it is | used for |
|---|---:|---|---|
| nycflights13 `flights` (Rdatasets) | 327,346 | every 2013 departure from New York, cancelled flights dropped, in scheduled-departure order; target `arr_delay` in minutes, or `arr_delay > 15`. Categoricals: `tailnum` 4,037 levels, `flight_id` (carrier and number) 5,706, `route` 223, `dest` 104, `carrier` 16, `origin` 3. A label is known only at arrival: `avail = t + dep_delay + air_time + 20` minutes | §2, §3, §4, §5, §7 |
| BTCUSDT 1-minute klines, 60 days from 2024-01-02 (Binance) | 86,390 | target `log1p(n_trades)` or `n_trades` (counts), features lagged EW means of the target at 5, 30, 240 and 1,440 minutes, the last absolute return and its 30-minute mean, minute of day, weekday | §3, §5, §7 |
| planted | 200,000–300,000 | per section, below; seeds 0, 1, 2 | all |

**Splits are by time, never shuffled.** Flights: train January–August,
validate (early stopping) September, test October–December. BTCUSDT: days
1–40, 41–45, 46–60. Planted: the first 40–60% of the clock, the next 10%,
the rest. The trees are LightGBM with `learning_rate` 0.05 (0.1 where
noted), 31 or 63 leaves, `min_data_in_leaf` 100, `deterministic=True`, 4
threads, early stopping on the validation slice; XGBoost `hist` where
noted.

**Load.** Each script logs `uptime` before and after. The runs were on an
Apple M4 Pro shared with another worker: load average 3.0 to 10.6 for every
number here, except one burst to 74 at 08:17, during which nothing ran.
No number here is a timing comparison between machines; the few timings
are within one run.

---

## 2. Target statistics of a categorical (rank 1)

### 2.1 What it is

A categorical feature with thousands of levels gives a tree little to
split on: LightGBM's own categorical handling and XGBoost's partition
splits sort the levels by their gradient statistics on the training rows,
which for a level seen ten times is noise. The standard remedy replaces the
level by a statistic of the target over the rows of that level, a *target
statistic* (TS). Computed over all rows, it leaks the row's own label into
its feature. CatBoost's answer (Prokhorenkova et al. 2018, §3.2, eq. 5) is
the *ordered* TS: each row's statistic from the rows before it alone,
shrunk to a prior,

```
x̂_k = (Σ_{j<k, c_j=c_k} y_j + a·p) / (#{j<k : c_j=c_k} + a)
```

with the order a random permutation for a batch. On a stream the order is
time, and the statistic is hard rule 2 itself: the row's value is read
before its label is learned. What this library adds to CatBoost's form is
what it adds everywhere: an exponential decay on the clock, labels folded
in when they become known rather than when their row arrives, a state
that continues past the training cut into serving, and bounded memory.

### 2.2 Measured

**On flights, arrival delay** (`scripts/tree_features/ts_flights.py`), the
four high-cardinality columns encoded each way, the rest of the features
the same:

| encoding of `dest`, `tailnum`, `flight_id`, `route` | test RMSE | test MAE |
|---|---:|---:|
| dropped | 37.293 | 24.250 |
| LightGBM's native categorical | 37.537 | 24.499 |
| codes as numbers | 37.391 | 24.396 |
| greedy TS on the training rows (leaks) | 37.614 | 24.645 |
| scikit-learn's `TargetEncoder` (5-fold cross-fitting) | 37.483 | 24.452 |
| ordered TS, no decay, frozen at the training cut | 37.553 | 24.674 |
| ordered TS, no decay, continuing | 37.292 | 24.194 |
| ordered TS, 30-day half-life, frozen at the cut | 37.481 | 24.563 |
| **ordered TS, 30-day half-life, continuing** | **36.774** | **22.467** |
| the same, with each level's label weight and decayed frequency | 36.883 | 22.549 |
| **the same, with short half-lives beside it** (the plane's last 6 hours, the origin's 2, the destination's and route's 4) | **34.472** | **20.923** |

Predicting the training mean scores 38.093. Every static encoding lands
within 0.3 of dropping the columns. The gain comes from three things
together: the decay, the state continuing through the test months with
each label folded in at its arrival, and, largest, short half-lives that
turn a level into its recent behaviour: a plane that has been late all
afternoon. A batch encoder cannot express the last one at all.

**The same as a classification**, `arr_delay > 15`, the statistic a rate
(`ts_flights_binary.py`):

| encoding | test log loss | test AUC |
|---|---:|---:|
| dropped | 0.5112 | 0.6397 |
| LightGBM's native categorical | 0.5164 | 0.6252 |
| scikit-learn's `TargetEncoder` | 0.5165 | 0.6241 |
| ordered rate, 30 days, continuing | 0.5010 | 0.6579 |
| with the short half-lives beside it | **0.4529** | **0.7460** |

**Planted** (`ts_synth.py`): 300,000 rows, a Zipf(1.3) categorical over
20,000 ranks whose population shifts so that half the levels in the test
period were never seen in training (50.4%), effects N(0, 1) of which half
drift as a sine of amplitude 1, plus `sin(2 x0) + x1` and unit noise. Test
RMSE over seeds 0, 1, 2 (the noise floor is 1.0):

| encoding | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| dropped | 1.4980 | 1.5207 | 1.5126 |
| LightGBM's native categorical | 1.4979 | 1.5206 | 1.5126 |
| greedy TS (leaks) | 1.4616 | 1.4781 | 1.4759 |
| scikit-learn's `TargetEncoder` | 1.5476 | 1.5540 | 1.5645 |
| ordered TS, no decay, frozen at the cut | 1.4522 | 1.4685 | 1.4647 |
| **ordered TS, no decay, continuing** | **1.1606** | **1.1671** | **1.1698** |
| ordered TS, half-life 20,000 rows, continuing | 1.1661 | 1.1732 | 1.1760 |

LightGBM's native handling ignores a feature of 22,600 levels here (it
scores as if dropped). The continuing statistic wins because half the test
rows carry a level no batch encoder has seen, which the stream learns as
the level appears.

**Bounded memory** (same data, 20,000-row half-life, continuing), at an
equal number of slots:

| memory | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| unbounded (22,600 levels) | 1.1661 | 1.1732 | 1.1760 |
| cap of 4,096 levels, the least decayed weight evicted | **1.1883** | **1.1988** | **1.1990** |
| one hashed table of 4,096 cells | 1.2521 | 1.2592 | 1.2644 |
| count-min, 4 × 1,024 cells, the cell of least weight read | 1.2985 | 1.3096 | 1.3113 |
| cap of 512 | 1.4667 | 1.4859 | 1.4825 |
| one hashed table of 512 | 1.4331 | 1.4419 | 1.4433 |
| count-min, 4 × 128 | 1.4763 | 1.4951 | 1.4908 |

**A cap with least-weight eviction errs toward the prior; a hashed table
errs toward a mixture of levels.** When the cap covers the levels in use,
it is nearly free (1.19 against 1.17), and the error it makes is the honest
one: an evicted or new level reads the prior, as a level never seen does.
A hashed table merges unrelated levels' statistics, a bias with no sign
that the tree then learns as signal. Count-min's minimum, the right read
for counts, is the wrong one for a mean: the least-collided cell is not the
least biased. When the cap is far too small, every scheme fails, the cap
somewhat worse, because it thrashes. So: a cap, sized to the levels in use,
and `weight` emitted beside the statistic so a tree can tell a level read
from data from one read from the prior.

**On flights, the cap costs nothing down to a quarter of the levels**
(`ts_flights_cap.py`, the best arm above with every table capped; v2's
measurement). A plane or a flight number that has not flown for weeks has
the least weight, and is the one evicted:

| levels kept per table | test RMSE | test MAE |
|---|---:|---:|
| all (`tailnum` 4,037, `flight_id` 5,706) | 34.472 | 20.969 |
| 1,024 | 34.377 | 21.005 |
| 256 | 34.453 | 21.204 |
| 64 | 34.852 | 21.996 |

(The uncapped MAE differs from the table above in the third digit because
this arm leaves out the frequency columns.)

### 2.3 The recipe it already is, and the defect that stops it

**The decayed, label-time statistic is already exact with the window
operators** (`ts_recipe.py`): a doubled stream, each row once at its own
time with a null target and once at its label's time with the target, then

```python
out = stream.online.with_windows(
    S=po.ewm_sum("y", half_life=hl, closed="left"),      # the decayed sum of labels before the stamp
    W=po.ewm_sum("one", half_life=hl, closed="left"),    # their decayed weight; null where none yet
    clock="c", gap_cap=1e12, group="tailnum",
)
```

reads, on the predict rows, the prototype's sums to a relative 3e-13 (`W`
to 2e-15). The prior is one more ungrouped `with_windows` call. But it ran
at 24 µs a row with 4,037 groups, where 3 groups take 182 ns (§6.1), so
the recipe is not usable at the cardinalities that need it until that
defect is fixed.

### 2.4 The design: a model kind

**Recommended: a bank kind, `po.spec.target_encoder`.** The bank is where
"read before the label is learned" is enforced by construction; it already
has late labels (`embargo`, and the doubled stream), groups, weights, a
versioned state that carries the encoder from the training run into
serving, the command line, and `po.gbdt` reads a bank's output query
directly (checked: `gbdt_compose.py`). One spec encodes many columns in one
pass and keeps one running prior.

```python
te = po.spec.target_encoder(
    "te", targets=["arr_delay"], features=["tailnum", "dest", "route"],
    clock="t", gap_cap="6h", half_life=["6h", "30d"],   # a list: one set of sums per half-life, as ewridge's grid
    prior_weight=10.0,         # a: the prior counts as this many rows
    prior="running",           # p: the target's own decayed mean before the row; or a number
    max_categories=100_000,    # per feature; the least decayed weight is evicted past it
)
```

The name follows scikit-learn's `TargetEncoder`, the one the field knows;
Polars has no such concept.

| per row, per feature `f`, half-life `h`, target `t` | |
|---|---|
| key | the value's 64-bit hash (xxh3 of its string form, fixed seed; a `pl.Enum` hashes its category string, not its code; a null is a level of its own, as `group_by` keeps it). The model receives it as two exact 32-bit halves in its `f64` features, so `OnlineModel` does not change; the state records the hash's name and seed, and a state with another is refused. Two levels collide with probability about n²/2⁶⁵: 3·10⁻⁶ at ten million levels |
| prior | `prior="running"`: the target's own decayed mean before the row, at `prior_half_life` (default the spec's half-life, or the longest of a list; flights' short statistics measured best with a 4-hour prior under a 2-to-6-hour statistic); or a number |
| names | with one half-life, `enc_<f>_<t>`; with a list, each half-life's fields carry the instance suffix `ewridge`'s grid gives its slots |
| read, before the row is learned | `enc = (S + a·p) / (W + a)`, with `W`, `S` the level's decayed label weight and label sum at the row's clock, `p` the prior. `a = 0` and `W = 0` give null, never 0/0 (rule 9) |
| learn | `W' = λW + w`, `S' = λS + w·y`, `λ = 0.5^(Δt/h)` from the level's last label; a null target learns nothing; weight 0 learns nothing and creates no level |
| emitted | `enc_<f>_<t>`, `weight_<f>_<t>` (the level's `W`, so a tree sees whether the value came from data or from the prior), and `weight_sum` with rule 8's meaning |
| classification | `classes=[...]` as `ew_class` takes them: one `S` per class, `enc_<f>_<class>` a rate, the prior the classes' running shares |
| eviction | past `max_categories`, the level of least decayed weight, ties by smaller key. All levels decay by the same factor, so the order is that of `ln W + ln 2 · t_last / h`, which does not change with time: a sorted set gives the victim in O(log n) |
| state | per feature and half-life, a map key → (`W`, `S` per target or class, `t_last`); about 48 + 8T bytes a level; sorted by key when saved, so the bytes do not depend on insertion order |

**Guarantees.** Rule 2: the statistic is read before the row's label, and a
label that arrives late is learned at its own time (doubled stream or
`embargo`). Rule 3: rows are taken one at a time, eviction is
deterministic, so any chunking gives the same output. Rule 8: `weight_sum`
is the stream's weight before the row and its decay; each target's level
weights are its own. Rule 9: a zero-weight row advances the clock and
creates nothing; a zero prior weight with no data gives null. Rule 5: a new
`ModelState` variant, schema bump, fixtures with an evicted level and every
key kind. Rules 12 and 13: nothing linked; no unstable interface.

**Tests and their oracles.** Undecayed, fixed prior, no embargo:
`category_encoders.CatBoostEncoder`, which computes the ordered TS in row
order (`ce_oracle.py`: our recursion with its prior agrees to 1.2e-14;
its prior is the mean of all labels, the future included, so the test
passes `prior=` a number). Decayed: the definition written by hand from eq.
5 with exponential weights, in numpy. Late labels: the `with_windows` recipe
of §2.3. Classification: `CatBoostEncoder` on each class indicator.

**Cost.** One hash of the value, one table lookup, one `exp` and a few
flops per feature per half-life per row: an estimate of 50–100 ns, against
107 ns a row for a five-feature `ewridge` (PERFORMANCE.md's headline).
Memory as above: a million levels of one target at about 56 MB.

### 2.5 Durability

| a user will want | taken by the design |
|---|---|
| several targets | `S` per target, `W` per target where it is present (rule 8, S2) |
| classification | `classes=` |
| weights | `weight=` multiplies |
| groups | `group=`: an encoder per group |
| windows | refused in v1 by name; `window_size` would follow `ewridge`'s snapshots |
| several half-lives | a list, one set of sums each, as `ewridge`'s grid |
| nulls | a level of its own |
| unbounded cardinality | `max_categories` |
| the command line | a TOML spec like any other |
| save and resume | the bank's state |
| combinations of categoricals (CatBoost's feature combinations) | not in v1; `features` should take `str \| list[str]` from the start, a list hashed as one tuple key, so that adding it breaks nothing |

**What would make us rethink it:** a keyed window operator in `po.ops`,
which would make the kind a second way to do the same thing. The recipe of
§2.3 is that operator in all but name. So the order is: fix §6.1 first,
which is a defect whatever is decided and makes the recipe exact and fast;
then build the kind for what the recipe cannot do, one spec over many
columns with a shared running prior, a cap, and the classification form,
in one pass rather than one `with_windows` call per column plus one for
the prior.

**The second thing a year would bring: labels with a time of their own.**
Flights' labels arrive hours after their rows, at a time that differs by
row. The doubled stream handles it today for every kind, and so does
nothing else; a `label_clock=` column on the bank, for every kind, is the
durable answer, and the encoder should not grow a mechanism of its own for
it.

---

## 3. Frozen trees as features of an online model (rank 3)

### 3.1 What it is

He et al. 2014 (Facebook's click prediction, §3.1) train boosted trees in
batch, take the leaf each row falls into in each tree as a one-hot
feature, and fit a linear model on those leaves: "a supervised feature
encoding". Their §3.2 finds that training daily rather than weekly cuts
normalized entropy by about 1%, and proposes retraining the trees daily or
every few days while the linear layer learns online. That split is exactly this library's place:
the structure from the library's batch fit, the adaptation from an online
model that refits the leaf weights with decay. XGBoost's R package ships
the encoding (`xgb.create.features`).

### 3.2 Measured

**Planted drift** (`leafrefit.py`, `leafrefit2.py`): 300,000 rows on a
clock of 3,000 steps, `y = a(s)·sin(2x0) + b(s)·[x1 > 0.5] + c(s)·x2·[x3 > 0]
+ level + noise`, the three weights drifting over the run (a sine, a ramp,
a sign change) and the level a slow random walk. Trees on the first 40%,
test on the last 50%:

| arm | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| the frozen trees (19–22 trees of 15 leaves) | 2.5472 | 2.4398 | 1.5186 |
| `ewridge` on each tree's leaf value (M features) | 1.2608 | 1.2790 | 1.2536 |
| **`ewridge` on one-hot leaves** (285–330 features) | **1.0206** | **1.0247** | **1.0180** |
| **`sgd`, adagrad, rate 0.05, on one-hot leaves** | **1.0250** | **1.0286** | **1.0241** |
| trees plus an online residual (`ewridge` on the raw features, target `y − trees`) | 1.3199 | 1.3264 | 1.3480 |
| `ewridge` on the raw features alone | 1.2653 | 1.2691 | 1.2628 |
| LightGBM's own `refit(decay_rate=0.5)` every 50 steps on the last 50 | 1.0583 | 1.0641 | 1.0483 |

The noise floor is 1.0. `sgd` on one-hot leaves matched the exact solve at
a fourteenth of the time (0.35 s against 5 s for 300,000 rows through the
bank, Python included). LightGBM's own periodic refit comes close, at the
price of a refit loop and the library at serving time.

**Real data.** Flights (`flights_more.py`, labels learned at arrival):
frozen 37.459, leaf values 36.856 (−1.6%), one-hot 37.076, the online
residual 36.910, and with the origin's recent delays in the residual model
36.098. BTCUSDT: frozen 0.4486, leaf values 0.4535, the residual 0.4481:
no gain, as its lagged features already carry the level.

### 3.3 A recipe today, measured

**In Python it needs nothing built** (`leaves_recipe.py`, v2's measurement):
the library's own `pred_leaf`, one-hot encoded chunk by chunk with
`LazyFrame.map_batches(..., streamable=True)`, feeds the bank through the
IO plugin, and the result equals the same columns made on the collected
frame to the bit:

```python
def add_leaves(df: pl.DataFrame) -> pl.DataFrame:
    leaves = booster.predict(df.select(names).to_numpy(), pred_leaf=True)   # (rows, trees)
    onehot = np.zeros((df.height, n_trees * n_leaves))
    onehot[np.arange(df.height)[:, None], np.arange(n_trees) * n_leaves + leaves] = 1.0
    return df.hstack(pl.DataFrame(onehot, schema=leaf_columns))

out = (pl.scan_parquet("rows.parquet")
       .map_batches(add_leaves, streamable=True, schema=schema)               # a row's leaves are its own: any chunking
       .online.fit_predict([po.spec.sgd("refit", targets=["y"], features=leaf_columns, schedule="adagrad",
                                        learning_rate=0.05, standardize=False, clock="c", gap_cap=10.0, half_life=20.0)])
       .collect())
```

On the planted drift, seed 0, it scored the 1.0250 of the table above, in
0.51 s for 300,000 rows. `standardize=False` matters: `sgd` and `pa`
standardize by default, and centring a one-hot column spreads every row's
update over every coefficient. What the recipe cannot do is run where
Python does not: the command line, or a server without XGBoost or LightGBM
installed. That is what the reader below would buy, and all it would buy.

### 3.4 Reading the model files without linking

**A reader of XGBoost's JSON and LightGBM's text model gives each row's
leaf exactly as the libraries' own `pred_leaf` does** (`tree_reader.py`):
0 mismatches over 20,000 rows × 60 trees for each of four models: XGBoost
with 786 categorical partition splits, LightGBM with categorical bitsets
and each of its three missing types (`NaN`, `Zero`, `None`). The rows held
10% nulls in every column, values on the thresholds, unseen and negative
category codes, zeros, and ±∞ (LightGBM; XGBoost refuses ∞ as input). The
rules, from the libraries' sources:

| library | numeric | missing | categorical |
|---|---|---|---|
| XGBoost (`predict_fn.h`, `categorical.h`) | in float32: `x < split_condition` goes left | the node's `default_left` | a category in the node's set goes **right**; a negative code or one past 2²⁴ goes left |
| LightGBM (`tree.h`) | in float64: `x <= threshold` goes left | by `decision_type`'s bits 2–3: `NaN` sends a NaN by `default_left`; `Zero` sends \|x\| ≤ 1e-35 (a NaN read as 0) by it; `None` reads a NaN as 0 | a category in the node's bitset goes **left**; NaN or a negative code goes right |

Each reader was about 40 lines of Python. In Rust, with parsing (XGBoost's
JSON through `serde_json`, already a workspace dependency; LightGBM's text
by hand), about 700 lines for both, plus tests.

### 3.5 The design: the reader, when the command line needs it

**Recommended, when the command line or library-free serving is wanted:
the reader in `online-core` (`forest.rs`, a parsed forest and its
evaluation, no Polars), and a `trees=` input that the bank turns into
columns appended to the kind's features.** The models do not change: the
bank assembles the row (rule 4), as it would the recipe's columns, and the
coefficient index names them.

```python
leaves = po.trees("model.json", features=["x0", "x1", "x2", "venue"], encode="one_hot")  # or "leaf_value"
spec = po.spec.sgd("refit", targets=["y"], features=["x0", "x1"], trees=leaves,
                   schedule="adagrad", learning_rate=0.05, clock="t", gap_cap=10.0, half_life=2_000.0)
```

| | |
|---|---|
| model files | XGBoost JSON (its documented schema, `version` checked and a version outside the range read refused by name, as `po.gbdt` refuses a library version), LightGBM's text model; a path, bytes, or a `Booster` (saved through its own `save_raw("json")` / `model_to_string()`, found by duck typing) |
| features | by the model's own `feature_names`; a model without them takes `features=` in order. A categorical feature is a `pl.Enum`'s codes, as `po.gbdt` builds it |
| `encode="one_hot"` | width Σ leaves, one 1 per tree; coefficient names `tree<k>_leaf<l>` |
| `encode="leaf_value"` | width M, each tree's leaf value (refused for LightGBM's linear trees and XGBoost's DART, whose leaf value is not the output) |
| width | `sgd`, `pa`, `ftrl`: O(width) a row, dense; the kinds that solve (`ewridge`, `huber`, `quantile`, `rls`, `lasso`, `kalman`) refuse a one-hot width past a budget and name `encode="leaf_value"`. A one-hot column is never standardized, whatever `standardize` says of the raw features (§3.3) |
| sparsity | not in v1. A one-hot row has M ones in Σ leaves columns, so `sgd`'s update could touch M coefficients rather than all; worth building only when a forest's width makes the dense row the cost (500 trees of 64 leaves: 32,000 columns) |
| state | the forest is not state: its SHA-256 goes into the spec, and a state loads only against the same forest; the coefficients are the kind's own |
| the CLI | `trees = { path = "model.json", encode = "one_hot" }` in the spec's TOML |

**Guarantees.** Rule 3: a row's leaves are a function of the row alone.
Rule 2 holds for the online layer only: trees trained on rows that the
stream later scores have seen those labels, so the docs must say to train
the trees on data before the scored segment. Rules 8 and 9 are the kind's
own. Rule 12: nothing linked; the files are read. Rule 13: a **fifth
unstable interface**, the libraries' model file formats. XGBoost documents
its JSON schema (`doc/model.schema`); LightGBM's text format is not
documented as a contract, though its reading has not changed since v3.

**Tests and their oracle.** `pred_leaf` of the library itself, to the bit,
on the edge cases above, for every model shape (regression, binary,
multiclass with a tree per class, categorical, each missing type).

**Cost.** Per row, M walks of depth d (100 trees of depth 6: 600
comparisons, an estimate of 0.3–1 µs), then the kind on the appended width:
`sgd` on 285 one-hot columns measured 1.2 µs a row through the bank.

### 3.6 Durability

| a user will want | taken |
|---|---|
| classification, counts | the kind's own losses: `sgd` logistic and Poisson, `ftrl` |
| several targets | the kind's own |
| a multiclass forest | one-hot per tree regardless of class |
| groups, decay, save and resume, the CLI | the kind's own |
| a newer XGBoost | the version check refuses by name; a reader update is maintenance |
| raw features beside the leaves | `features=` and `trees=` together |

**What would make us rethink it:** a model trained on pandas categoricals,
whose JSON (XGBoost 3.x) stores category names and recodes inputs by name
at prediction. v1 should refuse such a model by name rather than read codes
the model would not. And the recipe of §3.3 already covers Python: if the
command line never needs tree features, the reader is a cost with no
user, which is why it ranks below `offset=`.

---

## 4. `offset=`, and a model of the intercept alone (rank 2)

**An offset is a column added to the model's linear predictor, which the
model fits around.** It is the concept of `glmnet`'s and `statsmodels`'
`offset`, and the one XGBoost calls `base_margin` and LightGBM
`init_score`. With an external model's prediction as the offset, an online
model corrects its bias as the world drifts, and every diagnostic the bank
has (`emit_sigma`, `conformal`, `emit_drift`, `emit_calibration`) applies to
that external model.

**Measured** (`conformal_trees.py`): frozen trees, then the bank's
conformal tracker on the residual stream `y − trees`, through an
`ewridge` whose one feature is a column of zeros (the stand-in for a model
of the intercept alone, which `features=[]` refuses today). 90% intervals:

| | planted, seed 0 | seed 1 | seed 2 | flights |
|---|---:|---:|---:|---:|
| static split-conformal radius from the validation residuals: coverage | 0.749 | 0.586 | 0.769 | 0.917 |
| its coverage, by quarter of the test period | 0.78 0.70 0.73 0.79 | 0.69 0.56 0.52 0.57 | 0.85 0.75 0.73 0.75 | 0.93 0.95 0.92 0.87 |
| online conformal on the residual stream: coverage | **0.900** | **0.900** | **0.900** | 0.880 |
| test RMSE, frozen trees | 2.547 | 2.440 | 1.519 | 37.39 |
| test RMSE, trees plus the online bias | 1.467 | 1.477 | 1.498 | 36.74 |

On flights the online interval covered 88.0%, short of 90%, because the
labels arrive hours late and the tracker learns from them late; the static
one held on average and fell to 87% in the last quarter of the test period.

**For squared error it is a recipe today, and that is what was measured:**
the target `y − trees` and a column of zeros as the one feature. What a
build adds is what the recipe cannot reach: the offset inside a link (the
log of a Poisson `sgd`, the logit of `ftrl`), `features=[]` in place of the
column of zeros, and a `pred` that includes the offset, so that `resid`,
`sigma`, `conformal`'s `lo`/`hi` and the rest are on the target's own scale
without the user adding it back.

**Design.** `offset="col"` on every regression kind: the prediction is
`offset + f(x)` in the kind's link space (identity; the log of `sgd`'s
Poisson; the logit of `ftrl` and `sgd`'s logistic), and residuals and every
diagnostic are read on `y − pred`. A null offset withholds the row as a
null feature does. `features=[]` with `fit_intercept=True` becomes legal: a
model of the intercept alone. No state changes, so no schema bump; one
subtraction a row. The offset must itself be known before the row's label,
as any feature must: an external model's prediction made in advance, not a
fit that saw the row.

**Tests and their oracle.** statsmodels' GLM with `offset=` (Poisson,
binomial) and least squares on `y − offset`, at no decay, against the
kinds' fits.

**Durability.** It is the textbook concept under its textbook name; the
same column feeds `po.gbdt`'s base margin (§5) in the other direction. What
would make us rethink it: nothing found.

---

## 5. A base margin through `po.gbdt` (rank 4)

**The libraries fit trees around a given margin,** `base_margin` in
XGBoost and `init_score` in LightGBM. An online model's prediction is out
of sample by construction (rule 2), so as a margin it leaks nothing: the
trees fit what the online model leaves.

**Measured** (`margin_synth.py`, `margin_btc.py`, `flights_more.py`).
Planted: 100 rows per clock step over 2,000 steps, a level that random-walks
and that no feature carries, `f(x)` nonlinear; the online model is an
`ewridge` (Gaussian) or `ftrl` (Bernoulli) at a 5-step half-life.

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| the test level against the training range | inside | inside | **outside** (−9.5 to −6.0, training −4.5 to 0.5) |
| Gaussian RMSE: trees alone | 1.777 | 1.742 | 5.825 |
| the online prediction as a feature | 1.026 | 1.024 | 3.131 |
| **as the margin** | **1.021** | **1.022** | **1.023** |
| Bernoulli log loss: trees alone | 0.516 | 0.386 | 0.389 |
| the online logit as a feature | 0.480 | 0.353 | 0.204 |
| **as the margin** | **0.479** | **0.354** | **0.152** |

A tree cannot predict outside the range of the values it split on; a margin
is added after them. Real data: BTCUSDT `n_trades` under a Poisson loss,
log-RMSE 0.4682 alone, 0.4652 with the feature, 0.4603 with the margin;
flights, the origin's recent delays: 37.39 alone, 35.13 as a feature,
36.00 as the margin, 35.22 both. So a margin is the safe choice where the
level can leave the training range, and a feature can be better where it
does not. Pass it as both when unsure.

**Checked in the libraries.** XGBoost's callback route carries a
`base_margin` given batch by batch into the `QuantileDMatrix`
(`qid_check.py`), and with a margin XGBoost adds the margin in place of
`base_score` (`margin_btc.py`: the difference is 1.5e-5, float32
rounding). LightGBM's `LGBM_DatasetPushRowsWithMetadata` takes `init_score`
per batch once `LGBM_DatasetInitStreaming` is told `has_init_scores`, and
skips `boost_from_average` when one is set (`gbdt.cpp:331`).

**Design.** `po.gbdt.xgboost_matrix(..., base_margin="col")` and
`po.gbdt.lightgbm_dataset(..., init_score="col")`, each library's own word,
as `ref=` and `reference=` already are. A `List`/`Array` column of width K
for a K-class objective. The validation matrix takes its own column under
the same keyword. Refused: a null or non-finite margin. The cuts do not
depend on it. The docs must show the margin added at prediction: an
XGBoost `DMatrix` with `base_margin`, or LightGBM's `raw_score=True` plus
the margin. LightGBM's Python engine (`Sequence`) cannot stream it, so it
holds n·K doubles in memory there, stated.

**Tests and their oracle.** The same library's in-memory `DMatrix` /
`Dataset` with the margin: identical trees, by hash, as `po.gbdt`'s tests
already hold the build to.

---

## 6. Findings outside the candidates

### 6.1 `with_windows(group=...)` costs time in proportion to the groups

Measured (`ts_recipe_speed.py`, `ts_recipe_speed2.py`), one `ewm_sum` over
randomly interleaved groups, `closed="left"`:

| groups | ns a row (100,000 rows) | ns a row (400,000 rows) |
|---:|---:|---:|
| 10 | 259 | 168 |
| 100 | 516 | 512 |
| 1,000 | 3,931 | 3,959 |
| 10,000 | 79,905 | 88,386 |

The cost per row grows with the number of groups, not the rows. A profile
(`sample`, 4 s) puts the time in `Windows::push` and `flush_group`.
`flush_stamps_before` (`windows.rs:2357`) walks the list of groups in order
of their last row while each one's stamp has passed, calling
`flush_group`, which returns at once for a group with nothing pending; with
interleaved groups every other group's stamp has passed, so each row walks
all of them. Keeping only the groups with pending rows on the list walked
would make it O(1) a row. This is a hypothesis from the profile and the
code, not a fix measured; it should be a task of its own, since any
`with_windows(group=...)` with thousands of groups pays it.

### 6.2 XGBoost's callback route and ranking groups

`qid_check.py` builds a `QuantileDMatrix` from two batches through
XGBoost's `DataIter`: a query group split across the batch boundary comes
back as two groups (`group_ptr` [0 2 3 4 6] for three groups), and query
ids that step back from one batch to the next are accepted
(`MetaInfo::Extend`, `data.cc:717`: each batch's ids are run-length
encoded on their own, and the non-decreasing check, `data.cc:621`, is per
batch). LightGBM's streaming route keeps one array of ids and finds the
boundaries once, at `FinishLoad` (`metadata.cpp:191`), so a group may
straddle pushes there, but an id that comes back later starts a new group,
also silently.

---

## 7. Deferred, and recipes

### 7.1 Ranking groups (rank 5)

`po.gbdt.xgboost_matrix(..., qid="col")` and
`po.gbdt.lightgbm_dataset(..., group_column="col")` (LightGBM's own name
for a column of query ids), the column sorted. Because of §6.2, the build
must cut XGBoost's batches at a group boundary (carrying a group's head
into the next batch) and check the ids are non-decreasing across the whole
stream, refusing otherwise. XGBoost weights a ranking objective per group,
not per row, so `weight=` beside `qid=` is refused in v1. Value: a
capability, unmeasured. About 300 lines with tests; oracle: the library's
in-memory route with `qid`/`group`, identical trees.

### 7.2 A pairwise interaction screen: deferred

A B × B table of `(Σw, Σw·r, Σw·r²)` per feature pair, `r` the target or an
online model's residual, ranked by a two-way ANOVA interaction F on the
table, fits in `marginal`'s design (`bins=` already places the edges). On
20 planted features with four interacting pairs, a nonlinear main effect
and a correlated pair (`pairs.py`), the F found the pairs with average
precision 0.50–0.77. FAST (Lou et al. 2013, §4.1: the best 2 × 2 cut of the
table, on residuals) scored 0.04 on the raw target and needs the main
effects removed at the table's own resolution: 0.69–0.77 at 8 bins on an
8-bin additive model's residual, 0.04–0.08 at 16 bins. InterpretML's FAST
(`measure_interactions`, 0.7.8) agreed: 0.04 on the target, 0.16–0.29 with
the same residual as `init_score`. So the F, which removes the main effects
itself, is the score to build. What the pairs bought (`pairs_value.py`):
products of the top 8 pairs took a linear model from 1.337 to 1.148 test
RMSE (the true pairs: 1.148), and XGBoost constrained to them scored 1.025
against 1.034 unconstrained. On BTCUSDT and flights (`pairs_real.py`) the
screen's pairs bought nothing, linear or tree, and the F statistics sat far
above 1 for every pair, serial dependence inflating them. With no gain on
real data, it is deferred until a dataset shows one.

### 7.3 Decay-weighted batch retrains, and forward validation: a recipe

`weight = 0.5 ** ((cut − t) / half_life)` is one Polars expression. On the
planted drift it cut the next period's RMSE from 2.32 to 1.26 (seed 0; 1.91
to 1.44 and 1.19 to 1.15 on the others), but on real data uniform weights
were best: flights 37.40 against 37.55–37.78, BTCUSDT 0.4493 against
0.4493–0.4586 (`decayw.py`). A validation slice at the end of the training
time against a shuffled one was mixed: under drift the forward slice
removes the newest rows from training, which cost more than its honesty
gained (planted), and on flights it helped (37.40 against 37.56). A
paragraph in GBDT.md, not a feature.

---

## 8. Dropped

| idea | why |
|---|---|
| monotone constraints from `marginal`'s slopes | a marginal slope's sign is not the sign of the partial dependence under correlated features; a wrong constraint is a silent bias. Not measured |
| online quantile cuts replacing `po.gbdt`'s sample | GBDT-ADAPTER §4.1 measured the t-digest slower than a sample and no better |
| missing-value indicators | both libraries learn a default direction per split (§3.3) |
| a count encoding of its own | flights: adding decayed frequency and label weight to the statistic changed RMSE by +0.11 (36.774 → 36.883). `weight_<f>_<t>` comes with the encoder anyway |
| LightGBM's `refit` / XGBoost's `refresh` behind a library call | measured close to the online refit (1.05–1.06 against 1.02–1.03) but needs the library and a loop at serving; a recipe if wanted |
| target transforms, lag and window features | Polars' `with_columns` and `po.ewm_*` already make them |

---

## 9. Self-review: what a user would hit in a year

v1 ranked the items and designed each; this section attacks every
recommendation as a user a year from now would, and says what v2 changed.
The changes are folded into the sections above.

| item | the attack | the answer, and what changed |
|---|---|---|
| target encoder | "I already have this": `with_windows(group=cat)` with `ewm_sum(..., closed="left")` is exact | true (§2.3), and it is the oracle for late labels. v2 orders the work: the `with_windows` defect first (§6.1), the kind after it for what the recipe cannot do in one pass |
| target encoder | a cap that thrashes, or one that silently drops the levels that matter | v2 measured it on real data: 1,024 levels of 4,037 and 5,706 cost nothing, 64 cost 5% of MAE; the evicted level reads the prior, and `weight_<f>_<t>` says so to the tree |
| target encoder | keys through a model trait that only takes `f64` | two exact 32-bit halves of a 64-bit hash; the hash's name and seed in the state, so a build with another hash refuses the file rather than reading other levels |
| target encoder | my labels arrive late, each at its own time | the doubled stream, for every kind; the durable fix is a bank-level `label_clock=`, not a mechanism inside the encoder (§2.5) |
| target encoder | I want the short and the long half-life at once | measured as the largest gain on flights; the design takes a list, as `ewridge`'s grid does, from the first version |
| target encoder | the trees I trained see young, noisy statistics on the first rows of each level, and old, steady ones at serving | the prediction shift CatBoost names; `weight_<f>_<t>` beside each statistic lets a tree condition on it. Not measured separately |
| target encoder | one real dataset | flights is the only public set here with a high-cardinality categorical; the planted case reproduces the mechanism, not the size of the gain. A second real dataset is an open item |
| frozen trees | "I can do this in Python today" | measured true in v2 (§3.3): `map_batches(streamable=True)` and `pred_leaf`, equal to the bit to the eager run. The reader's only addition is the command line and serving without the libraries, so v2 ranks it below `offset=` |
| frozen trees | a one-hot row standardized by `sgd`'s default | every update then moves every coefficient; v2's design never standardizes the leaf columns |
| frozen trees | 500 trees of 64 leaves | 32,000 dense columns: the cost is the row, not the model; a sparse update is the follow-up, named in §3.5, not built first |
| frozen trees | trees trained on the rows the stream then scores | in sample for the trees, whatever rule 2 says of the online layer; the docs must say to train the trees before the scored segment |
| frozen trees | a newer XGBoost, a pandas-categorical model | a version check that refuses by name; a model with stored category names refused in v1 |
| `offset=` | "a target transform does this" | for squared error, yes, and that is what was measured; the build is for the link-space offset, `features=[]`, and diagnostics on the target's own scale (§4) |
| `offset=` | intervals under-cover when labels arrive late (0.880 on flights) | the tracker can only learn a miss once the label exists; the same bank-level `label_clock=` would let it learn each at its time, and the docs should state the shortfall |
| base margin | the margin was worse than a feature on flights | measured, and kept in the table: a margin is for a level that can leave the training range (planted seed 2: 1.02 against 3.13), and passing both was within 0.09 of the better of the two on both real datasets (flights 35.22 against 35.13; BTCUSDT 0.4615 against 0.4603) |
| base margin | prediction then needs the margin too | XGBoost's `DMatrix(base_margin=)` and LightGBM's `raw_score=True` plus the margin; the docs must show it, since forgetting it shifts every prediction |
| ranking groups | my groups straddle a file's batches | §6.2's finding: `po.gbdt` cuts XGBoost's batches at a group boundary and refuses ids that step back, so a straddling group is neither split nor refused silently |
| interaction screen | deferred on two datasets | no real gain was found; F statistics far above 1 on every real pair show it would need the serial-dependence count `marginal(lags=)` already keeps before its ranking means anything |

## 10. Open questions for the user

1. **The target encoder as a kind, or the recipe alone.** Once §6.1 is
   fixed, the recipe is exact and fast but takes one `with_windows` call
   per column plus one for the prior, and has no cap or classification
   form. The kind is about 1,600 lines. Recommended: fix §6.1 now; build
   the kind.
2. **The model-file reader.** About 1,650 lines and a fifth unstable
   interface, for tree features on the command line and serving without
   the libraries installed; Python has the recipe (§3.3). Recommended:
   document the recipe now, and build the reader only if the command line
   needs tree features.
3. **A bank-level `label_clock=`**, raised by the encoder, by `offset=`'s
   intervals and by flights' late labels: a column giving each row's label
   time, for every kind, in place of the doubled stream. Not designed here;
   it is a change to the bank, not to the tree work.

## 11. Reproducing

```sh
.venv/bin/python -I scripts/tree_features/fetch.py "$PWD"   # 60 days of BTCUSDT, nycflights13 into .cache/
cd scripts/tree_features
../../.venv/bin/python -I ts_flights.py          # §2.2, about 1 minute
../../.venv/bin/python -I ts_flights_binary.py
../../.venv/bin/python -I ts_flights_cap.py      # v2: the cap on real data
../../.venv/bin/python -I ts_synth.py            # about 15 minutes, the capped tables in pure Python
../../.venv/bin/python -I ts_recipe.py
../../.venv/bin/python -I ts_recipe_speed.py
../../.venv/bin/python -I ts_recipe_speed2.py
../../.venv/bin/python -I leafrefit.py
../../.venv/bin/python -I leafrefit2.py
../../.venv/bin/python -I leaves_recipe.py       # v2: the no-build recipe through map_batches
../../.venv/bin/python -I flights_more.py
../../.venv/bin/python -I tree_reader.py
../../.venv/bin/python -I margin_btc.py
../../.venv/bin/python -I margin_synth.py
../../.venv/bin/python -I conformal_trees.py
../../.venv/bin/python -I qid_check.py
../../.venv/bin/python -I gbdt_compose.py
../../.venv/bin/python -I pairs.py               # writes the data pairs_interpret.py reads
../../.venv/bin/python -I pairs_value.py
../../.venv/bin/python -I pairs_real.py
../../.venv/bin/python -I decayw.py
uv run --no-project --python 3.12 --with interpret-core --with numpy --with pandas --with scikit-learn python -I pairs_interpret.py "${TREE_FEATURES_OUT:-../../.cache/tree-features/results}"
uv run --no-project --python 3.12 --with category_encoders --with numpy --with pandas --with polars python -I ce_oracle.py
```

Each writes its numbers as JSON into `$TREE_FEATURES_OUT` (default
`.cache/tree-features/results`). InterpretML and category_encoders run in a
throwaway environment and are not dependencies of anything.

Sources: Prokhorenkova et al. 2018, *CatBoost: unbiased boosting with
categorical features* (§3.2); He et al. 2014, *Practical Lessons from
Predicting Clicks on Ads at Facebook* (§3.1–3.3); Lou, Caruana, Gehrke and
Hooker 2013, *Accurate Intelligible Models with Pairwise Interactions*
(§4.1, §5.2); XGBoost at `54155e3` and LightGBM 4.7.0, read at the lines
cited. The papers' text is under the gitignored `.cache/research/papers/`.
