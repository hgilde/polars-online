# What this library adds to XGBoost and LightGBM: designs, measured

**Status as of 2026-11-05: design and measurement for PLAN task 324, v3;
nothing is built.** v1 measured and ranked; v2 attacked its own
recommendations (§12); v3 answers a fresh reviewer's sixteen findings
(§13), with new measurements wherever they decide the ranking. The
question (the user, 2026-11-05): what can this library add, at high impact,
to the batch tree libraries without rewriting them, choosing ideas that
will not be rethought later. Each candidate was prototyped in Python
(numpy, Polars, the bank as it is, XGBoost 3.4.1, LightGBM 4.7.0) on
public data and on data that plants the effect. The prototypes are kept as
`scripts/tree_features/` (§15 runs them). [BOOSTED-TREES.md](../BOOSTED-TREES.md)
is the earlier question, an online tree model inside the library, and
[GBDT-ADAPTER.md](../GBDT-ADAPTER.md) the one that built `po.gbdt`.

---

## 0. The short version

**What to build, ranked.** Each gain is against the strongest arm that
needs no build, with its spread over seeds where the arm has one.

| rank | item | measured | cost | surface | risk |
|---:|---|---|---|---|---|
| 1 | **`label_clock=`**: each row's label time as a column; the row is scored where it sits and learned, and its served prediction scored, when its label arrives | the only route today for a delay that differs by row, the doubled stream (`po.stream.embargo`'s shape), scores every residual diagnostic against a prediction nobody served: flights' 90% interval covers 0.8803 of the served predictions while reporting the 0.8996 its tracker reaches on the learn rows. Scoring the served prediction closes it: 0.8992 at a half-width of 47.4 (simulated, the doubled case reproduced to 0.0003). The bank's `embargo` already holds each row's served prediction (§2.2), so this generalizes a delay to a per-row time | about 900 lines with tests; a schema bump | `label_clock=` on every spec | low: `embargo` is the mechanism |
| 2 | **Target statistics of a high-cardinality categorical**, a model kind | the part that needs high-cardinality keys, over recent means with no key and with low-cardinality keys: flights MAE 21.45 → 20.72 (−3.4%; seeds 1/2/3 within 0.03), → 20.57 (−4.1%) with longer statistics beside them that August's selection had declined. Batch baselines tuned on the earlier months, CatBoost with `has_time=True` among them, reach 24.0–24.5 | about 1,700 lines with tests; a new `ModelState` | `po.spec.target_encoder` | low |
| 3 | **A base margin through `po.gbdt`** | planted level outside the training range: RMSE 5.83 trees, 3.13 with the online prediction as a feature, 1.02 as the margin; inside it, margin ≈ feature; real: BTCUSDT log-RMSE 0.4682 → 0.4603, flights worse than the feature | about 400 lines with tests | `base_margin=` / `init_score=` | low |
| 4 | **`offset=`**, per target, and a model of the intercept alone | inside a link, measured against the same margin as a free feature: flights' delayed-flight logistic `sgd`, log loss 0.4822 against 0.4773, AUC 0.7089 against 0.7089; BTCUSDT's Poisson `sgd` lost to the trees alone in every arm. No measured gain; what it would add, `pred` and every diagnostic on the target's own scale, is unmeasured | about 700 lines with tests; a schema bump | `offset=` on the regression kinds | low |
| 5 | **A reader of the libraries' model files**, for tree features on the command line | none beyond §8.3's recipe, which does the same in Python today, to the bit; and refitting leaves stops working when the trees' thresholds drift (planted 1.28, against 1.08 for §8.4's combination of retrains) | about 1,700 lines with tests; a fifth unstable interface | `trees=po.trees(...)` | medium |
| 6 | **Ranking groups through `po.gbdt`** | a capability, unmeasured | about 300 lines with tests | `qid=` / `group_column=` | low |

**What is a recipe today, measured.** None of these needs a build; each
is a page of the docs.

| recipe | measured |
|---|---|
| **a nowcast of recent labels, no key**, with `with_windows` over the doubled stream, the same feature at training and at serving through a saved state | flights MAE 24.24 → 21.82 (−10.0%), the largest single gain in this record; two runs through a saved state equal one run to the bit |
| **low-cardinality recent means** (origin, carrier) beside it | → 21.45 (−1.7% more) |
| rolling split-conformal: the 90% quantile of the residuals known in a recent window | flights 0.896 at a half-width of 46.4; 0.897 at 42.8 on the residual after an online bias |
| an online combination of the last five retrained boosters | planted, magnitudes drifting: 1.028–1.032, as good as refitting leaves (1.024–1.029); thresholds moving: **1.077–1.082**, where refitting leaves stops at 1.277–1.283 and the latest booster alone gives 1.083–1.087 (noise floor 1.0) |
| input drift against `po.gbdt`'s cuts (PSI) | a planted shift of half a standard deviation passes PSI 0.1 within 27–29 clock steps and 0.2 within 60–74 (half-life 20), where unshifted features stay below 0.013; BTCUSDT's daily-mean feature read 1.7–6.0 every day of the test period, the level outside the training range that §4's margin is for |
| frozen trees' one-hot leaves into `sgd` through `map_batches(streamable=True)` | planted drift 2.44–2.55 → 1.02–1.03; flights −1.6%; equal to the eager run to the bit |
| decay-weighted retrains, forward validation | planted drift only; uniform weights best on real data |

Deferred: the pairwise interaction screen (§10). Dropped: §11.

Outside the candidates: the cost of `with_windows(group=...)` grows with
the groups it has ever seen, not with the rows (§9.1, dispatched as its
own task); and XGBoost's callback route splits a ranking group at a batch
boundary and accepts ids that step back, without a word (§9.2).

---

## 1. How it was measured

**Data.** Every file is downloaded by `tests/data.py`'s helpers into
`.cache/` and read with `python -I` from scripts outside it (hard rule 1).

| data | rows | what it is | used in |
|---|---:|---|---|
| nycflights13 `flights` (Rdatasets) | 327,346 | every 2013 departure from New York, cancelled flights dropped, in scheduled-departure order; target `arr_delay` in minutes, or `arr_delay > 15`. Categoricals: `tailnum` 4,037 levels, `flight_id` (carrier and number) 5,706, `route` 223, `dest` 104, `carrier` 16, `origin` 3. A label is known only at arrival: `avail = t + dep_delay + air_time + 20` minutes; moving every label 30 or 60 minutes later costs 0.25 and 0.4 MAE in every arm, so the approximation hides no leak | §2–§9 |
| BTCUSDT 1-minute klines, 60 days from 2024-01-02 (Binance) | 86,390 | target `log1p(n_trades)` or `n_trades`, features lagged EW means of the target at 5, 30, 240 and 1,440 minutes, the last absolute return and its 30-minute mean, minute of day, weekday | §4, §5, §8 |
| planted | 200,000–300,000 | per section; seeds 0, 1, 2 | all |

**Splits are by time, never shuffled.** Flights: train January–August,
validate (early stopping) September, test October–December. BTCUSDT: days
1–40, 41–45, 46–60. Planted: the first 40–60% of the clock, the next 10%,
the rest.

**Every half-life, smoothing and categorical setting reported in v3 was
chosen on the earlier months only.** The selection fits trees on
January–July and scores each candidate by August's L2 at LightGBM's
early-stopping best; the test months are read once, after. v1's flights
half-lives (6, 2 and 4 hours) were chosen by hand; v3's replace them.

**The tree models differ between sections,** and a comparison is only
within one: 63 leaves at learning rate 0.05 on the four numeric features
plus `carrier` and `origin` as categoricals (§3, test RMSE 37.29 alone);
31 leaves on those plus `dest`'s code (§2, §4, §5: 37.39); 15 leaves at
0.1, about 15 trees (§6: 37.46).

**Load.** Each script logs `uptime` before and after. The runs shared an
Apple M4 Pro with other workers: load average 3 to 12 for every number
here, except where a run is noted otherwise; nothing heavy ran above 20.
No number here compares the speed of two machines.

---

## 2. `label_clock=` (rank 1)

### 2.1 What is wrong today

A label that arrives late is the rule in this record's data: a flight's
delay is known at its arrival, hours after its row. The library offers two
routes. A spec's `embargo` delays every label by one constant. The
*doubled stream* feeds each row twice: once at its own time, to be scored
and not learned (a null target, or weight 0), and once at its label's
time, to be learned. `po.stream.embargo` builds it for a constant delay,
and a delay that differs by row, as flights' does, can only be fed this
way, built by hand.

**The doubled stream learns the right labels at the right times, but every
residual diagnostic scores the wrong prediction.** At the learn row the
model predicts again, from hours of labels that arrived in between, and
`resid`, `sigma`, `conformal`, `emit_metrics`, `emit_calibration` and
`emit_drift` all fold that fresher prediction. The reviewer measured it
on flights' residual stream behind a frozen tree model, through an
intercept model with a 90% interval:

| rows of the doubled stream | coverage of `y − trees` | half-width | RMSE |
|---|---:|---:|---:|
| predict rows: the interval served at `t` | **0.8803** | 45.71 | 36.738 |
| learn rows: the interval read at `avail` | **0.8996** | 45.34 | 36.646 |

The tracker reaches its 90% exactly on what it scores, the learn rows, and
the served interval falls short.

### 2.2 What `embargo` already does

**`embargo` keeps, for each row it holds, the prediction the row was
served with, and the conformal radius of its interval; the release scores
the label against them** (`crates/online-polars/src/stream.rs`: the
stream's `pending: VecDeque<PendingRow>` and `score_pred`, schema 18,
review 2026-09-12 C21). Everything else the README's "Labels that arrive
late" lists, `weight_sum`, the `min_weight` gate, the metrics, the
interval, is read from the labels that had arrived. So the design is not
new machinery: it is `embargo` with a release time per row in place of
`clock + delay`.

### 2.3 Measured: scoring the served prediction closes the gap

`label_clock_sim.py` simulates the intercept model, its spread and the
bank's conformal tracker under both semantics on the same events. The
doubled semantics reproduce the bank's own run (served 0.8806 against
0.8803; learn rows 0.8990 against 0.8996). The served predictions are the
same under both (RMSE 36.713), since the model learns the same labels at
the same clocks; only what the diagnostics score differs.

| flights, 90% interval | served coverage | served half-width |
|---|---:|---:|
| doubled stream (today) | 0.8806 | 45.6 |
| `label_clock`, `conformal_rate` 0.05 (the default) | 0.8995 | **64.3** |
| `label_clock`, rate 0.01 | **0.8992** | **47.4** |
| `label_clock`, rate 0.002 | 0.8972 | 42.9 |
| rolling split-conformal, the last 7 days of known labels, on `y − trees` | 0.8964 | 46.4 |
| the same on the residual after the online bias | 0.8969 | 42.8 |

On the planted drift with each label 20 steps late on average
(exponential), the doubled stream covered 0.8976–0.8979 and `label_clock`
0.8988–0.9002 at rate 0.05, at half-widths 2.40–2.45 against 2.72–2.81;
at rate 0.01, 0.8997–0.9000 at 2.42–2.46.

**The default rate is too fast once hundreds of predictions are in
flight.** The tracker learns of a miss only when its label arrives; by
then the rows served since have been shown the same too-narrow radius, and
the misses arrive in a burst that the step overshoots. The bank shows it
today under a long `embargo`, which scores served predictions already
(`label_clock_bank.py`, flights, `embargo` 8 hours, which 99.4% of labels
beat):

| one stream, `embargo` 8 h | served coverage | half-width |
|---|---:|---:|
| `conformal_rate` 0.05 (the default) | 0.9034 | **188.9** |
| `conformal_rate` 0.01 | 0.8992 | 50.4 |
| `conformal_rate` 0.002 | 0.8973 | 43.3 |
| `resid_quantiles=[0.9]`: the EW quantile of the served residuals, no feedback | 0.8934 | 41.3 |

**So `label_clock` needs a smaller step under delay, and so does
`embargo` today.** A rate of 0.01 held 0.899–0.900 on flights and on all
three planted seeds, at half-widths of 47–50 on flights against the
default's 64 (simulated) and 189 (the bank under `embargo`). A step
divided by the square root of the predictions in flight was tried and
over-covered flights (0.913 at 49.2; at a base rate of 0.5, 0.905 at
69.2), so no automatic rule is proposed: the design keeps
`conformal_rate`, documents the measured choice, and leaves the default
under a delay to the user (§14).

### 2.4 The design

```python
spec = po.spec.ewridge("delay", targets=["arr_delay"], features=[...], clock="t", gap_cap="10h",
                       half_life="7d", label_clock="avail",   # when each row's label became known
                       conformal=0.9)
```

| | |
|---|---|
| the column | a clock of the spec's clock type, one per row; `str \| Sequence[str]`, one per target, so that targets known at different times need no second spec later. v1 may refuse a list whose entries differ |
| scoring | each row is scored where it sits, from the labels released before it, as under `embargo`; `resid_<t>` is null on the row, as for a window target, since the label is not known when the row is served |
| release | a held row is released before the first row whose clock is at or past its label time, in order of (label time, arrival); exactly at the time, released, as `embargo` releases exactly at the delay. At release its label is scored against the prediction, interval and spread it was served with, then learned |
| learning clock | a released row is learned at its label time, not its own: the model's clock only goes forward, and a short-delay row may be learned before an earlier long-delay one. `embargo` learns a row at its own clock; with one delay for every row the two differ by that constant, so every decay is the same and only a step measured from the newest learned row moves, which is why `holt` and `kalman` are refused below |
| `n_eff` (rule 8) | `weight_sum` on a row is the weight of the labels released before it, decayed to the row's clock, before its own decay: `embargo`'s meaning |
| a zero-weight row (rule 9) | scored, and not held: its release would learn nothing and every weighted diagnostic steps by its weight. It advances the clock at its own time |
| a label that never arrives | a null `label_clock` is a row scored and never learned, not held; counted in `labels_missing`. A label time before its row's clock is refused by name, as a misaligned column |
| memory | the rows in flight: per held row, its features, targets, weight, label time and served prediction, interval and spread per slot. `max_held_rows` (per group) bounds it, with `held_overflow="refuse"` (the default) or `"drop_oldest"`, counted in `labels_dropped` |
| the end of the stream | held rows are not learned, unless the state is saved and a run resumes from it, as under `embargo` |
| state (rule 5) | `pending` gains each row's label time and becomes ordered by it; a layout change: `SCHEMA_VERSION` bump, the current fixtures copied to `v<N>/`, a loader that reads an `embargo` file's deque with `clock + delay` as each label time, and fixtures freezing a buffer released out of arrival order, a row held past the stream's end and a `labels_missing` count |
| chunk invariance (rule 3) | a release depends on the row sequence and the label times alone, never on where a chunk ends; the held rows live in the state between chunks, as `embargo`'s do |
| interactions | refused beside `embargo` (which it generalizes) and beside a window target (which has its own release); `holt` and `kalman` refused in v1, whose forecast and `error_inflation` reach from the newest learned clock, which would become a label time; a reset or a closed group drops its held rows, counted, as `embargo`'s reset does |
| the command line | `label_clock = "avail"` in the spec's TOML |
| cost | per row, one push into an ordered buffer and the releases due: O(log held) beside the model's own step; memory as above. It also halves the rows a late-label stream feeds, and removes the sort of the doubled stream by a second clock |

**Tests and their oracle.** The doubled stream itself: the served
predictions must equal the doubled stream's predict rows to the bit (the
same labels, learned at the same clocks); the residual diagnostics must
equal a numpy fold of `y − served` in release order; and for every kind it
takes, `embargo=d` must equal `label_clock = clock + d` to the bit.

**What would make us rethink it:** a label that arrives in pieces, or is
revised after it arrives. Neither is in the data here; a revised label
would be a second learn event for one row, which this design refuses
rather than guesses at.

---

## 3. Target statistics of a high-cardinality categorical (rank 2)

### 3.1 What it is

A categorical feature with thousands of levels gives a tree little to
split on. The standard remedy replaces the level by a statistic of the
target over the rows of that level, a *target statistic* (TS); over all
rows it leaks the row's own label. CatBoost's answer (Prokhorenkova et al.
2018, §3.2, eq. 5) is the *ordered* TS, each row's from the rows before it,
shrunk to a prior:

```
x̂_k = (Σ_{j<k, c_j=c_k} y_j + a·p) / (#{j<k : c_j=c_k} + a)
```

On a stream the order is time and the statistic is hard rule 2 itself.
What this library adds is a decay on the clock, labels folded in when they
arrive, a state that continues from training into serving, and bounded
memory.

### 3.2 Measured: the gain, decomposed

**Most of the gain needs no categorical at all** (the reviewer's finding,
re-measured with every setting chosen on August; `ts_decompose.py`). Each
arm adds a group to the one above; test MAE and RMSE for LightGBM seeds
1/2/3:

| arm | chosen on August (half-life, smoothing `a`) | MAE | RMSE |
|---|---|---|---|
| base: the numeric features, `carrier` and `origin` as categoricals | | 24.250 / 24.242 / 24.233 | 37.29 |
| + the recent mean of every known label, **no key** | 60 min | 21.814 / 21.849 / 21.796 | 34.96–35.05 |
| + recent means by `carrier` and by `origin` (16 and 3 levels) | carrier 120 min, a 10; origin 60 min, a 10 | 21.454 / 21.455 / 21.453 | 34.63–34.65 |
| + recent means by `tailnum`, `dest`, `route`, `flight_id` | 120 min a 1; 120 min a 1; 480 min a 10; 960 min a 1 | **20.706 / 20.739 / 20.706** | 34.12–34.20 |
| + a 7-day statistic of each of the four (a 10) | August declined it: L2 1,359 against 1,352 | 20.569 / 20.538 / 20.593 | 33.98–34.03 |

Each recent mean's prior is the keyless mean (`prior="running"` at 60
minutes); the shortest half-life tried was 60 minutes, so the keyless
nowcast's choice sits at the grid's edge.

| part | MAE change | its spread over seeds |
|---|---:|---:|
| recent labels, no key (a recipe today, §8.1) | −10.0% | ±0.03 |
| low-cardinality keys (a recipe today) | −1.7% | ±0.01 |
| **high-cardinality keys** (what the kind is for) | **−3.4%** (−4.1% with the 7-day statistics) | ±0.03 |

**Batch baselines, tuned on the earlier months** (`ts_baselines.py`,
`ts_catboost.py`). Each fits the same base features and splits:

| arm | MAE | RMSE | rounds |
|---|---|---|---:|
| the high-cardinality columns dropped, learning rate 0.05 | 24.250 | 37.29 | 22 |
| the same at 0.01, patience 300: it had converged | 24.241 | 37.28 | 103 |
| LightGBM's native categorical, defaults | 24.516 | 37.55 | 13 |
| native, 8,192 bins per categorical so no level folds into "other", `cat_smooth` 50, `min_data_per_group` 200, `max_cat_threshold` 32, `cat_l2` 1, chosen on August from 12 of 36 settings | 24.473 / 24.484 / 24.482 | 37.44–37.46 | 12–15 |
| scikit-learn's `TargetEncoder`, 5-fold cross-fitting, at 0.01 | 24.462 | 37.49 | 76 |
| **CatBoost, `has_time=True`**, the six categoricals as categories, depth 6, 0.05, early stopping on September | 24.415 / 24.396 / 24.406 | 37.52–37.55 | 18–21 |
| a monthly retrain: trees and `TargetEncoder` refitted on October 1, November 1 and December 1 on every label known by then | 24.005 | 37.56 | |
| CatBoost with the online statistics added as features | 21.106 / 21.262 / 21.284 | 34.09–34.22 | 1,640–1,971 |
| the monthly retrain with the online statistics added | 20.588 | 33.95 | |
| **the online statistics, trained once** (the 7-day row above) | **20.57** | **34.00** | |

Every static encoding, CatBoost's ordered statistic included, stops within
twenty rounds: on these months the static features describe January to
August and not the autumn, and a monthly refit recovers 1% of it. The
continuing statistics recover 15%, and a monthly retrain adds nothing on
top of them. CatBoost given the same online statistics reaches 21.1–21.3,
behind LightGBM's 20.6 with them, so its own ordered statistic adds nothing
the continuing ones do not already carry.

**The classification form** (v1's `ts_flights_binary.py`, half-lives by
hand): `arr_delay > 15`, AUC 0.640 dropped, 0.658 with 30-day rates, 0.746
with recent rates beside them; native and scikit-learn 0.625 and 0.624.

**Planted** (`ts_synth.py`, v1): a Zipf(1.3) categorical over 20,000
ranks, half the test period's levels unseen in training, half the effects
drifting. RMSE (noise floor 1.0), seeds 0/1/2: dropped 1.50/1.52/1.51;
native 1.50/1.52/1.51; greedy (leaks) 1.46/1.48/1.48; scikit-learn
1.55/1.55/1.56; ordered, frozen at the cut 1.45/1.47/1.46; **ordered,
continuing 1.16/1.17/1.17**. The case plants the continuing encoder's best
case (half the test levels new) and shows the mechanism, not its size.

### 3.3 Bounded memory

| planted, equal slots | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| unbounded (22,600 levels) | 1.1661 | 1.1732 | 1.1760 |
| cap of 4,096, the least decayed weight evicted | **1.1883** | **1.1988** | **1.1990** |
| one hashed table of 4,096 cells | 1.2521 | 1.2592 | 1.2644 |
| count-min, 4 × 1,024, the cell of least weight read | 1.2985 | 1.3096 | 1.3113 |
| cap of 512 | 1.4667 | 1.4859 | 1.4825 |
| one hashed table of 512 | 1.4331 | 1.4419 | 1.4433 |

| flights, every table capped (`ts_flights_cap.py`, v1's half-lives) | RMSE | MAE |
|---|---:|---:|
| all levels (`tailnum` 4,037, `flight_id` 5,706) | 34.472 | 20.969 |
| 1,024 | 34.377 | 21.005 |
| 256 | 34.453 | 21.204 |
| 64 | 34.852 | 21.996 |

**A cap errs toward the prior; a hashed table errs toward a mixture of
levels.** An evicted or new level reads the prior, as an unseen one does,
and its support (below) says so to the tree. A hashed table merges
unrelated levels, a bias with no sign. On flights a cap of a quarter of the
levels costs nothing: a plane that has not flown for weeks has the least
weight.

### 3.4 The recipe it is today

**The decayed, label-time statistic is exact with the window operators**
(`ts_recipe.py`): over the doubled stream,

```python
out = stream.online.with_windows(
    S=po.ewm_sum("y", half_life=hl, closed="left"),      # the decayed sum of labels before the stamp
    W=po.ewm_sum("one", half_life=hl, closed="left"),    # their decayed weight; null where none yet
    clock="c", gap_cap=20 * hl, group="tailnum",
)
```

agrees with the prototype's sums to a relative 3e-13. One call per
column, one more for the prior. **Its cost is not a blocker:** with
`gap_cap=1e12` it ran at 24 µs a row on 4,037 groups (§9.1), but a
finite `gap_cap` cuts a group idle that long, and at 20 half-lives the cut
changes a statistic by at most 2⁻²⁰ of its weight; the reviewer measured
533 ns a row on 10,000 interleaved groups at `gap_cap=50`. The 7-day
statistics need a long cap, and pay the walk until §9.1's fix lands.
**So the kind is not gated on §9.1.** It is for what the recipe cannot do
in one pass: many columns, a shared running prior, a cap, the
classification form.

### 3.5 The design: a model kind

```python
te = po.spec.target_encoder(
    "te", targets=["arr_delay"],
    features=["tailnum", "dest", "route", ("dest", "route")],   # a tuple is one key of several columns
    clock="t", gap_cap="2d", label_clock="avail",
    half_life={"tailnum": "2h", "dest": "2h", "route": "8h"},   # a scalar, a list, or a mapping per feature
    smooth=1.0,                 # scikit-learn's name for a; "auto" once v2 reads the levels' variance
    prior="running",            # the target's own decayed mean before the row; or a number
    prior_half_life="1h",       # the keyless nowcast's half-life
    max_categories=100_000,     # per feature; the level of least decayed weight is evicted past it
)
```

| per row, per feature `f`, half-life `h`, target `t` | |
|---|---|
| the level | **the value itself, typed**: a `pl.String` or a `pl.Enum`'s category string, an integer column's `i64`, a Boolean; a float column is refused by name (a float category is almost always a mistake), and the feature's dtype is in the state, so a resume on another dtype is refused. `1`, `1.0` and `"1"` are therefore three columns' values, never one level. A null is a level of its own, as `group_by` keeps it. A tuple is the tuple of its columns' values |
| how the model sees it | the bank keeps, per feature, a dictionary value → dense id, as it keys groups by their text (`windows.rs:1346`); the model receives the id in its `f64` features (exact below 2⁵³), so `OnlineModel` does not change, and the dictionary is the bank's state beside the model's. An evicted level leaves the dictionary too, so `max_categories` bounds both. A hashed, value-free mode (xxh3-64, linked already) can come later, as an option |
| read, before the row is learned | `enc = (S + a·p) / (support + a)`; `a = 0` with no support gives null, never 0/0 (rule 9) |
| learn, at the label's release | `W' = λW + w`, `S' = λS + w·y`, `Q' = λQ + w·y²`, `λ = 0.5^(Δt/h)` from the level's last label; a null target or a weight of 0 learns nothing and creates no level |
| emitted | `enc_<f>_<t>`; `support_<f>_<t>`, the level's label weight decayed to the row's clock (`λW`), which drives the shrinkage and is *not* rule 8's quantity, so it does not say "weight"; and `weight_sum` with rule 8's meaning. A tuple's stem is its columns joined by `__` (`enc_dest__route_arr_delay`), and a column whose name holds `__` is refused beside a tuple |
| several half-lives | each feature's fields carry the instance suffix `ewridge`'s grid gives its slots; one key table per feature holds every half-life's sums, and eviction reads the longest |
| classification | `classes=[...]` as `ew_class` takes them: one `S` per class, `enc_<f>_<class>` a rate, the prior the classes' running shares |
| eviction | past `max_categories`, the level of least decayed weight, ties by the smaller id. Every level decays by the same factor, so the order is that of `ln W + ln 2 · t_last / h`, which does not change with time, and a sorted set gives the victim in O(log n). **The key is never persisted**: the sorted set is rebuilt on load from (`W`, `t_last`), so no libm value enters the state |
| state | per feature: the dictionary, and per level (`W`, `S`, `Q` per target or class, `t_last`) for each half-life; about 56 + 16T bytes a level beside its value. Written sorted by value, so the bytes do not depend on insertion order |
| export | `bank.encodings("te")`: a frame of feature, level, half-life, target, `enc`, `support`, the level's spread, so a table can be read, served from SQL or a feature store, or inspected; the evicted levels are counted, not kept |

**Guarantees.** Rule 2: read before the row's label; a late label learned
at its time through `label_clock` (§2). Rule 3: one row at a time,
deterministic eviction. Rule 8: `weight_sum` as everywhere; `support_` is
named apart. Rule 9: above. Rule 5: a new `ModelState` variant, a bank
dictionary, a schema bump, and fixtures freezing an evicted level, a null
level, a tuple key, each dtype of key and several half-lives. **A near-tie
in eviction is a discrete divergence between operating systems** (one
keeps a level's whole statistics, the other loses them), not a last-bit
one, so the fixtures and the cross-OS tests run on `lam=` given as a
literal at a step of one, whose decays are products, with no libm. Rules
12 and 13: nothing linked; no unstable interface.

**Tests and their oracles.** Undecayed, fixed prior, no delay:
`category_encoders.CatBoostEncoder`, the ordered TS in row order
(`ce_oracle.py`: 1.2e-14; its own prior is the mean of every label, the
future included, so the test passes `prior=` its number). Decayed: the
definition by hand from eq. 5 with exponential weights. Late labels: the
`with_windows` recipe of §3.4. `smooth="auto"`, when built: scikit-learn's
`TargetEncoder(smooth="auto")`, undecayed, on one fold.

**Cost.** One dictionary lookup, one `exp` and a few flops per feature per
half-life per row: an estimate of 50–100 ns, beside 107 ns a row for a
five-feature `ewridge`. A million levels of one target: about 70 MB with
their values.

### 3.6 Durability

| a user will want | taken |
|---|---|
| several targets; classification; weights; groups | `S`, `Q` per target; `classes=`; `weight=`; `group=` |
| windows | refused in v1 by name; `window_size` would follow `ewridge`'s snapshots |
| a half-life per feature, several per feature | a scalar, a list, or a mapping |
| `smooth="auto"`, a per-level spread as a second encoding | `Q` is stored from v1 |
| the table itself | `bank.encodings()`, from the stored values |
| combinations of categoricals | tuples, from v1 |
| unbounded cardinality | `max_categories` |
| labels at their own times | `label_clock`, the bank's, not the kind's |
| a low-memory mode | hashed keys as an option, beside the values |

**What would make us rethink it:** a keyed window operator in `po.ops`,
which would make the kind a second way to do §3.4's recipe. The kind is
justified by the four things that recipe cannot do in one pass; if the user
does not want them, the recipe and §9.1's fix suffice.

---

## 4. A base margin through `po.gbdt` (rank 3)

**The libraries fit trees around a given margin,** `base_margin` in
XGBoost and `init_score` in LightGBM. An online model's prediction is out
of sample by construction (rule 2), so as a margin it leaks nothing.

**Measured** (`margin_synth.py`, `margin_btc.py`, `flights_more.py`).
Planted: 100 rows per clock step over 2,000 steps, a level that random-walks
and that no feature carries; the online model an `ewridge` or `ftrl` at a
5-step half-life.

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| the test level against the training range | inside | inside | **outside** (−9.5 to −6.0, training −4.5 to 0.5) |
| Gaussian RMSE: trees alone | 1.777 | 1.742 | 5.825 |
| the online prediction as a feature | 1.026 | 1.024 | 3.131 |
| **as the margin** | **1.021** | **1.022** | **1.023** |
| Bernoulli log loss: trees alone | 0.516 | 0.386 | 0.389 |
| the online logit as a feature | 0.480 | 0.353 | 0.204 |
| **as the margin** | **0.479** | **0.354** | **0.152** |

A tree cannot predict outside the range it split on; a margin is added
after it. Real data: BTCUSDT `n_trades`, Poisson, log-RMSE 0.4682 alone,
0.4652 with the feature, 0.4603 with the margin; flights, the origin's
recent delays (31-leaf trees): 37.39 alone, 35.13 as a feature, 36.00 as
the margin, 35.22 both. A margin is the safe choice where the level can
leave the training range, and a feature can be better where it cannot;
passing both was within 0.09 of the better on both real datasets.

**Checked in the libraries.** XGBoost's callback route carries a
`base_margin` given batch by batch (`qid_check.py`), and adds it in place
of `base_score` (`margin_btc.py`: 1.5e-5, float32 rounding). LightGBM's
`LGBM_DatasetPushRowsWithMetadata` takes `init_score` per batch once
`LGBM_DatasetInitStreaming` is told `has_init_scores`, and skips
`boost_from_average` when one is set (`gbdt.cpp:331`).

**Design.** `po.gbdt.xgboost_matrix(..., base_margin="col")` and
`po.gbdt.lightgbm_dataset(..., init_score="col")`, each the name its
library's Python API uses, as `ref=` and `reference=` already are. A
`List`/`Array` column of width K for a K-class objective; the validation
matrix takes its own column; a null or non-finite margin is refused; the
cuts do not depend on it. The docs must show the margin added at
prediction (an XGBoost `DMatrix` with `base_margin`, or LightGBM's
`raw_score=True` plus the margin). LightGBM's Python engine cannot stream
it and holds n·K doubles. **Tests:** the library's in-memory route with the
margin, identical trees by hash. No state; no schema change.

---

## 5. `offset=`, and a model of the intercept alone (rank 4)

**An offset is a column added to a model's linear predictor,** the concept
of `glmnet`'s and `statsmodels`' `offset`, and of XGBoost's `base_margin`.

**Measured inside a link** (`offset_link.py`, the reviewer's request). The
bank has no offset, but `sgd` can hold one exactly: the margin as a
feature whose coefficient is pinned at 1 (`coef_min = coef_max = 1`,
`standardize=False`; the starting point is projected, so it holds from the
first row). Each arm's learning rate was chosen on the validation period
from 0.01, 0.05 and 0.2.

| | trees alone | `sgd` alone | the margin as a free feature | **the margin as an offset** | the offset, intercept alone |
|---|---:|---:|---:|---:|---:|
| BTCUSDT `n_trades`, Poisson deviance | **355.5** | 419.5 | 450.9 | 395.3 | 362.3 |
| flights `arr_delay > 15` (labels at arrival), log loss | 0.5128 | 0.4933 | **0.4773** | 0.4822 | 0.4887 |
| the same, AUC | 0.6366 | 0.6798 | **0.7089** | **0.7089** | 0.6881 |

On flights the online layer, on two recent-rate features, adds a great
deal to the trees (log loss 0.513 → 0.477), but an offset adds nothing over
the same margin as a free feature. On BTCUSDT the Poisson `sgd` could not
beat the trees in any arm. **So `offset=` has no measured gain**; what it
would add is convenience: `pred`, `resid`, `sigma` and the interval on the
target's own scale without adding the margin back, and `features=[]` in
place of a column of zeros. For squared error it is a recipe today (target
`y − margin`). It ranks on that argument alone.

**Design, if built.** `offset: str | Sequence[str]`, one column per
target in `targets`' order, a length mismatch refused by name; a
`Mapping[target, column]` may come later without a break. The prediction
is `offset + f(x)` in the kind's link space (identity; the log of `sgd`'s
Poisson; the logit of `ftrl` and `sgd`'s logistic); residuals and
diagnostics on `y − pred`; a null offset withholds the row as a null
feature does; the offset must be known before the row's label, as a
feature must. statsmodels' `exposure=` for Poisson (the log of a column)
can come later beside it. `features=[]` with `fit_intercept=True` becomes
legal.

**Its state cost.** A spec field changes the bytes of every bank file,
since each carries its specs (EXTENDING.md, "Adding an output or a
parameter instead"), and this library is past 1.0: a `SCHEMA_VERSION`
bump, the current fixtures copied to `v<N>/`, and a loader. `features=[]`
is a new form of every regression kind's state (no slopes), so each kind
that takes it needs a fixture with none. **Tests:** statsmodels' GLM with
`offset=` (Poisson, binomial) and least squares on `y − offset`, at no
decay, against the kinds' fits.

**Naming.** Polars' `rolling` and `group_by_dynamic` use `offset` for a
window's shift. The specs share their clock keywords with `with_windows`,
which follows Polars' window API, so `with_windows` must never take
Polars' `offset`; a shift there would need another name.

---

## 6. A reader of the libraries' model files (rank 5)

### 6.1 What it would add

He et al. 2014 (Facebook's click prediction, §3.1) take the leaf each row
falls into in each boosted tree as a one-hot feature of a linear model; §3.2
finds that training daily rather than weekly cuts normalized entropy by
about 1%, and proposes retraining the trees every day or few while the
linear layer learns online. §8.3 does exactly that in Python today. A
reader would put it on the command line and on a server without XGBoost or
LightGBM installed, and nowhere else.

### 6.2 Measured

**Where refitting leaves works, and where it stops** (`leafrefit.py`,
`leafrefit2.py`, `boosters_k.py`). Planted, 300,000 rows on 3,000 steps,
trees on the first 40%, test on the last 50%, seeds 0/1/2:

| arm | magnitudes drift (thresholds fixed) | structure drifts (thresholds move) |
|---|---|---|
| the frozen trees | 2.547 / 2.440 / 1.519 | 1.531 / 1.544 / 1.534 |
| `sgd` on their one-hot leaves | **1.025 / 1.029 / 1.024** | 1.277 / 1.283 / 1.278 |
| a booster retrained every 150 steps on the last 300, served alone | 1.077 / 1.092 / 1.081 | 1.083 / 1.087 / 1.083 |
| `ewridge` over the last 5 retrained boosters | 1.028 / 1.029 / 1.032 | **1.079 / 1.082 / 1.077** |
| LightGBM's `refit(decay_rate=0.5)` every 50 steps | 1.058 / 1.064 / 1.048 | |

(The magnitudes case: `y = a(s)·sin(2x0) + b(s)·[x1 > 0.5] + c(s)·x2·[x3 > 0]
+ level + noise`; the structure case: `y = sin(2x0) + 1.5·[x1 > a(s)] +
1.5·x2·[x3 > b(s)] + noise`, `a` from 0.5 to −1.0, `b` from −1.0 to 1.0;
noise floor 1.0.) **Refitting leaves works only while the trees' cuts stay
right:** when the thresholds move, no reweighting of the old leaves finds
the new boundary, and it falls behind a plain retrain. The combination of
the last five retrains is within 0.005 of the best arm in both cases, and
needs no forest reader (§8.4).

**Real data.** Flights (15-leaf trees at 0.1): frozen 37.459, `ewridge` on
each tree's leaf value 36.856 (−1.6%), one-hot 37.076. BTCUSDT: no gain.

### 6.3 The reader, de-risked

**Readers of XGBoost's JSON and LightGBM's text model give each row's
leaf exactly as `pred_leaf` does** (`tree_reader.py`): 0 mismatches over
20,000 rows × 60 trees for each of four models, with categorical
partitions and bitsets, all three of LightGBM's missing types, 10% nulls,
values on the thresholds, unseen and negative codes, zeros and ±∞.

| library | numeric | missing | categorical |
|---|---|---|---|
| XGBoost (`predict_fn.h`, `categorical.h`) | in float32: `x < split_condition` goes left | the node's `default_left` | a category in the node's set goes **right**; a negative code or one past 2²⁴ goes left |
| LightGBM (`tree.h`) | in float64: `x <= threshold` goes left | by `decision_type`'s bits 2–3: `NaN` sends a NaN by `default_left`; `Zero` sends \|x\| ≤ 1e-35 (a NaN read as 0) by it; `None` reads a NaN as 0 | a category in the node's bitset goes **left**; NaN or a negative code goes right |

XGBoost's `pred_leaf` returns node ids, sparse over all nodes; the reader
renumbers each tree's leaves 0..L−1 in node order, and `tree<k>_leaf<l>`
names that index, so the one-hot width is the leaf count, not the node
count.

### 6.4 The design, if built

**In `online-polars`, beside `po.gbdt`**: a parser of third-party formats
is plumbing (rule 4), and the models never see it; the bank appends the
encoding to a kind's features.

| | |
|---|---|
| files | XGBoost JSON (its `version` read, a version outside the range read refused by name), LightGBM's text model; a path, bytes, or a `Booster` saved through its own `save_raw("json")` / `model_to_string()` |
| features | by the model's `feature_names`; a categorical as a `pl.Enum`'s codes. A model whose JSON stores category names (trained on pandas categoricals, XGBoost 3.x) is refused by name; a model `po.gbdt` built never has them |
| `encode` | `"one_hot"` (width Σ leaves) or `"leaf_value"` (width M; refused for LightGBM's linear trees and XGBoost's DART); a one-hot column is never standardized |
| width | O(width) a row for `sgd`, `pa`, `ftrl`; the kinds that solve refuse a one-hot width past a budget. A sparse update (M ones in Σ leaves columns) is the follow-up when a forest's width makes the dense row the cost |
| identity | the model's bytes hashed with xxh3-128 into the spec; a state loads only against the same forest |
| swapping a forest | **the online layer cannot be carried across a retrain**: its coefficients belong to the old forest's partition, and nothing maps a new tree's leaves onto the old ones. Two answers, both available now: two specs over an overlap, the new one warming while the old one serves, with a cut-over the user chooses; or the combination of §8.4, which carries the retrain cadence without any forest reader |
| the command line | `trees = { path = "model.json", encode = "one_hot" }` in the spec's TOML |

**Linking (rule 12), from `cargo tree -i`:** `serde_json` is a normal
dependency of `online-polars` already (`crates/online-polars/Cargo.toml`,
and through `polars-error`); `xxhash-rust` 0.8.18 with its `xxh3` feature
is linked into it through `polars-core`. Adding either as a direct
dependency at that version and feature set links nothing new. `sha2` is in
`Cargo.lock` only as a build dependency of `polars-plan`, so SHA-256 would
be new code in the wheel and the binary: the design uses xxh3-128 instead.

**Rule 5:** `trees=` is a spec field, so the same bump and loader as §5.
**Rule 13:** a fifth unstable interface, the model file formats; XGBoost
documents its JSON schema, LightGBM's text format is not a promised
contract. **Tests:** `pred_leaf` to the bit, for every model shape.

---

## 7. Ranking groups through `po.gbdt` (rank 6)

`po.gbdt.xgboost_matrix(..., qid="col")` and
`po.gbdt.lightgbm_dataset(..., group_column="col")`. `group_column` is
LightGBM's name for a column of query ids in its command-line and file
parameters; its Python `Dataset` takes `group=` as sizes, which a stream
cannot give before it ends. **The ids need only be contiguous:** natural
ids (a session hash, a UUID) are not sorted. The build renumbers each
contiguous run to a running counter, which is what both libraries need,
and cuts XGBoost's batches at a run's boundary, carrying a run's head into
the next batch (§9.2). An id that comes back after another cannot be seen
in bounded memory; `sorted=True` asserts the ids are non-decreasing and
refuses a step back. XGBoost weights a ranking objective per group, so
`weight=` beside `qid=` is refused in v1. Value: a capability, unmeasured.
**Tests:** the in-memory route with `qid`/`group`, identical trees.

---

## 8. Recipes, measured

### 8.1 A nowcast of recent labels, the same at training and at serving

The largest gain here needs no build: the decayed mean of every label
known before the row, through the doubled stream (or, once built,
`label_clock`), `with_windows(S=po.ewm_sum("y", ...), W=po.ewm_sum("one", ...))`
with no group. Flights MAE −10.0% (§3.2). **It is the same feature at
training and at serving**: computed over January–September with
`save_state=`, then October–December with `load_state=`, it equals one
run over the year to the bit (`parity.py`: 654,692 doubled rows in 0.33 s, equal: `True`). That is the library's
distinctive offer to a batch GBDT, whatever else is built.

### 8.2 Rolling split-conformal

The 90% quantile of `|y − pred|` over the labels known in a recent window,
one `rolling_quantile_by` over the label times: flights 0.8964 at a
half-width of 46.4 (7 days; 0.889–0.896 for 1 to 30 days), 42.8 on the
residual after an online bias; planted 0.893–0.899 for every window, at
half-widths 3.80–3.94 on the raw residual and 2.39–2.43 after the online
bias (`label_clock_sim.py`; the reviewer's `e2`, `e5`). It matches the
online tracker's coverage with no build; what it cannot give is an
interval on the row as the stream runs, inside the bank.

### 8.3 Frozen trees' leaves into an online model

`booster.predict(..., pred_leaf=True)` one-hot encoded chunk by chunk with
`LazyFrame.map_batches(..., streamable=True)`, into `sgd` through the IO
plugin (`leaves_recipe.py`): equal to the eager run to the bit, 1.0250 on
the planted magnitudes, 0.51 s for 300,000 rows. `standardize=False`: a
centered one-hot column spreads every update over every coefficient.

### 8.4 An online combination of the last K boosters

Retrain a booster on a schedule (here every 150 steps on the last 300),
let each score the rows after its own training data, and give an online
model one column per age: the newest booster's prediction, the one before
it, and so on to the fifth. `ewridge` with no intercept, half-life 20
steps, learns the weights as labels arrive (`boosters_k.py`). It is He et
al.'s retrain cadence carried by the library's online layer, without a
reader of the model files. Planted, seeds 0–2: 1.028–1.032 under drifting
magnitudes (refitting leaves: 1.024–1.029; the newest booster alone:
1.077–1.092), and 1.077–1.082 under moving thresholds (refitting leaves:
1.277–1.283; the newest alone: 1.083–1.087). Not measured on real data.

### 8.5 Input drift against `po.gbdt`'s cuts

A deployed GBDT fails silently when its inputs leave the ranges it split
on (§4's seed 2). `po.gbdt`'s own matrix holds the cuts it binned on
(`QuantileDMatrix.get_quantile_cut()`); the training rows' share in each
bin is the reference, a serving bin's share is decayed on the clock, and
the population stability index `Σ (q − p) ln(q / p)` is read per feature
(`psi_drift.py`, each share floored at 1e-4, 32 bins). Planted, a shift of
half a standard deviation in one of three features at step 2,000 of 3,000,
half-life 20 steps: its PSI passed 0.1 after 27–29 steps and 0.2 after
60–74, while before the shift it stayed below 0.010 and the unshifted
features below 0.013 (seeds 0–2). BTCUSDT against days 1–40, half-life one
day: the 30-minute mean read 0.47–1.96 on days 41–60, and the daily mean
1.7–6.0, every day: its level had left the training range, which the
trees' predictions cannot follow. The decayed histogram is a recursion per
bin, which `with_windows` can run as one `po.ewm_sum` per bin; a dedicated
operator would be worth building only if the recipe's width (features ×
bins columns) becomes the cost.

### 8.6 Decay-weighted retrains, and forward validation

`weight = 0.5 ** ((cut − t) / half_life)` is one Polars expression. On the
planted drift it cut the next period's RMSE from 2.32 to 1.26 (seed 0), but
on real data uniform weights were best: flights 37.40 against 37.55–37.78,
BTCUSDT 0.4493 against 0.4493–0.4586 (`decayw.py`). A forward validation
slice against a shuffled one was mixed: under drift it removes the newest
rows from training. A paragraph in GBDT.md.

---

## 9. Findings outside the candidates

### 9.1 `with_windows(group=...)`: a wasted walk, cured by `gap_cap`

| groups (one `ewm_sum`, `closed="left"`) | ns a row, 100,000 rows | 400,000 rows |
|---:|---:|---:|
| 10 | 259 | 168 |
| 100 | 516 | 512 |
| 1,000 | 3,931 | 3,959 |
| 10,000 | 79,905 | 88,386 |

**The reviewer's analysis, taken:** `flush_stamps_before`
(`windows.rs:2357`) walks the list of groups in order of their last row
while each one's stamp has passed, calling `flush_group`, which returns at
once when nothing is pending. The list holds every group with a clock and
loses one only through `end_group` or a reset, so with `gap_cap=1e12` it
holds every group ever seen. Under `closed="left"` nothing is ever pending
(`n_wait` is 0, line 2112), so the whole walk is wasted. The cost grows
with the groups ever linked, not with interleaving: 10,000 groups seen once
early and then 10 active cost 79,066 ns a row, and `gap_cap=50` cures each
case (533, 244 and 251 ns). **The fix:** a second list, the groups with
rows pending, which in clock order are the groups at the previous stamp, a
`Vec` cleared at each new stamp, `#[serde(skip)]` and rebuilt on load from
the groups whose `pending` is non-empty; `head`/`tail` stay as saved, so no
windows-state version change. Dispatched as its own task. It does not gate
the target encoder (§3.4).

### 9.2 XGBoost's callback route and ranking groups

`qid_check.py` builds a `QuantileDMatrix` from two batches through
XGBoost's `DataIter`: a query group split across the batch boundary comes
back as two groups (`group_ptr` [0 2 3 4 6]), and ids that step back from
one batch to the next are accepted (`MetaInfo::Extend`, `data.cc:717`: each
batch's ids are run-length encoded on their own; the non-decreasing check,
`data.cc:621`, is per batch). LightGBM's streaming route keeps one array of
ids and finds the boundaries once, at `FinishLoad` (`metadata.cpp:191`), so
a group may straddle pushes there; an id that comes back later starts a new
group, also silently.

---

## 10. Deferred: a pairwise interaction screen

A B × B table of `(Σw, Σw·r, Σw·r²)` per feature pair, `r` the target or an
online model's residual, ranked by a two-way ANOVA interaction F, fits in
`marginal`'s design. On 20 planted features with four interacting pairs
(`pairs.py`) the F found them with average precision 0.50–0.77; FAST (Lou
et al. 2013, §4.1) scored 0.04 on the raw target and needs the main effects
removed at the table's own resolution (0.69–0.77 at 8 bins on an 8-bin
additive residual, 0.04–0.08 at 16); InterpretML's FAST agreed (0.04 on the
target, 0.16–0.29 with the residual as `init_score`). Products of the top 8
pairs took a linear model from 1.337 to 1.148 test RMSE (the true pairs:
1.148), and XGBoost constrained to them scored 1.025 against 1.034. On
BTCUSDT and flights (`pairs_real.py`) the pairs bought nothing, and the F
of every real pair sat far above 1, serial dependence inflating it; a
ranking would need the serial count `marginal(lags=)` keeps. Deferred until
a real dataset shows a gain.

---

## 11. Dropped

| idea | why |
|---|---|
| monotone constraints from `marginal`'s slopes | a marginal slope's sign is not the partial dependence's under correlated features; a wrong constraint is a silent bias. Not measured |
| online quantile cuts replacing `po.gbdt`'s sample | GBDT-ADAPTER §4.1 measured the t-digest slower than a sample and no better |
| missing-value indicators | both libraries learn a default direction per split (§6.3's table) |
| a count encoding of its own | not shown to help: on one seed, adding decayed frequency and the label weight together moved flights' RMSE by +0.11, within the 0.06 the seeds vary by. `support_<f>_<t>` comes with the encoder anyway |
| LightGBM's `refit` / XGBoost's `refresh` behind a library call | close to the online refit (1.05–1.06 against 1.02–1.03), but needs the library and a loop at serving; a recipe if wanted |
| target transforms, lag and window features | Polars' `with_columns` and `po.ewm_*` already make them |

---

## 12. Self-review of v1 (v2)

v1 ranked the items and designed each; v2 attacked them as a user a year
on would. What it changed is folded into the sections above; v3 overturned
part of it (§13).

| item | the attack | v2's answer |
|---|---|---|
| target encoder | the `with_windows` recipe already does it | true, and it is the oracle; v2 ordered "fix §9.1 first", which v3 withdraws (§9.1 is cured by `gap_cap` and does not gate the kind) |
| target encoder | a cap that thrashes | measured on real data (§3.3) |
| target encoder | labels with their own times | named the bank-level `label_clock=`; v3 designs it and ranks it first |
| target encoder | short and long half-lives at once | a list; v3 adds a mapping per feature |
| target encoder | one real dataset | still one; the planted case shows the mechanism |
| frozen trees | Python does it today | measured true (§8.3); the reader ranks below `offset=`, and in v3 below `offset=` and the margin |
| frozen trees | standardized one-hot; very wide forests; trees in sample; a newer XGBoost | never standardized; sparse later; documented; refused by name |
| `offset=` | a target transform does this | for squared error; v3 measured the link case: no gain |
| `offset=` | intervals under-cover with late labels | v2 blamed the delay; v3 finds the doubled stream scores the wrong prediction (§2.1) |
| base margin | worse than a feature on flights | kept in the table; pass both when unsure |
| ranking groups | groups straddling batches | cut at boundaries; v3 asks only for contiguous ids |

---

## 13. Review response (v3)

A fresh reviewer read v2 and re-ran parts of it (load 4.3–5.8). Each
finding, and what changed:

| # | finding | what changed |
|---:|---|---|
| 1 | rank 1's gain comes mostly from recent labels with no key | taken and re-measured with every setting chosen on August (§3.2): no key −10.0%, low-cardinality keys −1.7%, high-cardinality keys −3.4% (−4.1% with 7-day statistics), each ±0.03 over seeds. The keyless nowcast is the first recipe (§8.1); the kind is ranked on the −3.4% |
| 2 | rank 2's intervals were against a static strawman; a rolling split-conformal recipe matches them | taken: widths beside coverage, the rolling arm added (§2.3, §8.2). `offset=` measured inside a link (§5): no gain over a free feature; ranked fourth, on argument |
| 3 | the doubled stream scores every diagnostic against the wrong prediction | taken, reproduced in simulation (0.8806/0.8990 against the bank's 0.8803/0.8996), and designed as `label_clock=` (§2), ranked first. Found beyond the review: `embargo` already holds served predictions, so the design generalizes it; and the default conformal step overshoots once hundreds of predictions are in flight, under `embargo` today too (half-width 188.9 at 8 hours) |
| 4 | `offset=` needs a schema bump and p = 0 fixtures | taken (§5), and the same for `trees=` (§6.4) |
| 5 | `offset=` per target; Polars' `offset` | taken (§5) |
| 6 | keep the level, not only its hash | taken: typed values, a dictionary per feature, `bank.encodings()`; hashing an option (§3.5) |
| 7 | store Σw·y²; name it `smooth`; a half-life per feature | taken (§3.5) |
| 8 | `weight_<f>_<t>` is a second meaning of weight | renamed `support_<f>_<t>`, defined as `λW`; tuple field names defined (§3.5) |
| 9 | v2's §6.1: the walk is wasted under `closed="left"`, cured by `gap_cap`, fixed by a second list; it does not gate the kind | taken as written (§9.1, §3.4); the fix is dispatched as its own task |
| 10 | baselines untuned, CatBoost missing, no monthly retrain, statics barely fit, no recorded selection | taken: CatBoost `has_time=True`, a tuned native, a monthly retrain, the statics at 0.01 with patience 300, every setting chosen on August (§1, §3.2). The ranking held: every batch arm stays at 24.0–24.5 MAE |
| 11 | no forest swap; placement and linking; leaf ids | taken: the swap cannot carry coefficients, and the two answers are stated (§6.4); the reader in `online-polars`; `cargo tree -i` shown, and xxh3-128 in place of SHA-256 (§6.4); leaves renumbered (§6.3) |
| 12 | the planted cases favor each candidate | taken: a structural drift added (§6.2); the target statistic's planted case is labeled as its best case (§3.2) |
| 13 | ranking ids need only be contiguous | taken (§7) |
| 14 | missed candidates | `label_clock=` designed and ranked first; the last-K combination, input drift against the cuts, parity through a saved state and rolling split-conformal measured as recipes (§8) |
| 15 | the eviction key and libm | taken: never persisted, rebuilt on load; cross-OS fixtures on a literal `lam`; one key table per feature (§3.5) |
| 16 | small points | §6.3's table referenced from §11; the count encoding "not shown to help"; the three tree configurations named in §1; the pandas-categorical refusal said never to hit a `po.gbdt` model (§6.4) |

**Where this record disagrees with the review:** nowhere on a number. On
one emphasis: the review asks whether rank 1 stands "on the right number";
v3 moves the encoder to second, below `label_clock=`, because the encoder
learns the same labels either way and only `label_clock` corrects numbers
the bank reports today.

---

## 14. Open questions for the user

1. **The order of the first two builds.** `label_clock=` first, as ranked,
   because it corrects every late-label diagnostic the bank emits today;
   or the target encoder first, whose −3.4% is the larger accuracy gain and
   which runs on the doubled stream until `label_clock` lands.
2. **The conformal step under a delay.** The default rate, 0.05, widened
   flights' 90% interval from 45 to 189 under an 8-hour `embargo` today, and
   to 64 under the simulated `label_clock`; 0.01 kept coverage at 0.899 at
   47–50. Change the default when a spec has a delay (to 0.01), or keep 0.05
   everywhere and document the case. *Taken up in [CONFORMAL-STEP.md](CONFORMAL-STEP.md)
   (2026-11-07): the step derived from the delay, measured, the default kept.*
3. **The model-file reader.** Build it only if the command line needs tree
   features; Python has §8.3's recipe.
4. **`offset=`.** No measured gain; build it for its semantics (the
   target's own scale, `features=[]`) or leave it as a recipe.

---

## 15. Reproducing

```sh
.venv/bin/python -I scripts/tree_features/fetch.py "$PWD"   # 60 days of BTCUSDT, nycflights13 into .cache/
cd scripts/tree_features
../../.venv/bin/python -I ts_decompose.py        # v3, §3.2: selection on August, then three seeds; ~10 min
../../.venv/bin/python -I ts_baselines.py        # v3, §3.2: needs ts_decompose.json
uv run --no-project --python 3.12 --with catboost --with pandas --with numpy --with polars python -I ts_catboost.py
../../.venv/bin/python -I label_clock_sim.py     # v3, §2.3
../../.venv/bin/python -I label_clock_bank.py    # v3, §2.3: embargo's served scoring and its step
../../.venv/bin/python -I offset_link.py         # v3, §5
../../.venv/bin/python -I boosters_k.py          # v3, §6.2, §8.4
../../.venv/bin/python -I psi_drift.py           # v3, §8.5
../../.venv/bin/python -I parity.py              # v3, §8.1
../../.venv/bin/python -I ts_flights.py          # v1
../../.venv/bin/python -I ts_flights_binary.py
../../.venv/bin/python -I ts_flights_cap.py
../../.venv/bin/python -I ts_synth.py            # ~15 minutes, the capped tables in pure Python
../../.venv/bin/python -I ts_recipe.py
../../.venv/bin/python -I ts_recipe_speed.py
../../.venv/bin/python -I ts_recipe_speed2.py
../../.venv/bin/python -I leafrefit.py
../../.venv/bin/python -I leafrefit2.py
../../.venv/bin/python -I leaves_recipe.py
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
`.cache/tree-features/results`). CatBoost, InterpretML and
category_encoders run in throwaway environments and are dependencies of
nothing. The reviewer's own scripts (`e1`–`e5`) were run from the session's
scratch space and are not kept; §3.2 and §8.2 re-measure what they showed.

Sources: Prokhorenkova et al. 2018, *CatBoost: unbiased boosting with
categorical features* (§3.2); He et al. 2014, *Practical Lessons from
Predicting Clicks on Ads at Facebook* (§3.1–3.3); Lou, Caruana, Gehrke and
Hooker 2013, *Accurate Intelligible Models with Pairwise Interactions*
(§4.1, §5.2); XGBoost at `54155e3` and LightGBM 4.7.0, read at the lines
cited. The papers' text is under the gitignored `.cache/research/papers/`.
