<!-- kind: reference -->
# Versions and the Polars pin

This page is the reference, for anyone who pins polars-online in their
requirements or upgrades it, on what each release may change and which
versions of Polars it runs with.

Two versions matter when you install this package. Its own version says
when to expect a breaking change. The range of Polars it accepts,
`polars>=1.37.0,<3`, is measured by this project, and Polars does not
promise it.

## This package's own versioning

**Version 1.0 starts this package's promise of a stable API.** A minor
release adds to it, and may rename a stable part while the old name keeps
working. Only a major release removes a stable part or changes a default
or a meaning. Minor releases changed a few defaults all the same, each
with the setting that restores it. The package follows semantic
versioning, and
[CHANGELOG.md](../CHANGELOG.md) lists what each release changed.

### Pinning the version

**To keep your code working across the 1.x series, pin the major version
in your requirements, as `polars-online~=1.0` or `polars-online>=1,<2`.**
A minor release adds models, parameters, outputs and functions, and breaks
nothing stable. It can also correct numbers that were wrong, or move last
bits when a sum is taken in another order, each declared with its size. So
pin the minor as well if your numbers must not move.
`polars-online~=1.5.0` pins the current minor: it takes that minor's patch
releases and stops before the next minor.
[Reproduce an earlier version's fit](how-to/reproduce-an-earlier-versions-fit.md)
shows how to get back an earlier release's numbers where a setting gives
them back.

**Code written for 0.x needs the changes the CHANGELOG lists under
*Changed (breaking)* before it runs on 1.0.** Version 1.0 renamed and
removed parts of the 0.x API, to make it more standard and consistent. A
[spec](GLOSSARY.md#the-model-bank-and-what-it-reads) keyword renamed up to
2026-10-07 is refused with an error that names the new one. One renamed on
2026-10-11, when the parameters took the names 1.0 keeps, is still read,
with a warning, until 2.0 ([Names that changed](#names-that-changed)).

### What is stable

**Stable means part of this package's API, and a breaking change to a
stable part needs a new major version.** `tests/test_api_surface.py`
compares the stable parts with a snapshot of them, `tests/api_surface.txt`,
so a change to one shows as a diff in review. The command line's [exit
status](RUNNER.md#exit-status) is held by a test of its own. An unstable part can change its form
in any minor release:

| | what it covers |
|---|---|
| **stable** | everything in `polars_online.__all__`; the `po.spec` constructors, their keywords and their defaults; the functions of the helper modules `po.eval`, `po.gram`, `po.ops` and `po.stream`; the `.online` namespace; the [output field names](how-to/find-an-output-field.md); the columns, and their types, of the frames a bank and `po.spec` return, and of the closed-groups frame; the words each string-valued parameter takes; the TOML config keys; the command line's flags, its exit status and the files it writes; and the environment variables the package reads |
| **unstable** | the file format of a window run's state (`with_windows(save_state=...)`); the written form of a formula target, in a TOML config or inside a bank's state; the Arrow export: `fit_predict_arrow`, `predict_arrow` and `ArrowStruct`; and the modules `po.sim`, `po.corr` and `po.gbdt`. Each says so in its docstring, and warns with `UnstableWarning` when used with `POLARS_ONLINE_WARN_UNSTABLE=1` set |
| **not stable** | any name that starts with an underscore, apart from the two columns `po.stream.embargo` adds, `_online_role` and `_online_role_weight`, which are stable; the Rust crates (`online-core`, `online-polars`, `online-cli` and `online-py`), which are not published to crates.io, so nothing outside this repository builds against them; a model's numbers within the 1e-12 tolerance its stored reference results are held to, which depend on the platform and the Polars version; and the text the command line prints (its summary, readiness warnings and errors) and the timing lines `POLARS_ONLINE_TIMING=1` asks for, which are written for a person and may be reworded in a minor release |

**Each unstable part is unstable for a reason that a promise would make
costly.** The window run's state reached version 10 in under four weeks
(2026-09-30 to 2026-10-24), versions 2 to 9 in the first nine days. The
formula tree is read from `expr.meta.serialize`, whose form Polars calls
unstable across its versions. The Arrow export's input side still arrives
as a Polars frame, and its import is not built. `po.sim` and `po.corr`
stand beside the bank: neither reads its outputs, as `po.eval` and
`po.gram` do. `po.sim`'s frames also rest on NumPy's `Generator`, which
promises no stream across NumPy versions. `po.gbdt` is built on the
boosting libraries' C interfaces and on how their Python classes hold a
native handle, so a new version of one of those libraries may need a new
release of this package.

### What the API is

**The API is bigger than `polars_online.__all__`: four surfaces can break
your code with no error at import.** Each is stable, and each
change to one shows as a diff in the snapshot:

| surface | what is API | how a change would break you |
|---|---|---|
| output field names, the largest surface | every field name you read from a spec's output column, as `out["m"].struct.field("pred_y")`: `pred_y__r0.000001@h100`, `resid_y__r0.5@h100`, `abs_resid_q0.95_y0`, `weight_sum@h500`. They are built from ridge values, half-life suffixes, feature-set labels, quantile levels and target names | a new output, a renamed suffix or a float rendered another way fails as a `StructFieldNotFoundError` in your pipeline |
| the spec constructors | the twenty-two model builders of `po.spec`, every keyword's name and every default. The keywords are keyword-only, so their order is not API | a renamed keyword is refused, or forwarded with a warning ([How a renamed name is still read](#how-a-renamed-name-is-still-read)) |
| defaults | every default, whether a builder sets it in Python or the bank resolves it in Rust from a keyword left as `None` | no error: it silently changes your numbers, which is worse than breaking your code |
| state files and the TOML config | a state file's layout and version, and every key of a run config, a spec and each model type | a file or a config that loaded before is refused |

**A state file records the schema version that decides whether it
loads.** `po.schema_version()` gives the installed release's, which moves
with every change to a state's layout. A release refuses a file whose
schema is older than the oldest it loads, by that version. A window run's
state and a `po.stream.refresh_time` state each carry a version of their
own.

### What each kind of release may change

**A changed default or meaning needs a major release, and a fix to numbers
that were wrong a minor one.** A number is wrong when it disagrees with
the model's stated definition, the update equations its docstring gives:

| a change | the release it needs |
|---|---|
| a fix that moves no number | patch |
| a fix to numbers that were wrong | minor, declared in the CHANGELOG with the difference `scripts/compare_release.py` measures from the last release |
| a change of arithmetic order or kernel that moves last bits within rounding | minor, declared in the CHANGELOG with its measured size |
| a new model, parameter, output or function | minor |
| a stable name renamed | minor, and the old name keeps working until the next major. Using it warns with a `PolarsOnlineDeprecationWarning` that names the new one, as [How a renamed name is still read](#how-a-renamed-name-is-still-read) says |
| a default or a meaning changed, or a stable name removed | major |
| a change to an unstable part | minor, declared in the CHANGELOG |

### Defaults changed in a minor release

**Some defaults have changed in a minor release, against the row for a
changed default in [What each kind of release may
change](#what-each-kind-of-release-may-change).** Each was the maintainer's
decision, and the CHANGELOG declares each with its size. Each moves a
fit's numbers, and each can be set back to its old value:

| parameter | release | old default | new default | to set the old default |
|---|---|---|---|---|
| `huber_scale` | 1.3.0 | `"rms"` | `"mad"` | `huber_scale="rms"` |
| `huber_refit_rows` | 1.4.0 | off | 500 | `huber_refit_rows=0` |
| `quantile_refit_rows` | 1.4.0 | off | 500 | `quantile_refit_rows=0` |
| `bin_refit_rows` | 1.4.0 | off | `max(500, 30 · bins)` | `bin_refit_rows=0` |
| `hmm`'s `warm_rows` | 1.4.0 | 50 | 1,000 | `warm_rows=50` |
| `refit_budget` | the release after 1.4.0 | 256 MiB | 320 MiB | `refit_budget=256` |
| `corrchange(kind="window")`'s `permute_every_rows` | the release after 1.4.0 | 50 | `window_rows` | `permute_every_rows=50` |

**`huber_scale` moved from `"rms"` to `"mad"` in 1.3.0.** It is the scale
`huber` and `sgd(loss="huber")` draw their cut in, and Huber's loss caps a
row's pull at `huber_delta` times that scale. A root mean square is set by
one wild row, so wild rows widened the cut meant to catch them, and the fit
drifted toward least squares. A median absolute deviation moves by a wild
row's weight, whatever the size of its miss.
[MODELS.md](MODELS.md#huber--robust-regression) gives the slopes measured
under each scale. `huber_scale="rms"` restores the earlier numbers to the
bit. A state saved under the old default loads under it.

**1.4.0 turned on `huber_refit_rows`, at 500.** One pass weighs each row by
the fit before it, so a share of wild rows can capture the first fits of
`huber` and `sgd(loss="huber")`, and without decay the fit stays captured.
The refit fits each target's first 500 learned rows to convergence, once
(the default is `max(500, 10 × features)` up to 100 features, and off above 100), and rebuilds the scale
from that fit's residuals. At one-sided wild rows on an eighth of 20,000
rows, 13 of 40 `huber` streams ended more than 0.05 from statsmodels'
converged fit, and none with the refit. MODELS.md gives the capture
measured with and without the refit. `huber_refit_rows=0` turns the refit
off, and the default MAD scale then reads its sketch on the fixed buckets
1.3.0 used. A state saved before it loads with the refit off and goes on
to the bit. A group a resumed bank first sees after the load, or one a
drift reset starts over, refits under the default.

**1.4.0 turned on `quantile`'s refit of its first rows,
`quantile_refit_rows`, at the same count.** Without decay its one pass
kept the rows its warm-up took as least squares in its sums at full
weight. Three wild first targets left the worst of 50 streams 0.22 from
statsmodels' `QuantReg` after 8,000 rows, against 0.0053 with the refit.
`quantile_refit_rows=0` turns it off. A state saved before it loads with
it off.

**1.4.0 turned on `marginal(bin_stat="huber")`'s refit of its first rows,
`bin_refit_rows`.** Spikes in the first rows pull the bins' locations,
and without decay the bins stay captured. The test ran 50 streams on 16 bins
with spikes of about 20 scales and no decay. After 2,000 rows at 12% and
20% spikes, the worst was 1.3 and 3.5 target scales from statsmodels'
M-location, and 0.015 and 0.14 with the refit. `bin_refit_rows=0` turns it
off. A state saved by 1.0 to 1.3 loads with the refit off and continues
exactly.

**1.4.0 increased `hmm`'s `warm_rows` from 50 to 1,000 rows** (or
`n_components` where that is more). The seeding places states where the
buffered rows are, and 50 rows hold one or two regimes. On three regimes
lasting about 50 rows each, the worst of 50 seeds put 0.46 of the rows in
their true state, and 1,000 rows put 0.95. `warm_rows=50` sets the old
count. A group a saved state had already seeded is past its warm-up, so
the new count does not reach it.

**The release after 1.4.0 increased `refit_budget` from 256 to 320 MiB.**
The budget bounds the rows that the refits of all of a spec's groups hold.
Each group reserves the most its refit can hold when the bank first sees
it. The group whose reservation would pass the budget keeps its one-pass
fit, and so does every new group after it. The same release keeps 48 more
bytes for each held row: 8 for the row's one-pass residual, and 40 for its
place in the tree that the residuals' exact median is read from while the
refit waits.
At 5 features and 500 rows a target now reserves 124 KB where it reserved
100 KB, so 256 MiB would refit 2,164 such targets where it refit 2,684.
320 MiB refits 2,706. Where the budget binds, a bank holds up to 64 MiB
more, the difference between the two bounds. `refit_budget=256` restores
the old bound. A bank saved under the old default resumes under the new
one. The groups it gave up stay given up, every group it holds goes on to
the bit, and the groups it first sees after the load refit until 320 MiB
is reserved.

**The same release changed `corrchange(kind="window")`'s
`permute_every_rows` from 50 to `window_rows`.** Without a fixed `crit`,
the critical value is a quantile of `n_perm` draws, each a shuffle of the
two windows' pooled rows. One redraw costs `n_perm · window_rows · k²`
for `k` features, so a redraw every 50 rows cost more a row the longer
the window. A redraw once a window's length of rows costs `n_perm · k²` a
row, whatever the window. Each critical value is then in force for one
shift of the windows, after which they share no row with the windows it
was drawn from. The test's size stayed near `alpha`: at alpha 0.05 and 200
draws, 4.8% to 5.2% of non-overlapping spans of rows with no change were
flagged, at windows of 50 to 400 rows. `permute_every_rows=50` restores
the old schedule of draws. A saved state keeps the value it was saved
with, given or a default. So a bank saved under the old default goes on
redrawing every 50 rows, and a group it first sees after the load redraws
every `window_rows` rows.

### How a renamed name is still read

**A renamed spec parameter, or a renamed word of one, is forwarded by a
table, and any other renamed name by a forwarding of its own.** This covers
the parameters, functions and keywords renamed since 2026-10-11. The
output fields and frame columns renamed at 1.0.0 are not read under their
old names, and a spec parameter renamed up to 2026-10-07 is refused, with
an error that names the new one. The tables
read a spec builder's old keyword, a spec dict's old key, a TOML file's old
key and an old word of a spec parameter as the new one, at any depth of a
spec. A table covers every model, or the one model in which the key or
word changed. They do not reach another function's keyword or the words
it takes, a function, a command-line flag, an environment variable, an
output field or a frame column. Each of those ships its own forwarding in the release that renames
it: a check at the function's top, a stub, an alias or a second column.
Both warn with `PolarsOnlineDeprecationWarning`, and both last until the
next major.

### Names that changed

**Each name renamed or removed since 0.x has one current name or
replacement, listed by its old name.** Older pages, notes and the
CHANGELOG use the names of their day. On 2026-10-02 the
public parameters and output fields were renamed to Polars' words. A few
other changes renamed one name each, and four surfaces were removed. A
spec that passes a parameter renamed on 2026-10-02 is refused, and the
message names the new one. On 2026-10-11 the parameters took the names they
keep at 1.0: scikit-learn's for the mixture, clustering and changepoint
models, and one name for one thing across the models. Those old names are
still read, with a `PolarsOnlineDeprecationWarning` naming the new one,
until the next major version. At 1.0.0 the output fields and frame columns
took one grammar: an old field or column name is no longer written, and a
function or keyword renamed then is still read, with the same warning.

| a dated section or record says | the name now | changed in |
|---|---|---|
| `halflife`, alone or inside a name: `long_halflife`, `coef_halflife`, `revert_halflife`, `select_halflife`, `level_halflife`, `trend_halflife` | `half_life`, and so `long_half_life` and the rest | 2026-10-02 |
| `max_dclock` | `gap_cap` | 2026-10-02 |
| `label_delay` | `embargo` | 2026-10-02 |
| a model's `window`, its hard cutoff; `po.eval.rolling_metrics`, its `window` and then its `window_size` | `window_size`; `po.eval.window_metrics` and `every`: its windows do not overlap, as Polars' `group_by_dynamic(every=)` | 2026-10-02, then 2026-10-07 |
| `min_periods` | `min_weight` | 2026-10-02 |
| `on_clock_reset` and its policies `"error"` and `"reset_state"`; `min_backwards_jump`; and before it `min_session_clock` | `restart_after_step_back`. Unset, every step back is refused, as `"error"` did; set, a step back larger than it resets the model, as `"reset_state"` did, and `0` resets it at every step back | 2026-10-02; `min_session_clock` was removed in 0.9.0, and the policies `"max"` and `"zero"` in 0.12.0 |
| `emit_resid_z`, and its field `resid_z_<t>` | `emit_zscore`, and `zscore_<t>` | 2026-10-02 |
| `sgd`'s `scale_features` | `standardize` | 2026-10-02 |
| `ridge_decay=True`, and its default `False` | `ridge_scale="sum"`, and the default `"mean"` | 2026-10-02 |
| `add_intercept` | `fit_intercept` | 2026-10-02 |
| `lasso`'s `max_cd_iters` and `cd_tol` | `max_iter` and `tol` | 2026-10-02 |
| `corrchange`'s `reset` | `reset_on_flag` | 2026-10-02 |
| `n_eff`, as a field a model writes, or a column or key of `closed_groups()`, `coef()`, `last_row()`, `marginal()` or `gram()` | `weight_sum`, as in `weight_sum@h10` and `pair_weight_sum`. The models' accessor keeps the name `n_eff`, and means the same number | 2026-10-02 |
| `lasso`'s `lam_selected_<t>` | `penalty_selected_<t>` | 2026-10-02 |
| `pcorr_<a>_<b>`, `absresid_q<p>`, and the PCA loadings `pc<j>_<feature>` | `partial_corr_<a>_<b>`, `abs_resid_q<p>`, and `pc<j>_loading_<feature>` | 2026-10-02 |
| `bocpd`'s `logscore`; `hmm`'s `p_<k>` and `p1_<k>` | `loglik`; `filtered_state<j>` and `predicted_state<j>` | 2026-10-02, then 1.0.0 |
| `po.prep`, and `refresh_time`'s `time=` and `by=` | `po.stream`, and `clock=` and `group=` | 0.12.0 |
| `po.corr.signal_share`'s `n_eff_blocks` | `n_kish_blocks` | 2026-10-02 |
| `po.eval`'s `by` and `min_obs` | `group`, which also takes one key as a bare string, and `min_samples` | 2026-10-07 |
| `po.corr.shift` | `po.corr.absorption_shift` | 2026-10-07 |
| `po.gram.lasso_path`'s `lambdas` | `penalties` | 2026-10-07 |
| `corrchange`'s `horizon` and `window`; `rcov`'s `window` | `span_rows`; `preavg_rows` | 0.3.0 |
| `po.run`, the Python runner | removed: the `polars-online` command line, or `ModelBank.fit(lf)` | 0.7.0 |
| the expression plugin: `pl.col(...).online.<model>`, `po.online` and `InMemoryExpressionWarning` | removed, because a stateful expression is handed its whole column: `lf.online.fit_predict` | 0.7.0 |
| `po.window.ewm` and `po.window.lookahead_rewm`, with `split=` | removed: the window operators, such as `po.ewm_mean` and `po.ewm_sum(..., reverse=True)`, run by `with_windows` or as a spec's target | 2026-10-03 |
| a spec's `type = "ew_ridge"` | `type = "ewridge"`, the builder's spelling | 2026-10-07 |
| `chunk_rows`, and the command line's `--chunk-rows` | `chunk_size` and `--chunk-size`, Polars' name | 2026-10-07 |
| the command line's `--resume` | `--load-state`, beside the TOML key `load_state` | 2026-10-07 |
| `kmeans`' field `dist2` | `dist_second` | 2026-10-07 |
| `ew_cov`'s statistic `lagcorr` and its fields `lagcorr_<a>_<b>_l<ℓ>`; `marginal`'s `lagcorr_xx`, `lagcorr_yy`, `lagcorr_xy` and `lagcorr_yx`, and their `pair_` columns | `lag_corr`, `lag_corr_l<ℓ>_<a>_<b>`, `lag_corr_xx` and the rest | 2026-10-07, the field's order 1.0.0 |
| `kmeans`' `update_every` and `split_merge_every`; `corrchange`'s `permute_every` | `update_every_rows`, `split_merge_every_rows`; `permute_every_rows` | 2026-10-07 |
| `kmeans`' `k`, and `seed_rule` with its word `"kmeanspp"`; `hmm`'s `k` and `seed_rule` | `n_clusters`, and `init` with `"k-means++"`; `n_components` and `init` | 2026-10-11 |
| `ew_class`'s and `hmm`'s `covariance`, and its words `"shared"` and `"diagonal"` | `covariance_type`, and `"tied"` and `"diag"` | 2026-10-11 |
| `bocpd`'s `emission`: `"gaussian"`, `"diag"` and `"robust"` | `covariance_type`: `"full"`, `"diag"`, and `"diag"` with `robust_beta` | 2026-10-11 |
| `bocpd`'s `prior_mean`, `prior_kappa`, `prior_nu` and `prior_scale` | `mean_prior`, `mean_precision_prior`, `degrees_of_freedom_prior` and `covariance_prior` | 2026-10-11 |
| `window_every` | `snapshot_every`, beside `max_rows_between_snapshots` | 2026-10-11 |
| `ew_cov`'s `max_rows_between_pca` | `max_rows_between_pcas`, as `max_rows_between_solves` | 2026-10-11 |
| `lasso`'s `lasso_path` | `penalties`, as `po.gram.lasso_path` names them | 2026-10-11 |
| `corrchange`'s `span_rows`, and `reset_on_flag` | `window_rows`, and `change_action`, `"flag"` or `"reset"` as `drift_action` | 2026-10-11 |
| `holt`'s `level_half_life` | the spec's `half_life` | 2026-10-07 |
| a model window that keeps a row exactly `window_size` old | `closed="both"`; the default, `closed="right"`, drops it | 2026-10-07 |
| a relative target: `po.target`'s `relative_to=` and `relative=`, and a target table's `relative_to` and `relative` keys | removed: the target made as a column, `with_columns(ret=(pl.col("p") / pl.col("mid")).log())` or a difference, made with Polars' `with_columns` in the query before the bank, or written into the file the command line reads; a window looking ahead less a column of the row stays a formula target | 2026-10-07 |
| a `lasso` grid point `__l<penalty>` | `__p<penalty>`, and a path of one penalty writes none | 1.0.0 |
| the Ramsey test's field `reset_<t>` | `ramsey_reset_<t>` | 1.0.0 |
| `groups()`' `rows_processed` and `last_clock`; `summary()`'s `last_clock` and `min_support_coef` (`_feature`) | `rows_fed` and `clock_last`; `clock_last` and `support_coef_min` (`_feature`) | 1.0.0 |
| `describe()`'s `null_count`, which counted every unusable value | `unusable_count` | 1.0.0 |
| `audit()`'s `rows`, `null`, `nan`, `pos_inf`, `neg_inf`, `beyond_bound` and `distinct`, and its clock table's `max_step` | `rows_fed`, `null_count`, `nan_count`, `pos_inf_count`, `neg_inf_count`, `beyond_bound_count` and `n_unique`, and `step_max` | 1.0.0 |
| `marginal()`'s `t_serial`; `closed_groups()`' `pair_t_serial`; `gram()`'s `cross_centred` | `t_stat_serial`; `pair_t_stat_serial`; `cross_centered` | 1.0.0 |
| the bank's `save(path=)`, `load(path=)`, `to_json` and `save_json`; the methods `rows_fed()` and `output_fields()` | `save(file=)`, `load(source=)` and `write_json(file=None)`; the properties `rows_fed` and `output_fields` | 1.0.0 |
| `po.rewm_mean`, `po.rewm_sum` and `po.rewm_rate` | `po.ewm_mean(..., reverse=True)` and the rest | 1.0.0 |
| `po.eval`'s `spec_name` and `spec_names` | `spec`, a name or the spec dict, and `specs` | 1.0.0 |
| `po.gram.subset(cols=)`; `lars_path`'s and `lars_paths`' `max_steps`; `po.corr.matrix` | `columns=`; `max_iter`; `po.gram.correlation` | 1.0.0 |
| the command `online`, and its variable `ONLINE_TIMING` | `polars-online`, and `POLARS_ONLINE_TIMING`, on at `1` only | 1.0.0 |

### State files across releases

**A state file a 1.x build saved loads in every 1.x build after it, and a
state file a 0.x release saved does not load in 1.0.** A
[state](GLOSSARY.md#saving-and-serving) is everything a bank has learned.
The oldest layout a build loads stays at 1.0's for the whole 1.x series,
and each change to a state's layout ships a loader for the layout before
it. A 0.x state is refused by its schema version, with an error that says
to fit the bank again on its input ([A bank's state and resuming
it](explanation/state-and-resuming.md)). The 0.x releases wrote schema 14
(0.10.0), 17 (0.11.1), 19 (0.12.0) and 20 (0.13.0), all below 1.0's 61,
so every 1.x release refuses each file they saved.

**A saved state of each kind, frozen in the test suite, checks the 1.x
promise three ways,** apart from the states of a model whose computation a
release changed. The current layout's
frozen states load, continue to the bit, and save back byte for byte. A
previous layout's frozen states load through its loader, continue to the
bit, and save back as the current layout's bytes of the same case. So,
outside those cases, the loader turns a state into what the newer release
writes from the same rows. [Why a state loads in later
versions](explanation/state-and-resuming.md#why-a-state-loads-in-later-versions)
gives the reasons.

**Where a release changed how a model computes, most states saved before
it keep the rule they were fitted under and go on to the bit.** The cases
that do not are described in the CHANGELOG entry of the release that made
the change, or in this list:

| a state saved by | holding | how it goes on after the load |
|---|---|---|
| 1.2.0 or earlier | a model under a [hard window](GLOSSARY.md#fitting-a-model) | its first read is the one it saved. Once a snapshot taken after the load is the window's edge, its outputs continue within 1e-12, not to the bit, because such a state never recorded the per-row decay factors |
| 1.3.0 or earlier | `kmeans`, before it seeded | it seeds by the new rule: the same centers and distances to the bit, with the cluster labels possibly renumbered, the same way on every row. It saves back bytes that load back the same, not the current layout's |
| 1.3.0 or earlier | `lasso` | within 1e-9 relative, not to the bit, because its stopping rule became relative to the target's spread. It saves back the same bytes |
| 1.3.0 or earlier | a spec's diagnostics | the `cusum_*` and `cusum_sq_*` fields read over a window take the current values while their sums are young. Every other field continues to the bit |
| 1.3.0 or earlier | `holt` | it loads with no line fit and continues with the gains it was fitted under: its predictions and state within 1e-14, and its diagnostics read over many rows within 1e-12. It saves back bytes that load back the same, not the current layout's |
| 1.3.0 or earlier | `ew_cov`, or a spec's diagnostics | the co-moment matrix is now kept as its lower triangle, so `ew_cov`'s covariances and correlations continue within 1e-15 of `1 + \|x\|`, and the specification, calibration, break and `ic` statistics within 1e-12. Every prediction continues to the bit |
| 1.3.0 or earlier | `ewridge` | every residual, spread, weight and Gram continues to the bit, and every prediction too, except where a target's weight has decayed to exactly 0 below its floor or every attempt to factor a fit fails. There a row may read the last fit solved. Until the next solve, `coef`, `support_coef` and `se_coef` report the file's fit |
| 1.4.0 or earlier | a median-absolute-deviation scale (`huber`, `quantile`, `sgd(loss="huber")`, `marginal`'s Huber bins) | it keeps the rule it was saved under and continues to the bit. It saves back in its own marked form, not the current layout's; a `marginal` state does so only until its bins are fixed |
| 1.4.0 or earlier | `corrchange(kind="window")` | where its windows span four blocks or more, `stat` continues within 1e-14 of `1 + \|stat\|`. Every other output continues to the bit, and a kept `permute_every_rows` stays as saved |
| 1.4.0 or earlier | `ew_class` under a hard window and `covariance_type="diag"` | a state below its `window_budget` continues to the bit. One that had reached it thins or refuses its snapshots later than before, and from that row does not continue to the bit |
| 1.4.0 or earlier | a bank under the default `refit_budget` | the groups it gave up stay given up, every group it holds continues to the bit, and a group it first sees after the load refits until 320 MiB is reserved |

**The two unstable formats are outside that promise:** a window run's
state, and a formula target inside a bank's state.

**The state files of every release on PyPI are checked against each new
release before it ships.** The 1.x releases are read from PyPI, so a
release is checked from the moment it is uploaded. Each spec of a fixed
workload is fitted under each release on the first half of a stream and
saved. The new release must refuse each 0.x file by its version, with the
message to "fit it on its input". It must load each 1.x file and go on
over the second half to the bit, against its own fit of both halves. Two
cases differ by design: the states of a model whose computation a release
changed, and the specs whose numbers a CHANGELOG entry moved since that release. An
`ewridge` with `gram_block_rows` set, saved by 1.0.0 or 1.0.1, whose target sums 1.1.0 took in
another order, goes on within 5.2e-16 of the new release's own. [The state
files of earlier
releases](RELEASE-READINESS.md#the-state-files-of-earlier-releases) has
how the test runs.

## The Polars range

**Moving the Polars floor up needs a minor release, and dropping a Polars
major version a major one.** A resolver that installs the new release
upgrades Polars with it, inside the same Polars major, so no code has to
change for a higher floor. If you must stay on an older Polars, stay on
the older minor of this package:

| a change to the Polars range | the release it needs |
|---|---|
| widening it to admit a newer Polars | minor |
| moving the floor up | minor |
| capping it below a Polars that broke this library, as [Move the Polars range](contributing/move-the-polars-range.md) describes | patch |
| dropping a Polars major version | major |

### What is pinned

**The wheel carries its own Rust `polars`, pinned exactly, and shares one
process with the py-polars you install.** Data crosses between the two
only through the interfaces that [Polars
interfaces](explanation/polars-interfaces.md) describes, so py-polars can
be any version in the range:

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.37.0, < 3** (built and tested against 2.0.0) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

**The lowest py-polars depends on what you use:**

| what you use | the lowest py-polars | why |
|---|---|---|
| `ModelBank` alone | 1.28.1 | the release that added the `PySeries._export` it calls |
| every call that reads a query, such as `lf.online.fit_predict`, `lf.online.predict`, `ModelBank.fit` and `ModelBank.fit_predict_batches`, `with_windows` and `refresh_time` | 1.37.0 | they read the query one chunk at a time, with `LazyFrame.collect_batches`, which before 1.37.0 reads ahead of a slower reader without limit ([The floor and the ceiling](#the-floor-and-the-ceiling)) |
| the examples that stream a DuckDB, ADBC or pyarrow source into a bank | 1.43.0 | for `pl.scan_arrow_c_stream` |

### The floor and the ceiling

**The floor is 1.37.0, because older py-polars reads a query faster than
the bank takes its chunks, and holds what it has read.** Below it, a
query reading a file held the whole file in memory: reading a 1 GB parquet
file to its end as a query held 1.6 GB on 1.34.0 and 1.36.1, against 0.33 GB
on 1.37.0.

**The ceiling, `<3`, admits every 2.x on the strength of 2.0.0, the pin,
on which the whole suite passes at every change.** Two runs hedge it: the blocking newest-Polars leg of every release,
and the weekly run on the newest Polars. [Why the range is 1.37.0 to
3](explanation/polars-interfaces.md#why-the-range-is-1370-to-3) gives the
measurements behind both ends.

### Where the suite has passed

**The suite passes on 2.0.0, the pin, at every change, and on the floor,
1.37.0, before every release and once a month**
([The runs that watch the
dependencies](RELEASE-READINESS.md#the-runs-that-watch-the-dependencies)). On
the floor on 2026-10-12, 6,762 tests passed and 27 skipped: two for a
Windows-only path spelling, and 25 each naming the newer Polars it needs.
Most of the 25 check against a Polars function that has changed since,
and the rest stream a database or a pyarrow reader through
`pl.scan_arrow_c_stream`. The suite has also passed on 1.38.1, 1.44.1,
1.44.2 (the pin until 2026-10-08) and 2.0.0rc1 with identical numbers.
`tests/test_scaffold.py` asserts the pin and the range.

**The Rust `polars` inside the wheel is 0.55.2, and the command line uses
it alone:** the `polars-online` command never touches py-polars, so the
py-polars range does not apply to it.

[Polars interfaces](explanation/polars-interfaces.md) explains why a
break on a new Polars is expected maintenance, and which paths to check
first.

## NumPy

**NumPy is the one optional dependency, `numpy>=1.26` with no ceiling.**
It is needed only by the `numpy` extra, `polars-online[numpy]`, for
`po.sim`, `po.corr`, `po.gram` and `ModelBank.gram`. The floor follows
the oldest Python this package supports: NumPy's first wheels for Python
3.12 are 1.26.0. No run on NumPy 1.26 is recorded.

**The newest NumPy blocks a release, and its next release candidate is
advisory.** Every other job runs on the locked NumPy, so the release runs
the Python suite with only NumPy upgraded, in two legs:

| leg | resolves to | blocks the publish |
|---|---|---|
| the newest NumPy | the newest stable that `numpy>=1.26` admits | yes |
| NumPy's next release candidate | a release candidate while one is out, and the newest stable again otherwise | no |

The weekly run on the newest dependencies, the canary ([The runs that watch
the dependencies](RELEASE-READINESS.md#the-runs-that-watch-the-dependencies)),
runs the advisory leg too. Only NumPy moves in these
runs, so a red one names it. With no ceiling, a NumPy that breaks this
library blocks every release until it is fixed or capped, the same trade
the Polars range makes.
