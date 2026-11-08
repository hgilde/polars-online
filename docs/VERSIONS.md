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
or a meaning, apart from a few defaults that minor releases changed, each
with the setting that restores it. The package follows semantic versioning, and
[CHANGELOG.md](../CHANGELOG.md) lists what each release changed.

### Pinning the version

**To keep your code working across the 1.x series, pin the major version
in your requirements, as `polars-online~=1.0` or `polars-online>=1,<2`.**
A minor release adds models, parameters, outputs and functions, and breaks
nothing stable. It can also correct numbers that were wrong, or move last
bits when a sum is taken in another order, each declared with its size. So
pin the minor as well if your numbers must not move.
`polars-online~=1.4.0` pins the current minor: it takes that minor's patch
releases and stops before the next minor.
[Reproduce an earlier version's fit](how-to/reproduce-an-earlier-versions-fit.md)
shows how to get back an earlier release's numbers where a setting gives
them back.

**Code written for 0.x needs the changes the CHANGELOG lists under
*Changed (breaking)* before it runs on 1.0.** Version 1.0 renamed and
removed parts of the 0.x API, to make it more standard and consistent. A
[spec](GLOSSARY.md#the-model-bank-and-what-it-reads) keyword that 1.0
renamed is refused with an error that names the new one.

### What is stable

**Stable means part of this package's API, and a breaking change to a
stable part needs a new major version.** `tests/test_api_surface.py`
compares the stable parts with a snapshot of them, `tests/api_surface.txt`,
so a change to one shows as a diff in review. The command line's exit
status is held by a test of its own. An unstable part can change its form
in any minor release:

| | what it covers |
|---|---|
| **stable** | everything in `polars_online.__all__`; the `po.spec` constructors, their keywords and their defaults; the functions of the helper modules `po.eval`, `po.gram`, `po.ops` and `po.stream`; the `.online` namespace; the [output field names](how-to/find-an-output-field.md); the columns, and their types, of the frames a bank and `po.spec` return, and of the closed-groups frame; the words each string-valued parameter takes; the TOML config keys; the command line's flags, its exit status and the files it writes; and the environment variables the package reads |
| **unstable** | the file format of a window run's state (`with_windows(save_state=...)`); the written form of a formula target, in a TOML config or inside a bank's state; the Arrow export: `fit_predict_arrow`, `predict_arrow` and `ArrowStruct`; and the modules `po.sim`, `po.corr` and `po.gbdt`. Each says so in its docstring, and warns with `UnstableWarning` when used with `POLARS_ONLINE_WARN_UNSTABLE=1` set |
| **not stable** | any name that starts with an underscore, apart from the two columns `po.stream.embargo` adds, `_online_role` and `_online_role_weight`, which are stable; the Rust crates (`online-core`, `online-polars`, `online-cli` and `online-py`), which are not published to crates.io, so nothing outside this repository builds against them; a model's numbers within the 1e-12 tolerance its stored reference results are held to, which depend on the platform and the Polars version; and the text the command line prints (its summary, readiness warnings and errors) and the timing lines `POLARS_ONLINE_TIMING=1` asks for, which are written for a person and may be reworded in a minor release |

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
| a stable name renamed | minor, and the old name keeps working until the next major. Using it warns with a `PolarsOnlineDeprecationWarning` that names the new one, as [Renamed names](#renamed-names) says |
| a default or a meaning changed, or a stable name removed | major |
| a change to an unstable part | minor, declared in the CHANGELOG |

### Defaults changed in a minor release

**Some defaults have changed in a minor release, against the table's row
for a changed default.** Each was the maintainer's decision, and the
CHANGELOG declares each with its size. Those below move a fit's numbers,
and each can be set back to its old value:

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
(10 a feature from 51 features, and off above 100), and rebuilds the scale
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
fit, and so does every new group after it. The same release keeps 40 more
bytes for each held row: the row's one-pass residual, and its place in the
tree that the residuals' exact median is read from while the refit waits.
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

### Renamed names

**A spec parameter renamed after 1.0 is forwarded by a table, and any
other renamed name by a forwarding of its own.** The table reads a spec
builder's old keyword, a spec dict's old key and a TOML file's old key as
the new one, at any depth of a spec. It does not reach another function's
keyword, a function, a word a parameter takes, a command-line flag, an
environment variable, an output field or a frame column. Each of those
ships its own forwarding in the release that renames it: a check at the
function's top, a stub, an alias or a second column. Both warn with
`PolarsOnlineDeprecationWarning`, and both last until the next major.

### State files across releases

**A state file a 1.x build saved loads in every 1.x build after it, and a
state file a 0.x release saved does not load in 1.0.** A
[state](GLOSSARY.md#saving-and-serving) is everything a bank has learned.
The oldest layout a build loads stays at 1.0's for the whole 1.x series,
and each change to a state's layout ships a loader for the layout before
it. A saved state of each kind, frozen in the test suite, checks it three
ways, apart from the cases listed under the next paragraph. The current layout's
frozen states load, continue to the bit, and save back byte for byte. A
previous layout's frozen states load through its loader, continue to the
bit, and save back as the current layout's bytes of the same case. So,
outside those cases, the loader turns a state into what this build writes
from the same rows. [Why a state loads in later
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
state, and a formula target inside a bank's state. A 0.x state is refused
by its schema version, with an error that says to refit it from its input
([A bank's state and resuming it](explanation/state-and-resuming.md)).

## The Polars range

**Moving the Polars floor up needs a minor release, and dropping a Polars
major version a major one.** A resolver that installs the new release
upgrades Polars with it, inside the same Polars major, so no code has to
change for a higher floor:

| a change to the Polars range | the release it needs |
|---|---|
| widening it to admit a newer Polars | minor |
| moving the floor up | minor |
| capping it below a Polars that broke this library, as [How the pin moves](RELEASE-READINESS.md#how-the-pin-moves) describes | patch |
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
| `ModelBank` alone | 1.28.1 | |
| every call that reads a query, such as `lf.online.fit_predict`, `lf.online.predict`, `ModelBank.fit` and `ModelBank.fit_predict_batches`, `with_windows` and `refresh_time` | 1.37.0 | they read with `LazyFrame.collect_batches`. Before 1.37.0 it reads ahead of a slower reader without limit: reading a 1 GB parquet file to its end as a query held 1.6 GB on 1.34.0 and 1.36.1, and 0.33 GB on 1.37.0 |
| the examples that stream a DuckDB, ADBC or pyarrow source into a bank | 1.43.0 | for `pl.scan_arrow_c_stream` |

### Where the suite has passed

**The suite passes on 2.0.0, the pin, at every change, and on the floor,
1.37.0, before every release and once a month**
([How the pin moves](RELEASE-READINESS.md#how-the-pin-moves)). It
passed on the floor on 2026-10-12. There, 25 tests skip, each naming the
newer Polars it needs: most check against a Polars function that has
changed since, and the rest stream a database or a pyarrow reader through
`pl.scan_arrow_c_stream`. The suite has also passed on 1.38.1, 1.44.1,
1.44.2 (the pin until 2026-10-08) and 2.0.0-rc.1 with identical numbers.
`tests/test_scaffold.py` asserts the pin and the range, and
[RELEASE-READINESS.md](RELEASE-READINESS.md) has each run with its date
and what differed.

[Polars interfaces](explanation/polars-interfaces.md) explains why a
break on a new Polars is expected maintenance, and which paths to check
first.
