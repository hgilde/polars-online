# Adding a model

A model touches every layer, from the recursion in `online-core` to a heading
in the README. None of the steps can be folded into another, because each is a
real decision: what the state is, what the spec calls it, what the outputs are
named, what the tests assert. What *can* be done is to make each omission
fail. This is the list, in the order the data flows, with the check that
catches skipping each step. Where a step says **no check**, that is the truth
and not a to-do.

It is written for a contributor adding a model kind, who knows the library's
words (spec, bank, state, `n_eff`) but not yet the names of its plumbing.
Adding a common parameter, an output field or a kind of input column instead
is [the last section](#adding-an-output-or-a-parameter-instead).

**The registry is `ModelKind::KINDS`** in `crates/online-polars/src/spec.rs`:
the spec `type` names, in enum order. Python reaches it as
`polars_online._polars_online.model_kinds()`. Everything else is held to it.

Two commits are complete worked examples of this list.
`git show --stat <commit>` on either is this list as a diff:

```sh
git show --stat aa96ad3   # holt: the smallest possible model, 17 files
git show --stat 47d3b35   # bocpd, task 55, the most recent addition: 23 files
```

`bocpd` touches more files because four of them did not exist when `holt` was
added: the pipeline's golden file, the model contract, the API snapshot and the
kwargs typing test.

The steps at a glance:

| step | file | what you add | the check that fails if you skip it |
|---|---|---|---|
| [1](#step-1--srcmodelrs) | `crates/online-core/src/<model>.rs` | the model: `<Model>Cfg`, `new`, `impl OnlineModel` | the trait, and the checks in the step |
| [2](#step-2--srclibrs) | `src/lib.rs` | the `pub use` | the compiler |
| [3](#step-3--srcmodelrs) | `src/model.rs` | a `ModelState` variant and its `kind()` arm | `kind()` is exhaustive; `every_model_state_variant_is_probed_here` |
| [4](#step-4--testsmodel_contractrs) | `tests/model_contract.rs` | the contract probe and the predict-parity test | `every_model_with_a_recovery_test_has_a_predict_parity_test` |
| [5](#step-5--testsgoldenrs) | `tests/golden.rs` | `fn <kind>_golden()` | `test_the_core_golden_file_pins_every_model` |
| [6](#step-6--srcspecrs) | `crates/online-polars/src/spec.rs` | the `ModelKind` variant, its `kind_name()` arm, the name in `KINDS` | `kinds_lists_every_variant_in_order`, and the checks in the step |
| [7](#step-7--srcstreamrs) | `src/stream.rs` | the `AnyModel` variant and its arms | exhaustive matches; the save/load sweeps for `restore` |
| [8](#step-8--srcbankrs) | `src/bank.rs` | nothing, unless the outputs or the `coef` slots differ | `test_names_match_the_realized_struct_for_every_model`; `test_every_model_lays_out_as_coef_index` |
| [9](#step-9--_specpy-and-specpy) | `python/polars_online/_spec.py`, `spec.py` | the builder | `test_every_rust_kind_has_exactly_one_builder`, and the checks in the step |
| [10](#step-10--testsapi_surfacetxt) | `tests/api_surface.txt` | the regenerated snapshot | `test_api_surface_matches_the_snapshot`; `test_the_api_snapshot_pins_every_models_output_fields` |
| [11](#step-11--teststest_modelpy) | `tests/test_<model>.py` | the Python-side oracle | `test_every_builder_has_a_per_model_test_file` |
| [12](#step-12--per-model-sweeps) | four sweep lists | one entry each | `test_the_sweeps_cover_every_regression_model` |
| [13](#step-13--teststest_model_registrypy) | `tests/test_model_registry.py` | the `MINIMAL` entry | `test_minimal_names_every_builder` |
| [14](#step-14--teststest_golden_pipelinepy) | `tests/test_golden_pipeline.py` | a spec in `specs()`, and its lines in `GOLDEN` | `test_the_golden_pipeline_pins_every_model` |
| [15](#step-15--readmemd) | `README.md` | the model's heading, and its row in the model table | `test_the_readme_documents_every_model` |
| [16](#step-16--changelogmd-and-the-design-note) | `CHANGELOG.md`, and the design note | an entry | **no check** |

## The recursion — `crates/online-core`

### Step 1 — `src/<model>.rs`

The new module holds three things:

| part | what it is |
|---|---|
| `<Model>Cfg` | the configuration struct |
| `<Model>::new(cfg) -> Result<Self, String>` | the constructor, which validates the cfg |
| `impl OnlineModel` | `step` *and* `predict`. `predict` is the step without the step: what `step` would report for the row, state untouched |

**Every parameter check belongs in `new`, not in the spec layer**: the CLI and
the bank both construct through `new`. **Derive `step`'s prediction from
`predict`**, so the two cannot drift (ENHANCEMENTS E31). The docstring states
the update equations. The module's unit tests need an **oracle**, not a golden
number alone: the recursion written out longhand, an equivalent model
configured differently, or the optimality conditions.

Four rules hold for every model. `n_eff` is the weight before this row's
update and before its own decay (CLAUDE.md rule 8). A zero-weight first row
must not divide 0/0 (rule 9). A zero-weight row **still decays** `n_eff`: it
advances the clock and nothing else. `zero_weight_rows_only_advance_the_clock`
checks that without naming a decay. The stream with the row and the stream
without it, its clock delta carried into the next row, must report the same
`n_eff`. And there is no `unsafe`, with `f64` everywhere.

**Check:** the trait. A missing method is a compile error; the shared contract
is step 4.

#### When the model holds more than a recursion

Some models keep state or read inputs that the trait's defaults do not handle.
Each such case overrides a method, and each has its check:

| if the model … | it implements | the check |
|---|---|---|
| keeps a ring of past rows: a lag ring, a window | **`clear_lags`**, dropping that ring and nothing else (docs/PLAN.md task 47) | `probe_with` in `tests/model_contract.rs`, against `KEEPS_LAGS` |
| has a `window` | its snapshots in `online_core::Snapshots`, a **`Footprint`** for the snapshot type (the heap it holds, in bytes), and **`set_window_budget`** and **`window_over_budget`** reaching the ring, so that `window_budget` bounds it (review 2026-09-12, P4) | `tests/test_window_budget.py`: `test_every_windowed_model_holds_its_budget` and `test_the_budget_table_names_every_windowed_builder` |
| is windowed and predicts a target | an entry in `ModelKind::window_and_every`, so the stream cuts its `sigma` and `resid_z` at the fit's window (review 2026-09-12, S1) | `test_second_opinion::TestWindowedSpread`, in whose parametrization it belongs |
| keeps a weight per target: the rows each target was present on | **`target_n_eff_into`** (review 2026-09-12, S2) | `test_a_sparse_target_warms_up_on_its_own_weight` in `tests/test_bank.py`, to which it is added |
| reads a number out of `y` rather than regressing it | **`predict_with(x, y, d_clock)`** (docs/REVIEW-E54-E64.md C1) | `predict_is_the_step_without_the_step` and `a_value_in_the_targets_slot_reaches_predict` |

**Row-lagged state.** The stream calls `clear_lags` on a session change and on
a capped clock gap (`ClockAdvance::capped`). Those are the two events after
which "the row `ℓ` back" no longer means a row `ℓ` ago. A reset already
rebuilds the model, and an ordinary gap is what decay is for. It is **not** a
reset: means, co-moments and `n_eff` must not move. `probe_with` serializes the
state, calls `clear_lags`, and requires the bytes to be identical unless the
model is listed in `KEEPS_LAGS`. So a model that quietly clears a mean fails,
and so does one with a ring that forgot to declare itself. It then restores a
copy from the *cleared* state, and requires the two to report the same rows
from there.

**Read a ring's depth from the configuration, never from a container's
`capacity()`.** A clone and a msgpack round-trip both shrink the capacity to
the length it holds. So a ring copied while short would keep that depth for
ever (docs/REVIEW-E54-E64.md L1). The contract's copy is taken with an *empty*
ring, which grows back. The case that bites is a partly filled one, so the
model's own unit tests need a save/restore at every depth from empty to full.

**A window.** The trait's defaults do nothing, so a model that forgets the two
overrides compiles and ignores the budget.
`test_every_windowed_model_holds_its_budget` feeds each windowed kind past a
tiny refusing budget, and fails for one that runs on.
`test_the_budget_table_names_every_windowed_builder` fails until a new builder
that takes `window_budget` is in that test's table.
`test_second_opinion::TestWindowedSpread` holds `ewridge` and `lasso` to
`numpy`, and a new windowed kind that predicts a target belongs in its
parametrization.

**A weight per target.** `ewridge`, `lasso`, `kalman`, `robust` and `holt`
keep one. With `target_n_eff_into` overridden, each target's `min_periods` is
checked against its own weight rather than the shared `n_eff`. The default
leaves every target on the shared one, and
`test_a_sparse_target_warms_up_on_its_own_weight` fails for a model that keeps
the default.

**A parameter in the targets slot.** `bocpd`'s hazard column and `hmm`'s
exogenous column are read out of `y`. Without `predict_with`, `predict`
answers from the configured default, and disagrees with the step on every row
where the column differs. The default ignores `y`, which is right for every
model that regresses its targets: that is what makes `predict` out of sample.
`predict_is_the_step_without_the_step` calls `predict_with`, and
`a_value_in_the_targets_slot_reaches_predict` feeds a column that varies and
counts the rows where ignoring it would change the answer.

#### Restoring

`restore` checks the state's shape against its cfg before returning it. Every
vector must be at the width `new` gives it, with one accumulator per target or
class, and a window or a lag ring exactly when the cfg asks for one. A
mismatch is refused as `StateError::Invalid(".. wrong shape")` (review
2026-09-18, B3). **A state's vectors are whatever a hand-edited JSON file or a
flipped length header makes them**, and an unchecked one loaded and panicked
on the first `step`'s indexing. The nested accumulators (`EwCov`, `EwLagCov`,
`MarginalLags`, `MarginalBins`, `FeatureMoments`) carry a `has_shape` for it.

**Check:** the model's own `a_state_of_the_wrong_shape_is_refused` unit test,
which 17 of the 20 model modules carry, and
`crates/online-polars/tests/summary.rs`'s bit-flip fuzz, which runs
`fit_predict` on every state that loads and fails on a panic. `ewridge`, `sgd`
and `kalman` refuse a wrong shape in `restore` without a unit test of that
name.

### Step 2 — `src/lib.rs`

`pub use <model>::{<Model>, <Model>Cfg};`.

**Check:** the compiler, as soon as `online-polars` names the type.

### Step 3 — `src/model.rs`

A `ModelState::<Model>(Box<...>)` variant, and its arm in `kind()`. The state
is what `save`/`load` serialize, so a new variant is a new schema. Bump
`SCHEMA_VERSION` if an existing layout changes (rule 5); appending a variant
does not.

**Check:** `kind()` is an exhaustive match. Then
`every_model_state_variant_is_probed_here` in `tests/model_contract.rs` fails
until the variant is listed in `PROBED`, which is the reminder for step 4.

### Step 4 — `tests/model_contract.rs`

A `<model>_cfg()`, a `#[test] fn <model>()` that runs `probe_with` and
`check`, and the variant name in `PROBED`. This is the one place every model
is held to the same `n_eff` recursion, slot counts, state round-trip and
bounded-input contract at once. Add a `#[test] fn <model>_predict_is_the_step()`
too, through `predict_is_the_step_without_the_step`, which holds `predict` to
`step`'s `pred`, `n_eff` and `extra`, row by row.

**Check:** the test in step 3, and
`every_model_with_a_recovery_test_has_a_predict_parity_test`, which reads this
file. It fails for a model with a `fn <model>()` but no
`fn <model>_predict_is_the_step()`.

### Step 5 — `tests/golden.rs`

A `#[test] fn <kind>_golden()`: the model on the fixed 60-row stream,
`check("<kind>", &signature(&mut m, slot), GOLDEN_X)`. Generate the constant,
and paste it under the `generated` marker:

```sh
PRINT_GOLDEN=1 cargo test -p online-core --test golden -- --nocapture
```

Prefer a configuration that exercises the busier path: a robust loss, an
annealed rate, standardization. Add a second, plain one if the model has one
(`sgd_squared_golden`, `ftrl_squared_golden`). This pins the core arithmetic
alone, so a move in the Python golden pipeline (step 14) can be placed in the
core or above it. It is also the only oracle `cargo mutants` can see for the
recursion (docs/TESTING.md,
[the mutation blind spot](TESTING.md#the-mutation-blind-spot)).

**Check:** `test_model_registry::test_the_core_golden_file_pins_every_model`
reads this file, and fails for a kind in `KINDS` with no `fn <kind>_golden()`.
Writing it found four models without one (`sgd`, `pa`, `holt`, `ew_cov`).

## The bank — `crates/online-polars`

### Step 6 — `src/spec.rs`

A `ModelKind::<Model> { ... }` variant: serde-tagged, so its snake_case name is
the spec `type`, with `#[serde(default)]` on every optional field. Add its arm
in `kind_name()`, and the name in **`ModelKind::KINDS`**. Add a clause in
`Spec::validate` only for a constraint that crosses the spec (`holt` takes no
features; `ew_cov`, `kmeans` and `micro` no targets). Per-parameter checks
stay in step 1.

**Check:** `kind_name` is exhaustive. `kinds_lists_every_variant_in_order`
fails until `KINDS` matches the enum, and `KINDS` is what every Python check
below reads.

#### A model that is not a regression

Two predicates classify a model that is not a regression:

| predicate | the model it takes | what it changes |
|---|---|---|
| `ModelKind::predicts_no_target` | a model that predicts no target | `validate` refuses every residual-based flag (`emit_sigma`, `emit_metrics`, `resid_quantiles`, `conformal`, ...) for it by name, rather than emitting nothing, and the bank counts its output slots from the schema |
| `ModelKind::is_unsupervised` | as well, a model with **no target column at all**: `ew_cov`, `kmeans`, `micro`, `deco`, `rcov`, `hmm`, `corrchange`, `bocpd` | the leak check exempts it |

`ew_class`, `seqtest` and `marginal` are in the first and not the second. For
`ew_class`, the label column travels as `targets[0]`, so everything that reads
the target column by name (`keep_columns`, the lazy source's projection)
works unchanged.

**Check:** `test_diagnostics::test_rejected_for_ew_cov` and
`test_kmeans::TestRefusals::test_residual_diagnostics_are_refused_by_name`
(and `test_micro`'s and `test_ew_class`'s twins) pin the refusal, for every
flag.

#### Two arms nothing is exhaustive over

Two more arms in the same file are easy to miss, because nothing is exhaustive
over them:

| arm | takes an arm when | for example |
|---|---|---|
| **`Spec::decays()`** | the model has no decay at all. Without the arm, the spec will demand a `halflife` it cannot use; `validate` should then refuse `halflife`/`lam` for it by name | `seqtest` counts trials, `rcov` accumulates a block, `corrchange` runs a test and `bocpd` has a run-length posterior instead |
| **`default_min_periods`** | the schema's own warm-up is the gate rather than `k + 1` | `corrchange`'s span, `bocpd`'s row one, `hmm`'s `warm_rows` |

**Check:** neither is exhaustive; `tests/test_<model>.py` is where the refusal
and the gate get pinned.

#### A block rather than a row

A model whose value is a **block** rather than a row, such as `rcov`, is a
third shape again. It needs `group` and `group_close`, emits `n_eff` alone per
row, and its output leaves the bank through `Bank::closed_groups` when the
group closes. `Spec::validate` refuses it without the two columns, and
`tests/test_closed_groups.py` is the file that covers the queue.

#### A parameter measured in clock units

**A parameter measured in clock units** — a window, a solve cadence, a
halflife of the model's own — is a `Span` (or a `SpanList` for one value per
slot), not an `f64`, so a temporal clock can give it as a duration
(docs/PLAN.md task 88). It goes into **`CLOCK_FIELDS`** under the model's
`type`, and into the match in **`Spec::clock_spans`**, which the bank uses to
refuse a number on a temporal clock and a duration on a numeric one. A rate
*per* clock unit, such as `kalman`'s `q`, has no duration form: it goes into
`CLOCK_RATES` and `Spec::clock_rate` instead.

**Check:** `clock_fields_are_exactly_the_fields_that_take_a_duration` walks
every field serde knows and fails on a `Span` missing from the table, or a
table entry that refuses a duration; `every_clock_field_is_walked` fails on
one `clock_spans` does not read.

### Step 7 — `src/stream.rs`

An `AnyModel::<Model>(Box<...>)` variant, and its arms:

| where | the arm |
|---|---|
| the `dispatch!` macro | the variant's arm |
| `solve_failures` | 0 for a model that never factorizes |
| `coefficients` | the per-target layout the `coef` field reports |
| `restore` | the `ModelState` arm |
| `build_one` | the `ModelKind` arm that turns spec fields into a `Cfg`. Defaults are decided here: `holt` reads the spec's `halflife` as its level halflife, so the shared parameter means the same thing everywhere |
| `combos` | `vec![Combo::default()]`, unless the model is a grid |

**Check:** every match but `restore` is exhaustive. `restore` has a
catch-all, because `ModelState::EwCov` is a component, not a bank model. So a
model left out of it restores as `WrongModel`. The save/load sweeps
(`test_properties::test_save_load_is_transparent`,
`test_semantics_all_models::test_save_load_mid_stream`) catch that once the
model is in their lists, which step 12 enforces. `Stream::restore` then
compares each restored model's `n_features`, `n_targets` and `n_outputs` with
the fresh model's. So a state of another width is refused before the first
row, rather than indexing past the spec's columns (review 2026-09-18, B3). A
model whose widths are not the cfg's own, such as a grid or a per-target
layout, must answer those three from the same cfg both times.

### Step 8 — `src/bank.rs`

**Nothing, unless the outputs are not one `pred`/`resid` pair per target per
combo, or the coefficient vector is not `[intercept] + features` per (target,
combo) slot.**

The outputs of these models are not one `pred`/`resid` pair per target per
combo, and are the cases in `output_index`:

| model | what it writes |
|---|---|
| `ew_cov` | statistics, no target |
| `kmeans` | an assignment and two distances, no target |
| `micro` | a label, an id, a flag, two counts |
| `ew_class` | a class and its posteriors |
| `seqtest` | two log e-values and two counts per target, no `coef` |
| `marginal` | `n_eff` alone: its pairs are state, read by `Bank::marginal` as a frame, and a spec that is not a `marginal` is refused there by name |
| `lasso` | a path |
| `deco` | `u`, `rho` and a `loglik`, per block and per pair of blocks |
| `rcov` | `n_eff` alone: its block leaves through `Bank::closed_groups` |
| `hmm` | a state and its posteriors, `ew_class`'s shape without the labels |
| `corrchange` | a statistic, a critical value, a flag and a counter, on the rows where a span closes |
| `bocpd` | a changepoint probability, two run lengths, a predictive mean per feature and a log score |

`coef_fields` names the slots, and these are its special cases:

| model | its `coef` slots |
|---|---|
| `holt` | `level`, `trend` |
| `ew_cov`, `seqtest` and `marginal` | none |
| `kmeans` | `k` slots `cluster{j}` in place of the targets, one coordinate per feature |
| `ew_class` | one slot per class, named by the class, one coordinate per feature |
| `micro` | none: its `coef` is one row per *live* summary, so the length is not a property of the spec, and `coef_index` refuses it by name |

An output that is not an `f64` needs its own `Source` variant and dtype. NaN
is null for all four:

| variant | materialized as | what it carries |
|---|---|---|
| `Source::Cluster` | `i32` | a small count, read out of the `pred` buffer |
| `Source::Id` | an `i64` | `micro`'s id, `seqtest`'s counts |
| `Source::Flag` | a `Boolean` | |
| `Source::Label` | its name (`F64Column::finish_label_array`) | a class index |

**An *input* that is not an `f64` column needs its own reader in `extract`.**
`ew_class`'s label goes through `label_column`. The adapter (`crate::arrow`,
`Want::Text`) has cast it to text as it does a group key, and the bank maps it
to the class index, with an undeclared value an error naming the row.

**An input that is *another spec's output* makes the spec a phase-two spec.**
`seqtest`'s `a`/`b` comparison reads `resid_<t>` from the two sides' structs.
`ModelKind::compares()` names the sides, and `resolve_compare` checks them at
`Bank::new`. `fit_predict` / `predict_on_pool` run every other spec first,
then fill the comparison's targets from the assembled structs
(`compare_targets`), returning the columns in spec order. A new model that
reads another's output goes through the same two switches, not a third phase.

**Check:**
`test_portability.TestOutputSchemaStability.test_names_match_the_realized_struct_for_every_model`
compares the declared field names with the struct the bank actually produces,
for every model in its list, times four output combinations (three when
written). `test_coef.test_every_model_lays_out_as_coef_index` checks each
model's `coef` list is as long as its named slots.

**`crates/online-cli` and `crates/online-py` need nothing**: both build from
the spec.

## The Python surface — `python/polars_online`

### Step 9 — `_spec.py` and `spec.py`

In `_spec.py`, write the builder:

| part | what it is |
|---|---|
| the signature | `@_checked def <name>(name, *, targets, features, <own parameters>, **common: Unpack[CommonKwargs])` |
| the body | it returns `_common(name, {"type": "<name>", ...}, ...)` |
| the docstring | the update equations |
| a clock-unit parameter | annotated `float \| Duration`, which is what makes the builder write a duration as text |
| a parameter where `inf` means something | a `_INF_OK` entry. Rust parses those as `Num`, and `validate` says `finite` of every other float |
| `__all__` | the name |

Then in **`spec.py`**, the import and `__all__`.

| check | what it holds |
|---|---|
| `test_model_registry::test_every_rust_kind_has_exactly_one_builder` | fails while a kind has no builder; `test_minimal_names_every_builder` then sends you to step 13 |
| `test_temporal_clock::TestEveryClockParameterTakesADuration` | the annotations to `CLOCK_FIELDS`; it fits each clock parameter both ways on a temporal clock |
| `test_error_messages::test_the_inf_table_matches_the_rust_side` | `_INF_OK` to what Rust's parser and `validate` accept |
| `crates/online-polars/tests/spec_inf.rs` | `validate` to the same verdicts from TOML, where it is the only gate |

### Step 10 — `tests/api_surface.txt`

**Field names are API, and the snapshot is where they are pinned.** Add a
`<model> minimal` case to the `[output field grammar]` list in
`tests/test_api_surface.py`, then regenerate the snapshot:

```sh
UPDATE_API_SURFACE=1 uv run pytest tests/test_api_surface.py
```

**Check:** `test_api_surface_matches_the_snapshot` fails on the new
constructor signature.
`test_model_registry::test_the_api_snapshot_pins_every_models_output_fields`
fails until the minimal case is there.

## Tests — `tests/`

### Step 11 — `tests/test_<model>.py`

The Python-side oracle, through `ModelBank`: a numpy reference in
`tests/reference.py` where the model has a closed form, and a longhand
recursion otherwise. `huber` and `quantile` share `test_robust.py`. The
per-model sweeps below cover the surfaces and the schema, so this file is for
the arithmetic, which they cannot see.

**Check:**
`test_model_registry::test_every_builder_has_a_per_model_test_file` fails for
a builder with no such file, or for one whose file never calls
`po.spec.<builder>(`. Writing it moved `ewridge` and `rls` out of
`test_bank.py`, where their oracles had lived unnamed.

### Step 12 — Per-model sweeps

One entry each in `test_semantics_all_models.MODELS`,
`test_properties.MODELS`, `test_edge_cases.MODELS` and
`test_portability.TestOutputSchemaStability._ALL_MODELS`. Every entry is
`(builder name, the least it needs to be constructible)`.

**The sweeps assert on `pred` and `resid`, so a model with no prediction sits
them out**, through `test_model_registry.REGRESSIONS`. Those models are
`ew_cov`, `kmeans`, `micro`, `ew_class`, `seqtest`, `marginal`, `deco`,
`rcov`, `hmm`, `corrchange` and `bocpd`. Each gets its own schema test
instead, such as
`test_portability.TestOutputSchemaStability.test_kmeans_names_match_the_realized_struct`,
`test_micro_names_match_the_realized_struct`,
`test_ew_class_names_match_the_realized_struct` and
`test_seqtest_names_match_the_realized_struct`. And it gets its own
chunk-invariance, save/load, null-row and zero-weight tests in its step-11
file.

**A model that reports on *some* rows by design also goes into
`model_contract.rs`'s `SPARSE_OUTPUT`.** A span-based test, for one, writes
its statistic where the span closes. Otherwise the predict-parity helper will
ask it for 300 rows with every slot filled, and fail.

**Check:** `test_model_registry::test_the_sweeps_cover_every_regression_model`.

### Step 13 — `tests/test_model_registry.py`

The model's `MINIMAL` entry. This is the file that holds every list in this
section to `KINDS`.

**Check:** `test_minimal_names_every_builder`.

### Step 14 — `tests/test_golden_pipeline.py`

A spec in `specs()`, then print the pinned values:

```sh
uv run python tests/test_golden_pipeline.py
```

**Copy only the new model's lines into `GOLDEN`.** If any other line moved,
that is a finding, not a regeneration.

**Check:** `test_model_registry::test_the_golden_pipeline_pins_every_model`.
Writing that check found `ftrl` missing: nine models had been pinned on three
operating systems, and the tenth was not.

## Docs

### Step 15 — `README.md`

A heading `` #### `<name>` — ... `` under its family's `###` heading in
`## Models`, with the equations, the parameters and when to reach for it. The
families are linear models; moments and correlation; clustering and
classification; and sequential tests and regimes. Add a row under the same
family in the model table.

**Check:** `test_model_registry::test_the_readme_documents_every_model`.
`huber` and `quantile` share a heading; the regex knows.

### Step 16 — `CHANGELOG.md` and the design note

**`CHANGELOG.md`**, and the design note wherever the model was proposed:
`docs/PLAN.md` §4 for the original six, `docs/ENHANCEMENTS.md` for the rest.

**No check.**

### Then the gate

Run the gate, unpiped. One commit per step group is fine: the registry tests
will fail in between, which is what they are for.

```sh
./scripts/gate.sh   # unpiped
```

## Adding an output or a parameter instead

| what you add | where it goes | what pins it |
|---|---|---|
| **a new *common* parameter**, one every model takes, like `label_delay` | the field on `Spec` in `crates/online-polars/src/spec.rs` with `#[serde(default)]`, and its validation in `Spec::validate`; `ExprKwargs` in `python/polars_online/_kwargs.py`; `_common`'s signature *and* the dict it builds, in `python/polars_online/_spec.py` | the API snapshot (`tests/api_surface.txt`) records the new keyword and its default, and regenerating it is the diff to read |
| **a new output field** on every model | `FieldMeta` in `crates/online-polars/src/bank.rs` carries the name and dtype (IMPROVEMENTS X1); the emit flag goes on `Spec` and `CommonKwargs` | `test_portability::test_exact_field_names_for_a_grid_spec`, plus the API snapshot |
| **a new kind of input column**, beyond features, targets and the weight | `DataSummary::layout` and `feed_row` in `crates/online-polars/src/summary.rs` decide which columns `describe()` lists, and in what order; `Bank::describe`'s `keep` decides which get moments | `tests/summary.rs` pins the frame's column names, and compares every statistic to an oracle computed over the frame, so a column the summary does not know is a failing count there |
| **a new parameter** on one model | the `Cfg` field and its validation in `new` (step 1), the `ModelKind` field with `#[serde(default)]` (step 6), the `build_one` default (step 7), the builder keyword (step 9), the snapshot (step 10); if `inf` means something for it, `_INF_OK` (step 9) | the inf-table test catches the Python half; the compiler catches the Rust half |

**A spec field changes the bytes of every bank file**, since each carries its
specs. So a new common parameter is a layout change under hard rule 5: bump
`SCHEMA_VERSION`, and freeze a fixture for the new one once the layout has
stopped moving (task 44). **If the Python builders write a default the field
would not otherwise have, add it to `Spec::fill_defaults` too.** Otherwise a
TOML spec and the same spec in Python will save different bytes
(`tests/test_no_output.py` compares the two state files).

**If a new output field gets its own `ChunkOut` buffer**, add it to
`LastRow::take` and `LastRow::to_chunk` in `crates/online-polars/src/stream.rs`.
Switch it on in the rich spec of `crates/online-polars/tests/last_row.rs`,
whose field-for-field comparison of the saved last row against the frame is
what catches the omission.
