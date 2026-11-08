<!-- kind: how-to -->
# Add a model

This how-to is for a contributor adding a model kind to polars-online: it
lists every layer a new model touches, from the recursion in `online-core`
to its line in the feature index, with the test that fails if you skip
each step. It assumes you know the library's words (spec, [model
bank](GLOSSARY.md#the-model-bank-and-what-it-reads), state, `weight_sum`)
but not yet the names of its plumbing.

**No step can be folded into another, because each is a real decision:**
what the state is, what the spec calls it, what the outputs are named and
what the tests assert. What can be done is to make each omission fail. The
steps run in the order the data flows, and each names the check that
catches skipping it. Where a step says **no check**, nothing catches the
omission: hold it yourself, by a test of the model's own or, for a page
of the docs, by reading it. Each step also says
what the layer it touches holds, since you need that to write the step.

**Put new code in the file each step names, which is often not the file
the type lives in.** `tests/test_repo_hygiene.py` holds every source file
to 250,000 bytes (`MAX_SOURCE_BYTES`). `crates/online-polars/src/bank.rs`
and `src/stream.rs` stand within a few hundred bytes of it, and
`python/polars_online/_spec.py` within ten kilobytes; `wc -c` gives each
one's size. So a new model's builder, its build configuration and its
output layout each go in a smaller file beside them: steps 7, 8 and 9 name
each one. Step 7 still adds arms to `stream.rs`, so it starts by moving
code out of it.

**The registry is `ModelKind::KINDS`** in `crates/online-polars/src/spec.rs`:
the spec `type` names, in enum order. Python reaches it as
`polars_online._polars_online.model_kinds()`. Every other list is held to
it.

To add a common parameter, an output field or a kind of input column
instead, see [Add an output field or a spec
parameter](contributing/add-an-output-or-a-parameter.md).

## Prerequisites

You need these before the first step:

- A checkout that builds and passes the test gate, as
  [CONTRIBUTING.md](../CONTRIBUTING.md) sets up.
- The model's update equations, and an oracle for them: a third-party
  library that computes the same quantity where one exists, or else the
  recursion written out longhand, an equivalent model configured
  differently, or the optimality conditions.
- The rules every model keeps, in the table of [step
  1](#step-1--srcmodelrs), and the rules a state file's layout keeps, in
  [what each change does to a saved
  state](contributing/add-an-output-or-a-parameter.md#what-each-change-does-to-a-saved-state).

Two commits show the change. List one's files with `git show --stat <commit>`, and read its diff with `git show <commit>`:

```sh
git show --stat aa96ad3   # holt, a small model, with an unrelated example: 17 files
git show --stat 47d3b35   # bocpd: 23 files
```

Both are older than several checks a model now meets, among them the
held-value test, the wrong-shape tests, `test_every_kind.py`, the
window-budget test, the outputs generator and `test_weight_scale.py`. Read
them for the shape of the change, and the steps for what a model touches
now.

The seventeen steps, each with the check that fails if you skip it:

| step | file | what you add | the check that fails if you skip it |
|---|---|---|---|
| [1](#step-1--srcmodelrs) | `crates/online-core/src/<model>.rs` | the model: `<Model>Cfg`, `new`, `impl OnlineModel` | the trait, and the checks in the step |
| [2](#step-2--srclibrs) | `src/lib.rs` | the `pub use` | the compiler |
| [3](#step-3--srcmodelrs) | `src/model.rs` | a `ModelState` variant and its `kind()` arm, its frozen name and its fixture case | `kind()` is exhaustive; `every_model_state_variant_is_probed_here`; `the_variant_names_are_frozen` |
| [4](#step-4--testsmodel_contractrs) | `tests/model_contract.rs` | the contract probe and the predict-parity test | `every_model_with_a_recovery_test_has_a_predict_parity_test` |
| [5](#step-5--testsgoldenrs) | `tests/golden.rs` | `fn <kind>_golden()` | `test_the_core_golden_file_pins_every_model` |
| [6](#step-6--srcspecrs) | `crates/online-polars/src/spec.rs` | the `ModelKind` variant, its `kind_name()` arm, the name in `KINDS` | `kinds_lists_every_variant_in_order`, and the checks in the step |
| [7](#step-7--srcstreamrs) | `src/stream.rs`, `src/stream/model_cfgs.rs` | the `AnyModel` variant and its arms, and the model's build configuration | exhaustive matches; the save/load sweeps for `restore` |
| [8](#step-8--srcbankfieldsrs) | `src/bank/fields.rs` | nothing, unless the outputs or the positions in `coef` differ | `test_names_match_the_realized_struct_for_every_model`; `test_every_model_lays_out_as_coef_index` |
| [9](#step-9--_spec_familypy-_specpy-and-specpy) | `python/polars_online/_spec_<family>.py`, `_spec.py`, `_spec_words.py`, `spec.py` | the builder | `test_every_rust_kind_has_exactly_one_builder`, and the checks in the step |
| [10](#step-10--testsapi_surfacetxt) | `tests/api_surface.txt` | the regenerated snapshot | `test_api_surface_matches_the_snapshot`; `test_the_api_snapshot_pins_every_models_output_fields` |
| [11](#step-11--teststest_modelpy) | `tests/test_<model>.py` | the Python-side oracle | `test_every_builder_has_a_per_model_test_file` |
| [12](#step-12--per-model-sweeps) | four sweep lists, and `tests/convergence.py` | one entry each; a model that predicts a target registers its convergence: a known-truth stream, a third-party oracle and its classes; a grid against an oracle registers with the oracle ledger | `test_the_sweeps_cover_every_regression_model`; `test_every_regression_registers_its_convergence`; `test_every_module_with_a_grid_registers_or_is_pending` |
| [13](#step-13--teststest_model_registrypy) | `tests/test_model_registry.py` | the `MINIMAL` entry | `test_minimal_names_every_builder` |
| [14](#step-14--teststest_golden_pipelinepy) | `tests/test_golden_pipeline.py` | a spec in `specs()`, and its lines in `GOLDEN` | `test_the_golden_pipeline_pins_every_model` |
| [15](#step-15--docsmodelsmd-and-docsfeaturesmd) | `docs/MODELS.md`, `docs/FEATURES.md`, `docs/OUTPUTS.md`, `llms.txt` | the model's heading, its row and its API links; its line in the feature index; its family and field meanings in the OUTPUTS generator; the model count | `test_the_models_page_documents_every_model`, the `test_api_links` pair, `test_feature_index`, the `test_outputs_doc` trio, `test_it_says_how_many_models_there_are` |
| [16](#step-16--changelogmd-and-the-design-note) | `CHANGELOG.md`, and the design note | an entry | **no check** |
| [17](#step-17--cratesonline-benchsrcscalingrs) | `crates/online-bench/src/scaling.rs`, `docs/PERFORMANCE.md` | a `<kind>.history` cell and a cell for each size the model has, each with its declared growth, and their rows in the table | `every_kind_has_its_cells`; each cell's own test in `crates/online-bench/tests/scaling.rs` |

## The recursion — `crates/online-core`

The model's arithmetic lives in `crates/online-core`, which knows nothing
of Polars, Python or the bank. Its Rust tests are the only ones `cargo
mutants` sees, so the recursion is pinned here.

### Step 1 — `src/<model>.rs`

Write the new module. It holds three things:

| part | what it is |
|---|---|
| `<Model>Cfg` | the struct of the model's settings |
| `<Model>::new(cfg) -> Result<Self, String>` | the constructor, which validates the cfg |
| `impl OnlineModel` | `step_into` *and* `predict`. `step_into` writes the row's `Step` into the caller's buffer. The trait provides `step` from it, into a new `Step`, and no model overrides `step`. `predict` is the step without the step: what `step` would report for the row, state untouched |

**Put every parameter check in `new`,** because the command line and the
bank both construct the model through `new`. The spec layer checks only
what crosses the spec (step 6). **Derive
`step`'s prediction from `predict`**, so the two cannot drift. State the
update equations in the docstring.

**Give the module's unit tests an oracle as well as a golden number.**
Where another library computes the quantity, the oracle is that library:
`faer` in Rust tests, and scikit-learn, scipy or statsmodels in
`tests/test_second_opinion.py`
([docs/TESTING.md](TESTING.md#libraries-the-package-does-not-depend-on)).
Otherwise it is the recursion written out longhand, an equivalent model
configured differently, or the optimality conditions.

These rules hold for every model:

| rule | why | the check |
|---|---|---|
| `weight_sum` is the weight before this row's update and before its own decay | one `min_weight` must mean the same thing across a bank (CLAUDE.md, hard rule 8) | `model_contract.rs`, every model against one recursion |
| a zero-weight first row divides no 0/0 | a NaN in the state never washes out (hard rule 9) | the contract's zero-weight probes |
| a zero-weight row still decays the weight, advancing the clock and nothing else | the stream with the row and the stream without it, its clock step carried into the next row, report the same `weight_sum` | `zero_weight_rows_only_advance_the_clock`, which names no decay |
| a parameter measured in clock units acts on the clock, so the numbers do not change with how densely rows arrive; anything that still counts rows says so in the docstring | a half-life or a window means the same at any row rate | the model's own tests; the docstring check of step 9 |
| a row weight scales evidence: in a model that keeps means, scaling every weight changes nothing but `weight_sum` | a weight is relative | `tests/test_weight_scale.py`, over `scripts/release_probe.py`'s `WORKLOAD`; a model whose outputs move with the scale goes into its `EXCEPTIONS`, with the reason |
| a row of weight 0 past the underflow forgets as a row of weight 0 does | a tiny weight must not freeze the state | `a_zero_weight_row_past_the_underflow_forgets` |
| a value that is not `usable` -- not a number, infinite, or past `INPUT_BOUND` -- is refused by the trait's one rule: a target is absent; a row with a feature or a weight that is not usable is learned at weight 0 and keeps nothing of its own; it reports nothing for a feature, `predict`'s answer for a weight. `step` begins with `crate::model::refused_step`, and a `predict` that reads its features with `refused_predict` (`refused_predict_with` where `y` is read) | the contract is the core's own, not the bank's alone: a direct Rust caller gets the bank's reading, and a zero-weight update never forms `0 · NaN` | `refuses_unusable_values` in `tests/model_contract.rs`, from `predict_is_the_step_without_the_step` for every model, against the same row at weight 0 byte for byte, and from `probe_with` against the accessor |
| no `unsafe`, and `f64` everywhere | hard rule 6 | `unsafe_code = "forbid"`, and review |

**Check:** the trait. A missing method is a compile error. The shared
contract is step 4.

#### When the model holds more than a recursion

Some models keep state or read inputs that the trait's defaults do not
handle. Each such case overrides a method, and each has its check:

| if the model … | it implements | the check |
|---|---|---|
| keeps a ring of past rows: a lag ring, a window | **`clear_lags`**, dropping that ring and nothing else | `probe_with` in `tests/model_contract.rs`, against `KEEPS_LAGS` |
| has a `window_size` | its snapshots in `online_core::Snapshots`, a **`Footprint`** for the snapshot type (the numbers it holds, in bytes) and a **`Pack`** (`online_core::pack_fields!`, every field named), which writes every snapshot but the oldest as its numbers alone; where forming the snapshot first would cost a copy, an **`Offer`** view that writes the same words straight from the model, which debug builds check against the snapshot it forms; **`set_window_budget`** and **`window_over_budget`** reaching the ring, so that `window_budget` bounds it, with **`window_shadow`** and **`window_could_refuse`** sizing its next snapshot alike; **`set_window_closed`** reaching the ring too, so that the spec's `closed` decides its edge, and a `closed` field in its `ModelKind` variant | the compiler, for `Pack`; `tests/test_window_budget.py`: `test_every_windowed_model_holds_its_budget` and `test_the_budget_table_names_every_windowed_builder`; `crates/online-core/tests/window_refusal.rs`, in whose list it belongs; `crates/online-bench/tests/window_memory.rs`, likewise; `crates/online-polars/tests/window_edge.rs`, every test of which runs each model under a hard window under both edges |
| has a hard window and predicts a target | an entry in `ModelKind::window_and_cadence`, which returns the window and its snapshot cadence, so the stream cuts its `sigma` and `zscore` at the fit's window | `test_second_opinion::TestWindowedSpread`, in whose parametrization it belongs |
| has a `window_size` and decays its moments | **`Snapshots::learn_decayed(&mut clock, d_clock, &[lam], make)`** on every row it learns, a row of weight 0 included, so the ring's running product takes each row's decay factor; and its window's edge read through **`Snapshots::cut(clock, decay, which)`**, which decays the snapshot by that product (a model with two decays, as `lasso` has, passes both factors and reads each with its index `which`). One factor of the summed clock instead differs from the live moments' decay by the per-row factor's rounding times the rows in the window, so a target held at one value can read a variance near 1e-15 instead of 0. No model calls `offer` itself | **no check**: chunk invariance and the window tests pass either way, so the model's own unit tests must hold the edge |
| keeps a weight per target: the rows each target was present on | **`target_n_eff_into`** | `test_a_sparse_target_warms_up_on_its_own_weight` in `tests/test_bank.py`, to which it is added |
| reads a number out of `y` that it does not regress | **`predict_with(x, y, d_clock)`** | `predict_is_the_step_without_the_step` and `a_value_in_the_targets_slot_reaches_predict` |
| emits a `Step` (every model) | **`step_into`**, which the trait requires: every slot of the `Step`'s predictions, its weight and its `extra` written into the caller's `Step` whatever it held -- `pred` by `clear` and `resize` (or the model's own fill), and an `extra`'s vector reused where the buffer already holds that variant (`lasso`'s `lam_selected`, `huber`'s `scale`). A helper that builds the row's answer writes into a `&mut Vec<f64>` too, with a wrapper that returns a new one for `predict`. The bank steps every model through `step_into`, into one `Step` kept per model instance, so a row allocates no answer: a new `Step` a row would be a heap allocation and a free, on Linux a third of the cheapest model's row | `predict_is_the_step_without_the_step` in `tests/model_contract.rs`, which steps a twin through `step_into` on a buffer left with another row's answer (either `extra` variant, of the wrong length) and holds its answer and state to `step`'s bits, and `refuses_unusable_values`; `crates/online-bench/tests/allocations.rs` for the count a row |
| counts a value that is not usable, as `audit` does, where other models refuse it | no `refused_step`; its kind in `model_contract.rs`'s `COUNTS_UNUSABLE`, which `refuses_unusable_values` passes by; an arm in `ModelKind::reads_every_row` (`spec.rs`), so the stream hands it every row, and one in `Instance::reset` (`stream.rs`) if a reset must not rebuild it | the model's own tests of each kind of value; `crates/online-polars/tests/audit.rs` for the rows the stream hands it |
| solves on a schedule, as `ewridge`, `lasso`, `huber` and `quantile` do | **`set_solve_share`** and **`solve_share`**, reaching the cfg's `solve_share`. The spec's default share goes into the cfg when the model is built, and the stream calls `set_solve_share` with the spec's share on every model it restores (`crates/online-polars/src/stream.rs`), so a state saved before the rule runs at the spec's cadence | `exactly_the_scheduled_solvers_report_a_solve_share` in `tests/model_contract.rs`, which names the four, and to which it is added |
| refits its first rows to convergence, as `huber`, `sgd`'s Huber loss, `quantile` and `marginal(bin_stat="huber")` do | a buffer of its first learned rows and a `RefitStatus` a target ([`huber_refit.rs`](../crates/online-core/src/huber_refit.rs)), and: a default count beside **`default_refit_rows`** and `default_bin_refit_rows`, with `0` for off; its buffer's bytes through **`refit_reservation`**, against which an arm in **`ModelKind::refit_budget_mib`** (`spec.rs`) bounds a spec's groups under `refit_budget`; arms in `AnyModel::refit_status` and `refit_give_up` (`stream.rs`), which `summary()`'s `refit_given_up` and the budget's notice read; a refit at its pass cap counted in `solve_failures`, and named by `ModelBank.check`'s wording for its kind (`python/polars_online/_check.py`, the `solve_failures` finding); a state saved before the refit loading with it off | **no check**: `refit_status`, `refit_give_up` and `refit_budget_mib` end in a catch-all, so a model left out never refits under the budget and reports nothing. Its own test file holds it, as `tests/test_huber_refit.py` and `tests/test_quantile_refit.py` do: the converged fit against a third-party one on the buffer, one chunk or many, a bank saved before, at and after the refit, the activation (the fit moves only after the buffer fills), and the budget giving up the later groups, which `tests/test_summary.py`'s column list pins |
| reports the readiness statistics: effective degrees of freedom, each coefficient's data share, a row's leverage | **`error_inflation_into`**, **`error_inflation_gate_into`** (where a bound may stand in below the gate's limit), **`row_error_inflation_into`** and **`support_coef`** (MODELS.md's [Warm-up gates](MODELS.md#warm-up-gates) says what each gates). Which models report them is the table of predicates under [What a model reads for the diagnostics](#what-a-model-reads-for-the-diagnostics), and `Spec::validate` refuses the parameters that read them for every other model, so a new one that reports them is added there | `a_model_without_the_optional_statistics_reports_none_of_them` (`model.rs`) for the defaults; `tests/readiness.rs` for `ewridge` |

**Clear a lag ring on the two events that break it.** The stream calls
`clear_lags` on a session change and on a capped clock gap
(`ClockAdvance::capped`). After those two events, "the row `ℓ` back" no
longer means a row `ℓ` ago. A reset already rebuilds the model, and an
ordinary gap is what decay is for. `clear_lags` is **not** a reset: means,
co-moments and the weight must not move. `probe_with` serializes the
state, calls `clear_lags`, and requires the bytes to be identical unless
the model is listed in `KEEPS_LAGS`. So a model that quietly clears a mean
fails, and so does one with a ring that forgot to declare itself. It then
restores a copy from the *cleared* state, and requires the two to report
the same rows from there.

**Read a ring's depth from the model's settings, never from a container's
`capacity()`.** A clone and a msgpack round-trip both shrink the capacity
to the length it holds. So a ring copied while short would keep that depth
forever. The contract's copy is taken with an *empty* ring, which grows
back. The case that bites is a partly filled one, so the model's own unit
tests need a save/restore at every depth from empty to full.

**Override both window methods.** The trait's defaults do nothing, so a
model that forgets the two overrides compiles and ignores the budget.
`test_every_windowed_model_holds_its_budget` feeds each kind under a hard
window past a tiny refusing budget, and fails for one that runs on.
`test_the_budget_table_names_every_windowed_builder` fails until a new
builder that takes `window_budget` is in that test's table.
`test_second_opinion::TestWindowedSpread` holds `ewridge` and `lasso` to
`numpy`, and a new kind under a hard window that predicts a target belongs
in its parametrization.

**Override `target_n_eff_into` in a model that keeps a weight per
target.** `ewridge`, `lasso`, `kalman`, `huber`, `quantile`, `holt`, `sgd`, `pa`,
`ftrl` and `rls` keep one. With `target_n_eff_into` overridden, each
target's `min_weight` is checked against that target's own weight. The default leaves every target on the shared one, and
`test_a_sparse_target_warms_up_on_its_own_weight` fails for a model that
keeps the default. The case that tells is a list of unequal thresholds.
Take a model that gates each target on its own weight inside the model,
but against the smallest threshold, and keeps the default: it checks that
list against the shared weight.

**Override `predict_with` in a model that reads a parameter out of the
targets slot.** `bocpd`'s hazard column and `hmm`'s exogenous column are
read out of `y`. Without `predict_with`, `predict` answers from the
configured default, and disagrees with the step on every row where the
column differs. The default ignores `y`, which is right for every model
that regresses its targets: that is what makes `predict` out-of-sample.
`predict_is_the_step_without_the_step` calls `predict_with`, and
`a_value_in_the_targets_slot_reaches_predict` feeds a column that varies
and counts the rows where ignoring it would change the answer.

#### When the model keeps a running mean

**Keep a mean that anything is centered on as a pair.** A variance, a
co-moment or a standardized value is centered on a mean. The pair is `hi`,
the double, and `lo`, what it leaves out (`crate::comp`). Take each
deviation as `comp::dev(x, hi, lo)` and each step as `comp::add(hi, lo, b·d)`. Take no step on a row of weight 0, where adding zero could round the
pair afresh. A plain `m += b·(x − m)` stops a few rounding steps short of a
value the stream holds, and whatever is centered on it then rests on that
gap.

**Keep a `Runs` beside the moments of a window read by subtraction.** It
zeroes the spread of a value held over the whole window. Subtracting the
rows that leave would give that spread as rounding noise.

| if the model … | the check |
|---|---|
| standardizes by, divides by or reports a spread | `crates/online-core/tests/held_values.rs`, to which it is added |
| has a `window_size` over such moments | a held-slot test beside it, as `window.rs` and `marginal.rs` have |

#### Restoring

**Check the state's shape against its cfg in `restore`, before returning
it.** Every vector must be at the width `new` gives it, with one set of
running sums per target or class, and a window or a lag ring exactly when
the cfg asks for one. Refuse a mismatch with `crate::model::damaged(kind,
"its running sums have the wrong shape")`, which says the state file is
damaged and how to recover. **A state's vectors are whatever a hand-edited
JSON file or a flipped length header makes them**, and an unchecked one
would load and then panic on the first `step`'s indexing. The nested
running sums (`EwCov`, `EwLagCov`, `MarginalLags`, `MarginalBins`,
`FeatureMoments`) carry a `has_shape` for it.

**Check:** the model's own `a_state_of_the_wrong_shape_is_refused` unit
test, which every model module carries. `sgd` and `kalman` check the shape
in their `TryFrom`, as a saved state is read, so their tests go through the
encoding. `crates/online-polars/tests/summary.rs`'s corruption sweeps cut
and bit-flip a saved state of every model kind. They run `fit_predict` on
every one that loads, and fail on a panic
(`every_model_kind_refuses_or_loads_a_corrupt_file_never_panics`, held to
`ModelKind::KINDS`).

**Add the model's `_lo` fields to the low-parts test.** A state without
the means' low parts is of a layout older than this build's schema, so it
is refused. `model_contract.rs`'s
`every_model_refuses_a_state_without_the_low_parts` drops every `_lo` field
of every model's state and requires the load to fail.

### Step 2 — `src/lib.rs`

Export the model: `pub use <model>::{<Model>, <Model>Cfg};`.

**Check:** the compiler, as soon as `online-polars` names the type.

### Step 3 — `src/model.rs`

Add a `ModelState::<Model>(Box<...>)` variant, and its arm in `kind()`.
The state is what `save` and `load` serialize, so the variant joins every
state file's format. Appending a variant moves no existing layout, so it
needs no `SCHEMA_VERSION` bump. Changing an existing layout does ([what each change does to a saved
state](contributing/add-an-output-or-a-parameter.md#what-each-change-does-to-a-saved-state)).

**Add the variant's name to the frozen list, and a fixture case.** The
variant name is the tag in every state file, so the names are frozen in
`FROZEN`, in `crates/online-core/tests/state_fixtures.rs`'s
`the_variant_names_are_frozen`. Add a case to
`crates/online-core/tests/cases/mod.rs` that keeps what the model's state
can keep, then write its fixture:

```sh
PRINT_STATE_FIXTURES=1 cargo test -p online-core --test state_fixtures
```

**Check:** `kind()` is an exhaustive match. Then
`every_model_state_variant_is_probed_here` in `tests/model_contract.rs`
fails until the variant is listed in `PROBED`, which is the reminder for
step 4. `the_variant_names_are_frozen` fails until the name is in
`FROZEN`, and `every_variant_and_every_case_has_a_fixture` until its
fixture is written.

### Step 4 — `tests/model_contract.rs`

Add a `<model>_cfg()`, a `#[test] fn <model>()` that runs `probe_with` and
`check`, and the variant name in `PROBED`. This file is the one place every
model is held to the same weight recursion, slot counts, state round-trip
and bounded-input contract at once.

Add a `#[test] fn <model>_predict_is_the_step()` beside them, through
`predict_is_the_step_without_the_step`. It holds `predict` to `step`'s
predictions, weight and `extra`, row by row.

**Check:** the test in step 3, and
`every_model_with_a_recovery_test_has_a_predict_parity_test`, which reads
this file. It fails for a model with a `fn <model>()` but no
`fn <model>_predict_is_the_step()`.

### Step 5 — `tests/golden.rs`

Pin the core arithmetic on the fixed 60-row stream:

1. Write a `#[test] fn <kind>_golden()` that runs the model on the stream
   and calls `check("<kind>", &signature(&mut m, slot), GOLDEN_<NAME>)`, where
   `<NAME>` follows the other `GOLDEN_*` constants in the file
   (`ewridge`'s is `GOLDEN_EW_RIDGE`).
2. Generate the constant:

   ```sh
   PRINT_GOLDEN=1 cargo test -p online-core --test golden -- --nocapture
   ```

3. The run prints `GOLDEN <kind>: &[...];`. Declare
   `const GOLDEN_<NAME>: &[f64] = &[...];` with the printed list, under the
   `// --- generated` marker.

**Prefer a configuration that exercises the busier path:** a robust loss,
an annealed rate, standardization. Add a second, plain one if the model has
one (`sgd_squared_golden`, `ftrl_squared_golden`). This pins the core
arithmetic alone, so a move in the Python golden pipeline (step 14) can be
placed in the core or above it. It is also the only oracle `cargo mutants`
can see for the recursion, since `cargo mutants` runs the Rust tests and
never pytest.

**Check:** `test_model_registry::test_the_core_golden_file_pins_every_model`
reads this file, and fails for a kind in `KINDS` with no
`fn <kind>_golden()`.

## The bank — `crates/online-polars`

The bank in `crates/online-polars` turns a spec into a model, steps it over
each chunk's rows and lays its outputs out as columns. A new model joins it
in three files.

### Step 6 — `src/spec.rs`

Add a `ModelKind::<Model> { ... }` variant. It is serde-tagged, so its
snake_case name is the spec `type`. Put `#[serde(default)]` on every
optional field. Add its arm in `kind_name()`, and the name in
**`ModelKind::KINDS`**.

**Add a clause in `Spec::validate` only for a constraint that crosses the
spec.** For example, `holt` takes no features, and `ew_cov`, `kmeans` and
`micro` take no targets. Per-parameter checks stay in step 1.

**Check:** `kind_name` is exhaustive. `kinds_lists_every_variant_in_order`
fails until `KINDS` matches the enum, and `KINDS` is what every Python
check of the later steps reads.

#### A model that is not a regression

Two predicates classify a model that is not a regression:

| predicate | the model it takes | what it changes |
|---|---|---|
| `ModelKind::predicts_no_target` | a model that predicts no target | `validate` refuses every residual-based flag (`emit_sigma`, `emit_metrics`, `resid_quantiles`, `conformal`, ...) for it by name, rather than emitting nothing, and the bank counts its output slots from the schema |
| `ModelKind::is_unsupervised` | as well, a model with **no target column at all**: `ew_cov`, `kmeans`, `micro`, `deco`, `rcov`, `hmm`, `corrchange`, `bocpd`, `audit` | the leak check exempts it |

`ew_class`, `seqtest` and `marginal` are in the first and not the second.
For `ew_class`, the label column travels as `targets[0]`, so everything
that reads the target column by name (`keep_columns`, the lazy source's
projection) works unchanged.

**Check:** `test_diagnostics::test_rejected_for_ew_cov` and
`test_kmeans::TestRefusals::test_residual_diagnostics_are_refused_by_name`
(and `test_micro`'s and `test_ew_class`'s twins) pin the refusal, for every
flag.

#### What a model reads for the diagnostics

Five predicates on `Spec` say which per-model readings a model has. Each is
a `matches!` over a list of models, in `src/spec/readiness.rs` and
`src/spec/diagnostics.rs`, so a new model has none of them until it is
added to the lists it belongs in:

| predicate | the models today | what it decides |
|---|---|---|
| `has_error_inflation` | `ewridge`, `rls`, `kalman`, `lasso` | the noise gate, `max_error_inflation` |
| `has_row_error_inflation` | `ewridge`, `rls`, `kalman` | `emit_error_inflation` and `emit_influence`; the recursive residual of `emit_breaks` and `emit_tails` stands 1 in for the inflation without it |
| `has_se_coef` | `ewridge`, `rls`, `kalman` | `emit_se_coef` |
| `has_robust_se` | `ewridge`, `rls` | `emit_robust_se`, the least-squares sandwich |
| `has_support_coef` | `ewridge`, `huber`, `quantile` | the `support_coef` field beside `coef` |

**Give a new family its own reason for each refusal.** A model refused one
of these readings says why in the refusal. `no_noise_statistic` and the
`match` arms beside each refusal in `Spec::validate` name a reason per
family.

**Check:**
`test_model_registry::test_each_kind_takes_the_per_model_readings_its_predicates_say`
fails on a builder missing from its `DIAGNOSTIC_READS` table. It fails too
on a model that takes a reading the table does not give it, or is refused
one the table does.

#### Three arms nothing is exhaustive over

Three more arms in the same file are easy to miss, because nothing is
exhaustive over them:

| arm | takes an arm when | for example |
|---|---|---|
| **`Spec::decays()`** | the model has no decay at all. Without the arm, the spec will demand a `half_life` it cannot use. `validate` should then refuse `half_life` and `lam` for it by name | `seqtest` counts trials, `rcov` accumulates a block, `corrchange` runs a test and `bocpd` has a run-length posterior instead |
| **`default_min_periods`** | the model's own warm-up is the gate, in place of `k + 1` rows | `corrchange`'s span, `bocpd`'s row one, `hmm`'s `warm_rows` |
| **`Spec::validate`'s refusal of a [formula target](GLOSSARY.md#fitting-a-model)** | the model does not regress its targets, so a formula target does not apply. The match ends in `_ => None`, so a new model accepts one silently until it has an arm | `ew_class`, `seqtest`, a logistic `ftrl`, a logistic or Poisson `sgd` |

**Check:** none is exhaustive. Pin each refusal and the gate in
`tests/test_<model>.py`.

#### A block rather than a row

**Require `group` and `group_close` of a model whose value is a whole
block of rows,** such as `rcov`. It emits `weight_sum` alone per row, and its
output leaves the bank through `Bank::closed_groups` when the group closes.
`Spec::validate` refuses it without the two columns, and
`tests/test_closed_groups.py` is the file that covers the queue.

#### A parameter measured in clock units

**Give a parameter measured in clock units the type `Span`,** so a temporal clock can give it as a duration. A window, a solve cadence
and a half-life of the model's own are such parameters, and one value per
slot is a `SpanList`. Each goes into the tables of
`crates/online-polars/src/spec/clock.rs`:

| the parameter | where it goes |
|---|---|
| measured in clock units | **`CLOCK_FIELDS`**, under the model's `type`, and the match in **`Spec::clock_spans`**, which the bank uses to refuse a number on a temporal clock and a duration on a numeric one |
| a rate *per* clock unit, such as `kalman`'s `q`, which has no duration form | `CLOCK_RATES` and `Spec::clock_rate` |
| a number that means something other than clock units, whose duration is a clock parameter: `bocpd`'s `hazard`, rows or a duration | `DURATION_OR_UNIT_FREE_FIELDS`. `clock_spans` reads it only when it is a duration |

**Check:** `clock_fields_are_exactly_the_fields_that_take_a_duration`
walks every field serde knows. It fails on a `Span` missing from the table,
or a table entry that refuses a duration. `every_clock_field_is_walked`
fails on one `clock_spans` does not read.
`a_duration_or_a_count_is_read_only_as_a_duration` fails on a second-table
entry it reads as a number.

### Step 7 — `src/stream.rs`

**First make room in `stream.rs`.** It stands within a few hundred bytes
of the cap, and the variant and arms this step adds take a few hundred
bytes for any model. Move a self-contained block of it into a submodule under
`src/stream/`, declared with `mod` as `model_cfgs.rs` and `last_row.rs`
are, until `wc -c crates/online-polars/src/stream.rs` leaves room for the
addition. `tests/test_repo_hygiene.py`'s `test_no_tracked_file_is_data_sized`
fails while it does not.

Then add an `AnyModel::<Model>(Box<...>)` variant, and its arms:

| where | the arm |
|---|---|
| the `dispatch!` macro | the variant's arm |
| `solve_failures` | 0 for a model that never factorizes |
| `coefficients` | the per-target layout the `coef` field reports |
| `restore` | the `ModelState` arm |
| `build_one` | the `ModelKind` arm that turns spec fields into a `Cfg`. Write the function that builds the `Cfg` as a `<kind>_cfg` in `src/stream/model_cfgs.rs`, as `audit_cfg`, `bocpd_cfg` and `hmm_cfg` are, and call it from the arm, so the arm stays one line. Defaults are decided here: `holt` reads the spec's `half_life` as its level half-life, so the shared parameter means the same thing everywhere |
| `combos` | `vec![Combo::default()]`, unless the model fits a grid |

**Answer `n_features`, `n_targets` and `n_outputs` from the cfg.**
`Stream::restore` compares each restored model's three widths with the
fresh model's. So a state of another width is refused before the first
row, before it can index past the spec's columns. A model whose widths
are not the cfg's own, such as a grid or a per-target layout, must answer
those three from the same cfg both times.

**Check:** every match but `restore` is exhaustive. `restore` has a
catch-all, because `ModelState::EwCov` is a component that no spec builds
on its own.
So a model left out of it is refused as a damaged state, holding a kind no
spec builds. The save/load sweeps
(`test_properties::test_save_load_is_transparent`,
`test_semantics_all_models::test_save_load_mid_stream`) catch that once the
model is in their lists, which step 12 enforces.

### Step 8 — `src/bank/fields.rs`

**Add nothing, unless the outputs are not one `pred`/`resid` pair per
target per [grid point](GLOSSARY.md#grids), or the coefficient vector is
not `[intercept] + features` per target and grid point.** The output
layout lives in `crates/online-polars/src/bank/fields.rs`: `FieldMeta`,
`Source`, `output_index` and `coef_fields`. `src/bank.rs` is within a few
hundred bytes of the source cap, so put the new model's cases in `fields.rs`.

The outputs of these models are not one `pred`/`resid` pair per target per
grid point, and are the cases in `output_index`:

| model | what it writes |
|---|---|
| `ew_cov` | statistics, no target |
| `kmeans` | an assignment and two distances, no target |
| `micro` | a label, an id, a flag, two counts |
| `ew_class` | a class and its posteriors |
| `seqtest` | two log e-values and two counts per target, no `coef` |
| `marginal` | `weight_sum` alone: its pairs are state, read by `Bank::marginal` as a frame, and a spec that is not a `marginal` is refused there by name |
| `audit` | `weight_sum` alone: its counts are state, read by `Bank::audit` (`bank/audit.rs`) as three frames, and a spec that is not an `audit` is refused there by name |
| `lasso` | a path |
| `deco` | `u`, `rho` and a `loglik`, per block and per pair of blocks |
| `rcov` | `weight_sum` alone: its block leaves through `Bank::closed_groups` |
| `hmm` | a state and its posteriors, `ew_class`'s shape without the labels |
| `corrchange` | a statistic, a critical value, a flag and a counter, on the rows where a span closes |
| `bocpd` | a changepoint probability, two run lengths, a predictive mean per feature and a log-likelihood, `loglik` |

`coef_fields` names the positions in `coef`, and these are its special cases:

| model | its positions in `coef` |
|---|---|
| `holt` | `level`, `trend` |
| `ew_cov`, `seqtest`, `marginal` and `audit` | none |
| `kmeans` | `k` positions `cluster{j}` in place of the targets, one coordinate per feature; the center in `coef_index`'s `cluster` column, and no target |
| `hmm` | `k` positions `state{j}`, one mean per feature; the state in `coef_index`'s `state` column, and no target |
| `ew_class` | one position per class, named by the class, one coordinate per feature; the class in `coef_index`'s `class` column, and the label column as its target |
| `micro` | none: its `coef` is one row per *live* summary, so the length is not a property of the spec, and `coef_index` and `coef_fields` are empty for it |

**Give an output that is not an `f64` its own `Source` variant and
dtype.** NaN is null for all three:

| variant | materialized as | what it carries |
|---|---|---|
| `Source::Id` | an `i64` | every integer, read out of the `pred` buffer: a cluster label (`kmeans`' and `micro`'s), `micro`'s id and live counts, `hmm`'s state, `seqtest`'s counts. One integer dtype for every output, so a test holds a new model to it |
| `Source::Flag` | a `Boolean` | |
| `Source::Label` | its name (`F64Column::finish_label_array`) | a class index |

**Give an *input* that is not an `f64` column its own reader in
`extract`** (`src/bank.rs`). `ew_class`'s label goes through `label_column`. The adapter
(`crate::arrow`, `Want::Text`) has cast it to text as it does a group key,
and the bank maps it to the class index, with an undeclared value an error
naming the row.

**Read *another spec's output* through the phase-two switches.** A spec
whose input is another spec's output is a phase-two spec. `seqtest`'s
`a`/`b` comparison reads `resid_<t>` from the two sides' output columns.
`ModelKind::compares()` names the sides, and `resolve_compare` checks them
at `Bank::new`. `fit_predict` and `predict_on_pool` run every other spec
before it. They then fill the comparison's targets from the assembled
output columns (`compare_targets`), returning the columns in spec order. A new
model that reads another's output goes through the same two switches, not
a third phase.

**Check:**
`test_portability.TestOutputSchemaStability.test_names_match_the_realized_struct_for_every_model`
compares the declared field names with the output column the bank
actually produces, for every model in its list, times four output combinations
(three when written). `test_coef.test_every_model_lays_out_as_coef_index`
checks each model's `coef` list is as long as its named positions. Add a spec
to `COEF_SPECS` there, which
`test_the_layout_test_covers_every_kind_with_a_coef` holds to every builder
`coef_index` lays out.

**`crates/online-cli` needs nothing, and `crates/online-py` nothing for a
model whose product is its output**: both build from the spec. A model
whose product is read from its state needs a binding, as `marginal`'s
pairs have (`ModelBank.marginal`) and `audit`'s counts (`ModelBank.audit`).
A model whose state says something about the data also goes into
`python/polars_online/_check.py`, as `audit` does (`_audit_findings`), with
each threshold measured on `tests/check_streams.py`'s clean and planted
streams.

## The Python surface — `python/polars_online`

A model reaches Python users through a builder in `po.spec`, whose
signature and output fields the API snapshot pins.

### Step 9 — `_spec_<family>.py`, `_spec.py` and `spec.py`

Write the builder in a `python/polars_online/_spec_<family>.py` of its
family, as `_spec_robust.py` holds `huber` and `quantile` and
`_spec_changepoint.py` holds `bocpd` and `corrchange`, or in a new one.
`_spec.py` is within ten kilobytes of the source cap, so a builder and
its docstring do not fit there. Its parts:

| part | what it is |
|---|---|
| the signature | `@_checked def <name>(name, *, targets, features, <own parameters>, **common: Unpack[CommonKwargs])` |
| the body | it returns `_common(name, {"type": "<name>", ...}, ...)` |
| the docstring | the update equations |
| a clock-unit parameter | annotated `float \| Duration`, which is what makes the builder write a duration as text |
| a parameter where `inf` means something | a `_INF_OK` entry, in `_spec_words.py`. Rust parses those as `Num`, and `validate` says `finite` of every other float |

Then import the builder among the imports at the foot of `_spec.py`, after
everything the builder uses, and add its name to `_spec.py`'s `__all__`.
Then add the same import and name to **`spec.py`**.

**Write the docstring by [docs/WRITING.md](WRITING.md), since tests read
some of it.** Its examples run, and each parameter measured in clock units
says "clock units" on one line of its entry. A model that keeps a weight on
the sum scale says so in words `tests/test_weight_scale.py` pins.

| check | what it holds |
|---|---|
| `test_model_registry::test_every_rust_kind_has_exactly_one_builder` | fails while a kind has no builder; `test_minimal_names_every_builder` then sends you to step 13 |
| `test_temporal_clock::TestEveryClockParameterTakesADuration` | the annotations to `CLOCK_FIELDS`; it fits each clock parameter both ways on a temporal clock, and reads each docstring entry: one that says "clock units", or a name ending `half_life`, must be in `CLOCK_FIELDS` |
| `test_production_hardening`, the API reference's examples | every example in the docstring runs |
| `test_error_messages::test_the_inf_table_matches_the_rust_side` | `_INF_OK` to what Rust's parser and `validate` accept, for each builder in that file's `BUILDERS`, which `test_the_float_sweeps_name_every_builder` holds to `MINIMAL` |
| `test_kwargs_typing::test_each_builder_takes_common_as_the_typed_dict` | `**common` typed `Unpack[CommonKwargs]`; add the name to that file's `BUILDERS`, which `test_model_registry::test_the_builder_list_covers_every_builder` holds to the builders |
| `crates/online-polars/tests/spec_inf.rs` | `validate` to the same verdicts from TOML, where it is the only gate |

### Step 10 — `tests/api_surface.txt`

**Field names are API, and the snapshot is where they are pinned.**

1. Add a `<model> minimal` case to the `[output field grammar]` list in
   `tests/test_api_surface.py`.
2. Regenerate the snapshot:

   ```sh
   UPDATE_API_SURFACE=1 uv run pytest tests/test_api_surface.py
   ```

**Check:** `test_api_surface_matches_the_snapshot` fails on the new
constructor signature.
`test_model_registry::test_the_api_snapshot_pins_every_models_output_fields`
fails until the minimal case is there.

**The model's defaults and words come in by two lists in the same file.**
`[resolved defaults]` renders the new kind from its `MINIMAL` spec without
being asked. A parameter whose default only an option reads, such as
`sgd`'s `huber_delta` under `loss = "huber"`, needs a row in
`RESOLVED_VARIANTS`. A keyword that takes one word of a fixed set needs a
row in `ENUM_PROBES`, or `test_every_word_valued_parameter_is_probed`
fails. A keyword that names a column of the caller's goes in `FREE_TEXT`
instead.

## Tests — `tests/`

The Python tests hold the model through `ModelBank`: its own file checks
the arithmetic, and the shared sweeps check every surface and the schema.

### Step 11 — `tests/test_<model>.py`

Write the Python-side oracle, through `ModelBank`. Use a library that
computes the same quantity where one exists, in
`tests/test_second_opinion.py`. Use a numpy reference in
`tests/reference.py` where the model has a closed form, and a longhand
recursion otherwise. `huber` and `quantile` share `test_robust.py`. The
per-model sweeps of step 12 cover the surfaces and the schema, so this file
is for the arithmetic, which they cannot see.

**Check:**
`test_model_registry::test_every_builder_has_a_per_model_test_file` fails
for a builder with no such file, or for one whose file never calls
`po.spec.<builder>(`.

### Step 12 — Per-model sweeps

Add one entry each to `test_semantics_all_models.MODELS`,
`test_properties.MODELS`, `test_edge_cases.MODELS` and
`test_portability.TestOutputSchemaStability._ALL_MODELS`. Every entry is
`(builder name, the least it needs to be constructible)`.

**The sweeps assert on `pred` and `resid`, so a model with no prediction
sits them out**, through `test_model_registry.REGRESSIONS`. Those models
are `ew_cov`, `kmeans`, `micro`, `ew_class`, `seqtest`, `marginal`, `deco`,
`rcov`, `hmm`, `corrchange`, `bocpd` and `audit`. Each gets its own schema
test instead, such as
`test_portability.TestOutputSchemaStability.test_kmeans_names_match_the_realized_struct`,
`test_micro_names_match_the_realized_struct`,
`test_ew_class_names_match_the_realized_struct` and
`test_seqtest_names_match_the_realized_struct`. It also gets its own
chunk-invariance, save/load, null-row and zero-weight tests in its step-11
file.

**Add a model that reports on *some* rows by design to
`model_contract.rs`'s `SPARSE_OUTPUT`.** A span-based test, for one, writes
its statistic where the span closes. Otherwise the predict-parity helper
will ask it for 300 rows with every slot filled, and fail.

**Add the model to the three lists kept by hand.** `tests/test_every_kind.py`
sweeps every registered model without a list of its own. Three more lists
are kept by hand, and `tests/test_model_registry.py` holds all three:

| list | what it is | what to add |
|---|---|---|
| `scripts/release_probe.py`'s `WORKLOAD` | the specs the weight-scale, released-state and rename checks run | the model. Enter its builder in `tests/test_released_state.py`'s `INTRODUCED` under the next release, the first that has it, so no earlier release is asked to build it |
| `tests/test_budgets_memory.py`'s `WORKLOAD` | every model with every output it has switched on, whose fields' bytes per row are held to `budgets.toml` | the model at its fullest. Run the test with `POLARS_ONLINE_BUDGETS_OUT=<dir>`, then write its budgets with `python scripts/budgets.py apply <dir>/memory.json` |
| `_spec.UNSUPERVISED` | exactly the models whose empty `targets` the bank fills from `features[0]`, which `ModelKind::is_unsupervised` names in `spec.rs` | an unsupervised model, and its name in that function's doc comment, which nothing holds |

**Add a command-line case.** `tests/test_cli_every_kind.py`'s `CASES` runs
every kind at its fullest through the command line, from TOML. It holds
each field, the saved state and the closed groups to
`ModelBank.fit_predict` to the bit. Add a case for the model with every one
of its keys, and split keys that cannot sit together across several cases.
`test_every_kind_and_key_is_covered` reads the kinds and keys from
`tests/api_surface.txt`'s `[toml keys]`. It fails for a kind with no case,
a model key no case uses, or a spec key the kind takes that no case of it
uses.

**Add the model to two lists that have no check.** Nothing holds them to
the registered models. `crates/online-bench/src/lib.rs`'s `MODEL_CASES` is
the instruction-count cases: at least one per model, and one more for each
costlier option, as `huber_k20_sets` is. Add the model's case there,
with its `[instructions.<case>]` entry in `budgets.toml`, and set the budget
from CI's first `measured.json` with `python scripts/budgets.py apply`.
`scripts/benchmark.py`'s rows are the throughput table, [Rows a second for
each model](PERFORMANCE.md#rows-a-second-for-each-model). Add a row for the
model if the table should show it.

**Register the convergence of a model that predicts a target** in
`tests/convergence.py`'s `REGISTRY`, once per value of `loss=` if it takes
one. A registration brings four things:

- a generator of streams with a known truth;
- an oracle: a third-party batch fit (statsmodels, scikit-learn or scipy)
  of the problem the model means to solve, on the same rows at their
  decayed weights;
- the classes that apply to it;
- one sentence saying what the model converges to.

`tests/test_convergence.py` then runs it over a grid of 50 seeds and
asserts on the worst seed, never the median. One online pass can be pulled
away for good on a share of streams while the median looks perfect. The
classes are these:

- consistency without decay, at the rate theory gives;
- the same fit from a wrong start (a loaded state from another regime, an
  extreme prior, a large level);
- wild rows, a level shift or heavy tails in the first rows;
- contamination short of the breakdown point, for a robust model;
- the fit after a break, within a stated number of half-lives of the
  decayed batch fit.

Each class reads its bound from a table in the test, with the measurement
it came from. A model that cannot register yet goes into `PENDING` or
`NOT_REGISTERED`, with the reason.

**Hold a grid that compares the model with a third-party oracle over many
seeds to the oracle ledger** ([TESTING.md, "The oracle
ledger"](TESTING.md#the-oracle-ledger)), so its error only tightens:

1. Mark the test `@pytest.mark.oracle_grid(reason="...")`.
2. Register the group's inputs at module level with
   `ledger.guard(name, outputs, oracles=[...])`.
3. Record each quantity the test bounds with `ledger.record(name, case,
   values, ideal=...)`, one value a seed.
4. Name the group after the source module it tests, as `kmeans_blobs` is
   `kmeans.rs`'s. `scripts/quick.sh` runs the hash guard for the groups
   whose id names a changed module, so a group named otherwise is not run
   when that module changes.
5. Run the grid with `uv run python scripts/oracles.py rerun <group>`. It
   writes the grid's cases to `oracles.toml` and its hash to
   `oracle_hashes.toml`. CI's first run on Linux writes the numbers the
   ledger holds.

A module with a grid that cannot register yet goes into `tests/ledger.py`'s
`PENDING`, with the reason.

**Check:** in `tests/test_oracle_ledger.py`,
`test_every_module_with_a_grid_registers_or_is_pending` fails for a module
with an `oracle_grid` test that neither registers nor is pending.
`test_every_grid_of_a_registered_module_records` fails for a grid of a
registered module that records nothing. The hash guard,
`TestTheHashGuard::test_our_outputs_on_the_cases_inputs_hold`, fails for a
group with no hash or no case in the ledger. It fails on the platform the
hashes were written on, and reports elsewhere. The group's naming has **no
check**.

**Give a clustering or changepoint model's own keywords worst-seed grids**
in `tests/test_family_f_grids.py`. `kmeans`' dead rule, `corrchange`'s
permutation size, `bocpd`'s `robust_beta` and `hmm`'s transition prior have
them. Each relation the docstring states is derived where it can be and
measured over 50 seeds where it cannot, and held to the worst seed. **Give
a model whose state accumulates over the whole stream a long-run case** in
`tests/test_long_run.py`'s `CASES`, which `tests/test_long_run_soak.py`
runs a hundred times longer, at 1e8 rows, weekly. Neither file has a list
held to the registered models: **no check**.

**Add a regression to `python/polars_online/_check.py`'s `_REGRESSIONS`**,
which `ModelBank.check` reads for `few_rows`. If the model does not center
its features, measure the level and the spread ratio at which its
out-of-sample R² falls by 0.05, as the docstring of `check` describes. Add
them to `_UNCENTRED_LIMITS`.

**Check:** `test_convergence::test_every_regression_registers_its_convergence`
reads the models from `test_model_registry.REGRESSIONS` and the losses from
each builder's signature. `test_model_registry::test_the_sweeps_cover_every_regression_model`,
`test_the_release_workload_builds_every_model`,
`test_the_memory_budgets_hold_every_model`,
`test_unsupervised_is_the_models_the_bank_fills_a_target_for` and
`test_check::test_check_places_every_regression` hold the lists.

### Step 13 — `tests/test_model_registry.py`

Add the model's `MINIMAL` entry. `tests/test_model_registry.py` holds every
per-model test list to `KINDS`.

**Check:** `test_minimal_names_every_builder`.

### Step 14 — `tests/test_golden_pipeline.py`

Pin the whole pipeline's numbers for the model:

1. Add a spec for it in `specs()`.
2. Print the pinned values:

   ```sh
   uv run python tests/test_golden_pipeline.py
   ```

3. Copy only the new model's lines into `GOLDEN`.

**If any other line moved, find out why before you copy it.**

**Check:** `test_model_registry::test_the_golden_pipeline_pins_every_model`.

## Docs

A model is documented in its entry in MODELS.md, its line in the feature
index, its fields in the generated OUTPUTS.md and the model count in
`llms.txt`, and tests hold each of them to the registry.

### Step 15 — `docs/MODELS.md` and `docs/FEATURES.md`

Write a heading `` ### `<name>` — ... `` under its family's `##` heading
in `docs/MODELS.md`, with the equations, the parameters and when to reach
for it. The four families are linear models, moments and correlation,
clustering and classification, and sequential tests and regimes. Add a row
under the same family in MODELS.md's table *The models at a glance*, with
its `entry` link to the new heading. Add a line for its builder in
`docs/FEATURES.md`'s *Models*, linking its entry and its how-tos.

**Check:**
`test_model_registry::test_the_models_page_documents_every_model`. And
`test_api_links`: `test_the_model_table_links_every_builder_to_its_page`
wants the table row to link ``
[`<name>`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.<name>)
``. `test_every_model_section_links_its_builder_and_its_source` wants the
heading's section to carry an `*API:*` link to the builder and a `*Rust:*`
link to `crates/online-core/src/<file>.rs`, which must exist.
`test_feature_index::test_every_public_name_has_a_line_in_the_index` fails
until `docs/FEATURES.md` has the builder's line.

**Regenerate `docs/OUTPUTS.md`, which is generated:**

1. In `scripts/outputs_doc.py`, place the model in its family in
   `FAMILIES`.
2. Give each new field stem a `MEANING` line, and a `STATE_ONLY` note if
   the model writes only `weight_sum`.
3. Run `uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`.

**Check:** `test_outputs_doc`'s `test_the_document_is_what_the_generator_writes`,
`test_every_model_has_a_section` and `test_no_field_is_left_undocumented`.

**Bump the model count wherever it is spelled out.** Grep for
`twenty-two`, `22 model` and `all 22`; on 2026-11-08 the count stood in
these places:

| file | where | check |
|---|---|---|
| `llms.txt` | the *Models* line, "all 22" | `test_llms_txt::test_it_says_how_many_models_there_are` |
| `README.md` | the models section, "twenty-two models in four families" | **no check** |
| `docs/FEATURES.md` | the contents table's *Models* row | **no check** |
| `docs/MODELS.md` | the first sentence, and the opener of *The models at a glance* | **no check** |
| `docs/VERSIONS.md` | the row for the spec constructors | **no check** |
| `docs/TESTING.md` | the *Terms* table's `a model kind` row, "one of the 22 model types" | **no check** |
| `docs/TESTING.md` | the `instructions` budget row, "the 22 models" | **no check** |
| `docs/WRITING.md` | *Plan the map*, on the families | **no check** |
| `docs/PERFORMANCE.md` | *The table*'s count, "76 over the 22 model kinds' rows", with the cells step 17 adds | **no check** |
| `docs/how-to/choose-a-model.md` | the first sentence | **no check** |

**Add the model to `docs/how-to/choose-a-model.md`** too, where a reader
picks a model by their task. **No check.**

`test_llms_txt::test_every_builder_is_its_spec_type` fails when a new
builder's name differs from its kind.

### Step 16 — `CHANGELOG.md` and the design note

Write a **`CHANGELOG.md`** entry, and a design note: the update equations,
the alternatives weighed and why they lost, and the measurements behind
the defaults. The note is kept with the project's records, which
[docs/README.md](README.md#records) lists.

**No check.**

### Step 17 — `crates/online-bench/src/scaling.rs`

Add the model's cells to the scaling table:

- `<kind>.history`, which runs the model after 1,000 rows and after 8,000,
  and must cost the same and keep the same state;
- one cell for each size the model has (its features, its targets, its
  clusters, a window, a held buffer), with the growth its algorithm gives
  a row's work and its state.

Give each cell its case in `model_case`, and its row, with the reason, in
docs/PERFORMANCE.md's [table](PERFORMANCE.md#how-cost-grows-with-size). A
cell that fails is a growth the model was not meant to have: find it
before the model ships.

**Check:** `every_kind_has_its_cells` fails for a kind in `KINDS` with no
cell, and each cell fails past its declared growth.

### Then the test gate

Check the result by running the test gate, unpiped. The registry tests fail
until every step is in, which is what they are for.

1. Run the essentials, and commit once they pass:

   ```sh
   ./scripts/gate.sh              # unpiped; the essentials, for each commit
   ```

2. Before the push, run every test ([TESTING.md, "Two
   tiers"](TESTING.md#two-tiers)):

   ```sh
   ./scripts/gate.sh --extended   # unpiped; every test, before the push
   ```

While you work, `scripts/quick.sh` runs the tests for what the branch
changed, as [Iterate on a change](contributing/iterate-on-a-change.md)
describes.
