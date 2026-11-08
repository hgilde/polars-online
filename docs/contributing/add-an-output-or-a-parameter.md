<!-- kind: how-to -->
# Add an output field or a spec parameter

This how-to is for a contributor adding an output field, a spec parameter,
a kind of input column or a withheld reason to polars-online, rather than a
whole model. It says where each goes, what pins it, and what each changes
in a saved state.

**Every spec parameter rides in every bank file.** A bank file carries its
specs, so a new spec field changes the bytes of every bank file saved
after it. Whether that needs a new schema is the last thing to settle,
under [What each change does to a saved
state](#what-each-change-does-to-a-saved-state). To add a model kind, see
[Add a model](../EXTENDING.md), whose steps this page refers to by number.

## Prerequisites

You need these before the first step:

- A checkout that builds and passes the gate, as
  [CONTRIBUTING.md](../../CONTRIBUTING.md) sets up.
- The library's words: spec, bank, state, the output struct and its
  fields, as the [glossary](../GLOSSARY.md) defines them.
- `CLAUDE.md`'s hard rule 5, on state files and their frozen fixtures.

## Add what you need

Find the row for what you add, put it in each place the row names, and
read the check that pins it:

| what you add | where it goes | what pins it |
|---|---|---|
| **a new *common* parameter**, one every model takes, like `embargo` | the field on `Spec` in `crates/online-polars/src/spec.rs` with `#[serde(default)]`, and its validation in `Spec::validate`; `CommonKwargs` in `python/polars_online/_kwargs.py`; `_common`'s signature *and* the dict it builds, in `python/polars_online/_spec.py` | the API snapshot (`tests/api_surface.txt`) records the new keyword and its default, and regenerating it is the diff to read |
| **a new output field** on every model | `FieldMeta` in `crates/online-polars/src/bank.rs` carries the name and dtype; the emit flag goes on `Spec`, in `CommonKwargs` in `python/polars_online/_kwargs.py`, and in `_common`'s signature and the dict it builds | `test_portability::test_exact_field_names_for_a_grid_spec`, plus the API snapshot |
| **a new clock-policy parameter** | the `ClockPolicy` in `crates/online-polars/src/spec/clock.rs`, and `with_windows`' signatures beside the spec's | the rename tests (`tests/test_renames.py`), and the API snapshot |
| **a new kind of input column**, beyond features, targets and the weight | `DataSummary::layout` and `feed_row` in `crates/online-polars/src/summary.rs` decide which columns `describe()` lists, and in what order; `Bank::describe`'s `keep` decides which get moments | `crates/online-polars/tests/summary.rs` pins the frame's column names, and compares every statistic to an oracle computed over the frame, so a column the summary does not know is a failing count there |
| **a new `withheld_reason`**, a gate that withholds a row's predictions | its name at the end of `WITHHELD_REASONS` in `crates/online-polars/src/stream.rs`, and its `REASON_*` code after the last. Then its four spelled-out lists: `output_fields`' docstring in `python/polars_online/spec.py`, the `withheld_reason` line in `scripts/outputs_doc.py` (regenerate `docs/OUTPUTS.md`), the README's readiness table and `docs/DIAGNOSTICS.md`'s gate table | `test_portability`'s `REASON_DTYPE`, the dtype every model's field is held to, and `test_readiness`'s `REASONS`, against which each row's reason is read; the state fixtures, whose last rows keep the codes; the four lists in the docs have **no check** |
| **a new parameter** on one model | the `Cfg` field and its validation in `new` (step 1), the `ModelKind` field with `#[serde(default)]` (step 6), the `build_one` default (step 7), the builder keyword (step 9), the snapshot (step 10); if `inf` means something for it, `_INF_OK` (step 9). A spec of `scripts/release_probe.py`'s `WORKLOAD` that sets it goes into `tests/test_released_state.py`'s `PARAMETER_INTRODUCED` under the next release, the first that will take the keyword | the inf-table test catches the Python half; the compiler catches the Rust half; a workload spec left out of `PARAMETER_INTRODUCED` fails the released-state test, where every release refuses the keyword, and one entered under an older version than the release being published stops `scripts/release_version.py --publish` |

**Append a withheld reason, never insert one.** The list is append-only,
because a state's last row keeps the reason's code. Its order is the
precedence when two gates withhold one row.

**Rename a parameter through the deprecation tables.** A parameter renamed
after 1.0 goes into `DEPRECATED` in `crates/online-polars/src/spec.rs`,
beside its twin in `python/polars_online/_warnings.py`'s `_DEPRECATED`. A
spec that names the old name is read as naming the new one, with a
deprecation notice, until the next major version. A name renamed before
1.0 is refused, naming the new one: through `RENAMED` in `spec.rs`, which
`name_renamed` reports, and through `_RENAMED` in `_spec.py`.

**If a new output field gets its own `ChunkOut` buffer**, add it to
`LastRow::take` and `LastRow::to_chunk` in
`crates/online-polars/src/stream/last_row.rs`. Switch it on in the rich
spec of `crates/online-polars/tests/last_row.rs`. That test compares the
saved last row against the frame field for field, which is what catches
the omission.

**If the Python builders write a default the field would not otherwise
have, add it to `Spec::fill_defaults` too.** Otherwise a TOML spec and the
same spec in Python save different bytes. `tests/test_no_output.py`
compares the two state files.

## What each change does to a saved state

**A spec field changes the bytes of every bank file**, since each carries
its specs. Whether that is a new schema depends on whether a release has
shipped the schema the field joins:

| the change | what it needs |
|---|---|
| a field with `#[serde(default)]`, added inside a schema no release has shipped | no bump: it joins that schema |
| any other change to a layout a state file holds | a new schema, under hard rule 5: the steps below |
| a new model's `ModelState` variant, appended | no bump; a fixture case and its frozen name, as [Add a model](../EXTENDING.md#step-3--srcmodelrs)'s step 3 says |

**A layout change cannot land without its fixtures.** Every schema from
`MIN_SCHEMA_VERSION` and `MIN_BANK_SCHEMA_VERSION` to `SCHEMA_VERSION` has
a frozen set of state fixtures, in `crates/online-core/tests/state_fixtures/`
and `crates/online-polars/tests/state_fixtures/`. Each crate's `build.rs`
finds every `<case>.rs` there, and every one under each `v<N>/`. Since 1.0,
the two minimums stay at 1.0's schema, 61, so every state saved since then
keeps loading. Take a layout change from schema `N` to `N + 1` in these
steps, in each crate whose state moved:

1. Copy the current set, every `<case>.rs`, as it is to
   `tests/state_fixtures/v<N>/`. The directory is the kept set.
2. Set `SCHEMA_VERSION` (`crates/online-core/src/lib.rs`) to `N + 1`,
   and write the loader the change needs in the model's `Deserialize`, as
   `check_schema`'s doc describes.
3. Regenerate the current set, which is then schema `N + 1`'s:

   ```sh
   PRINT_STATE_FIXTURES=1 cargo test -p online-core --test state_fixtures
   PRINT_STATE_FIXTURES=1 cargo test -p online-polars --test state_fixtures
   ```

4. If the change adds a form a layout can take, add a case to
   `crates/online-core/tests/cases/mod.rs` (or the bank's cases in
   `crates/online-polars/tests/state_fixtures.rs`) that writes it, and
   regenerate again.

**A change to the windows state's version moves the bank's schema with
it.** `a_windows_state_version_moves_the_banks_schema_with_it`, in
`crates/online-polars/src/windows_frame.rs`, pairs `WINDOWS_VERSION` with
`SCHEMA_VERSION`, and fails until both are moved.

## Check the result

Run the state-fixture tests of both crates, then the gate. Each holds a
different part of the change:

| the check | what it fails on |
|---|---|
| `the_fixtures_cover_the_schemas_this_build_loads` | a schema between the minimum and `SCHEMA_VERSION` with no set |
| `every_previous_fixture_loads_goes_on_and_converts_to_the_current_bytes` | a kept fixture that does not load through the loader, go on to the bit against its own frozen continuation, and save as the current schema's fixture bytes of the same case |
| `every_layout_a_schema_moved_is_written_by_a_fixture` (`online-core`), `every_form_a_schema_moved_is_written_by_a_fixture` (`online-polars`) | a new form of a layout that no fixture writes |
| `test_api_surface_matches_the_snapshot` | a new keyword, default or field not in the regenerated snapshot |

```sh
cargo test -p online-core --test state_fixtures
cargo test -p online-polars --test state_fixtures
./scripts/gate.sh              # unpiped; the essentials, for each commit
./scripts/gate.sh --extended   # unpiped; every test, before the push
```
