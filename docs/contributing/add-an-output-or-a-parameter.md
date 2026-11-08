<!-- kind: how-to -->
# Add an output field or a spec parameter

This how-to is for a contributor adding an output field, a spec parameter,
a kind of input column or a withheld reason to polars-online, without a new
model. It says where each goes, what pins it, and what each changes in a
saved state. To add a model kind, see [Add a model](../EXTENDING.md), whose
steps this page refers to by number.

## Prerequisites

You need these before the first step:

- A checkout that builds and passes the gate, as
  [CONTRIBUTING.md](../../CONTRIBUTING.md) sets up.
- The library's words, as the [glossary](../GLOSSARY.md) defines them: a
  spec, a [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  (then *bank*), its state, and a spec's [output
  column](../GLOSSARY.md#what-a-bank-writes) and its fields.
- `CLAUDE.md`'s hard rule 5, on state files and their frozen fixtures.

## Find where the addition goes

Each kind of addition touches its own set of files. Follow these steps for
any of them:

1. **Find the row for what you add** in the table of additions.
2. **Add it in each place the row's second column names.**
3. **Rebuild the extension**, since the snapshot is read from the
   installed build:

   ```sh
   uv run maturin develop --release -m crates/online-py/Cargo.toml
   ```

4. **Regenerate the API snapshot**, which records every keyword, its
   default and every field name:

   ```sh
   UPDATE_API_SURFACE=1 uv run pytest tests/test_api_surface.py
   ```

   Read the snapshot's diff: it should show only what you added.
5. **Run the tests the row's third column names.**
6. **Settle the saved state**, under [What each change does to a saved
   state](#what-each-change-does-to-a-saved-state).

| what you add | where it goes | what pins it |
|---|---|---|
| **a new *common* parameter**, one every model takes, like `embargo` | the field on `Spec` in `crates/online-polars/src/spec.rs` with `#[serde(default)]`, and its validation in `Spec::validate`; `CommonKwargs` in `python/polars_online/_kwargs.py`; `_common`'s signature *and* the dict it builds, in `python/polars_online/_spec.py` | the API snapshot (`tests/api_surface.txt`) records the new keyword and its default |
| **a new output field** on every model | first make room in `crates/online-polars/src/stream.rs`, as the `withheld_reason` row says. Then `FieldMeta` in `crates/online-polars/src/bank/fields.rs` carries the name and dtype; the emit flag goes on `Spec`, in `CommonKwargs` in `python/polars_online/_kwargs.py`, and in `_common`'s signature and the dict it builds. The value is written by the stream into a buffer of `ChunkOut` (`crates/online-polars/src/stream/chunk_out.rs`), as `out.scored_clock` is in `stream.rs`; a field with a buffer of its own needs that buffer, its `Source` variant in `fields.rs`, and its read in `assemble` (`crates/online-polars/src/bank/assembly.rs`) | `test_portability`'s `test_names_match_the_realized_struct_for_every_model`, once the flag is in one of its `extra` output combinations, which compares the declared fields with the produced ones for every model; the snapshot's `[output field grammar]` |
| **a new clock-policy parameter** | everything the common-parameter row names; a field on `ClockPolicy` and its read in `clock_cfg_of`, in `crates/online-polars/src/spec/clock.rs`, and on `ClockCfg` (`crates/online-core/src/clock.rs`) if the clock itself acts on it; a field on `WindowsConfig` in `crates/online-polars/src/windows_frame.rs`; both places that build a `ClockPolicy`, `Spec`'s in `spec.rs` and `WindowsRun`'s in `windows_frame.rs`; and `with_windows`' signatures beside the spec's | the API snapshot, which lists `with_windows`' signatures; nothing else lists the clock-policy keywords: **no check** |
| **a new kind of input column**, beyond features, targets and the weight | `DataSummary::layout` and `feed_row` in `crates/online-polars/src/summary.rs` decide which columns `describe()` lists, and in what order; `Bank::describe`'s `keep` decides which get moments | `crates/online-polars/tests/summary.rs` pins the frame's column names, and compares every statistic to an oracle computed over the frame, so a column the summary does not know is a failing count there |
| **a new `withheld_reason`**, a gate that withholds a row's predictions | first make room in `crates/online-polars/src/stream.rs`, which stands within a few hundred bytes of the source cap: move a block of it into a submodule, as [Add a model](../EXTENDING.md#step-7--srcstreamrs)'s step 7 says. Then its name at the end of `WITHHELD_REASONS` in `stream.rs`, with the array's length one more, and its `REASON_*` code after the last. Then its five spelled-out lists: the count "`withheld_reason`'s six" in `scripts/outputs_doc.py`'s description of the enum fields, the `withheld_reason` entry of `python/polars_online/spec.py`'s module docstring, the `withheld_reason` description in `scripts/outputs_doc.py` that lists the reasons (regenerate `docs/OUTPUTS.md`), the `withheld_reason` row of the readiness-fields table under `docs/MODELS.md`'s *Warm-up gates*, and `docs/DIAGNOSTICS.md`'s gate table | `test_portability`'s `REASON_DTYPE`, the dtype every model's field is held to, and `test_readiness`'s `REASONS`, against which each row's reason is read; the state fixtures, whose last rows keep the codes; the five lists in the docs have **no check** |
| **a new parameter** on one model | the `Cfg` field and its validation in `new` (step 1), the `ModelKind` field with `#[serde(default)]` (step 6), the `build_one` default (step 7), the builder keyword (step 9), the snapshot (step 10); if `inf` means something for it, `_INF_OK` (step 9). A spec of `scripts/release_probe.py`'s `WORKLOAD` that sets it goes into `tests/test_released_state.py`'s `PARAMETER_INTRODUCED` under the next minor release, the first that will take the keyword | the inf-table test catches the Python half; the compiler catches the Rust half; a workload spec left out of `PARAMETER_INTRODUCED` fails the released-state test, where every release refuses the keyword; an entry under a version newer than the release being published stops `scripts/release_version.py --publish` |

**Append a withheld reason at the end of the list.** The list is
append-only, because a state's last row keeps the reason's code. Its order
is the precedence when two gates withhold one row.

**Rename a parameter through the deprecation tables.** After 1.0, a spec
that names an old name is read as naming the new one, with a deprecation
notice, until the next major version refuses it. Each table has a twin on
each side, and `tests/test_deprecation.py` holds the twins equal:

| the rename | Rust | Python |
|---|---|---|
| a key every spec takes | `DEPRECATED` in `crates/online-polars/src/spec.rs` | `_DEPRECATED` in `python/polars_online/_warnings.py` |
| a key of one model | `DEPRECATED_IN` in `crates/online-polars/src/spec/names.rs` | `_DEPRECATED_IN` in `_warnings.py` |
| a word of one model's parameter | `DEPRECATED_WORDS_IN` in `spec/names.rs` | `_DEPRECATED_WORDS_IN` in `_warnings.py` |

A model's key has a table of its own because one old name went two ways
(`kmeans`' `k` is `n_clusters`, `hmm`'s is `n_components`). It also keeps a
name another model still takes from moving. A name renamed before 1.0 is
refused, naming the new one: through `RENAMED` in `spec.rs`, which
`name_renamed` reports, and `_RENAMED` in `_spec.py`. A model's own refused
name is in `RENAMED_WHERE_EXPECTED` and `_RENAMED_IN`.
`tests/test_renames.py` holds the refusals.

**If a new output field gets its own `ChunkOut` buffer**, add it to
`LastRow::take` and `LastRow::to_chunk` in
`crates/online-polars/src/stream/last_row.rs`. Switch it on in the rich
spec of `crates/online-polars/tests/last_row.rs`. That test compares the
saved last row against the frame field for field, which is what catches
the omission.

**If the Python builders write a default the field would not otherwise
have, add it to `Spec::fill_defaults` too.** Otherwise a TOML spec and the
same spec in Python save different bytes.
`tests/test_no_output.py`'s `test_the_filled_spec_is_the_one_python_writes`
compares the two state files, for the specs it builds. Give it a spec that
leaves the new default out, so it compares the default's two fillings.

## What each change does to a saved state

**A spec field changes the bytes of every bank file**, since each carries
its specs. Whether that is a new schema depends on whether a release has
shipped the schema the field joins:

| the change | what it needs |
|---|---|
| a field with `#[serde(default)]`, added inside a schema no release has shipped | no bump: it joins that schema. The schema has shipped if the last release's `SCHEMA_VERSION` equals the current one: `git show v<last>:crates/online-core/src/lib.rs \| grep '^pub const SCHEMA_VERSION'` |
| any other change to a layout a state file holds | a new schema, under hard rule 5, by the numbered steps that follow this table |
| a new model's `ModelState` variant, appended | no bump; a fixture case and its frozen name, as [Add a model](../EXTENDING.md#step-3--srcmodelrs)'s step 3 says |

**A layout change cannot land without its fixtures.** Every schema from
`MIN_SCHEMA_VERSION` and `MIN_BANK_SCHEMA_VERSION` to `SCHEMA_VERSION` has
a frozen set of state fixtures, in `crates/online-core/tests/state_fixtures/`
and `crates/online-polars/tests/state_fixtures/`. Each crate's `build.rs`
finds every `<case>.rs` there, and every one under each `v<N>/`. Since 1.0,
the two minimums stay at 1.0's schema, 61, so every state saved since then
keeps loading. `SCHEMA_VERSION` is one number for both crates, and each
crate demands a set for every schema up to it. So take a layout change
from schema `N` to `N + 1` in these steps, in both crates:

1. Copy the current set, every `<case>.rs`, as it is to
   `tests/state_fixtures/v<N>/`. The directory is the kept set.
2. Set `SCHEMA_VERSION` (`crates/online-core/src/lib.rs`) to `N + 1`.
3. Write the loader the change needs, where the moved layout is read. A
   model's state is read in the model's `Deserialize`, as `check_schema`'s
   doc describes. A bank file is read by `Bank::load_bytes`, in
   `crates/online-polars/src/bank.rs`. A `with_windows` state is read by
   `WindowsRun::load_bytes` (`src/windows_frame.rs`), and a `refresh_time`
   state by `RefreshTime::load_bytes` (`src/refresh.rs`); the bank
   crate's fixtures freeze both.
4. Regenerate the current set, which is then schema `N + 1`'s:

   ```sh
   PRINT_STATE_FIXTURES=1 cargo test -p online-core --test state_fixtures
   PRINT_STATE_FIXTURES=1 cargo test -p online-polars --test state_fixtures
   ```

5. If the change adds a form a layout can take, add a case that writes it,
   and regenerate again. The model cases are in
   `crates/online-core/tests/cases/mod.rs`, and the bank's specs in
   `crates/online-polars/tests/state_fixtures.rs`.

**A change to the windows state's version moves the bank's schema with
it.** `a_windows_state_version_moves_the_banks_schema_with_it`, in
`crates/online-polars/src/windows_frame.rs`, pairs `WINDOWS_VERSION` with
`SCHEMA_VERSION`, and fails until both are moved.

## Check the result

Run the state-fixture tests of both crates, then the gate. Each holds a
different part of the change:

| the check | what it fails on |
|---|---|
| `the_fixtures_cover_the_schemas_this_build_loads` (`online-core`), `the_bank_fixtures_cover_the_schemas_a_bank_loads` (`online-polars`) | a schema between the minimum and `SCHEMA_VERSION` with no set |
| `every_previous_fixture_loads_goes_on_and_converts_to_the_current_bytes` | a kept fixture that does not load through the loader, go on to the bit against its own frozen continuation, and save as the current schema's fixture bytes of the same case |
| `every_layout_a_schema_moved_is_written_by_a_fixture` (`online-core`), `every_form_a_schema_moved_is_written_by_a_fixture` (`online-polars`) | a new form of a layout that no fixture writes |
| `test_api_surface_matches_the_snapshot` | a new keyword, default or field not in the regenerated snapshot |

```sh
cargo test -p online-core --test state_fixtures
cargo test -p online-polars --test state_fixtures
./scripts/gate.sh              # unpiped; the essentials, for each commit
./scripts/gate.sh --extended   # unpiped; every test, before the push
```
