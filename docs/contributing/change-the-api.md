<!-- kind: how-to -->
# Change the public API

This how-to is for a contributor who adds, renames or removes a public
name of polars-online, or changes a default, and must see the change as a
diff and ship it in the right kind of release.

**A change to the public API can break a user with no error at all.**
A changed default silently changes users' numbers,
which is worse than breaking their code. A renamed output field fails as a
`StructFieldNotFound` in someone else's pipeline. So one snapshot test,
`tests/test_api_surface.py`, renders the entire public surface to text and
compares it with a checked-in file, `tests/api_surface.txt`. Every change
to it is a diff someone has to accept. A rename shows up as `- pred_y__r1e-06 / + pred_y__r0.000001` in the pull
request, where a reviewer decides whether it is intended, and which
release it needs. Accidental changes fail, and deliberate ones are
visible. `tests/test_golden_pipeline.py` does the same for the numbers.

## Prerequisites

A change to the API needs these:

- A development build of the package, as [Getting set
  up](../../CONTRIBUTING.md#getting-set-up) describes, with the Rust
  toolchain: the snapshot test builds the command line with `cargo` to
  read its flags.
- [What is stable](../VERSIONS.md#what-is-stable), which says which parts
  are API, and [What each kind of release may
  change](../VERSIONS.md#what-each-kind-of-release-may-change), which says
  the release each change needs.

## Steps

The snapshot test turns the change into a diff, and the diff decides the
release.

1. **Make the change** in the code.
2. **Run the snapshot test,** with
   `uv run pytest tests/test_api_surface.py`. It fails on any change to
   the surface, with a unified diff and a message that says a changed
   field name or default needs a version bump.
3. **Read the diff against [What each kind of release may
   change](../VERSIONS.md#what-each-kind-of-release-may-change).** A
   changed default or meaning, or a removed stable name, needs a major
   release. A new name needs a minor one. A renamed stable name needs a
   forwarding ([Rename a stable name](#rename-a-stable-name)). [What the
   API snapshot pins](../RELEASE-READINESS.md#what-the-api-snapshot-pins)
   says which section of the snapshot holds each part.
4. **Regenerate the snapshot,** with
   `UPDATE_API_SURFACE=1 uv run pytest tests/test_api_surface.py`.
5. **Add or rename the name's line in [FEATURES.md](../FEATURES.md),**
   in backticks in the row's first cell, with a link to the page that
   shows it. `tests/test_feature_index.py` fails on a public name the
   snapshot holds that the feature index does not list.
6. **Declare the change in the CHANGELOG**, under `[Unreleased]`, with
   what moved and by how much.
7. **If a default changes in a minor release, record the departure.**
   Give the setting that restores the old default, say *against the
   policy* in the CHANGELOG entry with the setting as `` `name=value` ``,
   and add a row and a paragraph to [Defaults changed in a minor
   release](../VERSIONS.md#defaults-changed-in-a-minor-release).
   `test_a_change_against_the_policy_is_on_the_policy_page`
   (`tests/test_released_state.py`) holds the CHANGELOG entry to that
   section.
8. **Commit the new `tests/api_surface.txt` with the change,** so the
   diff is in the pull request beside the code that made it.

## Check the result

The change is ready when `uv run pytest tests/test_api_surface.py` passes
on the regenerated snapshot, its diff is in the pull request, and the
CHANGELOG entry names the release kind the diff needs.

## Rename a stable name

**A renamed stable name keeps working until the next major, with a
`PolarsOnlineDeprecationWarning` that names the new spelling.**
`polars_online.PolarsOnlineDeprecationWarning` is a `DeprecationWarning`.

1. **For a spec parameter, or a word a spec parameter takes, add the old
   name to the forwarding tables, on both sides.** Each Python table in
   `python/polars_online/_warnings.py` has a Rust twin in
   `crates/online-polars/src/spec.rs` and `spec/names.rs`, and
   `tests/test_deprecation.py` holds each pair equal:

   | the rename | Python | Rust |
   |---|---|---|
   | a key, in every model | `_DEPRECATED` | `DEPRECATED` |
   | a key, in one model | `_DEPRECATED_IN` | `DEPRECATED_IN` |
   | a word of one model's parameter | `_DEPRECATED_WORDS_IN` | `DEPRECATED_WORDS_IN` |

   The Python tables read a spec builder's keyword and a spec dict's key,
   `with_windows`' `like=` spec included, and the Rust tables a TOML key,
   at any depth of a spec.
2. **For any other name, ship a forwarding of its own.** A keyword of
   another function, a word another function's parameter takes, a
   function, a command-line flag, an environment variable, an output field
   or a frame column does not go through the tables. Use
   `forward_deprecated` at the top of the function whose keyword moved, a
   stub that warns and calls the new function, an alias for the word, flag
   or variable, or a second column beside the new one. Each warns with the
   same `PolarsOnlineDeprecationWarning`.
3. **Ship the rename in a minor release,** and remove the old name at the
   next major.

**The renames of 0.x up to 2026-10-07 stay refused,** by `_RENAMED` in
`python/polars_online/_spec.py` and `RENAMED` in
`crates/online-polars/src/spec.rs`. Those of 2026-10-11, the names 1.0
keeps, are forwarded by `_DEPRECATED_IN` and `_DEPRECATED_WORDS_IN` until
2.0. `renamed_function` in `python/polars_online/_renamed.py` is the
refusing stub a pre-1.0 rename keeps, not a forwarding one.

## Add a spec parameter

**A new parameter needs an entry for the released-state test only when
the release workload uses it.** `tests/test_released_state.py` fits the
workload, `WORKLOAD` in `scripts/release_probe.py`, under every release
on PyPI, and an older release refuses a keyword it does not take.

1. **If you add a spec that uses the new parameter to `WORKLOAD`, enter
   that spec's name in `PARAMETER_INTRODUCED`** (`tests/test_released_state.py`),
   under the next minor release. An older release is then not asked to
   build it, and the test stops excusing the release as soon as that
   release is on PyPI.
2. **For a new model kind, enter it in `INTRODUCED`** in the same file,
   the same way.
3. **Leave the entries when you release.**
   `scripts/release_version.py --publish` refuses to publish a version
   older than one named there, since the test would then expect that
   release to build a spec it cannot.
