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
to it is a diff someone has to accept:

```
tests/test_api_surface.py   ->  compares against  tests/api_surface.txt
```

A rename shows up as `- pred_y__r1e-06 / + pred_y__r0.000001` in the pull
request, where a reviewer decides whether it is intended, and which
release it needs. Accidental changes fail, and deliberate ones are
visible. `tests/test_golden_pipeline.py` does the same for the numbers.

## Prerequisites

A change to the API needs these:

- A development build of the package, as `CONTRIBUTING.md` describes.
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
   forwarding ([Rename a stable name](#rename-a-stable-name)).
4. **Regenerate the snapshot,** with
   `UPDATE_API_SURFACE=1 uv run pytest tests/test_api_surface.py`.
5. **Declare the change in the CHANGELOG**, under `[Unreleased]`, with
   what moved and by how much.
6. **Commit the new `tests/api_surface.txt` with the change,** so the
   diff is in the pull request beside the code that made it.

## What each section of the snapshot pins

**Each stable part is pinned by a section of `tests/api_surface.txt`,
except the command line's exit status, which a test holds.** The unstable
names that stand in a section, `po.sim`, `po.corr`, `po.gbdt`,
`ArrowStruct`, `fit_predict_arrow` and `predict_arrow`, each carry the
label `# unstable` on their line, so the snapshot says which of its lines
carry no promise:

| stable part | pinned by |
|---|---|
| `__all__`, each function with its signature | `[package]` |
| the constructors' keyword names and the defaults they set in Python | `[common parameters]`, which lists once the keywords every spec shares (`CommonKwargs` in `python/polars_online/_kwargs.py`), and `[spec constructors]` |
| the defaults that resolve in Rust, `min_weight`'s per-kind rule included | `[resolved defaults]`, read off the bank's own build of each kind (`_polars_online.resolved_defaults`); `tests/test_spec_defaults.py` holds the README's warm-up tables to it |
| `ModelBank`'s methods, `__init__` included | `[ModelBank]` |
| the `.online` namespace | `[frame namespaces]` |
| the frames the bank returns: `groups()`, `summary()`, `describe()`, `coef()`, `last_row()`, `marginal()`, `audit()`, `check()`, `solve_failures()`, `gram()`'s keys, and `po.spec`'s `output_index()`, `coef_index()`, `coef_fields()` | `[frame columns]`, each column with its dtype |
| the closed-groups frame's columns | `[closed_groups columns]`, each column with its dtype, in order |
| the helper modules' functions, each with its signature | `[helper modules]`, every module the package's directory holds: today `corr`, `eval`, `gbdt`, `gram`, `ops`, `sim` and `stream` |
| `po.stream.embargo`'s two columns, `_online_role` and `_online_role_weight` | `[helper modules]`, as the default of `embargo`'s `role`, the first column's name, which the second's extends; `tests/test_label_delay.py` reads both |
| output field names and their dtypes | `[output field grammar]`, 29 spec shapes, one with every `emit_*` turned on |
| the words each string-valued parameter takes, and `withheld_reason`'s | `[enum values]`, read from each parameter's refusal of a word it does not take |
| the TOML config keys: the run config's, a spec's, and each model type's | `[toml keys]`, read from serde's refusal of a key a table has not got |
| the command line's flags, and whether each takes a value | `[cli flags]`, from `polars-online --help` |
| the command line's exit status: 0, 1 for a refusal or a run error, 2 for a usage error | `docs/RUNNER.md`'s *Exit status* and a test of the command line, not the snapshot |
| the environment variables, `POLARS_ONLINE_MAX_THREADS` and `POLARS_ONLINE_TIMING` among them | `[env vars]`, from every read in `crates/*/src` and the package |

**A default that resolves in Rust is pinned where the bank builds it.**
A builder passes `None` for it, so a constructor's signature shows `None`.
The snapshot's `[resolved defaults]` reads each off the bank's own build
of every kind's least spec: `Spec::check`, then the constructor of every
group's stream, through `_polars_online.resolved_defaults`. It renders one
line per kind and field. The lines cover the model's configuration, what
the model derives from a field left unset (`bocpd`'s priors, `rcov`'s ring
and window, `marginal`'s bin budget), the stream's gates and diagnostics'
settings, and the clock policy. Variants add the defaults that only an
option reads, such as `sgd`'s `huber_delta` under `loss = "huber"`. Other
tests hold some of these defaults to what they are for:

| default | set in | pinned by |
|---|---|---|
| every default a builder leaves to Rust, `min_weight`'s per-kind rule included | Rust | the snapshot's `[resolved defaults]`; `tests/test_spec_defaults.py` holds the README's warm-up tables to it |
| the solve cadence of `ewridge`, `lasso`, `huber` and `quantile`: by weight, `half_life/50` of clock in steady state | Rust | `[resolved defaults]` (`solve_every`, `solve_share`); `tests/test_solve_cadence.py` measures it |
| the clock settings, the readiness gates, and `min_weight`'s count floor | Rust | `[resolved defaults]`; `crates/online-polars/tests/spec_defaults.rs` |
| `standardize`: true for `kalman`, false for `ewridge`, `huber`, `quantile` and `sgd`; `lasso` has no such keyword | Python | the snapshot |
| `average_eta = 1.0` | Rust | `[resolved defaults]`; `spec_defaults.rs` |
| `clip_gradient = 1e3`, for `sgd` | Rust | `[resolved defaults]`; `spec_defaults.rs`; `tests/test_sgd.py` checks that the default clip keeps a Poisson fit stable |

`docs/VALIDATION.md` justifies the defaults it measures, and
`tests/test_validation_doc.py` pins that they are still the measured
optimum.

**An output field's name is rendered in one place.** The names are
generated by `format!` over ridge values, half-life suffixes, feature-set
labels, quantile levels and target names. `num_label()` in
`crates/online-polars/src/spec.rs` is the only place a number in a name is
rendered. A number outside `[1e-6, 1e7)` renders compactly, as `1e-300`,
since Rust's float `Display` never uses scientific notation, and a legal
`ridge = 1e-300` would otherwise make a 311-character name. A target literally named
`y__r0.5` renders like a grid field, so `Spec::check` refuses a spec
whose two outputs render to one field name, and `Bank::new` checks again. `test_names_match_the_realized_struct_for_every_model`
(`tests/test_portability.py`) checks, in 40 cases, that the fields a spec
declares are the fields its output has.

## Rename a stable name

**A renamed stable name keeps working until the next major, with a
`PolarsOnlineDeprecationWarning` that names the new spelling.** `polars_online.PolarsOnlineDeprecationWarning` is a
`DeprecationWarning`.

1. **For a spec parameter, add the old name to the forwarding tables.**
   They carry each renamed spec parameter to its new one, at any depth of a
   spec. In Python, `_DEPRECATED` in `python/polars_online/_warnings.py`
   covers a spec builder's keyword and a spec dict's key, `with_windows`'
   `like=` spec included. In Rust, `DEPRECATED` in
   `crates/online-polars/src/spec.rs` covers a TOML key.
2. **For any other name, ship a forwarding of its own.** A keyword of
   another function, a function, a word a parameter takes, a command-line
   flag, an environment variable, an output field or a frame column does
   not go through the tables. Use `forward_deprecated` at the top of the
   function whose keyword moved, a stub that warns and calls the new
   function, an alias for the word, flag or variable, or a second column
   beside the new one. Each warns with the same
   `PolarsOnlineDeprecationWarning`.
3. **Ship the rename in a minor release,** and remove the old name at the
   next major.

**The renames made before 1.0 stay refused,** by `_RENAMED` in
`python/polars_online/_spec.py` and `RENAMED` in
`crates/online-polars/src/spec.rs`. `renamed_function` in
`python/polars_online/_renamed.py` is the refusing stub a pre-1.0 rename
keeps, not a forwarding one.

## Add a spec parameter

**Enter a new parameter's spec in `PARAMETER_INTRODUCED`, under the next
release, in the change that adds it.** `tests/test_released_state.py`
fits its workload under every release on PyPI, and an older release
refuses a keyword it does not take. `PARAMETER_INTRODUCED` names, for each
spec the workload builds with a newer parameter, the first release that
takes it, so an older release is not asked to build it. The test stops
excusing the release as soon as that release is on PyPI.
`scripts/release_version.py --publish` refuses to publish a version older
than one named there, since the test would then expect that release to
build a spec it cannot.

## What not to do

**Three measures look like stability, and would take more work here than
they are worth:**

| do not | why |
|---|---|
| `__getattr__`-guard the submodules | `polars_online._spec` is importable, and someone will reach into it; that is what the underscore is for. Enforcing it would take more work than the honesty is worth |
| freeze the API to prevent change | the snapshot test is for detecting change, not preventing it |
| add `cargo semver-checks` | it is the right tool for a published Rust API, and pure overhead otherwise. The Rust crates carry `publish = false` in their `Cargo.toml` and are not API; add it only if they are ever published |

## Check the result

The change is ready when `uv run pytest tests/test_api_surface.py` passes
on the regenerated snapshot, its diff is in the pull request, and the
CHANGELOG entry names the release kind the diff needs.
