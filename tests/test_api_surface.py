"""The public API, rendered to text and pinned (docs/RELEASE-READINESS.md).

The public surface is much larger than `__all__`: it includes every spec
constructor's keyword names **and default values** (changing a default silently
changes users' numbers), and — largest and least
obvious — the **output field names**. Users index the result struct by strings
like `pred_y__r0.5@h100`; those strings are produced by `format!` over floats,
and rustc's float formatting is an implementation detail that has changed
between compiler versions before. Nothing but this file pins the whole grammar.

Beside those it renders what the stability policy calls stable and no
signature shows (review 2026-10-06, task 190): the defaults that resolve in
Rust from a `None` (`[resolved defaults]`, read off the bank's own build), every
helper function's and operator's signature, the columns and dtypes of the
frames the bank returns, the CLI config's TOML keys, the CLI's flags, the
environment variables the shipped code reads, and the words each
string-valued parameter accepts (`[enum values]`).

`tests/api_surface.txt` is the contract. This test regenerates the same text
and diffs it, so **every API change becomes a reviewable diff in the PR that
makes it** — deliberate changes are visible and versioned, accidental ones
fail. Regenerate after an intended change with:

    UPDATE_API_SURFACE=1 uv run pytest tests/test_api_surface.py

and treat the diff to `api_surface.txt` as part of the change under review.
Pre-1.0, a diff here needs at least a minor version bump; a changed *default*
does too, because it changes results without an error.
"""

import difflib
import importlib
import inspect
import json
import os
import pkgutil
import re
import subprocess
import tempfile
from pathlib import Path

import polars as pl

import polars_online as po
from polars_online import _polars_online as native
from polars_online import _spec
from polars_online import spec as spec_mod

SNAPSHOT = Path(__file__).parent / "api_surface.txt"
REPO = Path(__file__).resolve().parent.parent


def helper_modules() -> list[str]:
    """Every public module of the package but `spec`, which has its own section.

    Read from the package's directory rather than listed by hand, so a new
    module lands in the snapshot's diff: the hand-kept list left `po.sim` out
    (docs/PLAN.md task 109).
    """
    return sorted(
        m.name
        for m in pkgutil.iter_modules(po.__path__)
        if not m.name.startswith("_") and m.name != "spec"
    )


def signature_of(obj: object) -> str:
    """A function's signature as `inspect` renders it, keyword names and
    defaults included; empty for a class, a module or a value, which are
    pinned by name."""
    if not callable(obj) or inspect.isclass(obj) or inspect.ismodule(obj):
        return ""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def describe_api(cli: Path | None = None) -> str:
    """The snapshot's text. `cli` is the `online` executable (the
    `online_cli` fixture): its configuration's own keys, its flags and the
    words its format options take are read from it, and left out without it
    (`test_model_registry` reads the field grammar alone)."""
    out: list[str] = []
    w = out.append

    w("# polars-online public API surface. Regenerate: UPDATE_API_SURFACE=1 pytest")
    w("")

    w("[package]  # a function with its signature, keyword names and defaults included")
    for name in sorted(po.__all__):
        w(f"  {name}{signature_of(getattr(po, name))}")
    w(f"  schema_version = {po.schema_version()}")
    w("")

    w("[common parameters]  # every constructor accepts these via **common")
    from polars_online import _spec as _impl

    for p in inspect.signature(_impl._common).parameters.values():
        if p.name in ("name", "model") or p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
        w(f"  {p.name}{default}")
    w("")

    w("[spec constructors]  # keyword names AND defaults are API")
    for name in sorted(dir(spec_mod)):
        if name.startswith("_"):
            continue
        fn = getattr(spec_mod, name)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        w(f"  {name}:")
        for p in sig.parameters.values():
            default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
            w(f"    {p.name}{default}")
    w("")

    out.extend(resolved_defaults_section())
    w("")

    w("[ModelBank]  # name and signature: a parameter added or a default moved is a diff here")
    for name in sorted(dir(po.ModelBank)):
        if not name.startswith("_") or name in ("__init__", "__reduce__"):
            obj = getattr(po.ModelBank, name)
            try:
                sig = str(inspect.signature(obj)) if callable(obj) else ""
            except (TypeError, ValueError):
                sig = ""
            w(f"  {name}{sig}")
    w("")

    w("[frame namespaces]  # lf.online.<method> -> LazyFrame; df.online.<method> -> DataFrame")
    for label, frame in (("LazyFrame", pl.LazyFrame()), ("DataFrame", pl.DataFrame())):
        for name in sorted(dir(frame.online)):
            if name.startswith("_"):
                continue
            sig = inspect.signature(getattr(frame.online, name))
            w(f"  {label}.online.{name}{sig}")
    w("")

    out.extend(frame_columns_section())
    w("")

    w("[helper modules]  # po.<module>.<function> with its signature, keyword names and defaults")
    for mod in helper_modules():
        w(f"  {mod}:")
        module = importlib.import_module(f"polars_online.{mod}")
        for name in sorted(module.__all__):
            w(f"    {name}{signature_of(getattr(module, name))}")
    w("")

    w("[output field grammar]  # the strings users index the output struct by")
    cases: list[tuple[str, dict]] = [
        ("ewridge minimal", dict(targets=["y"], features=["x0"], half_life=100.0)),
        (
            "ewridge full grid, every output",
            dict(
                targets=["y", "z"],
                features=["x0", "x1"],
                feature_sets={"a": ["x0"], "b": ["x0", "x1"]},
                ridge=[1e-6, 0.5],
                half_life=[100.0, 500.0],
                emit_sigma=True,
                emit_zscore=True,
                emit_drift=True,
                emit_metrics=True,
                emit_autocorr=True,
                resid_quantiles=[0.05, 0.95],
                conformal=0.9,
                emit_selected=True,
                emit_averaged=True,
            ),
        ),
        (
            "float rendering at the extremes",
            dict(targets=["y"], features=["x0"], ridge=[1e-300, 0.5], half_life=[100.0, 1e9]),
        ),
        (
            "lasso path",
            dict(targets=["y"], features=["x0"], lasso_path=[0.1, 0.0], half_life=100.0),
        ),
    ]
    for label, kw in cases:
        model = (
            kw.pop("_model", "ewridge")
            if "_model" in kw
            else ("lasso" if "lasso_path" in kw else "ewridge")
        )
        s = getattr(po.spec, model)("m", min_weight=2.0, **kw)
        w(f"  {label}:")
        for f in po.spec.output_fields(s):
            w(f"    {f}")
    for model, kw in [
        ("rls", dict(targets=["y"], features=["x0"], half_life=100.0)),
        ("lasso", dict(targets=["y"], features=["x0"], half_life=100.0, lasso_path=[0.1, 0.0])),
        ("kalman", dict(targets=["y"], features=["x0"], half_life=100.0, coef_half_life=50.0)),
        ("huber", dict(targets=["y"], features=["x0"], half_life=100.0)),
        ("quantile", dict(targets=["y"], features=["x0"], half_life=100.0, quantile=0.5)),
        ("sgd", dict(targets=["y"], features=["x0"], half_life=100.0, learning_rate=0.01)),
        ("pa", dict(targets=["y"], features=["x0"], half_life=100.0)),
        ("ftrl", dict(targets=["y"], features=["x0"], half_life=100.0)),
        ("holt", dict(targets=["y"], half_life=100.0)),
        (
            "ew_cov",
            dict(
                features=["x0", "x1"], stats=["mean", "var", "std", "cov", "corr"], half_life=100.0
            ),
        ),
        (
            "ew_cov every score",
            dict(
                features=["x0", "x1"],
                stats=["partial_corr", "mahal"],
                precision_prior=1e-6,
                mahal_quantiles=[0.5, 0.99],
                pca=2,
                pca_every=10,
                half_life=100.0,
            ),
        ),
        ("kmeans", dict(features=["x0", "x1"], k=2, half_life=100.0)),
        ("micro", dict(features=["x0", "x1"], eps=0.3, half_life=100.0)),
        (
            "ew_class",
            dict(
                features=["x0", "x1"],
                label="y",
                classes=["a", "b"],
                precision_prior=1.0,
                half_life=100.0,
            ),
        ),
        ("seqtest", dict(targets=["y", "z"])),
        (
            "seqtest comparing two specs",
            dict(targets=["y"], a="ridge", b="kalman", a_suffix="@h50"),
        ),
        ("marginal", dict(targets=["y", "z"], features=["x0", "x1"], half_life=100.0)),
        (
            "ew_cov with lags",
            dict(
                features=["x0", "x1"],
                stats=["corr", "lag_corr"],
                lags=[1, 5],
                half_life=100.0,
            ),
        ),
        ("bocpd", dict(features=["x0", "x1"], prior_scale=[1.0])),
        ("corrchange", dict(features=["x0", "x1"], span_rows=100)),
        (
            "hmm",
            dict(features=["x0", "x1"], k=3, precision_prior=0.1, half_life=100.0),
        ),
        (
            "rcov",
            dict(features=["x0", "x1"], group="g", group_close="monotone", block_rows=500),
        ),
        ("deco", dict(features=["x0", "x1", "x2"], half_life=100.0)),
        (
            "deco blocked",
            dict(
                features=["x0", "x1", "x2", "x3"],
                half_life=100.0,
                blocks={"a": ["x0", "x1"], "b": ["x2", "x3"]},
            ),
        ),
    ]:
        s = getattr(po.spec, model.split(" ")[0])("m", min_weight=2.0, **kw)
        w(f"  {model}{'' if ' ' in model else ' minimal'}:")
        for f in po.spec.output_fields(s):
            w(f"    {f}")
    w("")

    # The frame `closed_groups` drains is as much API as an output field
    # (docs/RELEASE-READINESS.md). One schema per bank, whatever has closed,
    # so a fresh bank already has every column. Each kind as the registry
    # builds it, closing on a key, plus the three blocks only an option
    # turns on: `ew_cov`'s PCA, and `marginal`'s lags and bins.
    # Each column as `name: dtype`, as `[frame columns]` gives the other
    # frames (task 194: the counts went to `UInt64`, and a dtype is as much
    # API as a name). A fresh bank has seen no clock, so its clock range is
    # the `Float64` of nulls a bank without a clock column gives.
    w(
        "[closed_groups columns]  # column: dtype, in order: a column renamed, retyped, moved"
        " or added is a diff here"
    )
    from test_model_registry import MINIMAL  # it imports this module

    variants = [(name, dict(MINIMAL[name])) for name in sorted(MINIMAL)] + [
        ("ew_cov with pca", {**MINIMAL["ew_cov"], "pca": 2}),
        ("marginal with lags and bins", {**MINIMAL["marginal"], "lags": [1], "bins": 4}),
    ]
    frames: list[tuple[str, list[str]]] = []
    for label, extra in variants:
        kw: dict[str, object] = {"targets": ["y"], "features": ["x0", "x1"], "half_life": 50.0}
        kw.update(extra)
        kw.setdefault("group", "g")
        kw.setdefault("group_close", "monotone")
        s = getattr(po.spec, label.split(" ")[0])(
            "m", **{k: v for k, v in kw.items() if v is not None}
        )
        schema = po.ModelBank([s]).closed_groups().schema
        frames.append((label, [f"{c}: {dtype_text(t)}" for c, t in schema.items()]))
    shared = [c for c in frames[0][1] if all(c in cols for _, cols in frames)]
    w("  shared:")
    for c in shared:
        w(f"    {c}")
    for label, cols in frames:
        if cols == shared:
            w(f"  {label}: the shared columns only")
            continue
        w(f"  {label}:")
        for c in cols:
            w(f"    {c}")
    w("")

    out.extend(enum_values_section(cli))
    w("")
    out.extend(toml_keys_section(cli))
    w("")
    if cli is not None:
        out.extend(cli_flags_section(cli))
        w("")
    out.extend(env_vars_section())
    return "\n".join(out) + "\n"


# --- [resolved defaults] -------------------------------------------------------

#: Options that bring defaults of their own, which their kind's minimal spec
#: leaves unread; each is rendered as the lines it adds to, or changes in, its
#: kind's minimal rendering.
RESOLVED_VARIANTS: list[tuple[str, dict[str, object]]] = [
    ("ewridge", {"half_life": None, "lam": 0.99}),
    ("ewridge", {"window_size": 100.0}),
    ("sgd", {"loss": "huber"}),
    ("sgd", {"loss": "epsilon_insensitive"}),
    ("sgd", {"schedule": "inv_scaling"}),
    ("ew_cov", {"pca": 2, "max_rows_between_pca": 10}),
    ("micro", {"prune_every": 10.0}),
    ("marginal", {"bins": 4}),
    ("corrchange", {"kind": "sequential"}),
    ("rcov", {"kind": "preavg"}),
    ("bocpd", {"emission": "robust"}),
]


def _registry_spec(name: str, extra: dict[str, object]) -> tuple[dict, dict]:
    """`name`'s minimal spec as `test_model_registry` builds it, with
    `extra` over it (`None` drops a keyword), and the keywords it was built
    with."""
    from test_model_registry import MINIMAL

    kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}
    kw.update(MINIMAL[name])
    kw.update(extra)
    kw = {k: v for k, v in kw.items() if v is not None}
    return getattr(po.spec, name)("m", **kw), kw


def _unneeded(name: str) -> dict[str, object]:
    """`MINIMAL`'s keywords for `name` that its builder does not need, each
    mapped to `None` (dropped): `MINIMAL` gives `sgd` a `learning_rate` equal
    to its default, and a spec that restates a default would hide it moving."""
    from test_model_registry import MINIMAL

    drop: dict[str, object] = {}
    for key, value in MINIMAL[name].items():
        if key in ("targets", "features", "half_life") or value is None:
            continue
        trial = {**drop, key: None}
        try:
            spec, _ = _registry_spec(name, trial)
            native.resolved_defaults(_spec._json(spec))
        except (TypeError, ValueError):
            continue
        drop = trial
    return drop


def _flatten(prefix: str, value: object, into: dict[str, str]) -> None:
    """One `path = value` per leaf, objects walked, everything else as JSON."""
    if isinstance(value, dict) and value:
        for key, inner in value.items():
            _flatten(f"{prefix}.{key}", inner, into)
    else:
        into[prefix] = json.dumps(value)


def resolved_lines(spec: dict) -> dict[str, str]:
    """`path -> value` for what `spec` resolves to in the bank
    (`_polars_online.resolved_defaults`)."""
    lines: dict[str, str] = {}
    for part, value in json.loads(native.resolved_defaults(_spec._json(spec))).items():
        _flatten(part, value, lines)
    return lines


def _call(name: str, kw: dict[str, object]) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in sorted(kw.items()))
    return f"{name}({args})"


def resolved_defaults_section() -> list[str]:
    """Each kind from the least spec its builder takes, and each option in
    `RESOLVED_VARIANTS` as what it moves; the lines every kind resolves
    alike once, at the top."""
    from test_model_registry import MINIMAL

    out = [
        "[resolved defaults]  # what a spec left out resolves to, read off the bank's own build"
        " of the spec shown (_polars_online.resolved_defaults): a moved default is a diff here",
        f"  chunk_size = {native.default_chunk_size()}",
    ]
    minimal: dict[str, tuple[dict[str, object], dict[str, str]]] = {}
    for name in sorted(MINIMAL):
        spec, kw = _registry_spec(name, _unneeded(name))
        minimal[name] = (kw, resolved_lines(spec))
    shared = {
        path: value
        for path, value in next(iter(minimal.values()))[1].items()
        if all(lines.get(path) == value for _, lines in minimal.values())
    }
    out.append("  shared by every kind below:")
    out.extend(f"    {path} = {value}" for path, value in sorted(shared.items()))
    for name, (kw, lines) in minimal.items():
        out.append(f"  {_call(name, kw)}:")
        out.extend(
            f"    {path} = {value}" for path, value in sorted(lines.items()) if path not in shared
        )
    for name, extra in RESOLVED_VARIANTS:
        spec, kw = _registry_spec(name, {**_unneeded(name), **extra})
        base = minimal[name][1]
        moved = {p: v for p, v in resolved_lines(spec).items() if base.get(p) != v}
        assert moved, f"{name} {extra} resolves as {name}'s minimal spec does: drop the variant"
        out.append(f"  {_call(name, kw)}, the lines that differ from {name}'s:")
        out.extend(f"    {path} = {value}" for path, value in sorted(moved.items()))
    return out


# --- [frame columns] -----------------------------------------------------------


def dtype_text(dtype: pl.DataType) -> str:
    """A dtype by its class and parameters, not by Polars' repr of it, which
    is Polars' to change (the weekly canary runs this file on the newest
    Polars)."""
    if isinstance(dtype, pl.List):
        return f"List[{dtype_text(dtype.inner)}]"
    if isinstance(dtype, pl.Enum):
        return f"Enum[{', '.join(dtype.categories.to_list())}]"
    if isinstance(dtype, pl.Struct):
        return "Struct[" + ", ".join(f"{f.name}: {dtype_text(f.dtype)}" for f in dtype.fields) + "]"
    if isinstance(dtype, pl.Datetime):
        return f"Datetime[{dtype.time_unit}, {dtype.time_zone}]"
    return type(dtype).__name__


def frame_columns_section() -> list[str]:
    """The frames a fitted bank returns, and the tables `po.spec` derives
    from a spec, as `column: dtype` in order. `closed_groups()` has its own
    section, and `last_row()` is shown for one spec: its columns after
    `spec` and `group` are that spec's output fields, which the grammar
    section pins for every kind."""
    n = 40
    df = pl.DataFrame(
        {
            "g": ["a", "b"] * (n // 2),
            "x0": [((i * 7) % 11) / 11.0 for i in range(n)],
            "x1": [((i * 5) % 13) / 13.0 for i in range(n)],
            "y": [((i * 3) % 17) / 17.0 for i in range(n)],
        }
    )
    common = dict(targets=["y"], features=["x0", "x1"], half_life=50.0, group="g")
    ridge = po.spec.ewridge("ridge", **common)
    pairs = po.spec.marginal("pairs", **common)
    pairs_lags_bins = po.spec.marginal(
        "pairs_lags_bins", lags=[1], serial_rule="bartlett", bins=4, **common
    )
    bank = po.ModelBank([ridge, pairs, pairs_lags_bins])
    bank.fit_predict(df)
    frames = [
        ("ModelBank.groups()", bank.groups()),
        ("ModelBank.summary()", bank.summary()),
        ("ModelBank.describe()", bank.describe()),
        ("ModelBank.coef()", bank.coef()),
        ("ModelBank.last_row() of an ewridge spec", bank.last_row("ridge")),
        ("ModelBank.marginal()", bank.marginal("pairs")),
        ("ModelBank.marginal() with lags, serial_rule and bins", bank.marginal("pairs_lags_bins")),
        ("po.spec.output_index()", po.spec.output_index(ridge)),
        ("po.spec.coef_index()", po.spec.coef_index(ridge)),
        ("po.spec.coef_fields()", po.spec.coef_fields(ridge)),
    ]
    out = [
        "[frame columns]  # column: dtype, in order: a column renamed, retyped, moved or added"
        " is a diff here"
    ]
    for label, frame in frames:
        out.append(f"  {label}:")
        out.extend(f"    {c}: {dtype_text(t)}" for c, t in frame.schema.items())
    out.append("  ModelBank.gram(), each entry's keys:")
    out.extend(f"    {key}" for key in bank.gram("ridge")[0])
    return out


# --- [enum values] -------------------------------------------------------------

#: A word no parameter takes. The refusal of it names the words a parameter
#: does take; each is then checked to be taken, so a misread message fails
#: here rather than being pinned.
BOGUS = "zz_bogus"

#: (owner.parameter, builder, keywords the builder needs for the parameter to
#: be read, where the word goes): every string-valued spec parameter, probed
#: at the bank's own door (`validate_spec`, which a TOML spec meets too). The
#: shape is a word, a list of words, or a table's key.
ENUM_PROBES: list[tuple[str, str, dict[str, object], str, str]] = [
    ("spec.drift_action", "ewridge", {}, "drift_action", "word"),
    ("spec.group_close", "rcov", {}, "group_close", "word"),
    (
        "spec.session_gap",
        "ewridge",
        {"clock": "t", "gap_cap": 10.0, "session": "s", "session_gap": 1.0},
        "session_gap",
        "word",
    ),
    ("bocpd.emission", "bocpd", {}, "model.emission", "word"),
    ("corrchange.alpha_adjust", "corrchange", {}, "model.alpha_adjust", "word"),
    ("corrchange.kind", "corrchange", {}, "model.kind", "word"),
    ("corrchange.norm", "corrchange", {"kind": "window"}, "model.norm", "word"),
    ("deco.dynamics", "deco", {}, "model.dynamics", "word"),
    # A windowed model's `closed` (docs/PLAN.md task 196): Polars' four words,
    # `"left"` and `"none"` refused by name with their reason.
    ("ew_class.closed", "ew_class", {"window_size": 100.0}, "model.closed", "word"),
    ("ew_cov.closed", "ew_cov", {"window_size": 100.0}, "model.closed", "word"),
    ("ewridge.closed", "ewridge", {"window_size": 100.0}, "model.closed", "word"),
    ("lasso.closed", "lasso", {"window_size": 100.0}, "model.closed", "word"),
    ("marginal.closed", "marginal", {"window_size": 100.0}, "model.closed", "word"),
    ("ew_class.covariance", "ew_class", {}, "model.covariance", "word"),
    ("ew_class.window_budget", "ew_class", {"window_size": 100.0}, "model.window_budget", "key"),
    ("ew_cov.stats", "ew_cov", {}, "model.stats", "list"),
    ("ew_cov.window_budget", "ew_cov", {"window_size": 100.0}, "model.window_budget", "key"),
    ("ewridge.ridge_scale", "ewridge", {}, "model.ridge_scale", "word"),
    ("ewridge.target_gaps", "ewridge", {}, "model.target_gaps", "word"),
    ("ewridge.window_budget", "ewridge", {"window_size": 100.0}, "model.window_budget", "key"),
    ("ftrl.loss", "ftrl", {}, "model.loss", "word"),
    ("hmm.covariance", "hmm", {}, "model.covariance", "word"),
    ("hmm.seed_rule", "hmm", {}, "model.seed_rule", "word"),
    ("kmeans.seed_rule", "kmeans", {}, "model.seed_rule", "word"),
    ("lasso.target_gaps", "lasso", {}, "model.target_gaps", "word"),
    ("lasso.window_budget", "lasso", {"window_size": 100.0}, "model.window_budget", "key"),
    ("marginal.bin_rule", "marginal", {"bins": 4}, "model.bin_rule", "word"),
    ("marginal.feature_moments", "marginal", {}, "model.feature_moments", "word"),
    ("marginal.serial_rule", "marginal", {"lags": [1]}, "model.serial_rule", "word"),
    ("marginal.shards", "marginal", {}, "model.shards", "word"),
    ("marginal.window_budget", "marginal", {"window_size": 100.0}, "model.window_budget", "key"),
    ("pa.mode", "pa", {}, "model.mode", "word"),
    ("rcov.kernel", "rcov", {}, "model.kernel", "word"),
    ("rcov.kind", "rcov", {}, "model.kind", "word"),
    ("sgd.loss", "sgd", {}, "model.loss", "word"),
    ("sgd.schedule", "sgd", {}, "model.schedule", "word"),
]

_QUOTED = re.compile(r'"([^"]*)"|`([^`]*)`|\'([^\']*)\'')
_WORD = re.compile(r"[a-z][a-z0-9_]*")


def words_in(message: str, parameter: str) -> list[str]:
    """The words a refusal names as the ones taken: quoted, after the
    parameter's name (a prefix naming the spec is not read), or else the
    bare list after `expected` or `one of`."""
    at = message.find(parameter)
    tail = message[at + len(parameter) :] if at >= 0 else message
    quoted = [next(g for g in m.groups() if g is not None) for m in _QUOTED.finditer(tail)]
    words = [q for q in quoted if q != BOGUS and _WORD.fullmatch(q)]
    listed = re.search(r"(?:expected|one of)(.*)", tail, flags=re.DOTALL)
    if not words and listed:
        bare = re.split(r"[;\n]|\(got|, got|\)", listed.group(1))[0].replace("one of", "")
        words = [x.strip() for x in re.split(r",| or ", bare) if _WORD.fullmatch(x.strip())]
    return words


def _spec_refusal(spec: dict, path: str, value: object) -> str:
    """What the bank's door says to `spec` with `value` at `path`; empty when
    it takes it. The spec is renamed to a name that cannot read as a word, as
    a refusal's `spec "<name>":` prefix would otherwise."""
    raw = json.loads(_spec._json(spec))
    raw["name"] = "Probe"
    where = raw["model"] if path.startswith("model.") else raw
    where[path.removeprefix("model.")] = value
    try:
        native.validate_spec(json.dumps(raw))
    except ValueError as e:
        return str(e)
    return ""


def _shaped(word: str, shape: str) -> object:
    return {"word": word, "list": [word], "key": {word: 1.0}}[shape]


def taken_words(label: str, refuse) -> list[str]:
    """The words a parameter takes: those its refusal of `BOGUS` names, each
    checked to be taken -- refused, if at all, for another reason than the
    word, which is refused with `BOGUS`'s message."""
    parameter = label.rsplit(".", 1)[-1]
    template = refuse(BOGUS)
    assert template, f"{label}: {BOGUS!r} was taken"
    words = words_in(template, parameter)
    assert words, f"{label}: no words read from {template!r}"
    for word in words:
        said = refuse(word)
        assert said != template.replace(BOGUS, word), f"{label}: {word!r} is refused: {said}"
    return words


def enum_values_section(cli: Path | None) -> list[str]:
    out = [
        "[enum values]  # the words each string-valued parameter takes, read from its refusal"
        " of one it does not: a word added, renamed or dropped is a diff here"
    ]
    for label, name, needs, path, shape in ENUM_PROBES:
        spec, _ = _registry_spec(name, needs)
        words = taken_words(
            label, lambda word, s=spec, p=path, sh=shape: _spec_refusal(s, p, _shaped(word, sh))
        )
        out.append(f"  {label}: {', '.join(sorted(words))}")

    def python_refusal(fn, kw: dict[str, object], key: str):
        def refuse(word: str) -> str:
            try:
                fn(**{**kw, key: word})
            except (TypeError, ValueError) as e:
                return str(e)
            return ""

        return refuse

    target = python_refusal(po.target, {"column": "Y", "relative_to": "X"}, "relative")
    out.append(f"  po.target.relative: {', '.join(sorted(taken_words('relative', target)))}")
    for op in ("ewm_mean", "ewm_rate", "ewm_sum", "rewm_mean", "rewm_rate", "rewm_sum"):
        for key in ("closed", "partial"):
            refuse = python_refusal(
                getattr(po, op), {"input": "X", "half_life": 10.0, "window_size": 5.0}, key
            )
            out.append(f"  po.{op}.{key}: {', '.join(sorted(taken_words(key, refuse)))}")
    reasons = (
        po.ModelBank([po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0)])
        .fit_predict(pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}))["m"]
        .struct.field("withheld_reason")
        .dtype
    )
    assert isinstance(reasons, pl.Enum)
    out.append("  withheld_reason, in precedence order: " + ", ".join(reasons.categories.to_list()))
    if cli is not None:
        for key in ("input_format", "output_format"):

            def toml_refusal(word: str, key: str = key) -> str:
                return _cli_refusal(cli, f'{key} = "{word}"\n')

            out.append(f"  toml.{key}: {', '.join(sorted(taken_words(key, toml_refusal)))}")
        for flag in ("--input-format", "--output-format"):

            def flag_refusal(word: str, flag: str = flag) -> str:
                return _cli_refusal(cli, "", [flag, word])

            words = taken_words(flag.lstrip("-").replace("-", "_"), flag_refusal)
            out.append(f"  cli {flag}: {', '.join(sorted(words))}")
    return out


# --- [toml keys] and [cli flags] -----------------------------------------------


def _cli_refusal(cli: Path, config: str, args: list[str] | None = None) -> str:
    """What `online` says, on stderr, to `config` and `args`; empty when it
    exits 0 (it never does here: no config below has specs)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bank.toml"
        path.write_text(config, encoding="utf-8")
        res = subprocess.run(
            [str(cli), "--config", str(path), *(args or [])],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    return "" if res.returncode == 0 else res.stderr


def keys_in(message: str) -> list[str]:
    """The keys serde lists after `expected` in its refusal of an unknown
    one, sorted: the set is the API, not the declaration order."""
    tail = message.split("expected", 1)[1]
    keys = [m.group(1) for m in re.finditer(r"`([^`]*)`", tail) if m.group(1) != BOGUS]
    assert keys, f"no keys read from {message!r}"
    return sorted(keys)


def _door_keys(spec: dict) -> list[str]:
    try:
        native.validate_spec(json.dumps(spec))
    except ValueError as e:
        return keys_in(str(e))
    raise AssertionError(f"{spec} was taken")


def toml_keys_section(cli: Path | None) -> list[str]:
    out = [
        "[toml keys]  # read from the refusal of a key the table has not got: a key added,"
        " renamed or removed is a diff here; the run config's keys need the CLI"
    ]
    if cli is not None:
        out.append("  run config:")
        out.extend(f"    {k}" for k in keys_in(_cli_refusal(cli, f"{BOGUS} = 0\n")))
    out.append("  [[specs]]:")
    out.extend(f"    {k}" for k in _door_keys({"name": "m", BOGUS: 0}))
    out.append("  [specs.model], by type:")
    for kind in sorted(native.model_kinds()):
        spec = {
            "name": "m",
            "model": {"type": kind, BOGUS: 0},
            "targets": ["y"],
            "features": ["x0"],
        }
        out.append(f"    {kind}: {', '.join(_door_keys(spec))}")
    return out


def cli_flags_section(cli: Path) -> list[str]:
    """`online --help`'s usage line and options, each with the value it
    takes, if any; the help text beside them is not pinned."""
    text = subprocess.run(
        [str(cli), "--help"], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout
    out = [
        "[cli flags]  # `online --help`: a flag added, renamed or removed, or one that starts or"
        " stops taking a value, is a diff here"
    ]
    # clap names the program by the file it was run from: `online.exe` on
    # Windows.
    out.extend(
        f"  {line.strip().replace(cli.name, cli.stem)}"
        for line in text.splitlines()
        if line.startswith("Usage:")
    )
    for line in text.splitlines():
        m = re.match(r"\s+((?:-\w, )?--[\w-]+(?: <\w+>)?)", line)
        if m:
            out.append(f"  {m.group(1)}")
    return out


# --- [env vars] ----------------------------------------------------------------

_RUST_READ = re.compile(r"\benv::var(?:_os)?\(\s*([^)]*?)\s*\)")
_PY_READ = re.compile(r"os\.(?:environ\.get|getenv)\(\s*([^,)]*)|os\.environ\[\s*([^\]]*)\]")


def env_vars() -> list[str]:
    """Every environment variable the shipped code reads at run time: the
    crates' `src/` and the package. An argument that is neither a string
    literal nor a `const` of one fails here, rather than be left out."""
    rust = sorted((REPO / "crates").glob("*/src/**/*.rs"))
    consts: dict[str, str] = {}
    for path in rust:
        for m in re.finditer(
            r'const\s+(\w+)\s*:\s*&str\s*=\s*"([^"]*)"', path.read_text(encoding="utf-8")
        ):
            consts[m.group(1)] = m.group(2)
    names: set[str] = set()
    sources = [(p, _RUST_READ) for p in rust] + [
        (p, _PY_READ) for p in sorted((REPO / "python" / "polars_online").glob("*.py"))
    ]
    for path, pattern in sources:
        for m in pattern.finditer(path.read_text(encoding="utf-8")):
            arg = next(g for g in m.groups() if g is not None).strip()
            literal = re.fullmatch(r"""["']([^"']*)["']""", arg)
            ident = arg.rsplit("::", 1)[-1]
            if literal:
                names.add(literal.group(1))
            elif ident in consts:
                names.add(consts[ident])
            else:
                raise AssertionError(f"{path}: an environment read this cannot name: {m.group(0)}")
    return sorted(names)


def env_vars_section() -> list[str]:
    return [
        "[env vars]  # read at run time by the shipped code (crates/*/src, the package):"
        " a variable added or renamed is a diff here",
        *(f"  {name}" for name in env_vars()),
    ]


#: Parameters typed `str` that name something of the caller's -- a column, a
#: spec, a grid suffix -- rather than take one word of a fixed set, so
#: `[enum values]` has no words to read for them.
FREE_TEXT = {
    "name",
    "clock",
    "group",
    "session",
    "weight",
    "label",
    "hazard_col",
    "exog_tvtp",
    "a",
    "b",
    "a_suffix",
    "b_suffix",
}


def test_every_word_valued_parameter_is_probed():
    """A builder's or the shared parameters' keyword typed as a bare `str`
    takes a word of a fixed set, unless it names something of the caller's
    (`FREE_TEXT`); each such keyword is in `ENUM_PROBES`, so a new one cannot
    stay out of `[enum values]`. (`session_gap`'s `"reset"`, `window_budget`'s
    keys and `ew_cov`'s `stats` are typed otherwise and probed by hand.)"""
    from test_model_registry import MINIMAL

    probed = {label for label, *_ in ENUM_PROBES}

    def worded(fn) -> list[str]:
        return [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if "str" in [part.strip() for part in str(p.annotation).split("|")]
            and p.name not in FREE_TEXT
        ]

    wanted = {f"spec.{p}" for p in worded(_spec._common)}
    for name in MINIMAL:
        wanted |= {f"{name}.{p}" for p in worded(getattr(po.spec, name))}
    assert wanted <= probed, f"string-valued parameters with no probe: {sorted(wanted - probed)}"


def test_api_surface_matches_the_snapshot(online_cli):
    got = describe_api(online_cli)
    if os.environ.get("UPDATE_API_SURFACE"):
        SNAPSHOT.write_text(got, encoding="utf-8")
        return
    assert SNAPSHOT.exists(), (
        "no snapshot yet — run UPDATE_API_SURFACE=1 pytest tests/test_api_surface.py"
    )
    want = SNAPSHOT.read_text(encoding="utf-8")
    if got != want:
        diff = "\n".join(
            difflib.unified_diff(
                want.splitlines(), got.splitlines(), "api_surface.txt", "current", lineterm=""
            )
        )
        raise AssertionError(
            "The public API changed. If intended, regenerate the snapshot with\n"
            "UPDATE_API_SURFACE=1 and include the diff in the PR — a changed\n"
            "field name or default needs a version bump.\n\n" + diff
        )
