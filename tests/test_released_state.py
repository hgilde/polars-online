"""A state written by a released wheel meets this build as the release policy
says it will (docs/PLAN.md tasks 109 and 198).

Each release is installed from PyPI into the cache
``scripts/compare_release.py`` keeps (``.cache/release-compare/<version>``),
and ``scripts/release_probe.py --states`` runs under it: every spec of the
release comparison's workload -- every kind the release has -- fitted on the first
half of its stream and saved.

**Before 1.0, a released file is refused by its version.** The bank's minimum
schema (``MIN_BANK_SCHEMA_VERSION``, crates/online-polars/src/bank.rs) is the
schema this build writes, above every release's: 0.10.0 wrote 14, 0.11.1 17,
0.12.0 19 and 0.13.0 20. Each such file is refused naming the way out, and
``test_a_pre_1_0_state_is_refused_by_its_version`` holds each to that.

**From 1.0, a file a 1.x release wrote loads in every later 1.x and goes on as
this build's own** (review round 4, D1, CI3 and TB3): this build loads it with
its own spec and fits the second half, beside a bank of its own that fitted
both halves, and every float, the closed groups, the coefficients and
``marginal``'s table agree. Until 1.0.0 is released and listed, that test runs
on the states this build writes itself, so the harness exists, and has run,
before the first release that must honour it. Every float goes on to the
bit, as the release comparison holds its outputs; a spec a CHANGELOG entry
moved since the release is named in ``DECLARED``, with the entry, and goes
on as the release itself goes on from the same file, run under the
release's own wheel (``test_a_moved_spec_goes_on_as_its_release_did``).

Until 2026-09-28 the loading test held 0.11.1's files (schema 17) bit for
bit and 0.10.0's (schema 14) within 2.2e-13; the pre-1.0 waiver (the user,
"Do not worry about old specs") made them refusals. Downloading needs the
network, so a release is skipped offline (hard rule 1); this build's own
states are not.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import is_transient

TIER = "extended"

REPO = Path(__file__).resolve().parents[1]

#: Every release from 0.10.0, the oldest whose files a loader once promised,
#: to the latest. A release joins the list on its release day
#: (docs/RELEASE-READINESS.md, "The steps of a release"): before 1.0 each
#: wrote a schema of its own and is refused; from 1.0 each loads and goes on.
RELEASES = ["0.10.0", "0.11.1", "0.12.0", "0.13.0", "1.0.0", "1.0.1", "1.1.0", "1.2.0", "1.3.0"]


def _major(version: str) -> int:
    return int(version.split(".")[0])


#: The releases before 1.0, whose files are refused by their version.
REFUSED = [v for v in RELEASES if _major(v) < 1]

#: The workload's kinds newer than a release, by the first version that has
#: each: a release before it has no builder for one, and its probe records
#: the refusal (`audit`, task 223 (b), after 0.13.0).
INTRODUCED = {"audit": "1.0.0"}

#: The workload's specs no listed release can build: a parameter added
#: since the newest. On the release day that ships it, its specs move to
#: `PARAMETER_INTRODUCED` under that release, which builds them.
UNRELEASED: set[str] = {"huber_refit", "sgd_huber_refit"}

#: The workload's specs that need a parameter newer than a release, by the
#: first release that takes it (task 271's `huber_scale`, in 1.3.0). An
#: older release refuses the keyword itself, a `TypeError`, not the
#: `AttributeError` of a builder it lacks.
PARAMETER_INTRODUCED = {name: "1.3.0" for name in ("huber_mad", "huber_rms", "sgd_huber_mad")}


def _parameter_newer(name: str, version: str) -> bool:
    """Whether the spec `name` needs a parameter `version` does not take."""
    return _version(version) < _version(PARAMETER_INTRODUCED.get(name, "0.0.0"))


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def _buildable(version: str) -> list[str]:
    """The workload's specs a release can build: every one, for this build."""
    return [
        name
        for name, builder, *_ in probe.WORKLOAD
        if version == THIS_BUILD
        or (
            name not in UNRELEASED
            and not _parameter_newer(name, version)
            and _version(version) >= _version(INTRODUCED.get(builder, "0.0.0"))
        )
    ]


#: The specs whose numbers a CHANGELOG entry declares moved since a release,
#: ``(spec, the newest release whose files it moves) -> the entry``. A file
#: of that release or an older one goes on from its own numbers, not this
#: build's: `test_a_moved_spec_goes_on_as_its_release_did` holds it to the
#: release's own continuation instead. Every
#: other spec of every 1.x release goes on to the bit: measured on
#: 2026-10-23, 68 specs of 1.0.0, 1.0.1, 1.1.0 and 1.2.0, every gap 0 but
#: these.
DECLARED: dict[tuple[str, str], str] = {
    ("ridge_blocked", "1.0.1"): "1.1.0, Changed: a blocked `ewridge`'s target cross-moments "
    "move in their last bits (5.2e-16 here)",
    # Task 271 (the next release's Changed): `huber` and `sgd(loss="huber")`
    # scale their cut by the residuals' MAD by default, where every release
    # through 1.2.0 used their plain RMS; a file of one goes on from the old
    # scale's fit (3e-2 to 4e-1 here).
    **{
        (name, "1.2.0"): "the next release, Changed: the Huber scale is the MAD by default"
        for name in ("huber", "full_huber", "sgd_huber")
    },
    # Task 270 (a), B-1 (the next release's Fixed): standardized with no
    # `obs_var`, the rows past `min_weight` before a target's prior is sized
    # predicted 0.0 and seeded the residual variance with the level squared;
    # a file of one goes on from that variance.
    ("kalman_unsized", "1.2.0"): "the next release, Fixed: `kalman` no longer predicts 0.0 "
    "before a target's prior is sized",
    # Task 278, batch 3 (the next release's Fixed): after `drift_action =
    # "reset"` the next row reads a clock step of 0, as a fresh bank's first
    # row does; a file whose reset fell in its first half went on from the
    # step the release read (1.4e-2 here).
    ("ridge_reset", "1.3.0"): "the next release, Fixed: a drift reset is a fresh bank from "
    "the next row",
    # Task 275 (the next release's Fixed, and rule 5's exception the user
    # allowed on 2026-10-24): a window's snapshot decays by the product of
    # the per-row factors its moments took. A windowed file saved by a
    # release recorded no such products, so once a snapshot taken after its
    # load is the window's edge it goes on within the golden tolerance, not
    # to the bit (1.8e-15 at most here).
    **{
        (name, "1.2.0"): "the next release, Fixed: a window's snapshot decays by the "
        "product of the per-row factors"
        for name in (
            "ridge_window",
            "ridge_both",
            "marginal_window",
            "lasso_window",
            "ew_class_window",
        )
    },
    # Task 278, D7 and F3 (the next release's Fixed and Changed): `hmm`'s
    # default seeding runs Lloyd's passes to a fixed point from greedy
    # k-means++ starts. On the probe's rows it finds the same two means and
    # numbers them the other way round (and in their last bits), so this
    # build's own bank reads each state under the other's number (9.9e-1 in
    # `filtered_state*` here). A file seeded by a release goes on exactly as
    # that release did (measured against 1.3.0's own continuation, every
    # value equal), as the entry says a state saved after seeding does.
    ("hmm", "1.3.0"): "the next release, Fixed: `kmeans` and `hmm` seed with Lloyd's "
    "passes to a fixed point, and may number their states differently",
    # Task 278, D3 (the next release's Fixed, and rule 5's third exception,
    # the user's): `lasso`'s `tol` is relative to the target's spread, so a
    # file saved by 1.0-1.3 goes on within 1e-9 of this build's own, not to
    # the bit (2.0e-10 at most here, in `full_lasso`'s diagnostics; 1.0e-11
    # in `lasso_window`'s predictions, which 1.2.0 and older move by the
    # window entry above too).
    **{
        (name, "1.3.0"): "the next release, Fixed: `lasso`'s `tol` is relative to the "
        "target's spread"
        for name in ("full_lasso", "lasso_window")
    },
    # Task 278 (the next release's Fixed): with no `obs_var`, `kalman` holds
    # `P` in units of the noise, its gain reading no noise estimate, and its
    # intercept's prior is diffuse. A file of a release has no such form and
    # goes on as it was fitted, the estimate in the gain, where this build's
    # own bank runs the new form; `seqtest` reads a `kalman`'s predictions.
    **{
        (name, "1.3.0"): "the next release, Fixed: `kalman` holds its covariance in units of "
        "the noise, and its intercept's prior is diffuse"
        for name in ("kalman", "full_kalman", "kalman_standardized", "kalman_unsized", "seqtest")
    },
    # Task 278, D-H (the next release's Fixed; rule 5's fifth exception, the
    # user's): `holt`'s gains are the larger of the exact weighted line
    # fit's and the textbook's. A file of a release goes on with the gains it
    # was fitted with (3.0e-2 in `holt`'s predictions here, 1.0e-1 in
    # `full_holt`'s diagnostics).
    **{
        (name, "1.3.0"): "the next release, Fixed: `holt`'s gains are the larger of the "
        "weighted line fit's and the textbook's"
        for name in ("holt", "full_holt")
    },
    # Task 278, D1, D2 and D-S (the next release's Fixed): `sgd`'s clip and
    # AdaGrad's epsilon are in the target's own spread, and its intercept
    # takes the running mean's step while the fit learns the level. A file
    # of a release loads with the flag off (`target_spread`) and goes on as
    # it was fitted, where this build's own bank runs the new steps (3.5e-1
    # in `sgd`'s predictions, 4.6e-1 in `sgd_quantile`'s, 4.2 in `full_sgd`'s
    # diagnostics here). `sgd_huber` moved at 1.2.0 too, by the MAD entry.
    **{
        (name, "1.3.0"): "the next release, Fixed: `sgd` steps in its target's own spread "
        "and learns its level"
        for name in (
            "sgd",
            "full_sgd",
            "sgd_quantile",
            "sgd_epsilon_insensitive",
            "sgd_adagrad",
            "sgd_huber",
            "sgd_huber_mad",
        )
    },
    # Task 278, batch 17 (the next release's Fixed): a stream a spec builds
    # folds into its residual spread each slot's own prediction on the rows
    # a readiness gate withholds, and weighs its residual autocorrelation by
    # the row's weight. A file of a release has neither flag and goes on
    # folding only the residuals it reported, each autocorrelation residual
    # at 1, where this build's own bank runs the new rules: `sigma` and
    # `zscore` 1e-3 to 6e-3 apart at the split here, the conformal interval
    # that starts from `sigma` 1.4 to 12 (of `1 + |v|`), `autocorr` 0.1 to
    # 0.2; no prediction moves.
    **{
        (name, "1.3.0"): "the next release, Fixed: a readiness gate no longer moves the "
        "residual spread, and the residual autocorrelation weighs each residual by its row"
        for name in (
            "ridge_diag",
            "full_ewridge",
            "full_rls",
            "full_quantile",
            "full_ftrl",
            "full_pa",
        )
    },
    # Task 278, the anchored sketch (the next release's Fixed): the MAD's
    # sketch keeps each residual in units of the first one it takes, where
    # 1.3.0's buckets were fixed. A file of 1.3.0 keeps its fixed buckets and
    # goes on as it was fitted, where this build's own bank anchors its
    # sketch: `huber` (the MAD by default) and `huber_mad` 9.5e-4 to 1.4e-3
    # apart in their predictions here, `huber_rms`, which keeps no sketch,
    # to the bit. `full_huber` moves by the sketch and by batch 17's entry
    # above as well (its `autocorr` 0.12 to 0.19).
    **{
        (name, "1.3.0"): "the next release, Fixed: the robust scale's sketch keeps each "
        "residual in units of the first one it takes, so it rescales with the target"
        for name in ("huber", "huber_mad")
    },
    ("full_huber", "1.3.0"): "the next release, Fixed: the robust scale's sketch keeps each "
    "residual in units of the first one it takes; and a readiness gate no longer moves the "
    "residual spread, and the residual autocorrelation weighs each residual by its row",
}

#: The releases from 1.0, whose files load and go on; and this build, whose
#: own files stand in until 1.0.0 is listed and run beside them after.
THIS_BUILD = "this build"
LOADED = [v for v in RELEASES if _major(v) >= 1] + [THIS_BUILD]

# The workload's specs are built to reach every path, not to be ready: a
# readiness warning about one of them is expected and says nothing here.
pytestmark = [
    pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning"),
    pytest.mark.extended(reason="the network: each release's wheel, downloaded from PyPI"),
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load("compare_release")
probe = _load("release_probe")


def _online() -> bool:
    """Whether PyPI answers. An HTTP error is an ``OSError``, and a 404 here
    is PyPI saying the package is gone, so only what a retry could fix reads
    as offline (`data.is_transient`, review 2026-10-05, TA3)."""
    try:
        urllib.request.urlopen("https://pypi.org/simple/polars-online/", timeout=15).close()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if not is_transient(e):
            raise
        return False
    return True


#: Each build's states, written once a module run: the loading test and the
#: continuation test read the same files.
_SAVED: dict[str, tuple[dict, Path]] = {}


def _saved(version: str, factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    if version not in _SAVED:
        _SAVED[version] = _states(version, factory.mktemp("states"))
    return _SAVED[version]


def _states(version: str, out: Path) -> tuple[dict, Path]:
    """The workload's states as ``version`` writes them, and its manifest."""
    manifest = out / "manifest.json"
    if version == THIS_BUILD:
        probe.states(str(out), str(manifest))
    else:
        if not _online():
            pytest.skip("offline: the released wheel cannot be downloaded")
        python = compare.released_python(version)
        # `-I`: the released build imports its own package, not this checkout's.
        subprocess.run(
            [
                str(python),
                "-I",
                str(REPO / "scripts" / "release_probe.py"),
                "--states",
                str(out),
                str(manifest),
            ],
            check=True,
        )
    return json.loads(manifest.read_text(encoding="utf-8")), out


@pytest.fixture(scope="module", params=REFUSED)
def refused(request, tmp_path_factory):
    """A release before 1.0: its version, its manifest, its states."""
    version = request.param
    manifest, out = _states(version, tmp_path_factory.mktemp(f"states-{version}"))
    return version, manifest, out


@pytest.fixture(scope="module", params=LOADED)
def loaded(request, tmp_path_factory):
    """A release from 1.0, or this build: its version, manifest, states."""
    version = request.param
    manifest, out = _saved(version, tmp_path_factory)
    return version, manifest, out


def test_every_spec_was_saved_by_the_release_at_an_older_schema(refused):
    version, manifest, _ = refused
    assert manifest["version"] == version
    # The specs that reach the paths the first workload did not are written
    # against the 1.0 API: a release before it may refuse one by a parameter
    # it lacks (review round 9, J-2). The first workload's are held as before.
    reach = {name for name, *_ in probe.REACH}
    assert [n for n in manifest["saved"] if n not in reach] == [
        n for n in _buildable(version) if n not in reach
    ]
    newer = {name for name, *_ in probe.WORKLOAD} - set(manifest["saved"]) - reach
    assert set(manifest["refused"]) - reach == newer
    for name, why in manifest["refused"].items():
        if name not in reach | UNRELEASED and not _parameter_newer(name, version):
            assert why.startswith("AttributeError") and name in why, (name, why)
    # An older schema: the refusal below is by the version, not the layout.
    assert manifest["schema"] < po.schema_version()


def test_a_pre_1_0_state_is_refused_by_its_version(refused):
    version, manifest, states = refused
    built = {name: probe.build(b, name, kw) for name, b, kw, _ in probe.WORKLOAD}
    for name, _, _, reads in probe.WORKLOAD:
        if name not in manifest["saved"]:
            continue
        specs = [built[r] for r in reads] + [built[name]]
        with pytest.raises(ValueError) as e:
            po.ModelBank.load(states / f"{name}.state", specs=specs)
        message = str(e.value)
        assert f"state schema version {manifest['schema']} not supported" in message, message
        assert "refit it from its input" in message, message


def _floats(frame: pl.DataFrame, prefix: str = "") -> dict[str, np.ndarray]:
    """Every float in a frame, by column, with structs and lists opened."""
    out: dict[str, np.ndarray] = {}
    for name in frame.columns:
        s = frame[name]
        if s.dtype == pl.Struct:
            out.update(_floats(s.struct.unnest(), f"{prefix}{name}."))
            continue
        if isinstance(s.dtype, pl.List):
            # Flattened by hand, not with `explode`, whose default polars is
            # changing: nulls stay NaN, empty lists add nothing.
            flat = _flatten(s.to_list())
            if flat and all(isinstance(v, float) for v in flat):
                out[f"{prefix}{name}"] = np.array(flat, dtype=float)
            continue
        if s.dtype == pl.Float64:
            out[f"{prefix}{name}"] = s.to_numpy()
    return out


def _flatten(values: list) -> list:
    out: list = []
    for v in values:
        if isinstance(v, list):
            out.extend(_flatten(v))
        else:
            out.append(float("nan") if v is None else v)
    return out


def _worst_gap(got: pl.DataFrame, want: pl.DataFrame) -> float:
    """The largest gap between two frames' floats, relative to ``1 + |v|``,
    after their nulls are held to the same rows."""
    a, b = _floats(got), _floats(want)
    assert a.keys() == b.keys()
    worst = 0.0
    for name in a:
        u, v = a[name], b[name]
        assert u.shape == v.shape, name
        np.testing.assert_array_equal(np.isnan(u), np.isnan(v), err_msg=name)
        ok = np.isfinite(v)
        if ok.any():
            worst = max(worst, float((np.abs(u[ok] - v[ok]) / (1 + np.abs(v[ok]))).max()))
    return worst


def _tables(bank: po.ModelBank, name: str) -> list[pl.DataFrame]:
    """What a model keeps and a row does not show (``probe.tables``): the
    rows a closed group emits, the coefficients where the kind has them (the
    rest refuse by name), and ``marginal``'s pairs table."""
    return list(probe.tables(bank, name).values())


def test_a_released_state_loads_and_goes_on_as_this_builds_own(loaded):
    version, manifest, states = loaded
    assert manifest["saved"] == _buildable(version)
    assert set(manifest["refused"]) == {n for n, *_ in probe.WORKLOAD} - set(manifest["saved"])
    df = probe.stream()
    first, second = df.head(probe.HALF), df.slice(probe.HALF)
    built = {name: probe.build(b, name, kw) for name, b, kw, _ in probe.WORKLOAD}
    # Every file goes on to the bit, as the release comparison holds the
    # outputs (review round 9, J-5: a bound of 1e-12 was described as the
    # comparison's, which allows nothing). A spec a CHANGELOG entry moved
    # since the release is named in `DECLARED`, and held here to its row
    # count, since its rows withheld may move too (task 278's `kalman`);
    # `test_a_moved_spec_goes_on_as_its_release_did` holds every field of it
    # to the release's own continuation.
    moved = _moved(version)
    gaps = {}
    for name, _, _, reads in probe.WORKLOAD:
        if name not in manifest["saved"]:
            continue
        specs = [built[r] for r in reads] + [built[name]]
        file_bank = po.ModelBank.load(states / f"{name}.state", specs=specs)
        own = po.ModelBank(specs)
        own.fit_predict(first)
        got_rows = file_bank.fit_predict(second).select(name)
        want_rows = own.fit_predict(second).select(name)
        if name in moved:
            assert got_rows.height == want_rows.height, name
            continue
        gap = _worst_gap(got_rows, want_rows)
        for got, want in zip(_tables(file_bank, name), _tables(own, name), strict=True):
            assert got.height == want.height, name
            if got.height:
                gap = max(gap, _worst_gap(got, want))
        gaps[name] = gap
    wrong = {k: f"{v:.1e}" for k, v in gaps.items() if v > 0.0}
    assert not wrong, wrong


def _moved(version: str) -> set[str]:
    """The specs a CHANGELOG entry moved since ``version``: its files go on
    from their own numbers, not this build's."""
    if version == THIS_BUILD:
        return set()
    return {n for (n, since), _ in DECLARED.items() if _version(version) <= _version(since)}


#: The releases whose moved specs go on as the release itself goes on from
#: the same file, run under its own wheel: every release whose files load.
#: The CHANGELOG says so of each entry in `DECLARED`: `sgd` loads with the
#: old rules and goes on to the bit, `kalman` goes on as it was fitted,
#: `holt` with the textbook gains, the residual spread folding only the
#: residuals it reported, `hmm` exactly as that release did, `huber`'s
#: sketch with its fixed buckets, a Huber file of 1.2.0 on the scale it was
#: fitted with. Measured 2026-10-31: every field of every moved spec of the
#: five to the bit but those `WITHIN`, `ROUNDED`, `YOUNG` and `RESETS` name.
RESUMED = [v for v in RELEASES if _major(v) >= 1]

#: The golden pipeline's tolerance, the library's bound for a move within
#: rounding (`tests/test_golden_pipeline.py`).
GOLDEN = 1e-12

#: Where a moved spec's file goes on within a tolerance of its release's own
#: continuation, not to the bit, relative to ``1 + |v|``, as `DECLARED` is
#: keyed: ``(spec, the newest release whose files it covers)``.
#: - `lasso`'s `tol` is relative to the target's spread since 1.3.0, and a
#:   state of an earlier schema goes on within 1e-9 (CLAUDE.md rule 5's third
#:   exception): 6.8e-10 at most here.
#: - A windowed file of 1.2.0 or older recorded no per-row decay factors,
#:   so once a snapshot taken after its load is the window's edge it goes on
#:   within the golden tolerance (rule 5's first exception): 2.9e-15 here.
#: - A blocked `ewridge`'s target cross-moments moved in their last bits in
#:   1.1.0: a file of 1.0.x goes on within rounding, 8.3e-16 here.
WITHIN: dict[tuple[str, str], float] = {
    ("full_lasso", "1.3.0"): 1e-9,
    ("lasso_window", "1.3.0"): 1e-9,
    **{
        (name, "1.2.0"): GOLDEN
        for name in ("ridge_window", "ridge_both", "marginal_window", "lasso_window")
    },
    ("ew_class_window", "1.2.0"): GOLDEN,
    ("ridge_blocked", "1.0.1"): GOLDEN,
}


def _within(name: str, version: str) -> float:
    """The widest tolerance `WITHIN` gives ``name`` for a file of ``version``."""
    found = [
        t for (n, since), t in WITHIN.items() if n == name and _version(version) <= _version(since)
    ]
    return max(found, default=0.0)


#: A diagnosed state of schema 61-63 goes on to this build's values in its
#: windowed `cusum_*` and `cusum_sq_*` fields while their sums are young:
#: their null's share is read at the sums' age (CLAUDE.md rule 5's fourth
#: exception). Those fields are held to their nulls alone; every other field
#: of a diagnosed spec goes on to the bit.
YOUNG = ("cusum_",)

#: The fields whose arithmetic changed since 1.3.0 within rounding, by the
#: CHANGELOG's Fixed entry: the residuals' skewness reads `m_2·√m_2`, two
#: correctly rounded operations, where it read the platform's `pow`. They
#: and the Jarque-Bera statistic built on them move by an ulp or two (4e-16
#: of `1 + |v|` at most here), in every spec that emits its tails.
ROUNDED = (("skew_", "jarque_bera_"), 1e-15)

#: A spec whose drift detector resets the model, by its flag: from the row
#: after one that fires, every field is what a fresh bank fed those rows
#: returns (the CHANGELOG's Fixed entry, "starts the model over exactly"),
#: so a file goes on as its release did up to its first reset, and as a
#: fresh bank of this build after it.
RESETS = {"ridge_reset": "drift_y"}


def _plain(frame: pl.DataFrame) -> pl.DataFrame:
    """A frame's struct columns opened, an Enum as its values: a release's
    `withheld_reason` has fewer categories, and its rows compare by value."""
    out = frame.select(
        pl.col(c).struct.unnest().name.prefix(f"{c}.") if frame[c].dtype == pl.Struct else pl.col(c)
        for c in frame.columns
    )
    return out.with_columns(pl.col(pl.Enum, pl.Categorical).cast(pl.String))


def _gaps(u: pl.Series, v: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Each row's gap between two float columns or float lists, relative to
    ``1 + |v|`` (0 where they agree to the bit, NaN against NaN and null
    against null included; inf where their nulls, NaNs or lengths part), and
    whether the row agrees to the bit."""
    if isinstance(v.dtype, pl.List) or isinstance(u.dtype, pl.List):
        pairs = list(zip(u.to_list(), v.to_list(), strict=True))
    else:
        pairs = [([a], [b]) for a, b in zip(u.to_list(), v.to_list(), strict=True)]
    gap = np.zeros(len(pairs))
    exact = np.ones(len(pairs), dtype=bool)
    for i, (x, y) in enumerate(pairs):
        if (x is None) != (y is None) or (x is not None and len(x) != len(y)):
            gap[i], exact[i] = np.inf, False
            continue
        if x is None:
            continue
        a = np.array(_flatten(x), dtype=float)
        b = np.array(_flatten(y), dtype=float)
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        if (nan_a != nan_b).any():
            gap[i], exact[i] = np.inf, False
            continue
        ok = ~nan_b
        exact[i] = bool((a[ok] == b[ok]).all())
        if ok.any():
            gap[i] = float((np.abs(a[ok] - b[ok]) / (1 + np.abs(b[ok]))).max())
    return gap, exact


def _differences(
    got: pl.DataFrame,
    want: pl.DataFrame,
    *,
    within: float = 0.0,
    young: tuple[str, ...] = (),
    against: str = "the release's",
) -> list[str]:
    """Where ``got`` parts from ``want``, field by field: the first row, both
    values, how many rows and the largest gap relative to ``1 + |v|``. Every
    field ``want`` reports is compared; floats to the bit, or within
    ``within``; a field starting with one of ``young`` by its nulls alone."""
    got, want = _plain(got), _plain(want)
    if got.height != want.height:
        return [f"{got.height} rows against {against} {want.height}"]
    lost = sorted(set(want.columns) - set(got.columns))
    if lost:
        return [f"{against} fields {lost} are gone"]
    out = []
    for c in want.columns:
        u, v = got[c], want[c]
        floats = any(
            d.is_float() or (isinstance(d, pl.List) and d.inner.is_float())
            for d in (u.dtype, v.dtype)
        )
        if floats:
            gap, exact = _gaps(u, v)
            field = c.split(".", 1)[-1]
            if young and field.startswith(young):
                bad = gap == np.inf
            else:
                tol = max(within, ROUNDED[1]) if field.startswith(ROUNDED[0]) else within
                bad = ~exact & ~(gap <= tol)
        else:
            gap, bad = None, ~(u.eq_missing(v).to_numpy())
        if bad.any():
            row = int(np.flatnonzero(bad)[0])
            size = f"{int(bad.sum())} rows"
            if gap is not None:
                size += f", the largest gap {float(gap[bad].max()):.1e} of 1 + |v|"
            out.append(f"{c} at row {row}: {u[row]!r} against {against} {v[row]!r} ({size})")
    return out


@pytest.fixture(scope="module", params=RESUMED)
def resumed(request, tmp_path_factory):
    """A release's moved specs: its files, and what the release reports on
    the second half of the stream from each, run under its own wheel."""
    version = request.param
    manifest, states = _saved(version, tmp_path_factory)
    names = sorted(_moved(version) & set(manifest["saved"]))
    out = tmp_path_factory.mktemp(f"resumed-{version}")
    subprocess.run(
        [
            str(compare.released_python(version)),
            "-I",
            str(REPO / "scripts" / "release_probe.py"),
            "--resume",
            str(states),
            str(out),
            *names,
        ],
        check=True,
    )
    return version, names, states, out


def _after_reset(rows: pl.DataFrame, second: pl.DataFrame, name: str, flag: str) -> np.ndarray:
    """Each row after its group's first reset in ``rows``."""
    fired = rows[name].struct.field(flag).fill_null(False).to_numpy().astype(bool)
    groups = second["g"].to_numpy()
    after = np.zeros(len(fired), dtype=bool)
    for g in np.unique(groups):
        at = np.flatnonzero((groups == g) & fired)
        if at.size:
            after |= (groups == g) & (np.arange(len(fired)) > at[0])
    return after


def _fresh(spec: list[dict], second: pl.DataFrame, after: np.ndarray, name: str) -> pl.DataFrame:
    """A fresh bank's rows on each group's rows after its first reset, in
    the stream's order."""
    rows = second.with_row_index("_at").filter(pl.Series(after))
    parts = [
        po.ModelBank(spec).fit_predict(rows.filter(pl.col("g") == g)).select("_at", name)
        for g in rows["g"].unique(maintain_order=True)
    ]
    return pl.concat(parts).sort("_at").drop("_at")


def test_a_moved_spec_goes_on_as_its_release_did(resumed):
    """A spec a CHANGELOG entry moved since a release: this build loads the
    release's file and goes on as the release itself goes on from it, every
    field of every row and of the tables `probe.tables` reads, to the bit,
    run under the release's own wheel (`WITHIN`, `ROUNDED`, `YOUNG` and
    `RESETS` name the declared exceptions). `withheld_reason` compares by value, since this
    build's Enum has categories the release's has not."""
    version, names, states, out = resumed
    assert names, f"{version}: no spec of its moved since"
    built = {name: probe.build(b, name, kw) for name, b, kw, _ in probe.WORKLOAD}
    work = {name: reads for name, _, _, reads in probe.WORKLOAD}
    second = probe.stream().slice(probe.HALF)
    wrong: dict[str, list[str]] = {}
    used = {"young": 0, "reset": 0}
    for name in names:
        specs = [built[r] for r in work[name]] + [built[name]]
        bank = po.ModelBank.load(states / f"{name}.state", specs=specs)
        got = {"rows": bank.fit_predict(second).select(name), **probe.tables(bank, name)}
        rule = dict(within=_within(name, version), young=YOUNG if name.startswith("full_") else ())
        for table in sorted(p.name.split(".")[-2] for p in out.glob(f"{name}.*.parquet")):
            want = pl.read_parquet(out / f"{name}.{table}.parquet")
            if table not in got:
                wrong[f"{name}.{table}"] = ["not reported by this build"]
                continue
            mine = got[table]
            if table == "rows" and name in RESETS:
                after = _after_reset(mine, second, name, RESETS[name])
                assert after.any(), f"{name}: no reset fired in the second half"
                used["reset"] += 1
                fresh = _fresh(specs, second, after, name)
                wrong[f"{name}.rows after its resets"] = _differences(
                    mine.filter(pl.Series(after)), fresh, against="a fresh bank's"
                )
                mine, want = mine.filter(pl.Series(~after)), want.filter(pl.Series(~after))
            wrong[f"{name}.{table}"] = _differences(mine, want, **rule)
            if rule["young"] and _differences(mine, want, within=rule["within"]):
                used["young"] += 1
        # The rows were compared: the release wrote them, all of them.
        assert pl.read_parquet(out / f"{name}.rows.parquet").height == second.height, name
    # The exceptions were used: the comparisons they excuse ran, and moved.
    assert used["reset"] == len(set(RESETS) & set(names)) and used["young"], used
    wrong = {k: v for k, v in wrong.items() if v}
    assert not wrong, "\n".join(f"{k}: {line}" for k, v in sorted(wrong.items()) for line in v)


def test_every_declared_move_names_a_spec_and_a_release():
    names = {name for name, *_ in probe.WORKLOAD}
    for (name, since), why in DECLARED.items():
        assert name in names and since in RELEASES and why, (name, since, why)
    # A tolerance excuses only a declared move, of the same releases' files.
    for (name, since), tol in WITHIN.items():
        assert (name, since) in DECLARED and 0 < tol <= 1e-9, (name, since, tol)
    assert set(RESETS) <= {n for n, _ in DECLARED}


def test_the_policy_page_lists_every_release_the_test_holds():
    """``docs/RELEASE-READINESS.md``'s paragraph on ``RELEASES`` names the
    first and the newest release this test holds, and does not call every
    release's files refused once a 1.x release is listed (task 269, review 8
    F-2: it stopped at 1.0.0 and said every release was refused after 1.1.0
    joined the list and 1.x files loaded)."""
    page = (REPO / "docs" / "RELEASE-READINESS.md").read_text(encoding="utf-8")
    section = page.split("#### A released build's state files", 1)[1].split("\n#### ", 1)[0]
    lead = "**Add each release to `RELEASES` there.**"
    assert lead in section
    paragraph = section.split(lead, 1)[1].split("\n\n", 1)[0]
    for version in (RELEASES[0], RELEASES[-1]):
        assert version in paragraph, f"{version} is not named after {lead}"
    if any(_major(v) >= 1 for v in RELEASES):
        assert "any release so far is refused" not in section


def _changelog_entries() -> list[str]:
    """Each bullet of ``CHANGELOG.md``, its lines joined: an entry runs from
    a line opening with ``- `` to the next such line, blank line or heading."""
    entries: list[str] = []
    current: list[str] | None = None
    for line in (REPO / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            if current:
                entries.append(" ".join(current))
            current = [line[2:].strip()]
        elif current is not None and line.startswith("  ") and line.strip():
            current.append(line.strip())
        else:
            if current:
                entries.append(" ".join(current))
            current = None
    if current:
        entries.append(" ".join(current))
    return entries


def test_a_change_against_the_policy_is_on_the_policy_page():
    """A CHANGELOG entry that says a change went against the release policy
    names a setting, as ``name="value"`` in backticks, that
    ``docs/RELEASE-READINESS.md``'s policy names too: the page records each
    departure from its own table, with the reason (task 270, review 9's R4-2:
    ``huber_scale``'s default moved in a minor and the page did not say)."""
    import re

    page = (REPO / "docs" / "RELEASE-READINESS.md").read_text(encoding="utf-8")
    policy = page.split("### The policy", 1)[1].split("\n### ", 1)[0]
    against = [e for e in _changelog_entries() if "against the policy" in " ".join(e.split())]
    assert against, "no CHANGELOG entry says it went against the policy"
    for entry in against:
        settings = set(re.findall(r"`([a-z_][a-z0-9_]*)=", entry))
        assert settings, f"names no setting: {entry[:120]}"
        assert any(f"`{s}`" in policy or f"`{s}=" in policy for s in settings), (
            f"the policy names none of {sorted(settings)}: {entry[:120]}"
        )


#: The last release without the Huber refit (task 277). A `huber` or
#: `sgd(loss="huber")` state it saved goes on here to the bit; a fresh
#: `huber_refit_rows = 0` is its `huber` to the bit only where no MAD sketch
#: is kept (`huber_scale = "rms"`), since the next release anchors that
#: sketch (task 278).
BEFORE_REFIT = "1.3.0"

_REFIT_SCRIPT = """
import json, sys
import polars as pl
import polars_online as po
specs, data, out = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
frames = [po.ModelBank([s]).fit_predict(pl.read_parquet(data)).select(s["name"]) for s in specs]
pl.concat(frames, how="horizontal").write_parquet(out)
"""


def test_no_refit_is_the_release_before_it_to_the_bit(tmp_path):
    """``huber_refit_rows = 0`` turns off the refit 1.4.0 put on by default.
    On 2,000 rows of two targets at a level of 1,000 with 12.5% one-sided
    outliers, nulls and zero weights, where the default's refit runs on both
    targets and moves each (the activation check below):

    - **a state 1.3.0 saved goes on to the bit**, every spec: ``huber`` under
      both scales and with feature sets, and ``sgd``'s Huber loss under each
      schedule, saved by 1.3.0 at row 1,000 and loaded here, reports what
      1.3.0's own run reports on the rows after it. The file has no
      ``huber_refit_rows`` and loads with the refit off, so it never refits
      rows its stream has passed; it keeps 1.3.0's fixed sketch buckets and
      the ``sgd`` steps it was fitted with.
    - **a fresh ``huber_refit_rows = 0`` is 1.3.0's ``huber`` to the bit
      under ``huber_scale = "rms"``**, which keeps no MAD sketch. Under the
      MAD, the default, it parts from 1.3.0 by the sketch's anchor (task 278:
      the sketch keeps each residual in units of the first one it takes,
      where 1.3.0's buckets were fixed), by less than a sketch bucket moves a
      prediction; 6.0e-5 of ``1 + |v|`` here at the level of 1,000, 1.6e-3
      on the second target.
    - ``sgd``'s Huber loss is held fresh to the activation check alone: task
      278's sgd group (D1, D2, D-S: the clip, AdaGrad's epsilon and the
      intercept in the target's own spread and level) moved its steps, so
      with the refit off it does not give 1.3.0's numbers."""
    if not _online():
        pytest.skip("offline: the released wheel cannot be downloaded")
    rng = np.random.default_rng(12)
    n, half = 2000, 1000
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y0 = 1000.0 + 2.0 * x - z + 0.5 * rng.standard_normal(n)
    y0[(rng.random(n) < 0.25) & (x > 0)] += 50.0
    y1 = np.where(np.arange(n) % 7 == 3, np.nan, -3.0 * x + rng.standard_t(3, n))
    w = np.where(np.arange(n) % 11 == 5, 0.0, rng.uniform(0.5, 1.5, n))
    df = pl.DataFrame({"x": x, "z": z, "y0": y0, "y1": y1, "w": w}).with_columns(
        pl.col("y1").fill_nan(None)
    )
    data = tmp_path / "rows.parquet"
    df.write_parquet(data)
    head = tmp_path / "head.parquet"
    df.head(half).write_parquet(head)
    c = dict(targets=["y0", "y1"], features=["x", "z"], half_life=float("inf"), weight="w")
    released = [
        po.spec.huber("h_mad", **c),
        po.spec.huber("h_rms", huber_scale="rms", **c),
        po.spec.huber("h_sets", feature_sets={"x": ["x"], "both": ["x", "z"]}, **c),
        *[
            po.spec.sgd(f"s_{s}", loss="huber", schedule=s, learning_rate=0.05, **c)
            for s in ("constant", "inv_scaling", "adagrad")
        ],
    ]
    out = tmp_path / "released.parquet"
    saved = tmp_path / "bank.state"
    for script, rows, to in ((_REFIT_SCRIPT, data, out), (_SAVE_SCRIPT, head, saved)):
        subprocess.run(
            [
                str(compare.released_python(BEFORE_REFIT)),
                "-I",
                "-c",
                script,
                json.dumps(released),
                str(rows),
                str(to),
            ],
            check=True,
        )
    then = pl.read_parquet(out)

    def run(rows: int | None) -> pl.DataFrame:
        frames = []
        for s in released:
            model = dict(s["model"])
            if rows is not None:
                model["huber_refit_rows"] = rows
            spec = {**s, "model": model}
            frames.append(po.ModelBank([spec]).fit_predict(df).select(s["name"]))
        return pl.concat(frames, how="horizontal")

    def plain(frame: pl.DataFrame) -> pl.DataFrame:
        # `withheld_reason`'s Enum gained categories after the release (task
        # 278); its values are compared, as strings.
        return frame.select(
            pl.col(c).struct.unnest().name.prefix(f"{c}.") for c in frame.columns
        ).with_columns(pl.col(pl.Enum).cast(pl.String))

    # A state 1.3.0 saved goes on to the bit, every spec, past the 500 rows
    # the default would refit at.
    names = [s["name"] for s in released]
    resumed = po.ModelBank.load(saved, specs=released).fit_predict(df.slice(half))
    assert plain(resumed.select(names)).equals(
        plain(then.slice(half).select(names)), null_equal=True
    )

    off = run(0)
    assert plain(off.select("h_rms")).equals(plain(then.select("h_rms")), null_equal=True)
    for name in ("h_mad", "h_sets"):
        # The anchor moved it (activation), and by no more than a bucket of
        # the sketch moves a prediction.
        assert not off[name].equals(then[name], null_equal=True), name
        got, want = _floats(off.select(name)), _floats(then.select(name))
        for field in (f for f in want if ".pred_" in f):
            u, v = got[field], want[field]
            np.testing.assert_array_equal(np.isnan(u), np.isnan(v), err_msg=field)
            ok = np.isfinite(v)
            assert (np.abs(u[ok] - v[ok]) / (1 + np.abs(v[ok]))).max() < 1e-2, field
    moved = run(None)
    for s in released:
        assert not moved[s["name"]].equals(then[s["name"]], null_equal=True), s["name"]


_SAVE_SCRIPT = """
import json, sys
import polars as pl
import polars_online as po
specs, data, out = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
bank = po.ModelBank(specs)
bank.fit_predict(pl.read_parquet(data))
bank.save(out)
"""


def test_a_bank_of_the_release_before_the_refit_resumes_without_it_in_its_groups(tmp_path):
    """A bank 1.3.0 saved holds `huber` groups fitted without the refit, and
    its file has no `huber_refit_rows`: loaded here, each restored group goes
    on as 1.3.0 did -- its state says the refit is off, so it never refits
    rows its stream has passed, and it keeps 1.3.0's fixed sketch buckets --
    to the bit against 1.3.0's own continuation, past 500 rows; a group the
    bank first sees after the load is built from the spec, whose default
    refits at 500 rows, as a fresh bank's does. One bank, two rules: the
    restored groups' and the new one's. (A bank of this build with the refit
    off is not 1.3.0's: its MAD sketch is anchored, task 278.)"""
    if not _online():
        pytest.skip("offline: the released wheel cannot be downloaded")
    rng = np.random.default_rng(21)
    first, second = 150, 600

    def rows(groups: list[str], n: int, t0: float) -> pl.DataFrame:
        k = len(groups) * n
        x = rng.standard_normal(k)
        y = 1000.0 + 2.0 * x + 0.5 * rng.standard_normal(k)
        y[(rng.random(k) < 0.25) & (x > 0)] += 50.0
        t = t0 + np.repeat(np.arange(n, dtype=float), len(groups))
        return pl.DataFrame({"g": groups * n, "t": t, "x": x, "y": y})

    head = rows(["a", "b"], first, 0.0)
    tail = rows(["a", "b", "c"], second, float(first))
    data = tmp_path / "head.parquet"
    head.write_parquet(data)
    c = dict(targets=["y"], features=["x"], group="g", clock="t", gap_cap=10.0)
    spec = po.spec.huber("m", half_life=float("inf"), **c)
    saved = tmp_path / "bank.state"
    whole = tmp_path / "whole.parquet"
    pl.concat([head, tail]).write_parquet(whole)
    went_on = tmp_path / "went_on.parquet"
    for script, rows, to in ((_SAVE_SCRIPT, data, saved), (_REFIT_SCRIPT, whole, went_on)):
        subprocess.run(
            [
                str(compare.released_python(BEFORE_REFIT)),
                "-I",
                "-c",
                script,
                json.dumps([spec]),
                str(rows),
                str(to),
            ],
            check=True,
        )
    # 1.3.0's own run over both parts, its rows after the save.
    then = pl.read_parquet(went_on).with_columns(g=pl.concat([head, tail])["g"]).slice(head.height)
    resumed = po.ModelBank.load(saved, specs=[spec]).fit_predict(tail)
    off = {**spec, "model": {**spec["model"], "huber_refit_rows": 0}}
    one_pass = po.ModelBank([off])
    one_pass.fit_predict(head)
    restored = one_pass.fit_predict(tail)
    fresh_c = po.ModelBank([spec]).fit_predict(tail.filter(pl.col("g") == "c"))

    def plain(frame: pl.DataFrame) -> pl.DataFrame:
        # `withheld_reason`'s Enum gained categories after the release (task
        # 278); its values are compared, as strings.
        return frame["m"].struct.unnest().with_columns(pl.col(pl.Enum).cast(pl.String))

    for g in ("a", "b"):
        mine = plain(resumed.filter(pl.col("g") == g))
        assert mine.equals(plain(then.filter(pl.col("g") == g)), null_equal=True), g
    new = resumed.filter(pl.col("g") == "c")["m"]
    assert new.equals(fresh_c["m"], null_equal=True)
    # The new group's refit ran: it parts from the one pass after its 500th row.
    p = new.struct.field("pred_y").to_numpy()
    q = restored.filter(pl.col("g") == "c")["m"].struct.field("pred_y").to_numpy()
    differ = np.flatnonzero(~((p == q) | (np.isnan(p) & np.isnan(q))))
    assert differ.size and differ[0] >= 500, differ[:3]
