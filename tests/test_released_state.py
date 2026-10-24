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
moved since the release is named in ``DECLARED``, with the entry.

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
RELEASES = ["0.10.0", "0.11.1", "0.12.0", "0.13.0", "1.0.0", "1.0.1", "1.1.0", "1.2.0"]


def _major(version: str) -> int:
    return int(version.split(".")[0])


#: The releases before 1.0, whose files are refused by their version.
REFUSED = [v for v in RELEASES if _major(v) < 1]

#: The workload's kinds newer than a release, by the first version that has
#: each: a release before it has no builder for one, and its probe records
#: the refusal (`audit`, task 223 (b), after 0.13.0).
INTRODUCED = {"audit": "1.0.0"}

#: The workload's specs no listed release can build: a parameter added
#: since the newest (task 271's `huber_scale`). On the release day that
#: ships it, its specs leave this set, as that release builds them.
UNRELEASED = {"huber_mad", "huber_rms", "sgd_huber_mad"}


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
            and _version(version) >= _version(INTRODUCED.get(builder, "0.0.0"))
        )
    ]


#: The specs whose numbers a CHANGELOG entry declares moved since a release,
#: ``(spec, the newest release whose files it moves) -> the entry``. A file
#: of that release or an older one goes on from its own numbers, not this
#: build's, so its continuation is held to loading and running alone. Every
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
    manifest, out = _states(version, tmp_path_factory.mktemp("states-loaded"))
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
        if name not in reach | UNRELEASED:
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
    """What a model keeps and a row does not show: the rows a closed group
    emits, the coefficients where the kind has them (the rest refuse by
    name), and ``marginal``'s pairs table."""
    out = [bank.closed_groups()]
    try:
        out.append(bank.coef(name))
    except ValueError as e:
        assert "coef" in str(e) or "coefficients" in str(e), e
    if name == "marginal":
        out.append(bank.marginal(name))
    return out


def test_a_released_state_loads_and_goes_on_as_this_builds_own(loaded):
    version, manifest, states = loaded
    assert manifest["saved"] == _buildable(version)
    assert set(manifest["refused"]) == {n for n, *_ in probe.WORKLOAD} - set(manifest["saved"])
    df = probe.stream()
    first, second = df.head(probe.HALF), df.slice(probe.HALF)
    built = {name: probe.build(b, name, kw) for name, b, kw, _ in probe.WORKLOAD}
    gaps = {}
    for name, _, _, reads in probe.WORKLOAD:
        if name not in manifest["saved"]:
            continue
        specs = [built[r] for r in reads] + [built[name]]
        file_bank = po.ModelBank.load(states / f"{name}.state", specs=specs)
        own = po.ModelBank(specs)
        own.fit_predict(first)
        gap = _worst_gap(
            file_bank.fit_predict(second).select(name), own.fit_predict(second).select(name)
        )
        for got, want in zip(_tables(file_bank, name), _tables(own, name), strict=True):
            assert got.height == want.height, name
            if got.height:
                gap = max(gap, _worst_gap(got, want))
        gaps[name] = gap
    # Every file goes on to the bit, as the release comparison holds the
    # outputs (review round 9, J-5: a bound of 1e-12 was described as the
    # comparison's, which allows nothing). A spec a CHANGELOG entry moved
    # since the release is named in `DECLARED`, and held to having loaded
    # and run.
    moved = set()
    if version != THIS_BUILD:
        moved = {n for (n, since), _ in DECLARED.items() if _version(version) <= _version(since)}
    wrong = {k: f"{v:.1e}" for k, v in gaps.items() if v > 0.0 and k not in moved}
    assert not wrong, wrong


def test_every_declared_move_names_a_spec_and_a_release():
    names = {name for name, *_ in probe.WORKLOAD}
    for (name, since), why in DECLARED.items():
        assert name in names and since in RELEASES and why, (name, since, why)


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
