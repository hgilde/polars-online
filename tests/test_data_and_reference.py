"""Task 2: the synthetic generator and the numpy oracles are self-consistent."""

import numpy as np
import polars as pl
import pytest

from data import public_intraday_or_skip, synthetic
from reference import compute_dclock, ewridge_ref, kalman_ref, rls_ref

TIER = "mixed"


def _arrays(df: pl.DataFrame, k: int = 3):
    x = np.column_stack([df[f"x{j}"].to_numpy() for j in range(k)])
    y = df["y0"].to_numpy().reshape(-1, 1)
    return x, y, df["t"].to_numpy(), df["session"].to_numpy(), df["w"].to_numpy()


def test_synthetic_is_deterministic():
    df1, b1 = synthetic(seed=7)
    df2, b2 = synthetic(seed=7)
    assert df1.equals(df2)
    assert all(np.array_equal(b1[g], b2[g]) for g in b1)
    df3, _ = synthetic(seed=8)
    assert not df1.equals(df3)


def test_synthetic_shape_and_clock():
    df, betas = synthetic(n_groups=2, n_rows=100, k=4, n_targets=2)
    assert df.height == 200
    assert betas["g0"].shape == (100, 2, 4)
    for _, g in df.group_by("group"):
        t = g["t"].to_numpy()
        assert (np.diff(t) > 0).all()
        # volume clock resets at each session break
        vol = g["vol"].to_numpy()
        ses = g["session"].to_numpy()
        breaks = np.nonzero(np.diff(ses))[0] + 1
        assert (vol[breaks] < vol[breaks - 1]).all()


def test_ewridge_oracle_recovers_static_beta():
    rng = np.random.default_rng(0)
    n, k = 500, 3
    beta = np.array([0.5, -1.0, 2.0])
    x = rng.standard_normal((n, k))
    y = (x @ beta + 0.01 * rng.standard_normal(n)).reshape(-1, 1)
    dc = np.ones(n)
    dc[0] = 0.0
    out = ewridge_ref(x, y, dc, np.ones(n), half_life=1e6, ridge=1e-8)
    np.testing.assert_allclose(out["coef"][-1, 0, 1:], beta, atol=1e-3)
    assert abs(out["coef"][-1, 0, 0]) < 1e-2  # intercept ~ 0


def test_ewridge_pred_is_out_of_sample():
    # Pure-noise target => oracle predictions must not correlate with y.
    rng = np.random.default_rng(1)
    n, k = 2000, 2
    x = rng.standard_normal((n, k))
    y = rng.standard_normal((n, 1))
    dc = np.ones(n)
    dc[0] = 0.0
    out = ewridge_ref(x, y, dc, np.ones(n), half_life=200.0)
    m = ~np.isnan(out["pred"][:, 0])
    ic = np.corrcoef(out["pred"][m, 0], y[m, 0])[0, 1]
    assert abs(ic) < 0.08


def test_ridge_decay_matches_rls_exactly():
    df, _ = synthetic(n_groups=1, n_rows=300, k=3, null_frac=0.0)
    x, y, t, s, w = _arrays(df)
    dc, rs = compute_dclock(t, s, len(df), gap_cap=50.0, session_gap=25.0)
    a = ewridge_ref(x, y, dc, w, rs, half_life=300.0, ridge=1.0, ridge_scale=True)
    b = rls_ref(x, y, dc, w, rs, half_life=300.0, ridge=1.0)
    m = ~(np.isnan(a["pred"][:, 0]) | np.isnan(b["pred"][:, 0]))
    assert m.sum() > 250
    np.testing.assert_allclose(a["pred"][m, 0], b["pred"][m, 0], atol=1e-12)


def test_compute_dclock_semantics():
    """Mirrors `caps_and_negative_deltas` in crates/online-core/src/clock.rs."""
    t = np.array([0.0, 10.0, 5.0, 6.0, 200.0])
    with pytest.raises(ValueError, match="row 2: the clock goes backwards by 5"):
        compute_dclock(t, None, 5, gap_cap=50.0)
    d, r = compute_dclock(t, None, 5, gap_cap=50.0, restart_after_step_back=2.0)
    assert r[2] and d[2] == 0.0
    np.testing.assert_allclose(d, [0.0, 10.0, 0.0, 1.0, 50.0])
    # Inclusive: a step back as large as the minimum is a late row.
    with pytest.raises(ValueError, match="row 2: a late row"):
        compute_dclock(t, None, 5, gap_cap=50.0, restart_after_step_back=5.0)
    ses = np.array([0, 0, 1, 1, 1])
    d, r = compute_dclock(t, ses, 5, gap_cap=50.0, session_gap=7.5)
    assert d[2] == 7.5  # session change overrides the negative delta
    d, r = compute_dclock(t, ses, 5, gap_cap=50.0, session_gap="reset")
    assert r[2]


def test_a_zero_weight_row_leaves_the_oracles_finite():
    """Hard rule 9 holds of the oracles too: a row of weight 0 advances the
    clock and teaches nothing, the stream's first row and its first scored
    row included, and is never a 0/0 that poisons what follows. On this
    stream ``kalman_ref`` divided its feature moments by the weight after a
    zero-weight first row, so none of its 120 predictions was a number, and
    ``ewridge_ref``'s sigma2 by its weight after a zero-weight first scored
    row, NaN on each of the 114 rows after it (review 2026-10-06, TA1 and
    TA10)."""
    df, _ = synthetic(seed=71, n_groups=1, n_rows=120, k=3, null_frac=0.0)
    x, y, t, _, w = _arrays(df)
    dc, _ = compute_dclock(t, None, len(df), gap_cap=50.0)

    first_row = w.copy()
    first_row[0] = 0.0
    pred = kalman_ref(x, y, dc, first_row, gap_cap=50.0)["pred"][:, 0]
    scored = np.flatnonzero(np.isfinite(pred))
    assert scored.size > 100, scored.size
    assert np.isfinite(pred[scored[0] :]).all()

    first_scored = int(np.argmax(np.isfinite(ewridge_ref(x, y, dc, w, gap_cap=50.0)["pred"][:, 0])))
    zeroed = w.copy()
    zeroed[first_scored] = 0.0
    sig2 = ewridge_ref(x, y, dc, zeroed, gap_cap=50.0)["sig2"][:, 0]
    assert np.isfinite(sig2).all()
    # The rows after it still add their residuals.
    added = np.flatnonzero(sig2 > 0.0)
    assert added.size > 100 and added[0] > first_scored, added
    assert (sig2[added[0] :] > 0.0).all()


def test_null_policy_in_oracle():
    df, _ = synthetic(n_groups=1, n_rows=200, k=2, null_frac=0.0)
    x = np.column_stack([df["x0"].to_numpy(), df["x1"].to_numpy()])
    y = df["y0"].to_numpy().reshape(-1, 1)
    n = len(df)
    dc = np.ones(n)
    dc[0] = 0.0
    x_null = x.copy()
    x_null[50, 0] = np.nan  # feature null: row skipped
    y_null = y.copy()
    y_null[60, 0] = np.nan  # target null: predict-only
    out = ewridge_ref(x_null, y_null, dc, np.ones(n), half_life=100.0)
    assert np.isnan(out["pred"][50, 0]) and np.isnan(out["weight_sum"][50])
    assert np.isfinite(out["pred"][60, 0]) and np.isnan(out["resid"][60, 0])


@pytest.mark.extended(reason="the network: the public intraday days")
def test_public_intraday_download():
    df = public_intraday_or_skip()
    assert df.height > 1000
    assert (np.diff(df["t"].to_numpy()) > 0).all()
    assert df["close"].null_count() == 0


def test_a_truncated_download_is_retried_and_a_dead_one_is_offline(monkeypatch):
    """`http.client.IncompleteRead` is not an `OSError`; it once escaped as a
    crash from a CI run instead of the retry (or the skip) it deserves."""
    import http.client
    import io

    import data

    calls: list[str] = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def flaky(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise http.client.IncompleteRead(b"partial")
        return _Resp(b"whole")

    monkeypatch.delenv("CI", raising=False)  # three attempts, as on a desk
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    monkeypatch.setattr(data.urllib.request, "urlopen", flaky)
    assert data._download("http://example.invalid/x") == b"whole"
    assert len(calls) == 3

    def dead(url, timeout):
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(data.urllib.request, "urlopen", dead)
    with pytest.raises(RuntimeError, match="offline"):
        data._download("http://example.invalid/x")


def _benchmarks():
    """`tests/public_benchmarks.py`, the sets task 278's convergence classes
    download, held to the same rule."""
    import public_benchmarks

    return public_benchmarks


def _outcome(fetch):
    """What a fetch does: ``("skipped", reason)``, ``(exception name,
    exception)``, or ``("returned", None)``."""
    try:
        fetch()
    except pytest.skip.Exception as e:
        return "skipped", str(e)
    except Exception as e:
        return type(e).__name__, e
    return "returned", None


@pytest.mark.parametrize(
    "fetch",
    [
        lambda data: data.public_intraday_or_skip(("1999-01-01",)),
        lambda data: data.public_quotes_and_trades_or_skip("ZZZUSDT", "1999-01-01"),
        lambda data: _benchmarks().numacc_or_skip(1),
        lambda data: _benchmarks().franti_or_skip("s1"),
        lambda data: _benchmarks().tcpd_or_skip("nile"),
        lambda data: _benchmarks().libsvm_or_skip("regression/abalone", 8),
    ],
    ids=["intraday", "quotes", "numacc", "franti", "tcpd", "libsvm"],
)
def test_only_the_network_is_offline(monkeypatch, tmp_path, fetch):
    """Review 2026-10-05, TA3 and TB2. An HTTP error is an ``OSError``, so a
    404 or a 403 was retried and then called offline, and the test skipped
    instead of failing: a moved file would have hidden its tests for good.
    A 4xx is the server's answer, not the network's: it is raised at once.
    A 5xx or a 429 is retried as a dropped connection is, and offline after
    the last attempt; only offline skips."""
    import io
    import urllib.error

    import data

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("CI", raising=False)  # three attempts, as on a desk
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    calls: list[str] = []

    def status(code: int):
        def urlopen(url, timeout, **tls):
            calls.append(url)
            raise urllib.error.HTTPError(url, code, "status", {}, io.BytesIO(b""))

        return urlopen

    def unreachable(url, timeout, **tls):
        calls.append(url)
        raise urllib.error.URLError("network unreachable")

    for code in (403, 404, 410):
        calls.clear()
        monkeypatch.setattr(data.urllib.request, "urlopen", status(code))
        kind, err = _outcome(lambda: fetch(data))
        assert kind == "HTTPError" and err.code == code, (code, kind, err)
        assert len(calls) == 1, f"a {code} was retried"
    for answer in (status(500), status(503), status(429), unreachable):
        calls.clear()
        monkeypatch.setattr(data.urllib.request, "urlopen", answer)
        kind, reason = _outcome(lambda: fetch(data))
        assert kind == "skipped" and "offline" in reason, (kind, reason)
        assert len(calls) == 3, "a transient failure was not retried"


# ---------------------------------------------------------------------------
# The downloaded data survive an outage at their source (task 339)


class _Body:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self) -> bytes:
        return self.body


def _opener(answers: list):
    """An ``urlopen`` that gives each of ``answers`` in turn: an exception
    is raised, bytes are the response. Records each URL asked."""
    calls: list[str] = []

    def urlopen(request, timeout, **tls):
        calls.append(request.full_url)
        answer = answers[min(len(calls), len(answers)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        return _Body(answer)

    return urlopen, calls


@pytest.mark.parametrize(
    ("ci", "pauses"),
    [("true", [5.0, 10.0, 20.0, 40.0]), (None, [2.0, 4.0])],
    ids=["ci", "desk"],
)
def test_a_download_waits_out_a_short_outage_under_ci_and_gives_up_fast_elsewhere(
    monkeypatch, ci, pauses
):
    """Fränti's server timed out for minutes from GitHub's runners; three
    attempts 2 and 4 seconds apart gave up inside 100 seconds, so a CI run
    skipped or failed for it. Under CI (``CI`` set) five attempts, 5, 10, 20
    and 40 seconds apart; elsewhere three, so a run offline does not hang.
    The schedule is the pauses taken, read from a patched opener and sleep:
    no network."""
    import urllib.error

    import data

    if ci is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", ci)
    slept: list[float] = []
    monkeypatch.setattr(data.time, "sleep", slept.append)
    down = urllib.error.URLError(TimeoutError("timed out"))
    assert list(data.download_pauses()) == pauses

    # Down for all but the last attempt: waited out.
    urlopen, calls = _opener([down] * len(pauses) + [b"whole"])
    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    assert data._download("http://example.invalid/x") == b"whole"
    assert slept == pauses and len(calls) == len(pauses) + 1

    # Down throughout: offline after every pause, and no sleep after the last.
    slept.clear()
    urlopen, calls = _opener([down])
    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    with pytest.raises(data.Offline) as e:
        data._download("http://example.invalid/x")
    assert e.value.__cause__ is down
    assert slept == pauses and len(calls) == len(pauses) + 1

    # A caller's own count takes the schedule's first pauses.
    slept.clear()
    urlopen, calls = _opener([down])
    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    with pytest.raises(data.Offline):
        data._download("http://example.invalid/x", attempts=2)
    assert slept == pauses[:1] and len(calls) == 2


def test_fetch_downloads_only_what_is_not_already_on_disk(monkeypatch, tmp_path):
    """CI restores the sets from its cache or an earlier run's artifact and
    then fetches: a set on disk must not be downloaded again, or an outage
    at its source would still fail the run. The Binance sources go to their
    loaders by date and by symbol and date."""
    import data
    import public_benchmarks

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    urlopen, calls = _opener([b"rownames,a,b\n1,2,3\n2,4,5\n"])
    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    spec = "rdatasets:datasets/one,datasets/two"
    assert public_benchmarks.fetch(spec) == ["rdatasets:datasets/one", "rdatasets:datasets/two"]
    assert len(calls) == 2
    assert sorted(p.name for p in (tmp_path / "rdatasets").iterdir()) == [
        "datasets-one.parquet",
        "datasets-two.parquet",
    ]
    calls.clear()
    assert public_benchmarks.fetch(spec) == ["rdatasets:datasets/one", "rdatasets:datasets/two"]
    assert calls == [], "a set on disk was downloaded again"

    asked: list[tuple] = []
    monkeypatch.setattr(data, "public_intraday", lambda dates: asked.append(("day", dates)))
    monkeypatch.setattr(
        data, "public_quotes_and_trades", lambda s, d: asked.append(("quotes", s, d))
    )
    got = public_benchmarks.fetch("binance:2024-01-02,2024-01-03;binance-um:ETCUSDT/2024-01-02")
    assert got == [
        "binance:2024-01-02",
        "binance:2024-01-03",
        "binance-um:ETCUSDT/2024-01-02",
    ]
    assert asked == [
        ("day", ("2024-01-02",)),
        ("day", ("2024-01-03",)),
        ("quotes", "ETCUSDT", "2024-01-02"),
    ]


def _oracles():
    """``scripts/oracles.py``, loaded once under its own name, as
    tests/test_oracle_ledger.py loads it."""
    import importlib.util
    import pathlib
    import sys

    if "oracles" in sys.modules:
        return sys.modules["oracles"]
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "oracles.py"
    spec = importlib.util.spec_from_file_location("oracles", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["oracles"] = module  # a dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


def test_the_cache_key_moves_with_the_sets_alone(capsys):
    """The cache was keyed on oracles.toml, which every ledger `apply`
    rewrites, so each apply started a new entry. The key is now the list of
    sets: a measured number moving leaves it, a set added moves it, and the
    suite's sets are in the CI test jobs' list."""
    import copy

    import public_benchmarks

    oracles = _oracles()
    doc = oracles.load()
    before = oracles.wanted_datasets(doc=doc)
    gid = next(g for g, v in doc["group"].items() if v.get("dataset"))
    moved = copy.deepcopy(doc)
    for case in moved["group"][gid].get("case", {}).values():
        if isinstance(case, dict):
            for k, v in case.items():
                if isinstance(v, float):
                    case[k] = v * 0.5
    moved["meta"]["tolerance"] = 0.5
    assert oracles.wanted_datasets(doc=moved) == before
    moved["group"][gid]["dataset"] += ";nist:Longley"
    assert oracles.wanted_datasets(doc=moved) != before

    suite = oracles.wanted_datasets(suite=True, doc=doc)
    assert set(before) | set(public_benchmarks.SUITE) == set(suite)
    for spec in public_benchmarks.SUITE:
        assert spec.partition(":")[0] in public_benchmarks.SOURCES, spec

    import argparse

    oracles._cmd_datasets(argparse.Namespace(suite=False, digest=True))
    first = capsys.readouterr().out
    oracles._cmd_datasets(argparse.Namespace(suite=True, digest=True))
    second = capsys.readouterr().out
    assert len(first.strip()) == 16 and len(second.strip()) == 16 and first != second


def test_fetch_waits_out_a_dead_source_once_and_names_every_set_it_missed(monkeypatch, capsys):
    """When a source is down for every round, a later set from it is named
    as missed without waiting out its rounds again (Fränti's two sets cost
    two full waits); the others are fetched; the annotation is the level
    asked for, and the run fails."""
    import argparse

    import data
    import public_benchmarks

    oracles = _oracles()
    sets = ["franti:a1,s1", "franti:s1", "nist:Longley"]
    monkeypatch.setattr(oracles, "wanted_datasets", lambda suite, doc=None: sets)
    tried: list[str] = []

    def fetch(dataset):
        tried.append(dataset)
        if dataset.startswith("franti"):
            raise data.Offline("offline") from TimeoutError("timed out")
        return [dataset]

    monkeypatch.setattr(public_benchmarks, "fetch", fetch)
    args = argparse.Namespace(rounds=1, suite=False, annotate="warning")
    assert oracles._cmd_fetch(args) == 1
    assert tried == ["franti:a1,s1", "nist:Longley"]
    out = capsys.readouterr().out
    assert "fetch: nist:Longley" in out
    assert "::warning::fetch: could not download franti:a1,s1: TimeoutError" in out
    assert "::warning::fetch: could not download franti:s1: its source failed" in out
    assert "::error::" not in out


def _workflow(name: str) -> dict:
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


LAST = "./.github/actions/last-datasets"


def _uses(step: dict) -> str:
    return str(step.get("uses", ""))


def _index(steps: list, predicate) -> int:
    (i,) = [i for i, s in enumerate(steps) if predicate(s)]
    return i


def test_the_oracle_datasets_survive_an_evicted_cache_and_an_outage():
    """oracles.yml's `datasets` job: the cache keyed on the list of sets,
    then on a miss the newest run's `oracle-datasets` artifact, then `fetch`
    for what is still missing, and the cache saved where it missed. The
    weekly sweep skips the cache and fetches afresh, and only an outage
    sends it to the artifact."""
    job = _workflow("oracles.yml")["jobs"]["datasets"]
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    steps = job["steps"]
    key = _index(steps, lambda s: "oracles.py datasets --digest" in str(s.get("run", "")))
    assert "oracles.toml" not in str(steps[key]) and "hashFiles" not in str(steps[key])
    restore = _index(steps, lambda s: "actions/cache/restore@" in _uses(s))
    fresh = _index(steps, lambda s: s.get("id") == "fresh")
    last = _index(steps, lambda s: _uses(s) == LAST)
    fetch = _index(steps, lambda s: str(s.get("run", "")).strip().endswith("oracles.py fetch"))
    save = _index(steps, lambda s: "actions/cache/save@" in _uses(s))
    upload = _index(steps, lambda s: "upload-artifact" in _uses(s))
    assert key < restore < fresh < last < fetch < save < upload
    assert not [s for s in steps if _uses(s).startswith("actions/cache@")]
    assert steps[restore]["if"] == "github.event_name != 'schedule'"
    assert steps[fresh]["if"] == "github.event_name == 'schedule'"
    assert steps[fresh]["continue-on-error"] is True
    assert "--annotate warning" in steps[fresh]["run"]
    when = " ".join(steps[last]["if"].split())
    assert when == (
        "(github.event_name == 'schedule' && steps.fresh.outcome == 'failure') "
        "|| (github.event_name != 'schedule' && steps.cache.outputs.cache-hit != 'true')"
    )
    assert "if" not in steps[fetch] and "continue-on-error" not in steps[fetch]
    assert steps[save]["if"] == (
        "github.event_name != 'schedule' && steps.cache.outputs.cache-hit != 'true'"
    )
    for i in (restore, save):
        assert steps[i]["with"]["key"] == "${{ steps.key.outputs.key }}"
    paths = {steps[i]["with"]["path"] for i in (restore, save, upload)}
    assert paths == {"${{ env.DATASET_PATHS }}"}
    assert steps[upload]["with"]["name"] == "oracle-datasets"


def test_ci_restores_the_suites_datasets_before_it_runs():
    """ci.yml's test job: its OS's cache keyed on the suite's sets, the
    oracle runs' artifact on a miss, `fetch --suite` for the rest (a set it
    cannot get warns and its tests skip), and the cache saved only once
    every set is in hand, all before pytest."""
    job = _workflow("ci.yml")["jobs"]["test"]
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    steps = job["steps"]
    sync = _index(steps, lambda s: s.get("run") == "uv sync --locked")
    key = _index(steps, lambda s: s.get("id") == "data-key")
    assert "oracles.py datasets --suite --digest" in steps[key]["run"]
    assert "runner.os" in steps[key]["run"] and steps[key]["shell"] == "bash"
    restore = _index(steps, lambda s: "actions/cache/restore@" in _uses(s))
    last = _index(steps, lambda s: _uses(s) == LAST)
    fetch = _index(steps, lambda s: s.get("id") == "data-fetch")
    save = _index(steps, lambda s: "actions/cache/save@" in _uses(s))
    pytest_step = _index(steps, lambda s: s.get("name") == "pytest")
    assert sync < key < restore < last < fetch < save < pytest_step
    miss = "steps.data-cache.outputs.cache-hit != 'true'"
    assert steps[last]["if"] == miss and steps[fetch]["if"] == miss
    assert "oracles.py fetch --suite" in steps[fetch]["run"]
    assert steps[fetch]["continue-on-error"] is True
    assert steps[save]["if"] == "steps.data-fetch.outcome == 'success'"
    for i in (restore, save):
        assert steps[i]["with"]["key"] == "${{ steps.data-key.outputs.key }}"
        assert steps[i]["with"]["path"] == "${{ env.DATASET_PATHS }}"
    paths = job["env"]["DATASET_PATHS"].split()
    assert ".cache/microstructure/*.parquet" in paths
    assert not any(p.endswith(".csv") or p == ".cache/microstructure" for p in paths)


def test_no_workflow_publishes_a_new_copy_of_a_dataset():
    """A public repository's artifacts can be downloaded by anyone signed
    in to GitHub, so a dataset in one is a published copy. The one that
    exists, `oracle-datasets`, holds the ledger's sets; nothing else uploads
    `.cache`, and the CI test jobs keep their sets in the Actions cache,
    which only this repository's workflows read."""
    import pathlib

    import yaml

    workflows = sorted((pathlib.Path(__file__).resolve().parents[1] / ".github").rglob("*.yml"))
    uploads = []
    for path in workflows:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        jobs = doc.get("jobs", {}) if isinstance(doc, dict) else {}
        steps = [s for j in jobs.values() for s in j.get("steps", [])]
        if path.name == "action.yml":
            steps = doc.get("runs", {}).get("steps", [])
        for s in steps:
            if "upload-artifact" in _uses(s) or "upload-pages-artifact" in _uses(s):
                where = str(s.get("with", {}).get("path", ""))
                if ".cache" in where or "DATASET" in where:
                    uploads.append((path.name, s["with"].get("name")))
    assert uploads == [("oracles.yml", "oracle-datasets")], uploads


def test_the_last_datasets_action_reads_main_of_this_repository_and_overwrites_nothing():
    """The artifact taken is the newest unexpired one from a run of this
    repository's main: a fork's pull request runs its own workflows and can
    upload an artifact of the same name from a branch it calls main. A file
    already in `.cache` (the weekly sweep's fresh download) is kept."""
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[1] / LAST.removeprefix("./") / "action.yml"
    action = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite"
    (step,) = action["runs"]["steps"]
    run = " ".join(step["run"].split())
    assert "name=oracle-datasets" in run and "--name oracle-datasets" in run
    assert "select(.expired | not)" in run
    assert 'select(.workflow_run.head_branch == "main")' in run
    assert "select(.workflow_run.head_repository_id == .workflow_run.repository_id)" in run
    assert 'if [ ! -e ".cache/$f" ]' in run
    assert step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert "exit 1" not in run, "never fails the job: fetch downloads what is missing"
