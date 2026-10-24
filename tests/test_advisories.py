"""`scripts/advisories.py` and `.github/workflows/advisories.yml`: both lock
files against OSV's published advisories, weekly, failing only on a
vulnerability in a crate the wheel or the command line links (review 9, X-1;
decision D3, task 270). Every test here runs on canned OSV answers: none
reaches the network."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import urllib.request
from pathlib import Path

import pytest
import yaml

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("advisories", REPO / "scripts/advisories.py")
assert _spec and _spec.loader
adv = importlib.util.module_from_spec(_spec)
sys.modules["advisories"] = adv
_spec.loader.exec_module(adv)

CARGO_LOCK = """
version = 4

[[package]]
name = "online-py"
version = "1.2.0"

[[package]]
name = "rustls"
version = "0.23.45"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "paste"
version = "1.0.15"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "quinn-proto"
version = "0.11.17"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "quick-xml"
version = "0.39.4"
source = "registry+https://github.com/rust-lang/crates.io-index"
"""

UV_LOCK = """
version = 1

[[package]]
name = "polars-online"
version = "1.2.0"
source = { editable = "." }

[[package]]
name = "pillow"
version = "12.3.0"
source = { registry = "https://pypi.org/simple" }
"""

TREE = """online-py v1.2.0 (/repo/crates/online-py)
rustls v0.23.45
rustls v0.23.45 (*)
paste v1.0.15 (proc-macro)
quick-xml v0.39.4
"""

#: OSV's records, as `GET /v1/vulns/<id>` gives them, cut to what is read.
DETAILS = {
    "RUSTSEC-2026-0346": {"id": "RUSTSEC-2026-0346", "summary": "TLS secrets in Debug"},
    "RUSTSEC-2024-0436": {
        "id": "RUSTSEC-2024-0436",
        "summary": "paste - no longer maintained",
        "affected": [{"database_specific": {"informational": "unmaintained"}}],
    },
    "RUSTSEC-2026-0194": {"id": "RUSTSEC-2026-0194", "summary": "quadratic time"},
    "RUSTSEC-2026-0338": {"id": "RUSTSEC-2026-0338", "summary": "panic on STREAM frames"},
    "GHSA-22hg-r758-rrw9": {"id": "GHSA-22hg-r758-rrw9", "summary": "decoder loop"},
    "RUSTSEC-2099-0001": {"id": "RUSTSEC-2099-0001", "withdrawn": "2099-01-01T00:00:00Z"},
}

#: What OSV's batch answers for the packages above, in their order.
BATCH = {
    ("crates.io", "paste", "1.0.15"): ["RUSTSEC-2024-0436"],
    ("crates.io", "quick-xml", "0.39.4"): ["RUSTSEC-2026-0194"],
    ("crates.io", "quinn-proto", "0.11.17"): ["RUSTSEC-2026-0338"],
    ("crates.io", "rustls", "0.23.45"): ["RUSTSEC-2026-0346", "RUSTSEC-2099-0001"],
    ("PyPI", "pillow", "12.3.0"): ["GHSA-22hg-r758-rrw9"],
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test reached the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_the_locks_are_read_for_their_registry_packages_alone():
    assert adv.cargo_lock_packages(CARGO_LOCK) == [
        ("paste", "1.0.15"),
        ("quick-xml", "0.39.4"),
        ("quinn-proto", "0.11.17"),
        ("rustls", "0.23.45"),
    ]
    assert adv.uv_lock_packages(UV_LOCK) == [("pillow", "12.3.0")]
    # The repository's own locks parse, and are what the job sends.
    crates = adv.cargo_lock_packages((REPO / "Cargo.lock").read_text(encoding="utf-8"))
    pypi = adv.uv_lock_packages((REPO / "uv.lock").read_text(encoding="utf-8"))
    assert len(crates) > 300 and len(pypi) > 50, (len(crates), len(pypi))
    assert not any(name.startswith("online-") for name, _ in crates)
    assert not any(name == "polars-online" for name, _ in pypi)


def test_cargo_trees_lines_give_name_and_version():
    assert adv.parse_cargo_tree(TREE) == {
        ("online-py", "1.2.0"),
        ("rustls", "0.23.45"),
        ("paste", "1.0.15"),
        ("quick-xml", "0.39.4"),
    }


def _hits():
    return [(eco, n, v, vid) for (eco, n, v), ids in BATCH.items() for vid in ids]


def test_a_vulnerability_in_a_shipped_crate_fails_and_the_rest_is_reported():
    shipped = adv.parse_cargo_tree(TREE)
    report = adv.classify(_hits(), DETAILS, shipped, {"RUSTSEC-2026-0194": "waits upstream"})
    assert [(h.name, h.id) for h in report.failing] == [("rustls", "RUSTSEC-2026-0346")]
    assert [(h.name, h.id, why) for h, why in report.notices] == [
        ("paste", "RUSTSEC-2024-0436", "informational: unmaintained"),
        ("quick-xml", "RUSTSEC-2026-0194", "accepted: waits upstream"),
    ]
    # A dev-dependency's crate and a Python package fail nothing, even where
    # a crate of the same name ships at another version.
    assert sorted((h.ecosystem, h.name) for h in report.reported) == [
        ("PyPI", "pillow"),
        ("crates.io", "quinn-proto"),
    ]
    # A withdrawn advisory is nowhere.
    everything = [*report.failing, *(h for h, _ in report.notices), *report.reported]
    assert "RUSTSEC-2099-0001" not in {h.id for h in everything}
    # The same crate at a version the graphs do not carry is reported.
    other = adv.classify(_hits(), DETAILS, {("rustls", "0.23.46")}, {})
    assert not other.failing
    # Not accepted, the quick-xml advisory fails.
    strict = adv.classify(_hits(), DETAILS, shipped, {})
    assert {h.id for h in strict.failing} == {"RUSTSEC-2026-0346", "RUSTSEC-2026-0194"}


def test_main_exits_1_on_a_failing_advisory_and_0_without(monkeypatch, tmp_path, capsys):
    posted = []

    def post(url, body):
        assert url == "https://api.osv.dev/v1/querybatch"
        posted.append(body)
        keys = [
            (q["package"]["ecosystem"], q["package"]["name"], q["version"]) for q in body["queries"]
        ]
        return {"results": [{"vulns": [{"id": i} for i in BATCH.get(k, [])]} for k in keys]}

    def get(url):
        return DETAILS[url.rsplit("/", 1)[1]]

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Cargo.lock").write_text(CARGO_LOCK, encoding="utf-8")
    (repo / "uv.lock").write_text(UV_LOCK, encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(adv, "REPO", repo)
    monkeypatch.setattr(adv, "_post", post)
    monkeypatch.setattr(adv, "_get", get)
    monkeypatch.setattr(adv, "shipped_crates", lambda: adv.parse_cargo_tree(TREE))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert adv.main() == 1
    out = capsys.readouterr().out
    assert "## Failing: in a shipped graph (1)" in out and "RUSTSEC-2026-0346" in out
    assert "Queried OSV for 4 crates.io, 1 PyPI packages." in out
    assert summary.read_text(encoding="utf-8") + "\n" == out
    assert len(posted) == 1 and len(posted[0]["queries"]) == 5
    # The lock moved past the advisory: nothing fails.
    monkeypatch.setattr(adv, "shipped_crates", lambda: {("rustls", "0.23.46")})
    assert adv.main() == 0
    # More packages than one batch takes go in batches, in order.
    monkeypatch.setattr(adv, "BATCH", 2)
    posted.clear()
    assert adv.main() == 0
    assert [len(b["queries"]) for b in posted] == [2, 2, 1]


def test_a_short_or_paged_answer_is_an_error_not_a_pass(monkeypatch):
    packages = [("crates.io", "rustls", "0.23.45"), ("crates.io", "paste", "1.0.15")]
    monkeypatch.setattr(adv, "_post", lambda url, body: {"results": [{}]})
    with pytest.raises(SystemExit, match="answered 1 of 2"):
        adv.query_osv(packages)
    monkeypatch.setattr(adv, "_post", lambda url, body: {"results": [{"next_page_token": "x"}, {}]})
    with pytest.raises(SystemExit, match="paged"):
        adv.query_osv(packages)


def test_every_accepted_advisory_says_why():
    for vid, why in adv.ACCEPTED.items():
        assert re.fullmatch(r"RUSTSEC-\d{4}-\d{4}", vid), vid
        assert len(why) > 40, (vid, why)


def test_the_script_uses_the_standard_library_alone():
    text = (REPO / "scripts/advisories.py").read_text(encoding="utf-8")
    imported = set(re.findall(r"^(?:from|import) (\w+)", text, re.M))
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported
    # And reads the graphs from the lock as committed.
    assert '"--locked"' in text


def test_the_workflow_runs_it_weekly_and_by_hand_with_least_privilege():
    wf = yaml.safe_load((REPO / ".github/workflows/advisories.yml").read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert set(on) == {"schedule", "workflow_dispatch"}
    (cron,) = (c["cron"] for c in on["schedule"])
    minute, hour, day, month, weekday = cron.split()
    assert (day, month) == ("*", "*") and weekday.isdigit(), cron
    assert wf["permissions"] == {}
    (job,) = wf["jobs"].values()
    assert job["permissions"] == {"contents": "read"}
    assert job["runs-on"] == "ubuntu-latest"
    runs = [s.get("run", "") for s in job["steps"]]
    assert any(r.strip() == "python3 scripts/advisories.py" for r in runs), runs
    assert json.dumps(wf).count("scripts/advisories.py") == 1
