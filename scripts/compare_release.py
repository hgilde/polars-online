"""This build's outputs against a released version's, bit for bit.

    uv run python scripts/compare_release.py                  # against the newest release on PyPI
    uv run python scripts/compare_release.py --against 0.9.1
    uv run python scripts/compare_release.py --report         # print; exit 0 whatever differs

The release step (docs/RELEASE-READINESS.md, "Cutting a release"). A test
with a tolerance lets a small numeric change through, and a change to the
numbers a model returns makes a release a minor under the pre-1.0 rule, so
before a release every output field is compared with the last release's,
bit for bit. A field that differs is either a change the CHANGELOG
declares, or a regression, whose cause then gets a test.

The released version is installed from PyPI into a cached venv under
`.cache/release-compare/`, beside the Polars this environment has, and
`scripts/release_probe.py` runs under both builds. Each run reports which
`polars_online` it imported, so the comparison cannot quietly compare a
build with itself. A spec the released version refuses is listed as not
comparable, not as a difference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "release_probe.py"
CACHE = REPO / ".cache" / "release-compare"


def newest_release() -> str:
    with urllib.request.urlopen("https://pypi.org/pypi/polars-online/json", timeout=30) as r:
        return json.load(r)["info"]["version"]


def released_python(version: str) -> Path:
    """A venv holding `polars-online==version` and this environment's Polars."""
    venv = CACHE / version
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        wanted = f"{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run(["uv", "venv", "-q", "--python", wanted, str(venv)], check=True)
        subprocess.run(
            [
                "uv", "pip", "install", "-q", "--python", str(python),
                f"polars-online=={version}", f"polars=={pl.__version__}", "numpy",
            ],
            check=True,
        )  # fmt: skip
    return python


def probe(python: Path, tag: str) -> tuple[pl.DataFrame, dict]:
    out, manifest = CACHE / f"{tag}.parquet", CACHE / f"{tag}.json"
    # `-I`: no PYTHONPATH and no script directory on the path, so each build
    # imports the package its own environment installed.
    subprocess.run([str(python), "-I", str(PROBE), str(out), str(manifest)], check=True)
    return pl.read_parquet(out), json.loads(manifest.read_text(encoding="utf-8"))


def float_bits(s: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Null mask and the raw bits, so -0.0 and a NaN's payload count."""
    return s.is_null().to_numpy(), s.fill_null(0.0).to_numpy().view(np.uint64)


def same(a: pl.Series, b: pl.Series) -> bool:
    if a.dtype != b.dtype:
        return False
    if a.dtype == pl.Float64:
        (na, va), (nb, vb) = float_bits(a), float_bits(b)
        return bool(np.array_equal(na, nb) and np.array_equal(va, vb))
    if isinstance(a.dtype, pl.List) and a.dtype.inner == pl.Float64:
        if not a.list.len().equals(b.list.len(), null_equal=True):
            return False
        return same(a.explode(empty_as_null=True), b.explode(empty_as_null=True))
    return a.equals(b, null_equal=True)


def first_difference(a: pl.Series, b: pl.Series) -> str:
    for i, (x, y) in enumerate(zip(a.to_list(), b.to_list(), strict=True)):
        if x != y and not (x != x and y != y):  # NaN == NaN here
            return f"row {i}: {x!r} -> {y!r}"
    return "bits differ (e.g. -0.0 against 0.0)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--against", help="the released version; default the newest on PyPI")
    ap.add_argument("--report", action="store_true", help="exit 0 whatever differs")
    args = ap.parse_args()
    version = args.against or newest_release()
    CACHE.mkdir(parents=True, exist_ok=True)

    old, old_meta = probe(released_python(version), f"released-{version}")
    new, new_meta = probe(Path(sys.executable), "this-build")
    lines: list[str] = []
    code = compare(version, old, old_meta, new, new_meta, lines)
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"## Against polars-online {version}, bit for bit\n\n```text\n{text}\n```\n")
    return 0 if args.report and code == 1 else code


def compare(
    version: str,
    old: pl.DataFrame,
    old_meta: dict,
    new: pl.DataFrame,
    new_meta: dict,
    out: list[str],
) -> int:
    """Write the comparison into `out`: 0 identical, 1 different, 2 nothing compared."""
    print = out.append  # noqa: A001 -- every line goes to the report
    print(f"released: polars-online {old_meta['version']} from {old_meta['file']}")
    print(f"this:     polars-online {new_meta['version']} from {new_meta['file']}")
    if old_meta["file"] == new_meta["file"]:
        print("both runs imported the same package; nothing was compared")
        return 2

    specs = [s for s in new_meta["ran"] if s in old_meta["ran"]]
    not_comparable = sorted(set(new_meta["ran"]) - set(old_meta["ran"]))
    differs: dict[str, list[str]] = {}
    compared = 0
    for col in new.columns:
        spec = col.split(".", 1)[0]
        if spec not in specs:
            continue
        if col not in old.columns:
            differs.setdefault(spec, []).append(f"{col}: new in this build")
            continue
        compared += 1
        if not same(old[col], new[col]):
            differs.setdefault(spec, []).append(f"{col}: {first_difference(old[col], new[col])}")
    gone = [c for c in old.columns if c.split(".", 1)[0] in specs and c not in new.columns]
    for col in gone:
        differs.setdefault(col.split(".", 1)[0], []).append(f"{col}: gone from this build")

    print(f"\n{compared} fields of {len(specs)} specs compared bit for bit over {new.height} rows")
    if not_comparable:
        print(f"not comparable (refused by {version}): {', '.join(not_comparable)}")
        for spec in not_comparable:
            print(f"  {spec}: {old_meta['refused'].get(spec, '?')}")
    if not differs:
        print(f"identical to {version}")
        return 0
    print(f"\nDIFFERENT from {version}, in {sum(map(len, differs.values()))} fields:")
    for spec, found in differs.items():
        print(f"  {spec}")
        for line in found:
            print(f"    {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
