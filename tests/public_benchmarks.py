"""Public benchmark sets with a known answer, downloaded and cached (task 278 (c)).

Hard rule 1 (CLAUDE.md): nothing here is in the repo. Each set is fetched by
:func:`data._download`, which retries what a retry can fix and raises
:class:`data.Offline` once the network is gone; the cached copy lives under
``.cache/benchmarks/``. A test calls the ``*_or_skip`` form, which turns
:class:`data.Offline` (and nothing else) into a skip.

The sets, each an input to this library's models with an answer taken as
given. None of them is used to score the libraries the tests take oracles
from (the user, 2026-10-25: "we do not want to retest existing libraries
just our part").

- **NIST StRD NumAcc1-4** (https://www.itl.nist.gov/div898/strd/univ/), a
  U.S. Government work, public domain: values at a large offset with a tiny
  spread, and a certified mean and standard deviation.
- **Fränti and Sieranoja's clustering basic benchmark**
  (https://cs.uef.fi/sipu/datasets/; P. Fränti and S. Sieranoja, "K-means
  properties on six clustering benchmark datasets", Applied Intelligence 48,
  2018), free for research use, cite the article: the S-sets (15 Gaussian
  clusters of rising overlap) and the A-sets (20, 35 and 50 clusters), with
  their ground-truth labels and centroids.
- **TCPD, the Turing Change Point Dataset**
  (https://github.com/alan-turing-institute/TCPD, MIT; van den Burg and
  Williams, "An Evaluation of Change Point Detection Algorithms", 2020):
  annotated series, the five annotators' change points each. Only the series
  the repository ships as JSON are read.
- **LIBSVM's regression and binary sets**
  (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), the copies of
  the UCI and StatLib sets every SGD paper reads: ``abalone`` (4,177 rows, 8
  features, 0.26 MB) and ``a9a`` (32,561 rows, 123 binary features, 2.3 MB).
"""

from __future__ import annotations

import io
import json
import re
import ssl
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np

import data

NIST_UNIV = "https://www.itl.nist.gov/div898/strd/univ/data/{name}.dat"
FRANTI = "https://cs.uef.fi/sipu/datasets/{name}"
TCPD = "https://raw.githubusercontent.com/alan-turing-institute/TCPD/master/{path}"
LIBSVM = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/{path}"
#: NIST's server answers urllib's default agent with a 403.
USER_AGENT = "polars-online-tests (https://github.com/hgilde/polars-online)"


def _lenient_tls() -> ssl.SSLContext:
    """The default verification without the strict RFC 5280 checks Python
    3.13 turned on (``VERIFY_X509_STRICT``): the chain and the host are still
    verified. LIBSVM's server certificate lacks a Subject Key Identifier,
    which the strict check refuses (2026-10-25); urllib reports that as a
    ``URLError``, so without this every LIBSVM test skipped as offline on
    3.13 and later, which is a false skip."""
    context = ssl.create_default_context()
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def _cached(name: str, url: str, context: ssl.SSLContext | None = None) -> bytes:
    """The bytes at ``url``, from ``.cache/benchmarks/<name>`` once fetched
    (``data.CACHE_DIR`` read at the call, so a test can point it elsewhere)."""
    bench = data.CACHE_DIR / "benchmarks"
    bench.mkdir(parents=True, exist_ok=True)
    path = bench / name
    if not path.exists():
        raw = data._download(url, user_agent=USER_AGENT, context=context)
        data.write_atomically(path, lambda tmp: Path(tmp).write_bytes(raw))
    return path.read_bytes()


def _or_skip(fetch, what: str):
    import pytest

    try:
        return fetch()
    except data.Offline as e:
        pytest.skip(f"offline: could not download {what} ({e.__cause__})")


# ---------------------------------------------------------------------------
# NIST StRD univariate summary statistics


def numacc(i: int) -> tuple[np.ndarray, Fraction, Fraction]:
    """NumAcc``i``: the values as doubles, and the certified mean and sample
    standard deviation (``n - 1``) as exact decimals. The certificate is of
    the decimals; a double nearest each differs from it by half a rounding
    step, so a test compares against the exact moments of the doubles, and
    the certificate only within the doubles' own rounding."""
    name = f"NumAcc{i}"
    text = _cached(f"{name}.dat", NIST_UNIV.format(name=name)).decode("ascii")
    mean = Fraction(re.search(r"ybar:\s*([-\d.]+)", text).group(1))
    sd = Fraction(re.search(r"\bs:\s*([-\d.]+)", text).group(1))
    lines = text.splitlines()
    start = next(j for j, line in enumerate(lines) if line.startswith("Data: Y")) + 2
    values = np.array([float(line) for line in lines[start:] if line.strip()], dtype=float)
    return values, mean, sd


def numacc_or_skip(i: int) -> tuple[np.ndarray, Fraction, Fraction]:
    return _or_skip(lambda: numacc(i), f"NIST NumAcc{i}")


# ---------------------------------------------------------------------------
# Fränti's clustering basic benchmark


def _points(text: str) -> np.ndarray:
    """The lines of two numbers."""
    numeric = re.compile(r"^\s*-?\d+(\.\d*)?\s+-?\d+(\.\d*)?\s*$")
    rows = [line.split() for line in text.splitlines() if numeric.match(line)]
    return np.array([[float(v) for v in r] for r in rows], dtype=float)


def _labels(text: str) -> np.ndarray:
    lines = text.splitlines()
    start = next(j for j, line in enumerate(lines) if line.startswith("---")) + 1
    return np.array([int(v) for v in lines[start:] if v.strip()], dtype=np.int64)


def _zip_member(blob: bytes, member: str) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.read(member).decode("latin-1")


def franti(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``s1``-``s4`` or ``a1``-``a3``: the points (``n x 2``), the ground-truth
    label of each (``0 .. k-1``), and the ground-truth centroids (``k x 2``),
    the mean of each label's points. (The A-sets' codebooks are binary; the
    S-sets' generating centres differ from their labels' means by the
    sampling, a few thousandths of the spread.)"""
    points = _points(_cached(f"{name}.txt", FRANTI.format(name=f"{name}.txt")).decode("ascii"))
    if name.startswith("s"):
        blob = _cached("s-originals.zip", FRANTI.format(name="s-originals.zip"))
        labels = _labels(_zip_member(blob, f"{name}-label.pa"))
    else:
        blob = _cached("a-gt-pa.zip", FRANTI.format(name="a-gt-pa.zip"))
        labels = _labels(_zip_member(blob, f"{name}-ga.pa"))
    labels = labels - labels.min()
    assert len(labels) == len(points), (name, len(labels), len(points))
    k = int(labels.max()) + 1
    centres = np.array([points[labels == j].mean(axis=0) for j in range(k)])
    return points, labels, centres


def franti_or_skip(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _or_skip(lambda: franti(name), f"the {name} clustering benchmark")


# ---------------------------------------------------------------------------
# TCPD


def tcpd(name: str) -> tuple[np.ndarray, dict[str, list[int]]]:
    """One TCPD series (its first dimension) and its annotations, the change
    points each annotator marked, by annotator id."""
    series = json.loads(
        _cached(f"tcpd-{name}.json", TCPD.format(path=f"datasets/{name}/{name}.json"))
    )
    annotations = json.loads(_cached("tcpd-annotations.json", TCPD.format(path="annotations.json")))
    raw = series["series"][0]["raw"]
    return np.array([np.nan if v is None else float(v) for v in raw]), annotations[name]


def tcpd_or_skip(name: str) -> tuple[np.ndarray, dict[str, list[int]]]:
    return _or_skip(lambda: tcpd(name), f"the TCPD series {name}")


# ---------------------------------------------------------------------------
# LIBSVM


def libsvm(path: str, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """A LIBSVM-format set as a dense ``X`` (``n x n_features``) and ``y``.
    ``path`` is under the datasets root, such as ``regression/abalone``."""
    name = f"libsvm-{path.replace('/', '-')}"
    text = _cached(name, LIBSVM.format(path=path), _lenient_tls()).decode("ascii")
    rows = [line.split() for line in text.splitlines() if line.strip()]
    y = np.array([float(r[0]) for r in rows])
    x = np.zeros((len(rows), n_features))
    for i, r in enumerate(rows):
        for item in r[1:]:
            j, v = item.split(":")
            x[i, int(j) - 1] = float(v)
    return x, y


def libsvm_or_skip(path: str, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    return _or_skip(lambda: libsvm(path, n_features), f"the LIBSVM set {path}")
