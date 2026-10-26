"""The gradient models cut every way give one run, to the bit (hard rule 3).

``sgd``, ``pa`` and ``ftrl`` keep more per-row state than a regression with
sufficient statistics -- a scaler and its warm-up count, AdaGrad's
accumulators, ``inv_scaling``'s weight sum, a projection onto a box and a sum,
FTRL's ``z`` and ``n`` under a half-life -- and each loss and schedule takes
its own path through it. Their chunking and save tests ran one setting each
(the squared loss at 37-row chunks, AdaGrad's save, ``pa1``'s), so a path
that read a chunk's edge or dropped a field from the state would have passed
on every other setting (task 278, batch 20; the inventory's D1-d).

Here every setting is cut every way: 1, 7 and 600 chunks of a 600-row stream,
and a save and load at three rows -- inside the scaler's warm-up, on the row
after the one whose update ends it, and on the row a gap past ``gap_cap``
opens -- across

- ``sgd``: every loss under every schedule, standardized and not;
- ``pa``: every mode, standardized and not, unconstrained and under a box
  and a sum;
- ``ftrl``: both losses under a half-life, with an L1.

The stream is the gradient oracles' (``test_oracles_gradient_paths._stream``):
an irregular clock with two gaps past ``gap_cap`` and 25 rows on a clock
already seen, two targets with scattered and blocked nulls, weights between
0.5 and 1.5 with a twentieth of them 0, the first row among them, and three
rows skipped for a null feature. Standardized legs read the features at
levels and in units of their own, so the scaler maps something.

Every output field is compared bit for bit, ``coef`` on every row
(``coef_every=0``), and so are the state's bytes: after the run, after a
reload (a re-save is the same bytes) and after the reloaded bank finishes.
Each check prints its evidence that the hard path ran: the save inside the
warm-up is inside it by the state's own count, and the box binds.

The grid is extended (CI runs it on every leg, ``scripts/gate.sh
--extended`` before every push); one case per model is in the essentials.
Every case's id starts with its model, so ``-k sgd`` and the like take one
model's cells.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_oracles_gradient_paths import FEATURES, MAX_DCLOCK, TARGETS, _scaled, _stream

TIER = "mixed"

#: 600 rows, so 1-row chunks are 600 chunks.
N = 600
#: Chunk lengths: one chunk, seven (86 rows, the last 84) and 600.
CHUNKS = (N, -(-N // 7), 1)
#: The row the second gap past ``gap_cap`` opens (``_stream``'s ``dt[290]``).
GAP_ROW = 290
#: Kish's count of the rows the scaler learns before the fit is held in the
#: caller's units (`online_core::WARMUP_ROWS`).
WARMUP_ROWS = 22.0

#: The grid's tier. Not `oracle_grid`, CI's sharded job for heavy grids
#: against a third-party library: nothing here is an oracle, and the whole
#: grid took 7.4 s in one process here (114 cells, 2026-10-26; with
#: ``sgd(loss="huber")``'s 18, 17.8 s at a load of 29). The extended tier
#: runs it on each of CI's legs, so every platform's arithmetic is cut every
#: way, which one sharded Linux job would not do. Each id leads with its
#: model, so ``-k sgd``, ``-k pa-`` and ``-k ftrl`` split it by model.
GRID = pytest.mark.extended(reason="a grid: every loss, schedule, mode and constraint, cut 6 ways")

#: Each loss with the stream it fits and what it needs beside it.
LOSSES = {
    "squared": ("linear", {}),
    "quantile": ("linear", {"quantile": 0.8}),
    "epsilon_insensitive": ("linear", {"eps": 0.2}),
    "poisson": ("poisson", {}),
    "logistic": ("logistic", {}),
    # Registered once task 277's refit of each target's first rows merged.
    "huber": ("linear", {"huber_delta": 0.3}),
}
SCHEDULES = {
    "constant": {"learning_rate": 0.05},
    "inv_scaling": {"learning_rate": 0.05, "power": 0.25},
    "adagrad": {"learning_rate": 0.2},
}
MODES = ("pa", "pa1", "pa2")
#: A box the slopes run into and a sum they must meet: the targets' slopes
#: are ``[1, -0.5, 0.25]`` and ``[0, 0.8, -0.6]`` on the plain features
#: (``[0.4, -0.3, 0.1]`` and ``[0, 0.3, -0.2]`` on the Poisson stream's log
#: scale, and about ``[1.5, -1, 0.5]`` on the logistic one's), and 100 times
#: larger on the standardized legs' ``x1``.
BOX_AND_SUM = {"coef_min": -0.2, "coef_max": 0.3, "coef_sum": 0.2}
#: Beside each loss, schedule and mode: the plain spec, the spec without an
#: intercept (whose scaler is a root mean square with no centre) and the
#: spec under a box and a sum, projected after every step.
VARIANTS = {
    "plain": {},
    "no-intercept": {"fit_intercept": False},
    "box-and-sum": BOX_AND_SUM,
}

#: What every spec shares: the stream's clock, gaps, weights and targets,
#: and ``coef`` on every row, so a chunk's end writes no extra one.
COMMON = dict(
    targets=TARGETS,
    features=FEATURES,
    clock="t",
    gap_cap=MAX_DCLOCK,
    weight="w",
    half_life=50.0,
    min_weight=5.0,
    coef_every=0,
)


def _sgd(loss: str, schedule: str, standardize: bool, variant: str) -> tuple[dict, str, bool]:
    kind, extra = LOSSES[loss]
    spec = po.spec.sgd(
        "m",
        loss=loss,
        schedule=schedule,
        standardize=standardize,
        l2=0.01,
        **extra,
        **SCHEDULES[schedule],
        **VARIANTS[variant],
        **COMMON,
    )
    return spec, kind, standardize


def _pa(mode: str, standardize: bool, variant: str) -> tuple[dict, str, bool]:
    kw = {} if mode == "pa" else {"c": 0.3}  # "pa" reads no `c`, and refuses one
    spec = po.spec.pa(
        "m", mode=mode, eps=0.05, standardize=standardize, **kw, **VARIANTS[variant], **COMMON
    )
    return spec, "linear", standardize


def _ftrl(loss: str, variant: str) -> tuple[dict, str, bool]:
    kw = {"alpha": 0.3, "beta": 0.5, "l1": 0.5, "l2": 0.5}
    if loss == "squared":
        kw = {"alpha": 0.2, "l1": 0.2}
    common = COMMON | {"half_life": 150.0}
    spec = po.spec.ftrl("m", loss=loss, **kw, **VARIANTS[variant], **common)
    return spec, ("linear" if loss == "squared" else "logistic"), False


def _std(standardize: bool) -> str:
    return "standardized" if standardize else "raw"


def _cells(losses: dict) -> list[tuple[str, tuple]]:
    """Every cell, its id led by its model."""
    cells = []
    for loss in losses:
        for schedule in SCHEDULES:
            for std in (True, False):
                for v in VARIANTS:
                    cid = f"sgd-{loss}-{schedule}-{_std(std)}-{v}"
                    cells.append((cid, ("sgd", loss, schedule, std, v)))
    for mode in MODES:
        for std in (True, False):
            for v in VARIANTS:
                cells.append((f"pa-{mode}-{_std(std)}-{v}", ("pa", mode, std, v)))
    for loss in ("logistic", "squared"):
        for v in ("plain", "no-intercept"):  # ftrl takes no box
            cells.append((f"ftrl-{loss}-{v}", ("ftrl", loss, v)))
    return cells


#: The essentials' cells, one a model.
ESSENTIAL = {
    "sgd-squared-adagrad-standardized-box-and-sum",
    "pa-pa1-standardized-box-and-sum",
    "ftrl-logistic-plain",
}


def _cases() -> list:
    cells = _cells(LOSSES)
    assert {cid for cid, _ in cells} >= ESSENTIAL
    return [
        pytest.param(cell, id=cid, marks=() if cid in ESSENTIAL else GRID) for cid, cell in cells
    ]


def _build(cell: tuple) -> tuple[dict, str, bool]:
    model, *args = cell
    return {"sgd": _sgd, "pa": _pa, "ftrl": _ftrl}[model](*args)


def _frame(kind: str, standardize: bool) -> pl.DataFrame:
    df = _stream(20, kind=kind, dup=True, n=N)
    return _scaled(df) if standardize else df


def _switch_row(df: pl.DataFrame) -> int:
    """The row whose scaler update brings Kish's count of the learned
    weights to `WARMUP_ROWS`: the rows the bank learns from are those with
    every feature present, each at its weight, undecayed (`warmup.rs`)."""
    learned = df.select(pl.all_horizontal(pl.col(f).is_not_null() for f in FEATURES)).to_series()
    w = np.where(learned.to_numpy(), df["w"].to_numpy(), 0.0)
    sw, sw2 = np.cumsum(w), np.cumsum(w * w)
    kish = np.divide(sw * sw, sw2, out=np.zeros_like(sw), where=sw2 > 0.0)
    return int(np.flatnonzero(kish >= WARMUP_ROWS)[0])


def _splits(df: pl.DataFrame) -> dict[str, int]:
    """The three rows a save is taken before: inside the warm-up, on the row
    after the switch, and on the gap's row."""
    switch = _switch_row(df)
    return {"inside the warm-up": switch // 2, "after the switch": switch + 1, "gap": GAP_ROW}


def _warmups(bank: po.ModelBank) -> list[bool]:
    """Each scaler's ``switched`` flag, read from the state."""
    found: list[bool] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "warmup" in node:
                found.append(bool(node["warmup"]["switched"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(json.loads(bank.write_json()))
    return found


def _columns(out: pl.DataFrame) -> pl.DataFrame:
    return out.select("m").unnest("m")


def _float_bits(s: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """A float column's null mask and its values' bits (a null as 0)."""
    nulls = s.is_null().to_numpy()
    bits = s.fill_null(0.0).to_numpy().astype(np.float64).view(np.uint64)
    return nulls, np.where(nulls, np.uint64(0), bits)


def _same_bits(want: pl.DataFrame, got: pl.DataFrame, what: str) -> None:
    """Every column of ``got`` is ``want``'s to the bit: a float's bits (so
    ``-0.0`` is not ``0.0`` and a NaN keeps its payload), a list's lengths
    and its items' bits, any other column by value; the first difference is
    named with its row."""
    assert want.columns == got.columns, f"{what}: columns {want.columns} != {got.columns}"
    assert want.height == got.height, f"{what}: {want.height} rows != {got.height}"
    for c in want.columns:
        a, b = want[c], got[c]
        assert a.dtype == b.dtype, f"{what}: {c} is {a.dtype} and {b.dtype}"
        if a.dtype == pl.List(pl.Float64):
            la, lb = (s.list.len().fill_null(-1).to_numpy() for s in (a, b))
            na, nb = a.is_null().to_numpy(), b.is_null().to_numpy()
            bad = np.flatnonzero((la != lb) | (na != nb))
            assert bad.size == 0, f"{what}: {c} is shaped differently at row {bad[0]}"
            ia, ib = a.explode(), b.explode()
            row = np.repeat(np.arange(a.len()), np.where(na, 1, np.maximum(la, 1)))
        elif a.dtype == pl.Float64:
            ia, ib, row = a, b, np.arange(a.len())
        else:
            assert a.equals(b, null_equal=True), f"{what}: {c} differs"
            continue
        (ma, va), (mb, vb) = _float_bits(ia), _float_bits(ib)
        bad = np.flatnonzero((ma != mb) | (va != vb))
        assert bad.size == 0, (
            f"{what}: {c} differs at row {row[bad[0]]}: {ia[int(bad[0])]!r} != {ib[int(bad[0])]!r}"
            f" ({bad.size} values)"
        )


def _binds(fields: pl.DataFrame) -> int:
    """How many slopes, over every row, sit exactly on the box's edge."""
    rows = [r for r in fields["coef"].to_list() if r is not None]
    coef = np.array(rows, float).reshape(len(rows), len(TARGETS), len(FEATURES) + 1)
    slopes = coef[:, :, 1:]
    return int(np.isin(slopes, [BOX_AND_SUM["coef_min"], BOX_AND_SUM["coef_max"]]).sum())


@pytest.mark.parametrize("cell", _cases())
def test_chunks_and_saves_give_one_run(cell):
    spec, kind, standardize = _build(cell)
    df = _frame(kind, standardize)
    whole = po.ModelBank([spec])
    want = _columns(whole.fit_predict(df))
    state = whole.save_bytes()
    assert want["coef"].null_count() < want.height // 2, "coef is reported too little"
    for n in TARGETS:
        assert want[f"pred_{n}"].is_not_null().sum() > N // 2, f"pred_{n} is scored too little"

    for chunk in CHUNKS:
        bank = po.ModelBank([spec])
        got = pl.concat([bank.fit_predict(df.slice(i, chunk)) for i in range(0, N, chunk)])
        _same_bits(want, _columns(got), f"{chunk}-row chunks")
        assert bank.save_bytes() == state, f"{chunk}-row chunks moved the state"

    splits = _splits(df)
    for where, split in splits.items():
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, split))
        saved = a.save_bytes()
        if standardize:
            # The hard path: the save is inside the warm-up, or past it, by
            # the state's own count, on each target's scaler.
            flags = _warmups(a)
            assert flags, "no scaler in the state"
            assert all(flags) == (where != "inside the warm-up"), (where, split, flags)
            assert not any(flags) or where != "inside the warm-up", (where, split, flags)
        b = po.ModelBank.load_bytes(saved, specs=[spec])
        assert b.save_bytes() == saved, f"a reload at row {split} ({where}) re-saved other bytes"
        rest = _columns(b.fit_predict(df.slice(split)))
        _same_bits(want.slice(split), rest, f"a save at row {split} ({where})")
        assert b.save_bytes() == state, f"a save at row {split} ({where}) moved the final state"
    if standardize:
        assert all(_warmups(whole)), "the scaler never switched"

    if spec["model"].get("coef_min") is not None:
        bound = _binds(want)
        print(f"{cell}: {bound} slopes on the box's edge")
        assert bound > 0, "the box never binds"


def test_the_splits_are_where_they_say():
    """The save points are distinct and ordered on every stream the grid
    reads, the warm-up's inside it with rows on either side."""
    for kind in ("linear", "poisson", "logistic"):
        for std in (True, False):
            s = _splits(_frame(kind, std))
            assert 0 < s["inside the warm-up"] < s["after the switch"] - 1 < s["gap"] < N, s
