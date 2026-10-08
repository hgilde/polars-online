"""The README's first fit prints five tables: its input, the fit over every
row, the betas the saved state holds, the local fit's betas hour by hour, and
the forecast of a target that looks ahead. A printed table is a claim like any
other (docs/WRITING.md §3), so this runs the README's block, as the README
test runs every block, and holds each shown value to the one the code returns:
a number to half a unit of its last shown digit, a time to the second, a text
or a null exactly. The claims the prose around the tables makes are held here
too, as `test_refresh_time.py` holds the refresh-time grid.
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from test_production_hardening import _readme_namespace

TIER = "essential"

README = Path(__file__).resolve().parent.parent / "README.md"
SECTION = "### A first fit"
A = pl.col("stock_id") == "A"

#: What each table after the block shows, in the order the README prints them.
#: A table's header names the columns it shows of the frame.
SHOWN = [
    ("the input's first rows", lambda ns: ns["prices"].head(6)),
    ("stock A's first rows of the fit over every row", lambda ns: ns["flat"].filter(A).head(5)),
    ("the betas in the saved state", lambda ns: ns["saved"]),
    (
        "stock A's local betas each hour from 10:30",
        lambda ns: ns["betas"].filter(A).gather_every(1200, offset=1200),
    ),
    (
        "stock A's forecast every five minutes from 9:30",
        lambda ns: ns["forecast"].filter(A).gather_every(100).head(5),
    ),
]
SEPARATOR = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")


def _first_fit() -> tuple[str, list[list[list[str]]]]:
    """The first fit's one python block, and each Markdown table after it as
    rows of cells, its header first."""
    text = README.read_text(encoding="utf-8")
    start = text.index(f"\n{SECTION}\n")
    end = text.index("\n### ", start + len(SECTION))
    section = text[start:end]
    blocks = re.findall(r"```python\n(.*?)\n```", section, re.S)
    assert len(blocks) == 1, f"{SECTION} should hold one python block, not {len(blocks)}"
    after = section[section.index(blocks[0]) + len(blocks[0]) :]
    tables: list[list[list[str]]] = []
    rows: list[list[str]] = []
    for line in [*after.splitlines(), ""]:
        if line.startswith("|"):
            if not SEPARATOR.match(line):
                rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif rows:
            tables.append(rows)
            rows = []
    return blocks[0], tables


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """The block run once, in the namespace and the directory every README
    block runs in, and that directory, which holds the state it saved."""
    code, tables = _first_fit()
    tmp = tmp_path_factory.mktemp("first_fit")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        ns = _readme_namespace(tmp)
        exec(compile(code, f"{README.name}: {SECTION}", "exec"), ns)
    return ns, tables, tmp


def _agrees(shown: str, value: object) -> bool:
    if shown == "null" or value is None:
        return shown == "null" and value is None
    if isinstance(value, datetime):
        return value.strftime("%H:%M:%S") == shown
    if isinstance(value, int | float):
        digits = len(shown.partition(".")[2])
        # 1e-9 of slack, so a value a platform's last bit puts on a rounding
        # edge cannot flip; every value shown sits at least 0.009 of a unit
        # away from one.
        return abs(float(shown.replace("−", "-")) - value) <= 0.5 * 10**-digits + 1e-9
    return str(value) == shown


def test_every_table_the_first_fit_prints_is_what_its_code_returns(ran):
    ns, tables, _ = ran
    assert len(tables) == len(SHOWN), (
        f"{SECTION} prints {len(tables)} tables and this test checks {len(SHOWN)}: "
        "a table added or removed needs its entry in SHOWN"
    )
    for (what, frame_of), (header, *rows) in zip(SHOWN, tables, strict=True):
        frame = frame_of(ns).select(header)
        assert frame.height == len(rows), f"{what}: {len(rows)} rows shown, {frame.height} computed"
        for i, (cells, values) in enumerate(zip(rows, frame.iter_rows(), strict=True), 1):
            for column, shown, value in zip(header, cells, values, strict=True):
                where = f"{what}, row {i}, {column}"
                assert _agrees(shown, value), f"{where}: shows {shown}, is {value}"


def test_the_first_fits_prose_holds(ran):
    """What the paragraphs around the tables say, beyond the numbers shown."""
    ns, _, tmp = ran
    po = ns["po"]
    n = ns["prices"].height
    true_a = np.linspace(0.2, 0.8, n)  # the block's beta on signal_a, row by row

    # "A stock's first three rows have too little behind them, so `pred_ret` is
    # null and `withheld_reason` says why."
    for _, rows in ns["flat"].group_by("stock_id", maintain_order=True):
        assert rows["pred_ret"].head(4).is_null().to_list() == [True, True, True, False]
        assert rows["withheld_reason"].head(3).null_count() == 0

    # "The state saved at 15:30 holds each stock's fit, near the average of the
    # true betas until then."
    saved = ns["saved"]
    assert saved["group"].to_list() == ["A", "B", "C"]
    assert (saved["signal_a"] - true_a[:21_600].mean()).abs().max() < 0.05
    assert (saved["signal_b"] + 0.2).abs().max() < 0.05

    # "`served` scores the rows from 15:30 with that state, and learns nothing
    # from them": every row is scored with the weight the state was saved with.
    served = ns["served"].unnest("whole")
    assert served["ts"].min() == datetime(2024, 1, 2, 15, 30)
    assert served["weight_sum"].unique().to_list() == [21_600 / 3]

    # "To keep learning from new rows instead, call
    # `.online.fit_predict(load_state="bank.state")` on a query over them."
    kept = (
        ns["prices"]
        .tail(1_800)
        .lazy()
        .online.fit_predict(load_state=str(tmp / "bank.state"))
        .collect()
        .unnest("whole")
    )
    by_stock = kept.group_by("stock_id", maintain_order=True).agg(pl.col("weight_sum"))
    for weights in by_stock["weight_sum"]:
        assert weights.to_list() == [21_600 / 3 + i for i in range(1_800 // 3)]

    # "Step 3 follows the recent rows, with a clock and the `gap_cap` a clock
    # requires."
    with pytest.raises(ValueError, match="gap_cap is required when clock is given"):
        po.spec.ewridge(
            "local",
            targets=["ret"],
            features=["signal_a", "signal_b"],
            group="stock_id",
            clock="ts",
            half_life="10m",
            coef_every=0,
        )

    # "Each hour from 10:30, stock A's beta on signal_a climbs with the true one."
    hourly = ns["betas"].with_row_index("i").filter(A).gather_every(1200, offset=1200)
    beta = hourly["coef_ret_signal_a"]
    assert (beta.diff().drop_nulls() > 0).all()
    assert (beta - true_a[hourly["i"].to_numpy()]).abs().max() < 0.05

    # "With `embargo="10m"`, each row is scored where it sits, before its
    # target is known, and learned from ten minutes later. So `ahead` learns
    # nothing from 9:30 to 9:40, and `resid_fwd_ret` is null on every row."
    forecast = ns["forecast"]
    first_ten = forecast.filter(pl.col("ts") <= datetime(2024, 1, 2, 9, 40))
    assert first_ten["weight_sum"].max() == 0
    assert forecast.filter(pl.col("ts") == datetime(2024, 1, 2, 9, 45))["weight_sum"].min() > 0
    assert forecast["resid_fwd_ret"].null_count() == forecast.height

    # "`fit_predict` requires an `embargo` of at least the target's
    # `window_size`; otherwise it refuses the spec, so for a shorter embargo
    # use `ModelBank.fit`, which keeps no predictions."
    short = po.spec.ewridge(
        "short",
        targets=[ns["fwd_ret"]],
        features=["signal_a", "signal_b"],
        group="stock_id",
        clock="ts",
        half_life="10m",
        gap_cap="5m",
        embargo="5m",
    )
    with pytest.raises(ValueError, match="needs an embargo of at least 10m"):
        ns["prices"].lazy().online.fit_predict([short]).collect()
    fitted = po.ModelBank([short])
    assert fitted.fit(ns["prices"]) is None
    assert (fitted.summary()["weight_sum"] > 0).all()
