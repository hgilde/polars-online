"""``docs/HIGHLIGHTS.md``: the four tasks of the highlights page, held to the code.

Each task on the page makes up its own data with a planted answer, writes
it to parquet files and scans them, and prints what the library found
(docs/WRITING.md §3, "A capability is shown by a recipe"). So the page's
python blocks run here in order, in one namespace and one directory, as a
reader runs them, and a block followed by a ```text block is held to it
with ``tests/doc_recipes.py``'s ``differences``. The page highlights the
Python API alone (the user, 2026-11-08: "we are not highlighting the
command line tool only the python api"), which a test holds too. The
README's section on the page links it (PLAN task 365).
"""

from __future__ import annotations

import contextlib
import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

import polars_online as po
from doc_recipes import differences, fenced_blocks

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "HIGHLIGHTS.md"


def test_every_block_runs_in_order_and_prints_what_the_page_shows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the blocks write their files where they run
    ns: dict[str, object] = {"__name__": "__recipe__"}
    blocks = fenced_blocks(PAGE)
    problems: list[str] = []
    ran = 0
    for j, (kind, line, code) in enumerate(blocks):
        if kind != "python":
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, f"HIGHLIGHTS.md:{line}", "exec"), ns)
        ran += 1
        after = blocks[j + 1] if j + 1 < len(blocks) else None
        if after is not None and after[0] == "text":
            problems += [
                f"HIGHLIGHTS.md:{line}:\n{p}" for p in differences(after[2], buf.getvalue())
            ]
    assert ran >= 5, ran  # four tasks, one of them in two blocks
    assert not problems, "\n\n".join(problems)


def test_the_page_shows_the_python_api_alone():
    """No shell block, and no run of the ``polars-online`` program."""
    blocks = fenced_blocks(PAGE)
    assert {kind for kind, _, _ in blocks} <= {"python", "text"}, {k for k, _, _ in blocks}
    text = PAGE.read_text(encoding="utf-8")
    for word in ("polars-online --", "command line", "command-line", "TOML"):
        assert word not in text, word


def test_every_capability_row_links_a_step_and_a_guide():
    """The index at the end names, for each capability, the task that shows
    it and the guide that covers it: two links a row."""
    text = PAGE.read_text(encoding="utf-8")
    index = text.split("\n## Capabilities\n", 1)[1]
    rows = [
        r for r in index.splitlines() if r.startswith("| ") and not r.startswith("| capability")
    ]
    assert len(rows) >= 16, len(rows)
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert "](#" in cells[2], row
        assert "](" in cells[3] and "](#" not in cells[3], row


def test_the_readme_section_links_the_page():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## What you can do with it\n", 1)[1].split("\n## ", 1)[0]
    assert "](docs/HIGHLIGHTS.md)" in section
    assert "](docs/HIGHLIGHTS.md#capabilities)" in section


def _signal_block() -> str:
    """The signal task's first block: its data, its windows, both stages."""
    text = PAGE.read_text(encoding="utf-8")
    section = text.split("\n## Build a forward-looking signal and resume it\n", 1)[1]
    blocks = fenced_blocks(PAGE)
    start = len(text) - len(section)
    first = text[:start].count("\n") + 1
    return next(code for kind, line, code in blocks if kind == "python" and line > first)


def test_the_signal_task_learns_nothing_from_the_future(tmp_path, monkeypatch):
    """The signal task's target reads the mid five minutes ahead, and its
    second stage learns each row only once the embargo has passed. So every
    feature, prediction and coefficient at or before a time T is the same
    whatever the rows after T hold. Run the page's own block twice, the
    second time with every input after the cut replaced, and fit its second
    stage (the page's ``common``) on each: before the cut nothing may move,
    while the target, which reads ahead, must. Without the embargo the
    predictions before the cut move, so the check can see a leak."""
    code = _signal_block()
    make_data, rest = code.split("\n# 2. ", 1)
    rest = "# 2. " + rest
    cut = datetime(2024, 1, 3, 12)  # where the page saves its state
    alt = np.random.default_rng(99)

    def run(where: Path, replace_after_cut: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
        where.mkdir()
        monkeypatch.chdir(where)
        ns: dict[str, object] = {"__name__": "__recipe__"}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(make_data, "HIGHLIGHTS.md signal data", "exec"), ns)
            if replace_after_cut:
                for f in sorted(Path("trades").glob("*.parquet")):
                    df = pl.read_parquet(f)
                    after = (df["ts"] > cut).to_numpy()
                    m = int(after.sum())
                    if m:
                        new = {
                            c: df[c].to_numpy().copy()
                            for c in ("flow", "mid", "d", "x0", "x1", "x2")
                        }
                        for c in ("flow", "x0", "x1", "x2"):
                            new[c][after] = alt.standard_normal(m)
                        new["d"][after] = 5.0 * alt.standard_normal(m)
                        new["mid"][after] = 100.0 + 3.0 * alt.standard_normal(m).cumsum()
                        df.with_columns(pl.Series(c, v) for c, v in new.items()).write_parquet(f)
            exec(compile(rest, "HIGHLIGHTS.md signal stages", "exec"), ns)
        raw = ns["common"]
        assert isinstance(raw, dict), raw
        common = dict(raw)
        assert common.get("embargo") == "5m", common
        features = pl.scan_parquet("features.parquet")

        def fit(**spec: object) -> pl.DataFrame:
            out = features.online.fit_predict(
                [po.spec.huber("robust", **spec), po.spec.ewridge("ridge", **spec)]
            ).collect()
            return out.select(
                "ts",
                "stretch",
                "resid_d",
                "fwd",
                *(
                    pl.col(m).struct.field(f).alias(f"{m}_{f}")
                    for m in ("robust", "ridge")
                    for f in ("pred_fwd", "coef")
                ),
            )

        embargoed = fit(**common)
        common.pop("embargo")
        return embargoed, fit(**common)

    base, base_free = run(tmp_path / "base", False)
    moved, moved_free = run(tmp_path / "moved", True)
    past = pl.col("ts") <= cut
    a = base.filter(past)
    b = moved.filter(past)
    assert a.height > 1_000 and a.height == b.height, (a.height, b.height)
    for column in a.columns:
        if column == "fwd":
            continue
        assert a[column].equals(b[column]), f"{column} before the cut moved with the rows after it"
    # The target reads ahead, so in the five minutes before the cut it must move.
    near = past & (pl.col("ts") > cut - timedelta(minutes=5))
    assert not base.filter(near)["fwd"].equals(moved.filter(near)["fwd"])
    # The control: with no embargo the fits learn targets that read past the cut.
    ua = base_free.filter(past)
    ub = moved_free.filter(past)
    assert not ua["robust_pred_fwd"].equals(ub["robust_pred_fwd"])
    assert not ua["ridge_pred_fwd"].equals(ub["ridge_pred_fwd"])
