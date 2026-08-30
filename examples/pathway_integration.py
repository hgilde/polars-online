"""Run a polars-online ModelBank inside a Pathway pipeline (ENHANCEMENTS E26).

The division of labour, and the reason this is an example rather than a
dependency:

* **Pathway** does the streaming plumbing — connectors, event-time alignment,
  windowing, late arrivals, exactly-once persistence — in its own Rust engine.
  It has no online regression.
* **polars-online** does the model. It expects a frame that is already aligned
  and ordered, and keeps O(state) per stream.

So they compose cleanly: Pathway hands us ordered batches, we hand back
predictions. Nothing here duplicates the other side.

**Licensing.** Pathway is distributed under the BSL and classified
"Other/Proprietary"; this project is Apache-2.0. Pathway is therefore *not* a
dependency, not even a dev one — this file imports it lazily and explains
itself if it is absent. Install it yourself, under its licence, if you want to
run this.

    uv run --with pathway python examples/pathway_integration.py
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

import polars_online as po

SPEC = po.spec.ewridge(
    "ridge",
    targets=["y"],
    features=["x0", "x1"],
    halflife=500.0,
    min_periods=20.0,
    emit_sigma=True,
    emit_drift=True,
)


def _synthetic(n: int = 5_000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    return pl.DataFrame(
        {
            "t": np.arange(float(n)),
            "x0": x0,
            "x1": x1,
            "y": 2 * x0 - 0.5 * x1 + 0.3 * rng.standard_normal(n),
        }
    )


class BankOperator:
    """A stateful operator: one `ModelBank`, fed ordered batches.

    The bank *is* the operator state. Two properties make it safe to run inside
    a streaming engine, and both are enforced by this project's own tests:

    * chunking never changes the numbers, so however the engine batches the
      stream, the output is the same;
    * `save_bytes()` / `load_bytes()` round-trip exactly, so the engine's
      snapshotting can checkpoint the model along with everything else.
    """

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.bank = po.ModelBank([spec])

    def __call__(self, batch: pl.DataFrame) -> pl.DataFrame:
        return self.bank.fit_predict(batch)

    # --- what a Pathway persistence hook would call ---
    def snapshot(self) -> bytes:
        return self.bank.save_bytes()

    def restore(self, blob: bytes) -> None:
        self.bank = po.ModelBank.load_bytes(blob, specs=[self.spec])


def run_without_pathway() -> pl.DataFrame:
    """The same operator driven by plain batches, so the example is runnable
    (and testable) whether or not Pathway is installed."""
    df = _synthetic()
    op = BankOperator(SPEC)
    out = pl.concat([op(df.slice(i, 500)) for i in range(0, df.height, 500)])

    # Checkpoint/restore mid-stream, the way an engine would.
    blob = op.snapshot()
    op.restore(blob)
    return out


def run_with_pathway() -> None:
    """Sketch of the same thing as a Pathway pipeline.

    Kept as a docstring-with-code rather than an executed path, because it
    needs a Pathway installation and a running input connector. The shape is
    the point: Pathway owns ingestion and ordering, the operator owns the model.
    """
    import pathway as pw  # noqa: PLC0415  (optional, licence-separated)

    op = BankOperator(SPEC)

    class Schema(pw.Schema):
        t: float
        x0: float
        x1: float
        y: float

    # Pathway supplies ordered rows; `fit_predict` is called per batch, and the
    # bank carries state across batches exactly as it does for `po.run`.
    table = pw.io.csv.read("./ticks", schema=Schema, mode="streaming")

    @pw.udf
    def _predict(t: float, x0: float, x1: float, y: float) -> float:
        frame = pl.DataFrame({"t": [t], "x0": [x0], "x1": [x1], "y": [y]})
        out = op(frame)
        value = out["ridge"].struct.field("pred_y").to_list()[0]
        return float("nan") if value is None else value

    result = table.select(pred=_predict(table.t, table.x0, table.x1, table.y))
    pw.io.csv.write(result, "./predictions")
    pw.run()


def main() -> int:
    out = run_without_pathway()
    preds = out["ridge"].struct.field("pred_y")
    print(f"rows: {out.height}, predictions: {preds.is_not_null().sum()}")
    print(out.select("t", "y", "ridge").tail(3))
    try:
        import pathway  # noqa: F401, PLC0415
    except ImportError:
        print(
            "\npathway is not installed, so only the plain-batch path ran.\n"
            "Install it under its own licence (BSL) to try run_with_pathway():\n"
            "    uv run --with pathway python examples/pathway_integration.py"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
