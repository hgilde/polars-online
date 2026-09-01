"""Memory and crash safety across the FFI boundary.

Two copies of Polars share this process: py-polars' and the one statically
linked into our extension. Data crosses on the Arrow C Data Interface, where a
`SeriesExport` carries a `release` callback back into the binary that produced
it. That design is what keeps the arrangement safe, and these tests are what
check the design is actually being honoured — every failure mode here is either
a leak or a hard crash, neither of which any other test in the suite would
notice.

**Why the assertion is "plateaus" and not "does not grow."** Allocators do not
return pages eagerly, rayon spawns worker threads lazily, and Polars caches the
loaded plugin. Measured here, `.over()` costs a one-time ~6 MB and is then flat
forever, while a real leak grows without bound. So the primitive compares the
*later* blocks against each other: a step is fine, a slope is not.
"""

import gc
import os
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

import polars_online as po

psutil = pytest.importorskip("psutil")
PROC = psutil.Process(os.getpid())


def rss_kb() -> float:
    return PROC.memory_info().rss / 1024


def assert_plateaus(fn, *, blocks=4, per_block=120, warmup=40, kb_per_iter=4.0):
    """Run `fn` in blocks and require RSS growth to flatten.

    The first block absorbs one-off costs — thread stacks, arena growth, the
    plugin's library cache. Later blocks are compared against each other, so a
    step passes and a slope fails.
    """
    for _ in range(warmup):
        fn()
    gc.collect()

    marks = []
    for _ in range(blocks):
        for _ in range(per_block):
            fn()
        gc.collect()
        marks.append(rss_kb())

    # Growth per iteration across everything after the first block.
    tail = marks[1:]
    grown = tail[-1] - tail[0]
    iters = per_block * (len(tail) - 1)
    per_iter = grown / iters
    assert per_iter < kb_per_iter, (
        f"RSS still climbing after the first block: {per_iter:.2f} KB/iter "
        f"over {iters} iterations (marks, KB: {[round(m) for m in marks]}). "
        "A plateau is expected; a slope means something is not being released."
    )


rng = np.random.default_rng(0)


def frame(n=1500):
    d = pl.DataFrame({"x0": rng.standard_normal(n), "x1": rng.standard_normal(n)})
    return d.with_columns(y=pl.col("x0") * 2)


SPEC = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], halflife=50.0, min_periods=2.0)


class TestNothingLeaksAcrossTheBoundary:
    def test_repeated_fit_predict(self):
        bank, df = po.ModelBank([SPEC]), frame()
        assert_plateaus(lambda: bank.fit_predict(df))

    def test_bank_churn(self):
        """A new bank per iteration: Rust-side state must be dropped with the
        Python object, not accumulated in the extension."""
        df = frame()
        assert_plateaus(lambda: po.ModelBank([SPEC]).fit_predict(df))

    def test_expression_plugin(self):
        df = frame()
        assert_plateaus(
            lambda: df.with_columns(
                pl.col("y").online.ewridge(features=["x0", "x1"], halflife=50.0, min_periods=2.0)
            )
        )

    def test_plugin_over_groups(self):
        """The one case with a real one-time step (~6 MB of thread stacks and
        arena), which is exactly why the primitive tolerates a step."""
        df = frame().with_columns(g=pl.Series(np.arange(1500) % 50))
        assert_plateaus(
            lambda: df.with_columns(
                pl.col("y")
                .online.ewridge(features=["x0"], halflife=50.0, min_periods=2.0)
                .over("g")
            )
        )

    def test_multi_chunk_input(self):
        """`SeriesExport` carries `arrays: **ArrowArray` plus a length, so a
        chunked Series exports one ArrowArray per chunk. Each needs releasing."""
        df = pl.concat([frame(300) for _ in range(5)], rechunk=False)
        assert df.n_chunks() > 1
        assert_plateaus(lambda: po.ModelBank([SPEC]).fit_predict(df))

    def test_sliced_frame_sharing_buffers(self):
        """A slice shares its parent's buffers; releasing the export must not
        free memory the parent still owns."""
        parent = frame(4000)
        sl = parent.slice(100, 1200)
        assert_plateaus(lambda: po.ModelBank([SPEC]).fit_predict(sl))
        assert parent["x0"].sum() == pytest.approx(parent["x0"].sum())

    def test_the_bank_error_path_still_releases(self):
        """The likeliest leak site in any FFI: a call that fails *after* the
        frame has crossed. A Categorical is the right trigger -- every other
        column imports cleanly first, and it is one of the few dtypes that is
        genuinely refused (a String feature is parsed back to f64, and a
        non-numeric one becomes nulls, so neither raises)."""
        bad = frame().with_columns(x1=pl.col("x1").cast(pl.String).cast(pl.Categorical))

        def raises():
            with pytest.raises(ValueError, match="categorical"):
                po.ModelBank([SPEC]).fit_predict(bad)

        assert_plateaus(raises)

    def test_the_plugin_error_path_still_releases(self):
        """The same question for the other FFI path, where it is sharper: the
        engine has already exported the inputs when our plugin returns an
        error, so releasing them is the loader's job on a path that only runs
        when something has gone wrong."""
        bad = frame().with_columns(c=pl.col("x1").cast(pl.String).cast(pl.Categorical))

        def raises():
            with pytest.raises(pl.exceptions.ComputeError, match="categorical"):
                bad.with_columns(
                    pl.col("y").online.ewridge(features=["c"], halflife=50.0, min_periods=2.0)
                )

        assert_plateaus(raises)

    def test_output_outliving_its_input(self):
        """Our output must own its data, not borrow from a frame that is gone."""

        def outlive():
            d = frame(400)
            out = po.ModelBank([SPEC]).fit_predict(d)
            del d
            gc.collect()
            return out["m"].struct.field("pred_y").sum()

        assert_plateaus(outlive)

    def test_state_round_trip(self):
        bank = po.ModelBank([SPEC])
        bank.fit_predict(frame(200))
        assert_plateaus(lambda: po.ModelBank.load_bytes(bank.save_bytes()))


def run_isolated(body: str, timeout=180):
    """Run `body` in a fresh interpreter and require a clean exit.

    A segfault or Rust abort takes the whole process down, which inside pytest
    kills the run and reports nothing useful. In a subprocess it is an exit
    code, and `faulthandler` turns it into a native traceback on stderr.
    """
    src = "import faulthandler; faulthandler.enable()\n" + textwrap.dedent(body)
    r = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=timeout,
    )
    assert r.returncode == 0, (
        f"exited {r.returncode} (negative means a fatal signal)\n"
        f"--- stdout ---\n{r.stdout[-2000:]}\n--- stderr ---\n{r.stderr[-4000:]}"
    )
    return r


class TestNothingCrashes:
    """Each runs in its own interpreter, so a crash is a failed assertion with
    a native traceback rather than a dead pytest session."""

    def test_gc_of_frames_mid_flight(self):
        run_isolated("""
            import gc
            import numpy as np, polars as pl, polars_online as po
            rng = np.random.default_rng(0)
            spec = po.spec.ewridge("m", targets=["y"], features=["x0"],
                                   halflife=20.0, min_periods=2.0)
            bank = po.ModelBank([spec])
            keep = []
            for i in range(200):
                df = pl.DataFrame({"x0": rng.standard_normal(300)})
                df = df.with_columns(y=pl.col("x0"))
                out = bank.fit_predict(df)
                # Drop the input immediately, keep some outputs, and collect
                # aggressively in between -- the ordering the release callbacks
                # have to survive.
                del df
                if i % 3 == 0:
                    keep.append(out["m"])
                del out
                gc.collect()
            assert sum(len(k) for k in keep) > 0
        """)

    def test_repeated_reference_and_dereference(self):
        run_isolated("""
            import gc, sys
            import numpy as np, polars as pl, polars_online as po
            rng = np.random.default_rng(1)
            spec = po.spec.ewridge("m", targets=["y"], features=["x0"],
                                   halflife=20.0, min_periods=2.0)
            df = pl.DataFrame({"x0": rng.standard_normal(800)})
            df = df.with_columns(y=pl.col("x0"))
            out = po.ModelBank([spec]).fit_predict(df)
            s = out["m"]
            refs = [s for _ in range(500)]
            base = sys.getrefcount(s)
            for _ in range(200):
                refs.append(s.struct.field("pred_y"))
                refs.pop(0)
                gc.collect()
            del refs
            gc.collect()
            assert sys.getrefcount(s) <= base
            assert s.struct.field("pred_y").len() == 800
        """)

    def test_reference_cycles_are_collectable(self):
        """A cycle holding a bank and its output must not be uncollectable --
        that would strand Rust state the GC can see but never free."""
        run_isolated("""
            import gc
            import numpy as np, polars as pl, polars_online as po
            # Deliberately NOT gc.DEBUG_SAVEALL: that parks *everything*
            # collected in gc.garbage, so it cannot distinguish "collected" from
            # "uncollectable". Left alone, only genuinely uncollectable objects
            # land there.
            rng = np.random.default_rng(2)
            spec = po.spec.ewridge("m", targets=["y"], features=["x0"],
                                   halflife=20.0, min_periods=2.0)
            for _ in range(30):
                d = {}
                d["self"] = d
                d["bank"] = po.ModelBank([spec])
                df = pl.DataFrame({"x0": rng.standard_normal(200)})
                d["out"] = d["bank"].fit_predict(df.with_columns(y=pl.col("x0")))
                del d, df
            gc.collect()
            assert not gc.garbage, (
                f"{len(gc.garbage)} uncollectable objects: "
                f"{[type(o).__name__ for o in gc.garbage[:5]]}"
            )
        """)

    def test_empty_and_degenerate_frames(self):
        run_isolated("""
            import polars as pl, polars_online as po
            spec = po.spec.ewridge("m", targets=["y"], features=["x0"],
                                   halflife=20.0, min_periods=2.0)
            bank = po.ModelBank([spec])
            empty = pl.DataFrame({"x0": pl.Series([], dtype=pl.Float64),
                                  "y": pl.Series([], dtype=pl.Float64)})
            for _ in range(50):
                out = bank.fit_predict(empty)
                assert len(out) == 0
            one = pl.DataFrame({"x0": [1.0], "y": [2.0]})
            for _ in range(50):
                bank.fit_predict(one)
            allnull = pl.DataFrame({"x0": [None, None, None], "y": [None, None, None]},
                                   schema={"x0": pl.Float64, "y": pl.Float64})
            for _ in range(50):
                bank.fit_predict(allnull)
        """)

    def test_interleaved_banks_and_plugin(self):
        """Both FFI paths alive at once, each holding exports from the other."""
        run_isolated("""
            import gc
            import numpy as np, polars as pl, polars_online as po
            rng = np.random.default_rng(3)
            spec = po.spec.ewridge("m", targets=["y"], features=["x0"],
                                   halflife=20.0, min_periods=2.0)
            banks = [po.ModelBank([spec]) for _ in range(4)]
            held = []
            for i in range(120):
                df = pl.DataFrame({"x0": rng.standard_normal(250)})
                df = df.with_columns(y=pl.col("x0"))
                held.append(banks[i % 4].fit_predict(df))
                held.append(df.with_columns(
                    pl.col("y").online.ewridge(features=["x0"], halflife=20.0,
                                               min_periods=2.0)))
                if len(held) > 8:
                    held.pop(0); held.pop(0)
                if i % 10 == 0:
                    gc.collect()
            assert len(held) > 0
        """)

    def test_pickled_bank_survives_a_round_trip(self):
        run_isolated("""
            import pickle
            import numpy as np, polars as pl, polars_online as po
            rng = np.random.default_rng(4)
            spec = po.spec.ewridge("m", targets=["y"], features=["x0"],
                                   halflife=20.0, min_periods=2.0)
            bank = po.ModelBank([spec])
            df = pl.DataFrame({"x0": rng.standard_normal(400)})
            df = df.with_columns(y=pl.col("x0"))
            bank.fit_predict(df)
            for _ in range(40):
                bank = pickle.loads(pickle.dumps(bank))
                bank.fit_predict(df)
            assert bank is not None
        """)
