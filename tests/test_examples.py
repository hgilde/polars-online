"""Everything under `examples/` must actually run.

A documented example that no longer works is worse than no example: it is a
claim the reader trusts. These are the only tests that exercise the files the
README points people at, end to end and unmodified.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"


def _without_coef(out):
    return out.select("ridge").unnest("ridge").drop("coef")


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, cwd=str(REPO), check=False, **kw)


class TestPathwayExample:
    """E26. Pathway itself is BSL-licensed and not a dependency, so what CI can
    check is the plain-batch path — which is the whole operator except for who
    hands it the batches."""

    def test_it_runs(self):
        res = _run([sys.executable, str(EXAMPLES / "pathway_integration.py")])
        assert res.returncode == 0, res.stderr
        assert "rows: 5000" in res.stdout

    def test_the_operator_is_chunk_invariant_and_checkpointable(self):
        sys.path.insert(0, str(EXAMPLES))
        try:
            import pathway_integration as ex
        finally:
            sys.path.pop(0)

        df = ex._synthetic(n=1000)
        whole = ex.BankOperator(ex.SPEC)(df)

        op = ex.BankOperator(ex.SPEC)
        chunked = pl.concat([op(df.slice(i, 97)) for i in range(0, df.height, 97)])
        # `coef` is excluded by design: it is snapshotted on each chunk's last
        # row, so it reports more often under smaller chunks. Every computed
        # field must match exactly.
        assert _without_coef(whole).equals(_without_coef(chunked), null_equal=True)

        # A checkpoint taken mid-stream resumes the same stream.
        a = ex.BankOperator(ex.SPEC)
        a(df.slice(0, 400))
        b = ex.BankOperator(ex.SPEC)
        b.restore(a.snapshot())
        rest = df.slice(400, 600)
        assert _without_coef(a(rest)).equals(_without_coef(b(rest)), null_equal=True)

    def test_pathway_is_not_a_dependency(self):
        """The licence separation is a property of the packaging, so assert it
        there rather than trusting the comment in the example."""
        pyproject = (REPO / "pyproject.toml").read_text()
        assert "pathway" not in pyproject


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """The input `examples/bank.toml` expects, built by the script the config's
    own comment points at."""
    d = tmp_path_factory.mktemp("bank_toml")
    res = _run([
        sys.executable,
        str(REPO / "scripts" / "make_example_parquet.py"),
        "--rows", "2000",
        "--out", str(d / "in.parquet"),
    ])  # fmt: skip
    assert res.returncode == 0, res.stderr
    return d


class TestBankToml:
    """`examples/bank.toml` is quoted in the README and in the CLI's own module
    docs; nothing until now ran it."""

    def _cli(self, *args):
        return _run(["cargo", "run", "-q", "-p", "online-cli", "--", *args])

    def test_dry_run_validates_the_shipped_config(self, data):
        res = self._cli(
            "--config", str(EXAMPLES / "bank.toml"),
            "--input", str(data / "in.parquet"),
            "--output", str(data / "out.parquet"),
            "--dry-run",
        )  # fmt: skip
        assert res.returncode == 0, res.stderr
        assert "ridge" in res.stdout and "kalman" in res.stdout

    def test_it_produces_both_banks_output(self, data):
        out = data / "out.parquet"
        state = data / "bank.state"
        res = self._cli(
            "--config", str(EXAMPLES / "bank.toml"),
            "--input", str(data / "in.parquet"),
            "--output", str(out),
            "--save-state", str(state),
        )  # fmt: skip
        assert res.returncode == 0, res.stderr
        df = pl.read_parquet(out)
        # --rows is per group, and the example generator makes three.
        assert df.height == pl.read_parquet(data / "in.parquet").height == 6000
        for name in ("ridge", "kalman"):
            assert name in df.columns
            fields = [f.name for f in df.schema[name].fields]
            assert any(f.startswith("pred_y") for f in fields), fields
        assert state.exists(), "save_state produced no file"

    def test_resuming_from_that_state_continues_the_stream(self, data):
        """The README advertises `--resume`; this proves the state the previous
        test wrote is loadable by the same config."""
        res = self._cli(
            "--config", str(EXAMPLES / "bank.toml"),
            "--input", str(data / "in.parquet"),
            "--output", str(data / "out2.parquet"),
            "--resume", str(data / "bank.state"),
            # The config's own `save_state` is relative, so without this the
            # run would drop a `bank.state` in the repo root.
            "--save-state", str(data / "bank2.state"),
        )  # fmt: skip
        assert res.returncode == 0, res.stderr
        first = pl.read_parquet(data / "out.parquet")
        second = pl.read_parquet(data / "out2.parquet")
        field = next(f.name for f in first.schema["ridge"].fields if f.name.startswith("pred_y"))
        p1 = first["ridge"].struct.field(field).to_list()
        p2 = second["ridge"].struct.field(field).to_list()
        neff = first["ridge"].struct.field("n_eff").to_list()
        # Compare at the first row that was not skipped for a null: the warmed
        # run predicts there, where the cold run was still inside min_periods.
        i = next(j for j, v in enumerate(neff) if v is not None)
        assert p1[i] is None, "cold run should still be warming up"
        assert p2[i] is not None, "resumed run should predict immediately"
