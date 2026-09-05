"""E50 and E53: a run whose product is its state, and a TOML spec that does
not have to invent a target.

The two go together. An accumulator-only spec (`ew_cov` with `stats=[]`, or a
`marginal`) emits `n_eff` a row and nothing else, and the value of running it
is the state it leaves behind -- so writing that column to a file, at 8 GB a
billion rows, is I/O nobody reads. And such a spec has no target, which a
TOML author should not have to make one up for.
"""

import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po


def frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )


def cov_spec(name="c", **kw):
    return po.spec.ew_cov(name, features=["x0", "x1"], stats=[], halflife=200.0, **kw)


def ridge_spec(name="m", **kw):
    return po.spec.ewridge(
        name, targets=["y"], features=["x0", "x1"], halflife=200.0, min_periods=3.0, **kw
    )


class TestARunWithNoOutput:
    def test_it_saves_the_state_and_writes_nothing(self, tmp_path):
        df = frame()
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        state = tmp_path / "b.state"
        stats = po.run(specs=[cov_spec()], input=src, save_state=state, chunk_rows=100)
        assert stats == {"rows": 500, "chunks": 5}
        # Nothing else was created -- not even the temporary the writer
        # renames into place, since no writer ran.
        assert {p.name for p in tmp_path.iterdir()} == {"in.parquet", "b.state"}

    def test_the_state_is_the_state_a_written_run_would_have_left(self, tmp_path):
        df = frame(seed=1)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        quiet, loud = tmp_path / "q.state", tmp_path / "l.state"
        po.run(specs=[cov_spec()], input=src, save_state=quiet, chunk_rows=64)
        po.run(
            specs=[cov_spec()],
            input=src,
            output=tmp_path / "out.parquet",
            save_state=loud,
            chunk_rows=64,
        )
        assert quiet.read_bytes() == loud.read_bytes(), "the same run, minus the file"

    def test_no_output_clears_a_config_that_names_one(self, tmp_path):
        df = frame()
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        cfg = {
            "output": str(tmp_path / "out.parquet"),
            "save_state": str(tmp_path / "b.state"),
            "specs": [cov_spec()],
            "chunk_rows": 100,
        }
        po.run(cfg, input=src, no_output=True)
        assert not (tmp_path / "out.parquet").exists()
        assert (tmp_path / "b.state").exists()

    def test_a_run_that_writes_nothing_and_saves_nothing_is_refused(self, tmp_path):
        df = frame(n=50)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        with pytest.raises(ValueError, match="somewhere to put its work"):
            po.run(specs=[cov_spec()], input=src)

    def test_an_empty_input_is_not_an_error(self, tmp_path):
        """With an output, an empty input still writes an empty frame of the
        right schema. Without one there is nothing to write, and the run is a
        no-op that still saves its (empty) state."""
        src = tmp_path / "in.parquet"
        frame(n=0).write_parquet(src)
        state = tmp_path / "b.state"
        stats = po.run(specs=[cov_spec()], input=src, save_state=state)
        assert stats["rows"] == 0
        assert state.exists()

    def test_it_works_for_a_learning_spec_too(self, tmp_path):
        """Nothing about this is `ew_cov`-only: a ridge fit whose product is
        the coefficients need not write its predictions either."""
        df = frame(seed=2)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        state = tmp_path / "b.state"
        po.run(specs=[ridge_spec()], input=src, save_state=state, chunk_rows=100)
        bank = po.ModelBank.load(state)
        want = po.ModelBank([ridge_spec()])
        want.fit_predict(df)
        assert bank.coef("m")["coef"].to_list() == want.coef("m")["coef"].to_list()

    def test_the_cli_flag(self, tmp_path, online_cli):
        df = frame(seed=3)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        state = tmp_path / "b.state"
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f"""
input = "{src.as_posix()}"
output = "{(tmp_path / "out.parquet").as_posix()}"
save_state = "{state.as_posix()}"
chunk_rows = 100

[[specs]]
name = "c"
targets = ["x0"]
features = ["x0", "x1"]
halflife = 200.0
[specs.model]
type = "ew_cov"
stats = []
"""
        )
        subprocess.run(
            [str(online_cli), "--config", str(cfg), "--no-output"],
            check=True,
            capture_output=True,
        )
        assert not (tmp_path / "out.parquet").exists()
        assert state.exists()

    def test_the_cli_dry_run_says_there_is_no_output(self, tmp_path, online_cli):
        src = tmp_path / "in.parquet"
        frame(n=10).write_parquet(src)
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f"""
input = "{src.as_posix()}"
save_state = "{(tmp_path / "b.state").as_posix()}"

[[specs]]
name = "c"
features = ["x0", "x1"]
halflife = 200.0
[specs.model]
type = "ew_cov"
"""
        )
        r = subprocess.run(
            [str(online_cli), "--config", str(cfg), "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "output: none (--no-output)" in r.stdout
        assert "b.state" in r.stdout

    def test_output_and_no_output_together_are_refused(self, tmp_path, online_cli):
        cfg = tmp_path / "c.toml"
        cfg.write_text('input = "x.parquet"\n[[specs]]\nname = "c"\nfeatures = ["x0"]\n')
        r = subprocess.run(
            [
                str(online_cli),
                "--config",
                str(cfg),
                "--no-output",
                "--output",
                str(tmp_path / "o.parquet"),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        assert "cannot be used with" in r.stderr


class TestTargetsAreOptionalWhereThereAreNone:
    @pytest.mark.parametrize("model", ["ew_cov", "kmeans", "micro"])
    def test_a_toml_spec_need_not_invent_a_target(self, tmp_path, model, online_cli):
        src = tmp_path / "in.parquet"
        frame(n=300, seed=4).write_parquet(src)
        state = tmp_path / "b.state"
        extra = {"ew_cov": "", "kmeans": "k = 3", "micro": "eps = 1.5"}[model]
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f"""
input = "{src.as_posix()}"
save_state = "{state.as_posix()}"

[[specs]]
name = "u"
features = ["x0", "x1"]
halflife = 200.0
[specs.model]
type = "{model}"
{extra}
"""
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        bank = po.ModelBank.load(state)
        assert bank.specs[0]["targets"] == ["x0"], "filled from features[0]"

    def test_the_filled_spec_is_the_one_python_writes(self, tmp_path):
        """E53's point: the two surfaces must produce the same spec, so a
        state saved from one resumes under the other."""
        df = frame(n=300, seed=5)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        from_toml = tmp_path / "toml.state"
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f"""
input = "{src.as_posix()}"
save_state = "{from_toml.as_posix()}"
chunk_rows = 100

[[specs]]
name = "c"
features = ["x0", "x1"]
halflife = 200.0
[specs.model]
type = "ew_cov"
stats = []
"""
        )
        po.run(str(cfg))
        from_python = tmp_path / "py.state"
        po.run(
            specs=[cov_spec(name="c")],
            input=src,
            save_state=from_python,
            chunk_rows=100,
        )
        assert from_toml.read_bytes() == from_python.read_bytes()

    def test_a_bank_fills_them_too(self):
        """Not only the TOML path: a hand-written dict gets the same
        treatment, so the two never disagree."""
        spec = dict(cov_spec())
        spec["targets"] = []
        bank = po.ModelBank([spec])
        assert bank.specs[0]["targets"] == []  # the caller's dict is untouched
        # ... but what the bank runs, and saves, has them filled.
        bank.fit_predict(frame(n=100))
        assert po.ModelBank.load_bytes(bank.save_bytes()).specs[0]["targets"] == ["x0"]

    def test_a_model_that_needs_targets_still_says_so(self):
        spec = dict(ridge_spec())
        spec["targets"] = []
        with pytest.raises(ValueError, match="targets must be non-empty"):
            po.ModelBank([spec])

    def test_an_unsupervised_spec_with_no_features_either(self):
        spec = dict(cov_spec())
        spec["targets"] = []
        spec["features"] = []
        with pytest.raises(ValueError, match="features must be non-empty"):
            po.ModelBank([spec])
