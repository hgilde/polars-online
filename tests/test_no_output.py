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
from conftest import run_online


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
    def test_it_saves_the_state_and_writes_nothing(self, tmp_path, online_cli):
        df = frame()
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        state = tmp_path / "b.state"
        run_online(
            online_cli, tmp_path, [cov_spec()], input=src, save_state=state, chunk_rows=100,
            args=["--no-output"],
        )  # fmt: skip
        # Nothing else was created -- not even the temporary the writer
        # renames into place, since no writer ran. (`bank.toml` is the config
        # this test wrote, not the run's doing.)
        assert {p.name for p in tmp_path.iterdir()} == {"in.parquet", "b.state", "bank.toml"}

    def test_the_state_is_the_state_a_written_run_would_have_left(self, tmp_path, online_cli):
        df = frame(seed=1)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        quiet, loud = tmp_path / "q.state", tmp_path / "l.state"
        run_online(
            online_cli, tmp_path, [cov_spec()], input=src, save_state=quiet, chunk_rows=64,
            args=["--no-output"],
        )  # fmt: skip
        run_online(
            online_cli,
            tmp_path,
            [cov_spec()],
            input=src,
            output=tmp_path / "out.parquet",
            save_state=loud,
            chunk_rows=64,
        )
        assert quiet.read_bytes() == loud.read_bytes(), "the same run, minus the file"

    def test_no_output_clears_a_config_that_names_one(self, tmp_path, online_cli):
        df = frame()
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        run_online(
            online_cli,
            tmp_path,
            [cov_spec()],
            input=src,
            output=tmp_path / "out.parquet",
            save_state=tmp_path / "b.state",
            chunk_rows=100,
            args=["--no-output"],
        )
        assert not (tmp_path / "out.parquet").exists()
        assert (tmp_path / "b.state").exists()

    def test_a_run_that_writes_nothing_and_saves_nothing_is_refused(self, tmp_path, online_cli):
        df = frame(n=50)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        res = run_online(
            online_cli, tmp_path, [cov_spec()], input=src, args=["--no-output"], check=False
        )
        assert res.returncode != 0
        assert "somewhere to put its work" in res.stderr, res.stderr

    def test_an_empty_input_is_not_an_error(self, tmp_path, online_cli):
        """With an output, an empty input still writes an empty frame of the
        right schema. Without one there is nothing to write, and the run is a
        no-op that still saves its (empty) state."""
        src = tmp_path / "in.parquet"
        frame(n=0).write_parquet(src)
        state = tmp_path / "b.state"
        run_online(
            online_cli, tmp_path, [cov_spec()], input=src, save_state=state, args=["--no-output"]
        )
        assert state.exists()

    def test_it_works_for_a_learning_spec_too(self, tmp_path, online_cli):
        """Nothing about this is `ew_cov`-only: a ridge fit whose product is
        the coefficients need not write its predictions either."""
        df = frame(seed=2)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        state = tmp_path / "b.state"
        run_online(
            online_cli, tmp_path, [ridge_spec()], input=src, save_state=state, chunk_rows=100,
            args=["--no-output"],
        )  # fmt: skip
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

    def test_the_filled_spec_is_the_one_python_writes(self, tmp_path, online_cli):
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
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        from_python = tmp_path / "py.state"
        bank = po.ModelBank([cov_spec(name="c")])
        for out in bank.fit_predict_batches(df.slice(i, 100) for i in range(0, df.height, 100)):
            del out
        bank.save(from_python)
        assert from_toml.read_bytes() == from_python.read_bytes()

    def test_a_bank_fills_them_too(self):
        """Not only the TOML path: a hand-written dict gets the same
        treatment, so the two never disagree."""
        spec = dict(cov_spec())
        spec["targets"] = []
        bank = po.ModelBank([spec])
        assert spec["targets"] == []  # the caller's dict is untouched
        # What the bank runs has them filled, before a save as after it; the
        # specs were the caller's dicts until one (review 2026-09-12, S21).
        assert bank.specs[0]["targets"] == ["x0"]
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
