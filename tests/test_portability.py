"""Determinism and cross-platform concerns testable without a Windows runner
(docs/TESTING.md sections D and E).

The parts that genuinely need Windows (T-W1, T-W2, T-W7, T-W8) run in CI; these
are the parts that can be pinned anywhere, so that when CI does run, a failure
points at the platform rather than at us.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_online as po

REPO = Path(__file__).resolve().parent.parent


def _frame(n=400, groups=6, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "g": [f"g{i % groups}" for i in range(n)],
            # rows of group j are i = j, j+groups, ...; their within-group
            # position (and so a monotone per-group clock) is i // groups
            "t": np.array([float(i // groups) for i in range(n)]),
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y0": rng.standard_normal(n),
        }
    )


def _spec(**kw):
    d = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        max_dclock=10.0,
        halflife=50.0,
        group="g",
        min_periods=5.0,
        max_rows_between_solves=1,
    )
    d.update(kw)
    return po.spec.ewridge("m", **d)


def _bank_specs():
    """The plain spec beside one with a halflife grid, `coef` on every row and
    every optional output on, so that a field of every kind is compared."""
    grid = po.spec.ewridge(
        "grid",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        max_dclock=10.0,
        halflife=[10.0, 50.0, 200.0],
        group="g",
        min_periods=5.0,
        max_rows_between_solves=1,
        coef_every=1,
        emit_sigma=True,
        emit_resid_z=True,
        emit_selected=True,
        emit_averaged=True,
        emit_metrics=True,
        resid_quantiles=[0.5, 0.9],
        emit_autocorr=True,
        emit_drift=True,
    )
    return [_spec(), grid]


class TestThreadDeterminism:
    """T-D3: the bank fans out over (spec x group) on its own pool, sized by
    `POLARS_ONLINE_MAX_THREADS`, and from 4096 rows a chunk's columns are
    also laid out, read and assembled in parallel (`PAR_MIN_ROWS`,
    docs/PERFORMANCE.md P9-P11). Work within one stream is serial, so thread
    count must not change a single number -- in any field, on either side
    of that threshold."""

    SNIPPET = """
import sys, numpy as np, polars as pl
sys.path.insert(0, {tests!r})
import polars_online as po
from test_portability import _frame, _bank_specs
out = po.ModelBank(_bank_specs()).fit_predict(_frame(n={n}, groups={groups}, seed=1))
out.write_ipc({path!r})
print(po.thread_pool_size())
"""

    # 37 groups round-robin over 5000 rows: every column is gathered into the
    # group layout and read on the pool, and every field is scattered back.
    FRAMES = [
        pytest.param(400, 6, id="below-PAR_MIN_ROWS"),
        pytest.param(5000, 37, id="above-PAR_MIN_ROWS"),
    ]

    def _run_with_threads(self, tmp_path, tag, n_threads, n, groups):
        env = dict(os.environ, POLARS_ONLINE_MAX_THREADS=str(n_threads), POLARS_MAX_THREADS="1")
        path = tmp_path / f"{tag}.arrow"
        code = self.SNIPPET.format(tests=str(REPO / "tests"), n=n, groups=groups, path=str(path))
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=str(REPO),
            check=False,
        )
        assert res.returncode == 0, res.stderr[-2000:]
        # The variable took: the pool the bank ran on is the size asked for.
        assert int(res.stdout.strip()) == n_threads
        out = pl.read_ipc(path)
        assert out.height == n and out.columns[-2:] == ["m", "grid"]
        return out

    @pytest.mark.parametrize(("n", "groups"), FRAMES)
    def test_one_thread_matches_many(self, tmp_path, n, groups):
        single = self._run_with_threads(tmp_path, "one", 1, n, groups)
        many = self._run_with_threads(tmp_path, "many", 8, n, groups)
        assert_frame_equal(single, many, check_exact=True)

    @pytest.mark.parametrize(("n", "groups"), FRAMES)
    def test_repeated_runs_are_identical(self, tmp_path, n, groups):
        a = self._run_with_threads(tmp_path, "a", 4, n, groups)
        b = self._run_with_threads(tmp_path, "b", 4, n, groups)
        assert_frame_equal(a, b, check_exact=True)


class TestThreadPoolKnob:
    """`POLARS_ONLINE_MAX_THREADS` is the bank's knob and the only one: it is
    read when the pool is built, at the first bank call; polars' and rayon's
    own variables do not reach it; a value that is not a count is refused
    by name, and nothing is built, so the process can go on."""

    def _python(self, code, **env):
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, **env},
            cwd=str(REPO),
            check=False,
        )
        assert res.returncode == 0, res.stderr[-2000:]
        return res.stdout.strip()

    def test_read_at_the_first_bank_call_and_fixed_after(self):
        out = self._python(
            "import os, polars as pl, polars_online as po\n"
            "os.environ['POLARS_ONLINE_MAX_THREADS'] = '3'\n"  # after import: still in time
            "print(po.thread_pool_size())\n"
            "os.environ['POLARS_ONLINE_MAX_THREADS'] = '5'\n"  # after the build: too late
            "print(po.thread_pool_size())\n"
        )
        assert out.split() == ["3", "3"]

    def test_other_pools_variables_do_not_size_it(self):
        cores = os.cpu_count() or 1
        out = self._python(
            "import polars_online as po; print(po.thread_pool_size())",
            RAYON_NUM_THREADS="1",
            POLARS_MAX_THREADS="1",
        )
        assert int(out) == cores

    def test_a_value_that_is_not_a_count_is_refused_by_name(self):
        tests = str(REPO / "tests")
        out = self._python(
            "import os, sys, polars_online as po\n"
            f"sys.path.insert(0, {tests!r})\n"
            "from test_portability import _frame, _spec\n"
            "bank = po.ModelBank([_spec()])\n"
            "try:\n"
            "    bank.fit_predict(_frame())\n"
            "except ValueError as e:\n"
            "    print(e)\n"
            "os.environ['POLARS_ONLINE_MAX_THREADS'] = '2'\n"  # fixed: the same bank goes on
            "print(bank.fit_predict(_frame()).height, po.thread_pool_size())\n",
            POLARS_ONLINE_MAX_THREADS="eight",
        )
        first, second = out.split("\n")
        refused = 'POLARS_ONLINE_MAX_THREADS="eight" is not a number of threads'
        assert first.startswith(refused), first
        assert second == f"{_frame().height} 2"


class TestOutputSchemaStability:
    """T-W6: output field names embed floats (`pred_y0__r0.000001`). Rust's
    float Display is locale-independent, so these must be byte-identical on
    every platform -- and they are part of the public schema, so a divergence
    would silently break `expression == bank`."""

    def test_exact_field_names_for_a_grid_spec(self):
        spec = po.spec.ewridge(
            "m",
            targets=["y0", "y1"],
            features=["x0", "x1", "x2"],
            ridge=[1e-6, 0.1, 10.0],
            feature_sets={"fast": ["x0", "x1"], "slow": ["x2"]},
            halflife=[100.0, 500.0],
            min_periods=5.0,
        )
        got = po.spec.output_fields(spec)
        assert got == [
            "pred_y0__fast_r0.000001@h100",
            "resid_y0__fast_r0.000001@h100",
            "pred_y0__fast_r0.1@h100",
            "resid_y0__fast_r0.1@h100",
            "pred_y0__fast_r10@h100",
            "resid_y0__fast_r10@h100",
            "pred_y0__slow_r0.000001@h100",
            "resid_y0__slow_r0.000001@h100",
            "pred_y0__slow_r0.1@h100",
            "resid_y0__slow_r0.1@h100",
            "pred_y0__slow_r10@h100",
            "resid_y0__slow_r10@h100",
            "pred_y1__fast_r0.000001@h100",
            "resid_y1__fast_r0.000001@h100",
            "pred_y1__fast_r0.1@h100",
            "resid_y1__fast_r0.1@h100",
            "pred_y1__fast_r10@h100",
            "resid_y1__fast_r10@h100",
            "pred_y1__slow_r0.000001@h100",
            "resid_y1__slow_r0.000001@h100",
            "pred_y1__slow_r0.1@h100",
            "resid_y1__slow_r0.1@h100",
            "pred_y1__slow_r10@h100",
            "resid_y1__slow_r10@h100",
            "n_eff@h100",
            "coef@h100",
            "pred_y0__fast_r0.000001@h500",
            "resid_y0__fast_r0.000001@h500",
            "pred_y0__fast_r0.1@h500",
            "resid_y0__fast_r0.1@h500",
            "pred_y0__fast_r10@h500",
            "resid_y0__fast_r10@h500",
            "pred_y0__slow_r0.000001@h500",
            "resid_y0__slow_r0.000001@h500",
            "pred_y0__slow_r0.1@h500",
            "resid_y0__slow_r0.1@h500",
            "pred_y0__slow_r10@h500",
            "resid_y0__slow_r10@h500",
            "pred_y1__fast_r0.000001@h500",
            "resid_y1__fast_r0.000001@h500",
            "pred_y1__fast_r0.1@h500",
            "resid_y1__fast_r0.1@h500",
            "pred_y1__fast_r10@h500",
            "resid_y1__fast_r10@h500",
            "pred_y1__slow_r0.000001@h500",
            "resid_y1__slow_r0.000001@h500",
            "pred_y1__slow_r0.1@h500",
            "resid_y1__slow_r0.1@h500",
            "pred_y1__slow_r10@h500",
            "resid_y1__slow_r10@h500",
            "n_eff@h500",
            "coef@h500",
        ]

    def test_lasso_field_names(self):
        spec = po.spec.lasso(
            "m",
            targets=["y0"],
            features=["x0"],
            lasso_path=[1.0, 0.01, 0.0],
            halflife=100.0,
        )
        assert po.spec.output_fields(spec) == [
            "pred_y0__l1",
            "resid_y0__l1",
            "pred_y0__l0.01",
            "resid_y0__l0.01",
            "pred_y0__l0",
            "resid_y0__l0",
            "n_eff",
            "coef",
            "lam_selected_y0",
        ]

    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"emit_sigma": True, "emit_resid_z": True},
            {"emit_selected": True, "emit_averaged": True},
            {"emit_drift": True},
            {"resid_quantiles": [0.5, 0.99], "emit_autocorr": True},
            {
                "emit_sigma": True,
                "emit_resid_z": True,
                "emit_drift": True,
                "emit_selected": True,
                "emit_averaged": True,
                "resid_quantiles": [0.1, 0.9],
                "emit_autocorr": True,
            },
        ],
        ids=["plain", "sigma+z", "selected+avg", "drift", "quantiles+autocorr", "all"],
    )
    def test_names_match_the_realized_struct(self, extra):
        # The declared schema and the produced struct must agree exactly, for
        # every combination of optional outputs. They diverged once, when an
        # output was added to the assembler but not to `output_fields`: the
        # expression plugin takes its dtype from the declaration, so such a
        # divergence breaks `.over()` while the bank keeps working.
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            ridge=[1e-6, 0.5],
            halflife=50.0,
            min_periods=2.0,
            max_rows_between_solves=1,
            **extra,
        )
        out = po.ModelBank([spec]).fit_predict(_frame().drop("g"))
        assert [f.name for f in out.schema["m"].fields] == po.spec.output_fields(spec)

    #: Every model, with whatever it needs to be constructible. `ew_cov` has no
    #: targets and `holt` no features, so both are given explicitly.
    _ALL_MODELS = [
        ("ewridge", {"features": ["x0", "x1"]}),
        ("rls", {"features": ["x0", "x1"], "ridge": 1.0}),
        ("kalman", {"features": ["x0", "x1"], "coef_halflife": 100.0}),
        ("lasso", {"features": ["x0", "x1"], "lasso_path": [0.1, 0.0]}),
        ("huber", {"features": ["x0", "x1"]}),
        ("quantile", {"features": ["x0", "x1"], "quantile": 0.5}),
        ("ftrl", {"features": ["x0", "x1"]}),
        ("sgd", {"features": ["x0", "x1"], "learning_rate": 0.01}),
        ("pa", {"features": ["x0", "x1"]}),
        ("holt", {}),
    ]

    @pytest.mark.parametrize(("model", "kw"), _ALL_MODELS, ids=[m for m, _ in _ALL_MODELS])
    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"emit_sigma": True, "emit_resid_z": True, "emit_drift": True},
            {"resid_quantiles": [0.5], "emit_autocorr": True, "emit_metrics": True},
        ],
        ids=["plain", "sigma+z+drift", "quantiles+autocorr+metrics"],
    )
    def test_names_match_the_realized_struct_for_every_model(self, model, kw, extra):
        """The same guard as above, across every model rather than just
        `ewridge`.

        The optional outputs are assembled in the stream layer, but each model
        contributes its own prediction and coefficient slots, so a model added
        after `output_fields` was written can declare a different schema from
        the one it produces -- and the expression plugin, which takes its dtype
        from the declaration, would break on it while the bank kept working.
        `sgd`, `pa`, `holt` and `ew_cov` all postdate that test.
        """
        spec = getattr(po.spec, model)(
            "m",
            targets=["y0"],
            halflife=50.0,
            min_periods=2.0,
            **kw,
            **extra,
        )
        out = po.ModelBank([spec]).fit_predict(_frame().drop("g"))
        assert [f.name for f in out.schema["m"].fields] == po.spec.output_fields(spec)

    @pytest.mark.parametrize("coef_every", [0, 7], ids=["plain", "coef_every"])
    def test_ew_cov_names_match_the_realized_struct(self, coef_every):
        """`ew_cov` has no targets, so its declared schema is built from the
        statistics and column pairs rather than from targets. (It used to be
        parametrized over `emit_sigma` / `emit_metrics`, which it silently
        ignored; since 0.2.0 an unsupervised spec refuses them by name.)"""
        spec = po.spec.ew_cov(
            "m",
            features=["x0", "x1"],
            stats=["mean", "var", "std", "cov", "corr"],
            halflife=50.0,
            min_periods=2.0,
            coef_every=coef_every,
        )
        out = po.ModelBank([spec]).fit_predict(_frame().drop("g"))
        assert [f.name for f in out.schema["m"].fields] == po.spec.output_fields(spec)

    @pytest.mark.parametrize("halflife", [50.0, [20.0, 50.0]], ids=["one", "grid"])
    @pytest.mark.parametrize("coef_every", [0, 7], ids=["plain", "coef_every"])
    def test_kmeans_names_match_the_realized_struct(self, halflife, coef_every):
        """`kmeans` has no targets either: an `i32` assignment, two distances,
        `n_eff` and the centres as `coef`, per instance."""
        spec = po.spec.kmeans(
            "m",
            features=["x0", "x1"],
            k=2,
            warm_rows=5,
            halflife=halflife,
            min_periods=2.0,
            coef_every=coef_every,
        )
        out = po.ModelBank([spec]).fit_predict(_frame().drop("g"))
        assert [f.name for f in out.schema["m"].fields] == po.spec.output_fields(spec)
        idx = po.spec.output_index(spec)
        names = {pl.Float64: "f64", pl.Int32: "i32", pl.List(pl.Float64): "list[f64]"}
        for f in out.schema["m"].fields:
            declared = idx.filter(pl.col("field") == f.name)["dtype"].item()
            assert names[f.dtype] == declared, (f, declared)


class TestConfigParsing:
    """T-W3/T-W4: the CLI reads a TOML config as text. Windows checkouts can
    have CRLF endings, and Windows paths need escaping in TOML strings."""

    @pytest.fixture(autouse=True)
    def _exe(self, online_cli):
        self.exe = online_cli

    def _cli(self, args, **kw):
        return subprocess.run(
            [str(self.exe), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(REPO),
            check=False,
            **kw,
        )

    def test_an_unescaped_windows_path_is_rejected_with_a_usable_hint(self, tmp_path):
        """The first Windows CI run ever attempted failed here.

        `input = "C:\\Users\\me\\in.parquet"` is invalid TOML -- `\\U` starts a
        unicode escape -- and TOML's own message is "too few unicode value
        digits", which says nothing about paths. The CLI now names the three
        ways to write it. Checked on every OS, because the mistake is about
        the *config text*, not about the host: a Linux user writing a Windows
        path in a config hits it too, and this is the test that would have
        caught it before a Windows runner existed.
        """
        cfg = tmp_path / "unescaped.toml"
        cfg.write_text(
            'input = "C:\\Users\\me\\in.parquet"\n'
            'output = "C:\\Users\\me\\out.parquet"\n'
            "\n[[specs]]\n"
            'name = "m"\n'
            'targets = ["y"]\n'
            'features = ["x0"]\n'
            "halflife = 100.0\n"
            "min_periods = 5.0\n"
            "\n[specs.model]\n"
            'type = "ew_ridge"\n'
            "ridge = 1e-6\n"
        )
        res = self._cli(["--config", str(cfg), "--dry-run"])
        assert res.returncode != 0, "invalid TOML must not be accepted"
        err = res.stdout + res.stderr
        assert "hint:" in err, f"no hint offered:\n{err}"
        # All three escapes the hint recommends must be named.
        assert "literal string" in err
        assert "backslashes doubled" in err
        assert "forward slashes" in err

    @pytest.mark.parametrize("escape", ["literal", "doubled", "forward"])
    def test_each_documented_windows_path_form_parses(self, tmp_path, escape):
        """The hint is only useful if all three forms it recommends work."""
        src = tmp_path / "in.parquet"
        pl.DataFrame({"x0": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}).write_parquet(src)
        out = tmp_path / "out.parquet"
        raw_in, raw_out = str(src), str(out)
        if escape == "literal":
            lines = f"input = '{raw_in}'\noutput = '{raw_out}'\n"
        elif escape == "doubled":
            lines = (
                f'input = "{raw_in.replace(chr(92), chr(92) * 2)}"\n'
                f'output = "{raw_out.replace(chr(92), chr(92) * 2)}"\n'
            )
        else:
            lines = (
                f'input = "{raw_in.replace(chr(92), "/")}"\n'
                f'output = "{raw_out.replace(chr(92), "/")}"\n'
            )
        cfg = tmp_path / f"{escape}.toml"
        cfg.write_text(
            lines + "\n[[specs]]\n"
            'name = "m"\n'
            'targets = ["y"]\n'
            'features = ["x0"]\n'
            "halflife = 100.0\n"
            "min_periods = 1.0\n"
            "\n[specs.model]\n"
            'type = "ew_ridge"\n'
            "ridge = 1e-6\n"
        )
        res = self._cli(["--config", str(cfg)])
        assert res.returncode == 0, f"{escape} form rejected:\n{res.stdout}{res.stderr}"
        assert out.exists()

    @pytest.mark.parametrize("newline", ["\n", "\r\n"])
    def test_config_parses_with_either_line_ending(self, tmp_path, newline):
        toml = newline.join(
            [
                'input = "in.parquet"',
                'output = "out.parquet"',
                "",
                "[[specs]]",
                'name = "ridge"',
                'targets = ["y"]',
                'features = ["x0"]',
                "halflife = 100.0",
                "min_periods = 5.0",
                "",
                "[specs.model]",
                'type = "ew_ridge"',
                "ridge = 1e-6",
                "",
            ]
        )
        cfg = tmp_path / "bank.toml"
        cfg.write_bytes(toml.encode())
        res = self._cli(["--config", str(cfg), "--dry-run"])
        assert res.returncode == 0, res.stderr[-1500:]
        assert "config OK" in res.stdout
        assert "ridge.pred_y" in res.stdout

    def test_windows_style_path_round_trips_through_toml(self, tmp_path):
        # A backslash path must be written with escaped separators (or a TOML
        # literal string); this pins that the parser keeps it intact, which is
        # what T-W3 will exercise for real on Windows.
        cfg = tmp_path / "win.toml"
        cfg.write_text(
            'input = "C:\\\\data\\\\in.parquet"\n'
            'output = "C:\\\\data\\\\out.parquet"\n'
            "\n[[specs]]\n"
            'name = "ridge"\n'
            'targets = ["y"]\n'
            'features = ["x0"]\n'
            "halflife = 100.0\n"
            "\n[specs.model]\n"
            'type = "ew_ridge"\n'
        )
        res = self._cli(["--config", str(cfg), "--dry-run"])
        assert res.returncode == 0, res.stderr[-1500:]
        assert "C:\\data\\in.parquet" in res.stdout

    def test_paths_with_spaces(self, tmp_path):
        d = tmp_path / "a directory with spaces"
        d.mkdir()
        cfg = d / "bank.toml"
        cfg.write_text(
            f'input = "{(d / "in.parquet").as_posix()}"\n'
            f'output = "{(d / "out.parquet").as_posix()}"\n'
            "\n[[specs]]\n"
            'name = "ridge"\n'
            'targets = ["y"]\n'
            'features = ["x0"]\n'
            "halflife = 100.0\n"
            "\n[specs.model]\n"
            'type = "ew_ridge"\n'
        )
        res = self._cli(["--config", str(cfg), "--dry-run"])
        assert res.returncode == 0, res.stderr[-1500:]
        assert "directory with spaces" in res.stdout


class TestStateFilePortability:
    """T-W2 precondition: the state file must contain nothing host-specific."""

    def test_state_bytes_are_deterministic_and_carry_no_paths(self, tmp_path):
        df = _frame(n=200, seed=3)
        spec = _spec()
        a, b = po.ModelBank([spec]), po.ModelBank([spec])
        a.fit_predict(df)
        b.fit_predict(df)
        assert a.save_bytes() == b.save_bytes(), "state bytes are not reproducible"

        blob = a.save_bytes()
        # No absolute paths, usernames or platform markers leaked into the file.
        for marker in (str(REPO).encode(), b"/Users/", b"C:\\", b"darwin", b"win32"):
            assert marker not in blob, f"state file contains {marker!r}"

    def test_state_survives_a_round_trip_through_bytes(self):
        df = _frame(n=200, seed=4)
        spec = _spec()
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 100))
        b = po.ModelBank.load_bytes(a.save_bytes(), specs=[spec])
        second = df.slice(100, 100)
        assert a.fit_predict(second).equals(b.fit_predict(second), null_equal=True)
