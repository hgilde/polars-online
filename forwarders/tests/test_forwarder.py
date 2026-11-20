"""The forwarders, in two modes.

**Stub mode** (the default). Nothing real is installed: `tests/stubs` holds a
stub of Polars and a stub of rheon 2.0.0 *with rheon's real layout* (what needs
no Polars is a module of `rheon`, what does is under `rheon.polars`), and `src`
of each forwarder is put on the path. It needs no network and no wheel.

**Real mode**, chosen by two environment variables:

* `FORWARDER_REAL_PYTHON`: a Python whose environment has `rheon[polars,numpy]`
  2.0.0 installed (and `polars-online` not). The stubs are not used.
* `FORWARDER_OLD_PYTHON`: a Python whose environment has `polars-online==1.6.0`
  from PyPI, which writes the 1.x pickle the forwarder must load, and says what
  1.6.0 exported.

The same suite runs against `rheon-compat` (with `polars-online-compat==1.6.0` as
the old Python): both forwarders are tested in either environment, and the
release workflow runs both pairs.

Each check runs in a fresh interpreter, because what is under test is what
happens on first import (a warning once, modules registered once). Written as
`unittest` cases so both pytest and `python3 -m unittest` run it.

    python3 -m pytest forwarders/tests -q                      # stub mode
    FORWARDER_REAL_PYTHON=/scratch/new/bin/python \\
    FORWARDER_OLD_PYTHON=/scratch/old/bin/python \\
        python3 -m pytest forwarders/tests -q                  # real mode
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUBS = ROOT / "tests" / "stubs"
PACKAGES = {
    "polars-online": ROOT / "polars-online",
    "polars-online-compat": ROOT / "polars-online-compat",
}

REAL_PYTHON = os.environ.get("FORWARDER_REAL_PYTHON")
OLD_PYTHON = os.environ.get("FORWARDER_OLD_PYTHON")
REAL = REAL_PYTHON is not None
if REAL and OLD_PYTHON is None:
    raise RuntimeError(
        "FORWARDER_REAL_PYTHON is set: FORWARDER_OLD_PYTHON "
        "(a polars-online==1.6.0 Python) must be too"
    )

#: The modules of polars_online 1.6.0, from `pkgutil.iter_modules` on the wheel from PyPI
#: (real mode checks this set against the installed 1.6.0).
OLD_MODULES = {
    "_bank", "_build_flags", "_check", "_cpu_check", "_duration", "_errors", "_formula",
    "_frame", "_kwargs", "_polars_online", "_renamed", "_spec", "_spec_audit",
    "_spec_changepoint", "_spec_class", "_spec_cluster", "_spec_fields", "_spec_gradient",
    "_spec_regimes", "_spec_robust", "_spec_trend", "_spec_words", "_warnings", "corr",
    "eval", "gbdt", "gram", "ops", "sim", "spec", "stream",
}  # fmt: skip

#: Where rheon 2.0.0 holds each module that is not a module of 1.6.0's name, by
#: the rule that what needs Polars is under `rheon.polars`. Spelled out so a wrong
#: table fails a stub run too, not only a real one.
EXPECTED_TABLE = {
    "_bank": "rheon.polars._bank", "_check": "rheon.polars._check",
    "_errors": "rheon.polars._errors", "_formula": "rheon.polars._formula",
    "_frame": "rheon.polars._frame", "eval": "rheon.polars.eval",
    "gbdt": "rheon.polars._gbdt", "ops": "rheon.polars.ops", "sim": "rheon.polars.sim",
    "stream": "rheon.polars.stream",
    "_build_flags": "rheon._build_flags", "_cpu_check": "rheon._cpu_check",
    "_duration": "rheon._duration", "_kwargs": "rheon._kwargs", "_renamed": "rheon._renamed",
    "_spec": "rheon._spec", "_spec_audit": "rheon._spec_audit",
    "_spec_changepoint": "rheon._spec_changepoint", "_spec_class": "rheon._spec_class",
    "_spec_cluster": "rheon._spec_cluster", "_spec_fields": "rheon._spec_fields",
    "_spec_gradient": "rheon._spec_gradient", "_spec_regimes": "rheon._spec_regimes",
    "_spec_robust": "rheon._spec_robust", "_spec_trend": "rheon._spec_trend",
    "_spec_words": "rheon._spec_words", "_warnings": "rheon._warnings",
    "corr": "rheon.corr", "gram": "rheon.gram", "spec": "rheon.spec",
    "_polars_online": "rheon._rheon",
}  # fmt: skip


def _env(package: str) -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(PACKAGES[package] / "src")]
    if not REAL:
        paths.insert(0, str(STUBS))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.pop("PYTHONWARNINGS", None)
    return env


def _python() -> str:
    return REAL_PYTHON if REAL_PYTHON is not None else sys.executable


def run_process(args: list[str], package: str = "polars-online", cwd: Path | None = None):
    """The forwarder's Python (stubs or the real environment), run on `args`."""
    return subprocess.run(
        [_python(), *args],
        env=_env(package), capture_output=True, text=True,
        cwd=str(cwd if cwd is not None else (STUBS if not REAL else tempfile.gettempdir())),
    )  # fmt: skip


def run(code: str, package: str = "polars-online", *, warn: str = "always") -> object:
    """Run `code` in a fresh interpreter; it prints one JSON value, returned."""
    done = run_process(["-W", warn, "-c", textwrap.dedent(code)], package)
    if done.returncode != 0:
        raise AssertionError(done.stderr)
    return json.loads(done.stdout.strip().splitlines()[-1])


def run_old(code: str, *args: str) -> object:
    """Run `code` on 1.6.0, in a directory with no `polars_online` of its own."""
    assert OLD_PYTHON is not None
    done = subprocess.run(
        [OLD_PYTHON, "-c", textwrap.dedent(code), *args],
        capture_output=True, text=True, cwd=tempfile.gettempdir(),
    )  # fmt: skip
    if done.returncode != 0:
        raise AssertionError(done.stderr)
    return json.loads(done.stdout.strip().splitlines()[-1])


def table(package: str) -> dict:
    """`_MODULES` read from the source, so a test of the keys needs no import."""
    tree = ast.parse((PACKAGES[package] / "src/polars_online/__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_MODULES":
            return ast.literal_eval(node.value)
    raise AssertionError("no _MODULES")


def per_package(test):
    """Run a `(self, name)` test once for each forwarder, as a subtest."""

    def wrapper(self):
        for name in PACKAGES:
            with self.subTest(package=name):
                test(self, name)

    wrapper.__name__ = test.__name__
    return wrapper


class EachPackage(unittest.TestCase):
    @per_package
    def test_import_warns_once(self, name):
        out = run(
            """
            import json, warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import polars_online
                import polars_online.ops, polars_online._bank
                import importlib; importlib.import_module("polars_online")
            print(json.dumps([[str(x.category.__name__), str(x.message)] for x in w]))
            """,
            name,
        )
        self.assertEqual(len(out), 1, out)
        self.assertEqual(out[0][0], "DeprecationWarning")
        self.assertIn("from rheon import polars as rp", out[0][1])
        self.assertIn(
            name + " is now " + ("rheon-compat" if "compat" in name else "rheon"), out[0][1]
        )

    @per_package
    def test_warning_is_attributed_to_the_importing_line(self, name):
        """Under Python's default filters, a DeprecationWarning shows only when it is
        attributed to `__main__`: the user's import, not this package or importlib."""
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "user_script.py"
            script.write_text("import polars_online\n")
            shown = {
                "-c": ["-c", "import polars_online"],
                "script": [str(script)],
                "from-import": ["-c", "from polars_online import ModelBank"],
                "submodule first": ["-c", "import polars_online.gram"],
                "importlib": ["-c", "import importlib; importlib.import_module('polars_online')"],
            }
            for label, args in shown.items():
                with self.subTest(label):
                    done = run_process(args, name, cwd=Path(tmp))
                    self.assertEqual(done.returncode, 0, done.stderr)
                    self.assertEqual(done.stderr.count("DeprecationWarning"), 1, done.stderr)
                    self.assertNotIn("polars_online/__init__.py", done.stderr)
                    self.assertNotIn("importlib", done.stderr.split("DeprecationWarning")[0])
                    self.assertRegex(
                        done.stderr, r"^(<string>|.*user_script\.py):1: DeprecationWarning"
                    )
            # Imported by a library, not by `__main__`, it is hidden, as Python's
            # default filters hide every library's DeprecationWarning.
            (Path(tmp) / "a_library.py").write_text("import polars_online\n")
            done = run_process(["-c", "import a_library"], name, cwd=Path(tmp))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stderr, "")

    @per_package
    def test_identity_across_the_two_names(self, name):
        out = run(
            """
            import json, polars_online, rheon.polars
            from polars_online import _bank, ops
            print(json.dumps([
                polars_online.ModelBank is rheon.polars.ModelBank,
                _bank.ModelBank is rheon.polars.ModelBank,
                ops is rheon.polars.ops and polars_online.ops is ops,
                polars_online.__version__,
            ]))
            """,
            name,
        )
        self.assertEqual(out, [True, True, True, "2.0.0"])

    @per_package
    def test_a_module_imports_by_its_old_path(self, name):
        """`import polars_online.gram` and `from polars_online.gram import solve`."""
        out = run(
            """
            import json, polars_online.gram
            from polars_online.gram import solve
            import rheon.gram, rheon.polars
            print(json.dumps([
                polars_online.gram is rheon.gram,
                polars_online.gram is rheon.polars.gram,
                solve is rheon.gram.solve,
            ]))
            """,
            name,
        )
        self.assertEqual(out, [True, True, True])

    @per_package
    def test_every_table_entry_is_the_mapped_module(self, name):
        mapping = table(name)
        out = run(
            f"""
            import importlib, json, sys, polars_online
            mapping = {mapping!r}
            res = {{}}
            for old, new in mapping.items():
                try:
                    mod = importlib.import_module("polars_online." + old)
                    res[old] = mod is importlib.import_module(new) and mod.__name__ == new
                except ModuleNotFoundError as e:
                    res[old] = str(e)
            print(json.dumps(res))
            """,
            name,
        )
        self.assertEqual({k: v for k, v in out.items() if v is not True}, {})

    @per_package
    def test_the_old_warning_classes_are_rheons(self, name):
        out = run(
            """
            import json, warnings, polars_online as po, rheon, rheon.polars as rp
            warnings.simplefilter("ignore")
            with warnings.catch_warnings():
                # A 1.x filter, by 1.x's class name: it must match what rheon raises.
                warnings.simplefilter("error", po.PolarsOnlineWarning)
                try:
                    warnings.warn("x", rheon.RheonWarning)
                    base = "not raised"
                except rheon.RheonWarning:
                    base = "raised"
                try:
                    warnings.warn("x", rp.RheonDeprecationWarning)
                    dep = "not raised"
                except po.PolarsOnlineDeprecationWarning:
                    dep = "raised"
            print(json.dumps([
                po.PolarsOnlineWarning is rheon.RheonWarning,
                po.PolarsOnlineDeprecationWarning is rheon.RheonDeprecationWarning,
                issubclass(po.PolarsOnlineDeprecationWarning, po.PolarsOnlineWarning),
                issubclass(po.PolarsOnlineDeprecationWarning, DeprecationWarning),
                {"PolarsOnlineWarning", "PolarsOnlineDeprecationWarning"} <= set(po.__all__),
                base, dep,
            ]))
            """,
            name,
        )
        self.assertEqual(out, [True, True, True, True, True, "raised", "raised"])

    def test_unmapped_name_is_an_import_error(self):
        out = run(
            """
            import json, polars_online
            try:
                import polars_online.nothing_here
                r = "imported"
            except ModuleNotFoundError:
                r = "missing"
            print(json.dumps(r))
            """
        )
        self.assertEqual(out, "missing")

    def test_missing_rheon_polars_names_the_extra(self):
        for name, extra in (
            ("polars-online", "rheon[polars]"),
            ("polars-online-compat", "rheon-compat[polars]"),
        ):
            out = run(
                """
                import json, sys
                sys.modules["rheon.polars"] = None   # as if rheon.polars cannot be found
                import importlib.util
                importlib.util.find_spec = lambda n, *a: None
                try:
                    import polars_online
                    r = "imported"
                except ImportError as e:
                    r = str(e)
                print(json.dumps(r))
                """,
                name,
            )
            self.assertIn(extra, out)


class Rheon(unittest.TestCase):
    def test_import_rheon_alone_does_not_import_rheon_polars(self):
        # (Not "does not import polars": the compiled extension's allocator imports
        # Polars's allocator capsule when Polars is installed.)
        out = run(
            "import json, sys, rheon; "
            "print(json.dumps(['rheon.polars' in sys.modules, rheon.__version__]))"
        )
        self.assertEqual(out, [False, "2.0.0"])

    def test_importing_rheon_polars_does_import_it(self):
        out = run("import json, sys, rheon.polars; print(json.dumps('polars' in sys.modules))")
        self.assertTrue(out)


class Table(unittest.TestCase):
    def test_keys_are_the_old_module_set(self):
        for name in PACKAGES:
            self.assertEqual(set(table(name)), OLD_MODULES)

    def test_the_table_is_rheon_2_0_0s_layout(self):
        for name in PACKAGES:
            self.assertEqual(table(name), EXPECTED_TABLE)

    def test_targets_are_rheon_modules(self):
        for name in PACKAGES:
            for old, new in table(name).items():
                self.assertRegex(new, r"^rheon(\.polars)?\.[A-Za-z_]+$", old)
        self.assertEqual(table("polars-online")["_polars_online"], "rheon._rheon")

    def test_the_two_packages_differ_only_in_strings(self):
        a = (PACKAGES["polars-online"] / "src/polars_online/__init__.py").read_text().splitlines()
        b = (
            (PACKAGES["polars-online-compat"] / "src/polars_online/__init__.py")
            .read_text()
            .splitlines()
        )
        self.assertEqual(len(a), len(b))
        changed = [(x, y) for x, y in zip(a, b, strict=True) if x != y]
        self.assertEqual(len(changed), 3)
        for x, y in changed:
            self.assertIn("compat", y)
            self.assertNotIn("compat", x)

    def test_pyproject_depends_on_the_extras(self):
        import tomllib

        for name, dep in (("polars-online", "rheon"), ("polars-online-compat", "rheon-compat")):
            meta = tomllib.loads((PACKAGES[name] / "pyproject.toml").read_text())["project"]
            self.assertEqual(meta["version"], "2.0.0")
            self.assertTrue(
                meta["dependencies"][0].startswith(dep + "[polars]>=2.0.0"), meta["dependencies"]
            )
            self.assertEqual(meta["optional-dependencies"], {"numpy": [dep + "[numpy]"]})


if not REAL:

    class StubLayout(unittest.TestCase):
        def test_every_table_target_has_a_stub(self):
            for old, new in table("polars-online").items():
                path = STUBS / Path(*new.split("."))
                self.assertTrue(
                    path.with_suffix(".py").is_file() or (path / "__init__.py").is_file(),
                    f"{old} -> {new}",
                )

    class StubFrames(unittest.TestCase):
        @per_package
        def test_pickle_by_the_old_module_name_loads(self, name):
            out = run(
                """
                import json, pickle, polars_online, rheon.polars
                # What a polars-online 1.x pickle holds: a global named by the old module.
                cls = pickle.loads(b"cpolars_online._bank\\nModelBank\\n.")
                cls2 = pickle.loads(b"cpolars_online._spec\\nNAME\\n.")
                obj = pickle.loads(pickle.dumps(rheon.polars.ModelBank(0.25)))
                print(json.dumps([cls is rheon.polars.ModelBank, cls2, obj.alpha]))
                """,
                name,
            )
            self.assertEqual(out, [True, "rheon._spec", 0.25])

        @per_package
        def test_online_namespace_delegates_to_rheon(self, name):
            out = run(
                """
                import json, warnings, polars, polars_online
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    a = polars.DataFrame([1.0, 2.0]).online.fit(alpha=2.0)
                    b = polars.LazyFrame([1.0, 2.0]).online.fit()
                    c = polars.DataFrame([1.0, 2.0]).rheon.fit(alpha=2.0)
                print(json.dumps([a, b, c, [str(x.message) for x in w]]))
                """,
                name,
            )
            self.assertEqual(out[:3], [6.0, 1.5, 6.0])
            self.assertEqual(len(out[3]), 1, out[3])  # warned on first use only
            self.assertIn(".online", out[3][0])

        def test_existing_online_namespace_is_left_alone(self):
            out = run(
                """
                import json, polars
                class Theirs:
                    def __init__(self, frame): pass
                    def fit(self): return "theirs"
                polars.api.register_dataframe_namespace("online")(Theirs)
                import polars_online
                d, lz = polars.DataFrame([1.0]), polars.LazyFrame([1.0])
                print(json.dumps([d.online.fit(), lz.online.fit()]))
                """
            )
            self.assertEqual(out, ["theirs", 0.5])  # the lazy frame gets the forwarder's


if REAL:
    #: One model, one frame: what both the 1.6.0 script and the forwarder run.
    FIT = textwrap.dedent(
        """
        import polars as pl
        def frame(lo, hi):
            return pl.DataFrame({
                "x": [float(i % 7) for i in range(lo, hi)],
                "y": [0.5 * (i % 7) + (i % 3) for i in range(lo, hi)],
            })
        """
    )

    def fit(code: str) -> str:
        """`code` after the helper that builds the frames."""
        return FIT + textwrap.dedent(code)

    class RealRheon(unittest.TestCase):
        """The forwarders against rheon 2.0.0 as built, and 1.6.0 from PyPI."""

        def test_the_stubs_have_only_modules_rheon_has(self):
            """The stub tree cannot drift from the real layout."""
            names = [
                ".".join(p.relative_to(STUBS).with_suffix("").parts).removesuffix(".__init__")
                for p in (STUBS / "rheon").rglob("*.py")
            ]
            out = run(
                f"""
                import importlib.util, json
                print(json.dumps([n for n in {names!r} if importlib.util.find_spec(n) is None]))
                """
            )
            self.assertEqual(out, [])

        def test_the_old_module_set_is_1_6_0s(self):
            out = run_old(
                """
                import json, pkgutil, polars_online
                names = [m.name for m in pkgutil.iter_modules(polars_online.__path__)]
                print(json.dumps(sorted(names)))
                """
            )
            self.assertEqual(set(out), OLD_MODULES)

        @per_package
        def test_every_name_1_6_0_exported_is_exported(self, name):
            old_all = run_old(
                "import json, polars_online; print(json.dumps(polars_online.__all__))"
            )
            out = run(
                f"""
                import json, polars_online as po
                names = {old_all!r}
                print(json.dumps([[n for n in names if n not in po.__all__],
                                  [n for n in names if not hasattr(po, n)]]))
                """,
                name,
            )
            self.assertEqual(out, [[], []])

        @per_package
        def test_every_old_warning_class_is_a_polars_online_warning_still(self, name):
            classes = run_old(
                """
                import json, polars_online as po
                print(json.dumps([n for n in po.__all__ if n.endswith("Warning")
                                  and issubclass(getattr(po, n), po.PolarsOnlineWarning)]))
                """
            )
            self.assertGreaterEqual(len(classes), 5)
            out = run(
                f"""
                import json, polars_online as po
                print(json.dumps([n for n in {classes!r}
                                  if not issubclass(getattr(po, n), po.PolarsOnlineWarning)]))
                """,
                name,
            )
            self.assertEqual(out, [])

        @per_package
        def test_online_equals_rheon_on_a_lazyframe_and_a_dataframe(self, name):
            out = run(
                fit(
                    """
                import json, warnings
                import polars_online as po
                from polars.testing import assert_frame_equal
                spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=20.0)
                df = frame(0, 60)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    old_lazy = df.lazy().online.fit_predict([spec]).collect()
                    old_eager = df.online.fit_predict([spec])
                new_lazy = df.lazy().rheon.fit_predict([spec]).collect()
                new_eager = df.rheon.fit_predict([spec])
                assert_frame_equal(old_lazy, new_lazy)
                assert_frame_equal(old_eager, new_eager)
                added = [c for c in new_lazy.columns if c not in df.columns]
                print(json.dumps([added, len(w), [str(x.category.__name__) for x in w]]))
                """
                ),
                name,
            )
            self.assertEqual(out[0], ["m"])
            self.assertEqual(out[1:], [1, ["DeprecationWarning"]])  # `.online` warns once

        @per_package
        def test_a_1_x_pickle_loads_and_continues(self, name):
            with tempfile.TemporaryDirectory() as tmp:
                blob = Path(tmp) / "bank_1_6_0.pkl"
                # polars-online 1.6.0, from PyPI, fits 40 rows, pickles the bank, then
                # continues on 20 more and says what that gave.
                expected = run_old(
                    fit(
                        """
                    import json, pickle, sys
                    import polars_online as po
                    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=20.0)
                    bank = po.ModelBank([spec])
                    bank.fit_predict(frame(0, 40))
                    with open(sys.argv[1], "wb") as f:
                        pickle.dump(bank, f)
                    print(json.dumps(bank.fit_predict(frame(40, 60))["m"].to_list()))
                    """
                    ),
                    str(blob),
                )
                self.assertGreater(blob.stat().st_size, 1000)
                self.assertIn(b"polars_online._bank", blob.read_bytes())
                out = run(
                    fit(
                        f"""
                    import json, pickle
                    import polars_online
                    import rheon.polars
                    with open({str(blob)!r}, "rb") as f:
                        bank = pickle.load(f)
                    assert type(bank) is rheon.polars.ModelBank, type(bank)
                    print(json.dumps(bank.fit_predict(frame(40, 60))["m"].to_list()))
                    """
                    ),
                    name,
                )
            self.assertEqual(out, expected)  # the same floats: states are unchanged

        def test_a_1_x_pickle_needs_the_forwarder_imported_first(self):
            """Not a promise, a measurement: unpickling names `polars_online._bank`, and
            the import of that name runs the package's `__init__` first."""
            with tempfile.TemporaryDirectory() as tmp:
                blob = Path(tmp) / "bank.pkl"
                run_old(
                    """
                    import json, pickle, sys
                    import polars_online as po
                    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=20.0)
                    with open(sys.argv[1], "wb") as f:
                        pickle.dump(po.ModelBank([spec]), f)
                    print(json.dumps(1))
                    """,
                    str(blob),
                )
                out = run(
                    f"""
                    import json, pickle
                    with open({str(blob)!r}, "rb") as f:   # no import of polars_online
                        bank = pickle.load(f)
                    import rheon.polars
                    print(json.dumps(type(bank) is rheon.polars.ModelBank))
                    """
                )
            self.assertTrue(out)
