"""The forwarder against a stub rheon and a stub polars.

Nothing real is installed: `tests/stubs` holds the two stubs, and `src` of each
forwarder is put on the path. Each check runs in a fresh interpreter, because
what is under test is what happens on first import (a warning once, modules
registered once). Written as `unittest` cases so both pytest and
`python3 -m unittest` run it.

    PYTHONPATH=forwarders/tests/stubs:forwarders/polars-online/src \
        python3 -m pytest forwarders/tests -q
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUBS = ROOT / "tests" / "stubs"
PACKAGES = {"polars-online": ROOT / "polars-online", "polars-online-compat": ROOT / "polars-online-compat"}

#: The modules of polars_online 1.6.0 and its rename, from `ls python/polars_online`.
OLD_MODULES = {
    "_bank", "_check", "_cpu_check", "_duration", "_errors", "_formula", "_frame",
    "_kwargs", "_polars_online", "_renamed", "_spec", "_spec_audit", "_spec_changepoint",
    "_spec_class", "_spec_cluster", "_spec_fields", "_spec_gradient", "_spec_regimes",
    "_spec_robust", "_spec_trend", "_spec_words", "_warnings", "corr", "eval", "gbdt",
    "gram", "ops", "sim", "spec", "stream",
}  # fmt: skip


def run(code: str, package: str = "polars-online", *, warn: str = "always") -> dict:
    """Run `code` in a fresh interpreter; it prints one JSON value, returned."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(STUBS), str(PACKAGES[package] / "src")])
    done = subprocess.run(
        [sys.executable, "-W", warn, "-c", textwrap.dedent(code)],
        env=env, capture_output=True, text=True, cwd=str(STUBS),
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
        self.assertIn(name + " is now " + ("rheon-compat" if "compat" in name else "rheon"), out[0][1])

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
                except ModuleNotFoundError:
                    res[old] = None
            print(json.dumps(res))
            """,
            name,
        )
        # The stub has no gbdt (a 2.0.0 that leaves it out): absent, not wrong.
        self.assertEqual({k for k, v in out.items() if v is None}, {"gbdt"})
        self.assertEqual({k for k, v in out.items() if v is False}, set())

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

    @per_package
    def test_pickle_by_the_old_module_name_loads(self, name):
        out = run(
            """
            import json, pickle, polars_online, rheon.polars
            # What a polars-online 1.x pickle holds: a global named by the old module.
            cls = pickle.loads(b"cpolars_online._bank\\nModelBank\\n.")
            cls2 = pickle.loads(b"cpolars_online.ops\\nNAME\\n.")
            obj = pickle.loads(pickle.dumps(rheon.polars.ModelBank(0.25)))
            print(json.dumps([cls is rheon.polars.ModelBank, cls2, obj.alpha]))
            """,
            name,
        )
        self.assertEqual(out, [True, "rheon.polars.ops", 0.25])

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
            print(json.dumps([polars.DataFrame([1.0]).online.fit(), polars.LazyFrame([1.0]).online.fit()]))
            """
        )
        self.assertEqual(out, ["theirs", 0.5])  # the lazy frame gets the forwarder's

    def test_missing_rheon_polars_names_the_extra(self):
        for name, extra in (("polars-online", "rheon[polars]"), ("polars-online-compat", "rheon-compat[polars]")):
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
    def test_import_rheon_alone_does_not_import_polars(self):
        out = run("import json, sys, rheon; print(json.dumps(['polars' in sys.modules, rheon.__version__]))")
        self.assertEqual(out, [False, "2.0.0"])

    def test_importing_rheon_polars_does_import_it(self):
        out = run("import json, sys, rheon.polars; print(json.dumps('polars' in sys.modules))")
        self.assertTrue(out)


class Table(unittest.TestCase):
    def test_keys_are_the_old_module_set(self):
        for name in PACKAGES:
            self.assertEqual(set(table(name)), OLD_MODULES)

    def test_targets_are_rheon_modules(self):
        for name in PACKAGES:
            for old, new in table(name).items():
                self.assertRegex(new, r"^rheon(\.polars)?\.[A-Za-z_]+$", old)
        self.assertEqual(table("polars-online")["_polars_online"], "rheon._rheon")

    def test_the_two_packages_differ_only_in_strings(self):
        a = (PACKAGES["polars-online"] / "src/polars_online/__init__.py").read_text().splitlines()
        b = (PACKAGES["polars-online-compat"] / "src/polars_online/__init__.py").read_text().splitlines()
        self.assertEqual(len(a), len(b))
        changed = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(changed), 3)
        for x, y in changed:
            self.assertIn("compat", y)
            self.assertNotIn("compat", x)

    def test_pyproject_depends_on_the_extra(self):
        import tomllib

        for name, dep in (("polars-online", "rheon[polars]"), ("polars-online-compat", "rheon-compat[polars]")):
            meta = tomllib.loads((PACKAGES[name] / "pyproject.toml").read_text())["project"]
            self.assertEqual(meta["version"], "2.0.0")
            self.assertTrue(meta["dependencies"][0].startswith(dep + ">=2.0.0"), meta["dependencies"])


if __name__ == "__main__":
    unittest.main()
