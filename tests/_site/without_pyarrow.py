"""pyarrow is unimportable, as it is for a user of the package, whose one
dependency is polars.

pyarrow is installed with the dev group for tests/test_pyarrow_interop.py,
which runs its pyarrow half in child interpreters of its own. Everywhere
else -- the pytest process (tests/conftest.py) and the child interpreters
the other tests spawn (tests/child.py, through sitecustomize.py beside this
file) -- this finder makes pyarrow read as absent, both to ``import
pyarrow`` and to ``importlib.util.find_spec``, which is how polars probes
for it. A ``None`` in ``sys.modules`` would not do: polars reads any entry
there as loaded. One difference from a real absence: ``find_spec`` raises
here where it would return ``None``; polars, pandas and duckdb all probe
inside ``try``, and erring on the strict side is the right side, since a
spec with a failing loader would read as present.
"""

from __future__ import annotations

import sys


class WithoutPyarrow:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError(
                f"No module named {name!r} (tests/_site/without_pyarrow.py)", name=name
            )
        return None


def install() -> None:
    assert "pyarrow" not in sys.modules, "pyarrow was imported before it could be blocked"
    if WithoutPyarrow not in sys.meta_path:
        sys.meta_path.insert(0, WithoutPyarrow)
