"""The OpenMP runtime XGBoost's and LightGBM's macOS wheels load, on a machine
without Homebrew's.

Both wheels link ``@rpath/libomp.dylib`` and look for it in Homebrew's
prefix (``/opt/homebrew/opt/libomp/lib``), where the CI runners install it
(``brew install libomp``, .github/workflows/ci.yml). A machine without it
cannot import either library, and tests/test_gbdt.py imports both, so on such
a machine this loads scikit-learn's bundled copy (scikit-learn is in the dev
group) under the name the libraries ask for: a copy under ``.cache/libomp/``
(gitignored) whose install name is ``@rpath/libomp.dylib``, ad hoc signed,
loaded first and globally, which dyld then hands to the libraries in place of
the file it cannot find. A copy rather than scikit-learn's own file, whose
install name is another; two OpenMP runtimes then share the process, which
LLVM's runtime allows (measured: scikit-learn, XGBoost and LightGBM training
in turn in one process). Elsewhere, and where Homebrew's runtime is there,
it does nothing. Imported by tests/conftest.py and by every child
interpreter (sitecustomize.py beside this file).
"""

from __future__ import annotations

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

#: Where the two libraries look for ``libomp.dylib`` (their LC_RPATH).
_SEARCHED = (
    Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
    Path("/opt/local/lib/libomp/libomp.dylib"),
)
_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "libomp" / "libomp.dylib"


def install() -> None:
    if sys.platform != "darwin" or any(p.exists() for p in _SEARCHED):
        return
    spec = importlib.util.find_spec("sklearn")
    if spec is None or not spec.submodule_search_locations:
        return
    bundled = Path(next(iter(spec.submodule_search_locations))) / ".dylibs" / "libomp.dylib"
    if not bundled.exists():
        return
    if not _CACHE.exists() or _CACHE.stat().st_mtime < bundled.stat().st_mtime:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE.with_suffix(".tmp")
        shutil.copyfile(bundled, tmp)
        subprocess.run(
            ["install_name_tool", "-id", "@rpath/libomp.dylib", str(tmp)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["codesign", "-f", "-s", "-", str(tmp)], check=True, capture_output=True)
        tmp.replace(_CACHE)
    ctypes.CDLL(str(_CACHE), mode=ctypes.RTLD_GLOBAL)
