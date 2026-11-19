"""polars-online-compat is now rheon-compat: this package forwards to it.

``import polars_online`` warns once and hands back rheon's Polars interface,
so existing code and existing pickles keep working::

    from rheon import polars as rp     # the new spelling

Every ``polars_online.<module>`` is the same module object as its rheon
counterpart (``polars_online.ModelBank is rheon.polars.ModelBank``), and the
``.online`` frame namespace delegates to rheon's ``.rheon`` namespace.

Where a module lives in rheon is set by ``_MODULES`` below, and nowhere else
in this file: correct the table, not the code.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from types import ModuleType

#: Old module name (the part after ``polars_online.``) -> the rheon module that
#: now holds it. A module that imports Polars lives in ``rheon.polars``; the
#: rest in ``rheon``, which imports without Polars installed. The test for
#: "imports Polars" is ``grep -l "import polars\|from polars"`` on the 1.6.0
#: rename; it looks at each file only, not what the file imports in turn.
_MODULES: dict[str, str] = {
    # Polars-bound.
    "_bank": "rheon.polars._bank",
    "_check": "rheon.polars._check",
    "_duration": "rheon.polars._duration",
    "_errors": "rheon.polars._errors",
    "_formula": "rheon.polars._formula",
    "_frame": "rheon.polars._frame",
    "_spec": "rheon.polars._spec",
    "_spec_fields": "rheon.polars._spec_fields",
    "_warnings": "rheon.polars._warnings",
    "eval": "rheon.polars.eval",
    # TODO(2.0.0): rheon leaves gbdt out until it is fixed. If 2.0.0 does not
    # ship it, `polars_online.gbdt` is an ImportError; if it ships it, this is
    # where it is expected.
    "gbdt": "rheon.polars.gbdt",
    "gram": "rheon.polars.gram",
    "ops": "rheon.polars.ops",
    "sim": "rheon.polars.sim",
    "stream": "rheon.polars.stream",
    # No Polars import of their own, by the same test. TODO(2.0.0): each of the
    # following imports a Polars-bound module above, so rheon's split may have
    # put it in ``rheon.polars`` instead: _kwargs (_duration), _renamed
    # (_warnings), corr (_errors, _warnings, gram), spec (_spec), and
    # _spec_audit/_spec_changepoint/_spec_class/_spec_cluster/_spec_gradient/
    # _spec_regimes/_spec_robust/_spec_trend (_spec, some _duration).
    "_cpu_check": "rheon._cpu_check",
    "_kwargs": "rheon._kwargs",
    "_renamed": "rheon._renamed",
    "_spec_audit": "rheon._spec_audit",
    "_spec_changepoint": "rheon._spec_changepoint",
    "_spec_class": "rheon._spec_class",
    "_spec_cluster": "rheon._spec_cluster",
    "_spec_gradient": "rheon._spec_gradient",
    "_spec_regimes": "rheon._spec_regimes",
    "_spec_robust": "rheon._spec_robust",
    "_spec_trend": "rheon._spec_trend",
    "_spec_words": "rheon._spec_words",
    "corr": "rheon.corr",
    "spec": "rheon.spec",
    # The compiled module, renamed with the package.
    "_polars_online": "rheon._rheon",
}

#: The public modules, aliased as soon as this package is imported; every other
#: entry of the table is aliased on first import.
_PUBLIC_MODULES = ("corr", "eval", "gbdt", "gram", "ops", "sim", "spec", "stream")

#: Where the Polars interface lives, and the name of its frame namespace.
_TARGET = "rheon.polars"
_NAMESPACE = "rheon"

_PREFIX = f"{__name__}."
_MESSAGE = (
    "polars-online-compat is now rheon-compat: the package `polars_online` only forwards to it. "
    "Use `from rheon import polars as rp` and call `rp.`; `df.online` is now `df.rheon`."
)
_NAMESPACE_MESSAGE = "`.online` is now `.rheon`: `polars_online` only forwards to rheon."

__version__ = "2.0.0"

_warned: set[str] = set()


def _warn_once(key: str, message: str, stacklevel: int) -> None:
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def _exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a parent package is missing
        return False


def _resolve(suffix: str) -> str | None:
    """The rheon module a ``polars_online.<suffix>`` stands for, or ``None``."""
    real = _MODULES.get(suffix)
    return real if real is not None and _exists(real) else None


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves ``polars_online.<name>`` as the rheon module the table names.

    The import system would otherwise look for the file in this package's
    directory, which holds none. The module object is rheon's own, so a
    pickle that names ``polars_online._bank.ModelBank`` finds rheon's class.
    """

    def __init__(self) -> None:
        self._spec_of: dict[str, object] = {}

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        real = _resolve(fullname[len(_PREFIX) :])
        if real is None:
            return None
        spec = importlib.util.spec_from_loader(fullname, self)
        if spec is not None:
            spec.loader_state = real
        return spec

    def create_module(self, spec):
        module = importlib.import_module(spec.loader_state)
        # `module_from_spec` stamps this spec on the module; exec_module puts the
        # module's own back.
        self._spec_of[spec.name] = module.__spec__
        return module

    def exec_module(self, module) -> None:
        original = self._spec_of.pop(module.__spec__.name, None)
        if original is not None:
            module.__spec__ = original


def _install() -> None:
    if not any(isinstance(f, _Alias) for f in sys.meta_path):
        sys.meta_path.insert(0, _Alias())


def _load_target() -> ModuleType:
    if not _exists(_TARGET):
        raise ImportError(
            "polars_online 2.x forwards to rheon, which is not installed: "
            "`pip install 'rheon-compat[polars]'`"
        )
    return importlib.import_module(_TARGET)


def _forward(target: ModuleType) -> None:
    for suffix in _PUBLIC_MODULES:
        real = _resolve(suffix)
        if real is not None:
            sys.modules[_PREFIX + suffix] = importlib.import_module(real)
    names = getattr(target, "__all__", None)
    if names is None:
        names = [n for n in dir(target) if not n.startswith("_")]
    here = globals()
    for name in names:
        if name != "__version__" and hasattr(target, name):
            here[name] = getattr(target, name)
    # Submodules found above are attributes too, as after a real import.
    for suffix in _PUBLIC_MODULES:
        module = sys.modules.get(_PREFIX + suffix)
        if module is not None:
            here[suffix] = module
    exported = set(names)
    here["__all__"] = sorted({n for n in here if n in exported} | {"__version__"})


class _OnlineNamespace:
    """``df.online`` / ``lf.online``: every attribute is read from ``.rheon``."""

    def __init__(self, frame) -> None:
        self._frame = frame

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        _warn_once("namespace", _NAMESPACE_MESSAGE, stacklevel=2)
        return getattr(getattr(self._frame, _NAMESPACE), name)

    def __dir__(self):
        return dir(getattr(self._frame, _NAMESPACE))


def _register_namespace() -> None:
    import polars as pl

    for frame_type, register in (
        (pl.LazyFrame, pl.api.register_lazyframe_namespace),
        (pl.DataFrame, pl.api.register_dataframe_namespace),
    ):
        # rheon has no `.online` of its own; if one is there anyway (another
        # package's, or a second copy of this one), leave it alone.
        if not hasattr(frame_type, "online"):
            register("online")(_OnlineNamespace)


_install()
_forward(_load_target())
_register_namespace()
_warn_once("import", _MESSAGE, stacklevel=2)
