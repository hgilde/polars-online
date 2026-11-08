"""The API reference's families: every public object, placed once.

Each module page of the reference (``polars_online.html``, ``spec.html``,
``eval.html`` and the rest) keeps its address and its module docstring, and
lists its objects grouped by family, each with a link to a page of its own.
Those pages are generated at build time (``conf.py``) from the lists here.

A module page once documented its objects in place, so a link such as
``spec.html#polars_online.spec.ewridge`` pointed into it. The README of every
release already on PyPI carries those links, so each module page keeps an
anchor with every old id beside the link to the object's new page
(``tests/test_reference_pages.py`` holds them). The build fails if a public
object is placed in no family, or in two (``conf.py``, ``_check_placed``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Family:
    """One family of the reference: its slug (the ``.. entries::`` argument),
    the module page that lists it, and its objects by dotted name. A class in
    ``paged`` gets a page for each public method and property, linked from
    its own page; any other class is documented whole on one page."""

    slug: str
    page: str
    entries: tuple[str, ...]
    paged: frozenset[str] = field(default_factory=frozenset)


_PO = "polars_online"
_SPEC = "polars_online.spec"
_FRAME = "polars_online._frame"

FAMILIES: tuple[Family, ...] = (
    Family(
        "bank",
        "polars_online",
        (
            f"{_PO}.ModelBank",
            f"{_PO}.fit_predict",
            f"{_PO}.predict",
            f"{_PO}.unnest",
            f"{_PO}.target",
            f"{_PO}.Target",
            f"{_PO}.FormulaTarget",
            f"{_PO}.ArrowStruct",
            f"{_PO}.thread_pool_size",
            f"{_PO}.native_version",
            f"{_PO}.schema_version",
        ),
        paged=frozenset({f"{_PO}.ModelBank"}),
    ),
    Family(
        "errors",
        "polars_online",
        (
            f"{_PO}.ColumnNotFoundError",
            f"{_PO}.OutputColumnNotFoundError",
            f"{_PO}.WrongTypeError",
            f"{_PO}.RefusedValueError",
        ),
    ),
    Family(
        "warnings",
        "polars_online",
        (
            f"{_PO}.PolarsOnlineWarning",
            f"{_PO}.ReadinessWarning",
            f"{_PO}.OrderNotGuaranteedWarning",
            f"{_PO}.ConsumedSourceWarning",
            f"{_PO}.PolarsOnlineDeprecationWarning",
            f"{_PO}.UnstableWarning",
        ),
    ),
    Family(
        "namespaces",
        "namespaces",
        (f"{_FRAME}.LazyFrameOnlineNamespace", f"{_FRAME}.DataFrameOnlineNamespace"),
        paged=frozenset(
            {f"{_FRAME}.LazyFrameOnlineNamespace", f"{_FRAME}.DataFrameOnlineNamespace"}
        ),
    ),
    Family(
        "regressions",
        "spec",
        tuple(
            f"{_SPEC}.{m}"
            for m in ("ewridge", "rls", "lasso", "kalman", "sgd", "pa", "ftrl", "holt")
        ),
    ),
    Family("robust", "spec", (f"{_SPEC}.huber", f"{_SPEC}.quantile")),
    Family(
        "covariance",
        "spec",
        tuple(f"{_SPEC}.{m}" for m in ("ew_cov", "marginal", "deco", "rcov", "audit", "hmm")),
    ),
    Family(
        "sequential",
        "spec",
        tuple(f"{_SPEC}.{m}" for m in ("seqtest", "corrchange", "bocpd")),
    ),
    Family(
        "clustering",
        "spec",
        tuple(f"{_SPEC}.{m}" for m in ("kmeans", "micro", "ew_class")),
    ),
    Family(
        "fields",
        "spec",
        tuple(
            f"{_SPEC}.{m}" for m in ("output_fields", "output_index", "coef_fields", "coef_index")
        ),
    ),
    Family(
        "stream",
        "stream",
        tuple(f"{_PO}.stream.{m}" for m in ("with_windows", "embargo", "refresh_time")),
    ),
    Family(
        "ops",
        "ops",
        tuple(
            f"{_PO}.ops.{m}"
            for m in (
                "ewm_mean",
                "ewm_sum",
                "ewm_var",
                "ewm_std",
                "ewm_quantile",
                "ewm_rate",
                "increment",
            )
        ),
    ),
    Family(
        "eval-scores",
        "eval",
        tuple(
            f"{_PO}.eval.{m}"
            for m in ("unpack", "metrics", "window_metrics", "sums", "merge_sums", "from_sums")
        ),
    ),
    Family(
        "eval-compare",
        "eval",
        tuple(
            f"{_PO}.eval.{m}"
            for m in ("compare_specs", "grid_search", "diebold_mariano", "clark_west", "seqtest")
        ),
    ),
    Family(
        "gram-pool",
        "gram",
        tuple(f"{_PO}.gram.{m}" for m in ("Gram", "merge", "from_row", "subset", "correlation")),
    ),
    Family(
        "gram-solve",
        "gram",
        tuple(
            f"{_PO}.gram.{m}"
            for m in (
                "solve",
                "lasso_path",
                "lars_path",
                "lars_paths",
                "solve_subsets",
                "coef_stats",
                "vif",
                "condition",
            )
        ),
    ),
    Family(
        "corr-matrix",
        "corr",
        tuple(f"{_PO}.corr.{m}" for m in ("nearest", "shrink", "to_z", "from_z", "fisher_se")),
    ),
    Family(
        "corr-structure",
        "corr",
        tuple(
            f"{_PO}.corr.{m}"
            for m in (
                "equicorr",
                "equicorr_row",
                "equicorr_loglik",
                "absorption",
                "absorption_shift",
                "spectral",
                "from_spectral",
                "block_means",
                "from_blocks",
            )
        ),
    ),
    Family(
        "corr-noise",
        "corr",
        tuple(
            f"{_PO}.corr.{m}"
            for m in ("mp_edge", "mp_density", "signal_share", "loss", "epps_invert")
        ),
    ),
    Family("sim", "sim", (f"{_PO}.sim.regimes",)),
    Family("gbdt", "gbdt", (f"{_PO}.gbdt.xgboost_matrix", f"{_PO}.gbdt.lightgbm_dataset")),
)

#: The module pages, in the reader's order, each the old address of the
#: objects its families list.
PAGES: tuple[str, ...] = (
    "polars_online",
    "spec",
    "namespaces",
    "stream",
    "ops",
    "eval",
    "gram",
    "corr",
    "sim",
    "gbdt",
)


def title(name: str) -> str:
    """An entry's page title: its dotted name as a reader calls it, with the
    private module of the frame namespaces read as the namespace."""
    for cls, spelled in (
        ("LazyFrameOnlineNamespace", "LazyFrame.online"),
        ("DataFrameOnlineNamespace", "DataFrame.online"),
    ):
        if name.startswith(f"{_FRAME}.{cls}"):
            return spelled + name[len(f"{_FRAME}.{cls}") :]
    return name


def page_name(name: str, kind: str) -> str:
    """The file an entry's page is written to, which is its URL. Two names
    that differ only in case, such as ``polars_online.Target`` and
    ``polars_online.target``, would be one file on macOS and Windows, so a
    class whose name has a twin of another case takes ``-class`` after it."""
    lowered = [n.lower() for f in FAMILIES for n in f.entries]
    if kind == "class" and lowered.count(name.lower()) > 1:
        return f"{name}-class"
    return name


def family(slug: str) -> Family:
    for f in FAMILIES:
        if f.slug == slug:
            return f
    raise KeyError(slug)
