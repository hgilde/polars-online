"""Type stub for the pyo3 module (crates/online-py/src/lib.rs). The wrappers in
this package are what users call; this is what lets a type checker follow them
into the native calls. ``tests/test_kwargs_typing.py`` checks it names every
attribute the built module has, so it cannot go stale silently."""

import polars as pl

class RefreshTime:
    """Refresh-time sampling (E58), fed frames of the long input in order."""

    def __init__(
        self,
        names: list[str],
        series: str,
        clock: str,
        value: str,
        group: str | None = None,
        pairs: bool = False,
        keep: list[str] | None = None,
    ) -> None: ...
    @staticmethod
    def load_bytes(
        state: bytes,
        names: list[str],
        series: str,
        clock: str,
        value: str,
        group: str | None = None,
        pairs: bool = False,
        keep: list[str] | None = None,
    ) -> RefreshTime: ...
    def save(self, path: str) -> None: ...
    def feed(self, df: pl.DataFrame, limit: int | None = None) -> pl.DataFrame: ...

class Windows:
    """``po.stream.with_windows`` (task 78), fed a stream's chunks in order."""

    def __init__(self, config: str, input: pl.DataFrame) -> None: ...
    @staticmethod
    def load_bytes(state: bytes, config: str, input: pl.DataFrame) -> Windows: ...
    def save(self, path: str, skip_on_resume: int = 0, input_ended: bool = False) -> None: ...
    def feed(self, df: pl.DataFrame, limit: int | None = None) -> pl.DataFrame: ...
    def finish(self) -> pl.DataFrame: ...
    def needed(self) -> list[str]: ...
    def held(self) -> int: ...
    def consumed(self) -> int: ...
    def queued(self) -> int: ...

class ArrowStruct:
    """One spec's output struct, over the Arrow PyCapsule interface.

    Handed back by ``ModelBank.fit_predict_arrow``. A reader of
    ``__arrow_c_array__`` takes it directly -- ``pl.Series(obj)``,
    ``pa.array(obj)`` -- and duckdb, which reads streams, through
    ``pl.Series(obj)``. Exporting consumes it, so it can be read once.

    .. warning::
        The Arrow output is considered **unstable**: it may change in any
        release without that counting as a breaking change. Under
        ``POLARS_ONLINE_WARN_UNSTABLE=1`` the calls that return one raise
        ``polars_online.UnstableWarning``.
    """

    @property
    def name(self) -> str: ...
    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...

class ModelBank:
    def __init__(self, specs_json: str) -> None: ...
    def fit_predict(
        self, df: pl.DataFrame, row_base: int = 0, learn_only: bool = False
    ) -> list[pl.Series]: ...
    def predict(self, df: pl.DataFrame, row_base: int = 0) -> list[pl.Series]: ...
    def fit_predict_arrow(self, df: pl.DataFrame) -> list[ArrowStruct]: ...
    def predict_arrow(self, df: pl.DataFrame) -> list[ArrowStruct]: ...
    def save(self, path: str) -> None: ...
    def save_bytes(self) -> bytes: ...
    def save_json_string(self, pretty: bool = True) -> str: ...
    @staticmethod
    def load_bytes(bytes: bytes, specs_json: str | None = None) -> ModelBank: ...
    def take_notices(self) -> list[str]: ...
    def output_fields(self) -> list[list[str]]: ...
    def solve_failures(self) -> list[list[tuple[str | None, int]]]: ...
    def gram(
        self,
        spec: int,
        group: list[str | None] | None = None,
        float32: bool = False,
        packed: bool = False,
        columns: list[str] | None = None,
    ) -> list[
        tuple[
            tuple[
                str | None,  # group
                str,  # instance
                int,  # k
                float,  # weight_sum
                float | None,  # n_kish
                list[float],  # means
                # comoments: the bytes of k*k row-major, or k(k+1)/2 packed,
                # float64s or float32s (docs/PLAN.md task 229)
                bytearray,
                list[list[float]],  # cross_moments, one row per target
                list[float],  # target_weights
                list[float],  # target_means
                list[float],  # target_vars
                list[float | None],  # target_n_kish
            ],
            # (lags, L*k*k cross-moments as bytes), or None without lags (E56)
            tuple[list[int], bytearray] | None,
            # this Gram's targets, as indices into the spec's, each one's
            # column means over its own rows (docs/PLAN.md task 81), and its
            # cross-moments centred at them (review 2026-09-12, N4)
            tuple[list[int], list[list[float]], list[list[float]]],
        ]
    ]: ...
    def coef(
        self, spec: int, group: list[str | None] | None = None
    ) -> list[tuple[str | None, str, float, list[float] | None]]: ...
    def last_row(
        self, spec: int, group: list[str | None] | None = None
    ) -> tuple[list[str | None], pl.Series]: ...
    def summary(self, specs: list[int], group: list[str | None] | None = None) -> pl.DataFrame: ...
    def describe(self, spec: int, group: list[str | None] | None = None) -> pl.DataFrame: ...
    def marginal(self, spec: int, group: list[str | None] | None = None) -> pl.DataFrame: ...
    def audit(
        self,
        spec: int,
        group: list[str | None] | None = None,
        table: str = "columns",
        pooled: bool = False,
    ) -> pl.DataFrame: ...
    def closed_groups(self, spec: int | None = None, drop: bool = True) -> pl.DataFrame: ...
    def spec_names(self) -> list[str]: ...
    def specs_json(self) -> str: ...
    def groups(self, specs: list[int]) -> pl.DataFrame: ...
    def group_counts(self) -> list[int]: ...
    def drop_groups(self, keys: list[str | None], spec: int | None = None) -> int: ...
    def last_clocks(
        self,
    ) -> list[list[tuple[str | None, float | None, int | None, int | None]]]: ...
    def rows_fed(self) -> int: ...

def validate_spec(spec_json: str) -> None: ...
def parse_duration(text: str) -> int: ...
def format_duration(ns: int) -> str: ...
def spec_clock_fields() -> tuple[
    list[tuple[str, list[str]]], list[tuple[str, list[str]]], list[tuple[str, list[str]]]
]: ...
def format_of_path(path: str) -> str: ...
def default_chunk_size() -> int: ...
def spec_output_index(spec_json: str) -> str: ...
def spec_coef_fields(spec_json: str) -> str: ...
def spec_output_fields(spec_json: str) -> list[str]: ...
def resolved_defaults(spec_json: str) -> str: ...
def native_version() -> str: ...
def schema_version() -> int: ...
def thread_pool_size() -> int: ...
def model_kinds() -> list[str]: ...

# A Gram as the fits in `polars_online.gram` hand it over: `k` and
# `weight_sum`, then `means`, `comoments`, `cross_moments`, `means_by_target`,
# `cross_centred`, `target_weights`, `target_means`, `target_vars` and
# `target_n_kish`, each a C-contiguous float64 numpy array.
_GramIn = tuple[int, float, object, object, object, object, object, object, object, object, object]
# A lasso path: the knots' penalties, their coefficients (the bytes of
# native-endian float64s, flat), the active columns at each, and why it stopped.
_PathOut = tuple[list[float], bytearray, list[list[int]], str]

def gram_lars_paths(
    grams: list[_GramIn],
    targets: list[list[int]],
    slots: list[int],
    icept: int | None,
    weights: list[float],
    max_steps: int | None,
    max_active: int | None,
) -> list[list[_PathOut]]: ...
def gram_cd_path(
    gram: _GramIn,
    target: int,
    slots: list[int],
    icept: int | None,
    penalties: list[float],
    l1_ratio: float,
    weights: list[float],
    max_iter: int,
    tol: float,
) -> list[float]: ...

# A ridge fit: `coef`, `se` and `t` (each the bytes of native-endian float64s),
# then `resid_var`, `sigma2`, `r2` and `n`.
_RidgeOut = tuple[bytearray, bytearray, bytearray, float, float, float, float]

def gram_ridge_subsets(
    grams: list[_GramIn],
    subsets: list[list[int]],
    targets: list[int],
    slots: list[int],
    icept: int | None,
    ridge: float,
    standardize: bool,
) -> list[list[_RidgeOut]]: ...
def gram_path_subsets(
    grams: list[_GramIn],
    subsets: list[list[int]],
    targets: list[int],
    slots: list[int],
    icept: int | None,
    weights: list[float],
    max_steps: int | None,
    max_active: int | None,
) -> list[list[_PathOut]]: ...
