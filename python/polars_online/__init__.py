"""Streaming / online regression models for Polars.

Three entry points share one Rust core (see ``docs/PLAN.md``):

1. an expression plugin -- ``pl.col("y").online.<model>(...)`` (task 8);
2. :class:`ModelBank`, chunk-fed, memory O(state) not O(data);
3. the ``online`` CLI, parquet in -> parquet out (task 15).
"""

from polars_online import spec
from polars_online._bank import ModelBank
from polars_online._polars_online import native_version, schema_version

__version__ = "0.1.0"

__all__ = ["ModelBank", "__version__", "native_version", "schema_version", "spec"]
