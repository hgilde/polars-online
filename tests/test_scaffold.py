"""Task 1 smoke tests: the extension builds, imports, and agrees with the Python package."""

import polars as pl

import polars_online


def test_native_extension_loads() -> None:
    assert polars_online.native_version() == polars_online.__version__


def test_schema_version_is_positive() -> None:
    assert polars_online.schema_version() >= 1


def test_polars_versions_match_the_pin() -> None:
    # Hard rule: the Cargo.toml and pyproject.toml Polars pins must match.
    assert pl.__version__ == "1.44.1"
