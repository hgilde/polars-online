"""Namespace registration as polars does it: the class is built per access."""

from __future__ import annotations

import polars


def _register(cls):
    def decorator_for(name):
        def decorator(namespace):
            if hasattr(cls, name):
                raise AssertionError(f"stub polars: {cls.__name__}.{name} registered twice")

            class _Descriptor:
                def __get__(self, instance, owner=None):
                    return namespace(instance) if instance is not None else namespace

            setattr(cls, name, _Descriptor())
            return namespace

        return decorator

    return decorator_for


def register_dataframe_namespace(name):
    return _register(polars.DataFrame)(name)


def register_lazyframe_namespace(name):
    return _register(polars.LazyFrame)(name)
