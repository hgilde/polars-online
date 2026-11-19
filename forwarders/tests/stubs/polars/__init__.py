"""A stub of polars: two frame classes and the namespace registry, nothing else."""

from polars import api  # noqa: F401


class DataFrame:
    def __init__(self, data=None):
        self.data = data


class LazyFrame:
    def __init__(self, data=None):
        self.data = data
