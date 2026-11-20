"""The warning classes, as rheon 2.0.0 has them: no Polars parent."""

NAME = "rheon._warnings"


class RheonWarning(Warning):
    pass


class RheonDeprecationWarning(RheonWarning, DeprecationWarning):
    pass


class UnstableWarning(RheonWarning, UserWarning):
    pass


class ReadinessWarning(RheonWarning, UserWarning):
    pass
