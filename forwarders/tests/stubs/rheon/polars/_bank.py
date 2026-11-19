NAME = "rheon.polars._bank"


class ModelBank:
    """Pickles by reference to this module, as the real class does."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, frame):
        return sum(frame.data) * self.alpha

    def __eq__(self, other):
        return type(other) is type(self) and other.alpha == self.alpha


def thread_pool_size():
    return 1
