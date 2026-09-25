"""Imported by every interpreter whose path holds this directory (tests/child.py
puts it on PYTHONPATH): the child runs as a user without pyarrow does."""

import without_pyarrow

without_pyarrow.install()
