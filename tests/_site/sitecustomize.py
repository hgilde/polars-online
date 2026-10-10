"""Imported by every interpreter whose path holds this directory (tests/child.py
puts it on PYTHONPATH): the child runs as a user without pyarrow does, and on
macOS finds the OpenMP runtime XGBoost and LightGBM load (libomp.py)."""

import libomp
import without_pyarrow

without_pyarrow.install()
libomp.install()
