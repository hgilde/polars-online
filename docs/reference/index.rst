polars-online
=============

Streaming / online regression models for `Polars <https://pola.rs>`_: a
Rust core exposed as a chunk-fed ``ModelBank``, as a ``LazyFrame`` plan
that streams (``lf.online.fit_predict(specs)``), as a runner and CLI for
files, and as an expression for a frame in memory. Predictions are
out-of-sample by construction and the numbers are identical however the
model is called.

This is the API reference, built from the docstrings. The `README
<https://github.com/hgilde/polars-online#readme>`_ is the guide: what
streams and what does not, the models and their update equations, the
state-file workflow, and performance. Install with ``pip install
polars-online``.

.. toctree::
   :maxdepth: 2

   polars_online
   spec
   namespaces
   eval
   gram
   prep
