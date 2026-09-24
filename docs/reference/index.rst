polars-online
=============

Online model fitting for `Polars <https://pola.rs>`_: linear models,
streaming moments, clustering and regime detection, for data too large to
hold in memory at once. The Rust core runs three ways: inside a Polars
query (``lf.online.fit_predict(specs)``), fed chunks by a ``ModelBank``,
and from the ``online`` command line. Every row is predicted before it is
learned from, and the numbers are the same whichever way the bank runs.

This is the API reference, built from the docstrings. The `README
<https://github.com/hgilde/polars-online#readme>`_ is the guide: the
models and their update equations, which calls stream, the state-file
workflow, and performance. Install with ``pip install polars-online``.

.. toctree::
   :maxdepth: 2

   polars_online
   spec
   namespaces
   eval
   gram
   corr
   prep
   sim
