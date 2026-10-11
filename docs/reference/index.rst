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

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - page
     - what it documents
   * - :doc:`polars_online`
     - ``ModelBank``, the package's own functions (``fit_predict``,
       ``predict``, ``unnest``, ``target``), its warnings and its versions
   * - :doc:`spec`
     - the model builders, the parameters every spec shares, and the fields
       each model writes
   * - :doc:`namespaces`
     - ``lf.online`` and ``df.online``: a bank run inside a query, or on a
       frame in memory
   * - :doc:`eval`
     - scores of a bank's output, over a frame or one chunk at a time, and
       tests of one spec's errors against another's
   * - :doc:`gram`
     - the running sums a bank exports: pooled, narrowed, solved, and
       given standard errors
   * - :doc:`corr`
     - correlation matrices: read, repaired, summarised, told from noise and
       scored
   * - :doc:`stream`
     - transforms of a stream before a bank: windows as columns, labels that
       arrive late, and series that tick at their own times
   * - :doc:`ops`
     - the window operators, Polars expressions over a stream's clock that
       look back or ahead
   * - :doc:`sim`
     - a seeded simulator of correlation regimes, its truth beside the data
   * - :doc:`gbdt`
     - XGBoost's and LightGBM's training inputs, binned from a stream

.. toctree::
   :maxdepth: 2
   :hidden:

   polars_online
   spec
   namespaces
   eval
   gram
   corr
   stream
   ops
   sim
   gbdt
