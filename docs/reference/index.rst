polars-online
=============

This is the API reference of polars-online, built from its docstrings, for
a reader who has met the library in the `README
<https://github.com/hgilde/polars-online#readme>`_ and wants one call's
parameters, output and refusals. Install it with ``pip install
polars-online``.

polars-online fits models online for `Polars <https://pola.rs>`_: linear
models, streaming moments, clustering and regime detection, for data too
large to hold in memory at once. The Rust core runs three ways: inside a
Polars query (``lf.online.fit_predict(specs)``), fed chunks by a
``ModelBank``, and from the ``polars-online`` command line. Every row is
predicted before it is learned from, and the numbers are the same whichever
way the bank runs.

Each module's page lists its functions and classes by family, and each
function and class has a page of its own. The guides, which the README's
map leads to, show each call at work: the models, which calls stream, the
state-file workflow, and performance. The `feature index
<https://github.com/hgilde/polars-online/blob/main/docs/FEATURES.md>`_ lists
every function and class with a line each. Each model's update equations
are in its builder's docstring.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - page
     - what it documents
   * - :doc:`polars_online`
     - ``ModelBank``, the package's own functions (``fit_predict``,
       ``predict``, ``unnest``, ``target``), its errors, its warnings and
       its versions
   * - :doc:`spec`
     - the model builders by family (regressions, robust regressions,
       covariance and regimes, changepoints and sequential tests,
       clustering and classification), the parameters every spec shares, and the fields each
       model writes
   * - :doc:`namespaces`
     - ``lf.online`` and ``df.online``: a bank run inside a query, or on a
       frame in memory
   * - :doc:`stream`
     - transforms of a stream before a bank: windows as columns, targets whose
       values arrive late, and series that tick at their own times
   * - :doc:`ops`
     - the window operators, Polars expressions over a stream's clock that
       look back or ahead
   * - :doc:`eval`
     - scores of a bank's output, over a frame or one chunk at a time, and
       tests of one spec's residuals against another's
   * - :doc:`gram`
     - the Grams a bank exports: pooled, narrowed, solved, and
       given standard errors
   * - :doc:`corr`
     - correlation matrices: read, repaired, summarized, told from noise and
       scored
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
   stream
   ops
   eval
   gram
   corr
   sim
   gbdt
