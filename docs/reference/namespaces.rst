The ``rheon`` namespaces
========================

Importing ``rheon.polars`` registers ``rheon`` on ``pl.LazyFrame`` and
``pl.DataFrame``. Each runs a model bank over the query's or the frame's rows
(``fit_predict``, ``predict``), turns the bank's output into one column per
field (``unnest``), and adds window operators' values as columns
(``with_windows``). ``lf.rheon`` does it inside a query, which streams its
rows through when it runs, and ``df.rheon`` at once, on a frame in memory.

``lf.rheon`` / ``df.rheon``
-----------------------------

.. automodule:: rheon.polars._frame
   :no-members:

.. entries:: namespaces
