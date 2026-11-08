The ``online`` namespaces
=========================

Importing ``polars_online`` registers ``online`` on ``pl.LazyFrame`` and
``pl.DataFrame``. Each runs a model bank over the query's or the frame's rows
(``fit_predict``, ``predict``), turns the bank's output into one column per
field (``unnest``), and adds window operators' values as columns
(``with_windows``). ``lf.online`` does it inside a query, which streams its
rows through when it runs, and ``df.online`` at once, on a frame in memory.

``lf.online`` / ``df.online``
-----------------------------

.. automodule:: polars_online._frame
   :no-members:

.. entries:: namespaces
