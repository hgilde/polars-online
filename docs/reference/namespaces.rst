The ``online`` namespaces
=========================

Importing ``polars_online`` registers ``online`` on ``pl.LazyFrame`` and
``pl.DataFrame``. Each runs a model bank over the frame (``fit_predict``,
``predict``, ``unnest``) and adds windows over it as columns
(``with_windows``). ``lf.online`` does it inside a query, which streams its
rows through when it runs, and ``df.online`` at once, on a frame in memory.

``lf.online`` / ``df.online``
-----------------------------

.. automodule:: polars_online._frame
   :no-members:

.. autoclass:: polars_online._frame.LazyFrameOnlineNamespace
   :members:

.. autoclass:: polars_online._frame.DataFrameOnlineNamespace
   :members:
