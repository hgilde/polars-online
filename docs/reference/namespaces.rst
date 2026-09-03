The ``online`` namespaces
=========================

Importing ``polars_online`` registers ``online`` on ``pl.LazyFrame``,
``pl.DataFrame`` and ``pl.Expr``. The frame namespaces run a model bank over
the frame -- as a plan that streams, or eagerly; the expression namespace
runs one model over the calling column, for a frame in memory.

``lf.online`` / ``df.online``
-----------------------------

.. automodule:: polars_online._frame
   :no-members:

.. autoclass:: polars_online._frame.LazyFrameOnlineNamespace
   :members:

.. autoclass:: polars_online._frame.DataFrameOnlineNamespace
   :members:

``pl.col("y").online``
----------------------

.. automodule:: polars_online._expr
   :no-members:

.. autoclass:: polars_online._expr.OnlineNamespace
   :members:
