The ``online`` namespaces
=========================

Importing ``polars_online`` registers ``online`` on ``pl.LazyFrame`` and
``pl.DataFrame``. Both run a model bank over the frame -- as a plan that
streams, or eagerly.

``lf.online`` / ``df.online``
-----------------------------

.. automodule:: polars_online._frame
   :no-members:

.. autoclass:: polars_online._frame.LazyFrameOnlineNamespace
   :members:

.. autoclass:: polars_online._frame.DataFrameOnlineNamespace
   :members:
