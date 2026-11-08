``polars_online.spec``
======================

.. automodule:: polars_online.spec
   :no-members:

Regressions
-----------

Models that predict each target from the features, or from the target's own
past.

.. entries:: regressions

Robust regressions
------------------

Regressions whose fit an outlying row cannot pull far.

.. entries:: robust

Covariance and regimes
----------------------

Moments and correlations of a stream's columns, what the columns hold, and
which regime the stream is in.

.. entries:: covariance

Changepoints and sequential tests
---------------------------------

Tests that read a stream row by row and say when it has changed, or whether a
sign, or one spec against another, favors one side.

.. entries:: sequential

Clustering and classification
-----------------------------

Models that place each row in a cluster or a class.

.. entries:: clustering

Reading a spec's fields
-----------------------

The names of the fields a bank's output carries, and where each coefficient
sits in ``coef``.

.. entries:: fields

Typed keyword sets
------------------

The keywords every builder shares, as a typed dictionary a type checker can
read.

.. automodule:: polars_online._kwargs
   :members:
   :undoc-members:
