``polars_online``
=================

.. automodule:: polars_online
   :no-members:

The model bank
--------------

Run a model bank over a query, a frame or chunks you feed it, and turn the
output column it adds into columns, or hand it over as Arrow. Give a target's
output fields a name of your own, or write a target as a formula that looks
ahead. Read the compiled extension's version, the version of the state files
it writes, and the number of threads a bank runs on.

.. entries:: bank

Errors
------

This library's own errors, each also a subclass of the built-in or Polars
class a handler may already catch.

.. entries:: errors

Warnings
--------

``PolarsOnlineWarning``, and the warnings the library issues, each a subclass
of it.

.. entries:: warnings
