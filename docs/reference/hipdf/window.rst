.. _api.window:

======
Window
======

Rolling objects are returned by ``.rolling`` calls: :func:`hipdf.DataFrame.rolling`, :func:`hipdf.Series.rolling`, etc.

.. _api.functions_rolling:

Rolling window functions
------------------------
.. currentmodule:: hipdf.core.window.rolling

.. autosummary::
   :toctree: api/

   Rolling.count
   Rolling.sum
   Rolling.mean
   Rolling.var
   Rolling.std
   Rolling.min
   Rolling.max
   Rolling.apply
