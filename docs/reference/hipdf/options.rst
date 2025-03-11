.. _api.options:

====================
Options and settings
====================

.. autosummary::
   :toctree: api/

   hipdf.get_option
   hipdf.set_option
   hipdf.describe_option
   hipdf.option_context


Available options
-----------------

You can get a list of available options and their descriptions with :func:`~hipdf.describe_option`. When called
with no argument :func:`~hipdf.describe_option` will print out the descriptions for all available options.

.. ipython:: python

   import hipdf
   hipdf.describe_option()
