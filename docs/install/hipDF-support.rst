*******************************************************
hipDF supported environments, features, and interfaces
*******************************************************
.. note::

    The focus of this EA release of hipDF is on functionality. Performance is defocused in favor of functionality.

hipDF requires ROCm 6.4.0 or later running on Ubuntu 22.04 or later. See
`ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_
for installation instructions.

hipDF is supported on gfx942 and gfx90a only.

There is no support for:

* zstd compression and decompression
* GPU direct storage (KvikIO, cuFile)
* rocTX tracing

Support is limited to C++ and Python interfaces. There is no official Java interface support.

hipDF only supports features from cuDF 23.10.

There is no official support for:

* Per-thread default stream (PTDS)
* Integration with Dask hipDF or Dask HIP
* Interoperability with Polars
* The ``hipstreamz`` and ``hipdf_kafka`` Python packages
