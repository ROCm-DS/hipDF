#include <hipcub/hipcub.hpp>
#include "hipcub_ext.cuh"

struct Policy900 : hipcub_extensions::ChainedPolicy<900, Policy900, Policy900> {
    enum {
      BLOCK_THREADS    = 128,
      ITEMS_PER_THREAD = 32,
    };

  };