#ifndef AMD_COOPERATIVE_GROUPS_EXT
#define AMD_COOPERATIVE_GROUPS_EXT

#include "../hip_warp_primitives/amd_warp_primitives.cuh"
#include <hip/device_functions.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#ifdef __AMDGCN_WAVEFRONT_SIZE
#undef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE __AMDGCN_WAVEFRONT_SIZE
#endif

#if !(__gfx1010__ || __gfx1011__ || __gfx1012__ || __gfx1030__ || __gfx1031__)
#if WAVEFRONT_SIZE != 64
#error "WAVEFRONT_SIZE 64 required"
#endif
#endif

namespace hip_extensions {

using namespace hip_warp_primitives;

/**
 * @brief Namespace containing extensions to HIP cooperative groups.
 **/
namespace hip_cooperative_groups_ext {

/**
 * @brief A base class that represents a cooperative group with some extensions
 * to the HIP-native implementation.
 *
 * This is a variant of cooperative groups that adds some
 * additional APIs that are presently not part of HIP cooperative groups
 * (e.g., any() and ballot()).
 *
 * All extended cooperative groups are convertible into this base class.
 *
 * CAUTION: currently, only cooperative groups of size <=64 are supported.
 **/
class cooperative_group_base {
 private:
  uint32_t __size;         ///< size of cooperative group
  lane_mask __group_mask;  ///< mask of the cooperative group

 public:
  /**
   * @brief Constructs a cooperative group base class instance.
   *
   * @param size Number of work items in the cooperative group.
   * @param mask Lane mask with Nth bit set to 1 if and only if the Nth work item
   * in the calling wavefront belongs to the cooperative group.
   */
  __device__ cooperative_group_base(uint32_t size, lane_mask mask)
  {
    __size       = size;
    __group_mask = mask;
  }

  /**
   * @brief Sets the size (number of work items) of the cooperative group.
   * @param size Size (number of work items) of the cooperative group.
   */
  __device__ void set_size(uint32_t size) { __size = size; }

  /**
   * @brief Gets the size (number of work items) of the cooperative group.
   * @return Size (number of work items) of the cooperative group.
   */
  __device__ uint32_t size() const { return __size; }

  /**
   * @brief Evaluate predicate for all work items in the cooperative group and returns non-zero if
   * and only if predicate evaluates to non-zero for any of them.
   * @param pred The predicate to evaluate.
   * @return True if the predicate evaluates to true in any work item in the cooperative group.
   */
  __device__ inline bool any(int pred) const
  {
    assert(__is_thread_in_mask(__group_mask));
    return __any_sync(__group_mask, pred);
  }

  /**
   * @brief Evaluate predicate for all work items in the cooperative group and returns an integer
   * whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth work item.
   * @param pred The predicate to evaluate.
   * @return An integer whose Nth bit is set if and only if predicate evaluates to non-zero for the
   * Nth work item.
   */
  __device__ inline lane_mask ballot(int pred) const
  {
    auto result_ballot_sync = __ballot_sync(__group_mask, pred);
    result_ballot_sync      = (__size == WAVEFRONT_SIZE)
                                ? result_ballot_sync
                                : result_ballot_sync >> __fns64(__group_mask, 0, 1);
    return result_ballot_sync;
  }

  /**
   * @brief Returns the thread rank of the calling work item in [0,size()-1]
   * @return The thread rank of the calling work item in [0,size()-1]
   */
  __device__ inline int thread_rank() const
  {
    auto lane_id = __lane_id();
    int rank =
      (__size == WAVEFRONT_SIZE) ? lane_id : __popcll(__group_mask & ((1L << (lane_id)) - 1));
    // printf("mask: %llx lane_id: %d rank %d size %d\n", __group_mask, lane_id, rank, __size);
    return rank;
  }

  /**
   * @brief Synchronizes the threads in the cooperative group.
   */
  __device__ inline void sync() const { return __sync_active_threads(); }

  /**
   * @brief Copies a variable from a source rank to all other ranks in the cooperative group.
   *
   * @tparam T the type of the variable that should be shuffled.
   *
   * @param var The variable to broadcast (from source rank)
   * @param srcRank The rank in the cooperative group from which the variable is broadcasted.
   *
   * @return The value of var from the work item with rank srcRank.
   */
  template <class T>
  __device__ inline T shfl(T var, int srcRank) const
  {
    int srcLane = (__size == WAVEFRONT_SIZE) ? srcRank : __fns64(__group_mask, 0, srcRank + 1);
    // printf("mask %llx rank: %ld lane: %ld\n", __group_mask, srcRank, srcLane);
    return __shfl_sync(__group_mask, var, srcLane);
  }

  /**
   * @brief Gets the lane mask of the cooperative group.
   * @return The lane mask of the cooperative group.
   */
  __device__ lane_mask get_mask() const { return __group_mask; }

  protected:
  /**
   * @brief Sets the lane mask of the cooperative group.
   * @param lm The lane mask of the cooperative group.
   */
  __device__ void set_mask(lane_mask lm) { __group_mask = lm; }
};

/**
 * @brief A tiled cooperative group.
 *
 * @tparam Size The size of the tile. Currently, only sizes <=64 are supported.
 */
template <uint32_t Size>
class tiled_partition_internal_ext : public cooperative_group_base {
 public:
  __device__ tiled_partition_internal_ext() : cooperative_group_base(Size, ~0)
  {  // Include all threads
    lane_mask __group_mask = __match_any_sync(get_mask(), threadIdx.x / size());
    set_mask(__group_mask);
  }
};

/**
 * @brief A coalesced cooperative group type.
 *
 * We follow the CUDA documentation and define a coalesced group as follows:
 * "If there exists a data-dependent conditional branch in the application code such
 *  that threads within a warp diverge, then the warp serially executes each branch disabling
 *  threads not on that path. The threads that remain active on the path are referred to as
 * coalesced." (Quote from:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#coalesced-groups)
 *
 * CAUTION: currently, only cooperative groups of size <=64 are supported.
 */
class coalesced_group_ext : public cooperative_group_base {
 public:  // todo(hip): hide constructor
  /**
   * @brief Creates a coalesced group with the given input lane mask.
   * @param lm Lane mask in which bit N is set if and only if the Nth thread belongs to the
   * coalesced group.
   */
  __device__ coalesced_group_ext(lane_mask lm) : cooperative_group_base(__popcll(lm), lm)
  {
    set_mask(lm);
  }
};

/**
 * @brief Partitions the parent thread block tile into coalesced subgroups.
 *
 * The binary partition is created depending on a predicate: work items will be
 * grouped into the same group if and only if they have the same predicate value.
 *
 * @param parent_tile The parent thread block tile to be partitioned.
 * @param pred The predicate to evaluate. Work items with the same predicate value will
 * be assigned to the same partition.
 */
template <uint32_t Size>
__device__ inline coalesced_group_ext binary_partition(
  tiled_partition_internal_ext<Size>& parent_tile, bool pred)
{
  lane_mask pred_mask = __ballot(pred);
  if (pred) {
    return coalesced_group_ext(pred_mask & parent_tile.get_mask());
  } else {
    return coalesced_group_ext(~(pred_mask & parent_tile.get_mask()));
  }
}

/**
 * @brief A cooperative group that represents a tiled thread block.
 *
 * @tparam Size The size of the thread block tile. CAUTION: Currently, only cooperative groups of
 * size <=64 are supported.
 */
template <uint32_t Size>
class thread_block_tile : public tiled_partition_internal_ext<Size> {
 public:
  __device__ thread_block_tile() : tiled_partition_internal_ext<Size>() {}
};

/**
 * @brief Creates a thread_block_tile from a parent cooperative group (HIP implementation).
 *
 * @tparam Size The size of the thread block tile. CAUTION: Currently, only cooperative groups of
 * size <=64 are supported.
 *
 * @return The thread_block_tile cooperative group, which the calling work item belongs to.
 */
template <uint32_t Size>
__device__ thread_block_tile<Size> tiled_partition(cooperative_groups::thread_block tb)
{
  return thread_block_tile<Size>();
}

/**
 * @brief Wrapper for return a HIP cooperative group for the active thread block.
 * @return The cooperative group that represents the active thread block.
 */
__device__ cooperative_groups::thread_block this_thread_block()
{
  // Todo(HIP): complete the implementation
  return cooperative_groups::this_thread_block();
}

}  // namespace hip_cooperative_groups_ext

}  // namespace hip_extensions
#endif
