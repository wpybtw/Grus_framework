#ifndef _COMPRESS_CUH
#define _COMPRESS_CUH

#include "common.cuh"
#include "graph.cuh"
#include "intrinsics.cuh"
#include "print.cuh"
#include "timer.cuh"

#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvrtc.h>

#include <algorithm>
#include <assert.h>
#include <gflags/gflags.h>

// frontier + xadj = active list
// active list--> compute offset,
// as active list are not known ahead, edges of a vtx must be consistent
// todo How to avoid reduplicative transfer
//      use a bitmap to indicate if in previous local sbgraph, then compute
//      resident vertices first.
// gap coding
// Variable-Length Encoding
// Interval Encoding

// struct compressedChunk {
//   /* data */
// };

namespace compress {
// log2f

// alignment
__forceinline__ __device__ __host__ void unary(vtx_t src, void *current_ptr) {

}

__forceinline__ __device__ __host__ void unary(vtx_t src) {}

} // namespace compress

#endif