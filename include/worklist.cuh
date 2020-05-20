#ifndef __WORKLIST_CUH
#define __WORKLIST_CUH

#include "common.cuh"
#include "intrinsics.cuh"
// DECLARE_bool(wlum);
// DECLARE_int32(device);

using namespace intrinsics;
#define WARP_SIZE 32

namespace mgg {
namespace worklist {

class Worklist {
public:
  vtx_t *data;
  vtx_t *count;
  vtx_t c_count;
  vtx_t capacity;

  __host__ Worklist(vtx_t capacity_p) {
    capacity = capacity_p;
    c_count = 0;

    // if (true)
    {
      H_ERR(cudaMallocManaged(&data, capacity * sizeof(vtx_t)));
      // H_ERR(cudaMemAdvise(data, capacity * sizeof(vtx_t),
      // cudaMemAdviseSetAccessedBy, FLAGS_device));
      // H_ERR(cudaMemPrefetchAsync(data, capacity * sizeof(vtx_t),
      // FLAGS_device, 0));
    }
    // else
    //   H_ERR(cudaMalloc(&data, capacity * sizeof(vtx_t)));
    H_ERR(cudaMalloc(&count, sizeof(vtx_t)));
    H_ERR(cudaMemcpy(count, &c_count, sizeof(vtx_t), cudaMemcpyHostToDevice));
  }
  __host__ Worklist() {}
  __host__ void init(vtx_t capacity_p, cudaStream_t stream = NULL) {
    capacity = capacity_p;
    c_count = 0;

    H_ERR(cudaMallocManaged(&data, capacity * sizeof(vtx_t)));
    // H_ERR(cudaMemAdvise(data, capacity * sizeof(vtx_t),
    // cudaMemAdviseSetAccessedBy, FLAGS_device));
    // H_ERR(cudaMemPrefetchAsync(data, capacity * sizeof(vtx_t), FLAGS_device,
    // 0));
    H_ERR(cudaMalloc(&count, sizeof(vtx_t)));
    H_ERR(cudaMemcpyAsync(count, &c_count, sizeof(vtx_t),
                          cudaMemcpyHostToDevice, stream));
  }
  __host__ void free() {
    H_ERR(cudaFree(data));
    H_ERR(cudaFree(count));
  }
  __host__ void initFull(vtx_t n);
  __host__ vtx_t get_sz();
  __host__ vtx_t get_sz(cudaStream_t streams);
  __host__ void reset();
  __host__ void add_item(vtx_t item, cudaStream_t streams);
  __host__ void creset() { c_count = 0; }

  __device__ __forceinline__ void append(const vtx_t &item) {
    vtx_t allocation = atomicAdd((vtx_t *)count, 1);
    data[allocation] = item;
  }
  __device__ __forceinline__ void warp_append(const vtx_t &item) {
    vtx_t allocation = atomicAggInc((vtx_t *)count);
    data[allocation] = item;
  }
  __device__ __forceinline__ void append_all(vtx_t *cache, const vtx_t length) {
    vtx_t allocation = atomicAdd((vtx_t *)count, length);
    for (int i = 0; i < length; i++) {
      data[allocation + i] = cache[i];
    }
  }
  __device__ __forceinline__ void reset_d() const { *count = 0; }
  __device__ __forceinline__ vtx_t read(vtx_t i) const { return data[i]; }
  __device__ __forceinline__ vtx_t size() const { return *count; }
  __device__ __forceinline__ void assign(vtx_t i, vtx_t t) const {
    data[i] = t;
  }
};
__global__ void flag_to_wl(Worklist wl, char *flag1, vtx_t size);
__global__ void get_outdegree(Worklist wl, vtx_t *vtx_ptr, vtx_t *xadj,
                              uint *outDegree, vtx_t size);
} // namespace worklist
} // namespace mgg
#endif