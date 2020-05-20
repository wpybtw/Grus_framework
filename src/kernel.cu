#include "subgraph.cuh"

__global__ void compute_lookup_buffer(Worklist wl, vtx_t *vtx, vtx_t *vtx_ptr,
                                      vtx_t *xadj, uint *lookup_buffer,
                                      vtx_t size) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) {
    lookup_buffer[vtx[tid]]=tid;
     outDegree[tid] = xadj[tid + 1] - xadj[tid];
  }
}