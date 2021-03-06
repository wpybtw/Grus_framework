#include "worklist.cuh"

namespace mgg {



namespace worklist {

__global__ void get_flag_num(char *flag1, vtx_t size, vtx_t *num)
{
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < size)
  {
    if (flag1[id])
      atomicAdd(num, 1);
  }
}
__global__ void compute_lookup_buffer(vtx_t *vtx,
                                      vtx_t *vtx_ptr, vtx_t *xadj,
                                      uint *lookup_buffer, uint *outDegree,
                                      vtx_t size) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) {
    // vtx_t id=wl.data[tid];
    lookup_buffer[vtx[tid]] = tid;
    outDegree[tid] = xadj[vtx[tid] + 1] - xadj[vtx[tid]];
  }
}

__global__ void worklist_reset(Worklist wl) { wl.reset_d(); }

__global__ void worklist_add(Worklist wl, vtx_t item) {
  if (blockDim.x * blockIdx.x + threadIdx.x == 0) {
    wl.append(item);
  }
}

__global__ void worklist_init_full(Worklist wl, vtx_t n) {
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n) {
    wl.data[id] = id;
  }
  if (id == 0) {
    *wl.count = n;
  }
}

__global__ void worklist_min_max(Worklist wl, vtx_t n, vtx_t *min, vtx_t *max) {
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n) {
    vtx_t tmp = wl.data[id];
    atomicMin(min, tmp);
    atomicMax(max, tmp);
  }
}

void Worklist::initFull(vtx_t n) {
  worklist_init_full<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(*this, n);
}
// void Worklist::initFull(Worklist wl, vtx_t n, cudaStream_t stream)
// {
//   worklist_init_full<<<n / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(wl, n);
// }
vtx_t Worklist::get_sz() {
  H_ERR(cudaMemcpy(&this->c_count, this->count, sizeof(vtx_t),
                   cudaMemcpyDeviceToHost));
  return this->c_count;
}
vtx_t Worklist::get_sz(cudaStream_t streams) {
  H_ERR(cudaMemcpyAsync(&this->c_count, this->count, sizeof(vtx_t),
                        cudaMemcpyDeviceToHost, streams));
  return this->c_count;
}
// void wlGetMinMax(Worklist wl, vtx_t n, vtx_t *min, vtx_t *max)
// {
//   // worklist_init_full<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(wl, n);
//   vtx_t size = wl_get_sz(&wl);
//   worklist_min_max<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(wl, n, min, max);
// }

// void wl_sync(Worklist wl)
// {
//   H_ERR(
//       cudaMemcpy(&wl.c_count, wl.count, sizeof(vtx_t),
//       cudaMemcpyDeviceToHost));
// }
__global__ void flag_to_wl(Worklist wl, char *flag1, vtx_t size) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) {
    if (flag1[tid])
      wl.warp_append(tid);
  }
}
__global__ void wl_to_flag(Worklist wl, char *flag1, vtx_t size) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) {
    vtx_t src = wl.data[tid];
    flag1[src] = 1;
  }
}
// after that, update flag_local
__global__ void flag_to_wl_remote_local(Worklist wl_remote, Worklist wl_local,
                                        char *flag_local, char *flag_active,
                                        vtx_t size) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) {
    if (flag_active[tid]) {
      if (flag_local[tid])
        wl_local.warp_append(tid);
      else
        wl_remote.warp_append(tid);
    }
  }
}

// __global__ void compute_offset(Worklist wl, vtx_t *vtx_ptr, vtx_t *xadj,
//                                vtx_t size) {
//   size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
//   if (tid < size) {
//   }
// }
void Worklist::reset() {
  worklist_reset<<<1, 1>>>(*this);
  this->creset();
}
// void wl_reset(Worklist wl, cudaStream_t streams)
// {
//   worklist_reset<<<1, 1, 0, streams>>>(wl);
//   wl.creset();
// }

void Worklist::add_item(vtx_t item, cudaStream_t streams) {
  worklist_add<<<1, 1, 0, streams>>>(*this, item);
}

} // namespace worklist

} // namespace mgg