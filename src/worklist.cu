#include "worklist.cuh"

namespace mgg
{

namespace worklist
{

__global__ void worklist_reset(Worklist wl) { wl.reset(); }

__global__ void worklist_add(Worklist wl, vtx_t item)
{
  if (blockDim.x * blockIdx.x + threadIdx.x == 0)
  {
    wl.append(item);
  }
}

__global__ void worklist_init_full(Worklist wl, vtx_t n)
{
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n)
  {
    wl.data[id] = id;
  }
  if (id == 0)
  {
    *wl.count = n;
  }
}

__global__ void worklist_min_max(Worklist wl, vtx_t n, vtx_t *min, vtx_t *max)
{
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n)
  {
    vtx_t tmp = wl.data[id];
    atomicMin(min, tmp);
    atomicMax(max, tmp);
  }
}

void wlInitFull(Worklist wl, vtx_t n)
{
  worklist_init_full<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(wl, n);
}
void wlInitFull(Worklist wl, vtx_t n, cudaStream_t stream)
{
  worklist_init_full<<<n / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(wl, n);
}
vtx_t wl_get_sz(Worklist *wl)
{
  H_ERR(cudaMemcpy(&wl->c_count, wl->count, sizeof(vtx_t),
                   cudaMemcpyDeviceToHost));
  return wl->c_count;
}
vtx_t wl_get_sz(Worklist *wl, cudaStream_t streams)
{
  H_ERR(cudaMemcpyAsync(&wl->c_count, wl->count, sizeof(vtx_t),
                        cudaMemcpyDeviceToHost, streams));
  return wl->c_count;
}
void wlGetMinMax(Worklist wl, vtx_t n, vtx_t *min, vtx_t *max)
{
  // worklist_init_full<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(wl, n);
  vtx_t size = wl_get_sz(&wl);
  worklist_min_max<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(wl, n, min, max);
}

void wl_sync(Worklist wl)
{
  H_ERR(
      cudaMemcpy(&wl.c_count, wl.count, sizeof(vtx_t), cudaMemcpyDeviceToHost));
}

void wl_reset(Worklist wl)
{
  worklist_reset<<<1, 1>>>(wl);
  wl.creset();
}
void wl_reset(Worklist wl, cudaStream_t streams)
{
  worklist_reset<<<1, 1, 0, streams>>>(wl);
  wl.creset();
}

void wl_add_item(Worklist wl, vtx_t item, cudaStream_t streams)
{
  worklist_add<<<1, 1, 0, streams>>>(wl, item);
}

} // namespace worklist

} // namespace mgg