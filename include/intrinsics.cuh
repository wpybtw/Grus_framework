#ifndef __INTRINSICS_CUH
#define __INTRINSICS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <cooperative_groups.h>

using namespace cooperative_groups; 

namespace intrinsics
{
// HandleError
// inline void HandleError(cudaError_t err, const char *file, int line)
// {
//   if (err != cudaSuccess)
//   {
//     printf("%s in %s at line %d\n",
//            cudaGetErrorString(err), file, line);
//     // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//     fflush(stdout);
//     exit(EXIT_FAILURE);
//   }
// }
// #define H_ERR(err) (HandleError(err, __FILE__, __LINE__))
// #define TOTAL_THREADS_1D (gridDim.x * blockDim.x)
// #define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)

inline  void query_device_prop(int nDevices)
{
  // int nDevices;
  for (int i = 0; i < nDevices; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf(" Device name: %s\n", prop.name);
    printf(" Device Capability: %d.%d\n", prop.major, prop.minor);
    printf(" Device Overlap: %s\n", (prop.deviceOverlap ? "yes" : "no"));
    printf(" Device canMapHostMemory: %s\n", (prop.canMapHostMemory ? "yes" : "no"));
    printf(" Memory Detils\n");
    printf("  - registers per Block (KB): %d\n", (prop.regsPerBlock));
    printf("  - registers per Thread (1024): %d\n", (prop.regsPerBlock / 1024));
    printf("  - Share Memory per Block (KB): %.2f\n", (prop.sharedMemPerBlock + .0) / (1 << 10));
    printf("  - Total Global Memory (GB): %.2f\n", (prop.totalGlobalMem + .0) / (1 << 30));
    printf("  - Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  - Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  - Peak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf(" Thread Detils\n");
    printf("  - max threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  - processor Count: %d\n", prop.multiProcessorCount);
    printf("\n");
  }
}

// timming
#define RDTSC(val)                                     \
  do                                                   \
  {                                                    \
    uint64_t __a, __d;                                 \
    asm volatile("rdtsc"                               \
                 : "=a"(__a), "=d"(__d));              \
    (val) = ((uint64_t)__a) | (((uint64_t)__d) << 32); \
  } while (0)

static inline uint64_t rdtsc()
{
  uint64_t val;
  RDTSC(val);
  return val;
}

inline double wtime()
{
  double time[2];
  struct timeval time1;
  gettimeofday(&time1, NULL);

  time[0] = time1.tv_sec;
  time[1] = time1.tv_usec;

  return time[0] + time[1] * 1.0e-6;
}

__forceinline__ __global__ void dArrayInit(uint *ctr, uint n, uint k = 0)
{
  uint tid = blockDim.x * blockIdx.x + threadIdx.x;
  for (; tid < n; tid += blockDim.x * gridDim.x)
  {
    ctr[tid] = k;
  }
}

__forceinline__ __device__ uint atomicAggInc(uint *ctr)
{
  auto g = coalesced_threads();
  uint warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

} // namespace intrinsics
#endif