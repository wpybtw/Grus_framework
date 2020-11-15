
#include "common.cuh"
#include "frontier.cuh"
#include "graph.cuh"
#include "graph_loader.cuh"
#include "kernel.cuh"
#include "worklist.cuh"
#include <gflags/gflags.h>
using namespace mgg;

DECLARE_int32(device);
DECLARE_string(input);
DECLARE_string(output);
DECLARE_int32(src);
DECLARE_bool(pull);
namespace cc {
__device__ char char_atomicCAS(char *addr, char cmp, char val) {
  unsigned *al_addr = reinterpret_cast<unsigned *>(((unsigned long long)addr) &
                                                   (0xFFFFFFFFFFFFFFFCULL));
  unsigned al_offset = ((unsigned)(((unsigned long long)addr) & 3)) * 8;
  unsigned mask = 0xFFU;
  mask <<= al_offset;
  mask = ~mask;
  unsigned sval = val;
  sval <<= al_offset;
  unsigned old = *al_addr, assumed, setval;
  do {
    assumed = old;
    setval = assumed & mask;
    setval |= sval;
    old = atomicCAS(al_addr, assumed, setval);
  } while (assumed != old);
  return (char)((assumed >> al_offset) & 0xFFU);
}
__global__ void CCInit(uint *label, int nnodes, vtx_t source) {
  int tid = TID_1D;
  if (tid < nnodes) {
    label[tid] = tid;
  }
}
// template<typename graph_t>
class job_t {

public:
  uint src;
  uint *label;
  uint itr = 0;
  vtx_t numNode;
  weight_t *adjwgt = nullptr;
  void operator()(vtx_t _numNode, uint _src) {
    numNode = _numNode;
    src = _src;
    init();
  }
  void init() {
    H_ERR(cudaMallocManaged(&label, numNode * sizeof(uint)));
    CCInit<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE>>>(label, numNode, src);
  }
  void prepare() {}
  void clean() {
  // __host__ __device__ ~job_t() {
#if !defined(__CUDA_ARCH__)
    if (!gflags::GetCommandLineFlagInfoOrDie("output").is_default)
      print::SaveResults(FLAGS_output, label, numNode);
#endif
  }
};

struct updater {
  __forceinline__ __device__ bool operator()(vtx_t src, vtx_t dst,
                                             vtx_t edge_id, job_t job) {
    if (job.label[dst] > job.label[src]) {
      atomicMin(&job.label[dst], job.label[src]);
      // if (!char_atomicCAS(&flag[dstId], 0, 1)) //need F.flag
      return true;
    }
    return false;
  }
};
struct generator {
  __forceinline__ __device__ void operator()(bool updated,
                                             worklist::Worklist wl, vtx_t dst) {
    if (updated)
      wl.append(dst);
  }
  __forceinline__ __device__ void operator()(bool updated, char *flag,
                                             vtx_t dst) {
    if (updated)
      if (!char_atomicCAS(&flag[dst], 0, 1))
        flag[dst] = true;
  }
  __forceinline__ __device__ void operator()(bool updated, char *flag,
                                             vtx_t dst, char *finished) {
    if (updated) {
      flag[dst] = 1;
      *finished = false;
    }
  }
};
struct pull_selector {
  __forceinline__ __device__ bool operator()(vtx_t id, job_t job) {
    if (job.label[id] == id) {
      return true;
    }
    return false;
  }
};
} // namespace cc
bool CC_multi_gpu() {
  graph_t<CSR> G_csr;
  graph_loader loader;
  loader.Load(G_csr, false);
  graph_t<CSC> G;
  G.CSR2CSC(G_csr);
  G.make_chunks(4);
  // for (size_t i = 0; i < 4; i++) {
  //   cout << "G " << i << G.chunks[i] << endl;
  // }
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  LOG("distributing\n");
  G.distribute_chunks(&stream);
}
bool CC_pull_single_gpu() {
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G_csr;
  graph_loader loader;
  loader.Load(G_csr, false);
  graph_t<CSC> G;
  G.CSR2CSC(G_csr);
  LOG("CC pull single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cc::job_t job;
  job(G.numNode, FLAGS_src);
  frontier::Frontier<BITMAP> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, true);
  G.Set_Mem_Policy(&stream); // stream
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  kernel_pull<cc::updater, cc::generator, cc::pull_selector, cc::job_t> K;
  while (!F.finish()) {
    // cout << "itr " << job.itr << " wl_sz " << F.wl_sz << endl;
    K(G, F, job);
    cudaDeviceSynchronize();
    // H_ERR(cudaStreamSynchronize(stream));
    F.Next();
    job.itr++;
  }
  cout << "itr " << job.itr << " in " << t.Finish() << endl;
  return 0;
}
bool CC_single_gpu() {
  if (FLAGS_pull) {
    return CC_pull_single_gpu();
  }
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G;
  graph_loader loader;
  loader.Load(G, false);
  LOG("CC single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // G.Init(false);
  cc::job_t job;
  job(G.numNode, FLAGS_src);
  frontier::Frontier<BDF> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, true);
  G.Set_Mem_Policy(&stream); // stream
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  kernel<graph_t<CSR>, frontier::Frontier<BDF>, cc::updater, cc::generator,
         cc::job_t>
      K;
  while (!F.finish()) {
    // cout << "itr " << job.itr << " wl_sz " << F.wl_sz << endl;
    K(G, F, job);
    cudaDeviceSynchronize();
    // H_ERR(cudaStreamSynchronize(stream));
    F.Next();
    job.itr++;
  }
  cout << "itr " << job.itr << " in " << t.Finish() << endl;
  job.clean();
  return 0;
}
