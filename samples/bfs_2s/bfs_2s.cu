
#include "common.cuh"
#include <gflags/gflags.h>

#include "graph.cuh"
#include "graph_loader.cuh"
#include "worklist.cuh"

#include "frontier_2s.cuh"
#include "kernel_2s.cuh"
#include "subgraph.cuh"

using namespace mgg;

DECLARE_int32(device);
DECLARE_string(input);
DECLARE_string(output);
DECLARE_int32(src);
DECLARE_bool(pull);
namespace bfs_2s {

__global__ void BFS_2SInit(uint *label, int nnodes, vtx_t source) {
  int tid = TID_1D;
  if (tid < nnodes) {
    label[tid] = tid == source ? 0 : INFINIT;
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
    BFS_2SInit<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE>>>(label, numNode, src);
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
    if (job.label[dst] > job.itr + 1) {
      job.label[dst] = job.itr + 1;
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
      flag[dst] = true;
  }
  __forceinline__ __device__ void operator()(bool updated, char *flag,
                                             vtx_t dst, char *finished) {
    if (updated) {
      flag[dst] = true;
      *finished = false;
    }
  }
};
struct pull_selector {
  __forceinline__ __device__ bool operator()(vtx_t id, job_t job) {
    if (job.label[id] == INFINIT) {
      return true;
    }
    return false;
  }
};
} // namespace bfs_2s

bool BFS_2S_single_gpu() {

  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G;
  graph_loader loader;
  loader.Load(G, false);
  LOG("BFS_2S single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  bfs_2s::job_t job;
  job(G.numNode, FLAGS_src);
  frontier::Frontier_2S<BDF> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, false);
  // G.Set_Mem_Policy(&stream); // stream
  // cudaDeviceSynchronize(); //todo
  Subgraph<NORMAL> SG1, SG2;
  SG1.reserve(G.numNode);
  SG2.reserve(G.numNode);
  Timer t;
  t.Start();
  kernel_2s<graph_t<CSR>, frontier::Frontier_2S<BDF>, bfs_2s::updater,
         bfs_2s::generator, bfs_2s::job_t>
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
