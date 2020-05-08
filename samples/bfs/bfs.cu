
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
DECLARE_int32(src);

namespace bfs {
__global__ void BFSInit(uint *levels, int nnodes, vtx_t source) {
  int tid = TID_1D;
  if (tid < nnodes) {
    levels[tid] = tid == source ? 0 : INFINIT;
  }
}
struct updater {
  __forceinline__ __device__ bool operator()(vtx_t src, vtx_t dst, uint *label,
                                             uint level) {
    if (label[dst] > level) {
      label[dst] = level;
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
class job_t {
public:
  uint src;
  uint *level;
  uint itr = 0;
  vtx_t num_Node;
  void operator()(vtx_t _num_Node, uint _src) {
    num_Node = _num_Node;
    src = _src;
    init();
  }
  void init() {
    H_ERR(cudaMalloc(&level, num_Node * sizeof(uint)));
    BFSInit<<<num_Node / BLOCK_SIZE + 1, BLOCK_SIZE>>>(level, num_Node, src);
  }
};

} // namespace bfs

bool BFSSingle() {
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G;
  graph_loader loader;
  loader.Load(G, false);
  // graph_t<CSC> G2;
  // G2.CSR2CSC(G);
  LOG("BFS single\n");
  cudaStream_t stream;
  // G.Init(false);
  bfs::job_t job;
  job(G.numNode, FLAGS_src);
  frontier::Frontier<BDF_AUTO> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, false);
  G.Set_Mem_Policy(stream);
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  kernel<graph_t<CSR>, frontier::Frontier<BDF_AUTO>, bfs::updater, bfs::generator,
         bfs::job_t>
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
  return 0;
}
