
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
namespace pagerank {
__global__ void pr_init(float *rank, float *delta, vtx_t *xadj, vtx_t *adjncy,
                        vtx_t numNode) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  vtx_t lid = threadIdx.x % 32;
  vtx_t wpid = tid / 32;
  if (wpid < numNode) {
    vtx_t id = wpid;
    vtx_t start = xadj[id];
    if (lid == 0)
      rank[id] = 1.0 - ALPHA;
    vtx_t degree = (xadj[id + 1] - xadj[id]);
    float update = ((1.0 - ALPHA) * ALPHA) / degree;
    for (size_t i = start + lid; i < xadj[id + 1]; i += 32) {
      atomicAdd(&delta[adjncy[i]], update);
    }
  }
}
__global__ void pr_dd0(float *rank, float *delta, float *delta2, vtx_t wl_sz) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < wl_sz) {
    float res;
    res = delta[tid];
    delta[tid] = 0.0;
    delta2[tid] = res;
    rank[tid] += res;
  }
}
// template<typename graph_t>
class job_t {
public:
  float *rank, *delta, *delta2;
  uint itr = 0;
  vtx_t numNode;
  weight_t *adjwgt = nullptr;
  // uint *out_degree;
  vtx_t *xadj;
  void operator()(vtx_t _numNode, uint *_xadj) {
    numNode = _numNode;
    xadj = _xadj;
    init();
  }
  void init() {
    H_ERR(cudaMallocManaged(&rank, numNode * sizeof(float)));
    H_ERR(cudaMallocManaged(&delta, numNode * sizeof(float)));
    H_ERR(cudaMallocManaged(&delta2, numNode * sizeof(float)));
  }
  void prepare() { // cudaStream_t *stream = NULL
    pr_dd0<<<numNode / 1024 + 1, 1024>>>(rank, delta, delta2, numNode);
  }
  void clean() {
// __host__ __device__ ~job_t() {
#if !defined(__CUDA_ARCH__)
    if (!gflags::GetCommandLineFlagInfoOrDie("output").is_default)
      print::SaveResults(FLAGS_output, rank, numNode);
#endif
  }
  __forceinline__ __device__ uint get_out_degree(vtx_t id) {
    return xadj[id + 1] - xadj[id];
  }
};

struct updater {
  __forceinline__ __device__ bool operator()(vtx_t src, vtx_t dst,
                                             vtx_t edge_id, job_t job) {
    float dt, update, res;
    res = job.delta2[src];
    update = res * ALPHA / job.get_out_degree(src);
    dt = atomicAdd(&job.delta[dst], update);
    if ((dt + update > EPSILON) && (dt < EPSILON)) {
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
    // if (job.label[id] == INFINIT) {
    return true;
    // }
    // return false;
  }
};
} // namespace pagerank
bool PR_multi_gpu() {
  //   graph_t<CSR> G_csr;
  //   graph_loader loader;
  //   loader.Load(G_csr, false);
  //   graph_t<CSC> G;
  //   G.CSR2CSC(G_csr);
  //   G.make_chunks(4);
  //   // for (size_t i = 0; i < 4; i++) {
  //   //   cout << "G " << i << G.chunks[i] << endl;
  //   // }
  //   cudaStream_t stream;
  //   cudaStreamCreate(&stream);
  //   LOG("distributing\n");
  //   G.distribute_chunks(&stream);
}
bool PR_pull_single_gpu() {
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G_csr;
  graph_loader loader;
  loader.Load(G_csr, false);
  graph_t<CSC> G;
  G.CSR2CSC(G_csr);
  LOG("PR pull single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  pagerank::job_t job;
  job(G.numNode, G.xadj);
  frontier::Frontier<BITMAP> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, true);
  G.Set_Mem_Policy(&stream); // stream
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  pagerank::pr_init<<<G.numNode * 32 / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(
      job.rank, job.delta, G.xadj, G.adjncy, G.numNode);
  kernel_pull<pagerank::updater, pagerank::generator, pagerank::pull_selector,
              pagerank::job_t>
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
bool PR_single_gpu() {
  if (FLAGS_pull) {
    return PR_pull_single_gpu();
  }
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G;
  graph_loader loader;
  loader.Load(G, false);
  LOG("PR single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  pagerank::job_t job;
  job(G.numNode, G.xadj);
  frontier::Frontier<BDF> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, true);
  G.Set_Mem_Policy(&stream); // stream
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  pagerank::pr_init<<<G.numNode * 32 / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(
      job.rank, job.delta, G.xadj, G.adjncy, G.numNode);
  kernel<graph_t<CSR>, frontier::Frontier<BDF>, pagerank::updater,
         pagerank::generator, pagerank::job_t>
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
