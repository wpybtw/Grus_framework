
#include "common.cuh"
#include "frontier.cuh"
#include "graph.cuh"
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
__forceinline__ __device__ bool update(vtx_t src, vtx_t dst, uint *label,
                                       uint level) {
  if (label[dst] > level) {
    label[dst] = level;
    return true;
  }
  return false;
}
// template <typename worklist>
__forceinline__ __device__ void generate(bool updated, worklist::Worklist wl,
                                         vtx_t dst) {
  if (updated)
    wl.append(dst);
}

class Job_data {
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

template <typename graph_t, typename frontier_t>
__global__ void push_kernel(graph_t G, worklist::Worklist wl_c,
                            worklist::Worklist wl_n, Job_data job) {
  int tid = TID_1D;
  vtx_t id, level, laneid, tmpid, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  if (wpid < *wl_c.count) {
    id = wl_c.data[wpid];
    level = job.itr + 1;
    for (vtx_t i = G.xadj[id] + laneid; i < G.xadj[id + 1]; i += 32) {
      tmpid = G.adjncy[i];
      //   generate(update(id, tmpid, job.level, level), wl_n, tmpid);
      if (job.level[tmpid] > level) {
        job.level[tmpid] = level;
        wl_n.append(tmpid);
      }
    }
  }
}
template <typename graph_t, typename frontier_t> class kernel {
public:
  void operator()(graph_t G, frontier_t F, Job_data job) {}
};
template <typename graph_t> class kernel<graph_t, mgg::frontier::Frontier> {
public:
  void operator()(graph_t G, mgg::frontier::Frontier F, Job_data job) {
    bfs::push_kernel<graph_t, mgg::frontier::Frontier><<<
        F.get_work_size_h() * 32 / BLOCK_SIZE + 1, BLOCK_SIZE>>>(G, *F.wl_c,
                                                                 *F.wl_n, job);
  }
};

} // namespace bfs

bool BFSSingle() {
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  mgg::graph::Graph G;
  printf("BFS single\n");
  cudaStream_t stream;
  G.Init(false);
  bfs::Job_data job;
  job(G.numNode, FLAGS_src);
  mgg::frontier::Frontier F;
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, false);
  G.Set_Mem_Policy(stream);
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  bfs::kernel<mgg::graph::Graph, mgg::frontier::Frontier> K;
  while (F.get_work_size_h() != 0) {
    K(G, F, job);
    cudaDeviceSynchronize();
    // H_ERR(cudaStreamSynchronize(stream));
    F.Next();
    job.itr++;
  }
  cout << "itr " << job.itr << " in " << t.Finish() << endl;
  return 0;
}
//   cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
//   H_ERR(cudaMallocManaged(&F, sizeof(mgg::frontier::Frontier)));
//   H_ERR(cudaMemAdvise(F, sizeof(mgg::frontier::Frontier),
//                       cudaMemAdviseSetAccessedBy, FLAGS_device));

// cout << "itr " << job.itr << "wl size" <<F.get_work_size_h() << endl;
// bfs::push_kernel<mgg::graph::Graph, mgg::frontier::Frontier><<<
//     F.get_work_size_h() * 32 / BLOCK_SIZE + 1, BLOCK_SIZE>>>(G, *F.wl_c,
//                                                              *F.wl_n,
//                                                              job);

// template <typename graph_t, typename frontier_t>
// __global__ void push_kernel(graph_t G, frontier_t *F, Job_data job) {
//   int tid = TID_1D;
//   vtx_t id, level, laneid, tmpid, wpid;
//   wpid = tid / 32;
//   laneid = threadIdx.x % 32;
//   if (wpid < F.get_work_size()) {
//     id = F.get_work_item(wpid);
//     level = job.itr + 1;
//     for (vtx_t i = G.xadj[id] + laneid; i < G.xadj[id + 1]; i += 32) {
//       tmpid = G.adjncy[i];
//       if (job.level[tmpid] > level) {
//         job.level[tmpid] = level;
//         F.add_work_item(tmpid);
//       }
//     }
//   }
// }