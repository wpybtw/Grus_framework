#ifndef __KERNEL_2S_CUH
#define __KERNEL_2S_CUH

#include "common.cuh"
#include "frontier_2s.cuh"
#include "graph.cuh"
#include "worklist.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
namespace mgg {

// wl to flag
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel_2s(const graph_t __restrict__ G,
                               worklist::Worklist wl_c, char *flag, job_t job) {
  int tid = TID_1D;
  vtx_t src, level, laneid, dst, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < *wl_c.count) {
    src = wl_c.data[wpid];
    for (vtx_t edge_id = G.xadj[src] + laneid; edge_id < G.xadj[src + 1];
         edge_id += 32) {
      dst = G.adjncy[edge_id];
      generator(updater(src, dst, edge_id, job), flag, dst);
    }
  }
}

template <typename graph_t, typename subgraph_t, typename frontier_t,
          typename updater_t, typename generator_t, typename job_t>
class kernel_2s {
public:
  void operator()(graph_t G, frontier_t F, job_t job) {}
};

template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
class kernel_2s<graph_t, Subgraph<NORMAL>, frontier::Frontier_2S<BDF>,
                updater_t, generator_t, job_t> {
public:
  void operator()(graph_t G, Subgraph<NORMAL> SG1, Subgraph<NORMAL> SG2,
                  frontier::Frontier_2S<BDF> F, job_t job) {
    job.prepare();
    // process previous subgraph
    push_kernel_2s<graph_t, updater_t, generator_t, job_t><<<
        F.get_work_size_local_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
        SG1, *F.wl_local, F.flag2, job);
    // transfer
    push_kernel_2s<graph_t, updater_t, generator_t, job_t><<<
        F.get_work_size_remote_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
        SG2, *F.wl_remote, F.flag2, job);
  }
};
// template <typename graph_t, typename updater_t, typename generator_t,
//           typename job_t>
// class kernel_2s<graph_t, frontier::Frontier<WL>, updater_t, generator_t,
// job_t>
// {
// public:
//   void operator()(graph_t G, frontier::Frontier<WL> F, job_t job) {
//     job.prepare();
//     push_kernel_2s<
//         graph_t, updater_t, generator_t,
//         job_t><<<F.get_work_size_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
//         G, *F.wl_c, *F.wl_n, job);
//   }
// };

// template <typename graph_t, typename updater_t, typename generator_t,
//           typename job_t>
// class kernel_2s<graph_t, frontier::Frontier<BITMAP>, updater_t, generator_t,
//              job_t> {
// public:
//   void operator()(graph_t G, frontier::Frontier<BITMAP> F, job_t job) {
//     job.prepare();
//     push_kernel_2s<graph_t, updater_t, generator_t,
//                 job_t><<<F.numNode / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
//         G, F.flag1, F.flag2, F.finished_d, job);
//   }
// };
// template <typename graph_t, typename updater_t, typename generator_t,
//           typename job_t>
// class kernel_2s<graph_t, frontier::Frontier<BDF_AUTO>, updater_t,
// generator_t,
//              job_t> {
// public:
//   void operator()(graph_t G, frontier::Frontier<BDF_AUTO> F, job_t job) {
//     job.prepare();
//     if (F.current_wl)
//       push_kernel_2s<
//           graph_t, updater_t, generator_t,
//           job_t><<<F.get_work_size_h() / (BLOCK_SIZE >> 5) + 1,
//           BLOCK_SIZE>>>(
//           G, *F.wl_c, F.flag2, job);
//     else
//       push_kernel_2s<graph_t, updater_t, generator_t,
//                   job_t><<<F.numNode / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
//           G, F.flag1, F.flag2, job);
//   }
// };

} // namespace mgg

#endif