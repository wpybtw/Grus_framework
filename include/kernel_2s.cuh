#ifndef __KERNEL_2S_CUH
#define __KERNEL_2S_CUH

#include "common.cuh"
#include "frontier_2s.cuh"
#include "graph.cuh"
#include "subgraph.cuh"
#include "worklist.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
namespace mgg {

// wl to flag
// template <typename subgraph_t, typename updater_t, typename generator_t,
//           typename job_t>
// __global__ void push_kernel_subgraph_reverse(subgraph_t G,
//                                              worklist::Worklist wl_c,
//                                              char *flag, job_t job) {
//   int tid = TID_1D;
//   vtx_t src, level, laneid, dst, wpid;
//   wpid = tid / 32;
//   laneid = threadIdx.x % 32;
//   updater_t updater;
//   generator_t generator;
//   if (wpid < *wl_c.count) {
//     src = wl_c.data[wpid];
//     vtx_t local_id=G.get_vtx_from_id(src);
//     for (size_t offset = laneid; offset < G.get_vtx_degree[wpid];
//          offset += 32) {
//       dst = G.get_edge_dst(wpid, offset);
//       edge_id = G.get_edge_id(wpid, offset);
//       generator(updater(src, dst, edge_id, job), flag, dst);
//     }
//   }
// }
template <typename subgraph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel_subgraph(subgraph_t G, worklist::Worklist wl_c,
                                     char *flag, job_t job) {
  int tid = TID_1D;
  vtx_t src, level, laneid, dst, wpid, edge_id, local_id;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < *wl_c.count) {
    src = wl_c.data[wpid];
    local_id = G.get_vtx_from_id(src);
    // if (tid == 0)
    //   printf("tid 0 process %d with local_id %d \n", src,local_id);
    for (size_t offset = laneid; offset < G.get_vtx_degree(wpid);
         offset += 32) {
      edge_id = G.get_edge_id(local_id, offset);
      dst = G.get_edge_dst(local_id, offset);
      generator(updater(src, dst, edge_id, job), flag, dst);
    }
  }
}

template <typename graph_t, typename subgraph_t, typename frontier_t,
          typename updater_t, typename generator_t, typename job_t>
class kernel_2s {
public:
  void operator()(graph_t G, subgraph_t SG1, subgraph_t SG2, frontier_t F,
                  job_t job) {}
};

template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
class kernel_2s<graph_t, Subgraph<NORMAL>, frontier::Frontier_2S<BDF>,
                updater_t, generator_t, job_t> {
public:
  void operator()(graph_t G, Subgraph<NORMAL> SG1, Subgraph<NORMAL> SG2,
                  frontier::Frontier_2S<BDF> F, job_t job) {
    job.prepare();
    LOG("-----------------------------------------------\n");
    // LOG("F.get_work_size_local_h() sz %d\n ", F.get_work_size_local_h());
    // LOG("F.get_work_size_remote_h() sz %d\n ", F.get_work_size_remote_h());
    // process previous subgraph
    push_kernel_subgraph<Subgraph<NORMAL>, updater_t, generator_t, job_t><<<
        F.get_work_size_local_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
        SG1, *F.wl_local, F.flag2, job);
    SG1.clean(); // read-only copy will be better??
    // build and transfer
    SG2.build(G, *F.wl_remote);
    push_kernel_subgraph<Subgraph<NORMAL>, updater_t, generator_t, job_t><<<
        F.get_work_size_remote_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
        SG2, *F.wl_remote, F.flag2, job);
    // LOG("SG1 numNode%d\n", SG1.numNode);
    // LOG("SG2 numNode%d\n", SG2.numNode);
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