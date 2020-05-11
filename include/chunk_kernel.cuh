#ifndef __CHUNK_KERNEL_CUH
#define __CHUNK_KERNEL_CUH

#include "common.cuh"
#include "frontier.cuh"
#include "graph.cuh"
#include "worklist.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
namespace mgg {

// template <typename graph_t, typename frontier_t, typename updater_t,
//           typename generator_t, typename job_t>
// __global__ void push_kernel(graph_t G, worklist::Worklist wl_c,
//                             worklist::Worklist wl_n, job_t job);

// todo: flag1 are a local copy, how about flag2
// pull kernel on CSC
template <typename updater_t, typename generator_t, typename pull_selector_t,
          typename job_t>
__global__ void chunk_pull_kernel(graph_chunk<CSC> G, char *flag1, char *flag2,
                                  job_t job) {
  int tid = TID_1D;
  vtx_t dst, level, laneid, src, wpid;
  wpid = tid / 32 + G.start_v;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  pull_selector_t pull_selector;
  if (wpid < G.end_v) {
    if (pull_selector(wpid)) {
      dst = wpid; //blobal vtx id
      // use chunk offset
      for (vtx_t i = laneid; i < G.get_degree(dst); i += 32) {
        // src = G.adjncy[i];
        src = G.access_edge(dst, i);
        if (flag1[src])
          generator(updater(src, dst, edge_id, job), flag2, dst);
      }
    }
  }
}
// todo 
// --------------------------push kernels--------------------
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(graph_t G, worklist::Worklist wl_c,
                            worklist::Worklist wl_n, job_t job) {
  int tid = TID_1D;
  vtx_t src, laneid, dst, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < *wl_c.count) {
    src = wl_c.data[wpid];
    for (vtx_t i = G.xadj[src] + laneid; i < G.xadj[src + 1]; i += 32) {
      dst = G.adjncy[i];
      generator(updater(src, dst, edge_id, job), wl_n, dst);
    }
  }
}
// wl to flag
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(graph_t G, worklist::Worklist wl_c, char *flag,
                            job_t job) {
  int tid = TID_1D;
  vtx_t src, level, laneid, dst, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < *wl_c.count) {
    src = wl_c.data[wpid];
    for (vtx_t i = G.xadj[src] + laneid; i < G.xadj[src + 1]; i += 32) {
      dst = G.adjncy[i];
      generator(updater(src, dst, edge_id, job), flag, dst);
    }
  }
}
// flag to flag
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(graph_t G, char *flag1, char *flag2, job_t job) {
  int tid = TID_1D;
  vtx_t src, level, laneid, dst, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < job.num_Node) {
    if (flag1[wpid]) {
      src = wpid;
      for (vtx_t i = G.xadj[src] + laneid; i < G.xadj[src + 1]; i += 32) {
        dst = G.adjncy[i];
        generator(updater(src, dst, edge_id, job), flag2, dst);
      }
    }
  }
}
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(graph_t G, char *flag1, char *flag2, char *finished,
                            job_t job) {
  int tid = TID_1D;
  vtx_t src, level, laneid, dst, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < job.num_Node) {
    if (flag1[wpid]) {
      src = wpid;
      for (vtx_t i = G.xadj[src] + laneid; i < G.xadj[src + 1]; i += 32) {
        dst = G.adjncy[i];
        generator(updater(src, dst, edge_id, job), flag2, dst, finished);
      }
    }
  }
}
template <typename graph_t, typename frontier_t, typename updater_t,
          typename generator_t, typename job_t>
class kernel {
public:
  void operator()(graph_t G, frontier_t F, job_t job) {}
};

template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
class kernel<graph_t, frontier::Frontier<WL>, updater_t, generator_t, job_t> {
public:
  void operator()(graph_t G, frontier::Frontier<WL> F, job_t job) {
    push_kernel<
        graph_t, updater_t, generator_t,
        job_t><<<F.get_work_size_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
        G, *F.wl_c, *F.wl_n, job);
  }
};
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
class kernel<graph_t, frontier::Frontier<BDF>, updater_t, generator_t, job_t> {
public:
  void operator()(graph_t G, frontier::Frontier<BDF> F, job_t job) {
    push_kernel<
        graph_t, updater_t, generator_t,
        job_t><<<F.get_work_size_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
        G, *F.wl_c, F.flag2, job);
  }
};
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
class kernel<graph_t, frontier::Frontier<BITMAP>, updater_t, generator_t,
             job_t> {
public:
  void operator()(graph_t G, frontier::Frontier<BITMAP> F, job_t job) {
    push_kernel<graph_t, updater_t, generator_t,
                job_t><<<F.numNode / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
        G, F.flag1, F.flag2, F.finished_d, job);
  }
};
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
class kernel<graph_t, frontier::Frontier<BDF_AUTO>, updater_t, generator_t,
             job_t> { //<graph_t, frontier::Frontier<BDF_AUTO>, updater_t,
                      // generator_t, job_t
public:
  void operator()(graph_t G, frontier::Frontier<BDF_AUTO> F, job_t job) {
    if (F.current_wl)
      push_kernel<
          graph_t, updater_t, generator_t,
          job_t><<<F.get_work_size_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
          G, *F.wl_c, F.flag2, job);
    else
      push_kernel<graph_t, updater_t, generator_t,
                  job_t><<<F.numNode / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
          G, F.flag1, F.flag2, job);
  }
};

__global__ void pull_kernel() {}

} // namespace mgg

#endif