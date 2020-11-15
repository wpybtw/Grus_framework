#ifndef __KERNEL_CUH
#define __KERNEL_CUH

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

// pull kernel on CSC
template <typename updater_t, typename generator_t, typename pull_selector_t,
          typename job_t>
__global__ void pull_kernel(graph_t<CSC> G, char *flag1, char *flag2,
                            char *finished, job_t job) {
  int tid = TID_1D;
  vtx_t dst, level, laneid, src, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  pull_selector_t pull_selector;
  if (wpid < job.numNode) {
    if (pull_selector(wpid, job)) {
      dst = wpid;
      for (vtx_t edge_id = G.xadj[dst] + laneid; edge_id < G.xadj[dst + 1];
           edge_id += 32) {
        src = G.adjncy[edge_id];
        if (flag1[src])
          generator(updater(src, dst, edge_id, job), flag2, dst, finished);
      }
    }
  }
}
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
    for (vtx_t edge_id = G.xadj[src] + laneid; edge_id < G.xadj[src + 1];
         edge_id += 32) {
      dst = G.adjncy[edge_id];
      generator(updater(src, dst, edge_id, job), wl_n, dst);
    }
  }
}
// wl to flag
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(const graph_t __restrict__ G,
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
// flag to flag
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(const graph_t __restrict__ G, char *flag1,
                            char *flag2, job_t job) {
  int tid = TID_1D;
  vtx_t src, level, laneid, dst, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < job.numNode) {
    if (flag1[wpid]) {
      src = wpid;
      for (vtx_t edge_id = G.xadj[src] + laneid; edge_id < G.xadj[src + 1];
           edge_id += 32) {
        dst = G.adjncy[edge_id];
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
  if (wpid < job.numNode) {
    if (flag1[wpid]) {
      src = wpid;
      for (vtx_t edge_id = G.xadj[src] + laneid; edge_id < G.xadj[src + 1];
           edge_id += 32) {
        dst = G.adjncy[edge_id];
        generator(updater(src, dst, edge_id, job), flag2, dst, finished);
      }
    }
  }
}
template <typename updater_t, typename generator_t, typename pull_selector_t,
          typename job_t>
class kernel_pull {
public:
  void operator()(graph_t<CSC> G, frontier::Frontier<BITMAP> F, job_t job) {
    pull_kernel<updater_t, generator_t, pull_selector_t,
                job_t><<<F.numNode / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
        G, F.flag1, F.flag2, F.finished_d, job);
  }
};

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
    job.prepare();
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
    job.prepare();
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
    job.prepare();
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
    job.prepare();
    switch (F.current_f) {
    case WL:
      push_kernel<
          graph_t, updater_t, generator_t,
          job_t><<<F.get_work_size_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
          G, *F.wl_c, *F.wl_n, job);
    case BITMAP:
      push_kernel<graph_t, updater_t, generator_t,
                  job_t><<<F.numNode / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
          G, F.flag1, F.flag2, job);
    case BDF:
      push_kernel<
          graph_t, updater_t, generator_t,
          job_t><<<F.get_work_size_h() / (BLOCK_SIZE >> 5) + 1, BLOCK_SIZE>>>(
          G, *F.wl_c, F.flag2, job);
    }
  }
};

} // namespace mgg

#endif