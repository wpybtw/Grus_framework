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

template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(graph_t G, worklist::Worklist wl_c,
                            worklist::Worklist wl_n, job_t job) {
  int tid = TID_1D;
  vtx_t id, level, laneid, tmpid, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < *wl_c.count) {
    id = wl_c.data[wpid];
    level = job.itr + 1;
    for (vtx_t i = G.xadj[id] + laneid; i < G.xadj[id + 1]; i += 32) {
      tmpid = G.adjncy[i];
      generator(updater(id, tmpid, job.level, level), wl_n, tmpid);
    }
  }
}
// wl to flag
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(graph_t G, worklist::Worklist wl_c, char *flag,
                            job_t job) {
  int tid = TID_1D;
  vtx_t id, level, laneid, tmpid, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < *wl_c.count) {
    id = wl_c.data[wpid];
    level = job.itr + 1;
    for (vtx_t i = G.xadj[id] + laneid; i < G.xadj[id + 1]; i += 32) {
      tmpid = G.adjncy[i];
      generator(updater(id, tmpid, job.level, level), flag, tmpid);
    }
  }
}
// flag to flag
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(graph_t G, char *flag1, char *flag2, job_t job) {
  int tid = TID_1D;
  vtx_t id, level, laneid, tmpid, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < job.num_Node) {
    if (flag1[wpid]) {
      id = wpid;
      level = job.itr + 1;
      for (vtx_t i = G.xadj[id] + laneid; i < G.xadj[id + 1]; i += 32) {
        tmpid = G.adjncy[i];
        generator(updater(id, tmpid, job.level, level), flag2, tmpid);
      }
    }
  }
}
template <typename graph_t, typename updater_t, typename generator_t,
          typename job_t>
__global__ void push_kernel(graph_t G, char *flag1, char *flag2, char *finished, job_t job) {
  int tid = TID_1D;
  vtx_t id, level, laneid, tmpid, wpid;
  wpid = tid / 32;
  laneid = threadIdx.x % 32;
  updater_t updater;
  generator_t generator;
  if (wpid < job.num_Node) {
    if (flag1[wpid]) {
      id = wpid;
      level = job.itr + 1;
      for (vtx_t i = G.xadj[id] + laneid; i < G.xadj[id + 1]; i += 32) {
        tmpid = G.adjncy[i];
        generator(updater(id, tmpid, job.level, level), flag2, tmpid, finished);
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
        job_t><<<F.get_work_size_h() * 32 / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
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
        job_t><<<F.get_work_size_h() * 32 / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
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
                job_t><<<F.numNode * 32 / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
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
          job_t><<<F.get_work_size_h() * 32 / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
          G, *F.wl_c, F.flag2, job);
    else
      push_kernel<graph_t, updater_t, generator_t,
                  job_t><<<F.numNode * 32 / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
          G, F.flag1, F.flag2, job);
  }
};

__global__ void pull_kernel() {}

} // namespace mgg

#endif