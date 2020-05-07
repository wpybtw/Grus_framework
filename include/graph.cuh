#ifndef _GRAPH_CUH
#define _GRAPH_CUH

#include "common.cuh"
#include "intrinsics.cuh"
#include "timer.cuh"
// #include "job.cuh"
#include "print.cuh"

#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <assert.h>
#include <gflags/gflags.h>

// using namespace intrinsics;
// using namespace grus;
// using namespace frontier;

DECLARE_string(input);
DECLARE_bool(pf);
DECLARE_bool(ab);
DECLARE_bool(rm);
DECLARE_bool(pl);
DECLARE_bool(opt);
DECLARE_int32(device);
namespace mgg {
namespace graph {

// template <typename T> void PrintResults(T *results, uint n);


template <graphFmt fmt> struct graph_t {};

template <> class graph_t<CSR> {
public:
  bool hasZeroID;
  uint64_t numNode;
  uint64_t numEdge;
  // graph
  vtx_t *xadj, *vwgt, *adjncy;
  vtx_t *xadj_d, *vwgt_d, *adjncy_d;
  weight_t *adjwgt, *adjwgt_d;
  uint *inDegree;
  uint *outDegree;
  bool weighted;
  bool needWeight;

  uint64_t mem_used = 0;

  graph_t(bool _needweight = false) {
    this->hasZeroID = false;
    this->needWeight = _needweight;
    // H_ERR(cudaMallocManaged(&this->xadj, (num_Node + 1) * sizeof(vtx_t)));
    // H_ERR(cudaMallocManaged(&this->adjncy, num_Edge * sizeof(vtx_t)));
    // if (_needweight)
    //   H_ERR(cudaMallocManaged(&G.adjwgt, num_Edge * sizeof(weight_t)));
  }
  ~graph_t() {
    // if (xadj != nullptr)
    //   H_ERR(cudaFree(xadj));
    // if (adjncy != nullptr)
    //   H_ERR(cudaFree(adjncy));
    // if (adjwgt != nullptr)
    //   H_ERR(cudaFree(adjwgt));
  }
  void Set_Mem_Policy(cudaStream_t stream) { //&
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    if (FLAGS_opt) {
      LOG("using opt\n");
      H_ERR(cudaMemPrefetchAsync(xadj_d, (numNode + 1) * sizeof(vtx_t),
                                 FLAGS_device, stream));
      if (mem_used < avail) {
        H_ERR(cudaMemPrefetchAsync(adjncy_d, numEdge * sizeof(vtx_t),
                                   FLAGS_device, stream));
        if (needWeight)
          H_ERR(cudaMemPrefetchAsync(adjwgt_d, numEdge * sizeof(weight_t),
                                     FLAGS_device, stream));
      } else {
        if (needWeight) {
          H_ERR(cudaMemPrefetchAsync(
              adjncy_d, (avail - (numNode + 1) * sizeof(vtx_t)) / 2,
              FLAGS_device, stream));
          H_ERR(cudaMemPrefetchAsync(
              adjwgt_d, (avail - (numNode + 1) * sizeof(vtx_t)) / 2,
              FLAGS_device, stream));
        } else
          H_ERR(cudaMemPrefetchAsync(adjncy_d,
                                     avail - (numNode + 1) * sizeof(vtx_t),
                                     FLAGS_device, stream));
      }
      if (mem_used > avail) //
      {
        H_ERR(cudaMemAdvise(xadj_d, (numNode + 1) * sizeof(vtx_t),
                            cudaMemAdviseSetAccessedBy, FLAGS_device));
        H_ERR(cudaMemAdvise(adjncy_d, numEdge * sizeof(vtx_t),
                            cudaMemAdviseSetAccessedBy, FLAGS_device));
        if (needWeight)
          H_ERR(cudaMemAdvise(adjwgt_d, numEdge * sizeof(weight_t),
                              cudaMemAdviseSetAccessedBy, FLAGS_device));
      }
    } else {
      if (FLAGS_pf) {
        LOG("pfing\n");
        H_ERR(cudaMemPrefetchAsync(xadj_d, (numNode + 1) * sizeof(vtx_t),
                                   FLAGS_device, stream));
        H_ERR(cudaMemPrefetchAsync(adjncy_d, numEdge * sizeof(vtx_t),
                                   FLAGS_device, stream));
        if (needWeight)
          H_ERR(cudaMemPrefetchAsync(adjwgt_d, numEdge * sizeof(weight_t),
                                     FLAGS_device, stream));
      }
      if (FLAGS_ab) //
      {
        LOG("AB hint\n");
        H_ERR(cudaMemAdvise(xadj_d, (numNode + 1) * sizeof(vtx_t),
                            cudaMemAdviseSetAccessedBy, FLAGS_device));
        H_ERR(cudaMemAdvise(adjncy_d, numEdge * sizeof(vtx_t),
                            cudaMemAdviseSetAccessedBy, FLAGS_device));
        if (needWeight)
          H_ERR(cudaMemAdvise(adjwgt_d, numEdge * sizeof(weight_t),
                              cudaMemAdviseSetAccessedBy, FLAGS_device));
      }
      if (FLAGS_rm) //
      {
        LOG("RM hint\n");
        H_ERR(cudaMemAdvise(xadj_d, (numNode + 1) * sizeof(vtx_t),
                            cudaMemAdviseSetReadMostly, FLAGS_device));
        H_ERR(cudaMemAdvise(adjncy_d, numEdge * sizeof(vtx_t),
                            cudaMemAdviseSetReadMostly, FLAGS_device));
        if (needWeight)
          H_ERR(cudaMemAdvise(adjwgt_d, numEdge * sizeof(weight_t),
                              cudaMemAdviseSetReadMostly, FLAGS_device));
      }
      if (FLAGS_pl) //
      {
        LOG("PL hint\n");
        H_ERR(cudaMemAdvise(xadj_d, (numNode + 1) * sizeof(vtx_t),
                            cudaMemAdviseSetPreferredLocation, FLAGS_device));
        H_ERR(cudaMemAdvise(adjncy_d, numEdge * sizeof(vtx_t),
                            cudaMemAdviseSetPreferredLocation, FLAGS_device));
        if (needWeight)
          H_ERR(cudaMemAdvise(adjwgt_d, numEdge * sizeof(weight_t),
                              cudaMemAdviseSetPreferredLocation, FLAGS_device));
      }
    }
  }
};
} // namespace graph
} // namespace mgg
#endif
