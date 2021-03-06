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
#include <iostream>
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

template <typename T> void PrintResults(T *results, uint n);

template <graphFmt fmt> class graph_chunk {
public:
  uint64_t numNode;
  uint64_t numEdge;
  vtx_t *xadj, *vwgt, *adjncy, *adjncy_global;
  // vtx_t *xadj_d, *vwgt_d, *adjncy_d;
  weight_t *adjwgt, *adjwgt_global;
  uint64_t node_in_chunk;
  uint64_t edge_in_chunk;
  vtx_t start_v, end_v, start_e, end_e;
  uint device_id;
  bool weighted = false;

  graph_chunk(uint64_t _numNode, uint64_t _numEdge, uint64_t _node_in_chunk,
              uint64_t _edge_in_chunk, vtx_t _start_v, vtx_t _start_e,
              uint _device_id, vtx_t *_adjncy_global,
              weight_t *adjwgt_global = nullptr) {
    numNode = _numNode;
    numEdge = _numEdge;
    node_in_chunk = _node_in_chunk;
    edge_in_chunk = _edge_in_chunk;
    start_v = _start_v;
    end_v = start_v + _node_in_chunk;
    // start_e = xadj[start_v];
    start_e = _start_e;
    end_e = start_e + _edge_in_chunk;
    adjncy_global = _adjncy_global;
    device_id = _device_id;
    if (adjwgt_global != nullptr)
      weighted = true;
    H_ERR(cudaMallocManaged(&xadj, (numNode + 1) *
                                       sizeof(vtx_t))); // change to all xadj
    H_ERR(cudaMallocManaged(&adjncy, edge_in_chunk * sizeof(vtx_t)));
    if (weighted)
      H_ERR(cudaMallocManaged(&adjwgt, edge_in_chunk * sizeof(weight_t)));
  }
  __forceinline__ __device__ vtx_t access_edge(vtx_t src, vtx_t offset) {
    if ((start_v <= src) && (src <= end_v)) // local
      return adjncy[xadj[src] + offset - start_e];
    else
      return adjncy_global[xadj[src] + offset];
  }
  __forceinline__ __device__ vtx_t access_weight(vtx_t src, vtx_t offset) {
    if ((start_v <= src) && (src <= end_v))
      return adjwgt[xadj[src] + offset - start_e];
    else
      return adjwgt_global[xadj[src] + offset];
  }
  __forceinline__ __device__ vtx_t get_degree(vtx_t id) {
    return xadj[id + 1] - xadj[id];
  }
  void distribute(int deviceId, cudaStream_t *stream) { // = NULL
    // cudaSetDevice(deviceId);
    // LOG("distributing %d\n", deviceId);
    // print::PrintResults(xadj, 10);
    // print::PrintResults(adjncy, 10);
    // print::PrintResults(adjwgt, 10);
    H_ERR(cudaMemPrefetchAsync(xadj, (numNode + 1) * sizeof(vtx_t), deviceId,
                               nullptr));
    H_ERR(cudaMemPrefetchAsync(adjncy, edge_in_chunk * sizeof(vtx_t), deviceId,
                               *stream));
    if (weighted)
      H_ERR(cudaMemPrefetchAsync(adjwgt, edge_in_chunk * sizeof(weight_t),
                                 deviceId, *stream));
  }
};
template <graphFmt fmt>
std::ostream &operator<<(std::ostream &output, graph_chunk<fmt> &chunk) {
  output << "numNode: \t" << chunk.node_in_chunk << " numEdge: \t"
         << chunk.edge_in_chunk << endl;
  // output << "xadj: \t" << endl;
  // print::PrintResults(chunk.xadj, 10);
  // output << "adjncy: \t" << endl;
  // print::PrintResults(chunk.adjncy, 10);
  // output << "adjwgt: \t" << endl;
  // if (chunk.weighted)
  //   print::PrintResults(chunk.adjwgt, 10);
  return output;
}
template <graphFmt fmt> class graph_base {
public:
  bool hasZeroID;
  uint64_t numNode;
  uint64_t numEdge;
  // graph
  vtx_t *xadj, *vwgt, *adjncy;
  // vtx_t *xadj_d, *vwgt_d, *adjncy_d;
  weight_t *adjwgt = nullptr, *adjwgt_d;
  uint *inDegree;
  uint *outDegree;
  bool weighted;
  bool needWeight;
  uint64_t mem_used = 0;
  vector<graph_chunk<fmt>> chunks;
  int numChunk;

  void distribute_chunks(cudaStream_t *stream) { // = NULL
    for (size_t i = 0; i < numChunk; i++) {
      chunks[i].distribute(i, stream);
    }
  }
  void make_chunks(int num_gpu) {
    numChunk = num_gpu;
    vtx_t num_vtx_per_chunk = numNode / num_gpu;
    for (size_t i = 0; i < num_gpu - 1; i++) {
      chunks.push_back(graph_chunk<fmt>(
          numNode, numEdge, num_vtx_per_chunk + 1,
          xadj[(i + 1) * num_vtx_per_chunk] - xadj[i * num_vtx_per_chunk],
          i * num_vtx_per_chunk, xadj[i * num_vtx_per_chunk], i, adjncy,
          adjwgt));
      // memcpy(chunks[i].xadj, &xadj[i * num_vtx_per_chunk],
      //        num_vtx_per_chunk + 1);
      memcpy(chunks[i].xadj, xadj, numNode + 1);
      memcpy(chunks[i].adjncy, &adjncy[xadj[i * num_vtx_per_chunk]],
             chunks[i].edge_in_chunk);
      if (needWeight)
        memcpy(chunks[i].adjwgt, &adjwgt[xadj[i * num_vtx_per_chunk]],
               chunks[i].edge_in_chunk);
    }
    chunks.push_back(graph_chunk<fmt>(
        numNode, numEdge, num_vtx_per_chunk + numNode % num_gpu + 1,
        xadj[numNode] - xadj[(num_gpu - 1) * num_vtx_per_chunk],
        (num_gpu - 1) * num_vtx_per_chunk,
        xadj[(num_gpu - 1) * num_vtx_per_chunk], num_gpu - 1, adjncy, adjwgt));
    // memcpy(chunks[num_gpu - 1].xadj, &xadj[(num_gpu - 1) *
    // num_vtx_per_chunk],
    //        chunks[num_gpu - 1].node_in_chunk+1);
    memcpy(chunks[num_gpu - 1].xadj, xadj, numNode + 1);
    memcpy(chunks[num_gpu - 1].adjncy,
           &adjncy[xadj[(num_gpu - 1) * num_vtx_per_chunk]],
           chunks[num_gpu - 1].edge_in_chunk);
    if (needWeight)
      memcpy(chunks[num_gpu - 1].adjwgt,
             &adjwgt[xadj[(num_gpu - 1) * num_vtx_per_chunk]],
             chunks[num_gpu - 1].edge_in_chunk);
  }
  void Set_Mem_Policy(cudaStream_t *stream = NULL) { //& =NULL
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    if (FLAGS_opt) {
      LOG("using opt\n");
      H_ERR(cudaMemPrefetchAsync(xadj, (numNode + 1) * sizeof(vtx_t),
                                 FLAGS_device, *stream));
      if (mem_used < avail) {
        H_ERR(cudaMemPrefetchAsync(adjncy, numEdge * sizeof(vtx_t),
                                   FLAGS_device, *stream));
        if (needWeight)
          H_ERR(cudaMemPrefetchAsync(adjwgt, numEdge * sizeof(weight_t),
                                     FLAGS_device, *stream));
      } else {
        if (needWeight) {
          H_ERR(cudaMemPrefetchAsync(
              adjncy, (avail - (numNode + 1) * sizeof(vtx_t)) / 2, FLAGS_device,
              *stream));
          H_ERR(cudaMemPrefetchAsync(
              adjwgt, (avail - (numNode + 1) * sizeof(vtx_t)) / 2, FLAGS_device,
              *stream));
        } else
          H_ERR(cudaMemPrefetchAsync(adjncy,
                                     avail - (numNode + 1) * sizeof(vtx_t),
                                     FLAGS_device, *stream));
      }
      if (mem_used > avail) //
      {
        H_ERR(cudaMemAdvise(xadj, (numNode + 1) * sizeof(vtx_t),
                            cudaMemAdviseSetAccessedBy, FLAGS_device));
        H_ERR(cudaMemAdvise(adjncy, numEdge * sizeof(vtx_t),
                            cudaMemAdviseSetAccessedBy, FLAGS_device));
        if (needWeight)
          H_ERR(cudaMemAdvise(adjwgt, numEdge * sizeof(weight_t),
                              cudaMemAdviseSetAccessedBy, FLAGS_device));
      }
    } else {
      if (FLAGS_pf) {
        LOG("pfing\n");
        H_ERR(cudaMemPrefetchAsync(xadj, (numNode + 1) * sizeof(vtx_t),
                                   FLAGS_device, *stream));
        H_ERR(cudaMemPrefetchAsync(adjncy, numEdge * sizeof(vtx_t),
                                   FLAGS_device, *stream));
        if (needWeight)
          H_ERR(cudaMemPrefetchAsync(adjwgt, numEdge * sizeof(weight_t),
                                     FLAGS_device, *stream));
      }
      if (FLAGS_ab) //
      {
        LOG("AB hint\n");
        H_ERR(cudaMemAdvise(xadj, (numNode + 1) * sizeof(vtx_t),
                            cudaMemAdviseSetAccessedBy, FLAGS_device));
        H_ERR(cudaMemAdvise(adjncy, numEdge * sizeof(vtx_t),
                            cudaMemAdviseSetAccessedBy, FLAGS_device));
        if (needWeight)
          H_ERR(cudaMemAdvise(adjwgt, numEdge * sizeof(weight_t),
                              cudaMemAdviseSetAccessedBy, FLAGS_device));
      }
      if (FLAGS_rm) //
      {
        LOG("RM hint\n");
        H_ERR(cudaMemAdvise(xadj, (numNode + 1) * sizeof(vtx_t),
                            cudaMemAdviseSetReadMostly, FLAGS_device));
        H_ERR(cudaMemAdvise(adjncy, numEdge * sizeof(vtx_t),
                            cudaMemAdviseSetReadMostly, FLAGS_device));
        if (needWeight)
          H_ERR(cudaMemAdvise(adjwgt, numEdge * sizeof(weight_t),
                              cudaMemAdviseSetReadMostly, FLAGS_device));
      }
      if (FLAGS_pl) //
      {
        LOG("PL hint\n");
        H_ERR(cudaMemAdvise(xadj, (numNode + 1) * sizeof(vtx_t),
                            cudaMemAdviseSetPreferredLocation, FLAGS_device));
        H_ERR(cudaMemAdvise(adjncy, numEdge * sizeof(vtx_t),
                            cudaMemAdviseSetPreferredLocation, FLAGS_device));
        if (needWeight)
          H_ERR(cudaMemAdvise(adjwgt, numEdge * sizeof(weight_t),
                              cudaMemAdviseSetPreferredLocation, FLAGS_device));
      }
    }
  }
};
template <graphFmt fmt> class graph_t : public graph_base<fmt> {};
template <> class graph_t<CSR> : public graph_base<CSR> {
public:
  graph_t(bool _needweight = false) {
    this->hasZeroID = false;
    this->needWeight = _needweight;
  }
  ~graph_t() {
    // if (xadj != nullptr)
    //   H_ERR(cudaFree(xadj));
    // if (adjncy != nullptr)
    //   H_ERR(cudaFree(adjncy));
    // if (adjwgt != nullptr)
    //   H_ERR(cudaFree(adjwgt));
  }
};

/* Modified from
 * https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
 * Compute B = A for CSR matrix A, CSC matrix B. */
template <class I, class T>
void csr_tocsc(const I n_row, const I n_col, const I Ap[], const I Aj[],
               const T Ax[], I Bp[], I Bi[], T Bx[], bool weighted) {
  const I nnz = Ap[n_row];
  // compute number of non-zero entries per column of A
  std::fill(Bp, Bp + n_col, 0);
  for (I n = 0; n < nnz; n++) {
    Bp[Aj[n]]++;
  }
  // cumsum the nnz per column to get Bp[]
  for (I col = 0, cumsum = 0; col < n_col; col++) {
    I temp = Bp[col];
    Bp[col] = cumsum;
    cumsum += temp;
  }
  Bp[n_col] = nnz;
  for (I row = 0; row < n_row; row++) {
    for (I jj = Ap[row]; jj < Ap[row + 1]; jj++) {
      I col = Aj[jj];
      I dest = Bp[col];
      Bi[dest] = row;
      // Bx[dest] = Ax[jj];
      Bp[col]++;
    }
  }
  for (I col = 0, last = 0; col <= n_col; col++) {
    I temp = Bp[col];
    Bp[col] = last;
    last = temp;
  }
}
template <> class graph_t<CSC> : public graph_base<CSC> {
public:
  graph_t() {}
  ~graph_t() {}
  void CSR2CSC(graph_t<CSR> G) {
    weighted = G.weighted;
    needWeight = G.needWeight;
    numNode = G.numNode;
    numEdge = G.numEdge;
    H_ERR(cudaMallocManaged(&xadj, (numNode + 1) * sizeof(vtx_t)));
    H_ERR(cudaMallocManaged(&adjncy, numEdge * sizeof(vtx_t)));
    if (needWeight)
      H_ERR(cudaMallocManaged(&adjwgt, numEdge * sizeof(weight_t)));
    LOG("transferring CSR to CSC\n");
    csr_tocsc<vtx_t, weight_t>(numNode, numNode, G.xadj, G.adjncy, G.adjwgt,
                               xadj, adjncy, adjwgt, needWeight);
  }
};

} // namespace mgg
#endif
