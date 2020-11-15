#ifndef _SUBGRAPH_CUH
#define _SUBGRAPH_CUH

#include "common.cuh"
#include "graph.cuh"
#include "intrinsics.cuh"
#include "print.cuh"
#include "timer.cuh"
#include "worklist.cuh"

#include <cub/cub.cuh>
#include <thrust/scan.h>

#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvrtc.h>

#include <algorithm>
#include <assert.h>
#include <gflags/gflags.h>

// frontier + xadj = active list
// active list--> compute offset,
// as active list are not known ahead, edges of a vtx must be consistent
// todo How to avoid reduplicative transfer
//      use a bitmap to indicate if in previous local sbgraph, then compute
//      resident vertices first.
// gap coding
// Variable-Length Encoding
// Interval Encoding

namespace mgg {

template <subgraphFmt subgraph_t> class Subgraph {
public:
  void reserve() {}
  __forceinline__ __device__ vtx_t get_vtx(vtx_t id) {}
  __forceinline__ __device__ vtx_t get_vtx_from_id(vtx_t id) {}
  __forceinline__ __device__ uint get_vtx_degree(vtx_t id) {}
  __forceinline__ __device__ vtx_t get_edge_dst(vtx_t id, vtx_t offset) {}
  __forceinline__ __device__ vtx_t get_edge_weight(vtx_t id, vtx_t offset) {}
};

template <> class Subgraph<NORMAL> {
private:
  /* data */
public:
  vtx_t numNode = 0, numEdge = 0, capacity = 0;
  vtx_t *lookup_buffer;
  vtx_t *vtx;
  uint *degreeBuffer;
  vtx_t *vtx_ptr;
  vtx_t *edges;
  weight_t *weight;

  uint *degreeBuffer_h;
  vtx_t *lookup_buffer_h, *vtx_h;
  // lookup buffer
  //   vtx_t *active_vtx_list;
  // worklist::Worklist active_list;
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0, temp_storage_bytes_reserved = 0;

  Subgraph() {}
  ~Subgraph() {}
  void reserve(vtx_t m, vtx_t n) {
    capacity = m;
    H_ERR(cudaMallocManaged(&vtx, m * sizeof(vtx_t)));
    H_ERR(cudaMallocManaged(&degreeBuffer, m * sizeof(uint)));
    H_ERR(cudaMallocManaged(&lookup_buffer, m * sizeof(weight_t)));

    H_ERR(cudaMallocManaged(&vtx_ptr, (m + 1) * sizeof(vtx_t)));
    H_ERR(cudaMallocManaged(&edges, n * sizeof(vtx_t)));
    H_ERR(cudaMallocManaged(&weight, n * sizeof(weight_t)));

    // malloc();
    // malloc();
    // malloc();
  }
  __forceinline__ __device__ vtx_t get_vtx_from_id(vtx_t id) {
    return lookup_buffer[id];
  }
  __forceinline__ __device__ vtx_t get_vtx(vtx_t id) { return vtx[id]; }
  __forceinline__ __device__ uint get_vtx_degree(vtx_t id) {
    return vtx_ptr[id + 1] - vtx_ptr[id];
  }
  __forceinline__ __device__ vtx_t get_edge_id(vtx_t id, vtx_t offset) {
    return vtx_ptr[id] + offset;
  }
  __forceinline__ __device__ vtx_t get_edge_dst(vtx_t id, vtx_t offset) {
    return edges[vtx_ptr[id] + offset];
  }
  __forceinline__ __device__ vtx_t get_edge_weight(vtx_t id, vtx_t offset) {
    return weight[vtx_ptr[id] + offset];
  }

  template <typename graph_t, typename worklist_t>
  void compute_ptr(graph_t G, worklist_t wl) {
    //   get F active vertices
    numNode = wl.get_sz();
    // LOG("compute_ptr for %d vtx subgraph\n",numNode);
    cudaMemcpy(vtx, wl.data, numNode * sizeof(vtx_t), cudaMemcpyDeviceToDevice);
    worklist::compute_lookup_buffer<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
        vtx, vtx_ptr, G.xadj, lookup_buffer, degreeBuffer, numNode);
    // compute offset
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  degreeBuffer, vtx_ptr + 1, numNode);
    if (temp_storage_bytes > temp_storage_bytes_reserved) {
      LOG("re-mallocing %d buffer for scan, old %d \n", temp_storage_bytes,
          temp_storage_bytes_reserved);
      cudaFree(d_temp_storage);
      cudaMalloc(&d_temp_storage, temp_storage_bytes);
      temp_storage_bytes_reserved = temp_storage_bytes;
    }
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  degreeBuffer, vtx_ptr + 1, numNode);
    // thrust::inclusive_scan(thrust::host, degreeBuffer, degreeBuffer +
    // numNode, vtx_ptr + 1);
  }
  template <typename graph_t, typename worklist_t>
  void construct(graph_t G, worklist_t wl) {
    // LOG("constructing for %d vtx subgraph\n",numNode);
    // numEdge = vtx_ptr[numNode];
    cudaMemcpy(&numEdge, &vtx_ptr[numNode], sizeof(vtx_t),
               cudaMemcpyDeviceToHost);
    //  let vtx_ptr,degreeBuffer,lookup_buffer be local
    for (size_t i = 0; i < numNode; i++) {
      memcpy(&edges[vtx_ptr[i]], &G.adjncy[G.xadj[lookup_buffer[i]]],
             degreeBuffer[i] * sizeof(vtx_t));
      //  weight
    }
  }
  // template <typename graph_t, typename worklist_t>
  void move() {
    H_ERR(cudaMemPrefetchAsync(vtx, numNode * sizeof(vtx_t), FLAGS_device,
                               nullptr));
    H_ERR(cudaMemPrefetchAsync(vtx_ptr, (numNode + 1) * sizeof(vtx_t),
                               FLAGS_device, nullptr));
    H_ERR(cudaMemPrefetchAsync(degreeBuffer, numNode * sizeof(vtx_t),
                               FLAGS_device, nullptr));
    H_ERR(cudaMemPrefetchAsync(lookup_buffer, capacity * sizeof(vtx_t),
                               FLAGS_device, nullptr));
    H_ERR(cudaMemPrefetchAsync(edges, numEdge * sizeof(vtx_t), FLAGS_device,
                               nullptr));
  }
  template <typename graph_t, typename worklist_t>
  void build(graph_t G, worklist_t wl) {
    compute_ptr<graph_t, worklist_t>(G, wl);
    construct<graph_t, worklist_t>(G, wl);
    move();
    // printf("SG numEdge: %d\n", numEdge);
    // printf("vtx\t");
    // print::PrintResults(vtx, numNode);
    // printf("vtx_ptr\t");
    // print::PrintResults(vtx_ptr, numNode + 1);
    // printf("degreeBuffer\t");
    // print::PrintResults(degreeBuffer, numNode);
    // printf("edges\t");
    // print::PrintResults(edges, numEdge);
  }
  void clean() {}
};
// template <>
// class Subgraph<NORMAL> {
// private:
//   /* data */
// public:
//   vtx_t numNode;
//   vtx_t *resident_vtx, *remote_vtx;
//   uint *degreeBuffer;
//   vtx_t *resident_vtx_ptr, *remote_vtx_ptr;
//   weight_t *resident_weight, *remote_weight;

//   //   vtx_t *active_vtx_list;
//   worklist::Worklist active_list;

//   subgraph(/* args */);
//   ~subgraph();
//   void reserve() {}

//   template <typename graph_t, typename frontier_t>
//   void compute_local_ptr(graph_t G, frontier_t F) {
//     //   get F active vertices
//     vtx_t numActive = F.get_active_num();

//     // active_list;   need order??
//     F.get_active_vtx(active_list);
//     // compute resident_vtx and remote_vtx

//     // return for GPU execution
//   }

//   template <typename graph_t, typename frontier_t>
//   void compute_ptr(graph_t G, frontier_t F) {

//     // compute degreeBuffer for remote
//     // compute offset
//   }
// };
}
#endif