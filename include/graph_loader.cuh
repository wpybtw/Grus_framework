#ifndef _GRAPH_LOADER_CUH
#define _GRAPH_LOADER_CUH

#include "common.cuh"
#include "graph.cuh"
#include "intrinsics.cuh"
#include "print.cuh"
#include "timer.cuh"

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
class graph_loader {
public:
  string graphFilePath;

  bool hasZeroID;
  uint64_t numNode;
  uint64_t numEdge;
  uint64_t sizeEdgeTy;

  // graph
  vtx_t *xadj, *vwgt, *adjncy;
  // vtx_t *xadj_d, *vwgt_d, *adjncy_d;
  weight_t *adjwgt, *adjwgt_d;
  uint *inDegree;
  uint *outDegree;
  bool weighted;
  bool needWeight;

  uint64_t mem_used = 0;

  graph_loader() {}
  ~graph_loader() {}
  void gk_fclose(FILE *fp) { fclose(fp); }
  FILE *gk_fopen(const char *fname, const char *mode, const char *msg) {
    FILE *fp;
    char errmsg[8192];
    fp = fopen(fname, mode);
    if (fp != NULL)
      return fp;
    sprintf(errmsg, "file: %s, mode: %s, [%s]", fname, mode, msg);
    perror(errmsg);
    printf("Failed on gk_fopen()\n");
    return NULL;
  }
  void ReadGraphGR(graph_t<CSR> &G) {
    // uint *vsize;
    FILE *fpin;
    bool readew;
    fpin = gk_fopen(graphFilePath.data(), "r", "ReadGraphGR: Graph");
    size_t read;
    uint64_t x[4];
    if (fread(x, sizeof(uint64_t), 4, fpin) != 4) {
      printf("Unable to read header\n");
    }
    if (x[0] != 1) /* version */
      printf("Unknown file version\n");
    sizeEdgeTy = x[1];
    // uint64_t sizeEdgeTy = le64toh(x[1]);
    uint64_t num_Node = x[2];
    uint64_t num_Edge = x[3];
    cout << graphFilePath + " has " << num_Node << " nodes and " << num_Edge
         << "  edges\n";

    // H_ERR(cudaMallocHost(&xadj, (num_Node + 1) * sizeof(uint)));
    // H_ERR(cudaMallocHost(&adjncy, num_Edge * sizeof(uint)));

    H_ERR(cudaMallocManaged(&G.xadj, (num_Node + 1) * sizeof(vtx_t)));
    H_ERR(cudaMallocManaged(&G.adjncy, num_Edge * sizeof(vtx_t)));
    mem_used += (num_Node + 1) * sizeof(vtx_t) + num_Edge * sizeof(vtx_t);

    // adjwgt = nullptr;
    H_ERR(cudaMallocManaged(&G.adjwgt, num_Edge * sizeof(weight_t)));
    // um_used += num_Edge * sizeof(uint);
    weighted = true;
    if (!sizeEdgeTy) {
      // adjwgt = new uint[num_Edge];
      for (size_t i = 0; i < num_Edge; i++) {
        G.adjwgt[i] = 1;
      }
      weighted = false;
    }
    G.outDegree = new uint[num_Node];
    assert(G.xadj != NULL);
    assert(G.adjncy != NULL);
    // assert(vwgt != NULL);
    // assert(adjwgt != NULL);
    if (sizeof(uint) == sizeof(uint64_t)) {
      read = fread(G.xadj + 1, sizeof(uint), num_Node,
                   fpin); // This is little-endian data
      if (read < num_Node)
        printf("Error: Partial read of node data\n");
      fprintf(stderr, "read %lu nodes\n", num_Node);
    } else {
      for (int i = 0; i < num_Node; i++) {
        uint64_t rs;
        if (fread(&rs, sizeof(uint64_t), 1, fpin) != 1)
          printf("Error: Unable to read node data\n");
        G.xadj[i + 1] = rs;
      }
    }
    // edges are 32-bit
    if (sizeof(uint) == sizeof(uint32_t)) {
      read = fread(G.adjncy, sizeof(uint), num_Edge,
                   fpin); // This is little-endian data
      if (read < num_Edge)
        printf("Error: Partial read of edge destinations\n");
      // fprintf(stderr, "read %lu edges\n", numEdge);
    } else {
      assert(false &&
             "Not implemented"); /* need to convert sizes when reading */
    }
    for (size_t i = 0; i < num_Node; i++) {
      G.outDegree[i] = G.xadj[i + 1] - G.xadj[i];
    }
    uint maxD = std::distance(
        G.outDegree, std::max_element(G.outDegree, G.outDegree + num_Node));
    printf("%d has max out degree %d\n", maxD, G.outDegree[maxD]);
    if (sizeEdgeTy) {
      if (num_Edge % 2)
        if (fseek(fpin, 4, SEEK_CUR) != 0) // skip
          printf("Error when seeking\n");
      if (sizeof(uint) == sizeof(uint32_t)) {
        read = fread(G.adjwgt, sizeof(uint), num_Edge,
                     fpin); // This is little-endian data
        readew = true;
        if (read < num_Edge)
          printf("Error: Partial read of edge data\n");

        // fprintf(stderr, "read data for %lu edges\n", num_Edge);
      } else {
        assert(false &&
               "Not implemented"); /* need to convert sizes when reading */
      }
    }
    G.mem_used = mem_used;
    G.numNode = num_Node;
    G.numEdge = num_Edge;
    gk_fclose(fpin);
  }
  void Load(graph_t<CSR> &G, bool _needweight = true) {
    this->graphFilePath = FLAGS_input;
    // this->weighted = FLAGS_weight||false;
    this->hasZeroID = false;
    this->needWeight = _needweight;
    ReadGraphGR(G);
  }
};

} // namespace mgg
#endif
