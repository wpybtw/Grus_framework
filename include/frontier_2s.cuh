#ifndef _FRONTIER_2S_CUH
#define _FRONTIER_2S_CUH
#include "common.cuh"
#include "print.cuh"
#include "timer.cuh"
#include "worklist.cuh"
#include <gflags/gflags.h>
DECLARE_double(wl_th);
// DEFINE_double(wl_th, 0.5, "wl switch threshold");

namespace mgg {

namespace frontier {
using namespace intrinsics;
using namespace worklist;

template <frontierType type> class Frontier_2S {
private:
public:
  Frontier_2S();
  ~Frontier_2S();
  void Init();
  void Next();
  __host__ vtx_t get_work_size_h();
};

template <> class Frontier_2S<BDF> {
  // private:
public:
  vtx_t numNode;
  Worklist wl1, wl2;
  Worklist *wl_remote, *wl_local;
  char *flag1, *flag2, *flag_local;
  vtx_t src;
  vtx_t wl_sz = 1, wl_sz_remote = 1, wl_sz_local = 0;
  vtx_t *flag_sz;
  float active_perct = 0;
  float switch_th = 1.0;
  char *finished_d;
  bool current_wl = true;

public:
  Frontier_2S() {}
  ~Frontier_2S() {}
  void Init(vtx_t _numNode, vtx_t _src = 0, int gpu_id = 0,
            float size_threshold = 1.0, bool _full = false) {
    numNode = _numNode;
    src = _src;
    switch_th = FLAGS_wl_th;
    H_ERR(cudaSetDevice(gpu_id));
    H_ERR(cudaMalloc(&flag_local, numNode * sizeof(char)));
    H_ERR(cudaMalloc(&flag2, numNode * sizeof(char)));
    wl1.init(numNode * size_threshold);
    wl2.init(numNode * size_threshold);
    wl_remote = &wl1;
    wl_local = &wl2;
    cudaMalloc(&flag_sz, sizeof(vtx_t));
    H_ERR(cudaMemsetAsync(flag2, 0, numNode * sizeof(char), NULL));
    H_ERR(cudaMemsetAsync(flag_local, 0, numNode * sizeof(char), NULL));
    if (_full) {
      wl_remote->initFull(numNode);
      wl_sz_remote = numNode;
    } else {
      wl_remote->add_item(src, 0);
    }
  }
  void Next() { // cudaStream_t &stream = NULL

    // flag_to_wl<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE, 0, NULL>>>(wl1, flag2,
    //                                                               numNode);
    // std::swap(wl_remote,wl_local)
    uint flag_num, *flag_num_ptr;
    cudaMalloc(&flag_num_ptr, sizeof(uint));
    worklist::get_flag_num<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE, 0, NULL>>>(
        flag2, numNode, flag_num_ptr);
    H_ERR(cudaMemcpy(&flag_num, flag_num_ptr, sizeof(vtx_t),
                     cudaMemcpyDeviceToHost));
    // LOG("active vtx from flag %u \n", flag_num);

    // previous remote become local
    wl_to_flag<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE, 0, NULL>>>(
        *wl_remote, flag_local, numNode);
    wl_remote->reset(); // flag2 active, flag1 0
    wl_local->reset();
    // flag_local is too expensive??
    flag_to_wl_remote_local<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE, 0, NULL>>>(
        *wl_remote, *wl_local, flag_local, flag2, numNode);
    H_ERR(cudaMemsetAsync(flag_local, 0,
                          numNode * sizeof(char))); // reset flag_local
    wl_sz_remote = wl_remote->get_sz();
    wl_sz_local = wl_local->get_sz();
    wl_sz = wl_sz_remote + wl_sz_local;
    H_ERR(cudaMemsetAsync(flag2, 0, numNode * sizeof(char))); // flag2 0
    LOG("wl_sz_remote %d wl_sz_local %d total %d \n", wl_sz_remote, wl_sz_local,
        wl_sz);
    // print::PrintResults(wl_remote->data, std::min(50, (int)wl_sz_remote));
  }
  // __device__ vtx_t get_work_item(vtx_t id) { return wl_remote->data[id]; }

  __device__ vtx_t get_work_size_remote() { return *wl_remote->count; }
  __device__ vtx_t get_work_size_local() { return *wl_local->count; }

  __host__ vtx_t get_work_size_local_h() { return wl_local->get_sz(); }
  __host__ vtx_t get_work_size_remote_h() { return wl_remote->get_sz(); }
  __host__ vtx_t get_work_size_totoal_h() { return wl_sz; }

  // __device__ void add_work_item(vtx_t id) { wl_n->append(id); }
  __host__ bool finish() { return wl_sz == 0 ? true : false; }
};
// template <> class Frontier_2S<BITMAP> {
//   // private:
// public:
//   vtx_t numNode;
//   char *flag1, *flag2;
//   Worklist *wl_c, *wl_n;
//   vtx_t src;
//   vtx_t wl_sz = 1;
//   vtx_t *flag_sz;
//   float active_perct = 0;
//   char *finished_d, finished = false;
//   // cudaStream_t &stream;

// public:
//   Frontier_2S() {}
//   ~Frontier_2S() {}
//   void Init(vtx_t _numNode, vtx_t _src = 0, int gpu_id = 0,
//             float size_threshold = 1.0, bool _full = false) {
//     numNode = _numNode;
//     src = _src;
//     H_ERR(cudaSetDevice(gpu_id));
//     H_ERR(cudaMalloc(&flag1, numNode * sizeof(char)));
//     H_ERR(cudaMalloc(&flag2, numNode * sizeof(char)));
//     H_ERR(cudaMalloc(&finished_d, sizeof(char)));
//     H_ERR(cudaMemset(finished_d, 1, sizeof(char)));
//     // cudaMalloc(&flag_sz, sizeof(vtx_t));
//     H_ERR(cudaMemsetAsync(flag2, 0, numNode * sizeof(char), NULL));
//     if (_full) {
//       H_ERR(cudaMemsetAsync(flag1, 1, numNode * sizeof(char), NULL));
//     } else {
//       H_ERR(cudaMemsetAsync(flag1, 0, numNode * sizeof(char), NULL));
//       H_ERR(cudaMemsetAsync(&flag1[src], 1, sizeof(char), NULL));
//     }
//   }
//   void Next() { // cudaStream_t &stream = NULL
//     // wl1.reset(); // flag2 active, flag1 dirty
//     H_ERR(cudaMemcpy(&finished, finished_d, sizeof(char),
//                      cudaMemcpyDeviceToHost));
//     H_ERR(cudaMemset(finished_d, 1, sizeof(char)));
//     std::swap(flag2, flag1);
//     if (!finished)
//       H_ERR(cudaMemsetAsync(flag2, 0,
//                             numNode * sizeof(char))); // flag1 active, flag2
//                             0
//   }
//   __device__ vtx_t get_work_item(vtx_t id) { return wl_c->data[id]; }
//   __device__ vtx_t get_work_size() { return *wl_c->count; }
//   __host__ vtx_t get_work_size_h() { return wl_c->get_sz(); }
//   __device__ void add_work_item(vtx_t id) { wl_n->append(id); }
//   // __host__ bool finish() { return wl_c->get_sz(); }
//   __host__ bool finish() { return finished; }
// };
// template <> class Frontier_2S<BDF_AUTO> {
//   // private:
// public:
//   vtx_t numNode;
//   Worklist wl1, wl2;
//   Worklist *wl_c, *wl_n;
//   char *flag1, *flag2;
//   vtx_t src;
//   vtx_t wl_sz = 1;
//   vtx_t *flag_sz;
//   float active_perct = 0;
//   float switch_th = 1.0;
//   char *finished_d;
//   bool current_wl = true;
//   // cudaStream_t &stream;

// public:
//   Frontier_2S() {}
//   ~Frontier_2S() {}
//   void Init(vtx_t _numNode, vtx_t _src = 0, int gpu_id = 0,
//             float size_threshold = 1.0, bool _full = false) {
//     numNode = _numNode;
//     src = _src;
//     switch_th = FLAGS_wl_th;
//     H_ERR(cudaSetDevice(gpu_id));
//     H_ERR(cudaMalloc(&flag1, numNode * sizeof(char)));
//     H_ERR(cudaMalloc(&flag2, numNode * sizeof(char)));
//     wl1.init(numNode * size_threshold);
//     wl_c = &wl1;
//     cudaMalloc(&flag_sz, sizeof(vtx_t));
//     H_ERR(cudaMemsetAsync(flag2, 0, numNode * sizeof(char), NULL));
//     if (_full) {
//       current_wl = false;
//       H_ERR(cudaMemsetAsync(flag1, 1, numNode * sizeof(char), NULL));
//       wl_sz = numNode;
//     } else {
//       H_ERR(cudaMemsetAsync(flag1, 0, numNode * sizeof(char), NULL));
//       wl_c->add_item(src, 0);
//     }
//   }
//   void Next() {  // cudaStream_t &stream = NULL
//     wl1.reset(); // flag2 active, flag1 0
//     flag_to_wl<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE, 0, NULL>>>(wl1, flag2,
//                                                                   numNode);
//     wl_sz = wl1.get_sz();
//     // LOG("wl_sz %d\n",wl_sz);
//     std::swap(flag2, flag1);
//     H_ERR(cudaMemsetAsync(flag2, 0,
//                           numNode * sizeof(char))); // flag1 active, flag2 0
//     current_wl = switch_th * numNode > wl_c->get_sz() ? true : false;
//   }
//   __device__ vtx_t get_work_item(vtx_t id) { return wl_c->data[id]; }
//   __device__ vtx_t get_work_size() { return *wl_c->count; }
//   __host__ vtx_t get_work_size_h() { return wl_c->get_sz(); }
//   __device__ void add_work_item(vtx_t id) { wl_n->append(id); }
//   __host__ bool finish() {
//     if (current_wl)
//       return get_work_size_h() == 0 ? true : false;
//     else
//       return wl_sz == 0 ? true : false;
//   }
// };
// template <> class Frontier_2S<WL> {
//   // private:
// public:
//   vtx_t numNode;
//   Worklist wl1, wl2;
//   Worklist *wl_c, *wl_n;
//   char *flag;
//   vtx_t src;
//   vtx_t wl_sz = 1;
//   vtx_t *flag_sz;
//   float active_perct = 0;
//   char *finished_d;

// public:
//   Frontier_2S() {}
//   ~Frontier_2S() {}
//   void Init(vtx_t _numNode, vtx_t _src = 0, int gpu_id = 0,
//             float size_threshold = 1.0, bool _full = false) {
//     numNode = _numNode;
//     src = _src;
//     H_ERR(cudaSetDevice(gpu_id));
//     wl1.init(numNode * size_threshold);
//     wl2.init(numNode * size_threshold);
//     wl_c = &wl1;
//     wl_n = &wl2;
//     if (_full) {
//       wl1.initFull(numNode);
//     } else {
//       cout << "add src" << endl;
//       wl_c->add_item(src, 0);
//     }
//   }
//   void Next() {
//     Worklist *tmp = wl_c;
//     wl_c = wl_n;
//     wl_n = tmp;
//     wl_n->reset();
//   }
//   __device__ vtx_t get_work_item(vtx_t id) { return wl_c->data[id]; }
//   __device__ vtx_t get_work_size() { return *wl_c->count; }
//   __host__ vtx_t get_work_size_h() { return wl_c->get_sz(); }
//   __device__ void add_work_item(vtx_t id) { wl_n->append(id); }
//   __host__ bool finish() { return get_work_size_h() == 0 ? true : false; }
// };

} // namespace frontier
} // namespace mgg
#endif
// if ((gpu_id / FLAGS_ngpu <= src / numNode) && ((gpu_id + 1) /
// FLAGS_ngpu > src / numNode))

// if (current_wl) {
//   if (switch_th * numNode > wl_n->get_sz()) {
//     std::swap(wl_c, wl_n);
//     wl_n->reset();
//   } else {
//     // wl to flag
//   }
// }