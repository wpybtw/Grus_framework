#ifndef FRONTIER_CUH
#define FRONTIER_CUH
#include "common.cuh"
#include "print.cuh"
#include "timer.cuh"
#include "worklist.cuh"

namespace mgg
{

namespace frontier
{
using namespace intrinsics;
using namespace worklist;

class Frontier
{
// private:
public:
    vtx_t numNode;
    Worklist wl1, wl2;
    Worklist *wl_c, *wl_n;
    char *flag;
    vtx_t src;
    vtx_t wl_sz = 1;
    // vtx_t *char_sz;
    // float active_perct = 0;
    // float switch_th;
    char *finished_d;
    // cudaStream_t &stream;

public:
    Frontier() {}
    ~Frontier() {}
    void Init(vtx_t _numNode, vtx_t _src = 0, int gpu_id = 0, float size_threshold = 1.0, bool _full = false)
    {
        numNode = _numNode;
        src = _src;
        H_ERR(cudaSetDevice(gpu_id));
        // H_ERR(cudaMallocManaged(&flag, numNode * sizeof(char)));
        wl1.init(numNode * size_threshold);
        wl2.init(numNode * size_threshold);
        wl_c = &wl1;
        wl_n = &wl2;
        // cudaMalloc(&char_sz, sizeof(vtx_t));
        // H_ERR(cudaMemsetAsync(flag, 0, numNode * sizeof(char), stream));
        if (_full)
        {
            wlInitFull(wl1, numNode);
        }
        else
        {
            // if ((gpu_id / FLAGS_ngpu <= src / numNode) && ((gpu_id + 1) / FLAGS_ngpu > src / numNode))
            if (gpu_id == 0)
            {
                cout << "add src" << endl;
                wl_add_item(*wl_c, src, 0);
            }
        }
    }
    void Next()
    {
        Worklist *tmp = wl_c;
        wl_c = wl_n;
        wl_n = tmp;
        wl_reset(*wl_n);
    }
    __device__ vtx_t get_work_item(vtx_t id)
    {
        return wl_c->data[id];
    }
    __device__ vtx_t get_work_size()
    {
        return *wl_c->count;
    }
    __host__ vtx_t get_work_size_h()
    {
        return wl_get_sz(wl_c);
    }
    __device__ void add_work_item(vtx_t id)
    {
        wl_n->append(id);
    }
    // void Generate()
    // {
    //     wl_reset(wl, stream);
    //     flag_to_wl<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(
    //         wl, flag, numNode);
    //     wl_sz = wl_get_sz(&wl, stream);
    // }
    // bool CheckConverge()
    // {
    //     if (wl_sz == 0)
    //         return true;
    //     return false;
    // }
    // void Free()
    // {
    //     free(flag);
    // }
};
// void frontier_group_gather(Frontier *frontier_group,int size){
//     for (size_t i = 0; i < size; i++)
//     {
//         /* code */
//     }

// }
} // namespace frontier
} // namespace mgg
#endif