
#include "common.cuh"
#include "graph.cuh"
#include "frontier.cuh"
#include <gflags/gflags.h>
using namespace mgg;

DECLARE_int32(device);
DECLARE_string(input);
DECLARE_int32(src);

namespace bfs
{
// __global__ void BFSInit(uint *levels, int nnodes, vtx_t source)
// {
//     int tid = TID_1D;
//     if (tid < nnodes)
//     {
//         levels[tid] = tid == source ? 0 : INFINIT;
//     }
// }
// class Job_data
// {
// public:
//     uint src;
//     uint *level;
//     uint itr;
//     vtx_t num_Node;
//     void operator()(vtx_t _num_Node, uint _src)
//     {
//         num_Node = _num_Node;
//         src = _src;
//         init();
//     }
//     void init()
//     {
//         H_ERR(cudaMalloc(&level, num_Node * sizeof(uint)));
//         BFSInit<<<num_Node / BLOCK_SIZE + 1, BLOCK_SIZE>>>(level, num_Node, src);
//     }
// };

// template <typename graph_t, typename frontier_t>
// __global__ void push_kernel(graph_t G, frontier_t F, Job_data job)
// {
//     int tid = TID_1D;
//     vtx_t id, level, laneid, tmpid, wpid;
//     wpid = tid / 32;
//     laneid = threadIdx.x % 32;
//     if (wpid < F.get_work_size())
//     {
//         id = F.get_work_item(wpid);
//         level = job.itr + 1;
//         for (vtx_t i = G.xadj[id] + laneid; i < G.xadj[id + 1]; i += 32)
//         {
//             tmpid = G.adjncy[i];
//             if (job.level[tmpid] > level)
//             {
//                 job.level[tmpid] = level;
//                 F.add_work_item(tmpid);
//             }
//         }
//     }
// }

} // namespace bfs

bool BFSSingle()
{
    mgg::graph::Graph G;
    printf("BFS single\n");
    G.Init(false);
    bfs::Job_data job;
    job(G.numNode, FLAGS_src);
    mgg::frontier::Frontier F;
    F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, false);
    while (F.get_work_size_h() != 0)
    {
        bfs::push_kernel<mgg::graph::Graph, mgg::frontier::Frontier><<<F.get_work_size_h() / BLOCK_SIZE + 1, BLOCK_SIZE>>>(G, F, job);
        F.Next();
        job.itr++;
    }
    cout << "itr " << job.itr << endl;
    return 0;
}