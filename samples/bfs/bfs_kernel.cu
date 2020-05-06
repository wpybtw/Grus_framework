// #include "common.cuh"
// #include "graph.cuh"
// #include "frontier.cuh"
// #include <gflags/gflags.h>
// using namespace mgg;

// namespace bfs
// {
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
// }