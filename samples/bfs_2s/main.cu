

#include "common.cuh"
#include "app.cuh"
#include <gflags/gflags.h>

bool BFS_2S_single_gpu();
// bool BFS_2S_multi_gpu();

namespace bfs_2s
{
struct App
{
    static bool Single() { return BFS_2S_single_gpu(); }
    static bool Multi() {return true;} //{ return BFS_2S_multi_gpu(); }
};
} // namespace bfs_2s

int main(int argc, char **argv)
{
    Skeleton<bfs_2s::App> app;
    int exit = app(argc, argv);
    return 0;
}