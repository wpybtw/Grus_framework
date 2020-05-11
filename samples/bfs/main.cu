

#include "common.cuh"
#include "app.cuh"
#include <gflags/gflags.h>

bool BFS_single_gpu();
bool BFS_multi_gpu();

namespace bfs
{
struct App
{
    static bool Single() { return BFS_single_gpu(); }
    static bool Multi() { return BFS_multi_gpu(); }
};
} // namespace bfs

int main(int argc, char **argv)
{
    Skeleton<bfs::App> app;
    int exit = app(argc, argv);
    return 0;
}