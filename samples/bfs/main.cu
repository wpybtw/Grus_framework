

#include "common.cuh"
#include "app.cuh"
#include <gflags/gflags.h>

bool BFSSingle();
namespace bfs
{
struct App
{
    static bool Single() { return BFSSingle(); }
};
} // namespace bfs

int main(int argc, char **argv)
{
    Skeleton<bfs::App> app;
    int exit = app(argc, argv);
    return 0;
}