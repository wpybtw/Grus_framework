

#include "common.cuh"
#include "app.cuh"
#include <gflags/gflags.h>

namespace bfs
{
struct App
{
    static bool Single() { return BFSSingle(); }
};
} // namespace bfs

int main(int argc, const char **argv)
{
    Skeleton<bfs::App> app;
    return 0;
}