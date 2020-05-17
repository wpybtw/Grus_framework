

#include "common.cuh"
#include "app.cuh"
#include <gflags/gflags.h>

bool PR_single_gpu();
bool PR_multi_gpu();

namespace pagerank
{
struct App
{
    static bool Single() { return PR_single_gpu(); }
    static bool Multi() { return PR_multi_gpu(); }
};
} // namespace pagerank

int main(int argc, char **argv)
{
    Skeleton<pagerank::App> app;
    int exit = app(argc, argv);
    return 0;
}