

#include "common.cuh"
#include "app.cuh"
#include <gflags/gflags.h>

bool SSSP_single_gpu();
bool SSSP_multi_gpu();
namespace sssp
{
struct App
{
    static bool Single() { return SSSP_single_gpu(); }
    static bool Multi() { return SSSP_multi_gpu(); }
};
} // namespace sssp

int main(int argc, char **argv)
{
    Skeleton<sssp::App> app;
    int exit = app(argc, argv);
    return 0;
}