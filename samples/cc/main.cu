

#include "common.cuh"
#include "app.cuh"
#include <gflags/gflags.h>

bool CC_single_gpu();
bool CC_multi_gpu();

namespace cc
{
struct App
{
    static bool Single() { return CC_single_gpu(); }
    static bool Multi() { return CC_multi_gpu(); }
};
} // namespace cc

int main(int argc, char **argv)
{
    Skeleton<cc::App> app;
    int exit = app(argc, argv);
    return 0;
}