

#include "common.cuh"
#include "app.cuh"
#include <gflags/gflags.h>

bool SSSPSingle();
namespace sssp
{
struct App
{
    static bool Single() { return SSSPSingle(); }
};
} // namespace sssp

int main(int argc, char **argv)
{
    Skeleton<sssp::App> app;
    int exit = app(argc, argv);
    return 0;
}