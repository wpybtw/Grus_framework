#ifndef __APP_CUH
#define __APP_CUH

#include "common.cuh"
#include <sstream>
#include <string>
#include <gflags/gflags.h>


DEFINE_string(file, "", "path to task file");
DEFINE_bool(v, false, "verbose results");
DEFINE_bool(fused, false, "fused kernel");

DEFINE_int32(device, 0, "GPU ID");
DEFINE_string(input, "", "graph file");
DEFINE_int32(src, 0, "app src");

// DEFINE_bool(one, false, "process one by one");
// DEFINE_bool(s, false, "single job");
// DEFINE_string(app, "bfs", "app name");


template<typename App>
struct Skeleton
{
    int operator() (int argc, char **argv)
    {
        gflags::ParseCommandLineFlags(&argc, &argv, true);

        Graph G;
        G.Init(input);
        return 0;
    }
}
