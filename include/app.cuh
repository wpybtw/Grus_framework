#ifndef __APP_CUH
#define __APP_CUH

#include "common.cuh"
#include "graph.cuh"
#include "frontier.cuh"
#include <sstream>
#include <string>
#include <gflags/gflags.h>


DEFINE_string(file, "", "path to task file");
DEFINE_bool(v, false, "verbose results");
DEFINE_bool(fused, false, "fused kernel");

DEFINE_int32(device, 0, "GPU ID");
DEFINE_string(input, "", "graph file");
DEFINE_int32(src, 0, "app src");

DEFINE_double(wl_th, 0.5, "wl switch threshold");
// DEFINE_bool(one, false, "process one by one");
// DEFINE_bool(s, false, "single job");
// DEFINE_string(app, "bfs", "app name");
DEFINE_bool(pf, false, "use UM prefetch");
DEFINE_bool(ab, false, "use UM hint: AccessedBy");
DEFINE_bool(rm, false, "use UM hint: ReadMostly");
DEFINE_bool(pl, false, "use UM hint: PreferredLocation");
DEFINE_bool(opt, true, "use opt UM policy");

template<typename App>
struct Skeleton
{
    int operator() (int argc, char **argv)
    {
        gflags::ParseCommandLineFlags(&argc, &argv, true);
        // G.Init();
        App::Single();
        return 0;
    }
};
#endif