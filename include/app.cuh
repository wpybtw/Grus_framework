#ifndef __APP_CUH
#define __APP_CUH

#include "common.cuh"
#include "frontier.cuh"
#include "graph.cuh"
#include <gflags/gflags.h>
#include <sstream>
#include <string>

DEFINE_string(file, "", "path to task file");
DEFINE_bool(v, false, "verbose results");
// DEFINE_bool(fused, false, "fused kernel");
DEFINE_bool(pull, false, "use pull-style execution");

DEFINE_int32(device, 0, "GPU ID");
DEFINE_string(input, "", "graph file");
DEFINE_string(output, "", "output result file");

DEFINE_int32(src, 0, "app src");
DEFINE_int32(ngpu, 1, "GPU number, 1 for single-GPU version ");

DEFINE_double(wl_th, 0.5, "wl switch threshold");
// DEFINE_bool(one, false, "process one by one");
// DEFINE_bool(s, false, "single job");
// DEFINE_string(app, "bfs", "app name");
DEFINE_bool(pf, false, "use UM prefetch");
DEFINE_bool(ab, false, "use UM hint: AccessedBy");
DEFINE_bool(rm, false, "use UM hint: ReadMostly");
DEFINE_bool(pl, false, "use UM hint: PreferredLocation");
DEFINE_bool(opt, true, "use opt UM policy");

template <typename App> struct Skeleton {
  int operator()(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // G.Init();
    if (FLAGS_ngpu == 1)
      App::Single();
    else
      App::Multi();
    return 0;
  }
};
#endif