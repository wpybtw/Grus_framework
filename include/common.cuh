#ifndef __COMMON_CUH
#define __COMMON_CUH

// #include <chrono>
// #include <cstdlib>
#include <cstring>
// #include <ctime>
#include <fstream>
#include <iostream>
// #include <iterator>
// #include <locale>
#include <math.h>
// #include <sstream>
// #include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <limits>
// #include <sys/stat.h>
// #include <sys/types.h>
#include <vector>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <gflags/gflags.h>

namespace mgg
{
namespace
{
#define BLOCK_SIZE 512

#define ALPHA 0.85
#define EPSILON 0.01

// #define ACT_TH 0.01

using std::cout;
using std::endl;
using std::flush;
using std::ifstream;
using std::ofstream;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

using uint = unsigned int;
using vtx_t = unsigned int; // vertex_num < 4B
using weight_t = unsigned int;
using ulong = unsigned long;

const unsigned int INFINIT = std::numeric_limits<uint>::max() - 1;

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)

inline void HandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
  {
    printf("%s in %s at line %d\n",
           cudaGetErrorString(err), file, line);
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    fflush(stdout);
    exit(EXIT_FAILURE);
  }
}
#define H_ERR(err) (HandleError(err, __FILE__, __LINE__))

enum Memtype
{
  normal,
  um,
};
enum graphFmt { CSR, CSC };
enum subgraphFmt { NORMAL, COMPRESSED };
enum frontierType { BDF_AUTO, BDF, WL, BITMAP };
// enum class APP
// {
//   BFS,
//   SSSP,
//   PR,
//   CC
// };

} // namespace
} // namespace mgg

// enum App
// {
//   BFS = 0,
//   SSSP = 1,
//   PR = 2,
//   CC = 3
// };

// struct Task
// {
//   App app;
//   int source;
// };

// struct Input
// {
//   string file;
//   vector<Task> tasks;
//   string output;
//   int device = 0;
//   int napp;
// };

// template <typename T>
// void printD(T *DeviceData, int n)
// {
//   T *tmp = new T[n];
//   cudaMemcpy(tmp, DeviceData, n * sizeof(T), cudaMemcpyDeviceToHost);
//   for (size_t i = 0; i < n; i++)
//   {
//     cout << tmp[i] << "\t";
//     if (i % 10 == 9)
//     {
//       cout << endl;
//     }
//   }
// }

#endif