#ifndef PRINT_CUH
#define PRINT_CUH

#include "common.cuh"
#include <algorithm>


#define LOG(...) print::myprintf(__FILE__, __LINE__, __VA_ARGS__)
#define VNAME(value) (#value)
namespace mgg {

namespace print {
template <typename... Args>
__host__ __device__ __forceinline__ void
myprintf(const char *file, int line, const char *__format, Args... args) {
#if defined(__CUDA_ARCH__)
  if (TID_1D == 0) {
    printf("%s:%d GPU: ", file, line);
    printf(__format, args...);
  }
#else
  printf("%s:%d HOST: ", file, line);
  printf(__format, args...);
#endif
}
template <typename T> void PrintResults(T *results, uint n) {
  cout << "First " << n << " elements of " << VNAME(results) << " :\n[";
  for (int i = 0; i < n; i++) {
    if (i > 0)
      cout << " ";
    cout << i << ":" << results[i];
  }
  cout << "]\n";
}

template <typename T> void SaveResults(string filepath, T *results, uint n) {
  cout << "Saving the results into the following file:\n";
  cout << ">> " << filepath << endl;
  ofstream outfile;
  outfile.open(filepath);
  for (int i = 0; i < n; i++)
    outfile << i << " " << results[i] << endl;
  outfile.close();
  cout << "Done saving.\n";
}
template <typename RankPair> bool PRCompare(RankPair elem1, RankPair elem2) {
  return elem1.page_rank > elem2.page_rank;
}

template <typename VertexId, typename Value> struct RankPair {
  VertexId vertex_id;
  Value page_rank;

  RankPair(VertexId vertex_id, Value page_rank)
      : vertex_id(vertex_id), page_rank(page_rank) {}
};

inline void topRank(float *results, uint n, uint m) {

  RankPair<uint, float> *pr_list =
      (RankPair<uint, float> *)malloc(sizeof(RankPair<uint, float>) * n);

  for (int i = 0; i < n; ++i) {
    pr_list[i].vertex_id = i;
    pr_list[i].page_rank = results[i];
  }

  std::stable_sort(pr_list, pr_list + n, PRCompare<RankPair<uint, float>>);

  uint *node_id = new uint[m];
  float *rank_d = new float[m];
  for (int i = 0; i < m; ++i) {
    node_id[i] = pr_list[i].vertex_id;
    rank_d[i] = pr_list[i].page_rank;
  }

  for (uint i = 0; i < m; ++i) {
    printf("EtaGraph: Vertex ID: %lld, PageRank: %.8le\n",
           (long long)node_id[i], (double)rank_d[i]);
  }
}
} // namespace print
} // namespace mgg
#endif //
