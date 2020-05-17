#!/bin/bash
# git pull;cd build;
make -j;
nvcc CMakeFiles/bfs.dir/src/worklist.cu.o CMakeFiles/bfs.dir/samples/bfs/bfs.cu.o CMakeFiles/bfs.dir/samples/bfs/main.cu.o -o bfs  -L/home/pywang/MGG/build/deps/gflags/lib  deps/gflags/libgflags_nothreads.a  -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -gencode arch=compute_75,code=sm_75
nvcc CMakeFiles/sssp.dir/src/worklist.cu.o CMakeFiles/sssp.dir/samples/sssp/sssp.cu.o CMakeFiles/sssp.dir/samples/sssp/main.cu.o -o sssp  -L/home/pywang/MGG/build/deps/gflags/lib  deps/gflags/libgflags_nothreads.a  -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -gencode arch=compute_75,code=sm_75
nvcc CMakeFiles/pr.dir/src/worklist.cu.o CMakeFiles/pr.dir/samples/pagerank/pr.cu.o CMakeFiles/pr.dir/samples/pagerank/main.cu.o -o pr  -L/home/pywang/MGG/build/deps/gflags/lib  deps/gflags/libgflags_nothreads.a  -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -gencode arch=compute_75,code=sm_75
nvcc CMakeFiles/cc.dir/src/worklist.cu.o CMakeFiles/cc.dir/samples/cc/cc.cu.o CMakeFiles/cc.dir/samples/cc/main.cu.o -o cc  -L/home/pywang/MGG/build/deps/gflags/lib  deps/gflags/libgflags_nothreads.a  -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -gencode arch=compute_75,code=sm_75
# cd ..;
#  git pull;cd build;cmake ..;make -j;cd ..;
