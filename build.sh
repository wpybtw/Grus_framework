#!/bin/bash
git pull;cd build;make -j;
nvcc CMakeFiles/bfs.dir/src/worklist.cu.o CMakeFiles/bfs.dir/samples/bfs/bfs.cu.o CMakeFiles/bfs.dir/samples/bfs/main.cu.o -o bfs  -L/home/pywang/MGG/build/deps/gflags/lib  deps/gflags/libgflags_nothreads.a  -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -gencode arch=compute_75,code=sm_75
cd ..;
#  git pull;cd build;cmake ..;make -j;cd ..;
