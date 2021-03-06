project(mgg LANGUAGES CUDA)
cmake_minimum_required(VERSION 3.8)

include(CheckLanguage)
check_language(CUDA)

#find_package(cub 1.1.4 REQUIRED)
#add_subdirectory(deps/cub)

set(CUDA_PATH "/usr/local/cuda")

# GFlags
add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)
#set_target_properties(mgg PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET(CUDA_SEPARABLE_COMPILATION ON)
set(BUILD_SHARED_LIBS OFF)

############## BUILD ##############
set(EXTRA_LIBS gflags )

#-Xcompiler -fopenmp

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75  -dc -std=c++11 ")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -Xcompiler   -ggdb  -lnvrt  ")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -O3 -Xcompiler  -DNDEBUG  ")
endif()
#-Woverloaded-virtual

set(CUDA_VERBOSE_BUILD ON)

include_directories(include deps/cub)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(bfs 
                src/worklist.cu
               samples/bfs/bfs.cu
					     samples/bfs/main.cu)
target_link_libraries(bfs ${EXTRA_LIBS})

add_executable(sssp 
                src/worklist.cu
               samples/sssp/sssp.cu
					     samples/sssp/main.cu)
target_link_libraries(sssp ${EXTRA_LIBS})

add_executable(pr 
                src/worklist.cu
               samples/pagerank/pr.cu
					     samples/pagerank/main.cu)
target_link_libraries(pr ${EXTRA_LIBS})

add_executable(cc 
                src/worklist.cu
               samples/cc/cc.cu
					     samples/cc/main.cu)
target_link_libraries(cc ${EXTRA_LIBS})

add_executable(bfs_2s 
                src/worklist.cu
                src/kernel.cu
               samples/bfs_2s/bfs_2s.cu
					     samples/bfs_2s/main.cu)
target_link_libraries(bfs_2s ${EXTRA_LIBS})

#add_executable(grus
#    src/app.cu
#    src/graph.cu
#    src/kernel_multi_gpu.cu
#    src/worklist.cu
#    )
#
#target_link_libraries(grus 
#    ${EXTRA_LIBS} 
#    -L/usr/local/cuda-10.1/targets/x86_64-linux/lib -fopenmp -ldl -lnvrtc  -lcudart -lcuda -lcudadevrt  
#    # -Wl,-b,binary,include/graph.cuh  -fopenmp
#    )





