# Grus_framework
Grus: Towards Unified-Memory-Efficient, High-Performance Graph Processing on GPU

Grus is an elaborately re-designed framework that aims to
unleash the performance potential of CPU-GPU architecture.

## Build from the source
+ Prerequisites: CUDA, C++11, CMake v3.15
+ Third party library: Gflags

```
mkdir build
cd build
cmake ..
make -j8
```



## Dataset
Grus uses [Galios](https://iss.oden.utexas.edu/?p=projects/galois) graph format (.gr) as the input. Other formats like Edgelist (form [SNAP](http://snap.stanford.edu/data/index.html)) or Matrix Market can be transformed into it with GALOIS' graph-convert tool. Compressed graphs like [Webgraph](http://law.di.unimi.it/datasets.php) need to be uncompressed first.

Here is an example:
```
wget http://snap.stanford.edu/data/wiki-Vote.txt.gz
gzip -d wiki-Vote.txt.gz
$GALOIS_PATH/build/tools/graph-convert/graph-convert -edgelist2gr  ~/data/wiki-Vote.txt  ~/data/wiki-Vote.gr
