#!/bin/bash

GR=".w.gr"
# ED=".w.edge"
# MTX=".w.mtx"
# FLAGS="--s --opt=1"
# --hint=0 --pf=0 --one=1

DATA=(lj orkut  uk-2005 twitter-2010 sk-2005   friendster uk-union)
SRC=( 0    0         0          0      10            101         1)

echo "--------------------Running MGG-----------------------"
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./build/bfs --input ~/data/${DATA[idx-1]}${GR} ${FLAGS} --src ${SRC[idx-1]} 
    # ./build/grus --input ~/data/${DATA[idx-1]}${GR} ${FLAGS} --app sssp --src ${SRC[idx-1]} 
    # ./build/grus --input ~/data/${DATA[idx-1]}${GR} ${FLAGS} --app pr  
    # ./build/grus --input ~/data/${DATA[idx-1]}${GR} ${FLAGS} --app cc  
done
