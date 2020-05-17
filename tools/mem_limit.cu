#include <stdio.h>
int main(int argc, char *argv[]){
    int *mem,*mem2;
    int i;
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    printf( "total available memory: %ld\n" ,avail / 1024 / 1024);

    long long size=(long)1024*1024*1024*atoi(argv[1]);
    long long size2=(long)1024*1024*atoi(argv[2]);
    cudaMalloc(&mem,avail-(long)1024*1024*1024*atoi(argv[1])-(long)1024*1024*atoi(argv[2]));
    //cudaMalloc(&mem2,size2);

//    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    printf( "available memory: %ld\n" ,avail / 1024 / 1024);
    printf("Press Enter key to continue...");
    fgetc(stdin);
}
