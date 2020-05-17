#include <stdio.h>

//  this should use with MPS

__global__ void k()
{
    int i = 0;
    while (true)
    {
        i++;
    }
}
int main(int argc, char *argv[])
{
    int *mem, *mem2;
    int i;
    cudaFree(0);
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    printf("total available memory: %ld\n", avail / 1024 / 1024);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    k<<<atoi(argv[1]), 1024, props.sharedMemPerBlock>>>();
    printf("Press Enter key to continue...");
    fgetc(stdin);
    return 0;
}
