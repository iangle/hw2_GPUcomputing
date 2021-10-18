
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "pgmUtility.h"

// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: Please follow the instructions stated in the write-up regarding the interface of the functions.
// NOTE: You might have to change the name of this file into pgmUtility.cu if needed.

int pgmDrawCircle( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header )
{
    int *d_a;

    int *p1;

    int *p2;

    size_t bytes = numCols*numRows*sizeof(int);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&p1, 2*sizeof(int));
    cudaMalloc(&p2, 2*sizeof(int));

    cudaMemcpy(d_a, pixels, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)numRows*numCols/blockSize);

    // Execute the kernel
    addCircle<<<gridSize, blockSize>>>(d_a, numRows, numCols, centerRow, centerCol, radius, p1, p2);

    cudaMemcpy(pixels, d_a, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);

    free(p1);
    free(p2);

    return 0;

}