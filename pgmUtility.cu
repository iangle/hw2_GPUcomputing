#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pgmUtility.h"
#include "pgmProcess.h"
// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: Please follow the instructions stated in the write-up regarding the interface of the func$
// NOTE: You might have to change the name of this file into pgmUtility.cu if needed.

int ** pgmRead( char **header, int *numRows, int *numCols, FILE *in){

        int i, j;

        // read in header of the image first
        for( i = 0; i < rowsInHeader; i ++)
        {
                if ( header[i] == NULL )
                {
                        return NULL;
                }
                if( fgets( header[i], maxSizeHeadRow, in ) == NULL )
                {
                        return NULL;
                }
        }
        // extract rows of pixels and columns of pixels
        sscanf( header[rowsInHeader - 2], "%d %d", numCols, numRows );  // in pgm the first number$

        // Now we can intialize the pixel of 2D array, allocating memory
        //need to change this to int*pixels as well as the logic
        int **pixels = ( int ** ) malloc( ( *numRows ) * sizeof( int * ) );
        for( i = 0; i < *numRows; i ++)
        {
                pixels[i] = ( int * ) malloc( ( *numCols ) * sizeof( int ) );
                if ( pixels[i] == NULL )
                {
                        return NULL;
                }
        }
  
        // read in all pixels into the pixels array.
        for( i = 0; i < *numRows; i ++ )
                for( j = 0; j < *numCols; j ++ )
                        if ( fscanf(in, "%d ", *( pixels + i ) + j) < 0 )
                                return NULL;

        return pixels;
}//end pgmRead

int pgmDrawCircle( int **pixels, int numRows, int numCols, int centerRow,
                  int centerCol, int radius, char **header ){

    int *d_a;

    int *p1;

    int *p2;

    int *flatArray =(int*) malloc(sizeof(int)*numCols*numRows);

    flattenArray(pixels, flatArray, numRows, numCols);

    size_t bytes = numCols*numRows*sizeof(int);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&p1, 2*sizeof(int));
    cudaMalloc(&p2, 2*sizeof(int));

    cudaMemcpy(d_a, flatArray, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)numRows*numCols/blockSize);

    // Execute the kernel
    addCircle<<<gridSize, blockSize>>>(d_a, numRows, numCols, centerRow, centerCol, radius, p1, p2);

    cudaMemcpy(flatArray, d_a, bytes, cudaMemcpyDeviceToHost);

    unFlattenArray(pixels, flatArray, numRows, numCols);

    cudaFree(d_a);
    cudaFree(p1);
    cudaFree(p2);

    free(flatArray);

    return 0;
}

//Draws an edge around the provided PGM.
__global__ void drawEdge (int* pixels, int numRows, int numCols, int edgeWidth)
{
  //Standard CUDA Variables
  int ix = blockIdx.x + blockDim.x + threadIdx.x;
  int iy = blockIdx.y + blockDim.y + threadIdx.y;
  int idx = iy*numCols + ix;
  
  if(ix < numCols && iy < numRows && (ix > numCols - edgeWidth || ix < edgewidth) && (iy > numRows - edgeWitdh || iy < edgeWidth))
    pixels[idx] = 0;
}

//Draws a line between two points within the provided PGM.
__global__ void drawLine (int* pixels, int numRows, int numCols, float slope, int* p1, int* p2)
{
  //Standard CUDA Variables.
  int ix = blockIdx.x + blockDim.x + threadIdx.x;
  int iy = blockIdx.y + blockDim.y + threadIdx.y;
  int idx = iy*numCols + ix;
  
  if((iy - (slope * ix) - p1[0]) == 0 && ix < numCols && iy < numRows && iy <= p2[0] && iy >= p1[0] && ix <= p2[1] && ix >= p1[1])
     pixels[idx] = 0;
}

int pgmWrite( const char **header, const int **pixels, int numRows, int numCols, FILE *out ){

    int i, j;

    // write the header
    for ( i = 0; i < rowsInHeader; i ++ )
    {
        fprintf(out, "%s", *( header + i ) );
    }

    // write the pixels
    for( i = 0; i < numRows; i ++ )
    {
        for ( j = 0; j < numCols; j ++ )
        {
                if ( j < numCols - 1 )
                        fprintf(out, "%d ", pixels[i][j]);
                else
                        fprintf(out, "%d\n", pixels[i][j]);
                }
        }
        return 0;
}

void flattenArray(int **pixels, int *storageArray, int rowSize, int colSize)
{

    int index = 0;

    for(int i = 0; i < rowSize; i++)
    {
        for(int j = 0; j < colSize; j++)
        {
            storageArray[index] = pixels[i][j];
            index++;
        }
    }
}

void unFlattenArray(int **pixels, int *storageArray, int rowSize, int colSize)
{
    int index = 0;

    for(int i = 0; i < rowSize; i++)
    {
        for(int j = 0; j < colSize; j++)
        {
            pixels[i][j] = storageArray[index];
            index++;
        }
    }
}
