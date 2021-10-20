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


int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header )
{
        return 0;
}

int pgmDrawEdgeSequential(int **pixels, int numRows, int numCols, int edgeWidth)
{
        int *flatArray =(int*) malloc(sizeof(int)*numCols*numRows);
        flattenArray(pixels, flatArray, numRows, numCols);

        for ( x = 0; x < numRows; x++) {
                for ( y = 0; y < numCols; y++ ) {
                        int idx = y*numCols + x;
                        if(x < numCols && y < numRows && (x > numCols - edgeWidth || x < edgewidth) && (y > numRows - edgeWitdh || y < edgeWidth))
                                flatArray[idx] = 0;
                }
        }

        unFlattenArray(pixels, flatArray, numRows, numCols);
        free(flatArray);
}

int pgmDrawLine( int **pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col)
{
        return 0;
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
