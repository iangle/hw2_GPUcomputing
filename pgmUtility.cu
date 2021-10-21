#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pgmUtility.h"
#include "pgmProcess.h"
#include "timing.h"
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

    int *flatArray =(int*) malloc(sizeof(int)*numCols*numRows);

    flattenArray(pixels, flatArray, numRows, numCols);

    size_t bytes = numCols*numRows*sizeof(int);

    cudaMalloc(&d_a, bytes);

    cudaMemcpy(d_a, flatArray, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize, gridSize;

    // Number of threads in each thread block
    blockSize.x = 3;
    blockSize.y = 4;

    gridSize.x = ceil( (float) numRows / blockSize.x );
    gridSize.y = ceil( (float) numCols / blockSize.y );

    // Execute the kernel
    double then = currentTime();
    addCircle<<<gridSize, blockSize>>>(d_a, numRows, numCols, centerRow, centerCol, radius);
    double now = currentTime();
    double cost = now - then;
    printf("\nThe time cost for the GPU add cirlce is : %f\n", cost);    
        
    cudaMemcpy(flatArray, d_a, bytes, cudaMemcpyDeviceToHost);

    unFlattenArray(pixels, flatArray, numRows, numCols);

    cudaFree(d_a);

    free(flatArray);

    return 0;
}

void addCircleSequential(int **pixels, int numRows, int numCols, int centerRow,
                  int centerCol, int radius, char **header)
{
    //creating two arrays that will act as pairs
    int p3[2];
    int p4[2];

    for(int x = 0; x < numRows; x++)
    {
        for(int y = 0; y < numCols; y++)
        {

            //add the current x and y location to a pair
            p3[0] = x;
            p3[1] = y;

            //add the center of the circle to a pair
            p4[0] = centerCol;
            p4[1] = centerRow;    

            //compute the total distance to the point from the center
            float totalDistance = distanceSequential(p3,p4);
            
            //if we are inside the circle then set the location to 0 or black
            if(totalDistance <= radius)
            {
                pixels[x][y] = 0;
            }
                
        }

    }
}

//Call for the GPU based drawing of an edge over a PGM.
int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header )
{
        //Initialize Variable.
        int* d_a;
        
        //Flatten the array.
        int* flatArray = (int*) malloc(sizeof(int)*numCols*numRows);
        flattenArray(pixels, flatArray, numRows, numCols);
        
        size_t bytes = numCols*numRows*sizeof(int);
        
        //Cuda Memory Work.
        cudaMalloc(&d_a, bytes);
        
        cudaMemcpy(d_a, flatArray, bytes, cudaMemcpyHostToDevice);
        
        //Initializing the Grid.
        dim3 grid, block;
        block.x = 4;
        block.y = 4;
        grid.x = ceil( (float)numCols/block.x);
        grid.y = ceil( (float)numRows/block.y);
        
        //Execute Kernel.
        double then = currentTime();
        drawEdge<<<grid, block>>>(d_a, numRows, numCols, edgeWidth);
        double now = currentTime();
        double cost = now - then;
        printf("\nThe time cost for the GPU draw edge is : %f\n", cost); 
        //Return From Kernel.
        cudaMemcpy(flatArray, d_a, bytes, cudaMemcpyDeviceToHost);
        
        //Unflatten the pixels array.
        unFlattenArray(pixels, flatArray, numRows, numCols);
        
        //Free Memory.
        cudaFree(d_a);
        free(flatArray);
        
        return 0;
}

int pgmDrawEdgeSequential(int **pixels, int numRows, int numCols, int edgeWidth, char **header)
{
        int x;
        int y;
        
        int *flatArray =(int*) malloc(sizeof(int)*numCols*numRows);
        flattenArray(pixels, flatArray, numRows, numCols);

        for ( x = 0; x < numCols; x++) {
                for ( y = 0; y < numRows; y++ ) {
                        int idx = y*numCols + x;
                        if((x < numCols && y < numRows) && ((x > numCols - edgeWidth || x < edgeWidth) || (y > numRows - edgeWidth || y < edgeWidth)))
                                flatArray[idx] = 0;
                }
        }

        unFlattenArray(pixels, flatArray, numRows, numCols);
        free(flatArray);
        
        return 0;
}

//Call for the GPU based drawing of a line within a PGM.
int pgmDrawLine( int **pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col)
{
        //Initialize Variables.
        int* d_a;
        
        //Flatten the pixels array.
        int* flatArray = (int*) malloc(sizeof(int)*numCols*numRows);
        flattenArray(pixels, flatArray, numRows, numCols);
        
        size_t bytes = numCols*numRows*sizeof(int);
        
        //Cuda Memory Work.
        cudaMalloc(&d_a, bytes);
        
        cudaMemcpy(d_a, flatArray, bytes, cudaMemcpyHostToDevice);

        float rise = (float) p2row-p1row;

        float run = (float) p2col-p1col;
        
        //Calculate Slope
        float slope = rise / run;        
        
        //Initializing the Grid.
        dim3 grid, block;
        block.x = 4;
        block.y = 4;
        grid.x = ceil( (float)numCols/block.x);
        grid.y = ceil( (float)numRows/block.y);
        
        //Execute Kernel.
        double then = currentTime();
        drawLine<<<grid, block>>>(d_a, numRows, numCols, slope, p1row, p1col, p2row, p2col);
        double now = currentTime();
        double cost = now - then;
        printf("\nThe time cost for the GPU add line is : %f\n", cost); 
        //Return From Kernel.
        cudaMemcpy(flatArray, d_a, bytes, cudaMemcpyDeviceToHost);
        
        //Unflatten the pixels array.
        unFlattenArray(pixels, flatArray, numRows, numCols);
        
        //Free Memory.
        cudaFree(d_a);
        free(flatArray);
        
        return 0;
}

//Call for the CPU based drawing of a line within a PGM.
int pgmDrawLineSequential(int** pixels, int numRows, int numCols, int p1row, int p1col, int p2row, int p2col)
{
        //Declare Variables
        int x;
        int y;
        
        //Flattening the pixels array.
        int* flatArray = (int*) malloc(sizeof(int)*numCols*numRows);
        flattenArray(pixels, flatArray, numRows, numCols);
        
        //Calculating line variables.
        float slope = (p2row-p1row)/(p2col-p1col);
        float b = p1row - (slope*p1col);
        
        //PGM Scan/Modify Loop for Line Pixels.
        for (x = 0; x < numCols; x++) 
        {
                for (y = 0; y < numRows; y++ ) 
                {
                        int idx = y*numCols + x;
                        if(x < numCols && y < numRows && y - (slope*x) - b < .001 && !(y < p1col || x > p2col) && !(y - (slope*x) - (b-1) < .001))
                                flatArray[idx] = 0;
                }
        }

        //Unflatten the pixels array.
        unFlattenArray(pixels, flatArray, numRows, numCols);
        
        //Free memory.
        free(flatArray);
        
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

float distanceSequential( int p1[], int p2[] )
{
    return sqrt( pow( p1[0] - p2[0], 2 ) + pow( p1[1] - p2[1], 2 ) );
}
