#include "pgmProcess.h"
#include <stdio.h>

/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] )
{
    float xDistance = p1[0] - p2[0];

    xDistance = pow(xDistance, 2);

    float yDistance = p1[1] - p2[1];

    yDistance = pow(yDistance, 2);

    return sqrt(xDistance + yDistance);

}

__global__ void addCircle(int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius)
{
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*numCols + ix;

    //creating two arrays that will act as pairs
    int p3[2];
    int p4[2];

    //add the current x and y location to a pair
    p3[0] = ix;
    p3[1] = iy;

    //add the center of the circle to a pair
    p4[0] = centerCol;
    p4[1] = centerRow;    

    //compute the total distance to the point from the center
    float totalDistance = distance(p3,p4);

    //if we are inside the circle then set the location to 0 or black
    if(totalDistance <= radius)
    {
       pixels[idx] = 0;
    }

}

//Draws an edge around the provided PGM.
__global__ void drawEdge (int* pixels, int numRows, int numCols, int edgeWidth)
{
  //Standard CUDA Variables
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = iy*numCols + ix;
  
  if(ix < numCols && iy < numRows && (ix > numCols - edgeWidth || ix < edgeWidth) || (iy > numRows - edgeWidth || iy < edgeWidth))
    pixels[idx] = 0;
}

//Draws a line between two points within the provided PGM.
__global__ void drawLine (int* pixels, int numRows, int numCols, float slope, int p1row, int p1col, int p2row, int p2col)
{
  //Standard CUDA Variables.
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = iy*numCols + ix;
    
  float b = p1row - (slope*p1col);
  
  if(ix < numCols && iy < numRows && iy - (slope*ix) - b < .001 && !(iy < p1col || ix > p2col) && !(iy - (slope*ix) - (b-1) < .001))
     pixels[idx] = 0;
}
