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

__global__ void addCircle(int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, int *p1, int *p2)
{
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*numCols + ix;

    int p3[2];
    int p4[2];

    p3[0] = ix / numCols;
    p3[1] = ix % numCols;

    p4[0] = centerRow;
    p4[1] = centerCol;    

    float totalDistance = distance(p3,p4);

    if(iy == 450)
        printf("totalDistance: %f", totalDistance);

    if(totalDistance <= radius)
    {
       pixels[idx] = 0;
    }

}
