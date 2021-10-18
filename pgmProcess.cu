

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
