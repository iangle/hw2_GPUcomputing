

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
