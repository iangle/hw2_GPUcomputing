
#include <stdio.h>
#include <stdlib.h>
#include "pgmUtility.h"

int main (int argc, char *argv[])
{
    int *numCols, *numRows;
    int x,y = 1;

    numRows = &x;
    numCols = &y;

    char ** header = (char**) malloc(sizeof(char) *4);

    FILE *in;

    in = fopen(argv[1], "r");

    printf("hello\n");

    int **pixels = pgmRead1(header, numRows, numCols, in);

    pgmDrawCircle(pixels, 480, 640, 5, 5, 3, header);
}