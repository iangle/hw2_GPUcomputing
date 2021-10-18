hw2: pgmUtility.cu pgmProcess.cu timing.c main.cu
	nvcc -arch=sm_52 -o hw2 pgmUtility.cu pgmProcess.cu timing.c main.cu 
clean:
	rm *.o hw2 

