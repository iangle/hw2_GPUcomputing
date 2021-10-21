hw2: main.o timing.o pgmProcess.o pgmUtility.o
	nvcc -arch=sm_52 -o hw2 main.o timing.o pgmProcess.o pgmUtility.o -I.

main.o: main.cu
	nvcc -arch=sm_52 -c main.cu

timing.o: timing.c timing.h
	g++ -c -o timing.o timing.c -I.

pgmProcess.o: pgmProcess.h pgmProcess.cu
	nvcc -arch=sm_52 -c pgmProcess.cu

pgmUtility.o: pgmUtility.h pgmUtility.cu
	nvcc -arch=sm_52 -c pgmUtility.cu

clean:
	rm -r *.o balloons_* hw2
