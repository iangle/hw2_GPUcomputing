hw2: pgmUtility.cu pgmProcess.cu main.cu
	nvcc -arch=sm_52 -o hw2 pgmUtility.cu pgmProcess.cu main.cu 
clean:
	rm hw2 

