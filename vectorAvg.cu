// A is input vector, B is output.

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// Used for timing host version
#include <time.h>

#define TIMING_SUPPORT

#define cudaCheckError() {                                                      \
 cudaError_t e=cudaGetLastError();                                              \
 if(e!=cudaSuccess) {                                                           \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
   exit(0);                                                                     \
 }                                                                              \
}

#ifdef TIMING_SUPPORT
#include <helper_cuda.h>
#include <helper_functions.h>
#endif
#define BLOCKSIZE 1024
// Only half number of threads are used in Scan when vector size = block size
// Therefore we can double to size of the vector used in the block scan

__global__ void block_scan(const int *d_X, int *d_Y, int n, int *extractArray) {

	__shared__ int sharedSum[BLOCKSIZE*2];
	int i = threadIdx.x;
	int blockOffset = blockIdx.x * BLOCKSIZE * 2;

	// Each thread copies an element from global into shared memory
	//Copy first block's worth of elements into shared mem
	if(i + blockOffset < n) {
		sharedSum[i] = d_X[blockOffset + i];
	} else {
		//Fill the remaining elements in the last non-full block with zero
		sharedSum[i] =  0;
	}
	//Copy first block's worth of elements into shared mem
	if(blockOffset + BLOCKSIZE + i < n) {
		sharedSum[i + BLOCKSIZE] = d_X[blockOffset + BLOCKSIZE + i];
	} else {
		sharedSum[i + BLOCKSIZE] = 0;
	}

	//Reduction phase
	for (uint stride = 1; stride <= BLOCKSIZE; stride *= 2) {
		//Minimize divergence
		__syncthreads();
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < BLOCKSIZE*2)
			sharedSum[index] += sharedSum[index - stride];
	}

	//Distribution phase
	for (uint stride = BLOCKSIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < BLOCKSIZE*2)
			sharedSum[index + stride] += sharedSum[index];
	}

	__syncthreads();

	// Copy results back to global memory
	if (i+blockOffset < n)
		d_Y[i+blockOffset] = sharedSum[i];
	if(i + blockOffset + BLOCKSIZE < n) {
		d_Y[i+blockOffset+BLOCKSIZE] = sharedSum[i+BLOCKSIZE];
	}

	//Extract sum
	if (extractArray && threadIdx.x == 0)
		extractArray[blockIdx.x] = sharedSum[2 * BLOCKSIZE - 1];

}

__global__ void add_blocks(int *d_Sum1_Scanned, int *d_Y, int sumSize, int n) {

	int blockOffset = blockIdx.x * blockDim.x + (BLOCKSIZE * 2);
	int i = threadIdx.x;

	if(i+blockOffset < n)
		d_Y[blockOffset + i] += d_Sum1_Scanned[blockIdx.x];

}

__global__ void average_scanned_block(int *d_Y, int n, int windowsize) {

//	__shared__ int read[BLOCKSIZE];
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//	read[i] = d_Y[i - windowSize]
}

void winAverage(const int *InputVector, int *Window, int size, int n) {

	for (int i = 0; i < size; ++i) {
		int temp = 0;
		uint x = 0;
		for (int j = i - n + 1; j <= i; ++j) {
			if (j >= 0) {
				temp += InputVector[j];
				x++;
			}
			if (x != 0)
				Window[i] = temp / (int) x;
		}
	}
}

void host_scan(const int *InputVector, int *OutputVector, int n) {

	OutputVector[0] = InputVector[0];
	for (int i = 1; i < n; ++i) {
		OutputVector[i] = OutputVector[i-1] + InputVector[i];
	}
}

int checkValid(int *hostWindow, int *deviceWindow, int size) {

	// -1 means invalid, 0 means valid
	for (int i = 0; i < size; ++i) {
		if (hostWindow[i] != deviceWindow[i]) {
			return -1;
		}
	}

	return 0;
}

__global__ void winAverageDeviceNaive(const int *InputVector, int *Window,
		int size, int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	uint x = 0;
	if (i < size) {
		int temp = 0;
		for (int j = i - n + 1; j <= i; ++j) {
			if (j >= 0) {
				temp += InputVector[j];
				x++;
			}
			if (x != 0)
				Window[i] = temp / (int) x;
		}
	}
}

void printVector(int *vector, int len) {
	for (int i = 0; i < len; ++i) {
        if(i%100 == 0)
            printf("\n");
		printf("%d, ", vector[i]);
	}
    printf("\n");
}

/**
 * Host main routine
 */
int main(void) {

	// Print the vector length to be used, and compute its size
	const int inputVectorLen = 1000000;  // This is 'size'
	int windowLen = 30; // This is 'n'

	printf("Input vector size %d\nWindow size = %d\n", inputVectorLen,
			windowLen);

	size_t inputVectorSize = inputVectorLen * sizeof(int);

	// Allocate the host vectors
	int *h_input = (int *) malloc(inputVectorSize);
	int *h_avg = (int *) malloc(inputVectorSize);
	int *h_avgVerify = (int *) malloc(inputVectorSize);

	// Verify that allocations succeeded
	if (h_input == NULL || h_avg == NULL || h_avgVerify == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	srand(time(NULL));
	// Randomize contents of input vector valuesd_Sum1
	for (int i = 0; i < inputVectorLen; ++i) {
		h_input[i] = rand() % 10;
	}

#ifdef TIMING_SUPPORT
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);             // create a timer
	sdkStartTimer(&timer);               // start the timer
#endif
	// Run winAverage on host for calculating speedup
	host_scan(h_input, h_avg, inputVectorLen);

#ifdef TIMING_SUPPORT
	// stop and destroy timer
	sdkStopTimer(&timer);
	double dSeconds = sdkGetTimerValue(&timer) / (1000.0);
	//Log throughput, etc
	printf("Total time on host = %.5f seconds\n", dSeconds);
	sdkDeleteTimer(&timer);
#endif

//	puts("Host scanned vector:");
//	printVector(h_avg, inputVectorLen);

	// Allocate device memory

	int *d_X = NULL;
	int *d_Y = NULL;
	int *d_Sum1 = NULL;
	int *d_Sum1_Scanned = NULL;
	int *d_Sum2 = NULL;
	int *d_Sum2_Scanned = NULL;

	cudaMalloc((void **) &d_X, inputVectorSize);
	cudaCheckError();

	cudaMalloc((void **) &d_Y, inputVectorSize);
	int *h_Y = (int *) malloc(inputVectorSize);
	cudaCheckError();

	//Calculate size of extract sum vector
	int sumLen = 1 + ((inputVectorLen-1) / (BLOCKSIZE*2));
	int sumSize = sumLen * sizeof(int);

	cudaMalloc((void **) &d_Sum1, sumSize);
	int *h_Sum1 = (int *) malloc(sumSize);
	cudaCheckError();

	cudaMalloc((void **) &d_Sum1_Scanned, sumSize);
	int *h_Sum1_Scanned = (int *) malloc(sumSize);
	cudaCheckError();

	// Copy contents of host memory into device memory
	cudaMemcpy(d_X, h_input, inputVectorSize, cudaMemcpyHostToDevice);
	cudaCheckError();

	int threadsPerBlock = 2048;
	int blocksPerGrid = 1 + ((inputVectorLen - 1) / threadsPerBlock);
	int blocksPerGrid2 = 1 + ((sumLen - 1) / threadsPerBlock);
	printf("Launched CUDA kernel with %d blocks of %d threads\n", blocksPerGrid,
			threadsPerBlock);

#ifdef TIMING_SUPPORT
	timer = NULL;
	sdkCreateTimer(&timer);             // create a timer
	sdkStartTimer(&timer);               // start the timer
#endif
	//Perform initial block scan
	block_scan<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, inputVectorLen, d_Sum1);
    cudaDeviceSynchronize();
	cudaCheckError();
	cudaMemcpy(h_Sum1, d_Sum1, sumSize, cudaMemcpyDeviceToHost);
	cudaCheckError();
	puts("\nd_Sum1 after first extract sum:");
	printVector(h_Sum1, sumLen);

	//Scan the extracted sum
	block_scan<<<blocksPerGrid2, threadsPerBlock>>>(d_Sum1, d_Sum1_Scanned, sumLen, NULL);
    cudaDeviceSynchronize();
	cudaCheckError();
	cudaMemcpy(h_Sum1_Scanned, d_Sum1_Scanned, sumSize, cudaMemcpyDeviceToHost);
	cudaCheckError();
	puts("\n\nd_Sum1_scanned after first block scan:");
	printVector(h_Sum1_Scanned, sumLen);

	//Add the scanned extract sum to the elements of the output vector
	add_blocks<<<blocksPerGrid, threadsPerBlock>>>(d_Sum1_Scanned, d_Y, sumLen, inputVectorLen);
	cudaCheckError();
    cudaDeviceSynchronize();
	cudaMemcpy(h_Y, d_Y, inputVectorSize, cudaMemcpyDeviceToHost);
	cudaCheckError();
	puts("\n\nd_Y after second block scan:");
	printVector(h_Y, inputVectorLen);

#ifdef TIMING_SUPPORT
	// stop and destroy timer
	sdkStopTimer(&timer);
	dSeconds = sdkGetTimerValue(&timer) / (1000.0);
	//Log throughput, etc
	printf("\n\nTotal time on device = %.5f seconds\n", dSeconds);
	sdkDeleteTimer(&timer);
#endif

	// Check if error occurred
	cudaCheckError();

//	// Copy the device result vector in device memory to the host result vector
//	// in host memory.
	cudaMemcpy(h_avgVerify, d_Y, inputVectorSize, cudaMemcpyDeviceToHost);
	cudaCheckError();

	if (checkValid(h_avg, h_avgVerify, inputVectorLen) == 0) {
		puts("Verification successful");
	} else {
		puts("Verification FAILED");
	}

	// Deallocate device memory
	cudaFree(d_X);
	cudaCheckError();

	cudaFree(d_Y);
	cudaCheckError();

	cudaFree(d_Sum1);
	cudaCheckError();

	cudaFree(d_Sum1_Scanned);
	cudaCheckError();

	// Reset the device and exit
	cudaDeviceReset();
	cudaCheckError();

	// Free host memory
	free(h_input);
	free(h_avg);
	free(h_avgVerify);


	printf("Done\n");
	return 0;
}
