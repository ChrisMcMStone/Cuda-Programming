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
// We can double the

#define MAXSIZE 2048

__global__ void block_scan(const int *d_X, int *d_Y, int n, int *extractArray) {

	__shared__ int sharedSum[BLOCKSIZE];
	int i = threadIdx.x;
	int blockOffset = blockIdx.x * BLOCKSIZE;

	// Each thread copies an element from global into shared memory
	if(i + blockOffset < n) {
		sharedSum[i] = d_X[blockOffset + i];
	} else {
		//Fill the remaining elements in the last non-full block with zero
		sharedSum[i] =  0;
	}


	//Reduction phase
	for (uint stride = 1; stride < BLOCKSIZE; stride *= 2) {
		//Minimize divergence
		__syncthreads();
		// When stride = 1
		//Index used by thread 0 = 1
		//Index used by thread 1 = 3
		//Index used by thread 2 = 5
		//Working threads are now contiguous -> MINIMAL DIVERGENCE
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < BLOCKSIZE)
			sharedSum[index] += sharedSum[index - stride];
	}

	//Distribution phase
	for (uint stride = BLOCKSIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();

		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < BLOCKSIZE)
			sharedSum[index + stride] += sharedSum[index];
	}

	__syncthreads();

	// Copy results back to global memory
	if (i+blockOffset < n)
		d_Y[i+blockOffset] = sharedSum[i];

	//Extract sum
	if (extractArray && threadIdx.x == 0)
		extractArray[blockIdx.x] = sharedSum[BLOCKSIZE - 1];

}

__global__ void addBlocks(int *d_Sum1, int *d_Sum1_Scanned, int n) {


	int blockOffset = blockIdx.x * BLOCKSIZE;


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

int checkValid(int *hostWindow, int *deviceWindow, int size) {

	// -1 means invalid, 0 means valid
	for (int i = 0; i < size; ++i) {
		if (hostWindow[i] != deviceWindow[i])
			return -1;
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

	// Randomize contents of input vector values
	for (int i = 0; i < inputVectorLen; ++i) {
		h_input[i] = rand() % 10;
	}

	// Run winAverage on host for calculating speedup
	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	winAverage(h_input, h_avg, inputVectorLen, windowLen);
	gettimeofday(&tv2, NULL);
	printf("Total time on host = %f seconds\n",
			(double) (tv2.tv_usec - tv1.tv_usec) / 1000000
					+ (double) (tv2.tv_sec - tv1.tv_sec));

	// Allocate device memory

	int *d_X = NULL;
	int *d_Y = NULL;
	int *d_Sum1 = NULL;
	int *d_Sum1_Scanned = NULL;

	cudaMalloc((void **) &d_X, inputVectorSize);
	cudaCheckError();

	cudaMalloc((void **) &d_Y, inputVectorSize);
	cudaCheckError();

	//Calculate size of extract sum vector
	int sumSize = ceil(inputVectorLen / BLOCKSIZE);

	cudaMalloc((void **) &d_Sum1, sumSize);
	cudaCheckError();

	cudaMalloc((void **) &d_Sum1_Scanned, sumSize);
	cudaCheckError();

	// Copy contents of host memory into device memory

	cudaMemcpy(d_X, h_input, inputVectorSize, cudaMemcpyHostToDevice);
	cudaCheckError();

	int threadsPerBlock = 1024;
	int blocksPerGrid = 1 + ((inputVectorLen - 1) / threadsPerBlock);
	printf("Launched CUDA kernel with %d blocks of %d threads\n", blocksPerGrid,
			threadsPerBlock);

#ifdef TIMING_SUPPORT
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);             // create a timer
	sdkStartTimer(&timer);               // start the timer
#endif

	block_scan<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, inputVectorLen, d_Sum1);

	cudaDeviceSynchronize();

	add_blocks<<<blocksPerGrid, threadsPerBlock>>>(d_Sum1, d_Sum1_Scanned, sumSize);

	// Wait for device to finish
	//cudaDeviceSynchronize();

	// Check if error occurred
	cudaCheckError();

#ifdef TIMING_SUPPORT
	// stop and destroy timer
	sdkStopTimer(&timer);
	double dSeconds = sdkGetTimerValue(&timer) / (1000.0);
	//Log throughput, etc
	printf("Total time on device = %.5f seconds\n", dSeconds);
	sdkDeleteTimer(&timer);
#endif

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
//	cudaMemcpy(h_avgVerify, d_Y, inputVectorSize, cudaMemcpyDeviceToHost);
//	cudaCheckError();
//
//	if (checkValid(h_avg, h_avgVerify, inputVectorLen) == 0) {
//		puts("Verification successful");
//	} else {
//		puts("Verification FAILED");
//	}

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
	free(h_avg);


	printf("Done\n");
	return 0;
}
