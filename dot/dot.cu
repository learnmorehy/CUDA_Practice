#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <chrono>

using namespace std;

const int N = 1024;

__global__ void cudaDotKernel(const float* a, const float* b, float* c, const uint size){

	__shared__ float tmp[N];

	uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	tmp[thread_index] = a[thread_index]*b[thread_index];
	__syncthreads();

	if(threadIdx.x == 0){
		int sum = 0;
		for(int i = 0; i < N; i++) sum += tmp[i];
		*c = sum;
		
	}

	thread_index += blockDim.x * gridDim.x;
}

void cudaCallDotKernel(
	const uint block_count,
	const uint per_block_thread_count,
	const float* a,
	const float* b,
	float* c,
	const uint size
){
	cudaDotKernel<<<block_count, per_block_thread_count>>>(a, b, c, size);
}

int main(int argc, char** argv){
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	const uint per_block_thread_count = prop.maxThreadsPerBlock;
	const uint max_block_count = 1024;

	int* a = (int*)malloc(N*sizeof(int));
	int* b = (int*)malloc(N*sizeof(int));
	int c;

	for(int i = 0; i < N; i++){
		a[i] = (int)i;
		b[i] = (int)i;
		c[i] = (int)i;
	}

	int* dev_a;
	int* dev_b;
	int dev_c;

	cudaMalloc((void**) &dev_a, N*sizeof(int));
	cudaMalloc((void**) &dev_b, N*sizeof(int));
	cudaMalloc((void**) &dev_c, sizeof(int));

	cudaMemcpy(dev_a, a, int*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, int*sizeof(int), cudaMemcpyHostToDevice);

	uint block_count = min(max_block_count, (uint)ceil(N / (int)per_block_thread_count));

	auto start = std::chrono::high_resolution_clock::now();
	cudaCallDotKernel(
		block_count,
		per_block_thread_count,
		dev_a,
		dev_b,
		dev_c,
		N
	);
	auto end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

	free(a);
	free(b);
	free(c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
