#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <chrono>

using namespace std;

__global__ void cudaAddVectorKernel(const float* a, const float* b, float* c, const uint size){

	uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	while(thread_index < size){
		c[thread_index] = a[thread_index] + b[thread_index];
		thread_index += blockDim.x * gridDim.x;
	}
}

void cudaCallAddVectorKernel(
	const uint block_count,
	const uint per_block_thread_count,
	const float* a,
	const float* b,
	float* c,
	const uint size
){
	cudaAddVectorKernel<<<block_count, per_block_thread_count>>>(a, b, c, size);
}

int main(int argc, char** argv){
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	// Maximum number of threads per block
    const int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    // Maximum number of blocks per dimension (Grid size)
    const int maxGridSizeX = prop.maxGridSize[0];
    const int maxGridSizeY = prop.maxGridSize[1];
    const int maxGridSizeZ = prop.maxGridSize[2];

	std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    std::cout << "Max grid size: (" << maxGridSizeX << ", " << maxGridSizeY << ", " << maxGridSizeZ << ")" << std::endl;

	const uint per_block_thread_count = 1024;
	const uint max_block_count = 1024;

	int length = 10000000;

	float* a = (float*)malloc(length*sizeof(float));
	float* b = (float*)malloc(length*sizeof(float));
	float* c = (float*)malloc(length*sizeof(float));

	for(int i = 0; i < length; i++){
		a[i] = (float)i;
		b[i] = (float)i;
		c[i] = (float)i;
	}

	float* dev_a;
	float* dev_b;
	float* dev_c;

	cudaMalloc((void**) &dev_a, length*sizeof(float));
	cudaMalloc((void**) &dev_b, length*sizeof(float));
	cudaMalloc((void**) &dev_c, length*sizeof(float));

	cudaMemcpy(dev_a, a, length*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, length*sizeof(float), cudaMemcpyHostToDevice);

	uint block_count = min(max_block_count, (uint)ceil(length / (float)per_block_thread_count));

	auto start = std::chrono::high_resolution_clock::now();

	cudaCallAddVectorKernel(
		block_count,
		maxThreadsPerBlock,
		dev_a,
		dev_b,
		dev_c,
		length
	);

	auto end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(c, dev_c, length * sizeof(float), cudaMemcpyDeviceToHost);
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
	free(a);
	free(b);
	free(c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
