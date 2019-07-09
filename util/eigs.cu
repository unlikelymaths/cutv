#include "eigs.h"
#include "CudaHelper.h"
#include <stdlib.h>

template <typename T>
__global__ void normalize(T* vecOut, T* vecIn, T* value, size_t length)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= length)
	{
		return;
	}
	vecOut[idx] = vecIn[idx] / value[0];
}

template<typename T>
T eigenvalue(op::Linear* op)
{
	size_t length = op->outputLength();

	// Allocate
	T* v0 = 0;
	cudaMalloc(&v0, length * sizeof(T));
	T* vk = 0;
	cudaMalloc(&vk, length * sizeof(T));
	T* vTilde = 0;
	cudaMalloc(&vTilde, length * sizeof(T));
	T* dDotProducts = 0;
	cudaMalloc(&dDotProducts, 2 * sizeof(T));
	T hDotProduct1[2] = { 0 };

	// Initialize
	size_t threads = 512;
	size_t blocks = iDivUp(length, threads);
	T* random = new T[length];
	for (int i = 0; i < length; ++i) {
		random[i] = 2.f * rand() / RAND_MAX - 1.f;
	}
	cudaMemcpy(v0, random, length * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(vk, v0, length * sizeof(T), cudaMemcpyDeviceToDevice);
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Power iteration
	int numIter = 1000;
	for (int iter = 0; iter < numIter; ++iter)
	{
		op->apply((const void*)vTilde, (const void*)vk);


		if (iter % 100 == 0) {
			// Compute EV
			cublasDot(handle, length, v0, 1, vTilde, 1, dDotProducts);
			cublasDot(handle, length, v0, 1, vk, 1, dDotProducts + 1);
			cudaMemcpy(&hDotProduct1, dDotProducts, 2 * sizeof(T), cudaMemcpyDeviceToHost);
			T ev = hDotProduct1[0] / hDotProduct1[1];
			std::cout << iter << ": " << ev << "\n";
		}

		if (iter < numIter - 1) {
			cublasDot(handle, length, v0, 1, vTilde, 1, dDotProducts);
			normalize KERNEL_ARGS2(blocks, threads) (vk, vTilde, dDotProducts, length);
		}
	}

	// Compute EV
	cublasDot(handle, length, v0, 1, vTilde, 1, dDotProducts);
	cublasDot(handle, length, v0, 1, vk, 1, dDotProducts + 1);
	cudaMemcpy(&hDotProduct1, dDotProducts, 2 * sizeof(T), cudaMemcpyDeviceToHost);
	T ev = hDotProduct1[0] / hDotProduct1[1];

	// Cleanup
	cublasDestroy(handle);
	cudaFree(v0);
	cudaFree(vk);
	cudaFree(vTilde);
	cudaFree(dDotProducts);

	return ev;
}

template float eigenvalue<float>(op::Linear* op);
template double eigenvalue<double>(op::Linear* op);