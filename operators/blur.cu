#include "blur.h"
#include <vector>

template <typename T>
T* createBinomialKernel(int halfWidth)
{
	// Construct the binomial
	std::vector<T> hBinomial{ 1,2,1 };
	for (int cHalfWidth = 1; cHalfWidth < halfWidth; ++cHalfWidth) {
		for (int step = 0; step < 2; ++step) {
			hBinomial.insert(hBinomial.begin(), 0);
			for (int pos = 0; pos < hBinomial.size() - 1; ++pos) {
				hBinomial[pos] = hBinomial[pos] + hBinomial[pos + 1];
			}
		}
	}

	// Allocate GPU array and copy data
	int length = (halfWidth * 2 + 1) * sizeof(T);
	T* dBinomial = 0;
	cudaMalloc(&dBinomial, length);
	cudaMemcpy(dBinomial, hBinomial.data(), length, cudaMemcpyHostToDevice);

	return dBinomial;
}

template <typename T>
T* createBinomialKernelAccum(int halfWidth)
{
	// Construct the binomial
	std::vector<T> hBinomial{ 1,2,1 };
	for (int cHalfWidth = 1; cHalfWidth < halfWidth; ++cHalfWidth) {
		for (int step = 0; step < 2; ++step) {
			hBinomial.insert(hBinomial.begin(), 0);
			for (int pos = 0; pos < hBinomial.size() - 1; ++pos) {
				hBinomial[pos] = hBinomial[pos] + hBinomial[pos + 1];
			}
		}
	}

	// Sum elements
	for (int i = 1; i < hBinomial.size(); ++i)
	{
		hBinomial[i] = hBinomial[i] + hBinomial[i - 1];
	}

	// Allocate GPU array and copy data
	int length = (halfWidth * 2 + 1) * sizeof(T);
	T* dBinomial = 0;
	cudaMalloc(&dBinomial, length);
	cudaMemcpy(dBinomial, hBinomial.data(), length, cudaMemcpyHostToDevice);

	return dBinomial;
}

template <typename T>
__global__ void op::binomialX(T* vecOut, T* vecIn, T* filter, int halfWidth, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	T accum = 0;
	T weightSum = 0;
	for (int dx = -halfWidth; dx <= halfWidth; ++dx)
	{
		size_t cx = x + dx;
		size_t idx = y + cx * height;
		T weight = filter[dx + halfWidth];
		if (cx >= 0 && cx < width) {
			accum += weight * vecIn[idx];
			weightSum += weight;
		}
	}
	vecOut[y + x * height] = accum / weightSum;
}

template <typename T>
__global__ void op::binomialXAdjoint(T* vecOut, T* vecIn, T* filter, T* filterAccum, int halfWidth, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	T accum = 0;
	T maxWeight = filterAccum[2 * halfWidth];
	for (int dx = -halfWidth; dx <= halfWidth; ++dx)
	{
		size_t cx = x + dx;
		size_t idx = y + cx * height;
		T weight = filter[dx + halfWidth];
		if (cx >= 0 && cx < width) {
			T normalize = maxWeight;
			if (cx > width - halfWidth - 1) {
				normalize -= filterAccum[halfWidth - (width - cx - 1)  - 1];
			}
			if (cx < halfWidth) {
				normalize -= filterAccum[halfWidth - cx - 1];
			}
			accum += vecIn[idx] * weight / normalize;
		}
	}
	vecOut[y + x * height] = accum;
}

template <typename T>
__global__ void op::binomialY(T* vecOut, T* vecIn, T* filter, int halfWidth, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	T accum = 0;
	T weightSum = 0;
	for (int dy = -halfWidth; dy <= halfWidth; ++dy)
	{
		size_t cy = y + dy;
		size_t idx = cy + x * height;
		T weight = filter[dy + halfWidth];
		if (cy >= 0 && cy < height) {
			accum += weight * vecIn[idx];
			weightSum += weight;
		}
	}
	vecOut[y + x * height] = accum / weightSum;
}

template <typename T>
__global__ void op::binomialYAdjoint(T* vecOut, T* vecIn, T* filter, T* filterAccum, int halfWidth, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	T accum = 0;
	T maxWeight = filterAccum[2 * halfWidth];
	for (int dy = -halfWidth; dy <= halfWidth; ++dy)
	{
		size_t cy = y + dy;
		size_t idx = cy + x * height;
		T weight = filter[dy + halfWidth];
		if (cy >= 0 && cy < height) {
			T normalize = maxWeight;
			if (cy > height - halfWidth - 1) {
				normalize -= filterAccum[halfWidth - (height - cy - 1) - 1];
			}
			if (cy < halfWidth) {
				normalize -= filterAccum[halfWidth - cy - 1];
			}
			accum += vecIn[idx] * weight / normalize;
		}
	}
	vecOut[y + x * height] = accum;
}
template<typename T>
op::SeparableFilter<T>::SeparableFilter<T>(size_t width, size_t height)
	: LinearImage(width, height)
{
	cudaMalloc(&dIntermediate, width * height * sizeof(T));
}

template<typename T>
op::Binomial<T>::Binomial<T>(unsigned int halfWidth, size_t width, size_t height)
	: SeparableFilter<T>(width, height)
	, halfWidth(halfWidth)
{
	dFilter = createBinomialKernel<T>(halfWidth);
}

template<typename T>
void op::Binomial<T>::apply(const void* dVectorOut, const void* dVectorIn)
{
	op::binomialX KERNEL_ARGS2(blocks, threads) (dIntermediate, (T*) dVectorIn, dFilter, halfWidth, width, height);
	op::binomialY KERNEL_ARGS2(blocks, threads) ((T*) dVectorOut, dIntermediate, dFilter, halfWidth, width, height);
}

template<typename T>
op::BinomialAdjoint<T>::BinomialAdjoint<T>(unsigned int halfWidth, size_t width, size_t height)
	: SeparableFilter<T>(width, height)
	, halfWidth(halfWidth)
{
	dFilter = createBinomialKernel<T>(halfWidth);
	dAccum = createBinomialKernelAccum<T>(halfWidth);
}

template<typename T>
void op::BinomialAdjoint<T>::apply(const void* dVectorOut, const void* dVectorIn)
{
	op::binomialYAdjoint KERNEL_ARGS2(blocks, threads) (dIntermediate, (T*)dVectorIn, dFilter, dAccum, halfWidth, width, height);
	op::binomialXAdjoint KERNEL_ARGS2(blocks, threads) ((T*)dVectorOut, dIntermediate, dFilter, dAccum, halfWidth, width, height);
}

template __global__ void op::binomialX(float* vecOut, float* vecIn, float* filter, int halfWidth, size_t width, size_t height);
template __global__ void op::binomialXAdjoint(float* vecOut, float* vecIn, float* filter, float* filterAccum, int halfWidth, size_t width, size_t height);
template __global__ void op::binomialY(float* vecOut, float* vecIn, float* filter, int halfWidth, size_t width, size_t height);
template __global__ void op::binomialYAdjoint(float* vecOut, float* vecIn, float* filter, float* filterAccum, int halfWidth, size_t width, size_t height);
template class op::Binomial<float>;
template class op::BinomialAdjoint<float>;

template __global__ void op::binomialX(double* vecOut, double* vecIn, double* filter, int halfWidth, size_t width, size_t height);
template __global__ void op::binomialXAdjoint(double* vecOut, double* vecIn, double* filter, double* filterAccum, int halfWidth, size_t width, size_t height);
template __global__ void op::binomialY(double* vecOut, double* vecIn, double* filter, int halfWidth, size_t width, size_t height);
template __global__ void op::binomialYAdjoint(double* vecOut, double* vecIn, double* filter, double* filterAccum, int halfWidth, size_t width, size_t height);
template class op::Binomial<double>;
template class op::BinomialAdjoint<double>;