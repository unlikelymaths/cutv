#include "derivative.h"


template <typename T>
__device__ T op::dx(T *image, size_t x, size_t y, size_t width, size_t height)
{
	if (x < width - 1) {
		return image[y + (x + 1) * height] - image[y + x * height];
	}
	else
	{
		return 0;
	}
}

template <typename T>
__device__ T op::dxT(T* image, size_t x, size_t y, size_t width, size_t height)
{
	T result = 0;
	if (x < width - 1) {
		result -= image[y + x * height];
	}
	if (x > 0) {
		result += image[y + (x - 1) * height];
	}
	return result;
}

template <typename T>
__device__ T op::dy(T* image, size_t x, size_t y, size_t width, size_t height)
{
	if (y < height - 1) {
		return image[y + 1 + x * height] - image[y + x * height];
	}
	else
	{
		return 0;
	}
}

template <typename T>
__device__ T op::dyT(T* image, size_t x, size_t y, size_t width, size_t height)
{
	T result = 0;
	if (y < height - 1) {
		result -= image[y + x * height];
	}
	if (y > 0) {
		result += image[y - 1 + x * height];
	}
	return result;
}

template <typename T>
__global__ void op::applyD(T* vecOut, T* vecIn, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	vecOut[idx] = dx<T>(vecIn, x, y, width, height);
	idx = idx + width * height;
	vecOut[idx] = dy<T>(vecIn, x, y, width, height);
}

template <typename T>
__global__ void op::applyDT(T* vecOut, T* vecIn, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	vecOut[idx] = dxT(vecIn, x, y, width, height) + dyT(vecIn + width * height, x, y, width, height);
}

template<typename T>
op::Derivative<T>::Derivative<T>(size_t width, size_t height)
	: LinearImage(width, height)
{}

template<typename T>
void op::Derivative<T>::apply(const void* dVectorOut, const void* dVectorIn)
{
	op::applyD<T> KERNEL_ARGS2(blocks, threads) ((T*)dVectorOut, (T*)dVectorIn, width, height);
}

template<typename T>
op::DerivativeAdjoint<T>::DerivativeAdjoint<T>(size_t width, size_t height)
	: LinearImage(width, height)
{}

template<typename T>
void op::DerivativeAdjoint<T>::apply(const void* dVectorOut, const void* dVectorIn)
{
	op::applyDT<T> KERNEL_ARGS2(blocks, threads) ((T*)dVectorOut, (T*)dVectorIn, width, height);
}

template<typename T>
op::DTD<T>::DTD<T>(size_t width, size_t height)
	: LinearImage(width, height)
	, derivative(width, height)
	, derivativeAdjoint(width, height)
{
	cudaMalloc(&dp, 2 * width * height * sizeof(T));
}

template<typename T>
void op::DTD<T>::apply(const void* dVectorOut, const void* dVectorIn)
{
	derivative.apply(dp, (T*)dVectorIn);
	derivativeAdjoint.apply((T*)dVectorOut, dp);
}


template __device__ float op::dx(float *image, size_t x, size_t y, size_t width, size_t height);
template __device__ float op::dxT(float* image, size_t x, size_t y, size_t width, size_t height);
template __device__ float op::dy(float* image, size_t x, size_t y, size_t width, size_t height);
template __device__ float op::dyT(float* image, size_t x, size_t y, size_t width, size_t height);
template __global__ void op::applyD(float* vecOut, float* vecIn, size_t width, size_t height);
template __global__ void op::applyDT(float* vecOut, float* vecIn, size_t width, size_t height);
template class op::Derivative<float>;
template class op::DerivativeAdjoint<float>;
template class op::DTD<float>;

template __device__ double op::dx(double *image, size_t x, size_t y, size_t width, size_t height);
template __device__ double op::dxT(double* image, size_t x, size_t y, size_t width, size_t height);
template __device__ double op::dy(double* image, size_t x, size_t y, size_t width, size_t height);
template __device__ double op::dyT(double* image, size_t x, size_t y, size_t width, size_t height);
template __global__ void op::applyD(double* vecOut, double* vecIn, size_t width, size_t height);
template __global__ void op::applyDT(double* vecOut, double* vecIn, size_t width, size_t height);
template class op::Derivative<double>;
template class op::DerivativeAdjoint<double>;
template class op::DTD<double>;