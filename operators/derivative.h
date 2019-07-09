#pragma once
#include "../util/cudahelper.h"
#include "linear.h"

namespace op {

template <typename T>
__device__ T dx(T *image, size_t x, size_t y, size_t width, size_t height);

template <typename T>
__device__ T dxT(T* image, size_t x, size_t y, size_t width, size_t height);

template <typename T>
__device__ T dy(T* image, size_t x, size_t y, size_t width, size_t height);

template <typename T>
__device__ T dyT(T* image, size_t x, size_t y, size_t width, size_t height);

template <typename T>
__global__ void applyD(T* vecOut, T* vecIn, size_t width, size_t height);

template <typename T>
__global__ void applyDT(T* vecOut, T* vecIn, size_t width, size_t height);

template <typename T>
class Derivative : public op::LinearImage {
public:
	Derivative<T>(size_t width, size_t height);
	void apply(const void* dVectorOut, const void* dVectorIn);
	unsigned int outputLength() { return 2 * width * height; };
};

template <typename T>
class DerivativeAdjoint : public op::LinearImage {
public:
	DerivativeAdjoint<T>(size_t width, size_t height);
	void apply(const void* dVectorOut, const void* dVectorIn);
	unsigned int outputLength() { return 2 * width * height; };
};

template <typename T>
class DTD : public op::LinearImage {
public:
	DTD<T>(size_t width, size_t height);
	void apply(const void* dVectorOut, const void* dVectorIn);
private:
	Derivative<T> derivative;
	DerivativeAdjoint<T> derivativeAdjoint;
	T* dp;
};

};