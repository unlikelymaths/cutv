#pragma once
#include "../util/cudahelper.h"
#include "linear.h"

namespace op {

template <typename T>
__global__ void binomialX(T* vecOut, T* vecIn, T* filter, int halfWidth, size_t width, size_t height);

template <typename T>
__global__ void binomialXAdjoint(T* vecOut, T* vecIn, T* filter, T* filterAccum, int halfWidth, size_t width, size_t height);

template <typename T>
__global__ void binomialY(T* vecOut, T* vecIn, T* filter, int halfWidth, size_t width, size_t height);

template <typename T>
__global__ void binomialYAdjoint(T* vecOut, T* vecIn, T* filter, T* filterAccum, int halfWidth, size_t width, size_t height);

template <typename T>
class SeparableFilter : public op::LinearImage {
public:
	SeparableFilter<T>(size_t width, size_t height);
protected:
	T* dIntermediate;
	T* dFilter;
};

template <typename T>
class Binomial : public SeparableFilter<T> {
public:
	Binomial<T>(unsigned int halfWidth, size_t width, size_t height);
	void apply(const void* dVectorOut, const void* dVectorIn);
private:
	int halfWidth;
};

template <typename T>
class BinomialAdjoint : public SeparableFilter<T> {
public:
	BinomialAdjoint<T>(unsigned int halfWidth, size_t width, size_t height);
	void apply(const void* dVectorOut, const void* dVectorIn);
private:
	int halfWidth;
	T* dAccum;
};

};