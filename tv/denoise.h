#pragma once

#include "tv.h"
#include "../util/cudahelper.h"

namespace TV {

template <typename T>
class Denoise : public TV::Base<T> {
public:
	Denoise<T>(size_t width, size_t height);
	~Denoise();
	void apply(T* hImage);
	void computeEnergy();
private:
	T sigma, tau, alpha;
	dim3 threads, blocks;
	T *du, *du2, *dubar, *df, *dp, *dTmp, *dTmp2;
	cublasHandle_t handle;
};

}