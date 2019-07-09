#pragma once

#include "tv.h"
#include "../util/cudahelper.h"
#include "../operators/linear.h"

namespace TV {

	template <typename T>
	class Deblur : public TV::Base<T> {
	public:
		Deblur<T>(size_t width, size_t height);
		~Deblur();
		void apply(T* hImage);
		void computeEnergy();
	private:
		T sigma, tau, alpha;
		dim3 threads, blocks;
		T *du, *du2, *dubar, *df, *dp, *dq, *dTmp, *dTmp2;
		op::Linear* binomial;
		op::Linear* binomialAdjoint;
		op::Linear* derivative;
		op::Linear* derivativeAdjoint;
		op::Linear* kTilde;
		op::Linear* kTildeAdjoint;
		op::Linear* kTildeSymm;
		cublasHandle_t handle;
	};

}