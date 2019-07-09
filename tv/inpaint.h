#pragma once

#include "tv.h"
#include "../util/cudahelper.h"

namespace TV {

	template <typename T>
	class Inpaint : public TV::Base<T> {
	public:
		Inpaint<T>(size_t width, size_t height, bool* mask);
		~Inpaint();
		void apply(T* hImage);
		void computeEnergy();
	private:
		T sigma, tau, alpha;
		dim3 threads, blocks;
		T *du, *du2, *dubar, *df, *dp, *dTmp2;
		bool* dmask;
		cublasHandle_t handle;
	};

}