#include "deblur.h"

#include "../util/eigs.h"
#include "../operators/derivative.h"
#include "../operators/blur.h"
#include "../operators/stack.h"
#include "kernels.h"

template<typename T>
TV::Deblur<T>::Deblur<T>(size_t width, size_t height)
	: TV::Base<T>(width, height)
{ 
	// Initialize Operators
	binomial = new op::Binomial<T>(50, width, height);
	binomialAdjoint = new op::BinomialAdjoint<T>(50, width, height);
	derivative = new op::Derivative<T>(width, height);
	derivativeAdjoint = new op::DerivativeAdjoint<T>(width, height);
	kTilde = new op::VStack<T>(binomial, derivative);
	kTildeAdjoint = new op::HStack<T>(binomialAdjoint, derivativeAdjoint);
	kTildeSymm = new op::Concat<T>(kTilde, kTildeAdjoint);


	// Initialize Arrays
	cudaMalloc(&df, width * height * sizeof(T));
	cudaMalloc(&du, width * height * sizeof(T));
	cudaMalloc(&du2, width * height * sizeof(T));
	cudaMalloc(&dubar, width * height * sizeof(T));
	cudaMalloc(&dq, width * height * sizeof(T));
	cudaMalloc(&dp, 2 * width * height * sizeof(T));
	cudaMalloc(&dTmp, width * height * sizeof(T));
	cudaMalloc(&dTmp2, 2 * width * height * sizeof(T));

	// Compute stepsize
	T ev = eigenvalue<float>(kTildeSymm);
	sigma = 0.8 / (ev);
	tau = sigma;
	alpha = 0.001f;

	// Kernel threads and blocks
	threads = dim3(32, 32, 1);
	blocks = dim3(iDivUp(width, 32), iDivUp(height, 32), 1);
	cublasCreate(&handle);
}

template<typename T>
TV::Deblur<T>::~Deblur()
{
	cudaFree(du);
	cudaFree(du2);
	cudaFree(dubar);
	cudaFree(dp);
	cudaFree(dq);
	cudaFree(df);
	cudaFree(dTmp);
	cudaFree(dTmp2);
	cublasDestroy(handle);
}

template<typename T>
void TV::Deblur<T>::apply(T* hImage)
{
	CUDA_SYNC_AND_ERRORCHECK();
	cudaMemcpy(df, hImage, width * height * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(du, df, width * height * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(du2, df, width * height * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dubar, df, width * height * sizeof(T), cudaMemcpyDeviceToDevice);

	// iterate
	int iterCount = 1000;
	for (int iter = 0; iter < iterCount; ++iter)
	{
		if (iter % 100 == 0) {
			computeEnergy();
		}
		// Update q
		binomial->apply(dTmp, dubar);
		deblurUpdateQ KERNEL_ARGS2(blocks, threads) (dq, dTmp, df, sigma, width, height);

		// Update p
		tvUpdateP KERNEL_ARGS2(blocks, threads) (dp, dubar, sigma, width, height);
		tvLimitP KERNEL_ARGS2(blocks, threads) (dp, alpha, width, height);

		// Update u
		derivativeAdjoint->apply(dTmp, dp);
		binomialAdjoint->apply(dTmp2, dq);
		deblurUpdateU KERNEL_ARGS2(blocks, threads)(du, du2, dTmp, dTmp2, tau, width, height);

		// Update u bar
		tvExterpolate KERNEL_ARGS2(blocks, threads)(du, du2, dubar, width, height);
	}
	cudaMemcpy(hImage, dubar, width * height * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void TV::Deblur<T>::computeEnergy()
{
	T hEnergy[2] = { 0 };

	T a = -1;
	binomial->apply(dTmp, du);
	cublasaxpy(handle, width * height, &a, df, 1, dTmp, 1);
	cublasDot(handle, width * height, dTmp, 1, dTmp, 1, &hEnergy[0]);

	derivative->apply(dp, du);
	norm KERNEL_ARGS2(blocks, threads) (dp, width, height);
	cublasasum(handle, width * height, dp, 1, &hEnergy[1]);

	T energy = hEnergy[0] + alpha * hEnergy[1];
	std::cout << "Energy: " << energy << "\t\t(" << hEnergy[0] << " - " << alpha * hEnergy[1] << ")\n";
}


template class TV::Deblur<float>;
template class TV::Deblur<double>;