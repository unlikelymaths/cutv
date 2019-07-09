#include "denoise.h"

#include "../util/eigs.h"
#include "../operators/derivative.h"
#include "kernels.h"

template<typename T>
TV::Denoise<T>::Denoise<T>(size_t width, size_t height)
	: TV::Base<T>(width, height)
{
	// Initialize
	cudaMalloc(&df, width * height * sizeof(T));
	cudaMalloc(&du, width * height * sizeof(T));
	cudaMalloc(&du2, width * height * sizeof(T));
	cudaMalloc(&dubar, width * height * sizeof(T));
	cudaMalloc(&dp, 2 * width * height * sizeof(T));
	cudaMalloc(&dTmp, width * height * sizeof(T));
	cudaMalloc(&dTmp2, 2 * width * height * sizeof(T));

	// Compute stepsize
	op::DTD<T> op(width, height);
	T ev = eigenvalue<T>(&op);
	sigma = 0.8 / (ev);
	tau = sigma;
	alpha = 0.1f;

	// Kernel threads and blocks
	threads = dim3(32, 32, 1);
	blocks = dim3(iDivUp(width, 32), iDivUp(height, 32), 1);
	cublasCreate(&handle);
}

template<typename T>
TV::Denoise<T>::~Denoise()
{
	cudaFree(du);
	cudaFree(du2);
	cudaFree(dubar);
	cudaFree(dp);
	cudaFree(df);
	cudaFree(dTmp);
	cudaFree(dTmp2);
	cublasDestroy(handle);
}

template<typename T>
void TV::Denoise<T>::apply(T* hImage)
{
	// Copy input image
	cudaMemcpy(df, hImage, width * height * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(du, df, width * height * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(du2, df, width * height * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dubar, df, width * height * sizeof(T), cudaMemcpyDeviceToDevice);

	// Initial P
	tvInitP KERNEL_ARGS2(blocks, threads) (dp, dubar, width, height);
	tvLimitP KERNEL_ARGS2(blocks, threads) (dp, alpha, width, height);

	// iterate
	int iterCount = 1000;
	for (int iter = 0; iter < iterCount; ++iter)
	{
		if (iter % 100 == 0) {
			computeEnergy();
		}
		// Update p
		tvUpdateP KERNEL_ARGS2(blocks, threads) (dp, dubar, sigma, width, height);
		tvLimitP KERNEL_ARGS2(blocks, threads) (dp, alpha, width, height);

		// Update u
		tvUpdateU KERNEL_ARGS2(blocks, threads)(du, du2, dp, df, tau, width, height);

		// Update u bar
		tvExterpolate KERNEL_ARGS2(blocks, threads)(du, du2, dubar, width, height);
	}
	cudaMemcpy(hImage, dubar, width * height * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void TV::Denoise<T>::computeEnergy()
{
	T hEnergy[2] = { 0 };
	cudaMemcpy(dTmp, du, width * height * sizeof(T), cudaMemcpyDeviceToDevice);

	T a = -1;
	cublasaxpy(handle, width * height, &a, df, 1, dTmp, 1);
	cublasDot(handle, width * height, dTmp, 1, dTmp, 1, &hEnergy[0]);

	op::Derivative<T> derivative(width, height);
	derivative.apply(dTmp2, du);
	norm KERNEL_ARGS2(blocks, threads) (dTmp2, width, height);
	cublasasum(handle, width * height, dTmp2, 1, &hEnergy[1]);

	float energy = hEnergy[0] + alpha * hEnergy[1];
	std::cout << "Energy: " << energy << "\t\t(" << hEnergy[0] << " - " << alpha * hEnergy[1] << ")\n";
}


template class TV::Denoise<float>;
template class TV::Denoise<double>;