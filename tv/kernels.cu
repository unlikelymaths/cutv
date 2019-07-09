#include "kernels.h"

#include "../operators/derivative.h"

template <typename T>
__global__ void tvInitP(T* dp, T* dubar, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	dp[idx] = op::dx(dubar, x, y, width, height);
	idx = idx + width * height;
	dp[idx] = op::dy(dubar, x, y, width, height);
}

template __global__ void tvInitP(float* dp, float* dubar, size_t width, size_t height);
template __global__ void tvInitP(double* dp, double* dubar, size_t width, size_t height);

template <typename T>
__global__ void tvUpdateP(T* dp, T* dubar, T sigma, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	dp[idx] = dp[idx] + sigma * op::dx(dubar, x, y, width, height);
	idx = idx + width * height;
	dp[idx] = dp[idx] + sigma * op::dy(dubar, x, y, width, height);
}


template __global__ void tvUpdateP(float* dp, float* dubar, float sigma, size_t width, size_t height);
template __global__ void tvUpdateP(double* dp, double* dubar, double sigma, size_t width, size_t height);

template <typename T>
__global__ void tvLimitP(T* dp, T alpha, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx1 = y + x * height;
	size_t idx2 = idx1 + width * height;
	T v1 = dp[idx1];
	T v2 = dp[idx2];
	T norm = sqrt(pow(v1, (T)2) + pow(v2, (T)2));
	if (norm > alpha) {
		dp[idx1] = v1 / norm * alpha;
		dp[idx2] = v2 / norm * alpha;
	}
}

template __global__ void tvLimitP(float* dp, float alpha, size_t width, size_t height);
template __global__ void tvLimitP(double* dp, double alpha, size_t width, size_t height);

template <typename T>
__global__ void tvUpdateU(T* du, T* du2, T* dp, T* df, T tau, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	du2[idx] = (du[idx]
		- tau * (op::dxT(dp, x, y, width, height) + op::dyT(dp + width * height, x, y, width, height))
		+ tau * df[idx]
		) / (1 + tau);
}

template __global__ void tvUpdateU(float* du, float* du2, float* dp, float* df, float tau, size_t width, size_t height);
template __global__ void tvUpdateU(double* du, double* du2, double* dp, double* df, double tau, size_t width, size_t height);

template <typename T>
__global__ void tvExterpolate(T* du, T* du2, T* dubar, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	dubar[idx] = 2 * du2[idx] - du[idx];
	du[idx] = du2[idx];
}

template __global__ void tvExterpolate(float* du, float* du2, float* dubar, size_t width, size_t height);
template __global__ void tvExterpolate(double* du, double* du2, double* dubar, size_t width, size_t height);

template <typename T>
__global__ void norm(T* vec, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	vec[idx] = sqrt(pow(vec[idx], (T)2) + pow(vec[idx + width * height], (T)2));
}

template __global__ void norm(float* vec, size_t width, size_t height);
template __global__ void norm(double* vec, size_t width, size_t height);

template <typename T>
__global__ void deblurUpdateQ(T* dq, T* dAubar, T* df, T sigma, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	dq[idx] = (dq[idx] + sigma * dAubar[idx] - sigma * df[idx]) / (1 + sigma);
}

template __global__ void deblurUpdateQ(float* dq, float* dAubar, float* df, float sigma, size_t width, size_t height);
template __global__ void deblurUpdateQ(double* dq, double* dAubar, double* df, double sigma, size_t width, size_t height);

template <typename T>
__global__ void deblurUpdateU(T* du, T* du2, T* dKp, T* dAq, T tau, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	du2[idx] = du[idx] - tau * dKp[idx] - tau * dAq[idx];
}

template __global__ void deblurUpdateU(float* du, float* du2, float* dKp, float* dAq, float tau, size_t width, size_t height);
template __global__ void deblurUpdateU(double* du, double* du2, double* dKp, double* dAq, double tau, size_t width, size_t height);

template <typename T>
__global__ void inpaintingUpdateU(T* du, T* du2, T* dp, T* df, bool* dmask, T tau, size_t width, size_t height)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	size_t idx = y + x * height;
	T value = 0.f;
	if (dmask[idx])
	{
		value = df[idx];
	}
	else
	{
		value = du[idx] - tau * (op::dxT(dp, x, y, width, height) + op::dyT(dp + width * height, x, y, width, height));
	}
	du2[idx] = value;
}

template __global__ void inpaintingUpdateU(float* du, float* du2, float* dp, float* df, bool* dmask, float tau, size_t width, size_t height);
template __global__ void inpaintingUpdateU(double* du, double* du2, double* dp, double* df, bool* dmask, double tau, size_t width, size_t height);
