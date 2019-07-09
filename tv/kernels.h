#pragma once

#include "../util/cudahelper.h"

template <typename T>
__global__ void tvInitP(T* dp, T* dubar, size_t width, size_t height);

template <typename T>
__global__ void tvUpdateP(T* dp, T* dubar, T sigma, size_t width, size_t height);

template <typename T>
__global__ void tvLimitP(T* dp, T alpha, size_t width, size_t height);

template <typename T>
__global__ void tvUpdateU(T* du, T* du2, T* dp, T* df, T tau, size_t width, size_t height);

template <typename T>
__global__ void tvExterpolate(T* du, T* du2, T* dubar, size_t width, size_t height);

template <typename T>
__global__ void norm(T* vec, size_t width, size_t height);

template <typename T>
__global__ void deblurUpdateQ(T* dq, T* dAubar, T* df, T sigma, size_t width, size_t height);

template <typename T>
__global__ void deblurUpdateU(T* du, T* du2, T* dKp, T* dAq, T tau, size_t width, size_t height);

template <typename T>
__global__ void inpaintingUpdateU(T* du, T* du2, T* dp, T* df, bool* dmask, T tau, size_t width, size_t height);
