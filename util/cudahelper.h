#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <cublas_v2.h>

#include "stdint.h"
#include <iostream>

/*************************************
DEFINITIONS:

CUDA_ERRORCHECK()
	Use after kernel calls that don't use UMem

CUDA_SYNC_AND_ERRORCHECK()
	Use after kernel calls that use UMem

CUDA_SAFE_CALL(call)
	Wrap cuda calls

CUBLAS_SAFE_CALL(call)
	Wrap cublas calls
***************************************/

#define CUDA_ERRORCHECK(); \
{ \
	cudaError err = cudaGetLastError(); \
	if (err != cudaSuccess) \
	{ \
		printf("######################################\n" \
		"CUDA ERROR %i in '%s"\
		"':\nIn File:\n\t %s\nLine %d:\n\t%s\n" \
		"Do you want to continue execution? (y/n)\n", \
		err, __FUNCTION__, __FILE__, __LINE__, \
		cudaGetErrorString(err)); \
		char i; std::cin >> i; \
		if (i != 'y') { exit(-1); } \
	} \
}

#define CUDA_SYNC_AND_ERRORCHECK(); \
{ \
	cudaDeviceSynchronize(); \
	cudaError err = cudaGetLastError(); \
	if (err != cudaSuccess) \
	{ \
		printf("######################################\n" \
		"CUDA ERROR %i in '%s"\
		"':\nIn File:\n\t %s\nLine %d:\n\t%s\n" \
		"Do you want to continue execution? (y/n)\n", \
		err, __FUNCTION__, __FILE__, __LINE__, \
		cudaGetErrorString(err)); \
		char i; std::cin >> i; \
		if (i != 'y') { exit(-1); } \
	} \
}

#define CUDA_SAFE_CALL(call)\
{ \
	cudaError err = call; \
	if (err != cudaSuccess) \
	{ \
		printf("######################################\n" \
		"CUDA ERROR %i in '%s"\
		"':\nIn File:\n\t %s\nLine %d:\n\t%s\n" \
		"Do you want to continue execution? (y/n)\n", \
		err, __FUNCTION__, __FILE__, __LINE__, \
		cudaGetErrorString(err)); \
		char i; std::cin >> i; \
		if (i != 'y') { exit(-1); } \
	} \
}

#define CUBLAS_SAFE_CALL(call)\
{ \
	cublasStatus_t status = call; \
if (status != CUBLAS_STATUS_SUCCESS) \
	{ \
	printf("######################################\n" \
	"CUBLAS ERROR %i in '%s"\
	"':\nIn File:\n\t %s\nLine:\n\t %d\n" \
	"Do you want to continue execution? (y/n)\n", \
	status, __FUNCTION__, __FILE__, __LINE__); \
	char i; std::cin >> i; \
if (i != 'y') { exit(-1); } \
	} \
}

/*************************************
DEFINES:

CUDART_PI_F
CUDA_CALLABLE
***************************************/

#define CUDART_PI_F 3.141592654f 

#ifdef __CUDA_ARCH__ // DEVICE Code
#define CUDA_CALLABLE __host__ __device__
#define CUDA_ONLY_CALLABLE __device__
#else // Host Code
#define CUDA_CALLABLE
#define CUDA_ONLY_CALLABLE 
#endif


/*************************************
DEFINES TO MAKE INTELLISENSE STOP WHINING ABOUT CUDA
***************************************/

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

/*************************************
FUNCTIONS:

iDivUp(...)
***************************************/

inline CUDA_CALLABLE
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline CUDA_CALLABLE
size_t iDivUpSize_t(size_t a, size_t b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*************************************
CUBLAS
***************************************/

inline void cublasDot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
	cublasSdot(handle, n, x, incx, y, incy, result);
}

inline void cublasDot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
	cublasDdot(handle, n, x, incx, y, incy, result);
}


inline void cublasaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float* y, int incy) {
	cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

inline void cublasaxpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double* y, int incy) {
	cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

inline void cublasasum(cublasHandle_t handle, int n, const float *x, int incx, float* result) {
	cublasSasum(handle, n, x, incx, result);
}

inline void cublasasum(cublasHandle_t handle, int n, const double *x, int incx, double* result) {
	cublasDasum(handle, n, x, incx, result);
}