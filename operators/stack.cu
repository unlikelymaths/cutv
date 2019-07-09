#include "stack.h"

template <typename T>
__global__ void add(T* vecOut, T* vecIn, size_t length)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= length)
	{
		return;
	}
	vecOut[idx] = vecOut[idx] + vecIn[idx];
}

template<typename T>
op::HStack<T>::HStack<T>(op::Linear* left, op::Linear* right)
	: Linear(dim3(512),dim3(left->outputLength()))
	, left(left)
	, right(right)
{
	cudaMalloc(&dTmp, right->outputLength() * sizeof(T));
}

template<typename T>
void op::HStack<T>::apply(const void* dVectorOut, const void* dVectorIn)
{
	right->apply(dTmp, (T*)dVectorIn);
	left->apply((T*)dVectorOut, (T*)dVectorIn);
	add KERNEL_ARGS2(blocks, threads) ((T*)dVectorOut, dTmp, left->outputLength());
}

template<typename T>
op::VStack<T>::VStack<T>(op::Linear* top, op::Linear* bottom)
	: Linear(dim3(), dim3())
	, top(top)
	, bottom(bottom)
{}

template<typename T>
void op::VStack<T>::apply(const void* dVectorOut, const void* dVectorIn)
{
	top->apply((T*)dVectorOut, (T*)dVectorIn);
	bottom->apply(((T*)dVectorOut) + top->outputLength(), (T*)dVectorIn);
}

template<typename T>
op::Concat<T>::Concat<T>(op::Linear* first, op::Linear* second)
	: Linear(dim3(), dim3())
	, first(first)
	, second(second)
{
	cudaMalloc(&dTmp, first->outputLength() * sizeof(T));
}

template<typename T>
void op::Concat<T>::apply(const void* dVectorOut, const void* dVectorIn)
{
	first->apply(dTmp, (T*)dVectorIn);
	second->apply(((T*)dVectorOut), dTmp);
}


template class op::HStack<float>;
template class op::VStack<float>;
template class op::Concat<float>;

template class op::HStack<double>;
template class op::VStack<double>;
template class op::Concat<double>;