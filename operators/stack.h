#pragma once

#include "linear.h"

namespace op {

template <typename T>
class HStack : public op::Linear {
public:
	HStack<T>(op::Linear* left, op::Linear* right);
	void apply(const void* dVectorOut, const void* dVectorIn);
	unsigned int inputLength() { return left->inputLength() + right->inputLength(); };
	unsigned int outputLength() { return left->outputLength(); };
private:
	T* dTmp;
	op::Linear* left;
	op::Linear* right;
};

template <typename T>
class VStack : public op::Linear {
public:
	VStack<T>(op::Linear* top, op::Linear* bottom);
	void apply(const void* dVectorOut, const void* dVectorIn);
	unsigned int inputLength() { return top->inputLength(); };
	unsigned int outputLength() { return top->outputLength() + bottom->outputLength(); };
private:
	op::Linear* top;
	op::Linear* bottom;
};

template <typename T>
class Concat : public op::Linear {
public:
	Concat<T>(op::Linear* first, op::Linear* second);
	void apply(const void* dVectorOut, const void* dVectorIn);
	unsigned int inputLength() { return first->inputLength(); };
	unsigned int outputLength() { return second->outputLength(); };
private:
	T* dTmp;
	op::Linear* first;
	op::Linear* second;
};

}