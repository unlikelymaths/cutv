#pragma once
#include "../util/cudahelper.h"

namespace op {

class Linear {
public:
	Linear(dim3 threads, dim3 blocks)
		: threads(threads)
		, blocks(blocks)
	{};
	virtual unsigned int inputLength() = 0;
	virtual unsigned int outputLength() = 0;
	virtual void apply(const void* dVectorOut, const void* dVectorIn) = 0;
protected:
	dim3 threads;
	dim3 blocks;
};


class LinearImage : public Linear {
public:
	LinearImage(size_t width, size_t height)
		: Linear(dim3(32, 32, 1), dim3(iDivUp(width, 32), iDivUp(height, 32), 1))
		, width(width)
		, height(height)
	{};
	virtual unsigned int inputLength() { return width * height; };
	virtual unsigned int outputLength() { return width * height; };
protected:
	const size_t width, height;
};

}
