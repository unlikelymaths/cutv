#pragma once

#include "image.h"
#include "cudahelper.h"
#include "../operators/blur.h"

void blurImage(const char* imageInPath, const char* imageOutPath, int halfWidth)
{
	Image image(imageInPath);
	float* din, *dout;
	cudaMalloc(&din, image.width * image.height * sizeof(float));
	cudaMalloc(&dout, image.width * image.height * sizeof(float));
	int channelStride = image.width * image.height;
	op::Binomial<float> binomial(halfWidth, image.width, image.height);
	for (int channel = 0; channel < 3; channel++) {
		cudaMemcpy(din, image.data + channel * channelStride, image.width * image.height * sizeof(float), cudaMemcpyHostToDevice);
		binomial.apply(dout, din);
		cudaMemcpy(image.data + channel * channelStride, dout, image.width * image.height * sizeof(float), cudaMemcpyDeviceToHost);
	}
	image.saveAs(imageOutPath);
	cudaFree(din);
	cudaFree(dout);
}

void destructImage(const char* imageInPath, const char* imageOutPath, const char* maskOutPath, float deleteProbability)
{
	Image image(imageInPath);
	size_t stride = image.width * image.height;
	float* mask = new float[stride * 3];
	float replaceValue = 0.f;
	for (int i = 0; i < stride; ++i) {
		mask[i] = (rand() < deleteProbability * RAND_MAX) ? 0.f : 1.f;
		mask[i + stride] = mask[i];
		mask[i + 2 * stride] = mask[i];
		if (mask[i] == 0.f) {
			image.data[i] = replaceValue;
			image.data[i + stride] = replaceValue;
			image.data[i + 2 * stride] = replaceValue;
		}
	}
	image.saveAs(imageOutPath);
	image.data = mask;
	image.saveAs(maskOutPath);
}