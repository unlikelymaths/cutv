#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include "util/image.h"
#include "util/tools.h"
#include "tv/denoise.h"
#include "tv/deblur.h"
#include "tv/inpaint.h"


bool* createMask(float* imageData, size_t width, size_t height)
{
	bool* mask = new bool[width * height];
	for (int i = 0; i < width * height; ++i)
	{
		mask[i] = imageData[i] > 0.5f;
	}
	return mask;
}

template<typename T>
void runDenoise() 
{
	auto start = std::chrono::high_resolution_clock::now();

	Image image("test.noise.jpg");
	int channelStride = image.width * image.height;
	TV::Denoise<T>* tv = new TV::Denoise<T>(image.width, image.height);
	for (int channel = 0; channel < 3; channel++) {
		tv->apply(image.data + channel * channelStride);
	}
	delete tv;
	image.saveAs("test.noise.out.jpg");

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Denoise time: " << elapsed.count() << " s\n";
}

template<typename T>
void runDeblur()
{
	auto start = std::chrono::high_resolution_clock::now();

	Image image("test.blur.jpg");
	int channelStride = image.width * image.height;
	TV::Deblur<T>* tv = new TV::Deblur<T>(image.width, image.height);
	for (int channel = 0; channel < 3; channel++) {
		tv->apply(image.data + channel * channelStride);
	}
	delete tv;
	image.saveAs("test.blur.out.jpg");

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Deblur time: " << elapsed.count() << " s\n";
}

template<typename T>
void runInpainting()
{
	auto start = std::chrono::high_resolution_clock::now();

	Image image("test.black.png");
	Image maskImage("test.black.mask.png");
	bool* mask = createMask(maskImage.data, maskImage.width, maskImage.height);
	int channelStride = image.width * image.height;
	TV::Inpaint<T>* tv = new TV::Inpaint<T>(image.width, image.height, mask);
	for (int channel = 0; channel < 3; channel++) {
		tv->apply(image.data + channel * channelStride);
	}
	delete tv;
	image.saveAs("test.black.out.png");

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Inpainting time: " << elapsed.count() << " s\n";
}

int main(int argc, const char* argv[])
{
	runDenoise<float>();
	runDeblur<float>();
	runInpainting<float>();
	return 0;
}
