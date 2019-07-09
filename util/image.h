#pragma once
#include <stdexcept>
#include "FreeImage.h"

typedef std::runtime_error ImageError;

class Image {

public:
	Image(const char* filePath);
	void saveAs(const char* filePath);

private:
	void loadFormat();
	void loadImage();
	void loadData();
	void saveData();

public:
	const char* filePath = NULL;
	float* data = NULL;
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int channels = 0;
	unsigned int length = 0;
	unsigned int pitch = 0;

private:
	FREE_IMAGE_FORMAT fiFormat = FIF_UNKNOWN;
	FIBITMAP* fiBitmap = NULL;
	FREE_IMAGE_TYPE fiType = FIT_UNKNOWN;
};
