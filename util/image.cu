#include "image.h"

unsigned char clamp(float value) {
	return (value > 255) ? 255 : (value < 0) ? 0 : value;
}

Image::Image(const char * filePath)
	: filePath(filePath)
{
	this->loadFormat();
	this->loadImage();
	this->loadData();
}

void Image::saveAs(const char * filePath)
{
	this->saveData();
	FreeImage_Save(fiFormat, fiBitmap, filePath);
}

void Image::loadFormat()
{
	fiFormat = FreeImage_GetFileType(filePath, 0);
	if (fiFormat == FIF_UNKNOWN) {
		fiFormat = FreeImage_GetFIFFromFilename(filePath);
	}
	if (fiFormat == FIF_UNKNOWN) {
		throw ImageError("Unknown file format.");
	}
}

void Image::loadImage()
{
	if (FreeImage_FIFSupportsReading(fiFormat)) {
		fiBitmap = FreeImage_Load(fiFormat, filePath);
		fiType = FreeImage_GetImageType(fiBitmap);
		width = FreeImage_GetWidth(fiBitmap);
		height = FreeImage_GetHeight(fiBitmap);
		pitch = FreeImage_GetPitch(fiBitmap);
	}
	else {
		throw ImageError("File format not supported.");
	}
}

void Image::loadData()
{
	if ((fiType == FIT_BITMAP) && (FreeImage_GetBPP(fiBitmap) == 24)) {
		channels = 3;
		length = width * height * channels;
		data = new float[length];
		BYTE *bits = (BYTE*)FreeImage_GetBits(fiBitmap);
		for (int y = 0; y < height; y++) {
			BYTE *pixel = (BYTE*)bits;
			for (int x = 0; x < width; x++) {
				data[y + x * height] = pixel[FI_RGBA_RED] / 255.f;
				data[y + x * height + width * height] = pixel[FI_RGBA_GREEN] / 255.f;
				data[y + x * height + 2 * width * height] = pixel[FI_RGBA_BLUE] / 255.f;
				pixel += 3;
			}
			bits += pitch;
		}
	}
}

void Image::saveData()
{
	if ((fiType == FIT_BITMAP) && (FreeImage_GetBPP(fiBitmap) == 24)) {
		BYTE *bits = (BYTE*)FreeImage_GetBits(fiBitmap);
		for (int y = 0; y < height; y++) {
			BYTE *pixel = (BYTE*)bits;
			for (int x = 0; x < width; x++) {
				pixel[FI_RGBA_RED] = clamp(data[y + x * height] * 255.f);
				pixel[FI_RGBA_GREEN] = clamp(data[y + x * height + width * height] * 255.f);
				pixel[FI_RGBA_BLUE] = clamp(data[y + x * height + 2 * width * height] * 255.f);
				pixel += 3;
			}
			bits += pitch;
		}
	}
}
