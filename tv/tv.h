#pragma once

namespace TV {

template <typename T>
class Base {
public:
	Base<T>(size_t width, size_t height) : width(width), height(height) {}
	virtual void apply(T* hImage) = 0;
protected:
	size_t width;
	size_t height;
};

}