// Description:
//   Tex2D<T> wraps 2-D texture memory

#pragma once

#include "cuda_runtime.h"

namespace JEngine
{
	template<typename T>
	class Tex2D
	{
	public:
		Tex2D(const size_t width = 0, const size_t height = 0);
		Tex2D(const Tex2D<T>& org);
		Tex2D(Tex2D<T>&& org) noexcept;
		~Tex2D();
		const cudaTextureObject_t& Get() const { return textureObject; }
		void Set(const T* pSrc, cudaStream_t cudaStream = NULL);
	private:
		const size_t width;
		const size_t height;
		cudaArray* pArrayOnDevice;
		cudaTextureObject_t textureObject;

		void Malloc();
		void CreateObject();
	};
}
