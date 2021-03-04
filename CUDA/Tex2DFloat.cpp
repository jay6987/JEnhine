// Description:
//   Tex2D<T> wraps 2-D texture memory
//   Tex2DFloat.cpp specialize implementation of Tex2D<float>


#include <iostream>
#include <sstream>

#include "Tex2D.h"
#include "../Common/Exception.h"
#include "ErrorChecking.h"

namespace JEngine
{
	Tex2D<float>::Tex2D(const size_t width, const size_t height)
		: width(width)
		, height(height)
		, pArrayOnDevice(nullptr)
	{
		Malloc();
		CreateObject();
	}

	Tex2D<float>::Tex2D(const Tex2D<float>& org)
		: width(org.width)
		, height(org.height)
	{
		Malloc();
		CreateObject();

		CUDA_SAFE_CALL(
			cudaMemcpy2DArrayToArray(
				pArrayOnDevice, 0, 0,
				org.pArrayOnDevice, 0, 0,
				width * sizeof(float), height,
				cudaMemcpyDeviceToDevice));
	}

	Tex2D<float>::Tex2D(Tex2D<float>&& org) noexcept
		: width(org.width)
		, height(org.height)
		, pArrayOnDevice(org.pArrayOnDevice)
		, textureObject(org.textureObject)
	{
		org.pArrayOnDevice = nullptr;
		org.textureObject = NULL;
	}

	Tex2D<float>::~Tex2D()
	{
		cudaFreeArray(pArrayOnDevice);
		cudaDestroyTextureObject(textureObject);
	}

	template<>
	void Tex2D<float>::Malloc()
	{
		cudaChannelFormatDesc m_channelDesc =
			cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		cudaError_t ec = cudaMallocArray(
			&pArrayOnDevice,
			&m_channelDesc,
			width,
			height
		);

		if (cudaErrorMemoryAllocation == ec)
		{
			std::stringstream ss;
			ss << "CUDA failed to malloc " <<
				width * height * sizeof(float) << " bytes of array memory, cudaError = " << cudaGetLastError();
			ThrowExceptionAndLog(ss.str());
		}
		else if (cudaSuccess != ec)
		{
			ThrowExceptionAndLog(cudaGetErrorString(cudaGetLastError()));
		}

	}

	template<>
	void Tex2D<float>::CreateObject()
	{

		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));

		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = pArrayOnDevice;

		cudaTextureDesc             texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = false;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		texDescr.borderColor[0] = NAN;
		texDescr.borderColor[1] = NAN;
		texDescr.borderColor[2] = NAN;
		texDescr.borderColor[3] = NAN;

		CUDA_SAFE_CALL(cudaCreateTextureObject(&textureObject, &texRes, &texDescr, NULL));

	}

	void Tex2D<float>::Set(const float* pSrc, cudaStream_t cudaStream)
	{
		CUDA_SAFE_CALL(
			cudaMemcpy2DToArrayAsync(
				pArrayOnDevice, 0, 0,
				pSrc,
				width * sizeof(float),
				width * sizeof(float),
				height,
				cudaMemcpyHostToDevice,
				cudaStream
			));
	}

	void Tex2D<float>::Set(const float* pSrc)
	{
		CUDA_SAFE_CALL(
			cudaMemcpy2DToArray(
				pArrayOnDevice, 0, 0,
				pSrc,
				width * sizeof(float),
				width * sizeof(float),
				height,
				cudaMemcpyHostToDevice
			));
	}
}