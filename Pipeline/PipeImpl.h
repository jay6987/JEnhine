// Description:
//   This is the implement of Pipe<T>
// 
// Copyright(c) 2019 - 2020 Fussen Technology Co., Ltd
#pragma once
#include <algorithm>
#include <sstream>
#include "Pipe.h"
#include "..\Common\Exception.h"

namespace JEngine
{
	template<typename T>
	inline T Pipe<T>::GetTemplate() const
	{
		if (buffers.empty())
			ThrowException("Pipe: pipe is not initialized.");
		return buffers[0];
	}


	//// Tell the pipe that each element should be constructed
	//// by copying the template.
	//template<typename T>
	//inline void Pipe<T>::SetTemplate(const T& elementTemplate)
	//{
	//	if (!buffers.empty())
	//		ThrowException("Pipe: buffer is already initialized.");
	//	buffers.emplace_back(elementTemplate);
	//	InitBufferIfParamsAreReady();
	//}

	// Tell the pipe that each element should be constructed
	// by copying the template.
	template<typename T>
	inline void Pipe<T>::SetTemplate(T&& elementTemplate, std::vector<size_t>&& dimentions)
	{
		if (!buffers.empty())
			ThrowExceptionAndLog("Pipe: buffer is already initialized.");
		buffers.emplace_back(std::move(elementTemplate));
		SetElementShape(dimentions);
		InitBufferIfParamsAreReady();
	}

	template<typename T>
	inline PipeWriteToken<T> Pipe<T>::GetWriteToken(size_t tokenWriteSize, bool isShotEnd)
	{
		WriteTokenInfoPtr tokenInfoPtr = GetWriteTokenInfo(tokenWriteSize, isShotEnd);
		return WriteToken(
			tokenInfoPtr,
			GetBufferPtrs(tokenInfoPtr->StartIndex)
		);
	}

	template<typename T>
	inline PipeReadToken<T> Pipe<T>::GetReadToken()
	{
		ReadTokenInfoPtr tokenInfoPtr =
			GetReadTokenInfo();
		return ReadToken(
			tokenInfoPtr,
			GetBufferPtrs(tokenInfoPtr->StartIndex)
		);
	}

	template<typename T>
	inline T* const* const Pipe<T>::GetBufferPtrs(size_t nStartIdx)
	{
		return bufferPtrs.data() + (nStartIdx % bufferSize);
	}

	template<typename T>
	inline void Pipe<T>::InitBufferIfParamsAreReady()
	{
		if (numReaders == 0 || numWriters == 0 || buffers.empty())
		{
			// parameters not ready
			return;
		}

		// if SetBufferSize have not been called, use default buffer size.
		// default buffer size ensure all writers and reader can work
		// in the same time.
		if (bufferSize == 0)
		{
			bufferSize = numWriters * writeSize +
				numReaders * (readSize - overlapSize) +
				overlapSize;
		}

		buffers.resize(bufferSize, buffers[0]);
		bufferPtrs.resize(2 * bufferSize - 1);

		for (size_t i = 0; i < bufferSize; ++i)
		{
			bufferPtrs[i] = &buffers[i];
		}
		for (size_t i = 0; i < bufferSize - 1; ++i)
		{
			bufferPtrs[i + bufferSize] = &buffers[i];
		}

	}

	template<typename T>
	inline void Pipe<T>::ClearBuffers()
	{
		buffers.clear();
		bufferPtrs.clear();
	}
}