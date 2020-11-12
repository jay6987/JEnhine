// Description:
//   This is a special implement of 
//   - Pipe<T>::SetDump
//   - Pipe<T>::SetLoad
//   - Pipe<T>::DumpWriteTokenImpl
//   - Pipe<T>::LoadReadTokenImpl
//   with common types

#include "Pipe.h"
#include "..\Common\TypeDefs.h"
#include "..\Common\Exception.h"

namespace JEngine
{
	template<>
	void Pipe<FloatVec>::SetDump(const std::filesystem::path& fileName)
	{
		if (pDumpFileStream.get())
			ThrowExceptionAndLog("Dump file is already set.");
		pDumpFileStream = std::make_shared<std::ofstream>(
			fileName, std::ios::binary);
	}

	template<>
	void Pipe<FloatVec>::SetLoad(const std::filesystem::path& fileName)
	{
		if (pLoadFileStream.get())
			ThrowExceptionAndLog("Load file is already set.");
		if (!std::filesystem::exists(fileName))
			ThrowExceptionAndLog("Can not open file: " + fileName.string());
		pLoadFileStream = std::make_shared<std::ifstream>(
			fileName, std::ios::binary);
	}

	template<>
	void Pipe<FloatVec>::DumpWriteToken(size_t nStart, const size_t nEnd)
	{
		while (nStart != nEnd)
		{
			FloatVec* pElement = bufferPtrs[nStart++ % bufferSize];
			pDumpFileStream->write(
				(char*)pElement->data(),
				pElement->size() * sizeof(float));
		}
	}

	template<>
	void Pipe<FloatVec>::LoadReadToken(size_t nStart, const size_t nEnd)
	{
		while (nStart != nEnd)
		{
			FloatVec* pElement = bufferPtrs[nStart++ % bufferSize];
			pLoadFileStream->read(
				(char*)pElement->data(),
				pElement->size() * sizeof(float));
		}
	}


	template<>
	void Pipe<UINT16Vec>::SetDump(const std::filesystem::path& fileName)
	{
		if (pDumpFileStream.get())
			ThrowException("Dump file is already set.");
		pDumpFileStream = std::make_shared<std::ofstream>(
			fileName, std::ios::binary);
	}

	template<>
	void Pipe<UINT16Vec>::SetLoad(const std::filesystem::path& fileName)
	{
		if (pLoadFileStream.get())
			ThrowExceptionAndLog("Load file is already set.");
		if (!std::filesystem::exists(fileName))
			ThrowExceptionAndLog("Can not open file: " + fileName.string());
		pLoadFileStream = std::make_shared<std::ifstream>(
			fileName, std::ios::binary);
	}

	template<>
	void Pipe<UINT16Vec>::DumpWriteToken(size_t nStart, const size_t nEnd)
	{
		while (nStart != nEnd)
		{
			UINT16Vec* pElement = bufferPtrs[nStart++ % bufferSize];
			pDumpFileStream->write(
				(char*)pElement->data(),
				pElement->size() * sizeof(unsigned short));
		}
	}

	template<>
	void Pipe<UINT16Vec>::LoadReadToken(size_t nStart, const size_t nEnd)
	{
		while (nStart != nEnd)
		{
			UINT16Vec* pElement = bufferPtrs[nStart++ % bufferSize];
			pLoadFileStream->read(
				(char*)pElement->data(),
				pElement->size() * sizeof(unsigned short));
		}
	}


	template<>
	void Pipe<INT16Vec>::SetDump(const std::filesystem::path& fileName)
	{
		if (pDumpFileStream.get())
			ThrowExceptionAndLog("Dump file is already set.");
		pDumpFileStream = std::make_shared<std::ofstream>(
			fileName, std::ios::binary);
	}

	template<>
	void Pipe<INT16Vec>::SetLoad(const std::filesystem::path& fileName)
	{
		if (pLoadFileStream.get())
			ThrowExceptionAndLog("Load file is already set.");
		if (!std::filesystem::exists(fileName))
			ThrowExceptionAndLog("Can not open file: " + fileName.string());
		pLoadFileStream = std::make_shared<std::ifstream>(
			fileName, std::ios::binary);
	}

	template<>
	void Pipe<INT16Vec>::DumpWriteToken(size_t nStart, const size_t nEnd)
	{
		while (nStart != nEnd)
		{
			INT16Vec* pElement = bufferPtrs[nStart++ % bufferSize];
			pDumpFileStream->write(
				(char*)pElement->data(),
				pElement->size() * sizeof(signed short));
		}
	}

	template<>
	void Pipe<INT16Vec>::LoadReadToken(size_t nStart, const size_t nEnd)
	{
		while (nStart != nEnd)
		{
			INT16Vec* pElement = bufferPtrs[nStart++ % bufferSize];
			pLoadFileStream->read(
				(char*)pElement->data(),
				pElement->size() * sizeof(signed short));
		}
	}

	template<>
	void Pipe<ByteVec>::SetDump(const std::filesystem::path& fileName)
	{
		if (pDumpFileStream.get())
			ThrowExceptionAndLog("Dump file is already set.");
		pDumpFileStream = std::make_shared<std::ofstream>(
			fileName, std::ios::binary);
	}

	template<>
	void Pipe<ByteVec>::SetLoad(const std::filesystem::path& fileName)
	{
		if (pLoadFileStream.get())
			ThrowExceptionAndLog("Load file is already set.");
		if (!std::filesystem::exists(fileName))
			ThrowExceptionAndLog("Can not open file: " + fileName.string());
		pLoadFileStream = std::make_shared<std::ifstream>(
			fileName, std::ios::binary);
	}

	template<>
	void Pipe<ByteVec>::DumpWriteToken(size_t nStart, const size_t nEnd)
	{
		while (nStart != nEnd)
		{
			ByteVec* pElement = bufferPtrs[nStart++ % bufferSize];
			pDumpFileStream->write(
				(char*)pElement->data(),
				pElement->size() * sizeof(unsigned char));
		}
	}

	template<>
	void Pipe<ByteVec>::LoadReadToken(size_t nStart, const size_t nEnd)
	{
		while (nStart != nEnd)
		{
			ByteVec* pElement = bufferPtrs[nStart++ % bufferSize];
			pLoadFileStream->read(
				(char*)pElement->data(),
				pElement->size() * sizeof(unsigned char));
		}
	}

}