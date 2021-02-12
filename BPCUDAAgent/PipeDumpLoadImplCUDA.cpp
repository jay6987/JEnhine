// Description:
//   PipeDumpLoadImplCUDA.cpp specialize implementation of 
//   pipe dump and load functions for
//   DeviceMemory<float> and Tex2D<float>

#include "..\Pipeline\Pipe.h"
#include "..\Common\TypeDefs.h"
#include "Tex2D.h"
#include "DeviceMemory.h"
#include "cuda_runtime.h"

namespace JEngine
{

	void Pipe<DeviceMemory<float>>::SetDump(const std::filesystem::path& fileName)
	{
		if (pDumpFileStream.get())
			ThrowExceptionAndLog("Dump file is already set.");
		pDumpFileStream = std::make_shared<std::ofstream>(
			fileName, std::ios::binary);
	}

	void Pipe<DeviceMemory<float>>::SetLoad(const std::filesystem::path& /*fileName*/)
	{
	}

	void Pipe<DeviceMemory<float>>::DumpWriteToken(size_t nStart, const size_t nEnd)
	{
		FloatVec buf;
		while (nStart != nEnd)
		{
			buf.resize(bufferPtrs[nStart % bufferSize]->Size());
			cudaMemcpy(
				buf.data(),
				bufferPtrs[nStart % bufferSize]->Data(),
				buf.size() * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaPeekAtLastError() != cudaSuccess)
				ThrowExceptionAndLog(cudaGetErrorString(cudaGetLastError()));
			pDumpFileStream->write(
				(char*)buf.data(),
				buf.size() * sizeof(float));
			++nStart;
		}
	}

	void Pipe<DeviceMemory<float>>::LoadReadToken(const size_t /*nStart*/, const size_t /*nEnd*/)
	{
	}

	void Pipe<Tex2D<float>>::SetDump(const std::filesystem::path& /*fileName*/)
	{
	}

	void Pipe<Tex2D<float>>::SetLoad(const std::filesystem::path& /*fileName*/)
	{
	}

	void Pipe<Tex2D<float>>::DumpWriteToken(const size_t /*nStart*/, const size_t /*nEnd*/)
	{
	}

	void Pipe<Tex2D<float>>::LoadReadToken(const size_t /*nStart*/, const size_t /*nEnd*/)
	{
	}
}