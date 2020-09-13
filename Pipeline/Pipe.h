// Description:
//   Pipe<T> is a template class of a circulum buffer.
//   It supports multi-thread writing and reading.
//   The reading threads are blocked when there is no data left.
//   The writing threads are blocked when buffer is full.
//   element type of Pipe should have copy constructor,
//   would be better to have move constructor

#pragma once

#include <vector>

#include "PipeReadToken.h"
#include "PipeWriteToken.h"
#include "PipeBase.h"

namespace JEngine
{
	template<typename T>
	class Pipe : public PipeBase
	{

	public:

		typedef PipeWriteToken<T>           WriteToken;
		typedef PipeReadToken<T>            ReadToken;

	public:

		Pipe<T>(const std::string& pipeName)
			: PipeBase(pipeName) {		};


		//typedef std::shared_ptr<WriteToken> WriteTokenPtr;
		//typedef std::shared_ptr<ReadToken>  ReadTokenPtr;

		void SetDump(const std::filesystem::path& fileName) override;

		void SetLoad(const std::filesystem::path& fileName) override;

		T GetTemplate() const;

		WriteToken GetWriteToken(size_t writeSize, bool isShotEnd);

		ReadToken GetReadToken();

		//void SetTemplate(const T& elementTemplate);

		void SetTemplate(T&& elementTemplate, std::vector<size_t>&& dimentions);

	private:

		T* const* const GetBufferPtrs(size_t nStartIdx);

		void InitBufferIfParamsAreReady() override;
		void ClearBuffers() override;

		// Each Type of Pipe<Type> should specialize this function
		void DumpWriteToken(size_t nStart, size_t nEnd) override;

		// Each Type of Pipe<Type> should specialize this function
		void LoadReadToken(size_t nStart, size_t nEnd) override;

		std::vector<T> buffers;
		std::vector<T*> bufferPtrs;
	};
}

#include "PipeImpl.h"