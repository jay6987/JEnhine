// Description:
//   Pipe token is a token that claims a pack of elements is occupied
//   Pipe token should include information such as position, size, and 
//   whether the pack is the begining or ending of a shot.
#pragma once

#include <memory>

#include "PipeWriteTokenInfo.h"

namespace JEngine
{
	template<typename T>
	class PipeWriteToken
	{
	public:
		PipeWriteToken()
			:pBufferPtrs(nullptr)
		{}

		PipeWriteToken(
			std::shared_ptr<PipeWriteTokenInfo> pTokenInfo,
			T* const* const pBufferPtrs)
			: pTokenInfo(pTokenInfo)
			, pBufferPtrs(pBufferPtrs)
		{
		}

		PipeWriteToken<T>& operator = (const PipeWriteToken<T>& right)
		{
			if (this->pTokenInfo.get())
				ThrowException("WriteToken is already assigned.");
			this->pTokenInfo = right.pTokenInfo;
			this->pBufferPtrs = right.pBufferPtrs;
			return *this;
		}

		~PipeWriteToken<T>()
		{
			if (pTokenInfo.get())
			{
				PipeBase* const pPipeBase = pTokenInfo->Owner;
				pTokenInfo.reset();
				pPipeBase->ClearFinishedWriteTokens();
				pBufferPtrs = nullptr;
			}
		}

		const size_t GetStartIndex() const
		{
			return pTokenInfo->StartIndex;
		}

		const size_t GetSize() const
		{
			return pTokenInfo->Size;
		}

		const bool IsShotStart() const
		{
			return pTokenInfo->IsShotStart;
		}

		const bool IsShotEnd() const
		{
			return pTokenInfo->IsShotEnd;
		}

		T* const* const GetBufferPtrs() const
		{
			return pBufferPtrs;
		}

		T& GetBuffer(size_t i) const
		{
			return *pBufferPtrs[i];
		}

	private:

		std::shared_ptr<PipeWriteTokenInfo> pTokenInfo;

		T* const* pBufferPtrs;

	};
}