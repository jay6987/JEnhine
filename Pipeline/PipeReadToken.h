// Description:
//   Pipe token is a token that claims a pack of elements is occupied
//   Pipe token should include information such as position, size, and 
//   whether the pack is the begining or ending of a shot.
#pragma once

#include <memory>
#include "../Common/Exception.h"
#include "PipeReadTokenInfo.h"

namespace JEngine
{
	template<typename T>
	class PipeReadToken
	{
	public:
		PipeReadToken()
			: pBufferPtrs(nullptr)
			, pTokenInfo()
		{}

		PipeReadToken<T>(
			std::shared_ptr<PipeReadTokenInfo> pTokenInfo,
			T* const* const pBufferPtrs
			)
			: pTokenInfo(pTokenInfo)
			, pBufferPtrs(pBufferPtrs)
		{
		}

		bool Null()
		{
			return !pTokenInfo.get();
		}

		PipeReadToken<T>& operator = (const PipeReadToken<T>& right)
		{
			if (this->pTokenInfo.get())
				ThrowException("ReadToken is already assigned.");
			this->pTokenInfo = right.pTokenInfo;
			this->pBufferPtrs = right.pBufferPtrs;
			return *this;
		}

		~PipeReadToken<T>()
		{
			if (pTokenInfo.get())
			{
				PipeBase* const pPipeBase = pTokenInfo->Owner;
				pTokenInfo.reset();
				pPipeBase->ClearFinishedReadTokens();
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

		const size_t GetOverlapSize() const
		{
			return pTokenInfo->OverlapSize;
		}

		const bool IsShotStart() const
		{
			return pTokenInfo->IsShotStart;
		}

		const bool IsShotEnd() const
		{
			return pTokenInfo->IsShotEnd;
		}

		const T* const* const GetBufferPtrs() const
		{
			return pBufferPtrs;
		}

		const T& GetBuffer(size_t i) const
		{
			return *pBufferPtrs[i];
		}

		T* const* const GetMutableBufferPtrs() const
		{
			return pBufferPtrs;
		}

		T& GetMutableBuffer(size_t i) const
		{
			return *pBufferPtrs[i];
		}

	private:

		std::shared_ptr<PipeReadTokenInfo> pTokenInfo;

		T* const* pBufferPtrs;

	};
}
