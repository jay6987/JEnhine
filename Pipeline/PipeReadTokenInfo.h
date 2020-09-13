// Description:
//   Pipe token is a token that claims a pack of elements is occupied
//   Pipe token should include information such as position, size, and 
//   whether the pack is the begining or ending of a shot.
//   This is a base class of PipeReadToekn, which contains all informations
//   about a PipeReadToken except the type-related pointer to buffer

#pragma once

#include "../Common/Noncopyable.h"

namespace JEngine
{
	class PipeBase;

	struct PipeReadTokenInfo : public Noncopyable
	{
		PipeReadTokenInfo(
			PipeBase* pOwner,
			const size_t nStartPos,
			const size_t nTokenSize,
			const size_t nOverlapSize,
			const bool bShotStart,
			const bool bShotEnd
		)
			: Owner(pOwner)
			, StartIndex(nStartPos)
			, Size(nTokenSize)
			, OverlapSize(nOverlapSize)
			, IsShotStart(bShotStart)
			, IsShotEnd(bShotEnd)
		{}

		const size_t StartIndex;
		const size_t Size;
		const size_t OverlapSize;
		const bool IsShotStart;
		const bool IsShotEnd;
		PipeBase* const Owner;
	};
}