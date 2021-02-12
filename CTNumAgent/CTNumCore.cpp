#include "CTNumCore.h"
#include "../Performance/BasicMathIPP.h"

namespace JEngine
{
	using namespace BasicMathIPP;

	CTNumCore::CTNumCore(
		const size_t width,
		const size_t height,
		const float norm0,
		const float norm1,
		const float muWater)
		: width(width)
		, height(height)
		, norm0(norm0)
		, norm1(norm1)
		, muWater(muWater)
	{
	}

	bool CTNumCore::Process(float* inout) const
	{
		Mul(inout, norm0 * 1000.f / muWater, width * height);
		Sub(inout, 1000.f, width * height);

		for (float* p = inout; p != inout + width * height; ++p)
		{
			if (*p > 300.0f)
			{
				*p = std::min(8191.f, (*p - 300.0f) * norm1 + 300.0f);
			}
		}

		return false;
	}
}
