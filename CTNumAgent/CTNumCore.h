// Description:
//   CTNumCore convert absortion value to CT#

#pragma once

#include "..\Common\TypeDefs.h"
#include "..\Common\Noncopyable.h"

namespace JEngine
{
	class CTNumCore : Noncopyable
	{
	public:
		CTNumCore(
			const size_t width,
			const size_t height,
			const float norm0,
			const float norm1,
			const float muWater);

		bool Process(
			float* inout) const;

	private:
		const size_t width;
		const size_t height;
		const float norm0;
		const float norm1;
		const float muWater;
	};
}
