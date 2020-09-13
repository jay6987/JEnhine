#pragma once

#include "../../Pipeline/SequentialAgentBase.h"

namespace JEngine
{

	class TestExceptionSequentialAgent :
		public SequentialAgentBase
	{
	public:
		TestExceptionSequentialAgent(const int errorNum);
		struct UnknownException {};

	private:

		const int errorNum;

		void SetPipesImpl() override {};

		void WorkFlow() override;

	};

}

