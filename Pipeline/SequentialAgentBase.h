// Description:
//   SequentialAgentBase a is working unit in pipeline while works sequentially

#pragma once

#include "AgentBase.h"

namespace JEngine
{
	class SequentialAgentBase : public AgentBase
	{
	public:
		SequentialAgentBase(
			const std::string& agentName);

		virtual ~SequentialAgentBase() {}

		void Start() override final;

		void Join() override final;

		size_t GetNumThreads() const override final { return 1; }

	protected:


		virtual void WorkFlow() = 0;

	private:

		std::future<void> worker;

		void WorkFlowWrappd();

	};
}
