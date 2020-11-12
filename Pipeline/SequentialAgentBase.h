// Description:
//   SequentialAgentBase a is working unit in pipeline whitch works sequentially.
//   SequentialAgent can includes multi steps, and multi steps can work concurrently,
//   but each step works sequentially.

#pragma once

#include "AgentBase.h"

namespace JEngine
{
	class SequentialAgentBase : public AgentBase
	{
	public:
		SequentialAgentBase(
			const std::string& agentName,
			const int numSteps);

		virtual ~SequentialAgentBase() {}

		void Start() override final;

		void Join() override final;

		size_t GetNumThreads() const override final { return 1; }

	protected:

		virtual void WorkFlow0() = 0;
		virtual void WorkFlow1() {}
		virtual void WorkFlow2() {}
		virtual void WorkFlow3() {}
		virtual void WorkFlow4() {}

	private:

		std::vector<std::future<void>> workers;

		void WorkFlowWrappd(int index);

		const int numSteps;

	};
}
