// Description:
//   SequentialAgentBase a is working unit in pipeline while works sequentially

#include "SequentialAgentBase.h"
#include "../Common/GLog.h"

namespace JEngine
{
	SequentialAgentBase::SequentialAgentBase(
		const std::string& agentName)
		: AgentBase(agentName)
	{
	}

	void SequentialAgentBase::Start()
	{
		worker = std::async(
			std::launch::async,
			&SequentialAgentBase::WorkFlowWrappd, this
		);
	}

	void SequentialAgentBase::Join()
	{
		worker.get();
		CloseAllUsedPipes();
	}

	void SequentialAgentBase::WorkFlowWrappd()
	{
		try
		{
			WorkFlow();
		}
		catch (Exception&)
		{
			std::stringstream ss;
			ss << GetAgentName() << " die, because of an Exception. ";
			GLog(ss.str());
		}
		catch (std::exception& e)
		{
			std::stringstream ss;
			ss << GetAgentName() << " die, " << e.what();
			GLog(ss.str());
		}
		catch (...)
		{
			std::stringstream ss;
			ss << GetAgentName() << " die, unknown exception.";
			GLog(ss.str());
		}
		CloseAllUsedPipes();
	}

}
