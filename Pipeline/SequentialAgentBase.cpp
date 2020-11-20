// Description:
//   SequentialAgentBase a is working unit in pipeline while works sequentially

#include "SequentialAgentBase.h"
#include "../Common/GLog.h"

namespace JEngine
{
	SequentialAgentBase::SequentialAgentBase(
		const std::string& name,
		const int numSteps)
		: AgentBase(name)
		, numSteps(numSteps)
	{
	}

	void SequentialAgentBase::Start()
	{
		for (int i = 0; i < numSteps; ++i)
		{
			workers.emplace_back(
				std::async(
					std::launch::async,
					&SequentialAgentBase::WorkFlowWrappd,
					this,
					i));
		}
	}

	void SequentialAgentBase::Join()
	{
		for (auto& w : workers)
		{
			w.get();
		}
		CloseConnectedPipes();
	}

	void SequentialAgentBase::WorkFlowWrappd(int index)
	{
		try
		{
			switch (index)
			{
			case 0:
				WorkFlow0(); break;
			case 1:
				WorkFlow1(); break;
			case 2:
				WorkFlow2(); break;
			case 3:
				WorkFlow3(); break;
			case 4:
				WorkFlow4(); break;
			default:
				ThrowException("SequentialAgentBase can only contains less than 6 steps");
			}
		}
		catch (Exception&)
		{
			std::stringstream ss;
			ss << GetAgentName() << " step #"<< index << " die, because of an Exception. ";
			GLog(ss.str());
			CloseConnectedPipes();
		}
		catch (std::exception& e)
		{
			std::stringstream ss;
			ss << GetAgentName() << " step #" << index << " die, " << e.what();
			GLog(ss.str());
			CloseConnectedPipes();
		}
		catch (...)
		{
			std::stringstream ss;
			ss << GetAgentName() << " step #" << index << " die, unknown exception.";
			GLog(ss.str());
			CloseConnectedPipes();
		}
	}

}
