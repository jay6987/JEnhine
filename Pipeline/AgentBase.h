// Description:
//   AgentBase a is working unit in pipeline
//   The base class provide some components to record performance

#pragma once

#include<mutex>
#include<map>
#include<set>
#include<memory>
#include<string>
#include<future>
#include<atomic>

#include "..\Common\Timer.h"
#include "..\Common\Noncopyable.h"
#include "Pipe.h"

namespace JEngine
{

	class AgentBase : public Noncopyable
	{
	public:
		AgentBase(const std::string& name);

		virtual ~AgentBase() {};

		void SetPipes(std::vector<std::shared_ptr<PipeBase>>& pipes);

		virtual void GetReady() {};

		virtual void Start() = 0;

		virtual void Join() = 0;

		std::string GetAgentName() const { return name; }
		virtual size_t GetNumThreads() const = 0;

		double GetOccupiedTime() const { return accumulatedOccupiedSpan; }
		double GetAsyncTime() const { return accumulatedWaitAsyncTime; }

		// The average occupied span should be smaller than the 
		// thread-switching period.
		// e.g. in Windows UI mode,  thread-switching rate is about 60-70 Hz
		// i.e. the thread-switching period is about 0.014s
		// if the occupied span is small than 0.014, a task is usually
		// not be interupted.
		double GetAverageOccupiedSpan() const { return accumulatedOccupiedSpan / occupiedCount; }
		double GetMaxOccupiedTime() const { return maximumOccupiedSpan; }

	protected:

		template<typename PipeType>
		std::shared_ptr<Pipe<PipeType>> BranchPipeFromTrunk(const std::string&& name);

		// use this function to create new pipe, 
		// then the pipe will be foced closed when a thread throw an exception.
		template<typename PipeType>
		std::shared_ptr<Pipe<PipeType>> CreateNewPipe(const std::string&& name);

		template<typename PipeType>
		void MergePipeToTrunk(std::shared_ptr<Pipe<PipeType>> branch);

		// collect all pipes that connected to other agents.
		std::set<std::shared_ptr<PipeBase>> connectedPipes;

		void CloseAllConnectedPipes();

		class WaitAsyncScope;

		class ThreadOccupiedScope;

	private:

		const std::string name;

		std::mutex closePipeMutex;

		// before running this function,
		// pipes are all pipes INPUT and PASSBY this agent;
		// after running this function,
		// pipes are all pipes OUTPUT and PASSBY this agent;
		virtual void SetPipesImpl() = 0;

		// TODO: 
		// accumulatedOccupiedSpan and accumulatedWaitAsyncTime should
		// use atomic<double> when we upgrade to C++20
		// C++20 support atomic<Floating>::operatior+=

		double accumulatedOccupiedSpan = 0.0;
		double maximumOccupiedSpan = 0.0;
		std::atomic<size_t> occupiedCount = 0;

		double accumulatedWaitAsyncTime = 0.0;

		std::vector<std::shared_ptr<PipeBase>> trunk;

	};

	// A WaitAsyncScope is a scoped signal that 
	// claims the codes during its life time are runing
	// asynchronously.
	// A WaitAsyncScope can split the life time of
	// a ThreadOccupiedScope into two parts
	class AgentBase::WaitAsyncScope : Noncopyable
	{
	public:
		WaitAsyncScope(ThreadOccupiedScope* const pOccupied);
		~WaitAsyncScope();
	private:
		ThreadOccupiedScope* const pOccupied;
	};

	// A ThreadOccupiedScope is a scoped signal that
	// claims the codes during its life time will 
	// occupy a CPU thread.
	// The accumulated occupied span and maximum occupied time of 
	// the agent will be record.
	// A WaitAsyncScope can split a ThreadOccupiedScope into two parts
	class AgentBase::ThreadOccupiedScope : Noncopyable {
	public:
		ThreadOccupiedScope(AgentBase* pOwner);
		~ThreadOccupiedScope();
		friend class WaitAsyncScope;
	private:
		void GetThread();
		void ReleaseThread();
		AgentBase* const pOwner;
		Timer timer;
	};

	template<typename PipeType>
	inline std::shared_ptr<Pipe<PipeType>> AgentBase::BranchPipeFromTrunk(const std::string&& name)
	{
		auto pipeIt = trunk.begin();
		for (; pipeIt != trunk.end(); ++pipeIt)
		{
			if (name == pipeIt->get()->GetName())
			{
				auto pPipeIn = std::dynamic_pointer_cast<Pipe<PipeType>>(*pipeIt);
				connectedPipes.insert(*pipeIt);
				trunk.erase(pipeIt);
				return std::shared_ptr<Pipe<PipeType>>(pPipeIn);
			}
		}
		std::stringstream ss;
		ss << GetAgentName() << ": Pipe trunk does not contain a pipe named " << name;
		ThrowExceptionAndLog(ss.str());
	}

	template<typename PipeType>
	inline void AgentBase::MergePipeToTrunk(std::shared_ptr<Pipe<PipeType>> branch)
	{
		if (connectedPipes.count(branch) == 0)
		{
			ThrowExceptionAndLog("Please use CreateNewPipe<PipeType>() to create a pipe");
		}
		trunk.push_back(branch);
	}

	template<typename PipeType>
	inline std::shared_ptr<Pipe<PipeType>> AgentBase::CreateNewPipe(const std::string&& name)
	{
		std::shared_ptr<Pipe<PipeType>> pipe = std::make_shared<Pipe<PipeType>>(name);
		connectedPipes.insert(pipe);
		return pipe;
	}

}