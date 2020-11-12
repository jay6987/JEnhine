#include "pch.h"
#include <string>
#include "../../Common/TypeDefs.h"
#include "../../Common/Timer.h"
#include "../../Common/GLog.h"
#include "../../Common/SyncMap.h"
#include "../../Pipeline/Pipe.h"
#include "../../Pipeline/SequentialAgentBase.h"

using namespace JEngine;

namespace UTPipeline
{
	class MakeOrderAgent : public SequentialAgentBase
	{
	public:
		MakeOrderAgent();
	private:
		void SetPipesImpl() override;
		void WorkFlow0() override;
		void WorkFlow1() override;

		std::shared_ptr<Pipe<ByteVec>> pipeIn;
		std::shared_ptr<Pipe<ByteVec>> pipeOut;
		SyncMap<char, char> queue;
	};

	TEST(SequenialAgent, MultiStep)
	{
		std::string slogen("We will rock you!");
		std::string outputSlogen;
		std::vector<char> wrongOrder =
		{ 2,3,8,4,0,11,5,6,7,1,12,13,9,10,15,16,14 };

		MakeOrderAgent agent;

		std::shared_ptr<Pipe<ByteVec>> pipeIn = std::make_shared< Pipe<ByteVec>>("InputCharacters");
		pipeIn->SetProducer("main", 1, 1);
		pipeIn->SetTemplate(ByteVec(2), { 2 });

		std::shared_ptr<Pipe<ByteVec>> pipeOut;
		{
			std::vector<std::shared_ptr<PipeBase>> bus;
			bus.push_back(pipeIn);
			agent.SetPipes(bus);
			pipeOut = std::dynamic_pointer_cast<Pipe<ByteVec>>(bus[0]);
			pipeOut->SetConsumer("main", 1, 1, 0);
		}

		agent.GetReady();

		agent.Start();

		std::future<void> pushThread =
			std::async(
				std::launch::async,
				[&]() {
					for (int i = 0; i < wrongOrder.size(); ++i)
					{
						auto writeToken = pipeIn->GetWriteToken(1, i + 1 == wrongOrder.size());
						writeToken.GetBuffer(0)[0] = wrongOrder[i];
						writeToken.GetBuffer(0)[1] = slogen[wrongOrder[i]];
					}
					pipeIn->Close();
				});

		std::future<void> pullThread =
			std::async(
				std::launch::async,
				[&]() {
					try
					{
						while (true)
						{
							auto readToken = pipeOut->GetReadToken();
							outputSlogen.push_back(readToken.GetBuffer(0)[0]);
						}
					}
					catch (PipeClosedAndEmptySignal&)
					{
					}
				});

		agent.Join();

		EXPECT_STREQ(slogen.c_str(), outputSlogen.c_str());
	}

	MakeOrderAgent::MakeOrderAgent()
		:SequentialAgentBase("MakeOrderAgent", 2)
	{
	}

	void MakeOrderAgent::SetPipesImpl()
	{
		pipeIn = BranchPipeFromTrunk<ByteVec>("InputCharacters");
		pipeIn->SetConsumer(GetAgentName(), GetNumThreads(), 1, 0);

		pipeOut = std::make_shared<Pipe<ByteVec>>("OutputCharacters");
		pipeOut->SetProducer(GetAgentName(), 1, 1);
		pipeOut->SetTemplate(ByteVec(1), { 1 });
		MergePipeToTrunk(pipeOut);
	}

	void MakeOrderAgent::WorkFlow0()
	{
		try
		{
			while (true)
			{
				auto readToken = pipeIn->GetReadToken();
				queue.Insert(readToken.GetBuffer(0)[0], readToken.GetBuffer(0)[1]);
				if (readToken.IsShotEnd())
					break;
			}
		}
		catch (PipeClosedAndEmptySignal&)
		{
		}
		queue.Close();
	}

	void MakeOrderAgent::WorkFlow1()
	{
		try
		{
			for (char i = 0;; ++i)
			{
				const char c = queue.Wait(i);
				auto writeToken = pipeOut->GetWriteToken(1, true);
				writeToken.GetBuffer(0)[0] = c;
				queue.Erase(i);
			}
		}
		catch (SyncMapClosedSignal&)
		{
			pipeOut->Close();
		}
	}
}