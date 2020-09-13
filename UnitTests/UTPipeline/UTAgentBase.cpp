#include "pch.h"
#include "SumAgent.h"
#include "TestExceptionConcurrentAgent.h"
#include "TestExceptionSequentialAgent.h"
#include "../../Common/GLog.h"

using namespace JEngine;

namespace UTPipeline
{

	bool isFileInitilized = false;

	class AgentBaseTest : public ::testing::Test {
		void SetUp() override {
			if (!isFileInitilized) {
				isFileInitilized = true;
				InitGLog("UTAgentBase.log");
			}
		}
	};
	TEST_F(AgentBaseTest, MultiThreadSum)
	{
		size_t n = 1000000;
		std::vector<int> a(n);
		std::vector<int> b(n);
		std::vector<int> sum(n);
		std::vector<int> sum_exp(n);

		for (size_t i = 0; i < n; ++i)
		{
			a[i] = static_cast<int>(i);
			b[i] = static_cast<int>(2 * i);
			sum_exp[i] = static_cast<int>(3 * i);
		}

		const int nThreads = 7;
		const int packSize = 10000;
		SumAgent agent(nThreads, packSize, a, b, sum);

		agent.GetReady();

		agent.Start();

		agent.Join();

		ASSERT_EQ(sum, sum_exp);

	}

	// Jifeng: ExceptionInThread has no self- validation, 
	// user should open the log file to check.

	TEST_F(AgentBaseTest, ExceptionInThreadCocurrent)
	{
		const int nThreads = 3;
		TestExceptionConcurrentAgent agent(nThreads);

		agent.GetReady();

		agent.Start();

		agent.Join();

	}

	TEST_F(AgentBaseTest, ExceptionInThreadSequential)
	{
		TestExceptionSequentialAgent agent1(0);
		TestExceptionSequentialAgent agent2(1);
		TestExceptionSequentialAgent agent3(2);

		agent1.GetReady();
		agent2.GetReady();
		agent3.GetReady();

		agent1.Start();
		agent2.Start();
		agent3.Start();

		agent1.Join();
		agent2.Join();
		agent3.Join();

	}
}