#include "pch.h"

#include "../../Common/LogMgr.h"

using namespace JEngine;

namespace UTCommon
{
	// Junjie: This unit test has no self- validation, 
	// user should open the log file to check.

	LogMgr logMgr;
	bool fileInitilized = false;

	class LogMgrTest : public ::testing::Test {
		void SetUp() override {
			if (!fileInitilized) {
				fileInitilized = true;
				logMgr.InitLogFile("UTLogMgr.log");
			}
		}
	};

	TEST_F(LogMgrTest, PushLog)
	{
		logMgr.Log("This log should be recorded.");
	}

	TEST_F(LogMgrTest, DebugLog)
	{
		// this line is not pushed
		logMgr.LogDebug("This debug log should NOT be recorded.");

		logMgr.EnableDebug();

		// this line should be pushed
		logMgr.LogDebug("This debug log should be recorded.");
	}

	void LogThreadID(size_t id)
	{
		std::string msg("This is thread #");
		msg += std::to_string(id);
		logMgr.Log(msg);
	}

	TEST_F(LogMgrTest, MultiThreadLogging)
	{
		std::vector<std::thread> threadPool;
		for (size_t i = 0; i < 10; ++i)
		{
			threadPool.emplace_back(std::thread(&LogThreadID, i));
		}
		for (std::thread& th : threadPool)
		{
			th.join();
		}
	}
}