// Description:
//   LogMgr is used to write logs into file

#pragma once

#include <fstream>
#include <string>
#include <future>
#include <filesystem>
#include <functional>
#include "SyncQueue.h"

namespace JEngine
{

	class LogMgr
	{
	public:
		LogMgr();
		~LogMgr();
		void InitLogFile(const std::filesystem::path& filePathWithPostFix);
		void EnableDebug();
		void Log(const std::string& message);
		std::function<void(const std::string&)> LogDebug;
		bool LogDebugEnabled() const { return logDebugEnabled; }
	private:
		void WriteLogs();
		void LogDebugReal(const std::string& message);
		void LogDebugFake(const std::string& /*message*/) {};
		std::string TimeStamp();

		std::future<void> writeLogThread;
		std::fstream fileStream;
		SyncQueue<std::string> messagesToWrite;
		bool logDebugEnabled;
	};


}
