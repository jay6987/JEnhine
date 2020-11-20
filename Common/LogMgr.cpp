//   LogMgr is used to write logs into file

#include <chrono>
#include <iostream>

#include "LogMgr.h"

namespace JEngine
{
	LogMgr::LogMgr(): logDebugEnabled(false)
	{
		LogDebug = std::bind(&LogMgr::LogDebugFake, this, std::placeholders::_1);
		Log("==================== LogMgr Initilized ===================================");
	}

	LogMgr::~LogMgr()
	{
		Log("==================== Log queue closed ====================================");

		messagesToWrite.Close();

		if (writeLogThread.valid())
		{
			writeLogThread.get();

			fileStream.close();
		}
	}
	void LogMgr::InitLogFile(const std::filesystem::path& filePath)
	{
		if (fileStream.is_open())
		{
			throw std::exception("LogMgr: log file is already opened.");
		}
		fileStream.open(filePath, std::ios::app);
		if (fileStream.good())
		{
			writeLogThread =
				std::async(
					std::launch::async,
					&LogMgr::WriteLogs, this
				);
		}
	}

	void LogMgr::EnableDebug()
	{
		logDebugEnabled = true;
		LogDebug = std::bind(&LogMgr::LogDebugReal, this, std::placeholders::_1);
	}

	void LogMgr::Log(const std::string& message)
	{
		messagesToWrite.Push(TimeStamp() + message);
	}

	void LogMgr::LogDebugReal(const std::string& message)
	{
		Log("[DEBUG]" + message);
	}

	std::string LogMgr::TimeStamp()
	{
		time_t timeStamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

		char sTimeStamp[32];
		struct tm buf;
		localtime_s(&buf, &timeStamp);
		strftime(sTimeStamp, sizeof(sTimeStamp), "[%Y-%m-%d %H:%M:%S] ", &buf);

		return sTimeStamp;
	}

	void LogMgr::WriteLogs()
	{
		fileStream << std::endl;
		try
		{
			while (true)
			{
				auto msg = messagesToWrite.Pop();
				std::cout << msg << std::endl;
				fileStream << msg << std::endl;
			}
		}
		catch (SyncQueueClosedAndEmptySignal&)
		{
		}
	}
}