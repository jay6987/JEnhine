// Description:
//   ProgressManager keeps writing Progress.ini file

#pragma once

#include <filesystem>
#include <vector>
#include <thread>
#include "../Common/Timer.h"

namespace JEngine
{
	class PipeBase;
	class IniFileReader;
	class ProgressManager
	{
	public:
		ProgressManager(const std::filesystem::path& fileName, double toleranceBlockedTime);
		~ProgressManager();
		void SetPipeToWatch(
			std::shared_ptr<PipeBase> pipe, size_t finishedIndex);
		void Start();
		void Stop();

	private:

		void WriteState(bool isRunning);

		void WriteProgress(float value);

		void WorkLoop();

		void WatchAndLog();

		bool stopped;

		std::shared_ptr<IniFileReader> file;

		std::vector<std::shared_ptr<PipeBase>> pipesToWatch;
		std::vector<size_t> pipesFinishedIndex;

		std::thread worker;

		const double toleranceBlockedTime;
		float lastProgress;

		Timer timer;
	};
}
