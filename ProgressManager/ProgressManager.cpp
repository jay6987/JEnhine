#include "ProgressManager.h"

#include "../Pipeline/PipeBase.h"
#include "../Config/IniFileReader.h"
#include "../Common/GLog.h"


namespace JEngine
{
	ProgressManager::ProgressManager(
		const std::filesystem::path& fileName,
		double toleranceBlockedTime)
		: file(new IniFileReader(fileName))
		, toleranceBlockedTime(toleranceBlockedTime)
		, stopped(false)
		, lastProgress(-1)
	{
		file->Write(L"Progress", L"IsPreviewUpdated", L"0");
		WriteProgress(0.f);
		WriteState(true);
	}

	ProgressManager::~ProgressManager()
	{
		Stop();
	}

	void ProgressManager::SetPipeToWatch(
		std::shared_ptr<PipeBase> pipe,
		size_t finishedIndex)
	{
		pipesToWatch.push_back(pipe);
		pipesFinishedIndex.push_back(finishedIndex);
	}
	void ProgressManager::Start()
	{
		worker = std::thread(&ProgressManager::WorkLoop, this);
	}

	void ProgressManager::Stop()
	{
		stopped = true;
		if (file.get())
		{
			if (worker.joinable())
				worker.join();
			WatchAndLog();
			WriteState(false);
		}
	}

	void ProgressManager::WriteState(bool isRunning)
	{
		file->Write(
			L"Progress", L"IsRunning",
			isRunning ? L"1" : L"0"
		);
	}
	void ProgressManager::WriteProgress(float value)
	{
		if (value == 0.0f)
		{
			file->Write(L"Progress", L"Progress", 0);
		}
		else if (value < 1.0f)
		{
			file->Write(L"Progress", L"Progress", value);
		}
		else
		{
			file->Write(L"Progress", L"Progress", 1);
		}
	}

	void ProgressManager::WorkLoop()
	{
		try
		{
			timer.Tic();
			while (!stopped)
			{
				WatchAndLog();
				Timer::Sleep(1);
			}
			return;
		}
		catch (Exception& e)
		{
			GLog("ProgressManager die due to an Exception: " + e.What());
		}
		catch (std::exception& e)
		{
			GLog(std::string("ProgressManager die due to an exception: ") + e.what());
		}
		catch (...)
		{
			GLog("ProgressManager die due to an unknown exception. ");
		}

		for (auto& pipe : pipesToWatch)
		{
			pipe->Close();
		}
		Timer::Sleep(2);
		WriteState(false);
		exit(1);
	}

	void ProgressManager::WatchAndLog()
	{
		size_t finished = 0;
		size_t total = 0;
		for (size_t i = 0; i < pipesToWatch.size(); ++i)
		{
			finished += pipesToWatch[i]->GetReadDonePos();
			total += pipesFinishedIndex[i];
		}
		float progress = (float)finished / (float)total;
		WriteProgress(progress);
		//GLog("Progress: " + std::to_string(progress));

		if (progress == lastProgress && progress != 1)
		{
			if (timer.Toc() > toleranceBlockedTime)
			{
				ThrowException("Progress has been blocked more than "
					+ std::to_string(toleranceBlockedTime) + "s. The process will be forced exit in 2 seconds.");
			}
		}
		else
		{
			lastProgress = progress;
			timer.Tic();
		}
	}
}
