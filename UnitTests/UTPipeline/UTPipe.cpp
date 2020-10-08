#include "pch.h"
#include <future>
#include "../../Pipeline/Pipe.h"
#include "../../Common/Timer.h"
#include "../../Common/SyncQueue.h"

using namespace JEngine;

namespace UTPipeline
{
	typedef size_t DataType;
	typedef Pipe<size_t> PipeType;
	typedef std::shared_ptr<PipeType> PipePtr;

	TEST(PipeTest, GetWriteToken)
	{
		const size_t numWriterThreads = 4;
		const size_t nWriteSize = 10;
		const size_t numReadThreads = 1;
		const size_t nReadSize = 5;
		const size_t nOverlapSize = 2;
		Pipe<size_t> pipe("PipeForTest");
		pipe.SetTemplate(0, {});
		pipe.SetProducer("producer", numWriterThreads, nWriteSize);
		pipe.SetConsumer("consumer", numReadThreads, nReadSize, nOverlapSize);

		// get first write token
		{
			auto wt =
				pipe.GetWriteToken(nWriteSize, false);
			EXPECT_FALSE(wt.IsShotEnd());
			EXPECT_TRUE(wt.IsShotStart());
			EXPECT_EQ(wt.GetSize(), nWriteSize);
		}

		// get shot-end token
		{
			PipeType::WriteToken wt =
				pipe.GetWriteToken(nWriteSize, true);
			EXPECT_FALSE(wt.IsShotStart());
			EXPECT_TRUE(wt.IsShotEnd());
			EXPECT_EQ(nWriteSize, wt.GetSize());
		}

		// after shot-end token
		{
			PipeType::WriteToken wt =
				pipe.GetWriteToken(nWriteSize, false);
			EXPECT_TRUE(wt.IsShotStart());
		}

		// get too-large token		
		try
		{
			PipeType::WriteToken wt =
				pipe.GetWriteToken(pipe.GetBufferSize() + 1, true);
			FAIL() << "Expected exception not caught";
		}
		catch (Exception& e)
		{
			EXPECT_STREQ(e.what(),
				"Pipe::GetWriteToken(): write size is larger than buffer size!");
		}
		catch (...)
		{
			FAIL() << "Expected exception not caught";
		}

		// write to closed pipe
		pipe.Close();
		try
		{
			PipeType::WriteToken wt =
				pipe.GetWriteToken(pipe.GetBufferSize(), true);
			FAIL() << "Expected exception not caught";
		}
		catch (Exception& e)
		{
			EXPECT_STREQ(e.what(),
				"Pipe: trying to get writoken when pipe is closed!");
		}
		catch (...)
		{
			FAIL() << "Expected exception not caught";
		}
	}


	void WaitAndRead(PipeType& pipe, double timeToWait)
	{
		Timer::Sleep(timeToWait);
		{
			auto rt = pipe.GetReadToken();
		}
	}

	TEST(PipeTest, GetWriteTokenBlocked)
	{
		const size_t numWriterThreads = 4;
		const size_t nWriteSize = 10;
		const size_t numReadThreads = 4;
		const size_t nReadSize = 10;
		const size_t nOverlapSize = 0;
		PipeType pipe("UTGetWriteTokenBlocked");
		pipe.SetTemplate(0, {});
		pipe.SetProducer("producer", numWriterThreads, nWriteSize);
		pipe.SetConsumer("consumer", numReadThreads, nReadSize, nOverlapSize);

		// full-filled the pipe
		{
			auto wt = pipe.GetWriteToken(pipe.GetBufferSize(), true);
		}

		double timeToWait = 0.2;

		Timer timer;
		timer.Tic();

		std::future<void> fut1(
			std::async(std::launch::async,
				&WaitAndRead, std::ref(pipe),
				timeToWait));

		auto wt = pipe.GetWriteToken(nWriteSize, true);
		double duration = timer.Toc();

		EXPECT_GT(duration, timeToWait);
	}

	void WaitAndWriteShot(PipeType& pipe, std::vector<DataType> content, double timeToWait)
	{
		Timer::Sleep(timeToWait);
		{
			while (!content.empty())
			{
				PipeWriteToken<DataType> wt;
				if (content.size() > pipe.GetWriteSize())
				{
					wt = pipe.GetWriteToken(content.size(), false);
				}
				else
				{
					wt = pipe.GetWriteToken(content.size(), true);
				}
				for (size_t i = 0; i < wt.GetSize(); ++i)
				{
					wt.GetBuffer(i) = content[i];
				}
				content.erase(content.begin(), content.begin() + wt.GetSize());
			}
		}
	}

	TEST(PipeTest, GetReadTokenBlocked)
	{
		const size_t numWriterThreads = 1;
		const size_t nWriteSize = 2;
		const size_t numReadThreads = 2;
		const size_t nReadSize = 1;
		const size_t nOverlapSize = 0;
		PipeType pipe("GetReadTokenBlocked");
		pipe.SetTemplate(0, {});
		pipe.SetProducer("producer", numWriterThreads, nWriteSize);
		pipe.SetConsumer("consumer", numReadThreads, nReadSize, nOverlapSize);

		double timeToWait = 0.2;

		std::vector<size_t> content(2);
		for (size_t i = 0; i < content.size(); ++i)
			content[i] = i;

		std::future<void> fut(
			std::async(std::launch::async,
				&WaitAndWriteShot,
				std::ref(pipe),
				content,
				timeToWait));

		Timer timer;
		timer.Tic();
		auto rt = pipe.GetReadToken();
		double duration = timer.Toc();

		EXPECT_GT(duration, timeToWait);

		EXPECT_TRUE(rt.IsShotStart());
		EXPECT_FALSE(rt.IsShotEnd());
		EXPECT_EQ(size_t(1), rt.GetSize());
		EXPECT_EQ(size_t(0), rt.GetOverlapSize());
		EXPECT_EQ(content[0], rt.GetBuffer(0));


		auto rt2 = pipe.GetReadToken();
		EXPECT_FALSE(rt2.IsShotStart());
		EXPECT_TRUE(rt2.IsShotEnd());
		EXPECT_EQ(size_t(1), rt2.GetSize());
		EXPECT_EQ(size_t(0), rt2.GetOverlapSize());
		EXPECT_EQ(content[0], rt.GetBuffer(0));
	}

	TEST(PipeTest, PipeClose)
	{
		const size_t numWriterThreads = 10;
		const size_t nWriteSize = 7;
		const size_t numReadThreads = 7;
		const size_t nReadSize = 5;
		const size_t nOverlapSize = 2;
		Pipe<size_t> pipe("UTPipeClose");
		pipe.SetTemplate(0, {});
		pipe.SetProducer("producer", numWriterThreads, nWriteSize);
		pipe.SetConsumer("consumer", numReadThreads, nReadSize, nOverlapSize);

		pipe.Close();

		size_t countPipeClosedSignalCatched = 0;
		for (size_t i = 0; i < 3; ++i)
		{
			try
			{
				auto rt = pipe.GetReadToken();
			}
			catch (PipeClosedAndEmptySignal&)
			{
				++countPipeClosedSignalCatched;
			}
		}
		EXPECT_EQ(countPipeClosedSignalCatched, 3);
	}

	TEST(PipeTest, AllReadersAndWritersWorkSimultaneously)
	{
		const size_t numWriterThreads = 4;
		const size_t nWriteSize = 10;
		const size_t numReadThreads = 3;
		const size_t nReadSize = 5;
		const size_t nOverlapSize = 1;
		Pipe<size_t> pipe("UTAllReadersAndWritersWorkSimultaneously");
		pipe.SetTemplate(0, {});
		pipe.SetProducer("producer", numWriterThreads, nWriteSize);
		pipe.SetConsumer("consumer", numReadThreads, nReadSize, nOverlapSize);

		// write something to read
		{
			size_t numFramesToWrite =
				numReadThreads * nReadSize - (numReadThreads - 1) * nOverlapSize;
			auto wt = pipe.GetWriteToken(numFramesToWrite, true);
		}

		std::vector<Pipe<size_t>::ReadToken> rtVec(numReadThreads);
		for (size_t i = 0; i < numReadThreads; ++i)
		{
			rtVec[i] = pipe.GetReadToken();
		}
		std::vector<Pipe<size_t>::WriteToken> wtVec(numWriterThreads);
		for (size_t i = 0; i < numWriterThreads; ++i)
		{
			wtVec[i] = pipe.GetWriteToken(nWriteSize, false);
		}
	}

	TEST(PipeTest, ConcurrentWritingAndReading)
	{
		// you can try different combinations

		const size_t numWriterThreads = 3;
		const size_t nWriteSize = 7;
		const size_t numReadThreads = 11;
		const size_t nReadSize = 5;
		const size_t nOverlapSize = 2;

		// initialize pipe

		Pipe<DataType> pipe("UTConcurrentWritingAndReading");
		pipe.SetTemplate(0, {});
		//pipe.SetBufferSize(10);
		pipe.SetProducer("producer", numWriterThreads, nWriteSize);
		pipe.SetConsumer("consumer", numReadThreads, nReadSize, nOverlapSize);

		// initialize data

		std::vector<std::vector<DataType>> originalData;
		{
			size_t index = 0;
			for (size_t i = nOverlapSize + 1; i < 100; ++i)
			{
				std::vector<DataType> shot(i);
				for (size_t j = 0; j < i; ++j)
				{
					shot[j] = ++index;
				}
				originalData.push_back(std::move(shot));
			}
		}

		std::vector<std::vector<DataType>> echoData;
		for (std::vector<DataType>& shot : originalData)
		{
			echoData.emplace_back(shot.size(), 0);
		}

		std::mutex syncWriterMutex;
		std::mutex syncReaderMutex;

		// deploy reading threads

		std::vector<std::future<void>> readDataThreads;
		{
			size_t readingFrameIndex = 0;
			size_t readingShotIndex = 0;
			for (size_t i = 0; i < numReadThreads; ++i)
			{
				readDataThreads.emplace_back(std::async([&] {
					try
					{
						while (true)
						{
							PipeType::ReadToken rt;
							DataType* pData;

							// get readToken in critical section
							{
								std::lock_guard lk(syncReaderMutex);
								rt = pipe.GetReadToken();

								pData = &echoData[readingShotIndex][readingFrameIndex - rt.GetOverlapSize()];

								if (rt.IsShotEnd())
								{
									++readingShotIndex;
									readingFrameIndex = 0;
								}
								else
								{
									readingFrameIndex += rt.GetSize() - rt.GetOverlapSize();
								}
							}

							// read data concurrently
							for (size_t i = 0; i < rt.GetSize(); ++i)
							{
								*pData = rt.GetBuffer(i);
								++pData;
							}
						}
					}
					catch (PipeClosedAndEmptySignal&)
					{
					}
					}
				));
			}
		}

		// deploy writing threads

		std::vector<std::future<void>> writeDataThreads;
		{
			size_t writingFrameIndex = 0;
			size_t writingShotIndex = 0;

			for (size_t i = 0; i < numWriterThreads; ++i)
			{
				writeDataThreads.emplace_back(std::async([&] {
					try
					{
						while (true)
						{
							PipeType::WriteToken wt;
							const DataType* pData;

							// get writeToken in critical section
							{
								std::lock_guard lk = std::lock_guard(syncWriterMutex);
								if (writingShotIndex == originalData.size())
								{
									pipe.Close();
									break;
								}

								pData = &originalData[writingShotIndex][writingFrameIndex];
								size_t currentWriteSize = originalData[writingShotIndex].size() - writingFrameIndex;
								if (currentWriteSize <= nWriteSize)
								{
									wt = pipe.GetWriteToken(currentWriteSize, true);

									writingFrameIndex = 0;
									++writingShotIndex;
								}
								else
								{
									wt = pipe.GetWriteToken(nWriteSize, false);
									writingFrameIndex += nWriteSize;
								}
							}

							// write data concurrently
							for (size_t i = 0; i < wt.GetSize(); ++i)
							{
								wt.GetBuffer(i) = *pData;
								++pData;
							}
						}
					}
					catch (Exception& e)
					{
						std::cerr << e.what() << std::endl;
					}
					})
				);
			}
		}


		// wait for join

		std::chrono::time_point timeout =
			std::chrono::high_resolution_clock::now() + std::chrono::seconds(10);
		for (auto& thread : readDataThreads)
		{
			EXPECT_EQ(thread.wait_until(timeout), std::future_status::ready);
		}
		for (auto& thread : writeDataThreads)
		{
			EXPECT_EQ(thread.wait_until(timeout), std::future_status::ready);
		}

		// check result

		EXPECT_EQ(echoData, originalData);

	}

	TEST(PipeTest, ConcurrentGetReadTokenFail)
	{
		const size_t numWriterThreads = 1;
		const size_t nWriteSize = 1;
		const size_t numReadThreads = 2;
		const size_t nReadSize = 1;
		const size_t nOverlapSize = 0;
		Pipe<size_t> pipe("UTConcurrentGetReadTokenFail");
		pipe.SetTemplate(0, {});
		pipe.SetProducer("producer", numWriterThreads, nWriteSize);
		pipe.SetConsumer("consumer", numReadThreads, nReadSize, nOverlapSize);

		std::future<void> fut1(
			std::async(
				std::launch::async,
				[&] { pipe.GetReadToken(); }
			)
		);

		std::future<void> fut2(
			std::async(
				std::launch::async,
				[&] { pipe.GetReadToken(); }
			)
		);

		Timer::Sleep(0.1);
		{
			pipe.GetWriteToken(1, true);
			fut1.get();
		}

		try {
			fut2.get();
		}
		catch (Exception& e)
		{
			ASSERT_STREQ(e.what(), "reentered sequential scope");
			return;
		}
		FAIL() << "exception not caught";

	}

	TEST(PipeTest, ConcurrentGetWriteTokenFail)
	{
		const size_t numWriterThreads = 2;
		const size_t nWriteSize = 1;
		const size_t numReadThreads = 1;
		const size_t nReadSize = 1;
		const size_t nOverlapSize = 0;
		Pipe<size_t> pipe("UTConcurrentGetWriteTokenFail");
		pipe.SetTemplate(0, {});
		pipe.SetProducer("producer", numWriterThreads, nWriteSize);
		pipe.SetConsumer("consumer", numReadThreads, nReadSize, nOverlapSize);

		{
			pipe.GetWriteToken(pipe.GetBufferSize(), true);
			// now the pipe is full
		}

		std::future<void> fut1(
			std::async(
				std::launch::async,
				[&] { pipe.GetWriteToken(1, true); }
			)
		);

		std::future<void> fut2(
			std::async(
				std::launch::async,
				[&] { pipe.GetWriteToken(1, true); }
			)
		);

		Timer::Sleep(0.1);
		{
			for (size_t i = 0; i < pipe.GetBufferSize(); ++i)
				pipe.GetReadToken();
			fut1.get();
		}

		try {
			fut2.get();
		}
		catch (Exception& e)
		{
			ASSERT_STREQ(e.what(), "reentered sequential scope");
			return;
		}
		FAIL() << "exception not caught";

	}

}