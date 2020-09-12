#include "pch.h"

#include <future>
#include "../../Common/SyncQueue.h"
#include "../../Common/Timer.h"

using namespace JEngine;

namespace UTCommon
{
	const double timeToWait = 0.2;
	const double tolerance = 0.0005;
	const int chances = 100;


	TEST(SyncQueueTest, Pop)
	{
		std::vector<int> elements{ 1,2 };

		for (int i = 0; i < chances; ++i)
		{
			std::cout << "try #" << i << ": ";

			SyncQueue<int> myQueue1;
			std::future<void> fut(
				std::async(
					std::launch::async,
					[&]()
					{
						myQueue1.Push(elements[0]);
						Timer::Sleep(timeToWait);
						myQueue1.Push(elements[1]);
					}
			));

			Timer timer;
			
			int rst0 = myQueue1.Pop();

			timer.Tic();
			int rst1 = myQueue1.Pop();

			const double span(timer.Toc());

			ASSERT_EQ(elements[0], rst0);
			ASSERT_EQ(elements[1], rst1);
			double err(span - timeToWait);
			std::cout << "actual - expected = " << err << std::endl;
			if (tolerance >= abs(err))
				return;
		}
		FAIL();

	}

	TEST(SyncQueueTest, Push)
	{

		for (int i = 0; i < chances; ++i)
		{
			std::cout << "try #" << i << ": ";


			SyncQueue<int> myQueue2(2);
			myQueue2.Push(0);
			myQueue2.Push(0);

			// now the queue is full

			Timer timer;

			std::future<void> fut(
				std::async(
					std::launch::async,
					[&]() {
						Timer::Sleep(timeToWait);
						myQueue2.Pop();
					}
			));

			timer.Tic();
			myQueue2.Push(0);
			double span(timer.Toc());
			double err(span - timeToWait);
			std::cout << "actual - expected = " << err << std::endl;
			if (tolerance >= abs(err))
				return;
			myQueue2.Pop();
			myQueue2.Pop();
		}
		FAIL();
	}

	TEST(SyncQueueTest, Close)
	{
		for (int i = 0; i < chances; ++i)
		{
			std::cout << "try #" << i << ": ";

			SyncQueue<int> myQueue3;
			std::future<void> fut(
				std::async(
					std::launch::async,
					[&]()
					{
						Timer::Sleep(timeToWait);
						myQueue3.Close();
					}
			));

			Timer timer;

			try
			{
				myQueue3.Pop();
			}
			catch (SyncQueueClosedAndEmptySignal&)
			{
				double span(timer.Toc());
				double err(span - timeToWait);
				std::cout << "actual - expected = " << err << std::endl;
				if (tolerance >= abs(err))
					return;
				else
					continue;
			}
			FAIL();
		}
		FAIL();
	}
}