#include "pch.h"

#include <future>
#include "..\..\Common\Semaphore.h"
#include "..\..\Common\Timer.h"

using namespace JEngine;

namespace UTCommon
{

	const int chances = 100;
	const double tolerance = 0.001;
	const double timeToWait = 0.2;

	TEST(SemaphoreTest, WaitSignal)
	{
		for (size_t i = 0; i < chances; ++i)
		{
			std::cout << "try #" << i << ": ...";
			Semaphore sema;

			std::future<double> waitFuture = std::async(
				std::launch::async,
				[&]()
				{
					Timer timer;
					sema.Wait();
					return timer.Toc();
				}
			);

			Timer::Sleep(timeToWait);
			sema.Signal();

			double actualWaitTime = waitFuture.get();

			if (std::abs(actualWaitTime - timeToWait) < tolerance)
			{
				std::cout << "pass" << std::endl;
				return;
			}

			std::cout << "fail: expected 0.2s, actual " << actualWaitTime << "s." << std::endl;

		}
		FAIL();
	}

	TEST(SemaphoreTest, WaitMultiSignal)
	{
		Semaphore sema;

		int a, b, c, sum;

		std::future<int> add = std::async(
			std::launch::async,
			[&]()
			{
				sema.Wait(3);
				return a + b + c;
			}
		);

		std::future<void> prepareSouces = std::async(
			std::launch::async,
			[&]()
			{
				Timer::Sleep(0.1);
				a = 1;
				b = 2;
				sema.Signal(2); // two sources are ready

				Timer::Sleep(0.1);
				c = 3;
				sema.Signal(1); // all sources are ready
			}
		);

		sum = add.get();
		prepareSouces.get();

		ASSERT_EQ(sum, 6);
	}

	TEST(SemaphoreTest, MultiWaiter)
	{
		Semaphore sema;

		std::future<void> waiter1 = std::async(
			std::launch::async,
			[&]()
			{
				sema.Wait(1);
				return;
			}
		);

		std::future<void> waiter2 = std::async(
			std::launch::async,
			[&]()
			{
				sema.Wait(1);
				return;
			}
		);

		sema.Signal(2);

		waiter1.get();
		waiter2.get();
	}


	TEST(SemaphoreTest, Close)
	{
		Semaphore sema;

		std::future<double> waitFuture = std::async(
			std::launch::async,
			[&]()
			{
				Timer timer;
				try {
					sema.Wait();
				}
				catch (SemaphoreClosedSignal&)
				{
					return -1.0;
				}
				return timer.Toc();
			}
		);

		Timer::Sleep(0.2);
		sema.Close();

		double waitTime = waitFuture.get();

		ASSERT_EQ(waitTime, -1.0);

	}

}