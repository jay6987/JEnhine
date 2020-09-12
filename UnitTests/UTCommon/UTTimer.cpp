#include "pch.h"

#include <thread>
#include "../../Common/Timer.h"

namespace UTCommon
{
	int chances = 100;
	double tolerace = 0.001;
	double timeToWait = 0.2;
	using namespace JEngine;
	Timer timer;

	TEST(TimerTest, TicToc) {
		for (int i = 0; i < chances; ++i)
		{
			std::cout << "try #" << i << ": ";
			timer.Tic();
			std::this_thread::sleep_for(std::chrono::milliseconds(int(timeToWait * 1e3)));
			double duration = timer.Toc();
			const double err = duration - timeToWait;
			std::cout << "actual - expected = " << err << std::endl;
			if (abs(err) < tolerace)
			{
				return;
			}
		}
		FAIL();
	}

	TEST(TimerTest, Sleep) {
		for (int i = 0; i < chances; ++i)
		{
			std::cout << "try #" << i << ": ";
			timer.Tic();
			Timer::Sleep(timeToWait);
			double duration = timer.Toc();
			const double err = duration - timeToWait;
			std::cout << "actual - expected = " << err << std::endl;
			if (abs(err) < tolerace)
			{
				return;
			}
		}
		FAIL();
	}
}