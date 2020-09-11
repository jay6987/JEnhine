#include <thread>
#include "Timer.h"

namespace JEngine
{
	using namespace std::chrono;

	Timer::Timer()
		:lastTic(high_resolution_clock::now())
	{
	}

	void Timer::Tic()
	{
		lastTic = high_resolution_clock::now();
	}

	double Timer::Toc() const
	{
		return duration_cast<duration<double>>(
			high_resolution_clock::now() - lastTic).count();
	}

	void Timer::Sleep(double seconds)
	{
		std::this_thread::sleep_for(
			microseconds(static_cast<long long>(seconds * 1e6)));
	}
}