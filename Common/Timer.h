// Timer provides accurate time measurement and pause

#pragma once
#include<chrono>

namespace JEngine
{
	// The smallest value of high_resolution clocks is nanosecond ( 1e-12 s)
	// a double float has 15-16 (decimal) effective digits
	// if the timer dose not run several days ( 1 day = 8.64e4 seconds)
	// it will not have any problem to use double to accumulate time span
	class Timer
	{
	public:
		Timer();
		void Tic();

		// return the tme span from last time Tic() is call (or Timer is constructed) to now
		// measured in second
		double Toc() const;

		static void Sleep(double seconds);

	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> lastTic;
	};
}