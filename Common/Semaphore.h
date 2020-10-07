// Description:
// Semaphore is a counter limiting the number of threads
// concurrently accessing a shared resource.
// When the counter is equal or greater than n,
// a call to Wait(n) returns immediately and decrements the counter by n.
// Otherwise, any subsequent calls to Semaphore::Wait blockand
// only return when the semaphore counter is equal or greater than n
// as the result of calling Signal(m) which increments the counter by m.
//
// TODO: Semaphore will be included in C++20, please use STL semaphore in the future
//

#pragma once
#include <mutex>

namespace JEngine
{

	struct SemaphoreClosedSignal {};

	class Semaphore
	{
	public:

		explicit Semaphore(const size_t initCount = 0)
			: count(initCount)
		{}

		void Signal(const size_t n = 1)
		{
			std::lock_guard<std::mutex> lock(mutex);
			count += n;
			condition.notify_all();
		}

		void Wait(const size_t n = 1)
		{
			std::unique_lock<std::mutex> lock(mutex);
			while (count < n)
			{
				if (isClosed)
					throw SemaphoreClosedSignal();
				else
					condition.wait(lock);
			}
			count -= n;
		}

		void Close()
		{
			std::lock_guard<std::mutex> lock(mutex);
			isClosed = true;
			condition.notify_all();
		}

	private:
		std::mutex mutex;
		std::condition_variable condition;
		size_t count;
		bool isClosed = false;
	};

}