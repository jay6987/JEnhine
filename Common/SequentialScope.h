// SequentialScope is a scope that can not be run sequentially
// if codes within a SequentialScope is run sequentially, an exception will be thrown


#pragma once

#include <mutex>
#include <string>
#include "Noncopyable.h"
#include "Exception.h"

namespace JEngine
{
	// uncomment the following line if you want to check sequential scope
//#define CHECK_SEQUENTIAL_SCOPE

#ifdef CHECK_SEQUENTIAL_SCOPE

#define DeclearSequentialMutex(enteredFlag) SequentialScope::EnteredFlag enteredFlag;

#define CheckSequential(enteredFlag) SequentialScope ss = SequentialScope(enteredFlag,__FILE__,__LINE__)

	class SequentialScope : Noncopyable
	{
	public:

		typedef std::mutex EnteredFlag;

		SequentialScope(SequentialScope::EnteredFlag& enteredFlag, std::string file, size_t line)
			:lock(enteredFlag, std::try_to_lock)
		{
			if (!lock.owns_lock())
			{
				throw Exception(file, line, "reentered sequential scope", true);
			}
		}

		~SequentialScope() {};

	private:
		std::unique_lock<std::mutex> lock;
	};

#else

#define DeclearSequentialMutex(enteredFlag) 

#define CheckSequential(enteredFlag) 

#endif

}