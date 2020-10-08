#include "pch.h"

#include <future>
#include "../../Common/SequentialScope.h"
#include "../../Common/Timer.h"

using namespace JEngine;

namespace UTCommon
{
	class SomeClass {
	public:
		void SequentialFunction()
		{
			CheckSequential(SequentialFunctionEntered);

			// some codes that may modify private buffers
			// ...
			// ...
			Timer::Sleep(0.1);
		}
	private:
		SequentialScope::EnteredFlag SequentialFunctionEntered;
	};

	TEST(SequentialScopeTest, ConcurrentCallShouldFail)
	{
		SomeClass object;

		std::future<void> fut1(
			std::async(
				std::launch::async,
				[&]() {
					object.SequentialFunction();
				}));

		std::future<void> fut2(
			std::async(
				std::launch::async,
				[&]() {
					object.SequentialFunction();
				}));

		fut1.get();// the first call will finish successfully

		try {
			fut2.get();
		}
		catch (Exception& e)
		{
			ASSERT_STREQ(e.what(), "reentered sequential scope");
			return;
		}
		FAIL();
	}

	TEST(SequentialScopeTest, MultiObjectIsOK)
	{
		SomeClass object1, object2;

		std::future<void> fut1(
			std::async(
				std::launch::async,
				[&]() {
					object1.SequentialFunction();
				}));

		std::future<void> fut2(
			std::async(
				std::launch::async,
				[&]() {
					object2.SequentialFunction();
				}));

		fut1.get(); // the first call will finish successfully
		fut2.get(); // the second call also success.
	}
}