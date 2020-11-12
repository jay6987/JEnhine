#include "pch.h"
#include <future>
#include <string>
#include "../../Common/SyncMap.h"
#include "../../Common/Timer.h"

using namespace JEngine;

namespace UTCommon
{
	TEST(SyncMapTest, InsertAndGetAndErase)
	{
		SyncMap<int, std::string> monthTable;
		monthTable.Insert(1, "Jan");
		EXPECT_STREQ(monthTable.Get(1).c_str(), "Jan");
		monthTable.Erase(1);

		try
		{
			monthTable.Get(1);
		}
		catch (std::exception& e)
		{
			EXPECT_STREQ(e.what(), "invalid map<K, T> key");
			return;
		}
		FAIL() << "Exception not caught";
	}

	TEST(SyncMapTest, Wait)
	{
		SyncMap<int, std::string> monthTable;

		std::future<void> fut =
			std::async(
				std::launch::async,
				[&]() {
					Timer::Sleep(0.1);
					monthTable.Insert(2, "Feb");
				}
		);

		Timer timer;
		monthTable.Wait(2);
		double waitTime = timer.Toc();


		EXPECT_STREQ(monthTable.Get(2).c_str(), "Feb");

		std::cout << "wait time should be 0.1, was " << waitTime << std::endl;

		fut.get();
	}
}