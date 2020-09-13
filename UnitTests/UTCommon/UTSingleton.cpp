#include "pch.h"

#include <queue>
#include "../../Common/Singleton.h"

using namespace JEngine;

namespace UTCommon
{

	int a = 1;

	TEST(SingletonTest, EqualValue)
	{
		Singleton<int>::Instance() = a;
		ASSERT_EQ(a, Singleton<int>::Instance());
		++Singleton<int>::Instance();
		ASSERT_EQ(a + 1, Singleton<int>::Instance());
	}

	TEST(SingletonTest, SameItem)
	{
		int& b = Singleton<int>::Instance();
		++b;
		ASSERT_EQ(&b, &Singleton<int>::Instance());
	}

	TEST(SingletonTest, SingletonQueue)
	{
		for (int i = 0; i < 10; ++i)
		{
			Singleton<std::queue<int>>::Instance().push(i);
		}

		for (int i = 0; i < 10; ++i)
		{
			ASSERT_EQ(i, Singleton<std::queue<int>>::Instance().front());
			Singleton<std::queue<int>>::Instance().pop();
		}
	}
}