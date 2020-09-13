#include "pch.h"

#include "../../Common/Exception.h"

using namespace JEngine;

namespace UTCommon
{
	TEST(ExceptionTest, ThrowExceptionTest)
	{
		ASSERT_ANY_THROW(
			ThrowException("An exception is thrown.")
		);
	}

	TEST(ExceptionTest, ThrowExceptionAndLogTest)
	{
		ASSERT_ANY_THROW(
			ThrowExceptionAndLog("An exception is thrown.")
		);
	}
}