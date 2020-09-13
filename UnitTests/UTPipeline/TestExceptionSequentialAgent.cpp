#include "pch.h"
#include "TestExceptionSequentialAgent.h"
#include "../../Common/Exception.h"

namespace JEngine
{
	TestExceptionSequentialAgent::TestExceptionSequentialAgent(const int errorNum)
		: SequentialAgentBase("CatchSeuqenceAgent")
		, errorNum(errorNum)
	{
	}

	void TestExceptionSequentialAgent::WorkFlow()
	{
		switch (TestExceptionSequentialAgent::errorNum)
		{
		case 0:
			throw std::exception("sequential thread throw an std exception");
			break;
		case 1:
			ThrowException("concurrent thread throw an exception");
			break;
		case 2:
			throw TestExceptionSequentialAgent::UnknownException();
			break;
		}
	}

}
