#include "pch.h"
#include "TestExceptionSequentialAgent.h"
#include "../../Common/Exception.h"

namespace JEngine
{
	TestExceptionSequentialAgent::TestExceptionSequentialAgent(const int errorNum)
		: SequentialAgentBase("TestExceptionSequentialAgent",1)
		, errorNum(errorNum)
	{
	}

	void TestExceptionSequentialAgent::WorkFlow0()
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
