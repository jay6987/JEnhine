// Description:
//    IPP initializer detects the processor type used in 
//    the user computer system and sets the processor-specific
//    code of the Intel IPP library most appropriate for
//    the current processor type.
//
// Usage:
//    Just include this file at the beginging of any code file
//    that using IPP.

#pragma once
#include "..\Common\Singleton.h"

namespace JEngine
{
	class IPPInitializer
	{
	public:
		IPPInitializer();

	private:
		bool isInitialized = false;
	};

	static IPPInitializer& GlobalIppInit
		= Singleton<IPPInitializer>::Instance();
}