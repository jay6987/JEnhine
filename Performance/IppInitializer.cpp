// Description:
//    IPP initializer detects the processor type used in 
//    the user computer system and sets the processor-specific
//    code of the Intel IPP library most appropriate for
//    the current processor type.

#include "IPPInitializer.h"
#include <ippcore.h>

JEngine::IPPInitializer::IPPInitializer()
{
	ippInit();
	isInitialized = true;
}
