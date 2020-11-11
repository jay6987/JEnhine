//   Exception is the base exception for the FEngine system.
//   All other exception shall derive from it.

#include "Exception.h"
#include "GLog.h"

namespace JEngine
{
	Exception::Exception(
		const std::string& file,
		const size_t line,
		const std::string& msg,
		const bool log)
		: std::exception(msg.c_str())
	{
		if (log)
			GLog("An exception is thrown in line "
				+ std::to_string(line)
				+ ", \""
				+ file
				+ "\": "
				+ what());
	}
}
