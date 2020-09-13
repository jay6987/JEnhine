//   Exception is the base exception for the FEngine system.
//   All other exception shall derive from it.

#pragma once

#include <exception>
#include <string>

namespace JEngine
{

#define ThrowExceptionAndLog(msg) throw Exception(__FILE__,__LINE__,msg,true)
#define ThrowException(msg) throw Exception(__FILE__,__LINE__,msg,false)

	namespace ErrComponents
	{
		enum
		{
			NONE = 0,
			Pipe
			// ...
		};
	}

	namespace ErrLevel
	{
		enum
		{
			NONE = 0
			// ...
		};
	}

	class Exception : public std::exception
	{
	public:
		Exception(const std::string file, const size_t nLine, const std::string msg, const bool log);
		std::string What() const { return what(); };
		// TODO:
		// add ErrLevel, ErrComponents...
	};
}
