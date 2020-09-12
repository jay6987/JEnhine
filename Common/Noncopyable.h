// Description:
//   Classes that derived from this class are noncopyable

#pragma once

namespace JEngine
{
	class Noncopyable
	{
	protected:
		Noncopyable(const Noncopyable&) = delete;
		Noncopyable& operator = (const Noncopyable) = delete;

		Noncopyable() = default;
		~Noncopyable() = default;
	};
}
