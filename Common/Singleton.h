//   This is a template class. 
//   The singleton class must have a default constructor.
//

#pragma once
#include"Noncopyable.h"

namespace JEngine
{
	template<typename T>
	class Singleton : public Noncopyable
	{
	public:

		static T& Instance()
		{
			static T instance;
			return instance;
		}

		Singleton() = delete;

		~Singleton() = delete;

	};
}
