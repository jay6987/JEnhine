// Description:
//   This is a special implement of 
//   - Pipe<T>::SetDump
//   - Pipe<T>::SetLoad
//   - Pipe<T>::DumpWriteTokenImpl
//   - Pipe<T>::LoadReadTokenImpl
//   with T = size_t

#include "pch.h"
#include "../../Pipeline/Pipe.h"

namespace JEngine
{
	template<>
	void Pipe<size_t>::SetDump(const std::filesystem::path& /*fileName*/)
	{
	}

	template<>
	void Pipe<size_t>::SetLoad(const std::filesystem::path& /*fileName*/)
	{
	}

	template<>
	void Pipe<size_t>::DumpWriteToken(const size_t /*nStart*/, const size_t /*nEnd*/)
	{
	}

	template<>
	void Pipe<size_t>::LoadReadToken(const size_t /*nStart*/, const size_t /*nEnd*/)
	{
	}
}