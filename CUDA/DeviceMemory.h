// Description:
//   DeviceMemory<T> wraps global memory on GPU device

#pragma once

namespace JEngine
{
	template<typename T>
	class DeviceMemory
	{
	public:
		DeviceMemory(const size_t size = 0);

		DeviceMemory(const DeviceMemory<T>& org);

		DeviceMemory(DeviceMemory<T>&& org) noexcept;

		~DeviceMemory();

		size_t Size() const { return size; }

		T* Data() { return pDataOnDevice; }

		const T* CData() const { return pDataOnDevice; };

		void Swap(DeviceMemory<T>& another);

	private:

		T* pDataOnDevice;

		size_t size;

		void Malloc();
	};
}
