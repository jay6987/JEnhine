// Description:
//   DeviceMemory<T> wraps global memory on GPU device
//   DeviceMemoryFloat.cpp specialize implementation of
//   DeviceMemory<float>

// The file cuda_tuntime_api.h contains a non-unicode character

#include <sstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DeviceMemory.h"
#include "../Common/Exception.h"
#include "ErrorChecking.h"

namespace JEngine
{
	template<typename T>
	DeviceMemory<T>::DeviceMemory(const size_t size)
		: size(size)
	{
		Malloc();
	}

	template<typename T>
	DeviceMemory<T>::DeviceMemory(const DeviceMemory<T>& org)
		: size(org.size)
	{
		Malloc();
		CUDA_SAFE_CALL(cudaMemcpy(
			pDataOnDevice, org.pDataOnDevice, size * sizeof(float), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	DeviceMemory<T>::DeviceMemory(DeviceMemory<T>&& org) noexcept
		:size(org.size)
		, pDataOnDevice(org.pDataOnDevice)
	{
		org.pDataOnDevice = nullptr;
	}

	template<typename T>
	DeviceMemory<T>::~DeviceMemory()
	{
		cudaFree(pDataOnDevice);
	}

	template<typename T>
	void DeviceMemory<T>::Swap(DeviceMemory<T>& another)
	{
		std::swap(pDataOnDevice, another.pDataOnDevice);
		std::swap(size, another.size);
	}

	template<typename T>
	void DeviceMemory<T>::Malloc()
	{
		cudaError_t ec = cudaMalloc(&pDataOnDevice, size * sizeof(T));
		if (cudaErrorMemoryAllocation == ec)
		{
			std::stringstream ss;
			ss << "CUDA failed to malloc " <<
				size * sizeof(T) << " bytes of memory, cudaError = " << ec;
			ThrowExceptionAndLog(ss.str());
		}
		else if (cudaSuccess != ec)
			ThrowExceptionAndLog(cudaGetErrorString(ec));

	}

	// this function is writen so that the template fuctions will be compile
	void InstanceSample()
	{
		DeviceMemory<float> instance1;
		DeviceMemory<float> instance2 = instance1;
		DeviceMemory<float> instance3(std::move(instance1));
		instance1.Swap(instance3);
	}
}