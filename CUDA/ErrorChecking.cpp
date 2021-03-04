#include "ErrorChecking.h"

#include "../Common/Exception.h"

namespace JEngine {


	void __cudaSafeCall(cudaError_t err, const char* file, const int line)
	{
		if (cudaSuccess != err)
		{
			ThrowExceptionAndLog(
				std::string("CUDA error at \"") +
				file + "\", line " + std::to_string(line) +
				"error code: " + std::to_string(err) + " " +
				cudaGetErrorString(err));
		}
	}
	void __cudaKernelLaunchPrepare(const char* file, const int line)
	{
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err)
		{
			ThrowExceptionAndLog(
				std::string("CUDA error before kernel launch at \"") +
				file + "\", line " + std::to_string(line) +
				"error code: " + std::to_string(err) + " " +
				cudaGetErrorString(err));
		}
	}

	void __cudaKernelLaunchCheck(const char* file, const int line)
	{
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err)
		{
			ThrowExceptionAndLog(
				std::string("CUDA error at kernel launch at \"") +
				file + "\", line " + std::to_string(line) +
				"error code: " + std::to_string(err) + " " +
				cudaGetErrorString(err));
		}
	}
	
}