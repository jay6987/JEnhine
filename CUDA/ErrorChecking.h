#pragma once
#include "cuda_runtime.h"
namespace JEngine
{
#define CUDA_SAFE_CALL(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CUDA_KERNEL_LAUNCH_PREPARE() __cudaKernelLaunchPrepare(__FILE__, __LINE__)
#define CUDA_KERNEL_LAUNCH_CHECK() __cudaKernelLaunchCheck(__FILE__, __LINE__)

	void __cudaSafeCall(cudaError_t err, const char* file, const int line);
	void __cudaKernelLaunchPrepare(const char* file, const int line);
	void __cudaKernelLaunchCheck(const char* file, const int line);
}
