#ifndef CUDA_RAYTRACER_CUDA_HELPER_H
#define CUDA_RAYTRACER_CUDA_HELPER_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace cray
{
#define CUDA_CHECK(err)	do {					\
	cuda_check_error(err, __FILE__, __LINE__);	\
	} while(0)

	inline void cuda_check_error(cudaError_t error, const char* file, unsigned int line)
	{

		if(error != cudaSuccess)
		{
			fprintf(stderr, "Cuda Error at: %s - %u: %s\n", file, line, cudaGetErrorString(error));
			cudaDeviceReset();
#ifdef BREAK_ON_CUDA_ERROR
			abort();
#endif
			exit(error);
		}
	}
}

#endif