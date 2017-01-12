#ifndef CUDA_RAYTRACER_MATH_HELPER_H
#define CUDA_RAYTRACER_MATH_HELPER_H

#include "cuda_runtime.h"

namespace cray
{
	const float _PI_ = 3.1415f;
	static inline __host__ __device__  float dot(const float3& a, const float3& b) {
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}

	static inline __host__ __device__  float3 cross(const float3& a, const float3& b) {
		return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
	}

	static inline __host__ __device__ float mag(const float3& a) {
		//to get rid of intellisense error
#ifdef __CUDACC__
		return sqrtf(dot(a,a));
#endif
	}

	static inline __host__ __device__ float3 norm(const float3& a) {
		float m = mag(a);
		return make_float3(a.x / m, a.y / m, a.z / m);
	}

	static inline __host__ __device__ float3 operator*(const float3& a, const float& b) {
		return make_float3(a.x *b, a.y *b, a.z *b);
	}

	static inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
		return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	static inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
		return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	
	static inline __host__ __device__ char sign(float a) {
		return a >= 0 ? 1 : -1;
	}

	static inline __host__ float deg_to_rad(float a) {
		return a * _PI_/180.0f;
	}
}
#endif