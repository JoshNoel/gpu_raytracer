#ifndef CUDA_RAYTRACER_MATH_HELPER_H
#define CUDA_RAYTRACER_MATH_HELPER_H

#include "cuda_runtime.h"
#include <cmath>

namespace cray
{
	const float _PI_ = 3.1415f;


	static inline __host__ __device__ float2 operator*(const float2& a, const float& b) {
		return make_float2(a.x * b, a.y * b);
	}

	static inline __host__ __device__ float2 operator/(const float2& a, const float& b) {
		return make_float2(a.x / b, a.y / b);
	}

	static inline __host__ __device__ float2 operator+(const float2& a, const float2& b) {
		return make_float2(a.x + b.x, a.y + b.y);
	}

	static inline __host__ __device__ float2 operator-(const float2& a, const float2& b) {
		return make_float2(a.x - b.x, a.y - b.y);
	}

	static inline __host__ __device__ float3 operator*(const float3& a, const float& b) {
		return make_float3(a.x * b, a.y * b, a.z * b);
	}

	static inline __host__ __device__ float3 operator/(const float3& a, const float& b) {
		return make_float3(a.x / b, a.y / b, a.z / b);
	}

	static inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
		return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	static inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
		return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

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
#else
		return sqrtf(dot(a, a));
#endif
	}

	static inline __host__ __device__ float3 norm(const float3& a) {
		return a / mag(a);
	}
	
	static inline __host__ __device__ char sign(float a) {
		return a >= 0 ? 1 : -1;
	}

	static inline __host__ float deg_to_rad(float a) {
		return a * _PI_/180.0f;
	}

	static inline __host__ __device__ float3 calc_points_to(float3 from, float3 to) {
		return to - from;
	}
}
#endif