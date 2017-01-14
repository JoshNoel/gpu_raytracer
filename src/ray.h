#ifndef CUDA_RAYTRACER_RAY_H
#define CUDA_RAYTRACER_RAY_H

#include "cuda_runtime.h"
#include "camera.h"

namespace cray
{
	class Ray {
	public:
		__device__ Ray(float3 p_ori, float3 p_dir);
		__device__ Ray(const Ray&);
		__device__ Ray& operator=(const Ray&);
		__device__ ~Ray();

		static __device__ Ray make_primary(unsigned int x, unsigned int y, const Camera& p_camera);

		__device__ float3 calc_point(float t) const;
		__device__ float3 calc_intersection_point_1() const;
		__device__ float3 calc_intersection_point_2() const;

		__device__ const float3& get_origin() const { return m_ori; }
		__device__ const float3& get_dir() const { return m_dir; }

		__device__ float get_thit0() const { return m_thit0; }
		__device__ float get_thit1() const { return m_thit1; }
		__device__ void set_thit0(float f) { m_thit0 = f; }
		__device__ void set_thit1(float f) { m_thit1 = f; }

	private:
		//32 bytes
		float3 m_ori;
		float3 m_dir;
		float m_thit0;
		float m_thit1;
	};
}
#endif
