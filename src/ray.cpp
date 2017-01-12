#include "ray.h"
#include <cfloat>

#include "math_helper.h"
#include "cuda_runtime.h"

namespace cray
{
	__device__ Ray::Ray(float3 p_ori, float3 p_dir)
		: m_ori(p_ori), m_dir(norm(p_dir)), m_thit0(FLT_MAX), m_thit1(FLT_MAX) {
	}

	__device__ Ray::Ray(const Ray& p_ray)
		: m_ori(p_ray.m_ori), m_dir(p_ray.m_dir), m_thit0(FLT_MAX), m_thit1(FLT_MAX) {

	}

	__device__ Ray& Ray::operator=(const Ray& p_ray) {
		m_ori = p_ray.m_ori;
		m_dir = p_ray.m_dir;
		m_thit0 = p_ray.m_thit0;
		m_thit1 = p_ray.m_thit1;
		return *this;
	}

	__device__ Ray::~Ray() {
	}

	__device__ Ray Ray::make_primary(unsigned x, unsigned y, const Camera& p_camera) {
		float3 dir = norm(p_camera.m_dir);
		dir = dir * p_camera.m_focal_length;
		dir = dir + p_camera.m_right * p_camera.m_world_width * (float(x) / float(p_camera.m_width) - 0.5f);
		dir = dir + p_camera.m_up * p_camera.m_world_height * (float(y) / float(p_camera.m_height) - 0.5f);

		return Ray(p_camera.m_position, norm(dir));
	}

	__device__ float3 Ray::calc_intersection_point_1() const {
		return m_ori + m_dir * m_thit0;
	}

	__device__ float3 Ray::calc_intersection_point_2() const {
		return m_ori + m_dir * m_thit1;
	} 

}
