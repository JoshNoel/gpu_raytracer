#ifndef CUDA_RAYTRACER_RAY_H
#define CUDA_RAYTRACER_RAY_H

#include "cuda_runtime.h"
#include "camera.h"
#include <cfloat>

namespace cray
{
	class Object;
	class Ray {
	public:
		__device__ Ray::Ray(float3 p_ori, float3 p_dir)
			: m_ori(p_ori), m_dir(norm(p_dir)), m_thit0(FLT_MAX), m_thit1(FLT_MAX) , m_hit_object(nullptr){
		}

		__device__ Ray::Ray(const Ray& p_ray)
			: m_ori(p_ray.m_ori), m_dir(p_ray.m_dir), m_thit0(FLT_MAX), m_thit1(FLT_MAX), m_hit_object(p_ray.m_hit_object) {

		}
		__device__ Ray& Ray::operator=(const Ray& p_ray) {
			m_ori = p_ray.m_ori;
			m_dir = p_ray.m_dir;
			m_thit0 = p_ray.m_thit0;
			m_thit1 = p_ray.m_thit1;
			m_hit_object = p_ray.m_hit_object;
			return *this;
		}		
		__device__ ~Ray() {};

		static __device__ Ray make_primary(unsigned int x, unsigned int y, const Camera* p_camera) {
			float3 dir = norm(p_camera->get_dir());
			dir = dir * p_camera->get_focal_length();
			dir = dir + p_camera->get_right() * p_camera->get_world_width() * ((float(x) / float(p_camera->get_width())) - 0.5f);
			dir = dir + p_camera->get_up() * p_camera->get_world_height() * ((float(y) / float(p_camera->get_height())) - 0.5f);

			return Ray(p_camera->get_position(), norm(dir));
		}

		__device__ float3 Ray::calc_point(float t) const { return m_ori + m_dir * t; }

		__device__ float3 Ray::calc_intersection_point_1() const { return m_ori + m_dir * m_thit0; }

		__device__ float3 Ray::calc_intersection_point_2() const { return m_ori + m_dir * m_thit1; }

		__device__ const float3& get_origin() const { return m_ori; }
		__device__ const float3& get_dir() const { return m_dir; }

		__device__ float get_thit0() const { return m_thit0; }
		__device__ float get_thit1() const { return m_thit1; }
		__device__ bool set_thit(float t) {
			if (t < m_thit0) {
				m_thit0 = t;
				return true;
			}
			if (t < m_thit1)
				m_thit1 = t;
			return false;
		}

		__device__ Object* get_hit_object() const { return m_hit_object; }
		__device__ void set_hit_object(Object* p_object) { m_hit_object = p_object; }

	private:
		//32 bytes
		float3 m_ori;
		float3 m_dir;
		float m_thit0;
		float m_thit1;
		Object* m_hit_object;
	};
}
#endif
