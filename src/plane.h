#ifndef CUDA_RAYTRACER_PLANE_H
#define CUDA_RAYTRACER_PLANE_H

#include <vector_functions.h>

#include "ray.h"
#include "light.h"
#include "material.h"

namespace cray
{
	class Plane
	{
	public:
		Plane(float3 p_position, float3 p_normal, float3 p_u_axis, float2 p_dimensions, const Material& p_material, float m_thickness = 0.01f);
		~Plane();

		__device__ bool intersects(Ray& p_ray) const;
		__device__ bool intersects_simple(const Ray& p_ray) const;

		__device__ float4 calc_lighting(const Ray& p_ray, Light* p_lights, unsigned int p_num_lights) const;

	private:
		//v-axis = normal X u-axis
		float3 m_normal;
		float3 m_u_axis, m_v_axis;
		float3 m_position;

		//(u-dimension, v-dimension)
		float2 m_half_dimensions;

		//to be used for refraction
		float m_thickness;

		Material m_material;
	};
}
#endif