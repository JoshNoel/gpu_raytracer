#ifndef CUDA_RAYTRACER_SPHERE_H
#define CUDA_RAYTRACER_SPHERE_H

#include "ray.h"
#include "light.h"
#include "material.h"

namespace cray {
	class Sphere
	{
	public:
		Sphere(float p_radius, float3 p_position, Material& p_material);
		~Sphere();

		__device__ bool intersects(Ray& ray) const;
		__device__ float4 calc_lighting(const Ray& p_ray, Light* p_lights, unsigned int p_num_lights) const;

		//32 bytes
		float m_radius;
		float3 m_position;
		Material m_material;
	};
}
#endif