#ifndef CUDA_RAYTRACER_SPHERE_H
#define CUDA_RAYTRACER_SPHERE_H

#include "ray.h"
#include "cuda_runtime.h"

namespace cray {
	class Sphere
	{
	public:
		Sphere(float p_radius, float3 p_position, float4 p_color);
		~Sphere();

		__device__ bool intersects(Ray& ray) const;

		//32 bytes
		float m_radius;
		float3 m_position;
		float4 m_color;
	};
}
#endif