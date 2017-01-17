#ifndef CUDA_RAYTRACER_SHAPE_H
#define CUDA_RAYTRACER_SHAPE_H

#include <host_defines.h>
#include <vector_functions.h>
#include "Material.h"


namespace cray
{
	class Light;
	class Ray;

	//https://en.wikipedia.org/wiki/Policy-based_design
	//Shape classes will implement functions, and the Shape class with be created
	//at compile time
	template<class ShapeType>
	class Shape : private ShapeType
	{
	public:
		//can't using virtual functions, as the vtable is invalid when copied to the device
		using ShapeType::ShapeType;


		__device__ bool intersects(Ray& p_ray) const {
			using ShapeType::intersects;
			return intersects(p_ray);
		}
		__device__ bool intersects_simple(const Ray& p_ray) const {
			using ShapeType::intersects_simple;
			return intersects_simple(p_ray);
		}

		__device__ float4 calc_lighting(const Ray& p_ray, Light* p_lights, unsigned int p_num_lights) const {
			using ShapeType::calc_lighting;
			calc_lighting(p_ray, p_lights, p_num_lights);
		}
	};

	class ShapeList {
		template <typename T>
		void add_shape(const T& shape) {
			
		}


	};
}
#endif