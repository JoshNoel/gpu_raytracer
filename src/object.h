#ifndef CUDA_RAYTRACER_OBJECT_H
#define CUDA_RAYTRACER_OBJECT_H

#include <vector_functions.h>

#include "material.h"
#include "ray.h"
#include "light.h"
#include "math_helper.h"

#include <string>

#define EPSILON 0.000001

namespace cray
{
	struct Triangle  {
		static inline Triangle make_triangle(const float3& vert1, const float3& vert2, const float3& vert3) {
			return Triangle{make_float4(vert1.x, vert1.y, vert1.z, 0), 
				make_float4(vert2.x - vert1.x, vert2.y - vert1.y, vert2.z - vert1.z, 0), 
				make_float4(vert3.x - vert1.x, vert3.y - vert1.y, vert3.z - vert1.z, 0) };
		}

		__device__ inline bool intersects(Ray& p_ray, int i) const {
			volatile int x = i;
			float3 cross_res = cross(p_ray.get_dir(), m_edge2);
			float determinant = dot(m_edge1, cross_res);
			//if det=0 no intersection
			if (determinant > -EPSILON && determinant < EPSILON)
				return false;
			float inv_determinant = 1.0f / determinant;
			float3 to_origin = p_ray.get_origin() - m_vert1;
			float u = dot(to_origin, cross_res) * inv_determinant;
			if (u < 0.0f || u > 1.0f)
				return false;
			
			float3 cross2_res = cross(to_origin, m_edge1);

			float v = dot(cross2_res, p_ray.get_dir()) * inv_determinant;
			if (v < 0.0f || u + v > 1.0f)
				return false;

			float t = dot(cross2_res, m_edge2) * inv_determinant;
			p_ray.set_thit(t);
			return true;			
		}

		//float4's can be accessed in 1 cuda instruction. Leave last component 0 for now.
		float4 m_vert1;
		float4 m_edge1;
		float4 m_edge2;
	};

	class Object {
	public:
		__host__ bool loadObj(const std::string& path, Material& p_material, float3 p_position);
		__device__ bool intersects(Ray& p_ray) {
			bool intersects = false;
			for (unsigned int i = 0; i < num_tris; i++) {
				if (tris[i].intersects(p_ray, i)) {
					p_ray.set_hit_object(this);
					intersects = true;
				}
			}
			return intersects;
		}

		__device__ float4 calc_lighting(const Ray& p_ray, Light* p_lights, unsigned int p_num_lights) const {
			return material.m_color;
		}

		//dynamically allocated array
		Triangle* tris;
		unsigned int num_tris;
		Material material;
	};
}
#endif

/*int triangle_intersection(const Vec3   V1,  // Triangle vertices
	const Vec3   V2,
	const Vec3   V3,
	const Vec3    O,  //Ray origin
	const Vec3    D,  //Ray direction
	float* out)
{
	Vec3 e1, e2;  //Edge1, Edge2
	Vec3 P, Q, T;
	float det, inv_det, u, v;
	float t;

	//Find vectors for two edges sharing V1
	SUB(e1, V2, V1);
	SUB(e2, V3, V1);
	//Begin calculating determinant - also used to calculate u parameter
	CROSS(P, D, e2);
	//if determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
	det = DOT(e1, P);
	//NOT CULLING
	if (det > -EPSILON && det < EPSILON) return 0;
	inv_det = 1.f / det;

	//calculate distance from V1 to ray origin
	SUB(T, O, V1);

	//Calculate u parameter and test bound
	u = DOT(T, P) * inv_det;
	//The intersection lies outside of the triangle
	if (u < 0.f || u > 1.f) return 0;

	//Prepare to test v parameter
	CROSS(Q, T, e1);

	//Calculate V parameter and test bound
	v = DOT(D, Q) * inv_det;
	//The intersection lies outside of the triangle
	if (v < 0.f || u + v  > 1.f) return 0;

	t = DOT(e2, Q) * inv_det;

	if (t > EPSILON) { //ray intersection
		*out = t;
		return 1;
	}

	// No hit, no win
	return 0;
}
//moller-trumbore
return true;
		}*/