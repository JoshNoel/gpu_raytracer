#include "sphere.h"
#include "math_helper.h"

namespace cray
{
	Sphere::Sphere(float p_radius, float3 p_position, Material& p_material)
		: m_radius(p_radius), m_position(p_position), m_material(p_material) {
	}


	Sphere::~Sphere() {
	}

	__device__ bool Sphere::intersects(Ray& ray) const {
		/*
		x = o + dt;
		r2 = |x-c|2;
		
		r2 = |o + dt - c|2;
		r2 = (o + dt - c) dot(o + dt - c);
		r2 = t2(d * d) + (o - c) * (o - c) + 2t(d*(o - c));
		ax2 + bx + c = 0;
			x = t;
			a = d*d
			b = 2(d*(o-c))
			c = |o-c|^2 - r^2
		*/

		float a = dot(ray.m_dir, ray.m_dir);
		float b = 2 * dot(ray.m_dir, ray.m_ori - this->m_position);
		float c = dot(ray.m_dir - this->m_position, ray.m_dir - this->m_position) - (this->m_radius * this->m_radius);

		//https://en.wikipedia.org/wiki/Quadratic_equation#Avoiding_loss_of_significance
		//http://stackoverflow.com/questions/898076/solve-quadratic-equation-in-c
		float t1, t2;
		/*if (b == 0 && c == 0)
			t1 = t2 = 0;
		else if(4*a*c == 0)
		{
			t1 = 0;
	//to get rid of intellisense error
	#ifdef __CUDACC__
			t2 = (-b - sqrtf(b*b - 4 * a*c)) / 2 * a;
	#endif
		}
		else*/ {
			if (b * b - 4 * a * c < 0)
				return false;
			//to get rid of intellisense error
#ifdef __CUDACC__
			float x = -0.5f * (b + sign(b) * sqrtf(b*b - 4 * a*c));
			t1 = x / a;
			t2 = c / x;
#endif
		}

		if (t2 < t1) {
			float t = t1;
			t1 = t2;
			t2 = t;
		}

		if (ray.m_thit0 > t1) {
			ray.m_thit0 = t1;
			ray.m_thit1 = t2;
		}

		return true;
	}

	__device__ float4 Sphere::calc_lighting(const Ray& p_ray, Light* p_lights, unsigned int p_num_lights) const {
		float3 finalColor = make_float3(0,0,0);
		float3 intersectionPoint = p_ray.calc_intersection_point_1();
		for (auto i = 0; i < p_num_lights; i++) {
			switch (p_lights->m_type) {
				case Light::LIGHT_TYPE::POINT:
					float3 normal = intersectionPoint - m_position;
					float3 toLight = p_lights[i].m_point_light.m_pos - intersectionPoint;
					finalColor = finalColor + m_material.m_color * dot(normal, toLight) * p_lights[i].m_intensity;
					break;
			}
		}
		return make_float4(finalColor.x, finalColor.y, finalColor.z, 1.0f);
	}
}
