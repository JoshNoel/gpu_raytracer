#include "sphere.h"
#include "math_helper.h"

namespace cray
{
	Sphere::Sphere(float p_radius, float3 p_position, Material& p_material)
		: m_radius(p_radius), m_position(p_position), m_material(p_material) {
		m_data = new 
	}
		
	Sphere::~Sphere() {
		delete m_data;
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
		
		//technically a is used to calculate intersection, but as ray.m_dir is always normalized, a can be removed to optimize
		//float a = 1.0f;//dot(ray.m_dir, ray.m_dir);
		float b = 2 * dot(ray.get_dir(), ray.get_origin() - this->m_position);
		float c = dot(ray.get_origin() - this->m_position, ray.get_origin() - this->m_position) - (this->m_radius * this->m_radius);


		//https://en.wikipedia.org/wiki/Quadratic_equation#Avoiding_loss_of_significance
		//http://stackoverflow.com/questions/898076/solve-quadratic-equation-in-c
		float t1, t2;
		if (b == 0 && c == 0)
			t1 = t2 = 0;
		else if (4 * c == 0)
		{
			t1 = 0;
			//to get rid of intellisense error
#ifdef __CUDACC__
			t2 = (-b - sqrtf(b*b - 4 * c)) / 2;
#endif
		}
		else 
		{
			if (b * b - 4 * c < 0)
				return false;
//to get rid of intellisense error
#ifdef __CUDACC__
			float x = -0.5f * (b + sign(b) * sqrtf(b*b - 4 * c));
			t1 = x; //really x/a, but a is always 1
			t2 = c / x;
#endif
		}

		if (t2 < t1) {
			float t = t1;
			t1 = t2;
			t2 = t;
		}

		if(t1 < 0 && t2 < 0) {
			return false;
		}

		if (ray.get_thit0() > t1) {
			ray.set_thit0(t1);
			ray.set_thit1(t2);
			return true;
		}
		return false;
	}

	__device__ bool Sphere::intersects_simple(const Ray& ray) const {
		float b = 2 * dot(ray.get_dir(), ray.get_origin() - this->m_position);
		float c = dot(ray.get_origin() - this->m_position, ray.get_origin() - this->m_position) - (this->m_radius * this->m_radius);

		if (b * b - 4 * c < 0)
			return false;

		float t1, t2;
		//to get rid of intellisense error
#ifdef __CUDACC__
		float x = -0.5f * (b + sign(b) * sqrtf(b*b - 4 * c));
		t1 = x; //really x/a, but a is always 1
		t2 = c / x;
#endif
		//check if behind ray origin
		if (t1 < 0.0f && t2 < 0.0f)
			return false;

		return true;
	}


	__device__ float4 Sphere::calc_lighting(const Ray& p_ray, Light* p_lights, unsigned int p_num_lights) const {
		float3 finalColor = make_float3(0,0,0);
		float3 intersectionPoint = p_ray.calc_intersection_point_1();
		float3 normal = norm(intersectionPoint - m_position);
		for (auto i = 0; i < p_num_lights; i++) {
			switch (p_lights->get_type()) {
				case Light::LIGHT_TYPE::POINT:
					float3 toLight = p_lights[i].get_point_light().m_pos - intersectionPoint;
					float m = mag(toLight);
					//linear falloff for now
					finalColor = finalColor + m_material.get_color() * dot(normal, norm(toLight)) * (p_lights[i].get_intensity() / (m * m));
					break;
				case Light::LIGHT_TYPE::DIRECTIONAL:
					finalColor = finalColor + m_material.get_color() * dot(normal, p_lights[i].get_dir_light().m_dir * -1.0f) * p_lights[i].get_intensity();
			}
		}
		return make_float4(finalColor.x, finalColor.y, finalColor.z, 1.0f);
	}
}
