#include "Plane.h"


namespace cray
{
	Plane::Plane(float3 p_position, float3 p_normal, float3 p_u_axis, float2 p_dimensions, const Material& p_material, float p_thickness)
		: m_position(p_position), m_normal(norm(p_normal)), m_u_axis(norm(p_u_axis)), m_half_dimensions(p_dimensions / 2.0f), m_thickness(p_thickness), m_material(p_material) {
		m_v_axis = cross(m_normal, m_u_axis);
	}

	Plane::~Plane() {
	}

	__device__ bool Plane::intersects(Ray& p_ray) const {
		float d = dot(p_ray.m_dir, m_normal);
		if (d >= 0)
			return false;
		//ax + by + cz = u | PLANE EQ.
		//norm dot (o + dt) = u
		//n.x*(o.x+d.x*t)... = u
		//n.x*d.x*t... = u - (n dot o)
		//t = (u - (n dot o))/(n dot d)

		float u = dot(m_position, m_normal);
		float t = (u - dot(m_normal, p_ray.m_ori)) / dot(m_normal, p_ray.m_dir);

		if (t < p_ray.m_thit0) {
			float3 hitPoint = p_ray.m_ori + p_ray.m_dir * t;
			//check if within dimensions
			if (abs(dot(hitPoint - m_position, m_u_axis)) < m_half_dimensions.x && abs(dot(hitPoint - m_position, m_v_axis)) < m_half_dimensions.y) {
				p_ray.m_thit0 = t;
				p_ray.m_thit1 = t + (m_thickness / d);
				return true;
			}
		}

		return false;
	}

	__device__ float4 Plane::calc_lighting(const Ray& p_ray, Light* p_lights, unsigned p_num_lights) const {
		float3 finalColor = make_float3(0, 0, 0);
		for(auto i = 0; i < p_num_lights; i++) {
			switch (p_lights[i].m_type) {
			case Light::LIGHT_TYPE::POINT:
				float3 toLight = p_lights->m_point_light.m_pos - p_ray.calc_intersection_point_1();
				float m = mag(toLight);
				//linear falloff for now
				finalColor = finalColor + m_material.m_color * dot(norm(toLight), m_normal) * (p_lights[i].m_intensity / (m * m));
				break;
			case Light::LIGHT_TYPE::DIRECTIONAL:
				finalColor = finalColor + m_material.m_color * dot(m_normal, p_lights[i].m_dir_light.m_dir * -1.0f) * p_lights[i].m_intensity;
			}
		}

		return make_float4(finalColor.x, finalColor.y, finalColor.z, 1.0f);
	}

}