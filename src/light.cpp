#include "Light.h"


namespace cray
{
	Light::Light()
		: m_type(LIGHT_TYPE::UNDEFINED)
	{
	}

	Light::Light(const Light& p_light)
	: m_type(p_light.m_type), m_color(p_light.m_color), m_intensity(p_light.m_intensity) {
		switch(m_type)
		{
		case LIGHT_TYPE::POINT:
			m_point_light = p_light.m_point_light;
			break;
		}
	}


	Light::~Light()
	{
	}

	Light Light::make_point_light(float3 p_pos, float3 p_color, float p_intensity) {
		Light light;
		light.m_type = LIGHT_TYPE::POINT;
		light.m_color = p_color;
		light.m_intensity = p_intensity;
		light.m_point_light = PointLight(p_pos);
		return light;
	}

}
