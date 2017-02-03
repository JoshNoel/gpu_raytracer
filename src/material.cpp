#include "Material.h"


namespace cray
{
	Material::Material(float3 p_color)
		: m_color(make_float4(p_color.x, p_color.y, p_color.z, 0)) {
	}

	Material::Material()
		: Material(make_float3(0.862f, 0.882f, 0.89f))  {
	}

	Material::~Material() {
	}
}
