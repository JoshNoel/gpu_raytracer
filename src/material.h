#pragma once
#include <vector_functions.h>

namespace cray
{
	class Material
	{
	public:
		Material();
		Material(float3 p_color);
		~Material();

		float4 m_color;
	};
}

