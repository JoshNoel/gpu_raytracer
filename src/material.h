#pragma once
#include <vector_functions.h>

namespace cray
{
	class Material
	{
	public:
		Material(float3 p_color);
		~Material();

		float3 m_color;
	};
}

