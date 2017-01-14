#pragma once
#include <vector_functions.h>

namespace cray
{
	class Material
	{
	public:
		Material(float3 p_color);
		~Material();


		//implement getters and setters as class is subject to major changes later
		//better to implement and be unnecessary than change later
		__device__ const float3& get_color() const { return m_color; }
		__device__ void set_color(const float3& c) { m_color = c; }

	private:
		float3 m_color;
	};
}

